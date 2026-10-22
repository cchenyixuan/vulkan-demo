"""
e8_single_ratio.py — E8 phase 2, deliverable 9: the single-GPU solver comparison (fig:reference's method) from the
e8_single_ratio pairs of a job: per pair the v7_single bench's fps (the RESULT {json} line of single_<case>[_<mode>]_p<i>_single.log:
measure steps / wall, drain to drain) and the chain's K = 1 fps on the same card (the results row of
single_<case>[_<mode>]_p<i>_chain), the ratio chain / single per pair, the ratio of the means and the min-max range of the
per-pair ratios.

Protocols (e7_lib.sh e8_single_ratio, revision 15):
  off      (jobs 1721839 / 1722130): --no-defrag on both halves; the chain's fps without its defrag events is used.
  cadence  (Sec. 5.1's, job e8_S2): both at the case's defrag cadence, the window holds one defrag per 1000 steps on both
           sides; the chain's fps WITH its defrag events is used (the bench's fps includes its own).
A pair is INVALID when its v7_single half never defragmented: protocol off and the bench without the bootstrap defrag
(RESULT bootstrap_defrag false or absent = the bench before 2026-10-22; job 1722130's 17 %, claude/e8_single_vs_chain_k1.md).
Invalid pairs are listed with the reason and excluded from the means.
The bootstrap-defrag control of e8_single_control (single_<case>_bd_boot.log / _noboot.log: --no-defrag with / without the
bootstrap defrag) is reported as its own table.

Usage:
    python docs/cluster_v6/scripts/e8_single_ratio.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/single_ratio.md, single_ratio.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402

SINGLE_LOG = re.compile(r"^single_(.+?)(?:_(cadence))?_p(\d+)_single\.log$")
CONTROL_LOG = re.compile(r"^single_(.+)_bd_(boot|noboot)\.log$")
INVALID_REASON = ("v7_single never defragmented: the bench before 2026-10-22 skipped the bootstrap defrag under --no-defrag "
                  "while the chain's bootstrap_all always runs one (claude/e8_single_vs_chain_k1.md)")


def single_result(path: pathlib.Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                return None
    return None


def bootstrap_defrag_ms(path: pathlib.Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"bootstrap defrag: ([0-9.]+) ms", line)
        if match:
            return float(match.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = []
    controls = []
    for directory in arguments.job:
        job = common.load_job(directory)
        pairs = {}
        for path in sorted(pathlib.Path(directory).glob("single_*_single.log")):
            match = SINGLE_LOG.match(path.name)
            if not match:
                continue
            short, mode, pair = match.group(1), match.group(2) or "off", int(match.group(3))
            label_base = f"single_{short}" + ("" if mode == "off" else f"_{mode}")
            single = single_result(path) or {}
            chain_row = job["rows"].get(f"{label_base}_p{pair}_chain")
            chain_fps, chain_nodefrag = common.fps_pair(chain_row)
            single_cadence = single.get("defrag_cadence")
            bootstrap = bool(single.get("bootstrap_defrag", False))
            protocol = "cadence" if (single_cadence or 0) > 0 else "off"
            valid = not (protocol == "off" and not bootstrap) if single else False
            chain_value = (chain_fps if protocol == "cadence" else (chain_nodefrag or chain_fps))
            entry = {"pair": pair, "mode": mode, "protocol": protocol if single else "-", "valid": valid,
                     "failed": not single,          # no RESULT line: the single half did not run (job 1721839's V7_WORKER_AFFINITY)
                     "invalid_reason": None if valid or not single else INVALID_REASON,
                     "single_fps": single.get("fps"), "single_us_per_step": single.get("us_per_step"),
                     "single_status": single.get("status"), "single_drift": single.get("drift"),
                     "single_overflow": single.get("overflow"), "single_gpu": single.get("gpu"),
                     "single_measure": single.get("measure"), "single_warmup": single.get("warmup"),
                     "single_in_flight": single.get("in_flight"), "single_defrag_cadence": single_cadence,
                     "single_bootstrap_defrag": bootstrap, "single_bootstrap_defrag_ms": bootstrap_defrag_ms(path),
                     "chain_label": f"{label_base}_p{pair}_chain", "chain_status": (chain_row or {}).get("status"),
                     "chain_fps": chain_fps, "chain_fps_nodefrag": chain_nodefrag, "chain_fps_used": chain_value,
                     "chain_device": (chain_row or {}).get("device_map")}
            entry["ratio_chain_over_single"] = (chain_value / entry["single_fps"]) if chain_value and entry["single_fps"] else None
            pairs.setdefault((short, mode), []).append(entry)
        for (short, mode), entries in pairs.items():
            entries.sort(key=lambda e: e["pair"])
            good = [e for e in entries if e["valid"] and e["ratio_chain_over_single"]]
            singles = [e["single_fps"] for e in good]
            chains = [e["chain_fps_used"] for e in good]
            ratios = [e["ratio_chain_over_single"] for e in good]
            results.append({"job": job["name"], "case": f"cavity{short}", "mode": mode, "pairs": entries,
                            "protocol": entries[0]["protocol"] if entries else None,
                            "single_fps_mean": statistics.fmean(singles) if singles else None,
                            "chain_fps_mean": statistics.fmean(chains) if chains else None,
                            "ratio_of_means": (statistics.fmean(chains) / statistics.fmean(singles)) if singles and chains else None,
                            "ratio_min": min(ratios) if ratios else None, "ratio_max": max(ratios) if ratios else None,
                            "complete_pairs": len(ratios), "invalid_pairs": sum(1 for e in entries if not e["valid"])})
        runs = {}
        for path in sorted(pathlib.Path(directory).glob("single_*_bd_*.log")):
            match = CONTROL_LOG.match(path.name)
            if not match:
                continue
            short, variant = match.group(1), match.group(2)
            single = single_result(path) or {}
            runs.setdefault(short, {})[variant] = {"fps": single.get("fps"), "status": single.get("status"),
                                                   "drift": single.get("drift"), "measure": single.get("measure"),
                                                   "warmup": single.get("warmup"), "defrag_cadence": single.get("defrag_cadence"),
                                                   "bootstrap_defrag": single.get("bootstrap_defrag"),
                                                   "bootstrap_defrag_ms": bootstrap_defrag_ms(path)}
        for short, variants in runs.items():
            boot, noboot = variants.get("boot", {}).get("fps"), variants.get("noboot", {}).get("fps")
            controls.append({"job": job["name"], "case": f"cavity{short}", "boot": variants.get("boot"), "noboot": variants.get("noboot"),
                             "ratio_boot_over_noboot": (boot / noboot) if boot and noboot else None})
    lines = ["## v7_single vs chain K = 1 (E8 phase 2; same card, 2 in flight)", "",
             "Protocol off = --no-defrag on both halves (the chain's fps without defrag events); cadence = Sec. 5.1's, both at the "
             "case's defrag cadence 1000 with the window from step warmup + 1 (the chain's fps with its defrag events). "
             "A pair whose v7_single half never defragmented (protocol off without the bench's bootstrap defrag) is INVALID "
             "and excluded from the means.", "",
             "| job | case | protocol | pair | steps (warmup + measured) | v7_single fps (bootstrap defrag) | chain K = 1 fps used (with / without defrag) | ratio chain / single | valid | single status / drift / overflow | chain status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for e in r["pairs"]:
            steps = f"{e['single_warmup']} + {e['single_measure']}" if e["single_measure"] is not None else "-"
            lines.append(f"| {r['job']} | {common.CASE_LABELS.get(r['case'], r['case'])} | {e['protocol']} | {e['pair']} | {steps} | "
                         f"{common.fmt(e['single_fps'], 3)} ({'yes' if e['single_bootstrap_defrag'] else 'NO'}) | "
                         f"{common.fmt(e['chain_fps_used'], 3)} ({common.fmt(e['chain_fps'], 3)} / {common.fmt(e['chain_fps_nodefrag'], 3)}) | "
                         f"{common.fmt(e['ratio_chain_over_single'], 4)} | {'yes' if e['valid'] else '**INVALID**'} | "
                         f"{e['single_status']} / {e['single_drift']} / {e['single_overflow']} | {e['chain_status']} |")
        if r["complete_pairs"]:
            lines.append(f"| {r['job']} | {common.CASE_LABELS.get(r['case'], r['case'])} | {r['protocol']} | **mean / mean** | | "
                         f"{common.fmt(r['single_fps_mean'], 3)} | {common.fmt(r['chain_fps_mean'], 3)} | "
                         f"**{common.fmt(r['ratio_of_means'], 4)}** (pairs {common.fmt(r['ratio_min'], 4)}–{common.fmt(r['ratio_max'], 4)}) | "
                         f"{r['complete_pairs']} valid pairs | | |")
        else:
            lines.append(f"| {r['job']} | {common.CASE_LABELS.get(r['case'], r['case'])} | {r['protocol']} | **mean / mean** | | - | - | "
                         f"**none: {r['invalid_pairs']} invalid pairs** | | | |")
    invalid = [(r, e) for r in results for e in r["pairs"] if not e["valid"] and e["single_fps"]]
    if invalid:
        lines += ["", "Invalid pairs: " + "; ".join(f"{r['job']} pair {e['pair']} (chain / single {common.fmt(e['ratio_chain_over_single'], 4)})"
                                                     for r, e in invalid) + f". Reason: {INVALID_REASON}."]
    if controls:
        lines += ["", "### Bootstrap-defrag control (v7_single alone, --no-defrag, job 1722130's steps; e8_single_control)", "",
                  "| job | case | steps (warmup + measured) | with bootstrap defrag: fps (defrag ms) | without: fps | ratio with / without |",
                  "|---|---|---|---|---|---|"]
        for c in controls:
            boot, noboot = c["boot"] or {}, c["noboot"] or {}
            steps = f"{boot.get('warmup')} + {boot.get('measure')}" if boot.get("measure") is not None else "-"
            lines.append(f"| {c['job']} | {common.CASE_LABELS.get(c['case'], c['case'])} | {steps} | "
                         f"{common.fmt(boot.get('fps'), 3)} ({common.fmt(boot.get('bootstrap_defrag_ms'), 1)}) | "
                         f"{common.fmt(noboot.get('fps'), 3)} | **{common.fmt(c['ratio_boot_over_noboot'], 4)}** |")
    text = "\n".join(lines)
    print(text)
    common.write_text(out / "single_ratio.md", text)
    common.write_json(out / "single_ratio.json", {"pairs": results, "controls": controls})
    print(f"[e8_single_ratio] {len(results)} case/protocol groups, {len(controls)} controls -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
