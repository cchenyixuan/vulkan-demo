"""
e8_single_ratio.py — E8 phase 2, deliverable 9: the single-GPU solver comparison (fig:reference's method) from the
e8_single_ratio pairs of a job: per pair the v7_single bench's fps (the RESULT {json} line of single_<case>_p<i>_single.log:
measure steps / wall, drain to drain, defrag off) and the chain's K = 1 fps on the same card (the results row of
single_<case>_p<i>_chain: the steady window without defrag), the ratio chain / single per pair, the ratio of the means
and the min-max range of the per-pair ratios.

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

SINGLE_LOG = re.compile(r"^single_(.+)_p(\d+)_single\.log$")


def single_result(path: pathlib.Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = []
    for directory in arguments.job:
        job = common.load_job(directory)
        pairs = {}
        for path in sorted(pathlib.Path(directory).glob("single_*_single.log")):
            match = SINGLE_LOG.match(path.name)
            if not match:
                continue
            short, pair = match.group(1), int(match.group(2))
            single = single_result(path)
            chain_row = job["rows"].get(f"single_{short}_p{pair}_chain")
            chain_fps, chain_nodefrag = common.fps_pair(chain_row)
            entry = {"pair": pair, "single_fps": (single or {}).get("fps"), "single_us_per_step": (single or {}).get("us_per_step"),
                     "single_status": (single or {}).get("status"), "single_drift": (single or {}).get("drift"),
                     "single_overflow": (single or {}).get("overflow"), "single_gpu": (single or {}).get("gpu"),
                     "single_measure": (single or {}).get("measure"), "single_in_flight": (single or {}).get("in_flight"),
                     "chain_label": f"single_{short}_p{pair}_chain", "chain_status": (chain_row or {}).get("status"),
                     "chain_fps": chain_fps, "chain_fps_nodefrag": chain_nodefrag, "chain_device": (chain_row or {}).get("device_map")}
            chain_value = chain_nodefrag or chain_fps
            entry["ratio_chain_over_single"] = (chain_value / entry["single_fps"]) if chain_value and entry["single_fps"] else None
            pairs.setdefault(short, []).append(entry)
        for short, entries in pairs.items():
            entries.sort(key=lambda e: e["pair"])
            singles = [e["single_fps"] for e in entries if e["single_fps"]]
            chains = [(e["chain_fps_nodefrag"] or e["chain_fps"]) for e in entries if (e["chain_fps_nodefrag"] or e["chain_fps"])]
            ratios = [e["ratio_chain_over_single"] for e in entries if e["ratio_chain_over_single"]]
            results.append({"job": job["name"], "case": f"cavity{short}", "pairs": entries,
                            "single_fps_mean": statistics.fmean(singles) if singles else None,
                            "chain_fps_mean": statistics.fmean(chains) if chains else None,
                            "ratio_of_means": (statistics.fmean(chains) / statistics.fmean(singles)) if singles and chains else None,
                            "ratio_min": min(ratios) if ratios else None, "ratio_max": max(ratios) if ratios else None,
                            "complete_pairs": len(ratios)})
    lines = ["## v7_single vs chain K = 1 (E8 phase 2; same card, 2 in flight, 1500 / 500 steps, defrag off in both windows)", "",
             "| job | case | pair | v7_single fps | chain K = 1 fps (nodefrag) | ratio chain / single | single status / drift / overflow | chain status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for e in r["pairs"]:
            lines.append(f"| {r['job']} | {common.CASE_LABELS.get(r['case'], r['case'])} | {e['pair']} | {common.fmt(e['single_fps'], 3)} | "
                         f"{common.fmt(e['chain_fps_nodefrag'] or e['chain_fps'], 3)} | {common.fmt(e['ratio_chain_over_single'], 4)} | "
                         f"{e['single_status']} / {e['single_drift']} / {e['single_overflow']} | {e['chain_status']} |")
        lines.append(f"| {r['job']} | {common.CASE_LABELS.get(r['case'], r['case'])} | **mean / mean** | {common.fmt(r['single_fps_mean'], 3)} | {common.fmt(r['chain_fps_mean'], 3)} | "
                     f"**{common.fmt(r['ratio_of_means'], 4)}** (pairs {common.fmt(r['ratio_min'], 4)}–{common.fmt(r['ratio_max'], 4)}) | {r['complete_pairs']} pairs | |")
    text = "\n".join(lines)
    print(text)
    common.write_text(out / "single_ratio.md", text)
    common.write_json(out / "single_ratio.json", results)
    print(f"[e8_single_ratio] {len(results)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
