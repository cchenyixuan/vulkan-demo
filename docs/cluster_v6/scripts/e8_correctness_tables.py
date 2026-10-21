"""
e8_correctness_tables.py — E8 phase 2, deliverable 8: the correctness tables of the dumped cases (4M, 8M) from the
job's compare/<label>.json (e8_field_compare.py with --reference2: K = 8 vs K = 1 a, the noise floor K = 1 a vs b, by
seam distance, per slab / device) and the dumps' sidecars (invariants, alive, frames).

Usage:
    python docs/cluster_v6/scripts/e8_correctness_tables.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/correctness.md, correctness.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402


def sci(value) -> str:
    return "-" if value is None else f"{value:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results, lines = [], ["## Correctness (E8 phase 2): K = 8 against K = 1 at step 1500, fluid particles, with the K = 1 noise floor", ""]
    for directory in arguments.job:
        directory = pathlib.Path(directory)
        compare_dir = directory / "compare"
        dumps_dir = directory / "dumps"
        for path in sorted(compare_dir.glob("*.json")) if compare_dir.exists() else []:
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["label"] = path.stem
            summary["job"] = directory.name
            results.append(summary)
            lines += [f"### {path.stem} ({directory.name}): K = {summary.get('slabs')}, cuts {summary.get('cuts')}, devices {summary.get('device_map')}, "
                      f"step {summary.get('horizon')} (references at {summary.get('reference_horizon')} / {summary.get('reference2_horizon')}); "
                      f"{summary.get('particles_compared', 0):,} fluid particles compared, ids missing {summary.get('missing_in_test')} / {summary.get('missing_in_reference')}; "
                      f"dumps valid: test {summary.get('test_valid')}, reference {summary.get('reference_valid')}, reference2 {summary.get('reference2_valid')}", "",
                      f"- test |B − A1|: max |dv|/U **{sci(summary.get('dv_max_over_U'))}**, rms {sci(summary.get('dv_rms_over_U'))}; max |drho|/rho0 "
                      f"**{sci(summary.get('drho_max_over_rho0'))}**, rms {sci(summary.get('drho_rms_over_rho0'))}; max |dx| {sci(summary.get('dx_max_m'))} m",
                      f"- noise |A1 − A2|: max |dv|/U {sci(summary.get('noise_dv_max_over_U'))}, rms {sci(summary.get('noise_dv_rms_over_U'))}; "
                      f"max |drho|/rho0 {sci(summary.get('noise_drho_max_over_rho0'))}, rms {sci(summary.get('noise_drho_rms_over_rho0'))}; "
                      f"rms ratios test / noise: dv {common.fmt(summary.get('dv_rms_ratio'), 2)}, drho {common.fmt(summary.get('drho_rms_ratio'), 2)}", ""]
            has_noise = summary.get("noise_dv_rms_over_U") is not None
            if has_noise:
                lines += ["| distance to the seam (columns) | particles | max dv/U test / noise | rms dv/U test / noise (ratio) | max drho/rho0 test / noise | rms drho/rho0 test / noise (ratio) | max dx (m) |",
                          "|---|---|---|---|---|---|---|"]
                for row in summary.get("bins", []):
                    lines.append(f"| {row['bin']} | {row['count']:,} | {sci(row['dv_max'])} / {sci(row.get('noise_dv_max'))} | {sci(row['dv_rms'])} / {sci(row.get('noise_dv_rms'))} "
                                 f"({common.fmt(row.get('dv_rms_ratio'), 2)}) | {sci(row['drho_max'])} / {sci(row.get('noise_drho_max'))} | {sci(row['drho_rms'])} / "
                                 f"{sci(row.get('noise_drho_rms'))} ({common.fmt(row.get('drho_rms_ratio'), 2)}) | {sci(row['dx_max_m'])} |")
            else:
                lines += ["| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |", "|---|---|---|---|---|---|---|"]
                for row in summary.get("bins", []):
                    lines.append(f"| {row['bin']} | {row['count']:,} | {sci(row['dv_max'])} | {sci(row['dv_rms'])} | {sci(row['drho_max'])} | {sci(row['drho_rms'])} | {sci(row['dx_max_m'])} |")
            lines += ["", "| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |", "|---|---|---|---|---|---|---|"]
            for row in summary.get("per_slab") or []:
                lines.append(f"| {row['slab']} | {row['device'] if row['device'] is not None else '-'} | {row['count']:,} | {sci(row['dv_max'])} | {sci(row['dv_rms'])} | {sci(row['drho_max'])} | {sci(row['drho_rms'])} |")
            if not summary.get("per_slab"):
                lines.append("| (no per-slab rows in this comparison) | | | | | | |")
            lines.append("")
        dumps = []
        for path in sorted(dumps_dir.glob("*.json")) if dumps_dir.exists() else []:
            document = json.loads(path.read_text(encoding="utf-8"))
            invariants = document.get("invariants", {})
            dumps.append({"dump": path.stem, "slabs": document.get("slabs"), "device_map": document.get("device_map"), "valid": invariants.get("valid"),
                          "drift": invariants.get("drift"), "overflow_total": invariants.get("overflow_total"), "missing_ids": invariants.get("missing_ids"),
                          "duplicate_ids": invariants.get("duplicate_ids"), "far_migration": invariants.get("far_migration_count"),
                          "alive": document.get("bookkeeping", {}).get("alive_total"), "frames_s": document.get("frames_time_s")})
        if dumps:
            lines += [f"### dumps of {directory.name}", "", "| dump | K | devices | valid | drift | overflow | missing | duplicates | far | alive | frames s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for d in dumps:
                lines.append(f"| {d['dump']} | {d['slabs']} | {d['device_map']} | {d['valid']} | {d['drift']} | {d['overflow_total']} | {d['missing_ids']} | {d['duplicate_ids']} | "
                             f"{d['far_migration']} | {common.fmt(d['alive'])} | {common.fmt(d['frames_s'], 1)} |")
            lines.append("")
            results.append({"job": directory.name, "dumps": dumps})
    text = "\n".join(lines)
    print(text[:3000])
    common.write_text(out / "correctness.md", text)
    common.write_json(out / "correctness.json", results)
    print(f"[e8_correctness_tables] {sum(1 for r in results if 'bins' in r)} comparisons -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
