"""
e8_migration_stats.py — E8 phase 2, deliverable 10: per traced chain point (the 8-card points are the deliverable) the
migrations per seam and step over the steady window, from the traced run's raw pool demand (raw/<point>_trace_C/
pool_demand_<link>.csv, column migrant = the packet's n_mig of the link, i.e. the particles that crossed the seam in the
link's direction on that frame): per link mean, peak, p95 and as a fraction of one voxel column's particles (a full
non-empty column holds 4 (n + 8)^2 particles incl. walls and 4 n^2 fluid particles); per seam (both directions summed).

Usage:
    python docs/cluster_v6/scripts/e8_migration_stats.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/migration_stats.md, migration_stats.json.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402

LINK = re.compile(r"^s(\d+)_to_s(\d+)$")


def analyze(raw_dir, case: str, warmup: int, steps: int) -> dict:
    demand = common.read_pool_demand(raw_dir)
    lattice = common.lattice_of(case)
    column_all = 4 * (lattice["n"] + 8) ** 2
    column_fluid = 4 * lattice["n"] ** 2
    links, seams = {}, {}
    for name, counts in demand.items():
        if name.startswith("_"):
            continue
        migrant = counts.get("migrant")
        if migrant is None:
            continue
        index = counts["index"].astype(int)
        window = (index >= warmup) & (index < steps)
        values = migrant[window] if window.any() else migrant
        entry = {"frames": int(len(values)), "mean": float(np.mean(values)), "peak": float(np.max(values)), "p95": float(np.percentile(values, 95)),
                 "mean_fraction_of_column": float(np.mean(values)) / column_all, "peak_fraction_of_column": float(np.max(values)) / column_all,
                 "mean_fraction_of_fluid_column": float(np.mean(values)) / column_fluid, "total": float(np.sum(values))}
        links[name] = entry
        match = LINK.match(name)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            seam = f"{min(a, b)}-{max(a, b)}"
            seams.setdefault(seam, []).append(values)
    seam_rows = {}
    for seam, series in seams.items():
        length = min(len(s) for s in series)
        total = sum(s[:length] for s in series)
        seam_rows[seam] = {"directions": len(series), "mean": float(np.mean(total)), "peak": float(np.max(total)), "p95": float(np.percentile(total, 95)),
                           "mean_fraction_of_column": float(np.mean(total)) / column_all, "peak_fraction_of_column": float(np.max(total)) / column_all}
    return {"raw": str(raw_dir), "case": case, "column_particles": column_all, "column_fluid": column_fluid, "window": [warmup, steps],
            "links": links, "seams": seam_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = []
    for directory in arguments.job:
        job = common.load_job(directory)
        for name, point in sorted(common.points_of(job).items(), key=lambda item: (item[1]["slabs"], item[1]["case"])):
            if not point.get("raw_dir") or point["slabs"] < 2:
                continue
            trace_row = (point["traces"].get("C") or point["traces"].get("E") or {}).get("row") or {}
            warmup, steps = 500, int(trace_row.get("total_steps") or 1500)
            if point.get("trace_dir"):
                try:
                    import json
                    meta = json.loads((pathlib.Path(point["trace_dir"]) / "run_meta.json").read_text(encoding="utf-8"))
                    warmup, steps = int(meta.get("warmup", warmup)), int(meta.get("max_steps") or steps)
                except Exception:                                           # noqa: BLE001
                    pass
            try:
                result = analyze(point["raw_dir"], point["case"], warmup, steps)
            except Exception as error:                                      # noqa: BLE001
                result = {"error": repr(error)}
            result.update({"point": name, "job": job["name"], "K": point["slabs"]})
            results.append(result)
    lines = ["## Migrations per seam and step (E8 phase 2; the traced run's packet counters, steady window)", "",
             "| point | K | seam | directions | mean / step | peak | p95 | mean / column (all) | peak / column | column = particles (fluid) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['point']} | {r['K']} | failed: {r['error']} | | | | | | | |")
            continue
        for seam, s in sorted(r["seams"].items(), key=lambda item: int(item[0].split("-")[0])):
            lines.append(f"| {r['point']} | {r['K']} | {seam} | {s['directions']} | {s['mean']:.2f} | {s['peak']:.0f} | {s['p95']:.1f} | "
                         f"{s['mean_fraction_of_column'] * 100:.4f} % | {s['peak_fraction_of_column'] * 100:.4f} % | {r['column_particles']:,} ({r['column_fluid']:,}) |")
    lines += ["", "### Per link (direction)", "", "| point | link | frames | mean | peak | p95 | mean / column | total |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for name, link in (r.get("links") or {}).items():
            lines.append(f"| {r['point']} | {name} | {link['frames']} | {link['mean']:.2f} | {link['peak']:.0f} | {link['p95']:.1f} | "
                         f"{link['mean_fraction_of_column'] * 100:.4f} % | {link['total']:.0f} |")
    text = "\n".join(lines)
    print(text[:3000])
    common.write_text(out / "migration_stats.md", text)
    common.write_json(out / "migration_stats.json", results)
    print(f"[e8_migration_stats] {len(results)} points -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
