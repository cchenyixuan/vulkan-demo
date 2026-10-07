"""
wall_option_timing.py — E37 timing tables of the wall option (case.yaml numerics.wall_boundary simple / adami, K = 1,
one GPU), CPU only.

(1) The E36 section 8.3 method: per-kernel GPU time of the last frame before each defrag boundary, median over the
    frames with step >= 3000, from canonical_dump --monitor --timestamps runs (<dir>/timing/<n250|n1000>_<wall>.npz
    and .monitor.jsonl), and the fps of those monitored runs.
(2) Production fps: the chain bench at K = 1 without monitor or timestamps (<dir>/bench/<n250|n1000>_<wall>_r*.out,
    the STEADY line after the 5,000-step warmup).

    .venv/Scripts/python.exe experiment/seam_audit/wall_option_timing.py logs/e37/A
"""
import json
import pathlib
import re
import statistics
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
KERNELS = ("predict", "update_voxel", "correction", "density", "wall_pass", "force", "phase_c")
results = {}
for resolution in ("n250", "n1000"):
    for wall in ("simple", "adami"):
        monitor = root / "timing" / f"{resolution}_{wall}.monitor.jsonl"
        if not monitor.exists():
            continue
        rows = [json.loads(line) for line in monitor.read_text(encoding="utf-8").splitlines() if line]
        rows = [row for row in rows if row["step"] >= 3000]
        parts = {key: [] for key in KERNELS}
        step = []
        for row in rows:
            ticks = row["ticks_ns"]

            def span(end, start):
                return (ticks[end] - ticks[start]) / 1000.0
            wall_pass = "b_wall_extrapolate_end" in ticks
            parts["predict"].append(span("a_predict_end", "a_start"))
            parts["update_voxel"].append(span("a_voxel_end", "a_predict_end"))
            parts["correction"].append(span("b_correction_interior_end", "b_start"))
            parts["density"].append(span("b_density_deep_interior_end", "b_correction_interior_end"))
            parts["wall_pass"].append(span("b_wall_extrapolate_end", "b_density_deep_interior_end") if wall_pass else 0.0)
            parts["force"].append(span("b_force_deep_interior_end",
                                       "b_wall_extrapolate_end" if wall_pass else "b_density_deep_interior_end"))
            parts["phase_c"].append(span("c_force_end", "c_start"))
            step.append(span("c_force_end", "a_start"))
        with np.load(root / "timing" / f"{resolution}_{wall}.npz") as archive:
            meta = json.loads(str(archive["meta"]))
        bench = []
        for path in sorted((root / "bench").glob(f"{resolution}_{wall}_r*.out")):
            match = re.search(r"STEADY \(post-warmup \d+\): \d+ steps in [\d.]+s = ([\d.]+) fps",
                              path.read_text(encoding="utf-8", errors="replace"))
            if match:
                bench.append(float(match.group(1)))
        results[(resolution, wall)] = {
            "medians": {key: float(np.median(values)) for key, values in parts.items()},
            "step_median": float(np.median(step)), "frames": len(rows), "fps_monitored": meta["fps"],
            "alive_ok": meta["alive"] == meta["expected"] and not meta.get("overflow"), "bench": bench}

print("内核时间(µs,E36 第 8.3 节方法:每个 defrag 边界前最后一帧,step ≥ 3000 的中位数)")
print("| 算例 | 壁面 | " + " | ".join(KERNELS) + " | 内核和 | a_start→c_force_end | 帧数 |")
print("|---|---|" + "---|" * (len(KERNELS) + 3))
for (resolution, wall), item in results.items():
    medians = item["medians"]
    print(f"| {resolution} | {wall} | " + " | ".join(f"{medians[key]:.1f}" for key in KERNELS)
          + f" | {sum(medians.values()):.1f} | {item['step_median']:.1f} | {item['frames']} |")
print()
print("| 算例 | 壁面 | 内核和 µs | 壁面 pass µs(占比) | fps(E36 方法,监视 + 时间戳) | fps(生产:链 bench K = 1,3 次) |")
print("|---|---|---|---|---|---|")
for (resolution, wall), item in results.items():
    medians = item["medians"]
    kernels = sum(medians.values())
    bench = item["bench"]
    bench_text = (f"{statistics.mean(bench):.1f} ± {statistics.stdev(bench):.1f}({', '.join(f'{v:.1f}' for v in bench)})"
                  if len(bench) >= 2 else str(bench))
    print(f"| {resolution} | {wall} | {kernels:.1f} | {medians['wall_pass']:.1f}({100 * medians['wall_pass'] / kernels:.1f} %) | "
          f"{item['fps_monitored']:.1f} | {bench_text} |")
print()
for resolution in ("n250", "n1000"):
    if (resolution, "simple") not in results or (resolution, "adami") not in results:
        continue
    simple, adami = results[(resolution, "simple")], results[(resolution, "adami")]
    kernel_simple, kernel_adami = sum(simple["medians"].values()), sum(adami["medians"].values())
    line = (f"{resolution}: adami/simple 内核和 {100 * (kernel_adami / kernel_simple - 1):+.1f} %, "
            f"a_start→c_force_end {100 * (adami['step_median'] / simple['step_median'] - 1):+.1f} %, "
            f"fps(E36 方法) {100 * (adami['fps_monitored'] / simple['fps_monitored'] - 1):+.1f} %"
            f"(每步时间 {100 * (simple['fps_monitored'] / adami['fps_monitored'] - 1):+.1f} %)")
    if simple["bench"] and adami["bench"]:
        ratio = statistics.mean(adami["bench"]) / statistics.mean(simple["bench"])
        line += f", fps(生产) {100 * (ratio - 1):+.1f} %(每步时间 {100 * (1 / ratio - 1):+.1f} %)"
    line += (f"; force {100 * (adami['medians']['force'] / simple['medians']['force'] - 1):+.1f} %, density "
             f"{100 * (adami['medians']['density'] / simple['medians']['density'] - 1):+.1f} %, 壁面 pass "
             f"{adami['medians']['wall_pass']:.1f} µs")
    print(line)
print("invariants:", {f"{key[0]}_{key[1]}": item["alive_ok"] for key, item in results.items()})
