"""
hydrostatic_spread.py - the residual speeds of the 2-D hydrostatic boxes over repeated runs of the tag crossing-pool and
of this branch (claude/wall_store_report.md, the hydrostatic regression). A run's voxel lists are filled in atomic
order, so two runs of one build are two realizations; one run per build cannot compare them.

    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic_spread

Per box (gravity -y, tilted 30 degrees) and build: the runs found (hydrostatic.py run, logs/wall_store/<build>/
hydrostatic/), and per run the mean over 5 <= t <= 10 of the largest and the rms speed (over sqrt(g H)) and, at the
end, the largest speed in the first layer from the wall (m/s); the mean and the standard deviation over the runs and
the difference of the means with its standard error. The single runs of the diagnostics (all wall layers kept, the
summed density diffusion) are listed beside them. Writes claude/wall_store/tables/hydrostatic_spread.md and
logs/wall_store/checks/hydrostatic_spread.json.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
LOG_ROOT = REPOSITORY_ROOT / "logs" / "wall_store"


TAIL_THRESHOLD = 0.005


def fisher_one_sided(first: tuple, second: tuple) -> float:
    """One-sided Fisher exact test: the probability of first having at least its tail count, given the totals."""
    from math import comb
    first_tails, first_runs = first
    second_tails, second_runs = second
    tails, runs = first_tails + second_tails, first_runs + second_runs
    total = comb(runs, tails)
    return sum(comb(first_runs, count) * comb(second_runs, tails - count)
               for count in range(first_tails, min(first_runs, tails) + 1)) / total


def run_metrics(directory: pathlib.Path) -> dict | None:
    if not (directory / "summary.json").exists():
        return None
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with open(directory / "series.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scale = summary["scales"]["velocity_sqrt_g_extent"]
    time = np.array([float(row["time"]) for row in rows])
    largest = np.array([float(row["speed_maximum"]) for row in rows]) / scale
    root_mean_square = np.array([float(row["speed_root_mean_square"]) for row in rows]) / scale
    chosen = time >= 5.0 - 1e-9
    return {"largest": float(largest[chosen].mean()), "root_mean_square": float(root_mean_square[chosen].mean()),
            "first_layer": float(summary["final"]["layers"][0]["speed_maximum"]),
            "final_largest": float(summary["final"]["speed_maximum_over_scale"]), "run_id": summary["run_id"]}


def main() -> int:
    from experiment.wall_crossing.wall_store_check import markdown_table, write_table
    result = {}
    rows = []
    for box, text in (("gy", "重力沿 −y"), ("g30", "重力倾斜 30°")):
        builds = {
            "tag": [LOG_ROOT / "reference" / "hydrostatic" / f"ref_2d_n250_{box}"]
                   + [LOG_ROOT / "reference" / "hydrostatic" / f"ref_2d_n250_{box}_repeat{index}" for index in range(1, 13)],
            "new": [LOG_ROOT / "new" / "hydrostatic" / f"ws_2d_n250_{box}_b"]
                   + [LOG_ROOT / "new" / "hydrostatic" / f"ws_2d_n250_{box}_repeat{index}" for index in range(1, 13)],
        }
        result[box] = {}
        statistics = {}
        for build, directories in builds.items():
            runs = [metrics for metrics in (run_metrics(directory) for directory in directories) if metrics]
            values = {key: np.array([run[key] for run in runs]) for key in ("largest", "root_mean_square", "first_layer")}
            statistics[build] = {key: (float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0,
                                       array.size) for key, array in values.items()}
            result[box][build] = {"runs": runs, "statistics": statistics[build]}
            label = "标签 crossing-pool" if build == "tag" else "壁面独立存储（逐对）"
            rows.append([text, label, f"{len(runs)}"]
                        + [f"{statistics[build][key][0]:.5f} ± {statistics[build][key][1]:.5f}"
                           for key in ("largest", "root_mean_square")]
                        + [f"{statistics[build]['first_layer'][0]:.4f} ± {statistics[build]['first_layer'][1]:.4f}"]
                        + [" / ".join(f"{run['largest']:.4f}" for run in runs)])
        differences = []
        for key in ("largest", "root_mean_square", "first_layer"):
            tag_mean, tag_deviation, tag_count = statistics["tag"][key]
            new_mean, new_deviation, new_count = statistics["new"][key]
            error = float(np.sqrt(tag_deviation ** 2 / tag_count + new_deviation ** 2 / new_count))
            differences.append(f"{new_mean - tag_mean:+.5f}（标准误 {error:.5f}）")
            result[box].setdefault("difference", {})[key] = {"value": new_mean - tag_mean, "standard_error": error}
        rows.append([text, "差（新 − 标签）", ""] + differences + [""])
        for label, name in (("诊断：保留全部壁面层", f"ws_2d_n250_{box}_all_walls"),
                            ("诊断：密度扩散壁面项按标签合成", f"ws_2d_n250_{box}_diffusion_summed")):
            metrics = run_metrics(LOG_ROOT / "new" / "hydrostatic" / name)
            if metrics:
                result[box][name] = metrics
                rows.append([text, label, "1", f"{metrics['largest']:.5f}", f"{metrics['root_mean_square']:.6f}",
                             f"{metrics['first_layer']:.4f}", ""])
    # the tilted box's tail: runs whose mean largest speed (5 <= t <= 10) exceeds TAIL_THRESHOLD, by form of the walls'
    # density diffusion (the tag, the pair-by-pair default, the tag's summed form in this build)
    groups = {
        "tag": [LOG_ROOT / "reference" / "hydrostatic" / "ref_2d_n250_g30"]
               + [LOG_ROOT / "reference" / "hydrostatic" / f"ref_2d_n250_g30_repeat{index}" for index in range(1, 13)],
        "pairwise": [LOG_ROOT / "new" / "hydrostatic" / "ws_2d_n250_g30_b"]
                    + [LOG_ROOT / "new" / "hydrostatic" / f"ws_2d_n250_g30_repeat{index}" for index in range(1, 13)]
                    + [LOG_ROOT / "new" / "hydrostatic" / f"ws_g30_locate{index}" for index in range(1, 13)],
        "summed": [LOG_ROOT / "new" / "hydrostatic" / "ws_2d_n250_g30_diffusion_summed"]
                  + [LOG_ROOT / "new" / "hydrostatic" / f"ws_g30_summed_repeat{index}" for index in range(1, 15)],
    }
    tail_rows = []
    counts = {}
    for name, directories in groups.items():
        runs = [metrics for metrics in (run_metrics(directory) for directory in directories) if metrics]
        largest = np.array([run["largest"] for run in runs])
        root_mean_square = np.array([run["root_mean_square"] for run in runs])
        tails = int((largest > TAIL_THRESHOLD).sum())
        counts[name] = (tails, len(runs))
        result.setdefault("tail", {})[name] = {"runs": len(runs), "tails": tails, "largest": largest.tolist(),
                                               "root_mean_square": root_mean_square.tolist()}
        label = {"tag": "标签 crossing-pool", "pairwise": "壁面独立存储，逐对（缺省）",
                 "summed": "壁面独立存储，诊断：壁面密度扩散按标签合成"}[name]
        below = largest[largest <= TAIL_THRESHOLD]
        tail_rows.append([label, f"{len(runs)}", f"{tails}", f"{largest.mean():.5f} ± {largest.std(ddof=1):.5f}",
                          f"{below.mean():.5f}" if below.size else "—",
                          f"{root_mean_square.mean():.6f} ± {root_mean_square.std(ddof=1):.6f}"])
    for first, second in (("pairwise", "tag"), ("pairwise", "summed"), ("summed", "tag")):
        probability = fisher_one_sided(counts[first], counts[second])
        result["tail"][f"fisher_{first}_more_than_{second}"] = probability
        tail_rows.append([f"{first} 的尾部比 {second} 多（Fisher 精确检验单侧 p）", "", f"{probability:.3f}", "", "", ""])
    write_table("hydrostatic_tail.md", markdown_table(
        ["倾斜 30° 静水盒，按壁面密度扩散的写法", "运行次数", f"最大速度 > {TAIL_THRESHOLD} √(gH) 的次数（尾部）",
         "最大速度 / √(gH)（5 ≤ t ≤ 10 平均）", "不算尾部的平均", "均方根速度 / √(gH)"], tail_rows))
    write_table("hydrostatic_spread.md", markdown_table(
        ["静水盒（2-D 250²，到 t = 10）", "版本", "运行次数", "最大速度 / √(gH)（5 ≤ t ≤ 10 平均；均值 ± 标准差）",
         "均方根速度 / √(gH)（同上）", "末态离壁第一层的最大速度 m/s", "各次的最大速度"], rows))
    out = LOG_ROOT / "checks" / "hydrostatic_spread.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
