"""
hydrostatic_round3.py - A3 of claude/wall_store_report.md section 9: the tilted (30 degrees) 2-D 250^2 hydrostatic box,
36 runs of each version (the wall store 21877e3, the tag crossing-pool): round 2's runs (logs/wall_store/new|reference/
hydrostatic) plus round 3's, run on the remote 5090 and pulled back (logs/wall_store/round3/remote/store|pool/
hydrostatic; the tag's driver there carries 351c27c's fastest-particle columns, no solver change).

    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic_round3

Per run (5 <= t <= 10, the samples of hydrostatic.py; section 8's measure): the mean of the largest fluid speed and of
the rms speed over sqrt(g H); the share of samples whose fastest fluid particle lies in the lower-left corner,
x < -0.45 and y < -0.45 (the corner of round 2: the fastest particle of every tail run sat at x ~ -0.48, y ~ -0.50);
a "tail" is a run whose mean largest speed exceeds 0.005 sqrt(g H) (section 8), a "lower-left tail" a tail whose
fastest particle is in the corner in at least half of its samples (only runs that record the position: round 2's
first 12 wall-store runs and 12 tag runs do not). Per version: the distribution (mean, standard deviation, median,
quartiles, extremes); tail counts with Fisher's exact test (one-sided: the wall store has more; and two-sided);
Mann-Whitney U on the largest and on the rms speed (two-sided, and one-sided wall store > tag).
Writes claude/wall_store/tables/round3_hydrostatic_*.md, claude/wall_store/figures/round3_hydrostatic.png and
logs/wall_store/round3/hydrostatic_round3.json.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from math import comb

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LOGS = REPOSITORY_ROOT / "logs" / "wall_store"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
TAIL_THRESHOLD = 0.005
CORNER = -0.45
GROUPS = {
    "store": ([LOGS / "new" / "hydrostatic" / "ws_2d_n250_g30_b"]
              + [LOGS / "new" / "hydrostatic" / f"ws_2d_n250_g30_repeat{index}" for index in range(1, 12)]
              + [LOGS / "new" / "hydrostatic" / f"ws_g30_locate{index}" for index in range(1, 13)]
              + [LOGS / "round3" / "remote" / "store" / "hydrostatic" / f"ws_2d_n250_g30_r3_repeat{index}"
                 for index in range(13, 25)]),
    "tag": ([LOGS / "reference" / "hydrostatic" / "ref_2d_n250_g30"]
            + [LOGS / "reference" / "hydrostatic" / f"ref_2d_n250_g30_repeat{index}" for index in range(1, 12)]
            + [LOGS / "round3" / "remote" / "pool" / "hydrostatic" / f"ref_2d_n250_g30_r3_repeat{index}"
               for index in range(13, 37)]),
}
LABELS = {"store": "壁面独立存储（21877e3）", "tag": "标签 crossing-pool"}


def run_metrics(directory: pathlib.Path) -> dict | None:
    if not (directory / "summary.json").exists():
        return None
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with open(directory / "series.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scale = summary["scales"]["velocity_sqrt_g_extent"]
    time = np.array([float(row["time"]) for row in rows])
    chosen = time >= 5.0 - 1e-9
    largest = np.array([float(row["speed_maximum"]) for row in rows])[chosen] / scale
    root_mean_square = np.array([float(row["speed_root_mean_square"]) for row in rows])[chosen] / scale
    entry = {"name": directory.name, "round": 3 if "round3" in directory.parts else 2,
             "largest": float(largest.mean()), "root_mean_square": float(root_mean_square.mean()),
             "samples": int(chosen.sum()), "run_id": summary.get("run_id"), "corner_share": None}
    if rows and "fastest_x" in rows[0] and rows[0]["fastest_x"] not in ("", None):
        x = np.array([float(row["fastest_x"]) for row in rows])[chosen]
        y = np.array([float(row["fastest_y"]) for row in rows])[chosen]
        entry["corner_share"] = float(((x < CORNER) & (y < CORNER)).mean())
        entry["median_fastest"] = [float(np.median(x)), float(np.median(y))]
    entry["tail"] = entry["largest"] > TAIL_THRESHOLD
    entry["lower_left_tail"] = (None if entry["corner_share"] is None
                                else bool(entry["tail"] and entry["corner_share"] >= 0.5))
    return entry


def fisher(first: tuple, second: tuple) -> tuple:
    """(one-sided p: first has at least its tail count, two-sided p) of a 2 x 2 table (tails, runs) x 2."""
    first_tails, first_runs = first
    second_tails, second_runs = second
    tails, runs = first_tails + second_tails, first_runs + second_runs
    total = comb(runs, tails)
    probabilities = {count: comb(first_runs, count) * comb(second_runs, tails - count) / total
                     for count in range(max(0, tails - second_runs), min(first_runs, tails) + 1)}
    one_sided = sum(value for count, value in probabilities.items() if count >= first_tails)
    observed = probabilities[first_tails]
    two_sided = sum(value for value in probabilities.values() if value <= observed * (1 + 1e-9))
    return float(one_sided), float(min(1.0, two_sided))


def main() -> int:
    from scipy.stats import mannwhitneyu
    result = {"tail_threshold": TAIL_THRESHOLD, "corner": CORNER, "runs": {}, "statistics": {}}
    for build, directories in GROUPS.items():
        runs = [entry for entry in (run_metrics(directory) for directory in directories) if entry]
        result["runs"][build] = runs
        for key in ("largest", "root_mean_square"):
            values = np.array([run[key] for run in runs])
            result["statistics"].setdefault(build, {})[key] = {
                "count": int(values.size), "mean": float(values.mean()), "standard_deviation": float(values.std(ddof=1)),
                "median": float(np.median(values)), "quartiles": [float(np.percentile(values, 25)),
                                                                  float(np.percentile(values, 75))],
                "minimum": float(values.min()), "maximum": float(values.max())}
        located = [run for run in runs if run["corner_share"] is not None]
        result["statistics"][build]["tails"] = int(sum(run["tail"] for run in runs))
        result["statistics"][build]["located_runs"] = len(located)
        result["statistics"][build]["located_tails"] = int(sum(run["tail"] for run in located))
        result["statistics"][build]["lower_left_tails"] = int(sum(bool(run["lower_left_tail"]) for run in located))
        result["statistics"][build]["corner_mostly"] = int(sum(run["corner_share"] >= 0.88 for run in located))
    store, tag = result["statistics"]["store"], result["statistics"]["tag"]
    tests = {}
    tests["fisher_tails"] = fisher((store["tails"], store["largest"]["count"]), (tag["tails"], tag["largest"]["count"]))
    tests["fisher_lower_left_tails"] = fisher((store["lower_left_tails"], store["located_runs"]),
                                              (tag["lower_left_tails"], tag["located_runs"]))
    for key in ("largest", "root_mean_square"):
        first = [run[key] for run in result["runs"]["store"]]
        second = [run[key] for run in result["runs"]["tag"]]
        two = mannwhitneyu(first, second, alternative="two-sided")
        greater = mannwhitneyu(first, second, alternative="greater")
        tests[f"mann_whitney_{key}"] = {"U": float(two.statistic), "p_two_sided": float(two.pvalue),
                                        "p_store_greater": float(greater.pvalue)}
    result["tests"] = tests

    lines = ["| 版本 | 运行次数 | 最大速度 / √(gH)：均值 ± 标准差 | 中位数（四分位） | 最小 … 最大 | 均方根速度 / √(gH)：均值 ± 标准差 | 中位数（四分位） |",
             "|---|---|---|---|---|---|---|"]
    for build in ("store", "tag"):
        largest, rms = result["statistics"][build]["largest"], result["statistics"][build]["root_mean_square"]
        lines.append(f"| {LABELS[build]} | {largest['count']} | {largest['mean']:.5f} ± {largest['standard_deviation']:.5f} | "
                     f"{largest['median']:.5f}（{largest['quartiles'][0]:.5f}–{largest['quartiles'][1]:.5f}） | "
                     f"{largest['minimum']:.5f} … {largest['maximum']:.5f} | "
                     f"{rms['mean']:.6f} ± {rms['standard_deviation']:.6f} | "
                     f"{rms['median']:.6f}（{rms['quartiles'][0]:.6f}–{rms['quartiles'][1]:.6f}） |")
    (TABLES / "round3_hydrostatic_distribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    lines = ["| 版本 | 运行次数 | 尾部（最大速度 > 0.005 √(gH)） | 记录了最快粒子位置的运行 | 其中尾部 | 其中左下角尾部 | 最快粒子 ≥ 88 % 的采样在左下角的运行 |",
             "|---|---|---|---|---|---|---|"]
    for build in ("store", "tag"):
        statistics = result["statistics"][build]
        lines.append(f"| {LABELS[build]} | {statistics['largest']['count']} | {statistics['tails']} | "
                     f"{statistics['located_runs']} | {statistics['located_tails']} | {statistics['lower_left_tails']} | "
                     f"{statistics['corner_mostly']} |")
    (TABLES / "round3_hydrostatic_tails.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    lines = ["| 检验 | 统计量 | p（双侧） | p（单侧：壁面独立存储更大 / 更多） |", "|---|---|---|---|"]
    lines.append(f"| 尾部次数，Fisher 精确检验（全部 {store['largest']['count']} 对 {tag['largest']['count']} 次） | "
                 f"{store['tails']} / {tag['tails']} | {tests['fisher_tails'][1]:.3f} | {tests['fisher_tails'][0]:.3f} |")
    lines.append(f"| 左下角尾部次数，Fisher 精确检验（记录位置的 {store['located_runs']} 对 {tag['located_runs']} 次） | "
                 f"{store['lower_left_tails']} / {tag['lower_left_tails']} | {tests['fisher_lower_left_tails'][1]:.3f} | "
                 f"{tests['fisher_lower_left_tails'][0]:.3f} |")
    for key, text in (("largest", "最大速度"), ("root_mean_square", "均方根速度")):
        test = tests[f"mann_whitney_{key}"]
        lines.append(f"| {text}，Mann–Whitney U | U = {test['U']:.0f} | {test['p_two_sided']:.3f} | {test['p_store_greater']:.3f} |")
    (TABLES / "round3_hydrostatic_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (LOGS / "round3" / "hydrostatic_round3.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_SECONDARY, save, setup_style
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    colours = {"store": "#2a78d6", "tag": "#eb6834"}
    for axis, key, text in ((axes[0], "largest", "最大速度 / √(gH)（5 ≤ t ≤ 10 平均）"),
                            (axes[1], "root_mean_square", "均方根速度 / √(gH)（同上）")):
        for position, build in enumerate(("store", "tag")):
            runs = result["runs"][build]
            values = np.array([run[key] for run in runs])
            jitter = (np.arange(values.size) % 9 - 4) * 0.035
            located = np.array([run["corner_share"] is not None for run in runs])
            axis.scatter(position + jitter[located], values[located], s=22, color=colours[build], edgecolor="white",
                         linewidth=0.6, zorder=3, label=f"{LABELS[build]}（记录位置）" if key == "largest" else None)
            axis.scatter(position + jitter[~located], values[~located], s=22, facecolor="white",
                         edgecolor=colours[build], linewidth=1.0, zorder=3,
                         label=f"{LABELS[build]}（第二轮，未记录位置）" if key == "largest" else None)
            quartiles = np.percentile(values, [25, 50, 75])
            axis.plot([position - 0.3, position + 0.3], [quartiles[1]] * 2, color=INK_SECONDARY, linewidth=2)
            axis.plot([position - 0.2, position + 0.2], [quartiles[0]] * 2, color=INK_SECONDARY, linewidth=1)
            axis.plot([position - 0.2, position + 0.2], [quartiles[2]] * 2, color=INK_SECONDARY, linewidth=1)
        if key == "largest":
            axis.axhline(TAIL_THRESHOLD, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([LABELS["store"], LABELS["tag"]])
        axis.set_xlim(-0.6, 1.6)
        axis.set_ylabel(text)
    axes[0].legend(loc="upper right", fontsize=8)
    figure.suptitle("倾斜 30° 静水盒（2-D 250²，t = 10），每个点一次运行；横线：中位数和四分位；虚线：尾部门槛 0.005")
    FIGURES.mkdir(parents=True, exist_ok=True)
    save(figure, FIGURES / "round3_hydrostatic.png")
    print(json.dumps({build: {key: value for key, value in statistics.items() if key in ("tails", "located_runs",
                                                                                      "lower_left_tails")}
                      for build, statistics in result["statistics"].items()}), json.dumps(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
