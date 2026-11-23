"""
round5_tables.py - the tables of claude/wall_store_report.md section 11 (wall store round 5).

    .venv/Scripts/python.exe -m experiment.wall_crossing.round5_tables lanes      # 11.3: the rule's threshold cases
    .venv/Scripts/python.exe -m experiment.wall_crossing.round5_tables physics    # 11.3: 3-D n50 k = 4 against k = 1
    .venv/Scripts/python.exe -m experiment.wall_crossing.round5_tables summary    # 11.6: the nine cases, final default
    .venv/Scripts/python.exe -m experiment.wall_crossing.round5_tables soak       # 11.2: the soak runs

Inputs: logs/wall_store/round5/timing/<case>/summary.json (kernel_timing, whole step, three rounds),
logs/wall_store/round4/timing{2,3}/<case>/summary.json (round 4's campaigns), logs/wall_store/round5/box_run/,
logs/wall_store/round5/soak/device*/runs.jsonl, logs/wall_store/round5/rule_inputs_new.json. Tables go to
claude/wall_store/tables/round5_*.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
ROUND4 = REPOSITORY_ROOT / "logs" / "wall_store" / "round4"
ROUND5 = REPOSITORY_ROOT / "logs" / "wall_store" / "round5"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
NEW_CASES = (("3-D n50（51³）", "c3d_n50", "cases/cavity3d_n50_crossing/case.yaml", False),
             ("3-D n70（71³）", "c3d_n70", "cases/cavity3d_n70_crossing/case.yaml", False),
             ("2-D 2000²", "n2000", "cases/lid_driven_cavity_2d_n2000_crossing/case.yaml", True))


def labels(path: pathlib.Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["labels"] if (path / "summary.json").exists() else {}


def step(entries: dict, label: str, dual: bool = False):
    entry = entries.get(label)
    if not entry:
        return None
    if dual:
        rate = entry["steady_fps"]
        return 1e6 / rate["median"], 1e6 / rate["maximum"], 1e6 / rate["minimum"]
    value = entry["kernels"]["step_total_us"]
    return value["median_us"], value["minimum_us"], value["maximum_us"]


def cell(value) -> str:
    return "—" if value is None else f"{value[0]:,.1f}（{value[1]:,.1f}–{value[2]:,.1f}）"


def change(value, reference) -> str:
    return "—" if value is None or reference is None else f"{100 * (value[0] / reference[0] - 1):+.1f} %"


def lanes_table() -> int:
    inputs = json.loads((ROUND5 / "rule_inputs_new.json").read_text(encoding="utf-8"))
    lines = ["| 算例 | 近壁流体粒子 | 近壁线程束 / SM | 规则选的 k | simple | simple (a) | k = 1 | k = 2 | k = 4 | 最好的 k | 规则选的 k 相对最好的 | 缺省（规则）相对 simple · 相对 simple (a) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    result = {}
    for text, key, case, two_dimensional in NEW_CASES:
        entries = labels(ROUND5 / "timing" / key)
        rule = inputs[case][0]
        rows = {k: step(entries, f"k{k}") for k in (1, 2, 4)}
        best = min(rows, key=lambda k: rows[k][0])
        chosen = rule["rule_k"]
        simple, simple_a, default = step(entries, "simple"), step(entries, "simple_a"), step(entries, "default")
        lines.append(
            f"| {text} | {rule['near_wall_fluid']:,} | {rule['near_wall_warps_per_sm']:.1f} | {chosen} | {cell(simple)} | "
            f"{cell(simple_a)} | {cell(rows[1])} | {cell(rows[2])} {change(rows[2], rows[1])} | "
            f"{cell(rows[4])} {change(rows[4], rows[1])} | {best} | {change(rows[chosen], rows[best])} | "
            f"{default[0] / simple[0]:.3f} · {'—' if simple_a is None else f'{default[0] / simple_a[0]:.3f}'} |")
        result[key] = {"rule": rule, "k": {str(k): rows[k] for k in rows}, "best": best, "simple": simple,
                       "simple_a": simple_a, "default": default}
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "round5_lanes.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROUND5 / "lanes.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


def physics_table() -> int:
    from experiment.wall_crossing.default_change_round4 import box_entry
    groups = {k: [f"n50_k{k}_r{index}" for index in (1, 2, 3)] for k in (4, 1)}
    present = {name: box_entry(ROUND5 / "box_run" / name) for names in groups.values() for name in names}
    rows = []
    for k, names in groups.items():
        for index, name in enumerate(names, 1):
            entry = present[name]
            rows.append(f"| k = {k}，第 {index} 次 | {entry['time']:.2f} | {entry['kinetic_energy']:.5f} | {entry['u_min']:.5f} | "
                        f"{entry['v_max']:.5f} | {entry['v_min']:.5f} | {entry['density_mean'] - 1000.0:+.5f} |")
    keys = (("kinetic_energy", 5), ("u_min", 5), ("v_max", 5), ("v_min", 5), ("density_mean", 5))

    def values(k, key):
        return [present[name][key] for name in groups[k]]
    rows.append("| 差：k = 4 平均 − k = 1 平均 | | " + " | ".join(
        f"{np.mean(values(4, key)) - np.mean(values(1, key)):+.{digits}f}" for key, digits in keys) + " |")
    rows.append("| 运行间散布（极差）：k = 1 三次 / k = 4 三次 | | " + " | ".join(
        f"{np.ptp(values(1, key)):.{digits}f} / {np.ptp(values(4, key)):.{digits}f}" for key, digits in keys) + " |")
    profile = {}
    for line in ("u", "v"):
        means = {k: np.mean([np.asarray(present[name][line]) for name in groups[k]], axis=0) for k in groups}
        same = [float(np.sqrt(np.nanmean((np.asarray(present[a][line]) - np.asarray(present[b][line])) ** 2)))
                for k in groups for position, a in enumerate(groups[k]) for b in groups[k][position + 1:]]
        profile[line] = {"between": float(np.sqrt(np.nanmean((means[4] - means[1]) ** 2))),
                         "same_median": float(np.median(same)), "same_largest": max(same)}
    text = ("| 3-D n50 方腔（51³，总体积修正打开），单卡，t = 2 | 时刻 | 动能（J） | u_min（竖直中心线） | v_max（水平中心线） | v_min | 平均密度 − ρ0 |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            f"两组平均剖面之差 L2：u(y) {profile['u']['between']:.5f}、v(x) {profile['v']['between']:.5f}；同一组两两之差 L2（中位数 / 最大）："
            f"u(y) {profile['u']['same_median']:.5f} / {profile['u']['same_largest']:.5f}、v(x) {profile['v']['same_median']:.5f} / "
            f"{profile['v']['same_largest']:.5f}（盖速为单位）。\n")
    (TABLES / "round5_n50_lanes_physics.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def summary_table() -> int:
    """The final default on the nine cases: crossing's whole step against simple and simple with (a); each row from
    one campaign (2-D 250^2-1000^2 and the dual: round 4's third, the lanes4 entry = the final default there; 3-D 1M
    single / dual: round 4's second, the default entry; n50, n70, 2000^2: round 5's, the default entry)."""
    rows_spec = (("2-D 250²", ROUND4 / "timing3" / "n250", "lanes4", False, "4"),
                 ("2-D 500²", ROUND4 / "timing3" / "n500", "lanes4", False, "4"),
                 ("2-D 1000²", ROUND4 / "timing3" / "n1000", "lanes4", False, "4"),
                 ("2-D 2000²", ROUND5 / "timing" / "n2000", "default", False, "4"),
                 ("2-D 1000²，双卡", ROUND4 / "timing3" / "dual_n1000", "lanes4", True, "4"),
                 ("3-D n50（51³）", ROUND5 / "timing" / "c3d_n50", "default", False, "1"),
                 ("3-D n70（71³）", ROUND5 / "timing" / "c3d_n70", "default", False, "1"),
                 ("3-D 1M（101³）", ROUND4 / "timing2" / "c3d", "default", False, "1"),
                 ("3-D 1M，双卡", ROUND4 / "timing2" / "dual_c3d", "default", True, "1"))
    lines = ["| 算例 | 卡数 | 规则选的 k | simple µs | simple (a) µs | crossing 最终缺省 µs | 相对 simple | 相对 simple (a) |",
             "|---|---|---|---|---|---|---|---|"]
    result = {}
    for text, path, label, dual, lanes in rows_spec:
        entries = labels(path)
        simple, simple_a, final = step(entries, "simple", dual), step(entries, "simple_a", dual), step(entries, label, dual)
        lines.append(f"| {text} | {2 if dual else 1} | {lanes} | {simple[0]:,.1f} | "
                     f"{'—' if simple_a is None else f'{simple_a[0]:,.1f}'} | {cell(final)} | {final[0] / simple[0]:.3f} | "
                     f"{'—' if simple_a is None else f'{final[0] / simple_a[0]:.3f}'} |")
        result[text] = {"simple": simple, "simple_a": simple_a, "final": final, "source": str(path.relative_to(REPOSITORY_ROOT))}
    (TABLES / "round5_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROUND5 / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


def soak_table() -> int:
    records = []
    for path in sorted((ROUND5 / "soak").glob("device*/runs.jsonl")):
        records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {}
    for record in records:
        groups.setdefault(record["label"], []).append(record)
    lines = ["| 构建 | 卡 | 次数 | 总帧数 | 每次的帧数 | 停住 | 恢复的停顿 | 其他失败 | 最大帧间隔 s（帧号） | 帧 / 秒（中位数） |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    result = {}
    for label in sorted(groups):
        for device in sorted({record["device"] for record in groups[label]}):
            chosen = [record for record in groups[label] if record["device"] == device]
            stalled = [record for record in chosen if record["status"].startswith("stall")]
            recovered = sum(1 for record in chosen for stall in record["stalls"] if stall.get("recovered"))
            other = [record for record in chosen if not record["status"].startswith(("ok", "stall"))]
            gap = max(chosen, key=lambda record: record["largest_completion_gap_s"])
            lines.append(f"| {label} | {device} | {len(chosen)} | {sum(record['frames_done'] for record in chosen):,} | "
                         f"{chosen[0]['frames_planned']:,} | {len(stalled)} | {recovered} | {len(other)} | "
                         f"{gap['largest_completion_gap_s']:.3f}（{gap['largest_gap_frame']}） | "
                         f"{np.median([record['frames_per_s'] for record in chosen]):,.0f} |")
            result[f"{label}_device{device}"] = {"runs": len(chosen), "frames": sum(r["frames_done"] for r in chosen),
                                                 "stalled": len(stalled), "recovered": recovered,
                                                 "other": [r["status"] for r in other]}
    (TABLES / "round5_soak.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROUND5 / "soak_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


def a2_history() -> int:
    """11.5: the 2-D 500^2 runs' mean density, mean pressure, theta and I (field_history.jsonl,
    volume_constraint.jsonl): window means t = 80-100 and a figure of the whole run."""
    runs = (("round5", ROUND5 / "remote" / "cavity" / "r5_n500", "最终缺省（0b92959，远端）"),
            ("store", REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "remote" / "store" / "cavity" / "r3_n500_store",
             "壁面独立存储（21877e3，远端，第 9 节）"),
            ("pool", REPOSITORY_ROOT / "logs" / "wall_crossing" / "cavity" / "n500_reflection_full_volume",
             "粒子池版本（第 15 节，无积分项）"))

    def read(path: pathlib.Path) -> list:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["| 2-D 500² 方腔，t = 80–100 平均 | 平均密度 − ρ0（kg/m³） | 平均压力（Pa） | θ | I | 全程 θ 的最大绝对值（t ≥ 5） |",
             "|---|---|---|---|---|---|"]
    loaded = {}
    for key, directory, text in runs:
        field, constraint = read(directory / "field_history.jsonl"), read(directory / "volume_constraint.jsonl")
        constraint = [record for record in constraint if "theta" in record]
        loaded[key] = (field, constraint, text)
        window = [record for record in field if 80.0 - 1e-9 <= record["time"] <= 100.0 + 1e-9]
        chosen = [record for record in constraint if 80.0 - 1e-9 <= record["time"] <= 100.0 + 1e-9]
        integral = [record["integral"] for record in chosen if "integral" in record]
        late = [abs(record["theta"]) for record in constraint if record["time"] >= 5.0]
        lines.append(f"| {text} | {np.mean([r['density_mean'] for r in window]) - 1000.0:+.5f} | "
                     f"{np.mean([r['pressure_mean'] for r in window]):+.2f} | {np.mean([r['theta'] for r in chosen]):+.2e} | "
                     f"{'—' if not integral else f'{np.mean(integral):+.2e}'} | {max(late):.2e} |")
    (TABLES / "round5_a2_history.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import save, setup_style
    setup_style()
    colours = {"round5": "#1baf7a", "store": "#2a78d6", "pool": "#eb6834"}
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for key, (field, constraint, text) in loaded.items():
        axes[0].plot([r["time"] for r in field], [r["density_mean"] - 1000.0 for r in field], color=colours[key],
                     linewidth=1.2, label=text)
        axes[1].plot([r["time"] for r in constraint], [r["theta"] for r in constraint], color=colours[key], linewidth=1.2)
        with_integral = [r for r in constraint if "integral" in r]
        if with_integral:
            axes[2].plot([r["time"] for r in with_integral], [r["integral"] for r in with_integral], color=colours[key],
                         linewidth=1.2)
    for axis, label in zip(axes, ("平均密度 − ρ0（kg/m³）", "θ", "I（积分项）")):
        axis.set_xlabel("t")
        axis.set_ylabel(label)
        axis.axvspan(80.0, 100.0, color="#e9e8e4", zorder=0)
    axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle("2-D 500² 方腔：平均密度、总体积修正的 θ 和 I（灰底：统计窗口 t = 80–100）", x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save(figure, REPOSITORY_ROOT / "claude" / "wall_store" / "figures" / "round5_a2_history.png")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("lanes", "physics", "summary", "soak", "a2_history"))
    arguments = parser.parse_args()
    return {"lanes": lanes_table, "physics": physics_table, "summary": summary_table, "soak": soak_table,
            "a2_history": a2_history}[arguments.action]()


if __name__ == "__main__":
    sys.exit(main())
