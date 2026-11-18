"""
long_validation.py - tables and figures of the long validation (2026-11-19): the short checks of its section 2
and, once they exist, the long runs of its section 3.

    .venv/Scripts/python.exe -m experiment.wall_crossing.long_validation short
    .venv/Scripts/python.exe -m experiment.wall_crossing.long_validation long

short:
  2.1  the diagnostic cases written out (their comment lines) and the three 700-step comparisons
       (outcome_classes restored_*);
  2.2  250^2, default + constraint, K = 1 and K = 2 to t = 5 (box_run k1_ / k2_n250_default_volume_r*): the first
       update of the constraint (target and current volume, theta) per run; theta, mean density and kinetic energy
       over time, K = 2 against K = 1 measured against the spread of the K = 1 runs; the invariants; the seam audit
       (seam_audit_run n250_crossing_default_volume) and the 3-D K = 2 bit comparison (k2_equivalence);
  2.3  the 3-D hydrostatic boxes, default + constraint, beside the runs of 2026-11-17 (first order, continuity
       wall component full) and adami;
  2.4  the 3-D cavity trial (box_run cavity3d_1m_crossing_trial_t5) and the speed probes of simple / adami.
Output: claude/wall_crossing/tables/long_*.md, claude/wall_crossing/figures/long_*.png,
claude/wall_crossing/long_validation/summary.json; one runs.jsonl record per call.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
OUT = REPORT_ROOT / "long_validation"
REST_DENSITY = 1000.0
K1_RUNS = [f"k1_n250_default_volume_r{index}" for index in (1, 2, 3)]
K2_RUNS = [f"k2_n250_default_volume_r{index}" for index in (1, 2, 3)]
RESTORED = (("crossing_quadratic_full", "db3d609", "二阶（全二阶），11-18 建"),
            ("crossing_delta_wall_rest_density", "81ae0d9", "诊断 delta_wall rest_density，11-17 建（连续性取完整向量的时期）"),
            ("crossing_reflection", "bd9fec1", "反射版（只写了逐对几何和两个反射），11-18 建（法向分量的时期）"))
HYDROSTATIC = (("3d_n48_gy_crossing_default_volume", "gy", "新缺省（完整方案）+ 约束"),
               ("3d_n48_gy_crossing", "gy", "11-17：一阶，连续性取完整向量"),
               ("3d_n48_gy_adami", "gy", "adami"),
               ("3d_n48_gtilt_crossing_default_volume", "gtilt", "新缺省（完整方案）+ 约束"),
               ("3d_n48_gtilt_crossing", "gtilt", "11-17：一阶，连续性取完整向量"),
               ("3d_n48_gtilt_adami", "gtilt", "adami"))


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(name: str, text: str) -> None:
    path = REPORT_ROOT / "tables" / name
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[long_validation] wrote {path.relative_to(REPOSITORY_ROOT)}")


def read_series(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] not in ("", None) else np.nan for row in rows])
            for key in rows[0]}


def read_jsonl(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def constraint_records(path: pathlib.Path) -> list:
    """The update records of a box_run volume constraint log with absolute steps (marker lines give the offset
    of a resumed segment)."""
    records, offset = [], 0
    for record in read_jsonl(path):
        if "segment_start_step" in record:
            offset = int(record["segment_start_step"])
            continue
        records.append(dict(record, absolute_step=offset + int(record["step"])))
    return records


def pair_differences(series_list: list, key: str) -> tuple:
    """(max over pairs and times of |a - b|, rms over pairs and times) for aligned series."""
    values = [np.asarray(series[key], dtype=np.float64) for series in series_list]
    length = min(len(value) for value in values)
    differences = [values[first][:length] - values[second][:length]
                   for first in range(len(values)) for second in range(first + 1, len(values))]
    if not differences:
        return float("nan"), float("nan")
    stacked = np.concatenate(differences)
    return float(np.nanmax(np.abs(stacked))), float(np.sqrt(np.nanmean(stacked ** 2)))


def cross_differences(first_list: list, second_list: list, key: str) -> tuple:
    values_first = [np.asarray(series[key], dtype=np.float64) for series in first_list]
    values_second = [np.asarray(series[key], dtype=np.float64) for series in second_list]
    length = min(len(value) for value in values_first + values_second)
    differences = [first[:length] - second[:length] for first in values_first for second in values_second]
    stacked = np.concatenate(differences)
    return float(np.nanmax(np.abs(stacked))), float(np.sqrt(np.nanmean(stacked ** 2)))


# =========================================================================================================== short
def short() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style

    summary = {}
    with LoggedRun(kind="analysis", case="none") as record:
        record.extra["label"] = "long_validation short (section 2 of the long validation)"
        summary["run_id"] = record.run_id

        # ------------------------------------------------------------------ 2.1 diagnostic cases written out
        written = {}
        for path in sorted((REPOSITORY_ROOT / "cases").rglob("case.yaml")):
            text = path.read_text(encoding="utf-8")
            lines = [line for line in text.splitlines() if "written out 2026-11-19" in line]
            if not lines:
                continue
            name = str(path.parent.relative_to(REPOSITORY_ROOT / "cases")).replace("\\", "/")
            commit = re.search(r"made \(([0-9a-f]{7})", lines[0]).group(1)
            keys = [line.split(":")[0].strip() + " " + line.split(":")[1].split("#")[0].strip() for line in lines]
            written[name] = {"commit": commit, "keys": keys}
        by_keys: dict = {}
        for name, entry in written.items():
            by_keys.setdefault("；".join(entry["keys"]), []).append(f"`{name}`（{entry['commit']}）")
        rows = [[f"{len(names)}", "<br>".join(names), keys] for keys, names in
                sorted(by_keys.items(), key=lambda item: -len(item[1]))]
        write_table("long_restored_cases.md", markdown_table(["个数", "算例（建立它的提交）", "补写的键和值"], rows))
        rows = []
        for variant, commit, text in RESTORED:
            path = LOG_ROOT / "outcome_classes" / f"restored_{variant}_against_{commit}_s700" / "classes.json"
            if not path.exists():
                continue
            classes = json.loads(path.read_text(encoding="utf-8"))
            members = "；".join(f"类 {index}：参照 {group['reference_runs']} 次，本次 {group['case_runs']} 次"
                               for index, group in enumerate(classes["classes"]))
            rows.append([f"`..._{variant}`", text, f"{commit} 上的同名算例（当时的文件、当时的求解器）",
                         f"{len(classes['classes'])} 个结果类。{members}",
                         "逐位相同" if classes["case_within_reference_classes"] else "**不同**", classes["run_id"]])
        write_table("long_restored_bitwise.md",
                    markdown_table(["算例（补写之后，当前提交）", "说明", "参照", "结果", "结论", "记录"], rows))
        summary["restored"] = {"cases": len(written)}

        # ------------------------------------------------------------------ 2.2 K = 1 against K = 2
        runs = {name: LOG_ROOT / "box_run" / name for name in K1_RUNS + K2_RUNS}
        present = {name: directory for name, directory in runs.items() if (directory / "series.csv").exists()}
        series = {name: read_series(directory / "series.csv") for name, directory in present.items()}
        logs = {name: constraint_records(directory / "volume_constraint.jsonl") for name, directory in present.items()}
        reference_name = K1_RUNS[0]
        rows = []
        first = logs[reference_name][0]
        for name in present:
            log = logs[name]
            slabs = log[0]["slabs"]
            target_difference = abs(log[0]["target_volume"] - first["target_volume"]) / first["target_volume"]
            volume_difference = abs(log[0]["volume"] - first["volume"]) / first["volume"]
            rows.append([name, slabs, f"{log[0]['target_volume']:.16g}", f"{log[0]['volume']:.16g}",
                         f"{log[0]['theta']:.3g}", f"{target_difference:.1e} / {volume_difference:.1e}",
                         f"{log[1]['theta']:+.3e}" if len(log) > 1 else "—", len(log)])
        write_table("long_k2_first_update.md", markdown_table(
            ["运行", "片数 K", "V_target", "第一次更新（t = 0）的 V_now", "θ", "与 K = 1 第 1 次的相对差（V_target / V_now）",
             "第二次更新（t = 0.18）的 θ", "更新次数"], rows))
        summary["k2_first_update"] = {name: {"target": logs[name][0]["target_volume"], "volume": logs[name][0]["volume"],
                                             "theta": logs[name][0]["theta"]} for name in present}

        def theta_series(name: str) -> dict:
            return {"theta": np.array([entry["theta"] for entry in logs[name]])}

        k1 = [name for name in K1_RUNS if name in present]
        k2 = [name for name in K2_RUNS if name in present]
        rows = []
        quantities = (("θ（每次更新）", None, 1.0, "{:.2e}"),
                      ("平均密度 − ρ0（kg/m³）", "density_mean", 1.0, "{:.2e}"),
                      ("动能", "kinetic_energy", 1.0, "{:.2e}"))
        summary["k2_histories"] = {}
        for text, key, scale, form in quantities:
            if key is None:
                within_k1 = pair_differences([theta_series(name) for name in k1], "theta")
                within_k2 = pair_differences([theta_series(name) for name in k2], "theta")
                across = cross_differences([theta_series(name) for name in k2], [theta_series(name) for name in k1],
                                           "theta")
            else:
                within_k1 = pair_differences([series[name] for name in k1], key)
                within_k2 = pair_differences([series[name] for name in k2], key)
                across = cross_differences([series[name] for name in k2], [series[name] for name in k1], key)
            rows.append([text, f"{form.format(within_k1[0])}（rms {form.format(within_k1[1])}）",
                         f"{form.format(within_k2[0])}（rms {form.format(within_k2[1])}）",
                         f"{form.format(across[0])}（rms {form.format(across[1])}）",
                         f"{across[1] / within_k1[1]:.2f}" if within_k1[1] > 0 else "—"])
            summary["k2_histories"][text] = {"within_k1": within_k1, "within_k2": within_k2, "across": across}
        write_table("long_k2_histories.md", markdown_table(
            ["量（t = 0 到 5）", "K = 1 各次之间：最大差（rms）", "K = 2 各次之间", "K = 2 对 K = 1：最大差（rms）",
             "rms 之比（跨组 ÷ K = 1 组内）"], rows))
        rows = []
        for name in present:
            data = series[name]
            rows.append([name, f"{data['time'][-1]:.2f}", f"{int(data['alive'][0]):,} → {int(data['alive'][-1]):,}",
                         int(np.nanmax(data["outside_box_count"])), int(np.nanmax(data["overflow"])),
                         f"{np.nanmax(data['speed_maximum']):.4f}", f"{data['steps_per_second'][-1]:.0f}"])
        write_table("long_k2_invariants.md", markdown_table(
            ["运行", "到达的 t", "粒子数（始 → 末）", "越出盒（最大）", "体素表溢出（最大）", "最大速度（最大）",
             "步 / 秒（显示卡上同时有别的作业时偏低）"], rows))

        setup_style()
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        for axis, (title, getter) in zip(axes, (
                ("θ（×10⁻⁶）", lambda name: ([entry["absolute_step"] * 3e-5 for entry in logs[name]],
                                           [1e6 * entry["theta"] for entry in logs[name]])),
                ("流体平均密度 − ρ0（kg/m³）", lambda name: (series[name]["time"],
                                                     series[name]["density_mean"] - REST_DENSITY)),
                ("流体动能", lambda name: (series[name]["time"], series[name]["kinetic_energy"])))):
            for name in present:
                times, values = getter(name)
                is_k2 = name.startswith("k2")
                axis.plot(times, values, color=INK_PRIMARY if is_k2 else INK_SECONDARY,
                          linewidth=0.9 if is_k2 else 2.4, alpha=1.0 if is_k2 else 0.45,
                          label=("K = 2" if is_k2 else "K = 1") if name.endswith("r1") else None)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        handles, labels = axes[0].get_legend_handles_labels()
        height = figure.get_figheight()
        figure.suptitle("250² 方腔，新缺省 + 体积约束：K = 1（宽的浅灰线，3 次）与 K = 2（细的黑线，3 次）", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=2,
                      handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
        figure.savefig(REPORT_ROOT / "figures" / "long_k2_histories.png")
        plt.close(figure)

        # seam audit and the 3-D bit comparison: their own summaries
        seam = LOG_ROOT / "seam_audit" / "n250_crossing_default_volume" / "summary.md"
        if seam.exists():
            (REPORT_ROOT / "tables" / "long_seam_audit.md").write_text(seam.read_text(encoding="utf-8"),
                                                                       encoding="utf-8", newline="\n")
        k2_3d = LOG_ROOT / "k2_equivalence" / "hydrostatic_3d_gtilt_default_volume_700" / "summary.md"
        if k2_3d.exists():
            (REPORT_ROOT / "tables" / "long_k2_3d.md").write_text(k2_3d.read_text(encoding="utf-8"),
                                                                  encoding="utf-8", newline="\n")

        # ------------------------------------------------------------------ 2.3 3-D hydrostatic boxes
        rows = []
        histories = {}
        for name, direction, text in HYDROSTATIC:
            path = LOG_ROOT / "hydrostatic" / name / "summary.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            final = data["final"]
            history = read_series(LOG_ROOT / "hydrostatic" / name / "series.csv")
            histories[name] = (history, data["scales"])
            density_offset = (f"{history['density_mean'][-1] - REST_DENSITY:+.3f}" if "density_mean" in history
                              else "—（当时没有记）")
            log = constraint_records(LOG_ROOT / "hydrostatic" / name / "volume_constraint.jsonl")
            rows.append([direction, text, f"{data['time']:.1f}", f"{final['speed_maximum_over_scale']:.2e}",
                         f"{final['speed_root_mean_square_over_scale']:.2e}", f"{final['pressure_level_over_scale']:+.3f}",
                         f"{final['pressure_error_root_mean_square_over_scale']:.2e}", density_offset,
                         f"{log[-1]['theta']:+.2e}" if log else "—", data["run_id"]])
        write_table("long_hydrostatic_3d.md", markdown_table(
            ["重力方向", "配置", "t", "末段最大速度 ÷ √(gH)", "均方根速度 ÷ √(gH)", "盒心压力电平 ÷ ρ0gH",
             "压力误差均方根（去电平）÷ ρ0gH", "平均密度 − ρ0（末，kg/m³）", "θ（末）", "记录"], rows))
        if histories:
            figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
            styles = {"新缺省（完整方案）+ 约束": (INK_PRIMARY, 1.2, "solid"),
                      "11-17：一阶，连续性取完整向量": (INK_SECONDARY, 2.2, "solid"),
                      "adami": (INK_SECONDARY, 1.0, (0, (4, 2)))}
            for name, direction, text in HYDROSTATIC:
                if name not in histories:
                    continue
                history, scales = histories[name]
                colour, width, style = styles[text]
                alpha = 1.0 if direction == "gy" else 0.5
                label = f"{text}，{direction}"
                axes[0].plot(history["time"], history["speed_root_mean_square"] / scales["velocity_sqrt_g_extent"],
                             color=colour, linewidth=width, linestyle=style, alpha=alpha, label=label)
                axes[1].plot(history["time"], history["pressure_level_at_centre"] / scales["pressure_rho_g_extent"],
                             color=colour, linewidth=width, linestyle=style, alpha=alpha)
            axes[0].set_yscale("log")
            axes[0].set_title("均方根速度 ÷ √(gH)")
            axes[1].set_title("盒心压力电平 ÷ ρ0gH")
            for axis in axes:
                axis.set_xlabel("时间（模拟单位）")
            handles, labels = axes[0].get_legend_handles_labels()
            height = figure.get_figheight()
            figure.suptitle("3-D 静水盒 48³：新缺省 + 约束与 11-17 的运行（gy 实线、gtilt 浅色）", x=0.01,
                            y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                            color=INK_PRIMARY)
            figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                          handlelength=2.6, labelcolor=INK_PRIMARY, borderaxespad=0.0, fontsize=8)
            figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.95 / height), w_pad=2.0)
            figure.savefig(REPORT_ROOT / "figures" / "long_hydrostatic_3d.png")
            plt.close(figure)

        # ------------------------------------------------------------------ 2.4 3-D cavity trial and speeds
        rows = []
        for name, text in (("cavity3d_1m_crossing_trial_t5", "crossing，新缺省 + 约束"),
                           ("cavity3d_1m_simple_speed", "simple（测速）"),
                           ("cavity3d_1m_adami_speed", "adami（测速）")):
            directory = LOG_ROOT / "box_run" / name
            if not (directory / "series.csv").exists():
                continue
            data = read_series(directory / "series.csv")
            segments = read_jsonl(directory / "segments.jsonl")
            log = constraint_records(directory / "volume_constraint.jsonl")
            rates = data["steps_per_second"][1:]
            rows.append([text, f"{data['time'][-1]:.2f}", f"{int(data['alive'][0]):,} → {int(data['alive'][-1]):,}",
                         int(np.nanmax(data["outside_box_count"])), f"{np.nanmax(data['speed_maximum']):.4f}",
                         f"{data['kinetic_energy'][-1]:.3f}", f"{data['density_mean'][-1] - REST_DENSITY:+.4f}",
                         f"{log[-1]['theta']:+.2e}" if log else "—", f"{np.nanmedian(rates):.1f}",
                         len(segments), "；".join(segment["status"][:60] for segment in segments)])
        if rows:
            write_table("long_cavity3d_trial.md", markdown_table(
                ["运行", "到达的 t", "粒子数（始 → 末）", "越出盒（最大）", "最大速度（最大）", "动能（末）",
                 "平均密度 − ρ0（末）", "θ（末）", "步 / 秒（中位数）", "段数", "各段"], rows))

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary_short.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                                encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("short", "long"))
    arguments = parser.parse_args()
    if arguments.action == "short":
        return short()
    raise SystemExit("long: written when the long runs exist")


if __name__ == "__main__":
    sys.exit(main())
