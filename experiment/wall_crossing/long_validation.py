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

        # the locating replicates (to t = 2, while the long runs held both cards): K = 1, K = 2 depth 2 / depth 1
        groups = (("K = 1", "k1_n250_default_volume_r{}", range(1, 7)),
                  ("K = 2，depth 2", "k2_n250_default_volume_r{}", range(1, 7)),
                  ("K = 2，depth 1", "k2d1_n250_default_volume_r{}", range(4, 7)))
        windows = ((0.03, 0.5), (0.5, 1.5), (1.5, 2.0))
        rows, grouped = [], {}
        for label, pattern, repeats in groups:
            for repeat in repeats:
                directory = LOG_ROOT / "box_run" / pattern.format(repeat)
                if not (directory / "series.csv").exists():
                    continue
                data = read_series(directory / "series.csv")
                log = constraint_records(directory / "volume_constraint.jsonl")
                cells = [label, f"r{repeat}"]
                for first, last in windows:
                    chosen = (data["time"] >= first) & (data["time"] < last)
                    theta = [entry["theta"] for entry in log if first <= entry["time"] < last]
                    density = float(np.mean(data["density_mean"][chosen] - REST_DENSITY))
                    cells.append(f"{density:+.4f} / {1e6 * np.mean(theta):+.2f}")
                    grouped.setdefault(label, {}).setdefault(first, []).append(density)
                cells += [f"{np.nanmedian(data['steps_per_second'][1:]):.0f}", f"{int(data['alive'][-1]):,}"]
                rows.append(cells)
        if rows:
            text = markdown_table(["组", "运行"] + [f"t ∈ [{first:g}, {last:g})：平均密度 − ρ0 / θ（×10⁻⁶）"
                                                    for first, last in windows]
                                  + ["步 / 秒（中位数，卡上负载的指标）", "粒子数（末）"], rows)
            text += "\n各组的平均 ± 标准差（平均密度 − ρ0，kg/m³）：\n\n" + markdown_table(
                ["组", "次数"] + [f"[{first:g}, {last:g})" for first, last in windows],
                [[label, f"{len(next(iter(values.values())))} 次"]
                 + [f"{np.mean(values[first]):+.4f} ± {np.std(values[first]):.4f}" for first, _ in windows]
                 for label, values in grouped.items()])
            write_table("long_k2_locate.md", text)
            summary["k2_locate"] = {label: {str(first): [float(np.mean(v)), float(np.std(v)), len(v)]
                                            for first, v in values.items()} for label, values in grouped.items()}

        # seam audit (after the fix of the constraint's start, f6ee476): the tool's own summaries
        parts = []
        for label, text in (("n250_crossing_default_volume_f6ee476", "2-D 250² 方腔，新缺省 + 约束"),
                            ("cavity3d_1m_crossing_default_volume", "3-D 方腔 1M，新缺省 + 约束")):
            seam = LOG_ROOT / "seam_audit" / label / "summary.md"
            if seam.exists():
                body = seam.read_text(encoding="utf-8").strip().splitlines()
                parts.append(f"**{text}**（`seam_audit/{label}`）\n\n" + "\n".join(
                    line for line in body if not line.startswith("#")) + "\n")
        if parts:
            (REPORT_ROOT / "tables" / "long_seam_audit.md").write_text("\n".join(parts), encoding="utf-8",
                                                                       newline="\n")
        # the canonical K = 2 dumps in 3-D (k2_equivalence): what the chain's dump reports
        rows = []
        for label, text in (("hydrostatic_3d_gtilt_default_volume_700", "3-D 静水盒 48³ gtilt，crossing 新缺省 + 约束"),
                            ("cavity3d_1m_crossing_default_volume_700", "3-D 方腔 1M，crossing 新缺省 + 约束"),
                            ("cavity3d_1m_simple_700_f6ee476", "3-D 方腔 1M，simple")):
            directory = LOG_ROOT / "k2_equivalence" / label
            if not (directory / "chain_s700.log").exists():
                continue
            chain = (directory / "chain_s700.log").read_text(encoding="utf-8", errors="replace")
            single = (directory / "single_s700.log").read_text(encoding="utf-8", errors="replace")
            alive_chain = re.search(r"alive (\d+)/(\d+)", chain)
            alive_single = re.search(r"alive (\d+)/(\d+)", single)
            warning = re.search(r"replica region ([0-9.]+) slots per voxel .*?= ([0-9.]+);", chain)
            migrations = re.search(r"'migration_install_count': \[(\d+), (\d+)\]", chain)
            rows.append([text, f"{int(alive_single.group(1)):,} / {int(alive_single.group(2)):,}" if alive_single else "—",
                         f"{int(alive_chain.group(1)):,} / {int(alive_chain.group(2)):,}" if alive_chain else "—",
                         f"{migrations.group(1)} / {migrations.group(2)}" if migrations else "—",
                         f"{warning.group(1)} 对 {warning.group(2)}" if warning else "—"])
        if rows:
            write_table("long_k2_3d.md", markdown_table(
                ["算例", "K = 1：700 步后存活 / 总数", "K = 2：700 步后存活 / 总数", "K = 2 两片的迁入粒子数",
                 "分区器的警告：每体素副本槽数 对 需要的 1.2 (h/dx)^d"], rows))

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
                if "pressure_level_at_centre" in history:
                    axes[1].plot(history["time"], history["pressure_level_at_centre"] / scales["pressure_rho_g_extent"],
                                 color=colour, linewidth=width, linestyle=style, alpha=alpha)
                else:
                    # the runs of 2026-11-17 kept the level at the end only (summary.json)
                    final = json.loads((LOG_ROOT / "hydrostatic" / name / "summary.json").read_text(
                        encoding="utf-8"))["final"]["pressure_level_over_scale"]
                    axes[1].plot([history["time"][-1]], [final], marker="o" if text.startswith("11-17") else "s",
                                 color=colour, alpha=alpha, markersize=7, linestyle="none")
            axes[0].set_yscale("log")
            axes[0].set_title("均方根速度 ÷ √(gH)")
            axes[1].set_title("盒心压力电平 ÷ ρ0gH（11-17 的运行只有末值：圆点 / 方块）")
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


# ============================================================================================================ long
SERIES_2D = (("n250_reflection_full_volume", 250), ("n500_reflection_full_volume", 500),
             ("n1000_complete_volume", 1000))
CONTEXT_2D = (("n250_crossing_normal", "250²，一阶（11-18 的缺省）"), ("n500_crossing_normal", "500²，一阶（11-18 的缺省）"),
              ("n250_adami", "250²，adami"), ("n500_adami", "500²，adami"), ("n1000_adami", "1000²，adami（远端）"),
              ("n250_simple", "250²，simple"), ("n500_simple", "500²，simple"))
# simple 1000^2 of docs/validation/cavity_re1000.md (main series: xi 0.001, eps^2 0.0025 h^2, the defaults of the
# cases since f2b6d67; v6 solver 9cde618, K = 2; normalisation of that document: the wall frame). Its 250^2 and
# 500^2 profile errors equal this series' simple runs (n250_simple, n500_simple) to 3-4 digits.
SIMPLE_1000_DOCUMENT = {"psi_wall_percent": -4.0, "extrema_percent": "−3.8 到 −4.1", "u_l2": 0.0108, "v_l2": 0.0126,
                        "commit": "9cde618（v6，K = 2；ξ / ε² 缺省 f2b6d67）"}
EXTREMA = ("u_min", "v_max", "v_min")
MODES_3D = (("cavity3d_1m_crossing_long", "crossing（新缺省 + 约束）"), ("cavity3d_1m_simple_long", "simple"),
            ("cavity3d_1m_adami_long", "adami"))
WINDOW_3D = (35.0, 45.0)


def convergence(errors: list, spacings: list) -> dict:
    """Three errors (signed, %) at spacings h, h/2, h/4: two-point orders; (a) e = c + C h^p fitted exactly;
    (b) e = C h^p fitted by least squares in log-log (the errors must share their sign)."""
    e1, e2, e3 = errors
    result = {"orders": [float(np.log(abs(e1 / e2)) / np.log(2.0)) if e2 else float("nan"),
                         float(np.log(abs(e2 / e3)) / np.log(2.0)) if e3 else float("nan")]}
    ratio = (e1 - e2) / (e2 - e3) if e2 != e3 else float("nan")
    if ratio > 0:
        p = float(np.log(ratio) / np.log(2.0))
        constant_coefficient = (e1 - e2) / (spacings[0] ** p - spacings[1] ** p)
        offset = e1 - constant_coefficient * spacings[0] ** p
        result["a"] = {"p": p, "C": float(constant_coefficient), "c": float(offset)}
    else:
        result["a"] = {"p": float("nan"), "C": float("nan"), "c": float("nan"),
                       "note": "the differences change sign: no c + C h^p"}
    if (np.sign(errors) == np.sign(errors[0])).all():
        slope, intercept = np.polyfit(np.log(spacings), np.log(np.abs(errors)), 1)
        fitted = np.exp(intercept) * np.asarray(spacings) ** slope
        result["b"] = {"p": float(slope), "C": float(np.sign(errors[0]) * np.exp(intercept)),
                       "residual_log": float(np.sqrt(np.mean((np.log(np.abs(errors)) - np.log(fitted)) ** 2)))}
    else:
        result["b"] = {"p": float("nan"), "note": "the errors change sign"}
    return result


def steady_step_rate(data: dict, from_time: float = 10.0) -> float:
    """The step rate of a box_run run over its rows with t >= from_time: box_run's steps_per_second is the average
    since the start of the segment, so the elapsed time of row i is (step_i - start) / rate_i; the rate between the
    first and the last of these rows of the last segment is the ratio of the step and elapsed differences."""
    segment = data["segment"]
    last = np.nanmax(segment)
    rows = np.flatnonzero((segment == last) & (data["time"] >= from_time) & (data["steps_per_second"] > 0))
    if rows.size < 2:
        rows = np.flatnonzero((segment == last) & (data["steps_per_second"] > 0))
        if rows.size < 2:
            return float("nan")
    # the segment's start step: 0 for the first segment; for a resumed one the checkpoint it started from, which is
    # the last row kept of the segments before it
    first_row = np.flatnonzero(segment == last)[0]
    earlier = np.flatnonzero((segment < last) & (data["step"] < data["step"][first_row]))
    start = data["step"][earlier[-1]] if earlier.size else 0.0
    elapsed = (data["step"][rows] - start) / data["steps_per_second"][rows]
    return float((data["step"][rows[-1]] - data["step"][rows[0]]) / (elapsed[-1] - elapsed[0]))


def window_mean_profiles(profiles: list, window: tuple) -> dict:
    chosen = [entry for entry in profiles if window[0] - 1e-9 <= entry["time"] <= window[1] + 1e-9]
    result = {"count": len(chosen)}
    for line in ("u_along_y", "v_along_x"):
        result[line] = {"coordinate": np.asarray(chosen[0][line]["coordinate"]),
                        "velocity": np.nanmean([entry[line]["velocity"] for entry in chosen], axis=0)}
    return result


def long_tables() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style

    tables = json.loads((REPORT_ROOT / "cavity" / "tables.json").read_text(encoding="utf-8"))["runs"]
    summary = {}
    with LoggedRun(kind="analysis", case="none") as record:
        record.extra["label"] = "long_validation long (section 3 of the long validation)"
        summary["run_id"] = record.run_id
        setup_style()

        # ------------------------------------------------------------------ 3.1 metrics beside 250^2 and 500^2
        rows = []

        def metric_row(name: str, text: str) -> list | None:
            if name not in tables:
                return None
            entry = tables[name]
            psi = entry["extrema"]["psi_min"]
            errors = entry["errors"]
            return [text, f"{entry['window'][0]:.1f}–{entry['window'][1]:.1f}",
                    f"{psi['deviation_mid_percent']:+.2f} %", f"{psi['deviation_wall_percent']:+.2f} %",
                    *[f"{entry['extrema'][key]['deviation_percent']:+.2f} %" for key in EXTREMA],
                    f"{errors['wall_u']['L2']:.4f} / {errors['wall_u']['Linf']:.4f}",
                    f"{errors['wall_v']['L2']:.4f} / {errors['wall_v']['Linf']:.4f}",
                    f"({psi['x_mid']:.4f}, {psi['y_mid']:.4f})",
                    f"({psi['x_mid'] - psi['x_reference']:+.4f}, {psi['y_mid'] - psi['y_reference']:+.4f})"]

        for name, resolution in SERIES_2D:
            row = metric_row(name, f"**{resolution}²，完整方案 + 约束（新缺省）**")
            if row:
                rows.append(row)
        for name, text in CONTEXT_2D:
            row = metric_row(name, text)
            if row:
                rows.append(row)
        rows.append([f"1000²，simple（{SIMPLE_1000_DOCUMENT['commit']}，docs/validation/cavity_re1000.md）", "80.0–99.8",
                     "—", f"{SIMPLE_1000_DOCUMENT['psi_wall_percent']:+.1f} %", SIMPLE_1000_DOCUMENT["extrema_percent"] + " %",
                     "", "", f"{SIMPLE_1000_DOCUMENT['u_l2']:.4f} / —", f"{SIMPLE_1000_DOCUMENT['v_l2']:.4f} / —", "—", "—"])
        write_table("long_n1000_metrics.md", markdown_table(
            ["运行", "统计窗口", "ψ_min（mid 帧：壁在盒面）", "ψ_min（wall 帧）", "u_min", "v_max", "v_min",
             "u 剖面 L2 / L∞（wall 帧）", "v 剖面 L2 / L∞", "涡心（mid 帧）", "涡心偏差"], rows))

        # ------------------------------------------------------------------ 3.1 three-point orders
        present = [name for name, _ in SERIES_2D if name in tables]
        if len(present) == 3:
            spacings = [1.0 / 250.0, 1.0 / 500.0, 1.0 / 1000.0]
            quantities = {"ψ_min（mid 帧）": [tables[name]["extrema"]["psi_min"]["deviation_mid_percent"] for name in present],
                          "ψ_min（wall 帧）": [tables[name]["extrema"]["psi_min"]["deviation_wall_percent"] for name in present]}
            for key in EXTREMA:
                quantities[key] = [tables[name]["extrema"][key]["deviation_percent"] for name in present]
            rows = []
            summary["convergence"] = {}
            for text, errors in quantities.items():
                fit = convergence(errors, spacings)
                summary["convergence"][text] = {"errors": errors, **fit}
                a, b = fit["a"], fit["b"]
                rows.append([text, " / ".join(f"{value:+.3f}" for value in errors),
                             f"{fit['orders'][0]:.2f} / {fit['orders'][1]:.2f}",
                             (f"p = {a['p']:.2f}，c = {a['c']:+.3f} %" if np.isfinite(a["p"]) else a.get("note", "—")),
                             (f"p = {b['p']:.2f}，对数残差 {b['residual_log']:.3f}" if np.isfinite(b.get("p", float("nan")))
                              else b.get("note", "—"))])
            write_table("long_convergence.md", markdown_table(
                ["量（相对 Marchi 2021 的偏差，%）", "250² / 500² / 1000²", "两点阶（250→500 / 500→1000）",
                 "(a) 偏差 = c + C·Δx^p（三点精确解）", "(b) 偏差 = C·Δx^p（对数最小二乘）"], rows))

        # ------------------------------------------------------------------ 3.1 histories of the 1000^2 run
        directory = LOG_ROOT / "cavity" / "n1000_complete_volume"
        history = [row for row in read_jsonl(directory / "field_history.jsonl") if row.get("step") is not None]
        if history:
            by_step = {}
            for row in history:
                by_step[row["step"]] = row
            history = [by_step[step] for step in sorted(by_step)]
            times = np.array([row["time"] for row in history])
            density = np.array([row["density_mean"] - REST_DENSITY for row in history])
            pressure = np.array([row["pressure_mean"] / 1000.0 for row in history])
            negative = np.array([row["pressure_negative_fraction"] for row in history])
            log = read_jsonl(directory / "volume_constraint.jsonl")
            samples = read_jsonl(directory / "samples.jsonl")
            rows = []
            for first, last in ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.05)):
                chosen = (times > first) & (times <= last)
                if chosen.sum() < 5:
                    continue
                theta = [entry["theta"] for entry in log if first < entry["time"] <= last]
                fps = [entry["fps"] for entry in samples if first < entry["time"] <= last and entry.get("fps")]
                from experiment.wall_crossing.pairwise_reflection import high_pass_fluctuation
                rows.append([f"{first:g}–{min(last, 100):g}", f"{density[chosen].mean():+.4f} ± {density[chosen].std():.4f}",
                             f"{np.mean(theta):+.2e}" if theta else "—", f"{pressure[chosen].mean():+.4f}",
                             f"{100 * negative[chosen].mean():.1f} %", f"{high_pass_fluctuation(pressure[chosen]):.4f}",
                             f"{np.median(fps):.0f}" if fps else "—"])
            write_table("long_n1000_history.md", markdown_table(
                ["时间段", "平均密度 − ρ0（kg/m³）", "θ（平均）", "平均压力（ρ0U²）", "负压粒子比例",
                 "平均压力的涨落（高通，ρ0U²）", "步 / 秒（中位数）"], rows))
            figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
            axes[0].plot(times, density, color=INK_PRIMARY, linewidth=0.8)
            axes[0].set_title("平均密度 − ρ0（kg/m³）")
            axes[1].plot([entry["time"] for entry in log], [1e6 * entry["theta"] for entry in log], color=INK_PRIMARY,
                         linewidth=0.8)
            axes[1].set_title("θ（×10⁻⁶）")
            axes[2].plot(times, pressure, color=INK_PRIMARY, linewidth=0.8)
            axes[2].set_title("平均压力（ρ0U²）")
            for axis in axes:
                axis.set_xlabel("时间（模拟单位）")
            height = figure.get_figheight()
            figure.suptitle("Re = 1000 方腔 1000²，新缺省（完整方案）+ 约束", x=0.01, y=1.0 - 0.05 / height, ha="left",
                            va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
            figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.4 / height), w_pad=2.0)
            figure.savefig(REPORT_ROOT / "figures" / "long_n1000_history.png")
            plt.close(figure)

        # ------------------------------------------------------------------ 3.2 the 3-D cavity, three modes
        runs = {}
        for name, text in MODES_3D:
            directory = LOG_ROOT / "box_run" / name
            if not (directory / "series.csv").exists():
                continue
            runs[name] = {"text": text, "series": read_series(directory / "series.csv"),
                          "profiles": json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
                          if (directory / "profiles.json").exists() else [],
                          "segments": read_jsonl(directory / "segments.jsonl"),
                          "stop": json.loads((directory / "watch_stop.json").read_text(encoding="utf-8"))
                          if (directory / "watch_stop.json").exists() else None,
                          "log": constraint_records(directory / "volume_constraint.jsonl")}
        rows = []
        means = {}
        for name, run in runs.items():
            data = run["series"]
            in_window = (data["time"] >= WINDOW_3D[0] - 1e-9) & (data["time"] <= WINDOW_3D[1] + 1e-9)
            kinetic = data["kinetic_energy"][in_window]
            slope = (float(np.polyfit(data["time"][in_window], kinetic, 1)[0]) if in_window.sum() > 5 else float("nan"))
            rows.append([run["text"], f"{data['time'][-1]:.2f}",
                         f"{int(data['alive'][0]):,} → {int(data['alive'][-1]):,}",
                         int(np.nanmax(data["outside_box_count"])), f"{np.nanmax(data['speed_maximum']):.4f}",
                         f"{kinetic.mean():.3f} ± {kinetic.std():.3f}" if in_window.any() else "—",
                         f"{slope:+.4f}" if np.isfinite(slope) else "—",
                         f"{data['density_mean'][-1] - REST_DENSITY:+.4f}",
                         f"{run['log'][-1]['theta']:+.2e}" if run["log"] else "—",
                         f"{steady_step_rate(data):.1f}",
                         f"{1e3 / steady_step_rate(data):.1f}",
                         f"{len({int(value) for value in data['segment'] if np.isfinite(value)})}",
                         run["stop"]["reason"] if run["stop"] else "没有"])
            if run["profiles"] and any(WINDOW_3D[0] - 1e-9 <= entry["time"] for entry in run["profiles"]):
                means[name] = window_mean_profiles(run["profiles"], WINDOW_3D)
        if rows:
            write_table("long_cavity3d_runs.md", markdown_table(
                ["模式", "到达的 t", "粒子数（始 → 末）", "越出盒（最大）", "最大速度（最大）",
                 f"动能（窗口 {WINDOW_3D[0]:g}–{WINDOW_3D[1]:g} 平均 ± 标准差）", "窗口内动能斜率（每时间单位）",
                 "平均密度 − ρ0（末）", "θ（末）", "步 / 秒（t ≥ 10，含采样）", "每步毫秒", "段数", "停止条件"], rows))
        if means:
            rows = []
            for name, profile in means.items():
                u, v = profile["u_along_y"], profile["v_along_x"]
                rows.append([runs[name]["text"], profile["count"],
                             f"{np.nanmin(u['velocity']):+.4f}（y = {u['coordinate'][np.nanargmin(u['velocity'])]:.3f}）",
                             f"{np.nanmax(v['velocity']):+.4f}（x = {v['coordinate'][np.nanargmax(v['velocity'])]:.3f}）",
                             f"{np.nanmin(v['velocity']):+.4f}（x = {v['coordinate'][np.nanargmin(v['velocity'])]:.3f}）"])
            names = list(means)
            for first in range(len(names)):
                for second in range(first + 1, len(names)):
                    a, b = means[names[first]], means[names[second]]
                    du = a["u_along_y"]["velocity"] - b["u_along_y"]["velocity"]
                    dv = a["v_along_x"]["velocity"] - b["v_along_x"]["velocity"]
                    rows.append([f"{runs[names[first]]['text']} − {runs[names[second]]['text']}", "",
                                 f"rms {np.sqrt(np.nanmean(du ** 2)):.4f}，最大 {np.nanmax(np.abs(du)):.4f}（u 线）",
                                 f"rms {np.sqrt(np.nanmean(dv ** 2)):.4f}，最大 {np.nanmax(np.abs(dv)):.4f}（v 线）", ""])
            write_table("long_cavity3d_profiles.md", markdown_table(
                ["模式 / 差", "窗口内剖面个数", "u 沿 y（过中心的竖线）的最小值", "v 沿 x（过中心的横线）的最大值",
                 "v 沿 x 的最小值"], rows))
            figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
            styles = {"cavity3d_1m_crossing_long": (INK_PRIMARY, 1.4, "solid"),
                      "cavity3d_1m_simple_long": (INK_SECONDARY, 2.2, "solid"),
                      "cavity3d_1m_adami_long": (INK_SECONDARY, 1.2, (0, (4, 2)))}
            for name, profile in means.items():
                colour, width, style = styles[name]
                axes[0].plot(profile["u_along_y"]["velocity"], profile["u_along_y"]["coordinate"], color=colour,
                             linewidth=width, linestyle=style, label=runs[name]["text"])
                axes[1].plot(profile["v_along_x"]["coordinate"], profile["v_along_x"]["velocity"], color=colour,
                             linewidth=width, linestyle=style)
            axes[0].set_xlabel("u（x 方向速度）")
            axes[0].set_ylabel("y（盒内相对坐标）")
            axes[0].set_title("过盒心的竖线（沿 y）")
            axes[1].set_xlabel("x（盒内相对坐标）")
            axes[1].set_ylabel("v（y 方向速度）")
            axes[1].set_title("过盒心的横线（沿 x）")
            handles, labels = axes[0].get_legend_handles_labels()
            height = figure.get_figheight()
            figure.suptitle(f"3-D 方腔 1M，Re = 1000：t = {WINDOW_3D[0]:g}–{WINDOW_3D[1]:g} 的平均中心线剖面", x=0.01,
                            y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                            color=INK_PRIMARY)
            figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                          handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
            figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
            figure.savefig(REPORT_ROOT / "figures" / "long_cavity3d_profiles.png")
            plt.close(figure)
        if runs:
            figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
            styles = {"cavity3d_1m_crossing_long": (INK_PRIMARY, 1.2, "solid"),
                      "cavity3d_1m_simple_long": (INK_SECONDARY, 2.2, "solid"),
                      "cavity3d_1m_adami_long": (INK_SECONDARY, 1.2, (0, (4, 2)))}
            for name, run in runs.items():
                colour, width, style = styles[name]
                axes[0].plot(run["series"]["time"], run["series"]["kinetic_energy"], color=colour, linewidth=width,
                             linestyle=style, label=run["text"])
                axes[1].plot(run["series"]["time"], run["series"]["density_mean"] - REST_DENSITY, color=colour,
                             linewidth=width, linestyle=style)
            axes[0].set_title("流体动能")
            axes[1].set_title("流体平均密度 − ρ0（kg/m³）")
            for axis in axes:
                axis.set_xlabel("时间（模拟单位）")
                axis.axvspan(*WINDOW_3D, color=INK_SECONDARY, alpha=0.08, linewidth=0)
            handles, labels = axes[0].get_legend_handles_labels()
            height = figure.get_figheight()
            figure.suptitle("3-D 方腔 1M，Re = 1000：动能和平均密度（灰底 = 统计窗口）", x=0.01, y=1.0 - 0.05 / height,
                            ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
            figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                          handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
            figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
            figure.savefig(REPORT_ROOT / "figures" / "long_cavity3d_history.png")
            plt.close(figure)

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary_long.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                               encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("short", "long"))
    arguments = parser.parse_args()
    if arguments.action == "short":
        return short()
    return long_tables()


if __name__ == "__main__":
    sys.exit(main())
