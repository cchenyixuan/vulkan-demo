"""
continuation_check.py - the 250^2 cavity of the complete scheme + volume constraint continued from t = 100 to
t = 200 (run n250_reflection_full_volume_to200, resumed from the original run's last checkpoint at t = 90.72 in a
copy of its directory; the original n250_reflection_full_volume is untouched).

    .venv/Scripts/python.exe -m experiment.wall_crossing.continuation_check grids
    .venv/Scripts/python.exe -m experiment.wall_crossing.continuation_check tables

Three windows, each analysed with the post-processing of the cavity tables (section 16 of the report describes
it; the functions are those of stream_function_check.py and cavity_sampling.py):
    80-100    the original run: its analysis window, its snapshots (one per 0.99) and samples (one per 0.21)
    130-150   the continued run: the extra snapshots of the window (one per 1.05; cavity_runner_entry.py item 5)
              and its samples with 130 < t <= 150
    180-200   the continued run: its analysis window, the runner's snapshots and samples
psi_min is normalised with the wall at the box face (L = 1 + dx: the "mid frame" of the tables; the wall frame,
L = 1 + 2 dx, is given beside it), the vortex centre likewise; velocity extrema from the time-mean centre lines.

Beside them: the kinetic energy from t = 80 to 200 (the runner's samples), the mean density (field_history.jsonl,
both runs), the volume constraint's log (volume_constraint.jsonl: the continued run's lines follow the original's
in the copied file; their "step" restarts at 0 with the process, so their time is the logged time + 90.72), and
the overlap 90.72 to 100 that both runs computed.
Output: claude/wall_crossing/tables/continuation_*.md, figures/continuation_history.png,
claude/wall_crossing/continuation/summary.json; one runs.jsonl record per call.
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

from experiment.validation import cavity_analysis, cavity_reference, cavity_sampling as sampling    # noqa: E402
from experiment.wall_crossing import stream_function_check as check                                 # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                    # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
ORIGINAL, CONTINUED = "n250_reflection_full_volume", "n250_reflection_full_volume_to200"
RESTART_TIME = 90.72
WINDOWS = (("80–100", ORIGINAL, None, "snapshots"),
           ("130–150", CONTINUED, (130.0, 150.0), "snapshots_extra"),
           ("180–200", CONTINUED, None, "snapshots"))
KINETIC_INSTANTS = (80.01, 90.09, 90.72, 99.96, 110.04, 120.12, 130.2, 140.07, 150.15, 160.02, 170.1, 180.18,
                    190.05, 199.92)
DYNAMIC_PRESSURE = 1000.0
REST_DENSITY = 1000.0


def window_of(name: str, window: tuple | None) -> tuple:
    return check.analysis_window(name) if window is None else window


def window_files(name: str, window: tuple, directory_name: str) -> list:
    run = cavity_analysis.load_run(name, LOG_ROOT / "cavity")
    timestep = float(run["meta"]["dt"])
    files = sorted((run["dir"] / directory_name).glob("t*.npz"))
    return [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * timestep <= window[1] + 1e-9]


def cache_path(label: str) -> pathlib.Path:
    return check.CACHE / f"continuation_{label.replace('–', '_')}.npz"


def make_grids(labels: list | None = None) -> int:
    check.CACHE.mkdir(parents=True, exist_ok=True)
    with LoggedRun(kind="analysis", case="none") as record:
        record.extra["label"] = "continuation_check grids" + (f" {','.join(labels)}" if labels else "")
        for label, name, window, directory_name in WINDOWS:
            if labels and label not in labels:
                continue
            window = window_of(name, window)
            files = window_files(name, window, directory_name)
            run = cavity_analysis.load_run(name, LOG_ROOT / "cavity")
            grids = check.grids_of_files(run, files)
            grids["window"] = np.array(window)
            grids["files"] = np.array([path.name for path in files])
            np.savez(cache_path(label), **grids)
            print(f"[continuation_check] {label}: {name}, {len(files)} snapshot files of {directory_name}, "
                  f"{int(grids['fallbacks'])} MLS fallbacks", flush=True)
    return 0


def raw_rows(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def samples(name: str) -> list:
    by_step = {}
    for row in raw_rows(LOG_ROOT / "cavity" / name / "samples.jsonl"):
        by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def centre_line_statistics(name: str, window: tuple) -> dict:
    """The time mean of the two dense centre lines over the samples with window[0] < t <= window[1] (the
    analysis takes the samples of its window the same way), and the extrema as cavity_analysis.analyse_run
    takes them."""
    directory = LOG_ROOT / "cavity" / name
    rows = [row for row in samples(name) if window[0] - 1e-9 <= row["time"] <= window[1] + 1e-9]
    u_lines, v_lines = [], []
    for row in rows:
        with np.load(directory / "samples" / f"s{row['step']:010d}.npz") as archive:
            u_lines.append(archive["u_dense_mls"])
            v_lines.append(archive["v_dense_mls"])
    dense = np.load(directory / "points.npz")["dense_reference"]
    means = {"u": np.mean(u_lines, axis=0), "v": np.mean(v_lines, axis=0)}
    extrema = {}
    for quantity, (line, kind, search) in cavity_analysis.EXTREMA_SEARCH.items():
        position, value = sampling.refine_extremum(dense, means[line], kind, search=search)
        extrema[quantity] = {"value": value, "position_wall": position}
    energies = np.array([row["kinetic_energy"] for row in rows])
    return {"samples": len(rows), "times": [rows[0]["time"], rows[-1]["time"]], "extrema": extrema,
            "kinetic_energy_mean": float(energies.mean()), "kinetic_energy_std": float(energies.std()),
            "kinetic_energy_slope": float(np.polyfit([row["time"] for row in rows], energies, 1)[0])}


def tables() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style

    reference = cavity_reference.marchi2021()["extrema"]
    reference_psi = float(reference["psi_min"][0])
    reference_centre = (float(reference["x_at_psi_min"][0]), float(reference["y_at_psi_min"][0]))
    analysed = json.loads((REPORT_ROOT / "cavity" / "tables.json").read_text(encoding="utf-8"))["runs"]
    with LoggedRun(kind="analysis", case="none") as record:
        record.extra["label"] = "continuation_check tables"
        summary = {"run_id": record.run_id, "windows": {}}

        # ---------------------------------------------------------------- the three windows
        rows, extrema_rows, check_rows = [], [], []
        for label, name, window, directory_name in WINDOWS:
            data = np.load(cache_path(label))
            axis, spacing = data["axis"], float(data["spacing"])
            lengths = check.frames(spacing)
            window = tuple(float(value) for value in data["window"])
            fields = check.stream_functions(data["u"], data["v"], axis)
            x, y, psi = check.refined_minimum(fields["A"], axis)
            x_c, y_c, psi_c = check.refined_minimum(fields["C"], axis)
            mid, wall = lengths["mid"], lengths["wall"]
            centre_mid = ((x + 0.5 * mid) / mid, (y + 0.5 * mid) / mid)
            lines = centre_line_statistics(name, window)
            entry = {
                "run": name, "window": window, "snapshots": int(data["snapshots"]), "samples": lines["samples"],
                "psi_physical": psi, "psi_mid": psi / mid, "psi_wall": psi / wall,
                "psi_mid_deviation_percent": 100.0 * (abs(psi / mid) / abs(reference_psi) - 1.0),
                "psi_wall_deviation_percent": 100.0 * (abs(psi / wall) / abs(reference_psi) - 1.0),
                "psi_mid_path_c_deviation_percent": 100.0 * (abs(psi_c / mid) / abs(reference_psi) - 1.0),
                "centre_mid": list(centre_mid),
                "centre_mid_deviation": [centre_mid[0] - reference_centre[0], centre_mid[1] - reference_centre[1]],
                "kinetic_energy_mean": lines["kinetic_energy_mean"], "kinetic_energy_std": lines["kinetic_energy_std"],
                "kinetic_energy_slope": lines["kinetic_energy_slope"], "extrema": {}}
            for quantity, values in lines["extrema"].items():
                physical = values["position_wall"] * wall - (0.5 + spacing)
                position_mid = (physical + 0.5 * mid) / mid
                reference_position = float(reference[{"u_min": "y_at_u_min", "v_max": "x_at_v_max",
                                                      "v_min": "x_at_v_min"}[quantity]][0])
                entry["extrema"][quantity] = {
                    "value": values["value"],
                    "deviation_percent": 100.0 * (abs(values["value"]) / abs(float(reference[quantity][0])) - 1.0),
                    "position_mid": position_mid, "position_deviation": position_mid - reference_position}
            summary["windows"][label] = entry
            rows.append([label, f"{name}，{directory_name}", f"{window[0]:.2f}–{window[1]:.2f}",
                         f"{entry['snapshots']} / {entry['samples']}",
                         f"{entry['psi_mid']:.5f}（{entry['psi_mid_deviation_percent']:+.2f} %）",
                         f"{entry['psi_wall_deviation_percent']:+.2f} %",
                         f"({centre_mid[0]:.4f}, {centre_mid[1]:.4f})",
                         f"({entry['centre_mid_deviation'][0]:+.4f}, {entry['centre_mid_deviation'][1]:+.4f})",
                         f"{entry['kinetic_energy_mean']:.4f} ± {entry['kinetic_energy_std']:.4f}"])
            extrema_rows.append([label] + [
                f"{entry['extrema'][quantity]['value']:+.5f}（{entry['extrema'][quantity]['deviation_percent']:+.2f} %）"
                f"，{entry['extrema'][quantity]['position_mid']:.4f}"
                for quantity in ("u_min", "v_max", "v_min")])
            # the standard tables, where they exist: the same window must give the same numbers
            tabled = analysed.get(name)
            if tabled and window is not None and list(tabled["window"]) == list(window):
                tabled_psi = tabled["extrema"]["psi_min"]
                check_rows.append([label, name, f"{abs(entry['psi_mid'] - tabled_psi['value_mid']):.1e}",
                                   f"{max(abs(entry['extrema'][quantity]['value'] - tabled['extrema'][quantity]['value']) for quantity in ('u_min', 'v_max', 'v_min')):.1e}"])
        check.write_table("continuation_windows.md",
                          f"250² 完整方案 + 体积约束，三个窗口。ψ_min 按壁面在盒面归一化（L = 1 + dx，即表里的 mid 帧），括号内"
                          f"是相对 Marchi 2021（{reference_psi:.5f}）的偏差；wall 帧（L = 1 + 2 dx）的偏差在旁边。涡心也在 mid 帧，"
                          f"Marchi 2021 的位置是 ({reference_centre[0]:.4f}, {reference_centre[1]:.4f})。动能是窗口内各样本的平均 ± "
                          "标准差。\n\n"
                          + check.markdown_table(["窗口", "运行，快照目录", "实际窗口", "快照数 / 样本数",
                                                  "ψ_min（mid 帧）", "ψ_min 偏差（wall 帧）", "涡心（mid 帧）", "涡心偏差",
                                                  "动能"], rows))
        check.write_table("continuation_extrema.md",
                          "同样三个窗口，两条中心线上的速度极值：值（相对 Marchi 2021 的偏差），位置（mid 帧）。Marchi 2021："
                          f"u_min {float(reference['u_min'][0]):.5f} 在 y = {float(reference['y_at_u_min'][0]):.4f}，"
                          f"v_max {float(reference['v_max'][0]):.5f} 在 x = {float(reference['x_at_v_max'][0]):.4f}，"
                          f"v_min {float(reference['v_min'][0]):.5f} 在 x = {float(reference['x_at_v_min'][0]):.4f}。\n\n"
                          + check.markdown_table(["窗口", "u_min", "v_max", "v_min"], extrema_rows))
        first = summary["windows"]["80–100"]
        difference_rows = []
        for label in ("130–150", "180–200"):
            other = summary["windows"][label]
            difference_rows.append([
                f"{label} − 80–100",
                f"{other['psi_mid_deviation_percent'] - first['psi_mid_deviation_percent']:+.3f}",
                *[f"{other['extrema'][quantity]['deviation_percent'] - first['extrema'][quantity]['deviation_percent']:+.3f}"
                  for quantity in ("u_min", "v_max", "v_min")],
                f"({other['centre_mid'][0] - first['centre_mid'][0]:+.5f}, {other['centre_mid'][1] - first['centre_mid'][1]:+.5f})",
                f"{100 * (other['kinetic_energy_mean'] / first['kinetic_energy_mean'] - 1):+.3f} %"])
        check.write_table("continuation_differences.md",
                          "后两个窗口减第一个窗口：ψ_min 和三个速度极值的偏差之差（百分点），涡心位置之差（mid 帧），平均动能"
                          "的相对变化。\n\n"
                          + check.markdown_table(["", "ψ_min（mid 帧）", "u_min", "v_max", "v_min", "涡心", "动能"],
                                                 difference_rows))
        if check_rows:
            check.write_table("continuation_consistency.md",
                              "本工具与方腔分析表在同一个窗口上的差（应为 0 或舍入量级）。\n\n"
                              + check.markdown_table(["窗口", "运行", "ψ_min（mid 帧）之差", "速度极值之差的最大值"],
                                                     check_rows))

        # ---------------------------------------------------------------- kinetic energy from t = 80 to 200
        original_samples, continued_samples = samples(ORIGINAL), samples(CONTINUED)
        original_by_step = {row["step"]: row for row in original_samples}
        overlap = [(row, original_by_step[row["step"]]) for row in continued_samples
                   if row["time"] > RESTART_TIME + 1e-9 and row["step"] in original_by_step]
        energy_rows = []
        # the continued run's own samples start after the restart (its file also holds the copied rows of the
        # original up to the checkpoint)
        continued_own = [row for row in continued_samples if row["time"] > RESTART_TIME + 1e-9]
        for instant in KINETIC_INSTANTS:
            cells = [f"{instant:.2f}" + ("（检查点）" if abs(instant - RESTART_TIME) < 1e-6 else "")]
            for source in (original_samples, continued_own):
                times = np.array([row["time"] for row in source])
                index = int(np.argmin(np.abs(times - instant)))
                cells.append("—" if abs(times[index] - instant) > 0.11 else f"{source[index]['kinetic_energy']:.4f}")
            energy_rows.append(cells)
        segments = []
        for first_time, last_time in ((100.0, 120.0), (120.0, 140.0), (140.0, 160.0), (160.0, 180.0),
                                      (180.0, 200.01)):
            chosen = [row for row in continued_samples if first_time < row["time"] <= last_time]
            energies = np.array([row["kinetic_energy"] for row in chosen])
            segments.append({"window": [first_time, last_time], "mean": float(energies.mean()),
                             "std": float(energies.std()),
                             "slope": float(np.polyfit([row["time"] for row in chosen], energies, 1)[0])})
        late = [row for row in continued_samples if row["time"] > 100.0]
        overall_slope = float(np.polyfit([row["time"] for row in late], [row["kinetic_energy"] for row in late], 1)[0])
        overlap_difference = np.array([continued["kinetic_energy"] - original["kinetic_energy"]
                                       for continued, original in overlap])
        summary["kinetic_energy"] = {
            "segments": segments, "slope_100_200": overall_slope,
            "overlap_samples": len(overlap),
            "overlap_difference_mean": float(overlap_difference.mean()) if overlap else None,
            "overlap_difference_maximum": float(np.abs(overlap_difference).max()) if overlap else None,
            "first_after_restart": [overlap[0][0]["time"], overlap[0][0]["kinetic_energy"],
                                    overlap[0][1]["kinetic_energy"]] if overlap else None}
        check.write_table("continuation_kinetic_energy.md",
                          "流体动能 ½ Σ m |v|²（运行器在每个样本记录的值）。原运行到 t = 99.96；续跑从 t = 90.72 的检查点开始"
                          "（它的第一个样本在 t = 90.93），90.72 到 99.96 两次都算了。\n\n"
                          + check.markdown_table(["t", "原运行", "续跑"], energy_rows)
                          + "\n每 20 个时间单位一段（续跑）：平均 ± 标准差，线性拟合斜率（每时间单位）。\n\n"
                          + check.markdown_table(["时间段", "平均", "标准差", "斜率"],
                                                 [[f"{segment['window'][0]:g}–{min(segment['window'][1], 200):g}",
                                                   f"{segment['mean']:.4f}", f"{segment['std']:.4f}",
                                                   f"{segment['slope']:+.5f}"] for segment in segments]))

        # ---------------------------------------------------------------- the constraint's target, theta, density
        log_rows = raw_rows(LOG_ROOT / "cavity" / CONTINUED / "volume_constraint.jsonl")
        restart_index = next(index for index in range(1, len(log_rows)) if log_rows[index]["step"] < log_rows[index - 1]["step"])
        original_log, continued_log = log_rows[:restart_index], log_rows[restart_index:]
        for row in continued_log:
            row["absolute_time"] = row["time"] + RESTART_TIME
        original_targets = sorted({row["target_volume"] for row in original_log})
        continued_targets = sorted({row["target_volume"] for row in continued_log})
        at_restart = next(row for row in original_log if abs(row["time"] - RESTART_TIME) < 1e-6)
        first_continued = continued_log[0]
        overlap_theta = [(row, next((other for other in original_log if abs(other["time"] - row["absolute_time"]) < 1e-6), None))
                         for row in continued_log if row["absolute_time"] <= 99.96 + 1e-6]
        overlap_theta = [(row, other) for row, other in overlap_theta if other is not None]
        theta_windows = {}
        for label, first_time, last_time in (("80–100", 80.0, 100.0), ("130–150", 130.0, 150.0),
                                             ("180–200", 180.0, 200.01)):
            source = original_log if label == "80–100" else continued_log
            key = "time" if label == "80–100" else "absolute_time"
            values = np.array([row["theta"] for row in source if first_time < row[key] <= last_time])
            theta_windows[label] = {"mean": float(values.mean()), "std": float(values.std()), "updates": int(values.size)}
        summary["constraint"] = {
            "original_targets": original_targets, "continued_targets": continued_targets,
            "targets_identical": original_targets == continued_targets and len(original_targets) == 1,
            "original_at_restart": {"theta": at_restart["theta"], "volume": at_restart["volume"]},
            "continued_first": {"theta": first_continued["theta"], "volume": first_continued["volume"]},
            "overlap_updates": len(overlap_theta),
            "overlap_theta_difference_maximum": float(max(abs(row["theta"] - other["theta"]) for row, other in overlap_theta)),
            "overlap_theta_identical_updates": int(sum(row["theta"] == other["theta"] for row, other in overlap_theta)),
            "continued_updates": len(continued_log), "theta_windows": theta_windows}

        def density_rows(name: str) -> list:
            return [row for row in raw_rows(LOG_ROOT / "cavity" / name / "field_history.jsonl")
                    if row.get("step") is not None]

        original_density = density_rows(ORIGINAL)
        continued_density = [row for row in density_rows(CONTINUED) if row.get("segment") == 1]
        original_density_by_step = {row["step"]: row for row in original_density}
        density_overlap = [(row, original_density_by_step[row["step"]]) for row in continued_density
                           if row["step"] in original_density_by_step]
        density_windows = {}
        for label, first_time, last_time in (("80–100", 80.0, 100.0), ("130–150", 130.0, 150.0),
                                             ("180–200", 180.0, 200.01)):
            source = original_density if label == "80–100" else continued_density
            chosen = [row for row in source if first_time < row["time"] <= last_time]
            values = np.array([row["density_mean"] - REST_DENSITY for row in chosen])
            pressures = np.array([row["pressure_mean"] / DYNAMIC_PRESSURE for row in chosen])
            negative = np.array([row["pressure_negative_fraction"] for row in chosen])
            density_windows[label] = {"mean": float(values.mean()), "std": float(values.std()),
                                      "pressure_mean": float(pressures.mean()),
                                      "negative_mean": float(negative.mean())}
        late = [row for row in continued_density if row["time"] > 100.0]
        density_slope = float(np.polyfit([row["time"] for row in late],
                                         [row["density_mean"] for row in late], 1)[0])
        overlap_density_difference = np.array([row["density_mean"] - other["density_mean"]
                                               for row, other in density_overlap])
        overlap_original = np.array([other["density_mean"] - REST_DENSITY for _, other in density_overlap])
        overlap_continued = np.array([row["density_mean"] - REST_DENSITY for row, _ in density_overlap])
        summary["density"] = {
            "windows": density_windows, "slope_100_200": density_slope,
            "overlap_rows": len(density_overlap),
            "overlap_original_mean": float(overlap_original.mean()),
            "overlap_continued_mean": float(overlap_continued.mean()),
            "overlap_difference_rms": float(np.sqrt((overlap_density_difference ** 2).mean())),
            "overlap_difference_maximum": float(np.abs(overlap_density_difference).max()),
            "original_at_restart": float(original_density_by_step[int(round(RESTART_TIME / 3e-5))]["density_mean"]
                                         - REST_DENSITY),
            "continued_first": [continued_density[0]["time"], continued_density[0]["density_mean"] - REST_DENSITY]}
        constraint_rows = [
            ["约束的目标值 V_target", f"{original_targets[0]:.16g}" if len(original_targets) == 1 else str(original_targets),
             f"{continued_targets[0]:.16g}" if len(continued_targets) == 1 else str(continued_targets),
             "相同" if summary["constraint"]["targets_identical"] else "**不同**"],
            ["t = 90.72 的那次更新：V_now", f"{at_restart['volume']:.16g}", f"{first_continued['volume']:.16g}",
             f"差 {first_continued['volume'] - at_restart['volume']:.1e}"],
            ["t = 90.72 的那次更新：θ", f"{at_restart['theta']:.6e}", f"{first_continued['theta']:.6e}",
             f"差 {first_continued['theta'] - at_restart['theta']:.1e}"],
            ["重叠段 90.72–99.96 的更新：θ 之差的最大值", "", "",
             f"{summary['constraint']['overlap_theta_difference_maximum']:.2e}（{len(overlap_theta)} 次更新，其中 "
             f"{summary['constraint']['overlap_theta_identical_updates']} 次逐位相同）"],
            ["重叠段的平均密度 − ρ0（平均）", f"{summary['density']['overlap_original_mean']:+.5f}",
             f"{summary['density']['overlap_continued_mean']:+.5f}",
             f"之差 rms {summary['density']['overlap_difference_rms']:.5f}，最大 {summary['density']['overlap_difference_maximum']:.5f}"],
        ]
        check.write_table("continuation_constraint.md",
                          "续跑前后的约束和平均密度。原运行和续跑的约束日志都在续跑目录的 volume_constraint.jsonl 里（续跑的行"
                          "接在后面，时间从 0 重新数，这里加上了 90.72）。\n\n"
                          + check.markdown_table(["", "原运行", "续跑", "比较"], constraint_rows))
        window_rows = []
        for label in ("80–100", "130–150", "180–200"):
            density, theta = density_windows[label], theta_windows[label]
            window_rows.append([label, f"{density['mean']:+.5f} ± {density['std']:.5f}",
                                f"{theta['mean']:+.2e} ± {theta['std']:.1e}（{theta['updates']} 次）",
                                f"{density['pressure_mean']:+.4f}", f"{100 * density['negative_mean']:.1f} %"])
        check.write_table("continuation_density_windows.md",
                          "三个窗口的平均密度 − ρ0（kg/m³，平均 ± 标准差）、约束的 θ、平均压力（ρ0U²）和负压粒子比例。"
                          f"续跑 t ∈ (100, 200] 的平均密度线性拟合斜率：{density_slope:+.6f} kg/m³ 每时间单位。\n\n"
                          + check.markdown_table(["窗口", "平均密度 − ρ0", "θ", "平均压力", "负压粒子比例"], window_rows))

        # ---------------------------------------------------------------- figure
        setup_style()
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        panels = (("kinetic_energy", "流体动能"), ("density", "流体平均密度 − ρ0（kg/m³）"), ("theta", "θ（×10⁻⁶）"))
        handles, labels = [], []
        for axis, (key, title) in zip(axes, panels):
            for source_label, colour, style in (("原运行（到 t = 100；宽的浅灰线）", INK_SECONDARY, "solid"),
                                                ("续跑（从 t = 90.72 的检查点；细的黑线）", INK_PRIMARY, "solid")):
                original = source_label.startswith("原")
                if key == "kinetic_energy":
                    source = original_samples if original else continued_samples
                    chosen = [row for row in source if row["time"] >= 80.0 - 1e-9 and (original or row["time"] > RESTART_TIME)]
                    times, values = [row["time"] for row in chosen], [row["kinetic_energy"] for row in chosen]
                elif key == "density":
                    source = original_density if original else continued_density
                    chosen = [row for row in source if row["time"] >= 80.0 - 1e-9]
                    times, values = [row["time"] for row in chosen], [row["density_mean"] - REST_DENSITY for row in chosen]
                else:
                    source = original_log if original else continued_log
                    time_key = "time" if original else "absolute_time"
                    chosen = [row for row in source if row[time_key] >= 80.0 - 1e-9]
                    times, values = [row[time_key] for row in chosen], [1.0e6 * row["theta"] for row in chosen]
                line, = axis.plot(times, values, color=colour, linestyle=style, linewidth=2.6 if original else 0.9,
                                  alpha=0.45 if original else 1.0, zorder=2 if original else 3)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(source_label)
            for first_time, last_time in ((80.0, 100.0), (130.0, 150.0), (180.0, 200.0)):
                axis.axvspan(first_time, last_time, color=INK_SECONDARY, alpha=0.08, zorder=1, linewidth=0)
            if key in ("density", "theta"):
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=1)
            axis.set_xlim(80, 200)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔 250²，完整方案 + 体积约束：t = 80 到 200（灰底 = 三个统计窗口）", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
        figure.savefig(REPORT_ROOT / "figures" / "continuation_history.png")
        plt.close(figure)

        out = REPORT_ROOT / "continuation"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                          encoding="utf-8", newline="\n")
        print(f"[continuation_check] tables written ({record.run_id})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("grids", "tables"))
    parser.add_argument("--window", action="append", default=None,
                        help="grids: only these windows (80–100, 130–150, 180–200; repeat)")
    arguments = parser.parse_args()
    return make_grids(arguments.window) if arguments.action == "grids" else tables()


if __name__ == "__main__":
    sys.exit(main())
