"""e36_compare.py - E36 comparison of the wall boundary conditions at one resolution (CPU only).

Runs (cavity_runner run directories; the same case, the same sampling): the v6 main series (K = 2), v6 K = 1 and
v7 K = 1 with the Adami wall (WALL_BC = 1). Every run is cut at --t-stop (100) and averaged over the last
--average-span (20) time units before it, exactly as cavity_analysis.analyse_run does for the main series
(window (t_last - 20, t_last], t_last = the last sample <= t_stop), with the existing tools:
  * cavity_analysis.analyse_run: errors at the 28 + 28 Marchi 2021 points (wall-row frame, MLS; L2 = rms,
    relative L2 = rms / rms(reference)), centre-line extrema, time-averaged dense centre lines;
  * the stream function of cavity_analysis.stream_function_minimum (time-averaged MLS grid of u over the window's
    snapshots, psi integrated upward from the bottom wall row, quadratic refinement), here kept as a field so the
    two bottom secondary vortices (BR1: x in (0.6, 1), y in (0, 0.45); BL1: x in (0, 0.35), y in (0, 0.35), the
    windows of cavity_fields.VORTEX_SEARCH) are refined the same way as the primary minimum;
  * effective_viscosity.evaluate: nu_eff / nu of the discrete viscous operator (the run's xi and eps^2) on the run's
    last snapshot, interior particles (>= 1.5 h from the wall rows), median of the two quadratic fields;
plus from the run files: the mass-weighted mean fluid velocity of the stored velocities (samples.jsonl mean_u,
mean_v) averaged over the window; u of the dense vertical centre line within 3 h of the lid row and of the bottom
row; the wall pressure near the two top corners from the checkpoints in the window (relative to the mean fluid
pressure, in units of rho U^2 / 2); the fluid density (checkpoints and the samples' min / max); fps (samples, and
the same-GPU timing runs if given).
Outputs: <out>/e36_summary.json, <out>/e36_tables.md, <out>/fig_centrelines.{png,pdf},
<out>/fig_near_wall.{png,pdf}, <out>/fig_density.{png,pdf}.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.e36_compare --out docs/wall_bc/e36 \\
        --run "v6 K=2 (main series)=C:/.../logs/validation/cavity_re1000/n250_k2_float32_xi0p001_eps0p0025" \\
        --run "v6 K=1=C:/.../logs/e36/runs/n250_k1_v6" --run "v7 K=1 Adami=C:/.../logs/e36/runs/n250_k1_v7_adami"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_analysis, cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation import effective_viscosity  # noqa: E402

GRID_POINTS = cavity_analysis.GRID_POINTS
SECONDARY_WINDOWS = {"BR1": ((0.6, 1.0), (0.0, 0.45)), "BL1": ((0.0, 0.35), (0.0, 0.35))}
# Ghia, Ghia & Shin (1982) Table V, Re = 1000 (129^2 grid), quoted for orientation only (not transcribed /
# checked in this repository; Marchi 2021 Table 19 lists the primary vortex only)
GHIA_VORTICES = {"primary": (-0.117929, 0.5313, 0.5625), "BR1": (1.75102e-3, 0.8594, 0.1094),
                 "BL1": (2.31129e-4, 0.0859, 0.0781)}
NEAR_WALL_DISTANCES_IN_H = (0.25, 0.5, 1.0, 2.0, 3.0)
DYNAMIC_PRESSURE = 0.5 * 1000.0 * 1.0 ** 2       # rho0 U^2 / 2


def truncate(run: dict, t_stop: float) -> dict:
    keep = run["times"] <= t_stop + 1e-9
    run = dict(run)
    run["rows"] = [row for row, flag in zip(run["rows"], keep) if flag]
    run["times"] = run["times"][keep]
    run["kinetic_energy"] = run["kinetic_energy"][keep]
    run["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    return run


def stream_function_field(run: dict, window: tuple[float, float]) -> dict:
    """cavity_analysis.stream_function_minimum's grid and psi (identical computation), with the primary minimum and
    the two bottom secondary maxima refined; positions and psi in the wall-row frame."""
    spacing = run["spacing"]
    dt = float(run["meta"]["dt"])
    support = float(run["meta"]["support_radius"])
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    selected = [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * dt <= window[1] + 1e-9]
    half = 0.5 + spacing
    axis = np.linspace(-half, half, GRID_POINTS)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    total = np.zeros(points.shape[0])
    for path in selected:
        with np.load(path) as archive:
            particles = sampling.ParticleSet(archive["positions"].astype(np.float64),
                                             archive["velocities"].astype(np.float64),
                                             archive["volumes"].astype(np.float64))
        tree = sampling.cKDTree(particles.positions)
        for start in range(0, points.shape[0], 40000):
            chunk = slice(start, start + 40000)
            total[chunk] += sampling.interpolate(particles, points[chunk], support, tree=tree)["mls"][:, 0]
    u_grid = (total / len(selected)).reshape(GRID_POINTS, GRID_POINTS)
    psi = sampling.stream_function(u_grid, axis + half)
    length = 1.0 + 2.0 * spacing
    reference = (axis + half) / length                       # wall-row frame coordinate of the grid lines
    vortices = {}
    interior = (np.abs(xx) < 0.3) & (np.abs(yy) < 0.3)       # stream_function_minimum's primary search box
    x_min, y_min, psi_min = sampling.refine_grid_minimum(axis, axis, np.where(interior, psi, np.nan))
    vortices["primary"] = {"psi": psi_min / length, "x": (x_min + half) / length, "y": (y_min + half) / length}
    for name, ((x0, x1), (y0, y1)) in SECONDARY_WINDOWS.items():
        box = ((reference[None, :] > x0) & (reference[None, :] < x1)
               & (reference[:, None] > y0) & (reference[:, None] < y1))
        x_max, y_max, minus_psi = sampling.refine_grid_minimum(axis, axis, np.where(box, -psi, np.nan))
        vortices[name] = {"psi": -minus_psi / length, "x": (x_max + half) / length, "y": (y_max + half) / length}
    # psi on the lid row (the top grid line) should be 0 (no net volume flux through a vertical line); a net mean
    # u of the stored velocities shows up here and biases psi_min by about the same order
    top = psi[-1] / length
    return {"psi_top_mean": float(top.mean()), "psi_top_max_abs": float(np.abs(top).max()),
            "psi_at_lid_above_primary": float(np.interp(vortices["primary"]["x"], reference, top)),
            "snapshots": len(selected), "first": int(selected[0].stem[1:]) * dt if selected else None,
            "last": int(selected[-1].stem[1:]) * dt if selected else None, "vortices": vortices}


def near_wall_profiles(analysis: dict, spacing: float, support: float) -> dict:
    """u(0.5, y) of the time-averaged dense line at distances d below the lid row and above the bottom row
    (wall-row frame: lid row y = 1, bottom row y = 0, physical d / L)."""
    dense = analysis["dense_reference"]
    u = analysis["statistics"]["u_dense_mls"]["mean"]
    length = 1.0 + 2.0 * spacing
    result = {"lid": {}, "bottom": {}}
    for factor in NEAR_WALL_DISTANCES_IN_H:
        distance = factor * support / length
        result["lid"][factor] = float(np.interp(1.0 - distance, dense, u))
        result["bottom"][factor] = float(np.interp(distance, dense, u))
    window = 3.0 * support / length
    result["lid_curve"] = {"y": dense[dense >= 1.0 - window].tolist(), "u": u[dense >= 1.0 - window].tolist()}
    result["bottom_curve"] = {"y": dense[dense <= window].tolist(), "u": u[dense <= window].tolist()}
    return result


def checkpoint_states(run: dict, window: tuple[float, float]) -> list[tuple[float, dict]]:
    dt = float(run["meta"]["dt"])
    states = []
    for path in sorted((run["dir"] / "checkpoints").glob("c*.npz")):
        time_value = int(path.stem[1:]) * dt
        if window[0] - 1e-9 <= time_value <= window[1] + 1e-9:
            with np.load(path) as archive:
                states.append((time_value, {name: archive[name] for name in archive.files}))
    return states


def corner_pressure(run: dict, window: tuple[float, float], support: float) -> dict:
    """Wall pressure near the two top corners (checkpoints in the window): wall / lid particles within 2 h of the
    corner of the wall-row frame (|x| = y = 0.5 + dx) that have a fluid particle within h, minus the mean fluid
    pressure, / (rho U^2 / 2); the fluid particles within 2 h of the corner likewise. Also the mean fluid density /
    pressure."""
    spacing = run["spacing"]
    fluid_groups = run["meta"]["fluid_groups"]
    per_state = []
    for time_value, state in checkpoint_states(run, window):
        positions = state["position_voxel_id"][:, :2].astype(np.float64)
        pressure = state["density_pressure"][:, 1].astype(np.float64)
        density = state["density_pressure"][:, 0].astype(np.float64) + float(state["_stored_density_offset"])
        fluid = np.isin(state["material"], fluid_groups)
        mean_pressure = float(pressure[fluid].mean())
        entry = {"time": time_value, "fluid_density_mean": float(density[fluid].mean()),
                 "fluid_pressure_mean": mean_pressure}
        # wall particles that are some fluid particle's neighbour (a fluid particle within h): the only ones whose
        # pressure enters the fluid's force (the deeper layers keep p = 0 under the Adami condition)
        distance_to_fluid, _ = sampling.cKDTree(positions[fluid]).query(positions, k=1)
        interacting = ~fluid & (distance_to_fluid < support)
        for corner, sign in (("top_left", -1.0), ("top_right", 1.0)):
            corner_point = np.array([sign * (0.5 + spacing), 0.5 + spacing])
            near = np.linalg.norm(positions - corner_point, axis=1) < 2.0 * support
            for kind, mask in (("wall", near & interacting), ("fluid", near & fluid)):
                values = (pressure[mask] - mean_pressure) / DYNAMIC_PRESSURE
                entry[f"{corner}_{kind}"] = {"particles": int(mask.sum()), "mean": float(values.mean()),
                                             "min": float(values.min()), "max": float(values.max())}
        per_state.append(entry)
    return {"checkpoints": per_state}


def run_fps(run: dict) -> dict:
    fps = np.array([row["fps"] for row in run["rows"]][5:], dtype=np.float64)
    return {"median": float(np.median(fps)), "mean": float(fps.mean()), "samples": int(fps.size),
            "device_uuids": run["segments"][0].get("device_uuids") if run["segments"] else None,
            "wall_hours": float(sum(segment["wall_s"] for segment in run["segments"]) / 3600.0)}


def viscosity_ratio(run_dir: pathlib.Path) -> dict:
    original = effective_viscosity.LOGS
    effective_viscosity.LOGS = run_dir.parent
    try:
        matrix_rows, operator_rows = effective_viscosity.evaluate(run_dir.name)
    finally:
        effective_viscosity.LOGS = original
    released = [float(row[10]) for row in operator_rows if row[3] == "released"]     # median per field
    return {"nu_eff_over_nu": float(np.mean(released)), "per_field_median": released,
            "snapshot": operator_rows[0][2]}


def analyse(label: str, run_dir: pathlib.Path, t_stop: float, average_span: float) -> dict:
    run = cavity_analysis.load_run(run_dir.name, root=run_dir.parent)
    if run is None:
        raise SystemExit(f"{run_dir}: no samples")
    run = truncate(run, t_stop)
    analysis = cavity_analysis.analyse_run(run, average_span, 1e-3, 10.0, with_stream_function=False)
    analysis["dense_reference"] = run["points"]["dense_reference"]
    window = analysis["window"]
    support = float(run["meta"]["support_radius"])
    rows = [row for row in run["rows"] if window[0] - 1e-9 <= row["time"] <= window[1] + 1e-9]
    stream = stream_function_field(run, window)
    reference = cavity_reference.marchi2021()
    result = {
        "label": label, "run": str(run_dir), "slabs": run["slabs"], "solver": run["meta"].get("solver", "v6"),
        "wall_bc": run["meta"].get("wall_bc"), "git": run["meta"].get("git"), "window": window,
        "samples_in_window": analysis["samples_in_window"], "t_steady_online": analysis["t_steady_online"],
        "stream": stream,
        "errors": {line: {key: analysis["errors"][("wall", "mls", line)][key] for key in ("L2", "L2_relative", "Linf", "Linf_at")}
                   for line in ("u", "v")},
        "errors_ghia": ghia_errors(analysis),
        "extrema": {name: {key: entry[key] for key in ("value", "position_wall")}
                    for name, entry in analysis["extrema"].items()},
        "mean_velocity": {"u": float(np.mean([row["mean_u"] for row in rows])),
                          "v": float(np.mean([row["mean_v"] for row in rows]))},
        "fluid_density_range_window": [float(min(row["density_min"] for row in rows)),
                                       float(max(row["density_max"] for row in rows))],
        "density_track": {"time": [row["time"] for row in run["rows"]],
                          "min": [row["density_min"] for row in run["rows"]],
                          "max": [row["density_max"] for row in run["rows"]]},
        "near_wall": near_wall_profiles(analysis, run["spacing"], support),
        "corners": corner_pressure(run, window, support),
        "viscosity": viscosity_ratio(run_dir),
        "fps": run_fps(run),
        "reference_points_error": {
            "u_near_lid": {f"{y:.7f}": float(error) for y, error in
                           zip(reference["u"][0], analysis["errors"][("wall", "mls", "u")]["pointwise"]) if y > 0.94},
            "u_near_bottom": {f"{y:.7f}": float(error) for y, error in
                              zip(reference["u"][0], analysis["errors"][("wall", "mls", "u")]["pointwise"]) if y < 0.065}},
        "profiles": {"dense": analysis["dense_reference"].tolist(),
                     "u": analysis["statistics"]["u_dense_mls"]["mean"].tolist(),
                     "v": analysis["statistics"]["v_dense_mls"]["mean"].tolist()},
    }
    return result


def gpu_label(item: dict) -> str:
    """GPU0 = the display 5090 (uuid fb83...), GPU1 = the headless one (ae13...)."""
    names = ["GPU0" if uuid.startswith("fb83") else "GPU1" for uuid in (item["fps"]["device_uuids"] or [])]
    return "+".join(names) + (f", K={item['slabs']}" if item["slabs"] > 1 else "")


def ghia_errors(analysis: dict) -> dict:
    """rms / max of (time-averaged dense MLS line - Ghia 1982) at Ghia's tabulated points without the two wall points
    (linear interpolation of the 1001-point line, wall-row frame), for orientation next to the Marchi errors."""
    ghia = cavity_reference.ghia1982()
    dense = analysis["dense_reference"]
    result = {}
    for line in ("u", "v"):
        coordinates, values = ghia[line]
        inside = (coordinates > 0.0) & (coordinates < 1.0)
        error = np.interp(coordinates[inside], dense, analysis["statistics"][f"{line}_dense_mls"]["mean"]) - values[inside]
        result[line] = {"L2": float(np.sqrt(np.mean(error ** 2))), "Linf": float(np.abs(error).max()),
                        "points": int(inside.sum())}
    return result


def deficit(value: float, reference: float) -> float:
    return 100.0 * (abs(value) - abs(reference)) / abs(reference)


def tables(results: list[dict], timing: dict | None) -> str:
    reference = cavity_reference.marchi2021()["extrema"]
    psi_ref, x_ref, y_ref = reference["psi_min"][0], reference["x_at_psi_min"][0], reference["y_at_psi_min"][0]
    lines = []
    header = ["量"] + [item["label"] for item in results]

    def row(name, cells):
        lines.append("| " + " | ".join([name] + cells) + " |")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    row("平均窗口(样本数)", [f"{item['window'][0]:.2f}–{item['window'][1]:.2f}({item['samples_in_window']})" for item in results])
    row("主涡 ψ_min(相对 Marchi)", [f"{item['stream']['vortices']['primary']['psi']:.5f}({deficit(item['stream']['vortices']['primary']['psi'], psi_ref):+.2f} %)"
                                   for item in results])
    row("主涡涡心 (x, y)(Δ 相对 Marchi)", [
        f"({item['stream']['vortices']['primary']['x']:.4f}, {item['stream']['vortices']['primary']['y']:.4f})"
        f"(Δ {item['stream']['vortices']['primary']['x'] - x_ref:+.4f}, {item['stream']['vortices']['primary']['y'] - y_ref:+.4f})"
        for item in results])
    row("lid 行上的 ψ:均值 / 最大绝对值", [f"{item['stream']['psi_top_mean']:+.2e} / {item['stream']['psi_top_max_abs']:.2e}"
                                             for item in results])
    for name, text in (("BR1", "右下二次涡 BR1"), ("BL1", "左下二次涡 BL1")):
        ghia = GHIA_VORTICES[name]
        row(f"{text} ψ(相对 Ghia 1982)", [f"{item['stream']['vortices'][name]['psi']:.3e}({deficit(item['stream']['vortices'][name]['psi'], ghia[0]):+.1f} %)"
                                       for item in results])
        row(f"{text} 位置", [f"({item['stream']['vortices'][name]['x']:.4f}, {item['stream']['vortices'][name]['y']:.4f})"
                             for item in results])
    for line, text in (("u", "竖直中线 u"), ("v", "水平中线 v")):
        row(f"{text}:L2 / 相对 L2 / L∞", [
            f"{item['errors'][line]['L2']:.4f} / {100 * item['errors'][line]['L2_relative']:.2f} % / {item['errors'][line]['Linf']:.4f}"
            for item in results])
    row("参考:相对 Ghia 1982 的 L2 / L∞(u;v,内部表列点)", [
        f"{item['errors_ghia']['u']['L2']:.4f} / {item['errors_ghia']['u']['Linf']:.4f};"
        f"{item['errors_ghia']['v']['L2']:.4f} / {item['errors_ghia']['v']['Linf']:.4f}" for item in results])
    for name, text in (("u_min", "u_min"), ("v_max", "v_max"), ("v_min", "v_min")):
        value_key, position_key = {"u_min": ("u_min", "y_at_u_min"), "v_max": ("v_max", "x_at_v_max"),
                                   "v_min": ("v_min", "x_at_v_min")}[name]
        row(f"{text}(相对 Marchi)", [f"{item['extrema'][name]['value']:.4f}({deficit(item['extrema'][name]['value'], reference[value_key][0]):+.2f} %)"
                                   for item in results])
    row("ν_eff / ν(内部,算子)", [f"{item['viscosity']['nu_eff_over_nu']:.4f}" for item in results])
    row("存储速度的质量平均 (ū, v̄)", [f"({item['mean_velocity']['u']:+.2e}, {item['mean_velocity']['v']:+.2e})" for item in results])
    row("窗口内流体密度 min / max", [f"{item['fluid_density_range_window'][0]:.2f} / {item['fluid_density_range_window'][1]:.2f}"
                                  for item in results])
    row("检查点流体平均密度", [", ".join(f"t={entry['time']:.1f}: {entry['fluid_density_mean']:.3f}" for entry in item["corners"]["checkpoints"])
                               for item in results])
    for factor in NEAR_WALL_DISTANCES_IN_H:
        row(f"u:lid 行下方 {factor:g}h", [f"{item['near_wall']['lid'][factor]:.4f}" for item in results])
    for factor in NEAR_WALL_DISTANCES_IN_H:
        row(f"u:底壁行上方 {factor:g}h", [f"{item['near_wall']['bottom'][factor]:.4f}" for item in results])
    for corner, text in (("top_left", "左上角"), ("top_right", "右上角")):
        for kind, kind_text in (("wall", "壁粒子"), ("fluid", "流体")):
            row(f"{text} 2h 内{kind_text}压力 (p − p̄_f)/(ρU²/2):均值 [min, max]" + ("(有流体邻居者)" if kind == "wall" else ""), [
                "; ".join(f"{entry[f'{corner}_{kind}']['mean']:+.2f} [{entry[f'{corner}_{kind}']['min']:+.2f}, {entry[f'{corner}_{kind}']['max']:+.2f}]"
                          for entry in item["corners"]["checkpoints"]) for item in results])
    row("fps(运行中位数;GPU)", [f"{item['fps']['median']:.0f}({gpu_label(item)})" for item in results])
    text = "\n".join(lines) + "\n"
    if timing:
        text += "\n" + timing_table(timing)
    return text


def timing_table(timing: dict) -> str:
    lines = ["| 配置(GPU1,21 000 步,最后一帧 GPU 时间戳中位数) | 每步内核 µs | 壁面 pass µs(占比) | force µs | density µs | fps |",
             "|---|---|---|---|---|---|"]
    for label, entry in timing.items():
        lines.append(f"| {label} | {entry['kernels']:.1f} | {entry['wall_pass']:.1f}({entry['wall_share']:.1f} %) | "
                     f"{entry['force']:.1f} | {entry['density']:.1f} | {entry['fps']:.0f} |")
    return "\n".join(lines) + "\n"


def timing_summary(directory: pathlib.Path) -> dict:
    result = {}
    for name, label in (("v6", "v6 K=1"), ("v7_bc0", "v7 K=1 WALL_BC=0"), ("v7_bc1", "v7 K=1 WALL_BC=1 (Adami)")):
        monitor = directory / f"{name}.monitor.jsonl"
        if not monitor.exists():
            continue
        rows = [json.loads(line) for line in monitor.read_text(encoding="utf-8").splitlines() if line]
        rows = [row for row in rows if row["step"] >= 3000]
        parts = {key: [] for key in ("predict", "update_voxel", "correction", "density", "wall_pass", "force", "phase_c")}
        for row in rows:
            ticks = row["ticks_ns"]

            def span(end, start):
                return (ticks[end] - ticks[start]) / 1000.0
            wall = "b_wall_extrapolate_end" in ticks
            parts["predict"].append(span("a_predict_end", "a_start"))
            parts["update_voxel"].append(span("a_voxel_end", "a_predict_end"))
            parts["correction"].append(span("b_correction_interior_end", "b_start"))
            parts["density"].append(span("b_density_deep_interior_end", "b_correction_interior_end"))
            parts["wall_pass"].append(span("b_wall_extrapolate_end", "b_density_deep_interior_end") if wall else 0.0)
            parts["force"].append(span("b_force_deep_interior_end",
                                       "b_wall_extrapolate_end" if wall else "b_density_deep_interior_end"))
            parts["phase_c"].append(span("c_force_end", "c_start"))
        medians = {key: float(np.median(values)) for key, values in parts.items()}
        kernels = sum(medians.values())
        with np.load(directory / f"{name}.npz") as archive:
            fps = json.loads(str(archive["meta"]))["fps"]
        result[label] = {**medians, "kernels": kernels, "wall_share": 100.0 * medians["wall_pass"] / kernels,
                         "fps": fps, "frames": len(rows)}
    return result


STYLES = [("#969696", "--"), ("#08519c", "-"), ("#d95f0e", "-"), ("#31a354", "-."), ("#756bb1", ":")]
DIAGNOSTIC_STYLES = [("#08519c", "-"), ("#31a354", "-."), ("#756bb1", ":"), ("#d95f0e", "-")]


def figures(results: list[dict], out: pathlib.Path) -> None:
    styles = STYLES if len(results) <= 3 else DIAGNOSTIC_STYLES
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cavity_analysis.setup_style()
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    insets = [axes[0].inset_axes([0.55, 0.09, 0.40, 0.42]), axes[1].inset_axes([0.15, 0.06, 0.34, 0.40])]
    for item, (color, style) in zip(results, styles):
        dense = np.array(item["profiles"]["dense"])
        u, v = np.array(item["profiles"]["u"]), np.array(item["profiles"]["v"])
        for axis, inset, x_data, y_data in ((axes[0], insets[0], u, dense), (axes[1], insets[1], dense, v)):
            axis.plot(x_data, y_data, style, color=color, lw=0.9, label=item["figure_label"])
            inset.plot(x_data, y_data, style, color=color, lw=0.8)
    for axis, inset, (x_data, y_data), (gx, gy) in (
            (axes[0], insets[0], (reference["u"][1], reference["u"][0]), (ghia["u"][1], ghia["u"][0])),
            (axes[1], insets[1], (reference["v"][0], reference["v"][1]), (ghia["v"][0], ghia["v"][1]))):
        axis.plot(x_data, y_data, "o", mfc="none", mec="k", ms=3.4, mew=0.7, label="Marchi et al. (2021)")
        axis.plot(gx, gy, "s", mfc="none", mec="#b2182b", ms=2.8, mew=0.6, label="Ghia et al. (1982)")
        inset.plot(x_data, y_data, "o", mfc="none", mec="k", ms=3.0, mew=0.6)
    insets[0].set_xlim(-0.40, -0.33)
    insets[0].set_ylim(0.12, 0.23)
    insets[1].set_xlim(0.86, 0.95)
    insets[1].set_ylim(-0.54, -0.44)
    for inset in insets:
        inset.tick_params(labelsize=6.5, length=2)
    axes[0].set_xlabel(r"$u(0.5, y)$")
    axes[0].set_ylabel(r"$y$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$v(x, 0.5)$")
    axes[0].set_ylim(0, 1)
    axes[1].set_xlim(0, 1)
    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.3)
    figure.tight_layout(w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"fig_centrelines.{suffix}", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    support = 0.02
    for item, (color, style) in zip(results, styles):
        for axis, key in ((axes[0], "lid_curve"), (axes[1], "bottom_curve")):
            curve = item["near_wall"][key]
            axis.plot(curve["u"], curve["y"], style, color=color, lw=0.9, label=item["figure_label"])
    for axis, mask in ((axes[0], reference["u"][0] > 0.93), (axes[1], reference["u"][0] < 0.065)):
        axis.plot(reference["u"][1][mask], reference["u"][0][mask], "o", mfc="none", mec="k", ms=3.4, mew=0.7,
                  label="Marchi et al. (2021)")
    length = 1.0 + 2.0 * 0.004
    axes[0].set_ylim(1.0 - 3 * support / length, 1.0)
    axes[1].set_ylim(0.0, 3 * support / length)
    axes[0].set_title(r"below the lid row ($1 - y \leq 3h$)", fontsize=8)
    axes[1].set_title(r"above the bottom row ($y \leq 3h$)", fontsize=8)
    for axis in axes:
        axis.set_xlabel(r"$u(0.5, y)$")
        axis.set_ylabel(r"$y$ (wall-row frame)")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    figure.tight_layout(w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"fig_near_wall.{suffix}", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(1, 1, figsize=(4.6, 2.8))
    for item, (color, style) in zip(results, styles):
        track = item["density_track"]
        axis.plot(track["time"], track["min"], style, color=color, lw=0.6, label=item["figure_label"] + " (min / max)")
        axis.plot(track["time"], track["max"], style, color=color, lw=0.6)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"fluid $\rho$ (kg/m$^3$)")
    axis.legend(frameon=False, fontsize=6.3)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"fig_density.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="table label[|figure label]::run directory (repeat, in table order)")
    parser.add_argument("--t-stop", type=float, default=100.0)
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--timing", default=None, help="directory of the same-GPU timing dumps (k1_dump --timestamps)")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in arguments.run:
        label, path = entry.rsplit("::", 1)
        label, figure_label = label.split("|", 1) if "|" in label else (label, label)
        print(f"[e36] analysing {label}: {path}", flush=True)
        result = analyse(label, pathlib.Path(path).resolve(), arguments.t_stop, arguments.average_span)
        result["figure_label"] = figure_label
        results.append(result)
    timing = timing_summary(pathlib.Path(arguments.timing)) if arguments.timing else None
    (out / "e36_summary.json").write_text(json.dumps({"runs": results, "timing": timing}, indent=1, default=float),
                                          encoding="utf-8")
    (out / "e36_tables.md").write_text(tables(results, timing), encoding="utf-8")
    figures(results, out)
    print((out / "e36_tables.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
