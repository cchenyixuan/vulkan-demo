"""cavity_v7_single_report.py - figures, data files, LaTeX rows and captions of one cavity_runner run of the
pure single-GPU build v7_single (docs/seam_audit/v7_single.md, the Re = 1000 accuracy validation; CPU only).

The analysis is cavity_analysis.analyse_run with the report's defaults (docs/validation/cavity_re1000.md): the
time average over the last --average-span time units, linear MLS (Wendland C4, support h) in the wall-row frame,
errors against Marchi et al. 2021 Tc (Tables 19-21), extrema from the 1001-point dense centre lines (parabolic
refinement), psi_min from the time-averaged 513 x 513 MLS grid of u integrated upward from the bottom wall, the
steady-state rule re-run sliding. Ghia et al. 1982 is drawn only.

Outputs (<out-fig>/<name>.pdf + .png, designed at the final double-column width, fonts >= 7 pt, no titles,
non-dimensional axes u/U, v/U, x/L, y/L):
  <prefix>_profiles   u(0.5, y) and v(x, 0.5): SPH dense lines, Marchi Tc (hollow circles), Ghia 1982 (hollow
                      squares), insets around u_min and v_min, the relative L2 of both profiles written in;
  <prefix>_field      (a) time-averaged stream function contours with the SPH primary-vortex centre and Marchi's,
                      (b) time-averaged vorticity omega_z L/U;
  <prefix>_steady     supplementary: fluid kinetic energy and the window criterion (cavity_re1000_steady's style);
  <prefix>_errors     supplementary: pointwise errors u_SPH - Tc, v_SPH - Tc at the 28 + 28 reference points.
Data (<out-data>): profile_<id>.csv and dense_<id>.csv (cavity_analysis.profile_csv), <prefix>_summary.json (every
number), <prefix>_tables.md (the LaTeX rows and the English captions).

    .venv/Scripts/python.exe -m experiment.validation.cavity_v7_single_report \\
        --run-root logs/validation/cavity_re1000_v7_single --run n1000_k1_float32_xi0p001_eps0p0025_adami \\
        --id-suffix _v7single --prefix cavity_re1000
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.cavity_analysis import (DATA, GRID_POINTS, analyse_run, load_run, profile_csv,  # noqa: E402
                                                   setup_style)
from experiment.validation.cavity_tables import invariants, wall_hours  # noqa: E402

SPH_COLOR = "#08519c"
MARCHI_COLOR = "#000000"
GHIA_COLOR = "#b2182b"
DOUBLE_COLUMN_WIDTH = 7.0          # inches, the final two-column text width
FONT_SIZE = 8.0


def velocity_grid_average(run: dict, window: tuple[float, float]) -> dict:
    """Time-averaged MLS velocity (u, v) on the 513 x 513 grid spanning the wall rows, over the snapshots inside
    ``window`` (cavity_analysis.stream_function_minimum's grid, both components), with psi integrated upward
    from the bottom wall and the vorticity by central differences; everything in the wall-row frame (positions
    scaled by 1 / (1 + 2 dx), psi by the same)."""
    spacing = run["spacing"]
    dt = float(run["meta"]["dt"])
    support = float(run["meta"]["support_radius"])
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    selected = [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * dt <= window[1] + 1e-9]
    if not selected:
        raise SystemExit("no snapshot inside the averaging window")
    half = 0.5 + spacing
    axis = np.linspace(-half, half, GRID_POINTS)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    total = np.zeros((points.shape[0], 2))
    for path in selected:
        with np.load(path) as archive:
            particles = sampling.ParticleSet(archive["positions"].astype(np.float64),
                                             archive["velocities"].astype(np.float64),
                                             archive["volumes"].astype(np.float64))
        tree = sampling.cKDTree(particles.positions)
        for start in range(0, points.shape[0], 40000):
            chunk = slice(start, start + 40000)
            total[chunk] += sampling.interpolate(particles, points[chunk], support, tree=tree)["mls"]
    velocity = total / len(selected)
    u_grid = velocity[:, 0].reshape(GRID_POINTS, GRID_POINTS)
    v_grid = velocity[:, 1].reshape(GRID_POINTS, GRID_POINTS)
    length = 1.0 + 2.0 * spacing
    psi = sampling.stream_function(u_grid, axis + half) / length          # psi / (U L)
    step = axis[1] - axis[0]
    vorticity = (np.gradient(v_grid, step, axis=1) - np.gradient(u_grid, step, axis=0)) * length   # omega L / U
    reference_axis = (axis + half) / length
    interior = (np.abs(xx) < 0.3) & (np.abs(yy) < 0.3)
    x_min, y_min, psi_min = sampling.refine_grid_minimum(axis, axis, np.where(interior, psi * length, np.nan))
    return {"axis": reference_axis, "u": u_grid, "v": v_grid, "psi": psi, "vorticity": vorticity,
            "snapshots": len(selected), "psi_min": psi_min / length, "x_psi_min": (x_min + half) / length,
            "y_psi_min": (y_min + half) / length}


def profiles_figure(analysis: dict, run: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    setup_style(FONT_SIZE)
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()
    dense = run["points"]["dense_reference"]
    u_mean = analysis["statistics"]["u_dense_mls"]["mean"]
    v_mean = analysis["statistics"]["v_dense_mls"]["mean"]
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.0))
    axes[0].plot(u_mean, dense, "-", color=SPH_COLOR, lw=1.0, label="SPH, time average")
    axes[1].plot(dense, v_mean, "-", color=SPH_COLOR, lw=1.0, label="SPH, time average")
    axes[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec=MARCHI_COLOR, ms=4.0, mew=0.7,
                 label="Marchi et al. (2021)")
    axes[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec=MARCHI_COLOR, ms=4.0, mew=0.7,
                 label="Marchi et al. (2021)")
    axes[0].plot(ghia["u"][1], ghia["u"][0], "s", mfc="none", mec=GHIA_COLOR, ms=3.2, mew=0.6, label="Ghia et al. (1982)")
    axes[1].plot(ghia["v"][0], ghia["v"][1], "s", mfc="none", mec=GHIA_COLOR, ms=3.2, mew=0.6, label="Ghia et al. (1982)")
    insets = [axes[0].inset_axes([0.52, 0.10, 0.42, 0.40]), axes[1].inset_axes([0.14, 0.08, 0.36, 0.40])]
    insets[0].plot(u_mean, dense, "-", color=SPH_COLOR, lw=0.9)
    insets[1].plot(dense, v_mean, "-", color=SPH_COLOR, lw=0.9)
    insets[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec=MARCHI_COLOR, ms=3.4, mew=0.6)
    insets[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec=MARCHI_COLOR, ms=3.4, mew=0.6)
    insets[0].plot(ghia["u"][1], ghia["u"][0], "s", mfc="none", mec=GHIA_COLOR, ms=2.8, mew=0.5)
    insets[1].plot(ghia["v"][0], ghia["v"][1], "s", mfc="none", mec=GHIA_COLOR, ms=2.8, mew=0.5)
    u_min = analysis["extrema"]["u_min"]
    v_min = analysis["extrema"]["v_min"]
    insets[0].set_xlim(u_min["value"] - 0.05, u_min["value"] + 0.06)
    insets[0].set_ylim(max(0.0, u_min["position_wall"] - 0.10), u_min["position_wall"] + 0.12)
    insets[1].set_xlim(max(0.0, v_min["position_wall"] - 0.06), min(1.0, v_min["position_wall"] + 0.05))
    insets[1].set_ylim(v_min["value"] - 0.04, v_min["value"] + 0.10)
    for inset in insets:
        inset.tick_params(labelsize=7, length=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
    error_u = analysis["errors"][("wall", "mls", "u")]["L2_relative"]
    error_v = analysis["errors"][("wall", "mls", "v")]["L2_relative"]
    axes[0].text(0.03, 0.56, rf"$L_2$ rel. $= {100 * error_u:.2f}\,\%$", transform=axes[0].transAxes, fontsize=7.5)
    axes[1].text(0.52, 0.93, rf"$L_2$ rel. $= {100 * error_v:.2f}\,\%$", transform=axes[1].transAxes, fontsize=7.5)
    axes[0].set_xlabel(r"$u/U$ at $x/L = 0.5$")
    axes[0].set_ylabel(r"$y/L$")
    axes[1].set_xlabel(r"$x/L$")
    axes[1].set_ylabel(r"$v/U$ at $y/L = 0.5$")
    axes[0].set_ylim(0, 1)
    axes[1].set_xlim(0, 1)
    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=7)
    figure.tight_layout(w_pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def field_figure(grid: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors
    setup_style(FONT_SIZE)
    reference = cavity_reference.marchi2021()["extrema"]
    axis = grid["axis"]
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.4))
    levels = [-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, -1e-5, -1e-7, -1e-10,
              1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3]   # Ghia 1982's psi levels
    axes[0].contour(axis, axis, grid["psi"], levels=sorted(levels), colors="#333333", linewidths=0.5)
    axes[0].plot(grid["x_psi_min"], grid["y_psi_min"], "o", color=SPH_COLOR, ms=5, mew=0, label="SPH primary vortex")
    axes[0].plot(reference["x_at_psi_min"][0], reference["y_at_psi_min"][0], "*", mfc="none", mec=MARCHI_COLOR, ms=9,
                 mew=0.8, label="Marchi et al. (2021)")
    axes[0].legend(frameon=False, loc="lower center", fontsize=7)
    clip = 5.0
    omega = np.clip(grid["vorticity"], -clip, clip)
    image = axes[1].imshow(omega, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                           norm=colors.Normalize(-clip, clip), interpolation="bilinear")
    axes[1].contour(axis, axis, grid["vorticity"], levels=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], colors="#222222",
                    linewidths=0.4)
    bar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    bar.set_label(r"$\omega_z L / U$")
    bar.ax.tick_params(labelsize=7)
    for axis_item, tag in zip(axes, ("(a)", "(b)")):
        axis_item.set_xlim(0, 1)
        axis_item.set_ylim(0, 1)
        axis_item.set_aspect("equal")
        axis_item.set_xlabel(r"$x/L$")
        axis_item.set_ylabel(r"$y/L$")
        axis_item.text(0.02, 0.98, tag, transform=axis_item.transAxes, va="top", ha="left",
                       bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.8))
    figure.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def steady_figure(analysis: dict, run: dict, tolerance: float, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    setup_style(FONT_SIZE)
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.7))
    times = run["times"]
    axes[0].plot(times, run["kinetic_energy"], "-", color=SPH_COLOR, lw=0.8)
    start, end = analysis["window"]
    axes[0].axvspan(start, end, color="#000000", alpha=0.06, lw=0)
    axes[0].text(0.5 * (start + end), 0.06, "time average", transform=axes[0].get_xaxis_transform(), fontsize=7,
                 color="#555555", ha="center", va="bottom")
    windows_path = run["dir"] / "windows.jsonl"
    if windows_path.exists():
        windows = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines() if line]
        window_times = [window["time"] for window in windows]
        axes[1].semilogy(window_times, [window["profile_change"] for window in windows], "-", marker="o", ms=3,
                         mfc=SPH_COLOR, mec=SPH_COLOR, color=SPH_COLOR, lw=0.8, label="profile (56 points)")
        axes[1].semilogy(window_times, [window["kinetic_energy_change"] for window in windows], "-", marker="o",
                         ms=3, mfc="none", mec=SPH_COLOR, color=SPH_COLOR, lw=0.5, label="kinetic energy")
    axes[1].axhline(tolerance, color="#777777", lw=0.6, ls=":")
    axes[1].text(2.0, tolerance * 1.25, f"{tolerance:g}", fontsize=7, color="#555555", va="bottom", ha="left")
    if analysis["t_steady_online"]:
        axes[1].axvline(analysis["t_steady_online"], color="#999999", lw=0.5, ls="--")
    axes[0].set_xlabel(r"$t\,U/L$")
    axes[0].set_ylabel(r"$\frac{1}{2}\sum_{\mathrm{fluid}} m\,|\mathbf{v}|^2 / (\rho_0 L^2 U^2)$")
    axes[1].set_xlabel(r"window end $t\,U/L$")
    axes[1].set_ylabel("relative change between windows")
    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.set_xlim(0, None)
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout(w_pad=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def errors_figure(analysis: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    setup_style(FONT_SIZE)
    reference = cavity_reference.marchi2021()
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.6), sharey=True)
    for axis, line, tag in zip(axes, ("u", "v"), ("(a)", "(b)")):
        error = analysis["errors"][("wall", "mls", line)]
        sem = analysis["statistics"][f"{line}_marchi_wall_mls"]["sem"]
        axis.errorbar(reference[line][0], error["pointwise"], yerr=2 * sem, fmt="o-", ms=3, mfc="none", mew=0.6,
                      lw=0.8, color=SPH_COLOR, elinewidth=0.5, capsize=1.5)
        axis.axhline(0.0, color="#777777", lw=0.5)
        axis.set_xlabel(r"$y/L$" if line == "u" else r"$x/L$")
        axis.set_xlim(0, 1)
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel(r"$\bar u/U - T_c$ (a), $\bar v/U - T_c$ (b)")
    figure.tight_layout(w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def latex_rows(analysis: dict, grid: dict, run: dict) -> tuple[str, dict]:
    reference = cavity_reference.marchi2021()["extrema"]
    inv = invariants({"segments": run["segments"]})
    rows = {}
    for name, (value_key, position_key) in (("u_min", ("u_min", "y_at_u_min")), ("v_max", ("v_max", "x_at_v_max")),
                                            ("v_min", ("v_min", "x_at_v_min"))):
        item = analysis["extrema"][name]
        reference_value, reference_position = reference[value_key][0], reference[position_key][0]
        rows[name] = {"value": item["value"], "sem": item["sem"], "position": item["position_wall"],
                      "reference_value": reference_value, "reference_position": reference_position,
                      "deviation_percent": 100.0 * (abs(item["value"]) - abs(reference_value)) / abs(reference_value),
                      "position_error": item["position_wall"] - reference_position}
    stream = analysis.get("stream") or {}
    psi_reference = reference["psi_min"][0]
    rows["psi_min"] = {"value": stream.get("psi_min_wall"), "position": (stream.get("x_wall"), stream.get("y_wall")),
                       "reference_value": psi_reference,
                       "reference_position": (reference["x_at_psi_min"][0], reference["y_at_psi_min"][0]),
                       "deviation_percent": 100.0 * (abs(stream["psi_min_wall"]) - abs(psi_reference)) / abs(psi_reference)
                       if stream else None,
                       "position_error": (stream["x_wall"] - reference["x_at_psi_min"][0],
                                          stream["y_wall"] - reference["y_at_psi_min"][0]) if stream else None,
                       "snapshots": stream.get("snapshots"), "sem": None}
    error_u, error_v = analysis["errors"][("wall", "mls", "u")], analysis["errors"][("wall", "mls", "v")]
    window = analysis["window"]
    lines = ["% v7_single, Re = 1000, 1000^2, adami walls: extrema and errors against Marchi et al. 2021 Tc (wall-row frame)",
             "% columns: quantity & SPH & Marchi 2021 & deviation (%) & std. error & position SPH & position Marchi & dx \\\\"]
    for name, label in (("u_min", r"$u_{\min}/U$"), ("v_max", r"$v_{\max}/U$"), ("v_min", r"$v_{\min}/U$")):
        row = rows[name]
        lines.append(f"{label} & {row['value']:.5f} & {row['reference_value']:.5f} & {row['deviation_percent']:+.2f} & "
                     f"{row['sem']:.1e} & {row['position']:.4f} & {row['reference_position']:.4f} & "
                     f"{row['position_error']:+.4f} \\\\")
    if stream:
        row = rows["psi_min"]
        lines.append(rf"$\psi_{{\min}}/(UL)$ & {row['value']:.5f} & {row['reference_value']:.5f} & "
                     f"{row['deviation_percent']:+.2f} & -- & ({row['position'][0]:.4f}, {row['position'][1]:.4f}) & "
                     f"({row['reference_position'][0]:.4f}, {row['reference_position'][1]:.4f}) & "
                     f"({row['position_error'][0]:+.4f}, {row['position_error'][1]:+.4f}) \\\\")
    lines.append("% errors: line & rel. L2 (%) & L2 & Linf & at \\\\")
    lines.append(rf"$u(0.5, y)$ & {100 * error_u['L2_relative']:.2f} & {error_u['L2']:.5f} & {error_u['Linf']:.5f} & "
                 f"$y = {error_u['Linf_at']:.4f}$ \\\\")
    lines.append(rf"$v(x, 0.5)$ & {100 * error_v['L2_relative']:.2f} & {error_v['L2']:.5f} & {error_v['Linf']:.5f} & "
                 f"$x = {error_v['Linf_at']:.4f}$ \\\\")
    lines.append(f"% steady state t_s = {analysis['t_steady_online']} (sliding {analysis['t_steady_sliding']}), "
                 f"average window ({window[0]:.2f}, {window[1]:.2f}], {analysis['samples_in_window']} samples, "
                 f"{stream.get('snapshots', 0)} snapshots; invariants drift {inv['drift']} overflow {inv['overflow']} "
                 f"far_migration {inv['far_migration']} stamp {inv['stamp_gpu']}/{inv['stamp_host']}; "
                 f"{inv['segments']} segment(s), {wall_hours({'segments': run['segments']}):.2f} h")
    summary = {"extrema": rows, "errors": {"u": {key: value for key, value in error_u.items() if key != "pointwise"},
                                           "v": {key: value for key, value in error_v.items() if key != "pointwise"}},
               "t_steady_online": analysis["t_steady_online"], "t_steady_sliding": analysis["t_steady_sliding"],
               "window": window, "samples_in_window": analysis["samples_in_window"],
               "kinetic_energy_window": {key: (value.tolist() if hasattr(value, "tolist") else value)
                                         for key, value in analysis["kinetic_energy_window"].items()},
               "invariants": inv, "wall_hours": wall_hours({"segments": run["segments"]}),
               "grid_psi_min": {"psi_min": grid["psi_min"], "x": grid["x_psi_min"], "y": grid["y_psi_min"],
                                "snapshots": grid["snapshots"]}}
    return "\n".join(lines), summary


def captions(analysis: dict, summary: dict, prefix: str) -> str:
    error_u = 100 * analysis["errors"][("wall", "mls", "u")]["L2_relative"]
    error_v = 100 * analysis["errors"][("wall", "mls", "v")]["L2_relative"]
    window = analysis["window"]
    psi = summary["extrema"]["psi_min"]
    return "\n\n".join([
        f"**{prefix}_profiles.** Lid-driven cavity at Re = 1000, 1000 x 1000 fluid particles (h = 5 dx), Adami wall "
        f"boundary, single GPU. Time-averaged centre-line velocity profiles over t U/L in ({window[0]:.1f}, {window[1]:.1f}] "
        f"({analysis['samples_in_window']} samples): (a) u/U along the vertical centre line x/L = 0.5, (b) v/U along the "
        f"horizontal centre line y/L = 0.5. Solid line: SPH, linear MLS interpolation on 1001 points; hollow circles: the "
        f"converged values T_c of Marchi et al. (2021); hollow squares: Ghia et al. (1982), for orientation only. The insets "
        f"enlarge the neighbourhoods of u_min and v_min. The relative L2 error at the 28 + 28 reference points is "
        f"{error_u:.2f} % (u) and {error_v:.2f} % (v).",
        f"**{prefix}_field.** The same run, time-averaged fields on a 513 x 513 grid (linear MLS of the particle velocities "
        f"over {summary['grid_psi_min']['snapshots']} snapshots inside the averaging window). (a) Stream function "
        f"psi/(UL) integrated upward from the bottom wall (contour levels of Ghia et al. 1982); the filled circle marks "
        f"the SPH primary-vortex centre at ({psi['position'][0]:.4f}, {psi['position'][1]:.4f}), the hollow star Marchi "
        f"et al.'s ({psi['reference_position'][0]:.4f}, {psi['reference_position'][1]:.4f}); psi_min deviates by "
        f"{psi['deviation_percent']:+.2f} %. (b) Vorticity omega_z L/U (central differences of the averaged field, "
        f"clipped to +-5) with contours at 0, +-0.5, +-1, +-2, +-3.",
        f"**{prefix}_steady (supplementary).** (a) Fluid kinetic energy against time; the shaded band is the averaging "
        f"window. (b) The steady-state criterion of the run: relative change, between consecutive windows of about 10 "
        f"time units, of the window-mean velocity at the 56 reference points (filled) and of the window-mean kinetic energy "
        f"(hollow); the dotted line is the threshold 10^-3, the dashed line the first window end satisfying it "
        f"(t_s = {analysis['t_steady_online']}).",
        f"**{prefix}_errors (supplementary).** Pointwise error of the time-averaged SPH velocity against T_c at the 28 "
        f"reference points of each centre line, (a) u and (b) v; error bars are +-2 standard errors of the time average "
        f"(integrated autocorrelation time included).",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="the campaign root holding the run directory")
    parser.add_argument("--run", required=True, help="the run id (directory name)")
    parser.add_argument("--id-suffix", default="_v7single", help="appended to the run id in the data file names")
    parser.add_argument("--prefix", default="cavity_re1000", help="figure name prefix")
    parser.add_argument("--out-fig", default="manuscripts/fig")
    parser.add_argument("--out-data", default=str(DATA))
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--no-field", action="store_true", help="skip the 513^2 grids (psi_min, the field figure)")
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    started = time.time()
    run = load_run(arguments.run, root=_REPO_ROOT / arguments.run_root)
    if run is None:
        raise SystemExit(f"no run at {arguments.run_root}/{arguments.run}")
    analysis = analyse_run(run, arguments.average_span, arguments.steady_tol, arguments.window,
                           with_stream_function=not arguments.no_field)
    analysis["id"] = arguments.run + arguments.id_suffix
    analysis["dense_reference"] = run["points"]["dense_reference"]
    out_fig = _REPO_ROOT / arguments.out_fig
    out_data = pathlib.Path(arguments.out_data)
    if not out_data.is_absolute():
        out_data = _REPO_ROOT / out_data
    import experiment.validation.cavity_analysis as analysis_module
    analysis_module.DATA = out_data
    profile_csv(analysis)
    prefix = arguments.prefix
    profiles_figure(analysis, run, out_fig / f"{prefix}_profiles")
    steady_figure(analysis, run, arguments.steady_tol, out_fig / f"{prefix}_steady")
    errors_figure(analysis, out_fig / f"{prefix}_errors")
    grid = None
    if not arguments.no_field:
        grid = velocity_grid_average(run, analysis["window"])
        field_figure(grid, out_fig / f"{prefix}_field")
    else:
        grid = {"psi_min": None, "x_psi_min": None, "y_psi_min": None, "snapshots": 0}
    rows, summary = latex_rows(analysis, grid, run)
    summary["run"] = {"root": arguments.run_root, "id": arguments.run, "meta": run["meta"], "result": run["result"],
                      "segments": run["segments"]}
    text = captions(analysis, summary, prefix)
    tables_path = out_data / f"{prefix}_tables{arguments.id_suffix}.md"
    tables_path.write_text(f"# {arguments.run} ({arguments.run_root})\n\n```latex\n{rows}\n```\n\n## Captions\n\n{text}\n",
                           encoding="utf-8")
    (out_data / f"{prefix}_summary{arguments.id_suffix}.json").write_text(json.dumps(summary, indent=1, default=float),
                                                                           encoding="utf-8")
    print(rows)
    print()
    print(text)
    print(f"\n[report] {time.time() - started:.0f} s -> {out_fig / prefix}_*.pdf/.png, {tables_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
