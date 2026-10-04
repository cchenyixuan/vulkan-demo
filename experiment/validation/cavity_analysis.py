"""cavity_analysis.py - analysis of the Re = 1000 cavity validation runs (CPU only).

For every run under logs/validation/cavity_re1000/<id>/ (written by cavity_runner.py):
  steady state: the runner's window decision (steady.json) and a sliding recomputation of the same rule;
  time average over the last --average-span time units (mean, std, sample count, standard error with the
      integrated autocorrelation time) of the MLS / Shepard velocities at the Marchi 2021 points (both
      frames) and on the dense centre lines;
  errors at the 28 + 28 reference points (L2 = rms of the pointwise error, Linf = max), extrema (u_min,
      v_max, v_min and positions, from the time-averaged dense lines) and the stream-function minimum and
      vortex centre (time-averaged MLS grid of u from the snapshots, psi integrated upward from the bottom wall).
Across runs: convergence orders, K = 1 vs K = 2, float32 vs delta density. Outputs: CSVs in
docs/validation/data/, vector figures in docs/validation/figures/, tables in
docs/validation/data/tables.md (for docs/validation/cavity_re1000.md).

    .venv/Scripts/python.exe -m experiment.validation.cavity_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402

LOGS = _REPO_ROOT / "logs" / "validation" / "cavity_re1000"
DATA = _REPO_ROOT / "docs" / "validation" / "data"
FIGURES = _REPO_ROOT / "docs" / "validation" / "figures"
RESOLUTIONS = {"n250": 250, "n500": 500, "n1000": 1000, "n2000": 2000}
EXTREMA_SEARCH = {"u_min": ("u", "min", (0.05, 0.45)), "v_max": ("v", "max", (0.05, 0.45)),
                  "v_min": ("v", "min", (0.60, 0.99))}
GRID_POINTS = 513
# numerics of the release cases; runs with other values get a variant suffix (e.g. float32_xi0.001)
BASELINE_NUMERICS = {"xi": 0.1, "epsilon_factor": 0.01}
# frames by the physical half width of the benchmark's unit square (dx = particle spacing): wall = centre lines of
# the innermost wall / lid rows, mid = half way between those rows and the outermost fluid rows (the usual SPH wall
# position), fluid = the fluid lattice. The wall-frame Reynolds number is U (1 + 2 dx) / nu, etc.
FRAME_HALF_WIDTH = {"wall": lambda dx: 0.5 + dx, "mid": lambda dx: 0.5 + dx / 2, "fluid": lambda dx: 0.5}


# ----------------------------------------------------------------------------------------------- loading
def load_run(run_id: str, root: pathlib.Path | None = None) -> dict | None:
    directory = (root or LOGS) / run_id
    if not (directory / "samples.jsonl").exists():
        return None
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    arrays: dict[str, list] = {}
    for row in rows:
        with np.load(directory / "samples" / f"s{row['step']:010d}.npz") as archive:
            for name in archive.files:
                arrays.setdefault(name, []).append(archive[name])
    run = {"id": run_id, "dir": directory, "meta": meta, "rows": rows,
           "times": np.array([row["time"] for row in rows]),
           "kinetic_energy": np.array([row["kinetic_energy"] for row in rows]),
           "arrays": {name: np.array(values) for name, values in arrays.items()},
           "points": dict(np.load(directory / "points.npz")),
           "steady": json.loads((directory / "steady.json").read_text(encoding="utf-8"))
           if (directory / "steady.json").exists() else {},
           "result": json.loads((directory / "result.json").read_text(encoding="utf-8"))
           if (directory / "result.json").exists() else {},
           "segments": [json.loads(line) for line in (directory / "segments.jsonl").read_text(encoding="utf-8").splitlines()
                        if line] if (directory / "segments.jsonl").exists() else []}
    spacing = float(meta["spacing"])
    numerics = yaml.safe_load((_REPO_ROOT / meta["case"]).read_text(encoding="utf-8"))["numerics"]
    xi = float(numerics["regularization"]["xi"])
    epsilon_factor = float(numerics.get("epsilon_squared_factor", BASELINE_NUMERICS["epsilon_factor"]))
    storage = "delta" if meta["expect"] == "release_delta" else "float32"
    run.update({"case": pathlib.Path(meta["case"]).parent.name, "slabs": int(meta["slabs"]),
                "variant": storage + numerics_suffix(xi, epsilon_factor), "storage": storage, "xi": xi,
                "epsilon_factor": epsilon_factor, "spacing": spacing, "resolution": int(round(1.0 / spacing))})
    return run


def numerics_suffix(xi: float, epsilon_factor: float) -> str:
    """'' for the release numerics, else '_xi<xi>' and / or '_eps<factor>'."""
    suffix = "" if xi == BASELINE_NUMERICS["xi"] else f"_xi{xi:g}"
    return suffix + ("" if epsilon_factor == BASELINE_NUMERICS["epsilon_factor"] else f"_eps{epsilon_factor:g}")


# ----------------------------------------------------------------------------------------------- statistics
def integrated_autocorrelation_time(series: np.ndarray) -> float:
    """1 + 2 sum of the normalised autocorrelation up to its first non-positive lag (in samples)."""
    values = np.asarray(series, dtype=np.float64) - np.mean(series)
    variance = np.dot(values, values)
    if variance <= 0 or values.size < 4:
        return 1.0
    tau = 1.0
    for lag in range(1, values.size // 3):
        rho = np.dot(values[:-lag], values[lag:]) / variance
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def window_statistics(matrix: np.ndarray) -> dict:
    """Mean, std, n and standard error (with the integrated autocorrelation time) per column."""
    count = matrix.shape[0]
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if count > 1 else np.zeros(matrix.shape[1])
    tau = np.array([integrated_autocorrelation_time(matrix[:, column]) for column in range(matrix.shape[1])])
    return {"mean": mean, "std": std, "n": count, "tau": tau, "sem": std * np.sqrt(tau / max(count, 1))}


def sliding_steady_time(times: np.ndarray, profile: np.ndarray, kinetic_energy: np.ndarray, span: float,
                        tolerance: float) -> tuple[float | None, np.ndarray, np.ndarray]:
    """The first time t after which, at every later sample time, the means over [t - span, t] and
    [t - 2 span, t - span] of the profile (relative L2) and of the kinetic energy (relative) differ by less
    than ``tolerance``. Returns (t_steady or None, profile changes, KE changes) per sample (NaN before 2 spans)."""
    changes, ke_changes = np.full(times.size, np.nan), np.full(times.size, np.nan)
    for index, time_value in enumerate(times):
        if time_value < 2 * span:
            continue
        now = (times > time_value - span) & (times <= time_value)
        before = (times > time_value - 2 * span) & (times <= time_value - span)
        if now.sum() < 3 or before.sum() < 3:
            continue
        mean_now, mean_before = profile[now].mean(axis=0), profile[before].mean(axis=0)
        changes[index] = np.linalg.norm(mean_now - mean_before) / np.linalg.norm(mean_now)
        ke_now, ke_before = kinetic_energy[now].mean(), kinetic_energy[before].mean()
        ke_changes[index] = abs(ke_now - ke_before) / ke_now
    ok = (changes < tolerance) & (ke_changes < tolerance)
    valid = np.isfinite(changes)
    steady = None
    for index in range(times.size):
        if valid[index] and ok[index:][valid[index:]].all():
            steady = float(times[index])
            break
    return steady, changes, ke_changes


# ----------------------------------------------------------------------------------------------- per run
def analyse_run(run: dict, average_span: float, tolerance: float, window: float, with_stream_function: bool) -> dict:
    reference = cavity_reference.marchi2021()
    times = run["times"]
    arrays = run["arrays"]
    t_last = float(times.max())
    in_window = times > t_last - average_span - 1e-9
    profile = np.concatenate([arrays["u_marchi_wall_mls"], arrays["v_marchi_wall_mls"]], axis=1)
    t_sliding, changes, ke_changes = sliding_steady_time(times, profile, run["kinetic_energy"], window, tolerance)
    analysis = {"id": run["id"], "case": run["case"], "slabs": run["slabs"], "variant": run["variant"],
                "storage": run["storage"], "xi": run["xi"], "epsilon_factor": run["epsilon_factor"],
                "resolution": run["resolution"], "spacing": run["spacing"], "t_last": t_last,
                "window": (float(times[in_window].min()), t_last), "samples_in_window": int(in_window.sum()),
                "t_steady_online": run["steady"].get("steady_time"), "t_steady_sliding": t_sliding,
                "kinetic_energy_window": window_statistics(run["kinetic_energy"][in_window, None]),
                "profile_changes": changes, "ke_changes": ke_changes}
    statistics = {}
    for name in ("u_marchi_wall_mls", "v_marchi_wall_mls", "u_marchi_wall_shepard", "v_marchi_wall_shepard",
                 "u_marchi_fluid_mls", "v_marchi_fluid_mls", "u_dense_mls", "v_dense_mls"):
        statistics[name] = window_statistics(arrays[name][in_window])
    analysis["statistics"] = statistics
    errors = {}
    for frame in ("wall", "fluid"):
        for method in ("mls", "shepard"):
            key = f"marchi_{frame}_{method}"
            if f"u_{key}" not in statistics:
                continue
            for line in ("u", "v"):
                error = statistics[f"{line}_{key}"]["mean"] - reference[line][1]
                errors[(frame, method, line)] = {
                    "L2": float(np.sqrt(np.mean(error ** 2))), "Linf": float(np.abs(error).max()),
                    "L2_relative": float(np.sqrt(np.mean(error ** 2)) / np.sqrt(np.mean(reference[line][1] ** 2))),
                    "Linf_at": float(reference[line][0][np.argmax(np.abs(error))]), "pointwise": error}
    # every frame from the time-averaged dense lines (cubic spline in the physical coordinate along the line);
    # for the wall and fluid frames this reproduces the directly sampled points (consistency check)
    from scipy.interpolate import CubicSpline
    dense_physical = sampling.Frame.make("wall", run["spacing"]).to_physical(run["points"]["dense_reference"])
    analysis["frame_reynolds"] = {}
    for frame_name, half_width in FRAME_HALF_WIDTH.items():
        half = half_width(run["spacing"])
        analysis["frame_reynolds"][frame_name] = 1000.0 * 2.0 * half
        for line in ("u", "v"):
            spline = CubicSpline(dense_physical, statistics[f"{line}_dense_mls"]["mean"])
            values = spline(reference[line][0] * 2.0 * half - half)
            error = values - reference[line][1]
            errors[(frame_name, "dense", line)] = {
                "L2": float(np.sqrt(np.mean(error ** 2))), "Linf": float(np.abs(error).max()),
                "L2_relative": float(np.sqrt(np.mean(error ** 2)) / np.sqrt(np.mean(reference[line][1] ** 2))),
                "Linf_at": float(reference[line][0][np.argmax(np.abs(error))]), "pointwise": error,
                "values": values}
    analysis["errors"] = errors
    dense = run["points"]["dense_reference"]
    wall_frame = sampling.Frame.make("wall", run["spacing"])
    extrema = {}
    for name, (line, kind, search) in EXTREMA_SEARCH.items():
        mean = statistics[f"{line}_dense_mls"]["mean"]
        position, value = sampling.refine_extremum(dense, mean, kind, search=search)
        index = int(np.argmin(np.abs(dense - position)))
        extrema[name] = {"value": value, "position_wall": position,
                         "position_fluid": float(wall_frame.to_physical(position) + 0.5),
                         "sem": float(statistics[f"{line}_dense_mls"]["sem"][index])}
    analysis["extrema"] = extrema
    if with_stream_function:
        analysis["stream"] = stream_function_minimum(run, analysis["window"])
    return analysis


def stream_function_minimum(run: dict, window: tuple[float, float]) -> dict | None:
    """Time-averaged MLS grid of u over the snapshots inside ``window`` (physical grid spanning the wall
    rows), psi integrated upward from the bottom wall row, minimum refined by a quadratic surface. The wall
    frame scales psi by 1 / (1 + 2 dx); the fluid frame shifts positions by 0.5."""
    spacing = run["spacing"]
    dt = float(run["meta"]["dt"])
    support = float(run["meta"]["support_radius"])
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    selected = [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * dt <= window[1] + 1e-9]
    if not selected:
        return None
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
    psi_physical = sampling.stream_function(u_grid, axis + half)
    interior = (np.abs(xx) < 0.3) & (np.abs(yy) < 0.3)
    masked = np.where(interior, psi_physical, np.nan)
    x_min, y_min, psi_min = sampling.refine_grid_minimum(axis, axis, masked)
    length = 1.0 + 2.0 * spacing
    return {"snapshots": len(selected), "psi_min_wall": psi_min / length,
            "x_wall": (x_min + half) / length, "y_wall": (y_min + half) / length,
            "psi_min_fluid": psi_min - float(np.interp(-0.5, axis, psi_physical[:, int(np.argmin(np.abs(axis - x_min)))])),
            "x_fluid": x_min + 0.5, "y_fluid": y_min + 0.5,
            "psi_top_max_abs": float(np.abs(psi_physical[-1]).max())}


# ----------------------------------------------------------------------------------------------- outputs
def write_csv(path: pathlib.Path, header: list[str], rows: list[list], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        for line in comment.strip().splitlines():
            handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def profile_csv(analysis: dict) -> None:
    reference = cavity_reference.marchi2021()
    statistics = analysis["statistics"]
    rows = []
    for line, coordinate_name in (("u", "y"), ("v", "x")):
        coordinates, values, uncertainties = reference[line]
        wall = statistics[f"{line}_marchi_wall_mls"]
        for index, coordinate in enumerate(coordinates):
            rows.append([f"{line}_at_{'x' if line == 'u' else 'y'}_0.5", coordinate_name, f"{coordinate:.7f}",
                         f"{values[index]:.10f}", f"{wall['mean'][index]:.7f}", f"{wall['sem'][index]:.2e}",
                         f"{wall['std'][index]:.2e}", wall["n"], f"{wall['mean'][index] - values[index]:+.7f}",
                         f"{statistics[f'{line}_marchi_wall_shepard']['mean'][index]:.7f}",
                         f"{statistics[f'{line}_marchi_fluid_mls']['mean'][index]:.7f}"])
    window = analysis["window"]
    write_csv(DATA / f"profile_{analysis['id']}.csv",
              ["line", "coordinate_name", "coordinate", "marchi2021_Tc", "sph_mean", "sph_sem", "sph_std", "samples",
               "error", "sph_mean_shepard", "sph_mean_fluid_frame"],
              rows, f"""Run {analysis['id']}: time-averaged SPH velocity at the Marchi 2021 points, t in [{window[0]:.2f}, {window[1]:.2f}].
sph_mean: linear MLS, Wendland C4 weight, support radius h, wall-row frame (unit square = centre lines of the innermost
wall / lid rows, L = 1 + 2 dx). sph_sem = std * sqrt(tau_int / n). error = sph_mean - Tc.
sph_mean_shepard: Shepard in the same frame; sph_mean_fluid_frame: MLS with the unit square = the fluid box (L = 1).""")
    dense = analysis["dense_reference"]
    rows = [[f"{dense[index]:.4f}", f"{analysis['statistics']['u_dense_mls']['mean'][index]:.7f}",
             f"{analysis['statistics']['v_dense_mls']['mean'][index]:.7f}"] for index in range(dense.size)]
    write_csv(DATA / f"dense_{analysis['id']}.csv", ["coordinate", "u_at_x_0.5", "v_at_y_0.5"], rows,
              f"Run {analysis['id']}: time-averaged MLS velocity on the dense centre lines (wall-row frame), "
              f"t in [{window[0]:.2f}, {window[1]:.2f}].")


def setup_style(font_size: float = 8.5) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": font_size, "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1, "xtick.labelsize": font_size - 0.5, "ytick.labelsize": font_size - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


SHADES = {250: "#9ecae1", 500: "#4292c6", 1000: "#08519c", 2000: "#08306b"}
CONTROL_COLOR = "#e6550d"
MAIN_VARIANT = "float32_xi0.001_eps0.0025"
# numerics settings compared in the report, release first
SETTINGS = ("float32", "float32_xi0.001", MAIN_VARIANT)
# (line style, marker, legend text) per variant
VARIANT_STYLES = {MAIN_VARIANT: ("-", "s", r"$\xi = 0.001$, $\varepsilon^2 = 0.0025h^2$"),
                  "float32_xi0.001": ("-.", "^", r"$\xi = 0.001$, $\varepsilon^2 = 0.01h^2$"),
                  "float32": ("--", "o", r"$\xi = 0.1$, $\varepsilon^2 = 0.01h^2$ (release)"),
                  "delta": (":", "x", r"$\delta\rho$, $\xi = 0.1$, $\varepsilon^2 = 0.01h^2$")}
# one colour per setting where the settings are drawn at a single resolution
SETTING_COLORS = {"float32": "#969696", "float32_xi0.001": "#6baed6", MAIN_VARIANT: "#08519c"}


def variant_style(variant: str) -> tuple[str, str, str]:
    return VARIANT_STYLES.get(variant, (":", "x", variant.replace("_", " ")))


def profile_figure(analyses: list[dict], variant: str, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    setup_style()
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    selected = sorted((item for item in analyses if item["variant"] == variant and item["slabs"] == 2),
                      key=lambda item: item["resolution"])
    for item in selected:
        dense = item["dense_reference"]
        label = rf"SPH, $\Delta x = 1/{item['resolution']}$"
        axes[0].plot(item["statistics"]["u_dense_mls"]["mean"], dense, "-", color=SHADES[item["resolution"]],
                     label=label, lw=0.9)
        axes[1].plot(dense, item["statistics"]["v_dense_mls"]["mean"], "-", color=SHADES[item["resolution"]],
                     label=label, lw=0.9)
    axes[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec="k", ms=3.6, mew=0.7,
                 label="Marchi et al. (2021)")
    axes[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec="k", ms=3.6, mew=0.7,
                 label="Marchi et al. (2021)")
    axes[0].plot(ghia["u"][1], ghia["u"][0], "s", mfc="none", mec="#b2182b", ms=3.0, mew=0.6, label="Ghia et al. (1982)")
    axes[1].plot(ghia["v"][0], ghia["v"][1], "s", mfc="none", mec="#b2182b", ms=3.0, mew=0.6, label="Ghia et al. (1982)")
    insets = [axes[0].inset_axes([0.55, 0.09, 0.40, 0.42]), axes[1].inset_axes([0.15, 0.06, 0.34, 0.40])]
    for item in selected:
        dense = item["dense_reference"]
        insets[0].plot(item["statistics"]["u_dense_mls"]["mean"], dense, "-", color=SHADES[item["resolution"]], lw=0.8)
        insets[1].plot(dense, item["statistics"]["v_dense_mls"]["mean"], "-", color=SHADES[item["resolution"]], lw=0.8)
    insets[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec="k", ms=3.2, mew=0.6)
    insets[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec="k", ms=3.2, mew=0.6)
    insets[0].set_xlim(-0.40, -0.30)
    insets[0].set_ylim(0.08, 0.30)
    insets[1].set_xlim(0.85, 0.96)
    insets[1].set_ylim(-0.54, -0.40)
    for inset in insets:
        inset.tick_params(labelsize=6.5, length=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
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
    axes[1].legend(frameon=False, loc="upper right")
    figure.tight_layout(w_pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def error_figure(analyses: list[dict], path: pathlib.Path, series: list[dict]) -> None:
    """Pointwise error (SPH minus Marchi 2021 Tc) at the 28 + 28 reference points, wall-row frame. One line per K = 2
    run of every series entry {variant, style, resolutions (optional), color (optional, default by resolution),
    label (optional) or suffix (appended to the resolution label)}."""
    import matplotlib.pyplot as plt
    setup_style()
    reference = cavity_reference.marchi2021()
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    entries = 0
    for spec in series:
        runs = sorted((entry for entry in analyses if entry["variant"] == spec["variant"] and entry["slabs"] == 2
                       and entry["resolution"] in spec.get("resolutions", SHADES)), key=lambda entry: entry["resolution"])
        for item in runs:
            color = spec.get("color") or SHADES[item["resolution"]]
            label = spec.get("label") or (rf"$\Delta x = 1/{item['resolution']}$" + spec.get("suffix", ""))
            entries += 1
            for axis, line in zip(axes, ("u", "v")):
                axis.plot(reference[line][0], item["errors"][("wall", "mls", line)]["pointwise"], spec["style"],
                          marker="o", ms=2.6, mfc="none", mew=0.6, lw=0.8, color=color, label=label)
    for axis, line, tag in zip(axes, ("u", "v"), ("(a)", "(b)")):
        axis.axhline(0.0, color="#777777", lw=0.5)
        axis.set_xlabel(r"$y$" if line == "u" else r"$x$")
        axis.set_xlim(0, 1)
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel(r"$u_{\mathrm{SPH}} - u_{\mathrm{ref}}$ (a), $v_{\mathrm{SPH}} - v_{\mathrm{ref}}$ (b)")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=1 if entries <= 3 else 2)
    figure.tight_layout(w_pad=1.0)
    figure.savefig(path)
    plt.close(figure)


def settings_profile_figure(analyses: list[dict], path: pathlib.Path, settings: list[str], resolution: int) -> None:
    """Centre-line profiles of the numerics settings at one resolution (K = 2), with insets around u_min and v_min
    that mark each extremum (filled dot: SPH, hollow star: Marchi 2021 Table 19)."""
    import matplotlib.pyplot as plt
    setup_style()
    reference = cavity_reference.marchi2021()
    extrema = reference["extrema"]
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    insets = [axes[0].inset_axes([0.55, 0.09, 0.40, 0.42]), axes[1].inset_axes([0.15, 0.06, 0.34, 0.40])]
    for variant in settings:
        item = next((entry for entry in analyses if entry["variant"] == variant and entry["slabs"] == 2
                     and entry["resolution"] == resolution), None)
        if item is None:
            continue
        style, _, label = variant_style(variant)
        color = SETTING_COLORS.get(variant, "#555555")
        dense = item["dense_reference"]
        u_mean, v_mean = item["statistics"]["u_dense_mls"]["mean"], item["statistics"]["v_dense_mls"]["mean"]
        axes[0].plot(u_mean, dense, style, color=color, lw=0.9, label=label)
        axes[1].plot(dense, v_mean, style, color=color, lw=0.9, label=label)
        insets[0].plot(u_mean, dense, style, color=color, lw=0.8)
        insets[1].plot(dense, v_mean, style, color=color, lw=0.8)
        insets[0].plot(item["extrema"]["u_min"]["value"], item["extrema"]["u_min"]["position_wall"], "o", color=color, ms=3)
        insets[1].plot(item["extrema"]["v_min"]["position_wall"], item["extrema"]["v_min"]["value"], "o", color=color, ms=3)
    for axis, inset, (x_data, y_data) in ((axes[0], insets[0], (reference["u"][1], reference["u"][0])),
                                          (axes[1], insets[1], (reference["v"][0], reference["v"][1]))):
        axis.plot(x_data, y_data, "o", mfc="none", mec="k", ms=3.6, mew=0.7, label="Marchi et al. (2021)")
        inset.plot(x_data, y_data, "o", mfc="none", mec="k", ms=3.2, mew=0.6)
    insets[0].plot(extrema["u_min"][0], extrema["y_at_u_min"][0], "*", mfc="none", mec="k", ms=7, mew=0.7)
    insets[1].plot(extrema["x_at_v_min"][0], extrema["v_min"][0], "*", mfc="none", mec="k", ms=7, mew=0.7)
    insets[0].set_xlim(-0.40, -0.34)
    insets[0].set_ylim(0.12, 0.22)
    insets[1].set_xlim(0.87, 0.95)
    insets[1].set_ylim(-0.54, -0.46)
    for inset in insets:
        inset.tick_params(labelsize=6.5, length=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
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
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.5)
    figure.tight_layout(w_pad=1.5)
    figure.savefig(path)
    plt.close(figure)


def convergence_figure(analyses: list[dict], path: pathlib.Path, variants: tuple[str, ...] = ("float32", "delta")) -> None:
    import matplotlib.pyplot as plt
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    styles = {variant: variant_style(variant)[:2] for variant in variants}
    names = {variant: variant_style(variant)[2] for variant in variants}
    for axis, norm in zip(axes, ("L2", "Linf")):
        for variant, (line_style, marker) in styles.items():
            selected = sorted((item for item in analyses if item["variant"] == variant and item["slabs"] == 2),
                              key=lambda item: item["resolution"])
            if len(selected) < 2:
                continue
            spacing = np.array([1.0 / item["resolution"] for item in selected])
            for line, color in (("u", "#08519c"), ("v", "#b2182b")):
                values = np.array([item["errors"][("wall", "mls", line)][norm] for item in selected])
                axis.loglog(spacing, values, line_style, marker=marker, color=color, mfc="none", ms=4, mew=0.7,
                            lw=0.9, label=f"{line}, {names[variant]}")
        coarse = [item for item in analyses if item["slabs"] == 2 and item["resolution"] == 250]
        if coarse:
            start = 0.75 * min(item["errors"][("wall", "mls", line)][norm] for item in coarse for line in ("u", "v"))
            spacing = np.array([1 / 250, 1 / 1000])
            for order, label in ((1, "order 1"), (2, "order 2")):
                values = start * (spacing / spacing[0]) ** order
                axis.loglog(spacing, values, ":", color="#777777", lw=0.7)
                axis.text(spacing[1] * 1.06, values[1] * 1.08, label, fontsize=7, color="#555555", va="bottom",
                          ha="left")
        from matplotlib import ticker
        axis.set_xticks([1 / 1000, 1 / 500, 1 / 250])
        axis.set_xticklabels(["1/1000", "1/500", "1/250"])
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_yticks([0.005, 0.01, 0.02, 0.05])
        axis.set_yticklabels(["0.005", "0.01", "0.02", "0.05"])
        axis.set_xlabel(r"$\Delta x$")
        axis.set_title(r"$L_2$" if norm == "L2" else r"$L_\infty$", fontsize=8.5)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("error at the reference points")
    axes[1].legend(frameon=False, fontsize=6.0 if len(variants) > 2 else 7, loc="lower right")
    figure.tight_layout(w_pad=1.0)
    figure.savefig(path)
    plt.close(figure)


def steady_figure(analyses: list[dict], tolerance: float, path: pathlib.Path) -> None:
    """(a) fluid kinetic energy against time for every run; (b) the runner's window test: relative change of the
    window-mean profile (56 reference points) and of the window-mean kinetic energy between consecutive windows."""
    import matplotlib.pyplot as plt
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ordered = sorted(analyses, key=lambda entry: (entry["slabs"] == 1, entry["variant"] != "float32", entry["resolution"]))
    for item in ordered:
        style = variant_style(item["variant"])[0]
        color = CONTROL_COLOR if item["slabs"] == 1 else SHADES[item["resolution"]]
        label = None
        times = np.array([row["time"] for row in item["rows"]])
        energy = np.array([row["kinetic_energy"] for row in item["rows"]])
        axes[0].plot(times, energy, style, color=color, lw=0.8, label=label)
        windows_path = LOGS / item["id"] / "windows.jsonl"
        if windows_path.exists():
            windows = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines() if line]
            window_times = [window["time"] for window in windows]
            axes[1].semilogy(window_times, [window["profile_change"] for window in windows], style, marker="o", ms=2.8,
                             mfc=color, mec=color, mew=0.6, color=color, lw=0.8)
            axes[1].semilogy(window_times, [window["kinetic_energy_change"] for window in windows], style, marker="o",
                             ms=2.8, mfc="none", mec=color, mew=0.6, color=color, lw=0.5)
    start = min(item["window"][0] for item in analyses)
    end = max(item["window"][1] for item in analyses)
    axes[0].axvspan(start, end, color="#000000", alpha=0.06, lw=0)
    axes[0].text(0.5 * (start + end), 0.06, "time average", transform=axes[0].get_xaxis_transform(), fontsize=7,
                 color="#555555", ha="center", va="bottom")
    axes[1].axhline(tolerance, color="#777777", lw=0.6, ls=":")
    axes[1].text(2.0, tolerance * 1.25, f"{tolerance:g}", fontsize=7, color="#555555", va="bottom", ha="left")
    axes[1].plot([], [], "o", ms=2.8, mfc="#555555", mec="#555555", label="profile")
    axes[1].plot([], [], "o", ms=2.8, mfc="none", mec="#555555", label="kinetic energy")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\frac{1}{2}\sum_{\mathrm{fluid}} m\,|\mathbf{v}|^2$")
    axes[1].set_xlabel(r"window end $t$")
    axes[1].set_ylabel("relative change between windows")
    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.set_xlim(0, None)
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    from matplotlib.lines import Line2D
    resolutions = sorted({item["resolution"] for item in analyses if item["slabs"] == 2})
    colour_handles = [Line2D([], [], color=SHADES[value], lw=1.0, label=rf"$\Delta x = 1/{value}$") for value in resolutions]
    if any(item["slabs"] == 1 for item in analyses):
        colour_handles.append(Line2D([], [], color=CONTROL_COLOR, lw=1.0, label=r"$K = 1$, $\Delta x = 1/1000$"))
    present = [variant for variant in VARIANT_STYLES if any(item["variant"] == variant for item in analyses)]
    style_handles = [Line2D([], [], color="#555555", lw=0.9, ls=variant_style(variant)[0], label=variant_style(variant)[2])
                     for variant in present]
    first = axes[0].legend(handles=colour_handles, frameon=False, fontsize=6.5, loc="center right",
                           bbox_to_anchor=(1.0, 0.55))
    axes[0].add_artist(first)
    if len(style_handles) > 1:
        axes[0].legend(handles=style_handles, frameon=False, fontsize=6.0, loc="center right", bbox_to_anchor=(1.0, 0.25))
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout(w_pad=1.2)
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--no-stream-function", action="store_true")
    parser.add_argument("--runs", default=None)
    parser.add_argument("--from-cache", action="store_true",
                        help="redo only the figures and tables from the analyses cached by the last full run")
    arguments = parser.parse_args()
    run_ids = arguments.runs.split(",") if arguments.runs else sorted(path.name for path in LOGS.iterdir()
                                                                       if (path / "result.json").exists())
    analyses = list(np.load(LOGS / "analysis_cache.npy", allow_pickle=True)) if arguments.from_cache else []
    for run_id in [] if arguments.from_cache else run_ids:
        run = load_run(run_id)
        if run is None:
            continue
        analysis = analyse_run(run, arguments.average_span, arguments.steady_tol, arguments.window,
                               not arguments.no_stream_function)
        analysis["dense_reference"] = run["points"]["dense_reference"]
        analysis["segments"] = run["segments"]
        analysis["rows"] = run["rows"]
        analyses.append(analysis)
        profile_csv(analysis)
        print(f"{run_id}: window {analysis['window'][0]:.2f}-{analysis['window'][1]:.2f} ({analysis['samples_in_window']} samples), "
              f"steady online {analysis['t_steady_online']} sliding {analysis['t_steady_sliding']}; "
              f"u L2 {analysis['errors'][('wall', 'mls', 'u')]['L2']:.4f} v L2 {analysis['errors'][('wall', 'mls', 'v')]['L2']:.4f}",
              flush=True)
    if not analyses:
        print("no completed runs")
        return 1
    if not arguments.from_cache:
        np.save(LOGS / "analysis_cache.npy", np.array(analyses, dtype=object), allow_pickle=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for variant in sorted({item["variant"] for item in analyses}):
        if len({item["resolution"] for item in analyses if item["variant"] == variant and item["slabs"] == 2}) >= 2:
            profile_figure(analyses, variant, FIGURES / f"cavity_re1000_profiles_{variant.replace('.', 'p')}.pdf")
    present = [variant for variant in SETTINGS if any(item["variant"] == variant and item["slabs"] == 2 for item in analyses)]
    convergence_figure(analyses, FIGURES / "cavity_re1000_convergence.pdf",
                       tuple(present) if len(present) > 1 else ("float32", "delta"))
    error_figure(analyses, FIGURES / "cavity_re1000_errors_release.pdf",
                 [{"variant": "float32", "style": "-"}, {"variant": "delta", "style": "--", "suffix": r", $\delta\rho$"}])
    if MAIN_VARIANT in present:
        error_figure(analyses, FIGURES / "cavity_re1000_errors.pdf", [{"variant": MAIN_VARIANT, "style": "-"}])
    if len(present) > 1:
        common = set.intersection(*({item["resolution"] for item in analyses if item["variant"] == variant
                                     and item["slabs"] == 2} for variant in present))
        if common:
            top = max(common)
            error_figure(analyses, FIGURES / "cavity_re1000_errors_settings.pdf",
                         [{"variant": variant, "style": variant_style(variant)[0], "resolutions": {top},
                           "color": SETTING_COLORS[variant], "label": variant_style(variant)[2]} for variant in present])
            settings_profile_figure(analyses, FIGURES / "cavity_re1000_profiles_settings.pdf", present, top)
    steady_figure(analyses, arguments.steady_tol, FIGURES / "cavity_re1000_steady.pdf")
    from experiment.validation import cavity_tables
    cavity_tables.write_all(analyses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
