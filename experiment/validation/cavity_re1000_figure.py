"""cavity_re1000_figure.py - paper section 6.2: the two Re = 1000 lid-driven cavity figures (split 2026-10-21) from the
finished runs (v7-rc3, adami walls, 1000^2): K = 1 (one RTX 5090), K = 2 (5090 + 7900 XTX, E40) and K = 8 (one whole N56
node, 8 x RTX 5090, SLURM job 1719093, continued from the K = 1 run's t = 89.3025 checkpoint). Uses existing data only:
nothing is run. EVERY time average is taken over the common window of the three runs, t = 89.505 .. 99.8325: the K = 8
run's 52 samples (every 0.2 time units) and 11 snapshots (every ~1 time unit); K = 1 / K = 2 restricted to the same steps.

Outputs (docs/validation/figures/, vector PDF + PNG preview; width = elsarticle single-column text width, 384 pt):
  cavity_re1000.pdf   three panels in a row:
    (a) u(0.5, y) and (b) v(x, 0.5): the time-averaged MLS centre lines of K = 1 (solid) against Marchi et al. 2021
        (open circles) and Ghia et al. 1982 (open squares), with zooms near u_min and v_min; the legend is in (a) only;
    (c) K = 8: time-averaged speed |v|/U (colour 0 - 1) with streamlines and the 7 seams (thin white dashed lines).
    The previous version (K = 1 + K = 2 lines, K = 2 field, each run's last 20 time units) is kept as cavity_re1000_v1.pdf.
  cavity_seams.pdf    two panels, the same width:
    (a) (K = 2 - K = 1) / sigma (open squares) and (K = 8 - K = 1) / sigma (filled circles) at the 56 Marchi 2021 points,
        the u line (28 points by y) first and then the v line (28 by x), separated by a thin vertical line, +-2 dashed;
        sigma = sqrt(sem_1^2 + sem_2^2) of the two time averages (sem = std * sqrt(tau_int / n) over the 52 samples);
    (b) rms |delta v| of K = 8 - K = 1 (time-averaged MLS velocity of the 11 snapshots, strips of 3 h on either side of
        every seam, 0.5 h columns, 397 points per column) against the distance to the nearest seam in support radii h
        (the two sides of a seam folded together), one thin line per seam; the min - max range of the 6 control strips
        (half way between neighbouring seams) as a grey band.
  cavity_re1000_k1_field.pdf   panel (c) for K = 1 on the same window (same size and colour scale, no seams, no label).

Data:
  * centre lines of (a) / (b): the K = 1 run's samples (logs/validation/cavity_re1000_v7/<run>/samples, dense MLS
    centre lines of 1001 points, wall-row frame) restricted to the K = 8 run's sample steps and averaged; written to
    docs/validation/data/dense_<K1 run>_common_window.csv;
  * the field of (c): the 11 snapshots of the window (the same steps in every run) interpolated with the linear MLS of
    cavity_sampling.interpolate (the method of the psi_min analysis in cavity_analysis.stream_function_minimum, there on
    513^2 and u only) onto a node grid that spans the innermost wall rows, then averaged; cached as
    mean_field_<run>_<points>.npz next to the runs;
  * streamlines: contours of the stream function of the averaged field, solved from its vorticity with psi = 0 on the
    effective walls (cavity_fields.stream_function_from_vorticity; the integral of u dy does not close because the
    stored velocities carry a net flux of ~3e-4 U, which would distort the small corner-eddy levels);
  * cavity_seams.pdf: docs/validation/data/adami_k8_point_differences.csv and adami_k8_strip_bins.csv
    (experiment/validation/adami_k8_report.py, docs/validation/adami_k8.md);
  * references: cavity_reference.marchi2021() (Tc, Tables 19-21) and ghia1982().

Coordinates: the wall-row frame of docs/validation/cavity_re1000.md (unit square between the innermost wall / lid rows,
X = (x + 0.5 + dx) / (1 + 2 dx)); velocities in U, psi in U L.

    .venv/Scripts/python.exe -m experiment.validation.cavity_re1000_figure [--grid-points 1025] [--workers 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
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
from experiment.validation.cavity_analysis import load_run, write_csv  # noqa: E402
from experiment.validation.cavity_fields import UniformGrid, read_case_information, stream_function_from_vorticity  # noqa: E402

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re1000_v7"
DATA = _REPO_ROOT / "docs" / "validation" / "data"
FIGURES = _REPO_ROOT / "docs" / "validation" / "figures"
RUN_IDS = {"K1": "n1000_k1_float32_xi0p001_eps0p0025_adami", "K8": "n1000_k8_float32_xi0p001_eps0p0025_adami"}
POINT_DIFFERENCES_CSV = DATA / "adami_k8_point_differences.csv"
STRIP_BINS_CSV = DATA / "adami_k8_strip_bins.csv"
SEAM_COUNT = 7
CONTROL_COUNT = 6
ELSARTICLE_SINGLE_COLUMN_WIDTH = 384.0 / 72.0   # preprint and 1p text width, 384 pt (elsdoc: 5.3 in)
CHUNK_POINTS = 40000

# figure style (the serif style of the existing validation figures, cavity_analysis.setup_style)
FONT_SIZE = 8.0
INSET_TICK_SIZE = 8.0                     # zoom tick labels (2026-10-15: was 6 pt)
FIELD_YLABEL_PAD = 4.5                    # (c) y label to its tick labels, points (was 1.5)
COLOR_K1 = "#08519c"
COLOR_K2 = "#6baed6"
COLOR_GHIA = "#b2182b"
COLOR_CONTROL_BAND = "#c6c6c6"
COLOR_GUIDE = "#777777"
STREAMLINE_COLOR = "#f0f0f0"
# stream-function levels: the primary vortex (psi < 0, psi_min = -0.119) and the two bottom corner eddies (psi > 0;
# BR1 max ~ 1.7e-3, BL1 max ~ 2.4e-4 in these runs)
PSI_LEVELS_NEGATIVE = (-0.115, -0.10, -0.08, -0.06, -0.04, -0.02, -0.005)
PSI_LEVELS_POSITIVE = (1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 1.5e-3)


# --------------------------------------------------------------------------------------------- data
def run_facts(run_id: str) -> dict:
    directory = RUNS_ROOT / run_id
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    steps, times = [], []
    with (directory / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                steps.append(int(row["step"]))
                times.append(float(row["time"]))
    information = read_case_information(meta)
    spacing = float(meta["spacing"])
    support = float(meta["support_radius"])
    length = 1.0 + 2.0 * spacing
    seams = [((information["grid_origin_x"] + int(column) * support) + 0.5 + spacing) / length
             for column in (meta.get("cuts") or [])]
    return {"id": run_id, "dir": directory, "meta": meta, "spacing": spacing, "support": support,
            "dt": float(meta["dt"]), "sample_steps": steps, "sample_times": times,
            "window": (min(times), max(times)), "seams_wall": seams}


def common_window(k8: dict) -> tuple[float, float]:
    """The K = 8 continuation's samples: t = 89.505 .. 99.8325 (steps 11,934,000 .. 13,311,000)."""
    return (min(k8["sample_times"]), max(k8["sample_times"]))


def k1_lines_on_window(k1: dict, k8: dict) -> dict:
    """Time-averaged dense MLS centre lines of the K = 1 run on the K = 8 run's sample steps (wall-row frame);
    written to docs/validation/data/dense_<run>_common_window.csv."""
    run = load_run(k1["id"], RUNS_ROOT)
    if run is None:
        raise FileNotFoundError(k1["dir"] / "samples.jsonl")
    wanted = set(k8["sample_steps"])
    keep = np.array([int(row["step"]) in wanted for row in run["rows"]], dtype=bool)
    if int(keep.sum()) != len(wanted):
        raise RuntimeError(f"{k1['id']}: {int(keep.sum())} of the {len(wanted)} K = 8 sample steps found")
    coordinate = run["points"]["dense_reference"]
    u = run["arrays"]["u_dense_mls"][keep].mean(axis=0)
    v = run["arrays"]["v_dense_mls"][keep].mean(axis=0)
    times = run["times"][keep]
    window = (float(times.min()), float(times.max()))
    path = DATA / f"dense_{k1['id']}_common_window.csv"
    write_csv(path, ["coordinate", "u_at_x_0.5", "v_at_y_0.5"],
              [[f"{coordinate[index]:.4f}", f"{u[index]:.7f}", f"{v[index]:.7f}"] for index in range(coordinate.size)],
              f"Run {k1['id']}: time-averaged MLS velocity on the dense centre lines (wall-row frame) over the common\n"
              f"window of the K = 1 / K = 2 / K = 8 comparison, t in [{window[0]:.4f}, {window[1]:.4f}] ({int(keep.sum())}\n"
              f"samples, the K = 8 continuation's sample steps {min(wanted):,} .. {max(wanted):,}); cavity_re1000_figure.py.")
    print(f"[figure] K1 lines: {int(keep.sum())} samples t = {window[0]:.4f} .. {window[1]:.4f} -> {path.name}", flush=True)
    return {"coordinate": coordinate, "u": u, "v": v, "window": window, "samples": int(keep.sum()),
            "source": str(path.relative_to(_REPO_ROOT)).replace("\\", "/")}


def window_snapshots(run: dict) -> list[pathlib.Path]:
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    low, high = run["window"]
    return [path for path in files if low - 1e-9 <= int(path.stem[1:]) * run["dt"] <= high + 1e-9]


def _snapshot_mls(arguments: tuple) -> tuple[str, np.ndarray, int]:
    """Worker: linear-MLS velocity of one snapshot at the grid nodes (chunked)."""
    path, axis, support = arguments
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    del xx, yy
    with np.load(path) as archive:
        particles = sampling.ParticleSet(archive["positions"].astype(np.float64),
                                         archive["velocities"].astype(np.float64),
                                         archive["volumes"].astype(np.float64))
    tree = sampling.cKDTree(particles.positions)
    velocity = np.empty((points.shape[0], 2))
    fallbacks = 0
    for start in range(0, points.shape[0], CHUNK_POINTS):
        chunk = slice(start, start + CHUNK_POINTS)
        result = sampling.interpolate(particles, points[chunk], support, tree=tree)
        velocity[chunk] = result["mls"]
        fallbacks += int(result["mls_fallback"].sum())
    return str(path), velocity.reshape(axis.size, axis.size, 2), fallbacks


def mean_field(run: dict, grid_points: int, workers: int, refresh: bool) -> dict:
    """Time-averaged MLS velocity on a node grid spanning the innermost wall rows (physical +-(0.5 + dx)) over the
    snapshots of run["window"]."""
    cache = RUNS_ROOT / f"mean_field_{run['id']}_{grid_points}.npz"
    files = window_snapshots(run)
    if not files:
        raise RuntimeError(f"{run['id']}: no snapshots in t = {run['window'][0]:.4f} .. {run['window'][1]:.4f}")
    if cache.exists() and not refresh:
        with np.load(cache) as archive:
            stored = {name: archive[name] for name in archive.files}
        if [pathlib.Path(name).name for name in stored["snapshots"]] == [path.name for path in files]:
            print(f"[figure] {run['id']}: mean field from {cache.name}", flush=True)
            return stored
    half = 0.5 + run["spacing"]
    axis = np.linspace(-half, half, grid_points)
    total = np.zeros((grid_points, grid_points, 2))
    fallbacks = 0
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (name, velocity, count) in enumerate(pool.map(_snapshot_mls, [(path, axis, run["support"])
                                                                               for path in files]), start=1):
            total += velocity
            fallbacks += count
            print(f"[figure] {run['id']}: {done}/{len(files)} {pathlib.Path(name).name} "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    mean = total / len(files)
    stored = {"axis_physical": axis, "u": mean[:, :, 0], "v": mean[:, :, 1],
              "snapshots": np.array([path.name for path in files]),
              "times": np.array([int(path.stem[1:]) * run["dt"] for path in files]),
              "mls_fallbacks": np.array(fallbacks), "spacing": np.array(run["spacing"])}
    np.savez_compressed(cache, **stored)
    print(f"[figure] {run['id']}: wrote {cache.name} ({time.perf_counter() - started:.0f} s)", flush=True)
    return stored


def field_quantities(field: dict) -> dict:
    """Speed, and the stream function of the averaged field (Poisson, psi = 0 on the effective walls), in the
    wall-row frame (lengths / (1 + 2 dx))."""
    axis = field["axis_physical"]
    spacing = float(field["spacing"])
    node_spacing = float(axis[1] - axis[0])
    u, v = field["u"], field["v"]
    vorticity = np.gradient(v, node_spacing, axis=1) - np.gradient(u, node_spacing, axis=0)
    grid = UniformGrid(float(axis[0]) - 0.5 * node_spacing, float(axis[0]) - 0.5 * node_spacing, node_spacing,
                       axis.size, axis.size)
    psi_physical = stream_function_from_vorticity(vorticity, grid, 0.5 + 0.5 * spacing)
    length = 1.0 + 2.0 * spacing
    wall_axis = (axis - axis[0]) / (axis[-1] - axis[0])
    speed = np.hypot(u, v)
    return {"axis": wall_axis, "speed": speed, "psi": psi_physical / length, "vorticity": vorticity * length,
            "speed_max": float(np.nanmax(speed)), "psi_min": float(np.nanmin(psi_physical / length)),
            "psi_max": float(np.nanmax(psi_physical / length))}


def _csv_rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(line for line in path.read_text(encoding="utf-8").splitlines()
                               if line and not line.startswith("#")))


def point_differences() -> dict[str, list[dict]]:
    """adami_k8_point_differences.csv by pair: 56 rows each, the u line (28 points by y) then the v line (28 by x)."""
    pairs: dict[str, list[dict]] = {}
    for row in _csv_rows(POINT_DIFFERENCES_CSV):
        pairs.setdefault(row["pair"], []).append(row)
    for pair, rows in pairs.items():
        if [row["line"] for row in rows] != ["u"] * 28 + ["v"] * 28:
            raise RuntimeError(f"{POINT_DIFFERENCES_CSV.name}: pair {pair} is not 28 u points followed by 28 v points")
    return pairs


def folded_strip_bins() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """adami_k8_strip_bins.csv (rms |delta v| per 0.5 h column at signed offsets -2.75 .. +2.75 h from the strip centre)
    folded to the distance from the strip centre: rms over both columns at the same |offset|, weighted by their points."""
    totals: dict[str, dict[float, list[float]]] = {}
    for row in _csv_rows(STRIP_BINS_CSV):
        distance = round(abs(float(row["offset_h"])), 6)
        count = int(row["points"])
        bucket = totals.setdefault(row["strip"], {}).setdefault(distance, [0.0, 0.0])
        bucket[0] += count
        bucket[1] += count * float(row["rms_abs_du"]) ** 2
    folded = {}
    for strip, buckets in totals.items():
        distances = sorted(buckets)
        folded[strip] = (np.array(distances),
                         np.array([math.sqrt(buckets[distance][1] / buckets[distance][0]) for distance in distances]))
    expected = [f"seam{index}" for index in range(1, SEAM_COUNT + 1)] + [f"control{index}" for index in range(1, CONTROL_COUNT + 1)]
    missing = [name for name in expected if name not in folded]
    if missing:
        raise RuntimeError(f"{STRIP_BINS_CSV.name}: strips missing: {missing}")
    return folded


# --------------------------------------------------------------------------------------------- drawing
def setup_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1.5, "xtick.labelsize": FONT_SIZE - 0.5, "ytick.labelsize": FONT_SIZE - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.linewidth": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": None, "contour.negative_linestyle": "solid",
    })


def axes_inches(figure, left: float, bottom: float, width: float, height: float):
    figure_width, figure_height = figure.get_size_inches()
    return figure.add_axes([left / figure_width, bottom / figure_height, width / figure_width, height / figure_height])


def panel_label(axis, text: str) -> None:
    # above the top-left corner of the axes, left of the frame: clear of the data, the ticks and the tick labels
    axis.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-2.0, 5.0), textcoords="offset points",
                  ha="right", va="bottom", fontsize=FONT_SIZE + 0.5)


def style_inset(inset) -> None:
    inset.tick_params(which="major", labelsize=INSET_TICK_SIZE, length=2.0, width=0.5, pad=1.5)
    inset.tick_params(which="minor", length=1.3, width=0.4)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)


def hide_top_right(axis) -> None:
    axis.tick_params(which="both", top=False, right=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_profiles(axis_u, axis_v, line: dict, reference: dict, ghia: dict, legend: bool) -> dict:
    """(a) u(0.5, y) and (b) v(x, 0.5): the K = 1 line (solid) against Marchi 2021 (circles) and Ghia 1982 (squares)."""
    from matplotlib.ticker import FixedLocator, MultipleLocator
    k1_style = dict(color=COLOR_K1, lw=1.0, ls="-", label=r"SPH, $K = 1$")
    marchi_style = dict(ls="none", marker="o", mfc="none", mec="k", ms=3.0, mew=0.6, label="Marchi et al. (2021)")
    ghia_style = dict(ls="none", marker="s", mfc="none", mec=COLOR_GHIA, ms=2.6, mew=0.55, label="Ghia et al. (1982)")
    axis_u.plot(line["u"], line["coordinate"], **k1_style)
    axis_v.plot(line["coordinate"], line["v"], **k1_style)
    axis_u.plot(reference["u"][1], reference["u"][0], **marchi_style)
    axis_u.plot(ghia["u"][1], ghia["u"][0], **ghia_style)
    axis_v.plot(reference["v"][0], reference["v"][1], **marchi_style)
    axis_v.plot(ghia["v"][0], ghia["v"][1], **ghia_style)

    axis_u.set_xlim(-0.46, 1.04)
    axis_u.set_ylim(0.0, 1.0)
    axis_u.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_u.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_u.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_u.set_xlabel(r"$u(0.5, y)$", labelpad=1.5)
    axis_u.set_ylabel(r"$y$", labelpad=1.5)
    axis_v.set_xlim(0.0, 1.0)
    axis_v.set_ylim(-0.62, 0.46)
    axis_v.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_v.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis_v.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_v.set_xlabel(r"$x$", labelpad=1.5)
    axis_v.set_ylabel(r"$v(x, 0.5)$", labelpad=1.0)
    for axis in (axis_u, axis_v):
        hide_top_right(axis)

    # zooms: u_min (lower right of (a)), v_min (lower left of (b)); ticks inside the main axes, clear of its ticks
    inset_u = axis_u.inset_axes([0.63, 0.54, 0.33, 0.27])
    inset_v = axis_v.inset_axes([0.27, 0.15, 0.21, 0.34])
    inset_u.plot(line["u"], line["coordinate"], **{**k1_style, "lw": 0.9})
    inset_v.plot(line["coordinate"], line["v"], **{**k1_style, "lw": 0.9})
    inset_u.plot(reference["u"][1], reference["u"][0], **{**marchi_style, "ms": 2.8})
    inset_u.plot(ghia["u"][1], ghia["u"][0], **{**ghia_style, "ms": 2.4})
    inset_v.plot(reference["v"][0], reference["v"][1], **{**marchi_style, "ms": 2.8})
    inset_v.plot(ghia["v"][0], ghia["v"][1], **{**ghia_style, "ms": 2.4})
    inset_u.set_xlim(-0.405, -0.29)
    inset_u.set_ylim(0.08, 0.30)
    inset_u.xaxis.set_major_locator(MultipleLocator(0.1))
    inset_u.yaxis.set_major_locator(FixedLocator([0.1, 0.2]))   # no 0.3 label: it would touch the curve at y = 0.75
    inset_v.set_xlim(0.84, 0.96)
    inset_v.set_ylim(-0.545, -0.375)
    # one labelled tick at 8 pt: two ("0.85", "0.95") would touch each other
    inset_v.xaxis.set_major_locator(FixedLocator([0.90]))
    inset_v.xaxis.set_minor_locator(FixedLocator([0.85, 0.95]))
    inset_v.yaxis.set_major_locator(MultipleLocator(0.1))
    for inset in (inset_u, inset_v):
        style_inset(inset)
    if legend:
        # lower right of (a), lifted off the bottom so that its left edge clears the curve below y = 0.1
        axis_u.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), frameon=False, handlelength=1.6,
                      handletextpad=0.45, borderaxespad=0.0, borderpad=0.0, labelspacing=0.3)
    return {"inset_u": inset_u, "inset_v": inset_v}


def draw_field(axis, colorbar_axis, quantities: dict, seams_wall: list[float]) -> dict:
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator
    wall_axis = quantities["axis"]
    image = axis.imshow(quantities["speed"], origin="lower", extent=[0.0, 1.0, 0.0, 1.0], cmap="viridis",
                        norm=Normalize(0.0, 1.0), interpolation="bilinear", aspect="equal")
    levels = sorted(PSI_LEVELS_NEGATIVE + PSI_LEVELS_POSITIVE)
    axis.contour(wall_axis, wall_axis, quantities["psi"], levels=levels, colors=STREAMLINE_COLOR, linewidths=0.35,
                 linestyles="solid", alpha=0.9)
    for seam in seams_wall:
        axis.axvline(seam, color="white", lw=0.6, ls=(0, (3.0, 2.0)))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.xaxis.set_major_locator(MultipleLocator(0.5))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.xaxis.set_minor_locator(MultipleLocator(0.25))
    axis.yaxis.set_minor_locator(MultipleLocator(0.25))
    axis.tick_params(which="both", direction="out", top=False, right=False)
    axis.set_xlabel(r"$x$", labelpad=1.5)
    axis.set_ylabel(r"$y$", labelpad=FIELD_YLABEL_PAD)
    colorbar = axis.figure.colorbar(image, cax=colorbar_axis, ticks=np.linspace(0.0, 1.0, 6))
    colorbar.outline.set_linewidth(0.5)
    colorbar_axis.tick_params(direction="out", length=2.0, width=0.5, labelsize=FONT_SIZE - 0.5, pad=1.5)
    colorbar.set_label(r"$|\boldsymbol{v}|/U$", labelpad=2.0)   # bold italic v: the velocity vector of the text
    return {"levels": levels}


# geometry of the row (inches): left margin, panel widths and gaps; the field panel is square
# 2026-10-15: gap_bc 0.37 -> 0.42 for the larger (c) y-label pad, taken from the two profile panels (1.20 -> 1.175),
# so that (c) keeps its size and the figure its height
LAYOUT = {"left": 0.37, "profile_width": 1.175, "gap_ab": 0.44, "gap_bc": 0.42, "colorbar_pad": 0.06,
          "colorbar_width": 0.075, "bottom": 0.34, "top_margin": 0.21}
# cavity_seams.pdf: two equal panels
SEAMS_LAYOUT = {"left": 0.50, "gap": 0.62, "right": 0.10, "bottom": 0.37, "height": 1.52, "top_margin": 0.20}


def _label_boxes(axis, renderer) -> list:
    """Tick labels of ``axis`` that are drawn: the locator also returns ticks outside the view limits."""
    frame = axis.get_window_extent(renderer)
    boxes = []
    for direction, texts in (("x", axis.get_xticklabels()), ("y", axis.get_yticklabels())):
        for text in texts:
            if not (text.get_visible() and text.get_text()):
                continue
            box = text.get_window_extent(renderer)
            centre = 0.5 * (box.x0 + box.x1) if direction == "x" else 0.5 * (box.y0 + box.y1)
            low, high = (frame.x0, frame.x1) if direction == "x" else (frame.y0, frame.y1)
            if low - 1.0 <= centre <= high + 1.0:
                boxes.append((text.get_text(), box))
    return boxes


def _drawn_points(axis, renderer) -> list:
    """Display-space points of every line of ``axis`` inside its frame (segments densified to 1 px), with the
    half-width of what is drawn there (marker radius or half the line width), in pixels."""
    dpi = axis.figure.dpi
    frame = axis.get_window_extent(renderer)
    result = []
    for line in axis.get_lines():
        data = line.get_xydata()
        if data.size == 0 or not line.get_visible():
            continue
        display = line.get_transform().transform(data)
        has_marker = line.get_marker() not in (None, "None", "", " ")
        if line.get_linestyle() not in ("None", "", " ") and len(display) > 1:
            pieces = [display[:1]]
            for start, end in zip(display[:-1], display[1:]):
                steps = max(int(np.ceil(np.hypot(*(end - start)))), 1)
                pieces.append(start + (end - start) * (np.arange(1, steps + 1)[:, None] / steps))
            display = np.concatenate(pieces)
        inside = ((display[:, 0] >= frame.x0 - 1) & (display[:, 0] <= frame.x1 + 1)
                  & (display[:, 1] >= frame.y0 - 1) & (display[:, 1] <= frame.y1 + 1))
        radius = 0.5 * (line.get_markersize() if has_marker else line.get_linewidth()) * dpi / 72.0
        result.append((display[inside], radius))
    return result


def _clearance_points(box, points: list, dpi: float) -> float:
    """Smallest gap (pt) between a box and drawn points (negative = overlap)."""
    best = np.inf
    for display, radius in points:
        if display.size == 0:
            continue
        dx = np.maximum(np.maximum(box.x0 - display[:, 0], display[:, 0] - box.x1), 0.0)
        dy = np.maximum(np.maximum(box.y0 - display[:, 1], display[:, 1] - box.y1), 0.0)
        best = min(best, float((np.hypot(dx, dy) - radius).min()))
    return best * 72.0 / dpi


def _clearance_boxes(box, others: list, dpi: float) -> float:
    """Smallest gap (pt) between a box and other boxes (negative = overlap depth)."""
    best = np.inf
    for other in others:
        dx = max(other.x0 - box.x1, box.x0 - other.x1, 0.0)
        dy = max(other.y0 - box.y1, box.y0 - other.y1, 0.0)
        if dx > 0 or dy > 0:
            gap = float(np.hypot(dx, dy))
        else:
            gap = -min(box.x1 - other.x0, other.x1 - box.x0, box.y1 - other.y0, other.y1 - box.y0)
        best = min(best, gap)
    return best * 72.0 / dpi


def _report_clearances(report: list, minimum_pt: float) -> None:
    failures = []
    for item, gaps in report:
        print(f"[check] {item}: " + ", ".join(f"{key} {value:+.1f} pt" for key, value in gaps.items()), flush=True)
        failures += [f"{item} vs {key} ({value:+.1f} pt)" for key, value in gaps.items() if value < minimum_pt]
    if failures:
        raise RuntimeError("overlaps: " + "; ".join(failures))


def check_clearances(figure, axis_u, axis_v, axis_field, insets: dict, minimum_pt: float = 0.5) -> None:
    """Measured on the rendered figure: the zoom tick labels against the main curves / reference markers, the main
    ticks and tick labels and the legend; each zoom frame against the main curves; the (b) and (c) y labels against
    everything of the panel to their left. Raises when anything overlaps (gap < minimum_pt)."""
    figure.canvas.draw()                  # tick-label and axis-label positions are set at draw time
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    report = []
    for name, axis, inset in (("(a) zoom", axis_u, insets["inset_u"]), ("(b) zoom", axis_v, insets["inset_v"])):
        points = _drawn_points(axis, renderer)
        main_ticks = []
        for tick in (axis.xaxis.get_major_ticks() + axis.xaxis.get_minor_ticks()
                     + axis.yaxis.get_major_ticks() + axis.yaxis.get_minor_ticks()):
            if tick.get_visible() and tick.tick1line.get_visible():
                main_ticks.append(tick.tick1line.get_window_extent(renderer))
        main_labels = [box for _, box in _label_boxes(axis, renderer)]
        legend = axis.get_legend()
        legend_boxes = [legend.get_window_extent(renderer)] if legend is not None else []
        for text, box in _label_boxes(inset, renderer):
            gaps = {"curves/markers": _clearance_points(box, points, dpi),
                    "main ticks": _clearance_boxes(box, main_ticks, dpi),
                    "main tick labels": _clearance_boxes(box, main_labels, dpi)}
            if legend_boxes:
                gaps["legend"] = _clearance_boxes(box, legend_boxes, dpi)
            frame = axis.get_window_extent(renderer)
            gaps["main axis lines"] = min(box.x0 - frame.x0, box.y0 - frame.y0) * 72.0 / dpi
            report.append((f"{name} tick label {text!r} ({box.height * 72.0 / dpi:.1f} pt tall)", gaps))
        report.append((f"{name} frame", {"curves/markers": _clearance_points(inset.get_window_extent(renderer),
                                                                             points, dpi)}))
    label_box = axis_field.yaxis.label.get_window_extent(renderer)
    field_ticks = [box for text, box in _label_boxes(axis_field, renderer)
                   if box.x1 <= axis_field.get_window_extent(renderer).x0]   # the y tick labels
    panel_b_texts = [box for _, box in _label_boxes(axis_v, renderer)] + [axis_v.xaxis.label.get_window_extent(renderer)]
    label_b = axis_v.yaxis.label.get_window_extent(renderer)
    panel_a_texts = [box for _, box in _label_boxes(axis_u, renderer)] + [box for _, box in _label_boxes(insets["inset_u"], renderer)]
    report.append(("(b) y label", {"(a) curves/markers": _clearance_points(label_b, _drawn_points(axis_u, renderer), dpi),
                                   "(a) tick labels (incl. zoom)": _clearance_boxes(label_b, panel_a_texts, dpi)}))
    report.append(("(c) y label", {"(b) curves/markers": _clearance_points(label_box, _drawn_points(axis_v, renderer), dpi),
                                   "(b) tick and axis labels": _clearance_boxes(label_box, panel_b_texts, dpi),
                                   "(c) tick labels": _clearance_boxes(label_box, field_ticks, dpi)}))
    _report_clearances(report, minimum_pt)


def composite_figure(line: dict, field_k8: dict, seams_wall: list[float], width: float, path: pathlib.Path) -> dict:
    import matplotlib.pyplot as plt
    layout = LAYOUT
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()
    right_reserve = 0.33                       # colourbar tick labels and label
    field_size = (width - layout["left"] - 2 * layout["profile_width"] - layout["gap_ab"] - layout["gap_bc"]
                  - layout["colorbar_pad"] - layout["colorbar_width"] - right_reserve)
    height = layout["bottom"] + field_size + layout["top_margin"]
    figure = plt.figure(figsize=(width, height))
    x = layout["left"]
    axis_u = axes_inches(figure, x, layout["bottom"], layout["profile_width"], field_size)
    x += layout["profile_width"] + layout["gap_ab"]
    axis_v = axes_inches(figure, x, layout["bottom"], layout["profile_width"], field_size)
    x += layout["profile_width"] + layout["gap_bc"]
    axis_field = axes_inches(figure, x, layout["bottom"], field_size, field_size)
    x += field_size + layout["colorbar_pad"]
    colorbar_axis = axes_inches(figure, x, layout["bottom"], layout["colorbar_width"], field_size)
    insets = draw_profiles(axis_u, axis_v, line, reference, ghia, legend=True)
    draw_field(axis_field, colorbar_axis, field_k8, seams_wall)
    for axis, text in zip((axis_u, axis_v, axis_field), ("(a)", "(b)", "(c)")):
        panel_label(axis, text)
    check_clearances(figure, axis_u, axis_v, axis_field, insets)
    save(figure, path)
    plt.close(figure)
    return {"width_in": width, "height_in": height, "field_size_in": field_size}


def field_only_figure(field: dict, size: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    layout = LAYOUT
    left = layout["gap_bc"] + 0.08
    width = left + size["field_size_in"] + layout["colorbar_pad"] + layout["colorbar_width"] + 0.33
    figure = plt.figure(figsize=(width, size["height_in"]))
    axis = axes_inches(figure, left, layout["bottom"], size["field_size_in"], size["field_size_in"])
    colorbar_axis = axes_inches(figure, left + size["field_size_in"] + layout["colorbar_pad"], layout["bottom"],
                                layout["colorbar_width"], size["field_size_in"])
    draw_field(axis, colorbar_axis, field, [])
    save(figure, path)
    plt.close(figure)


def draw_point_differences(axis, pairs: dict[str, list[dict]]) -> dict:
    """(a) of cavity_seams.pdf: delta / sigma at the 56 points for K = 2 - K = 1 (open squares) and K = 8 - K = 1
    (filled circles); the u line first, the v line after a thin vertical line; +-2 dashed."""
    from matplotlib.ticker import MultipleLocator
    styles = {"K2-K1": dict(ls="none", marker="s", mfc="none", mec=COLOR_K2, ms=3.0, mew=0.75, label=r"$K = 2 - K = 1$"),
              "K8-K1": dict(ls="none", marker="o", mfc=COLOR_K1, mec=COLOR_K1, ms=2.5, mew=0.5, label=r"$K = 8 - K = 1$")}
    index = np.arange(1, 57)
    largest = 0.0
    for pair in ("K2-K1", "K8-K1"):
        values = np.array([float(row["difference_over_sigma"]) for row in pairs[pair]])
        largest = max(largest, float(np.abs(values).max()))
        axis.plot(index, values, **styles[pair])
    axis.axhline(0.0, color="#333333", lw=0.5, zorder=0.5)
    for level in (-2.0, 2.0):
        axis.axhline(level, color=COLOR_GUIDE, lw=0.6, ls=(0, (3.0, 2.0)), zorder=0.5)
    top = max(3.0, largest + 0.8)
    # the u / v separator spans the data range only: the legend sits below it
    axis.vlines(28.5, -top, top, color=COLOR_GUIDE, lw=0.5, zorder=0.5)
    axis.set_xlim(0.0, 57.0)
    axis.set_ylim(-top - 0.9, top)               # the legend sits below the data
    axis.xaxis.set_major_locator(MultipleLocator(14))
    axis.xaxis.set_minor_locator(MultipleLocator(7))
    axis.yaxis.set_major_locator(MultipleLocator(1.0))
    axis.set_xlabel("reference point (Marchi et al. 2021, 28 + 28)", labelpad=1.5)
    axis.set_ylabel(r"$\Delta / \sigma$", labelpad=1.5)
    axis.text(14.5, top - 0.1, r"$u(0.5, y)$", ha="center", va="top")
    axis.text(42.5, top - 0.1, r"$v(x, 0.5)$", ha="center", va="top")
    axis.legend(loc="lower center", ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.3, borderaxespad=0.2,
                borderpad=0.0)
    hide_top_right(axis)
    return {"largest": largest}


def draw_strips(axis, folded: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """(b) of cavity_seams.pdf: rms |delta v| (K = 8 - K = 1) of the 7 seam strips against the distance to the nearest
    seam, one thin line per seam; the 6 control strips' min - max as a grey band. Values in 1e-5 U."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    scale = 1.0e5
    distances = folded["seam1"][0]
    controls = np.array([folded[f"control{index}"][1] for index in range(1, CONTROL_COUNT + 1)]) * scale
    axis.fill_between(distances, controls.min(axis=0), controls.max(axis=0), color=COLOR_CONTROL_BAND, lw=0,
                      label="controls (min – max)", zorder=0.8)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, SEAM_COUNT))
    peak = float(controls.max())
    for index in range(1, SEAM_COUNT + 1):
        values = folded[f"seam{index}"][1] * scale
        peak = max(peak, float(values.max()))
        axis.plot(distances, values, "-", color=colors[index - 1], lw=0.75, label=f"seam {index}")
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(0.0, peak * 1.6)                # the legend (two rows) sits above the data
    axis.xaxis.set_major_locator(MultipleLocator(1.0))
    axis.xaxis.set_minor_locator(MultipleLocator(0.5))
    axis.yaxis.set_major_locator(MultipleLocator(2.0))
    axis.set_xlabel(r"distance to the nearest seam ($h$)", labelpad=1.5)
    axis.set_ylabel(r"rms $|\Delta\boldsymbol{v}|/U$ ($10^{-5}$)", labelpad=1.5)
    # three columns: the control entry is wide; the pair is the legend's title
    axis.legend(loc="upper center", ncol=3, frameon=False, fontsize=FONT_SIZE - 2.0, title=r"$K = 8 - K = 1$",
                title_fontsize=FONT_SIZE - 1.5, columnspacing=1.0, handlelength=1.5, handletextpad=0.4,
                borderaxespad=0.2, borderpad=0.0, labelspacing=0.3)
    hide_top_right(axis)
    return {"peak": peak, "distances": distances}


def check_seams_clearances(figure, axis_a, axis_b, minimum_pt: float = 0.5) -> None:
    """Rendered-figure checks: the legend of (a) against its markers and the -2 guide line, the texts of (a) against the
    markers and the +2 line, the legend of (b) against the seam lines and the band, the (b) y label against (a)."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    report = []
    points_a = _drawn_points(axis_a, renderer)
    legend_a = axis_a.get_legend().get_window_extent(renderer)
    report.append(("(a) legend", {"markers/lines": _clearance_points(legend_a, points_a, dpi)}))
    for text in axis_a.texts:
        box = text.get_window_extent(renderer)
        report.append((f"(a) text {text.get_text()!r}", {"markers/lines": _clearance_points(box, points_a, dpi)}))
    points_b = _drawn_points(axis_b, renderer)
    legend_b = axis_b.get_legend().get_window_extent(renderer)
    band_top = max(collection.get_window_extent(renderer).y1 for collection in axis_b.collections)
    report.append(("(b) legend", {"seam lines": _clearance_points(legend_b, points_b, dpi),
                                  "control band": (legend_b.y0 - band_top) * 72.0 / dpi}))
    label_b = axis_b.yaxis.label.get_window_extent(renderer)
    panel_a_texts = [box for _, box in _label_boxes(axis_a, renderer)] + [axis_a.xaxis.label.get_window_extent(renderer)]
    report.append(("(b) y label", {"(a) markers/lines": _clearance_points(label_b, points_a, dpi),
                                   "(a) tick and axis labels": _clearance_boxes(label_b, panel_a_texts, dpi),
                                   "(b) tick labels": _clearance_boxes(label_b, [box for _, box in _label_boxes(axis_b, renderer)], dpi)}))
    _report_clearances(report, minimum_pt)


def seams_figure(width: float, path: pathlib.Path) -> dict:
    import matplotlib.pyplot as plt
    layout = SEAMS_LAYOUT
    pairs = point_differences()
    folded = folded_strip_bins()
    panel_width = (width - layout["left"] - layout["gap"] - layout["right"]) / 2.0
    height = layout["bottom"] + layout["height"] + layout["top_margin"]
    figure = plt.figure(figsize=(width, height))
    axis_a = axes_inches(figure, layout["left"], layout["bottom"], panel_width, layout["height"])
    axis_b = axes_inches(figure, layout["left"] + panel_width + layout["gap"], layout["bottom"], panel_width,
                         layout["height"])
    points = draw_point_differences(axis_a, pairs)
    strips = draw_strips(axis_b, folded)
    for axis, text in zip((axis_a, axis_b), ("(a)", "(b)")):
        panel_label(axis, text)
    check_seams_clearances(figure, axis_a, axis_b)
    save(figure, path)
    plt.close(figure)
    return {"width_in": width, "height_in": height, "largest_sigma_ratio": points["largest"],
            "peak_rms_1e-5": strips["peak"], "distances_h": [float(value) for value in strips["distances"]]}


def save(figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # every artist must lie inside the page (the page size is the layout, no bbox cropping)
    renderer = figure.canvas.get_renderer()
    box = figure.get_tightbbox(renderer)
    width, height = figure.get_size_inches()
    if box.x0 < -1e-3 or box.y0 < -1e-3 or box.x1 > width + 1e-3 or box.y1 > height + 1e-3:
        raise RuntimeError(f"{path.name}: artists outside the page: {box} vs {width:.3f} x {height:.3f} in")
    figure.savefig(path, dpi=600)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    print(f"[figure] wrote {path.relative_to(_REPO_ROOT)} (+ .png), {width:.3f} x {height:.3f} in", flush=True)


# --------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-points", type=int, default=1025, help="nodes per side of the averaged field (wall rows)")
    parser.add_argument("--workers", type=int, default=4, help="snapshots interpolated in parallel")
    parser.add_argument("--width", type=float, default=ELSARTICLE_SINGLE_COLUMN_WIDTH, help="figure width, inches")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached mean fields")
    parser.add_argument("--out", default=str(FIGURES / "cavity_re1000.pdf"))
    parser.add_argument("--out-seams", default=str(FIGURES / "cavity_seams.pdf"))
    parser.add_argument("--out-k1-field", default=str(FIGURES / "cavity_re1000_k1_field.pdf"))
    parser.add_argument("--seams-only", action="store_true", help="only cavity_seams.pdf (from the committed CSVs)")
    arguments = parser.parse_args()

    setup_style()
    if arguments.seams_only:
        print(json.dumps(seams_figure(arguments.width, pathlib.Path(arguments.out_seams)), indent=1), flush=True)
        return 0

    runs = {key: run_facts(run_id) for key, run_id in RUN_IDS.items()}
    window = common_window(runs["K8"])
    for run in runs.values():
        run["window"] = window
    print(f"[figure] common window t = {window[0]:.4f} .. {window[1]:.4f} ({len(runs['K8']['sample_steps'])} samples of "
          f"the K = 8 run, steps {min(runs['K8']['sample_steps']):,} .. {max(runs['K8']['sample_steps']):,})", flush=True)
    line = k1_lines_on_window(runs["K1"], runs["K8"])
    quantities = {}
    for key, run in runs.items():
        field = mean_field(run, arguments.grid_points, arguments.workers, arguments.refresh)
        quantities[key] = field_quantities(field)
        print(f"[figure] {key}: {len(field['snapshots'])} snapshots t = {float(field['times'][0]):.2f} .. "
              f"{float(field['times'][-1]):.2f}, MLS fallbacks {int(field['mls_fallbacks'])}, max |v| "
              f"{quantities[key]['speed_max']:.4f}, psi min / max {quantities[key]['psi_min']:.5f} / "
              f"{quantities[key]['psi_max']:.3e}", flush=True)
    seams = runs["K8"]["seams_wall"]
    if len(seams) != SEAM_COUNT:
        raise RuntimeError(f"K8: {len(seams)} seams in meta.json, expected {SEAM_COUNT}")
    print("[figure] seams (K = 8, wall frame) X = " + " ".join(f"{seam:.4f}" for seam in seams), flush=True)
    size = composite_figure(line, quantities["K8"], seams, arguments.width, pathlib.Path(arguments.out))
    field_only_figure(quantities["K1"], size, pathlib.Path(arguments.out_k1_field))
    print(json.dumps(seams_figure(arguments.width, pathlib.Path(arguments.out_seams)), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
