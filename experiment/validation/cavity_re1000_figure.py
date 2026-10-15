"""cavity_re1000_figure.py - paper section 6.2: the Re = 1000 lid-driven cavity validation figure from the finished E40
runs (v7-rc3, adami walls, 1000^2, K = 1 on GPU 1 and K = 2 on GPUs 0 + 1). Uses existing data only: nothing is run.

Outputs (docs/validation/figures/, vector PDF + PNG preview):
  cavity_re1000.pdf      three panels in a row, elsarticle single-column text width (preprint / 1p: 384 pt = 5.33 in):
    (a) u(0.5, y) and (b) v(x, 0.5): time-averaged MLS centre lines of K = 1 (solid) and K = 2 (dashed) against
        Marchi et al. 2021 (open circles) and Ghia et al. 1982 (open squares), with zooms near u_min and v_min; (b)
        marks the K = 2 seam with a thin dashed line (no text: the caption explains it); the legend is in (a) only;
    (c) K = 2: time-averaged speed |u|/U (colour 0 - 1) with streamlines and the seam (thin white dashed line).
  cavity_re1000_k1_field.pdf   panel (c) alone for K = 1 (same size and colour scale, no seam, no panel label).

Data:
  * centre lines: docs/validation/data/dense_<run>.csv (adami_chain_report: time-averaged MLS on 1001 points of each
    centre line, wall-row frame, averaging window = the last 20 time units of the run);
  * the field of (c): the snapshots of the same averaging window (logs/validation/cavity_re1000_v7/<run>/snapshots,
    20 per run) interpolated with the linear MLS of cavity_sampling.interpolate (the method of the psi_min analysis in
    cavity_analysis.stream_function_minimum, there on 513^2 and u only) onto a node grid that spans the innermost wall
    rows, then averaged; cached as mean_field_<run>_<points>.npz next to the runs;
  * streamlines: contours of the stream function of the averaged field, solved from its vorticity with psi = 0 on the
    effective walls (cavity_fields.stream_function_from_vorticity; the integral of u dy does not close because the
    stored velocities carry a net flux of ~3e-4 U, which would distort the small corner-eddy levels);
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
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.cavity_fields import UniformGrid, read_case_information, stream_function_from_vorticity  # noqa: E402

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re1000_v7"
DATA = _REPO_ROOT / "docs" / "validation" / "data"
FIGURES = _REPO_ROOT / "docs" / "validation" / "figures"
RUN_IDS = {"K1": "n1000_k1_float32_xi0p001_eps0p0025_adami", "K2": "n1000_k2_float32_xi0p001_eps0p0025_adami"}
AVERAGE_SPAN = 20.0                       # time units, as in the validation analysis
ELSARTICLE_SINGLE_COLUMN_WIDTH = 384.0 / 72.0   # preprint and 1p text width, 384 pt (elsdoc: 5.3 in)
CHUNK_POINTS = 40000

# figure style (the serif style of the existing validation figures, cavity_analysis.setup_style)
FONT_SIZE = 8.0
INSET_TICK_SIZE = 8.0                     # zoom tick labels (2026-10-15: was 6 pt)
FIELD_YLABEL_PAD = 4.5                    # (c) y label to its tick labels, points (was 1.5)
COLOR_K1 = "#08519c"
COLOR_K2 = "#6baed6"
COLOR_GHIA = "#b2182b"
SEAM_PROFILE_COLOR = "#555555"
STREAMLINE_COLOR = "#f0f0f0"
# stream-function levels: the primary vortex (psi < 0, psi_min = -0.119) and the two bottom corner eddies (psi > 0;
# BR1 max ~ 1.7e-3, BL1 max ~ 2.4e-4 in these runs)
PSI_LEVELS_NEGATIVE = (-0.115, -0.10, -0.08, -0.06, -0.04, -0.02, -0.005)
PSI_LEVELS_POSITIVE = (1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 1.5e-3)


# --------------------------------------------------------------------------------------------- data
def run_facts(run_id: str) -> dict:
    directory = RUNS_ROOT / run_id
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    last = None
    with (directory / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    t_last = float(json.loads(last)["time"])
    information = read_case_information(meta)
    spacing = float(meta["spacing"])
    support = float(meta["support_radius"])
    length = 1.0 + 2.0 * spacing
    seams = [((information["grid_origin_x"] + int(column) * support) + 0.5 + spacing) / length
             for column in (meta.get("cuts") or [])]
    return {"id": run_id, "dir": directory, "meta": meta, "spacing": spacing, "support": support,
            "dt": float(meta["dt"]), "window": (t_last - AVERAGE_SPAN, t_last), "seams_wall": seams}


def dense_lines(run_id: str) -> dict:
    path = DATA / f"dense_{run_id}.csv"
    rows = [row for row in csv.reader(line for line in path.read_text(encoding="utf-8").splitlines()
                                      if line and not line.startswith("#"))]
    header, values = rows[0], np.array(rows[1:], dtype=float)
    column = {name: index for index, name in enumerate(header)}
    return {"coordinate": values[:, column["coordinate"]], "u": values[:, column["u_at_x_0.5"]],
            "v": values[:, column["v_at_y_0.5"]], "source": str(path.relative_to(_REPO_ROOT)).replace("\\", "/")}


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
    """Time-averaged MLS velocity on a node grid spanning the innermost wall rows (physical +-(0.5 + dx))."""
    cache = RUNS_ROOT / f"mean_field_{run['id']}_{grid_points}.npz"
    files = window_snapshots(run)
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


def draw_profiles(axis_u, axis_v, lines: dict, reference: dict, ghia: dict, seam_wall: float | None,
                  legend: bool) -> None:
    from matplotlib.ticker import FixedLocator, MultipleLocator
    styles = {"K1": dict(color=COLOR_K1, lw=1.0, ls="-", label=r"SPH, $K = 1$"),
              "K2": dict(color=COLOR_K2, lw=1.0, ls=(0, (3.2, 1.8)), label=r"SPH, $K = 2$")}
    marchi_style = dict(ls="none", marker="o", mfc="none", mec="k", ms=3.0, mew=0.6, label="Marchi et al. (2021)")
    ghia_style = dict(ls="none", marker="s", mfc="none", mec=COLOR_GHIA, ms=2.6, mew=0.55, label="Ghia et al. (1982)")
    for key in ("K1", "K2"):
        axis_u.plot(lines[key]["u"], lines[key]["coordinate"], **styles[key])
        axis_v.plot(lines[key]["coordinate"], lines[key]["v"], **styles[key])
    axis_u.plot(reference["u"][1], reference["u"][0], **marchi_style)
    axis_u.plot(ghia["u"][1], ghia["u"][0], **ghia_style)
    axis_v.plot(reference["v"][0], reference["v"][1], **marchi_style)
    axis_v.plot(ghia["v"][0], ghia["v"][1], **ghia_style)
    if seam_wall is not None:
        axis_v.axvline(seam_wall, color=SEAM_PROFILE_COLOR, lw=0.5, ls=(0, (2.5, 2.0)), zorder=0.5)

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
        axis.tick_params(which="both", top=False, right=False)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    # zooms: u_min (lower right of (a)), v_min (lower left of (b)); ticks inside the main axes, clear of its ticks
    inset_u = axis_u.inset_axes([0.63, 0.54, 0.33, 0.27])
    inset_v = axis_v.inset_axes([0.27, 0.15, 0.21, 0.34])
    for key in ("K1", "K2"):
        inset_u.plot(lines[key]["u"], lines[key]["coordinate"], **{**styles[key], "lw": 0.9})
        inset_v.plot(lines[key]["coordinate"], lines[key]["v"], **{**styles[key], "lw": 0.9})
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
    # one labelled tick at 8 pt: two ("0.85", "0.95") would touch each other or the seam line
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


def draw_field(axis, colorbar_axis, quantities: dict, seam_wall: float | None) -> dict:
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator
    wall_axis = quantities["axis"]
    image = axis.imshow(quantities["speed"], origin="lower", extent=[0.0, 1.0, 0.0, 1.0], cmap="viridis",
                        norm=Normalize(0.0, 1.0), interpolation="bilinear", aspect="equal")
    levels = sorted(PSI_LEVELS_NEGATIVE + PSI_LEVELS_POSITIVE)
    axis.contour(wall_axis, wall_axis, quantities["psi"], levels=levels, colors=STREAMLINE_COLOR, linewidths=0.35,
                 linestyles="solid", alpha=0.9)
    if seam_wall is not None:
        axis.axvline(seam_wall, color="white", lw=0.6, ls=(0, (3.0, 2.0)))
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


def check_clearances(figure, axis_u, axis_v, axis_field, insets: dict, minimum_pt: float = 0.5) -> None:
    """Measured on the rendered figure: the zoom tick labels against the main curves / reference markers / seam
    line, the main ticks and tick labels and the legend; each zoom frame against the main curves; the (c) y label
    against everything of panel (b). Raises when anything overlaps (gap < minimum_pt)."""
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
            gaps = {"curves/markers/seam": _clearance_points(box, points, dpi),
                    "main ticks": _clearance_boxes(box, main_ticks, dpi),
                    "main tick labels": _clearance_boxes(box, main_labels, dpi)}
            if legend_boxes:
                gaps["legend"] = _clearance_boxes(box, legend_boxes, dpi)
            frame = axis.get_window_extent(renderer)
            gaps["main axis lines"] = min(box.x0 - frame.x0, box.y0 - frame.y0) * 72.0 / dpi
            report.append((f"{name} tick label {text!r} ({box.height * 72.0 / dpi:.1f} pt tall)", gaps))
        report.append((f"{name} frame", {"curves/markers/seam": _clearance_points(inset.get_window_extent(renderer),
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
    failures = []
    for item, gaps in report:
        print(f"[check] {item}: " + ", ".join(f"{key} {value:+.1f} pt" for key, value in gaps.items()), flush=True)
        failures += [f"{item} vs {key} ({value:+.1f} pt)" for key, value in gaps.items() if value < minimum_pt]
    if failures:
        raise RuntimeError("overlaps: " + "; ".join(failures))


def composite_figure(lines: dict, field_k2: dict, seam_wall: float, width: float, path: pathlib.Path) -> dict:
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
    insets = draw_profiles(axis_u, axis_v, lines, reference, ghia, seam_wall, legend=True)
    draw_field(axis_field, colorbar_axis, field_k2, seam_wall)
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
    draw_field(axis, colorbar_axis, field, None)
    save(figure, path)
    plt.close(figure)


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
    parser.add_argument("--out-k1-field", default=str(FIGURES / "cavity_re1000_k1_field.pdf"))
    arguments = parser.parse_args()

    runs = {key: run_facts(run_id) for key, run_id in RUN_IDS.items()}
    lines = {key: dense_lines(run["id"]) for key, run in runs.items()}
    quantities = {}
    for key, run in runs.items():
        field = mean_field(run, arguments.grid_points, arguments.workers, arguments.refresh)
        quantities[key] = field_quantities(field)
        print(f"[figure] {key}: {len(field['snapshots'])} snapshots t = {float(field['times'][0]):.2f} .. "
              f"{float(field['times'][-1]):.2f}, MLS fallbacks {int(field['mls_fallbacks'])}, max |u| "
              f"{quantities[key]['speed_max']:.4f}, psi min / max {quantities[key]['psi_min']:.5f} / "
              f"{quantities[key]['psi_max']:.3e}", flush=True)
    seam = runs["K2"]["seams_wall"][0]
    print(f"[figure] seam (K = 2, wall frame) X = {seam:.4f}", flush=True)
    setup_style()
    size = composite_figure(lines, quantities["K2"], seam, arguments.width, pathlib.Path(arguments.out))
    field_only_figure(quantities["K1"], size, pathlib.Path(arguments.out_k1_field))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
