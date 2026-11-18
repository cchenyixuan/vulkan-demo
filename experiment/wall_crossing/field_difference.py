"""
field_difference.py - where do two finished 250^2 cavity runs differ? No new run: everything comes from the
snapshots the runner kept (the last 21, one per time unit, t = 80.2 ... 100.0).

    .venv/Scripts/python.exe -m experiment.wall_crossing.field_difference

Runs compared: the old default (crossing, continuity_wall_component full: n250_crossing) minus the first order
of today (normal: n250_crossing_normal). Second realisations of either (n250_crossing_r2; the control
n250_crossing_normal_rho_minus, whose pressure is positive again from t = 50 on) give the size of a difference
that means nothing. n250_crossing_normal_rho_minus_0p45 (normal, pressure negative throughout) separates the
sign of the pressure from the continuity rule.

1. Time-mean fields. Every snapshot's fluid particles are put on the nodes of the initial lattice
   (x_k = -0.5 + k dx, 251 x 251 nodes; the first node row is the first particle layer, half a spacing from
   the wall plane +-0.502) with Shepard weights of a Wendland C2 kernel of radius 1.5 dx, summed over the
   snapshots: u, v and the level-free pressure p' = p - (mean over the fluid of that snapshot), p from the
   stored volume V = m / rho through the Tait EOS (rho / rho0 = V_wall / V: a wall particle stores rho0).
   The level is removed per snapshot because it differs by an order of magnitude between the runs and drifts
   during the window.
2. Difference fields (old default minus first order) and their distribution: the share of the integral of
   |delta u|^2 (and of delta p'^2) in the four corners, the middle sections of the four walls and the interior,
   for a band width of 0.1 L and of h = 0.02 L (the kernel support), beside the same numbers of the difference
   between two runs of the old default (what a difference of two time means is worth). And how much of the
   interior's difference is a uniform scaling of the first-order field: u_old = (1 + alpha) u_first + r, alpha
   by least squares over the interior (more than 0.1 L from every wall; the lid's speed is prescribed, so the
   field under it cannot scale).
3. Fluctuations: per region and layer, the rms of the particles' velocity and level-free pressure about the
   time-mean field (bilinear from the nodes): particle-scale noise plus what the mean field does not resolve.
4. The tangential part of the wall neighbours' drift, evaluated with the numpy reference (reference.py) on the
   particles of the last snapshots: T_i = (drift over the wall neighbours with the whole vector) - (the same
   with the normal component only), per fluid particle with a wall neighbour, as a velocity divergence (T / rho,
   units U / L), beside the divergence the normal-only rule keeps. This is the term the first order drops.

Output: claude/wall_crossing/figures/field_difference_*.png, claude/wall_crossing/tables/field_difference_*.md,
claude/wall_crossing/field_difference/summary.json; one runs.jsonl record.

Figure conventions: those of the approved cavity field figures (docs/cavity_validation/scripts/
plot_cavity_fields.py of the main checkout: a main panel and four corner zooms with their own colour scales and
their raw range in the title, walls gray and the lid near-black at true thickness, viridis for a magnitude,
RdBu_r for a signed quantity). The zooms show the lattice nodes as cells: a time mean has no particles to draw.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import yaml                         # noqa: E402
from matplotlib.cm import ScalarMappable            # noqa: E402
from matplotlib.colors import Normalize             # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402
from scipy.interpolate import RegularGridInterpolator   # noqa: E402
from scipy.spatial import cKDTree                   # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, SURFACE, setup_style   # noqa: E402
from experiment.wall_crossing.report_tables import markdown_table                                      # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                        # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
OLD, FIRST = "n250_crossing", "n250_crossing_normal"
OLD_REPEAT, FIRST_REPEAT, FIRST_NEGATIVE = "n250_crossing_r2", "n250_crossing_normal_rho_minus", \
    "n250_crossing_normal_rho_minus_0p45"
RUN_TEXT = {OLD: "旧缺省（完整向量）", OLD_REPEAT: "旧缺省，第二次运行", FIRST: "一阶（只取法向）",
            FIRST_REPEAT: "一阶，另一次运行（初始密度 ρ0 − 0.1 的对照）",
            FIRST_NEGATIVE: "一阶，全程负压（初始密度 ρ0 − 0.45 的对照）"}
WINDOW = (80.0, 100.5)
HALF_BOX = 0.502                    # the crossing box: the fluid lattice's bounding box + one particle radius
DYNAMIC_PRESSURE = 1000.0           # rho0 U^2
KERNEL_RADIUS_SPACINGS = 1.5
BANDS = (("0.1 L", 0.1), ("h = 0.02 L（核半径）", 0.02))
REGIONS = (("top_left", "左上角（盖的上游）"), ("top_right", "右上角（盖的下游）"), ("bottom_left", "左下角"),
           ("bottom_right", "右下角"), ("lid", "盖的中段"), ("bottom", "底壁中段"), ("left", "左壁中段"),
           ("right", "右壁中段"), ("interior", "内部"))
GROUPS = (("两个上角", ("top_left", "top_right")), ("两个下角", ("bottom_left", "bottom_right")),
          ("四面壁的中段", ("lid", "bottom", "left", "right")), ("内部", ("interior",)))
LAYERS = ((0.0, 1.0, "第 1 层（< 1 dx）"), (1.0, 2.0, "第 2 层"), (2.0, 3.0, "第 3 层"), (3.0, 5.0, "3 到 5 dx"))
WALL_COLOR, LID_COLOR, ANNOTATION_COLOR = "#8c8c8c", "#1a1a1a", "#e6007e"
ZOOM_WIDTH = 0.1
ARROW_THRESHOLD = 0.0015            # |delta u| / U below which no direction arrow is drawn
TERM_SNAPSHOTS = 3                  # the last snapshots on which the dropped term is evaluated


def load_run(name: str) -> dict:
    directory = LOG_ROOT / "cavity" / name
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    case = yaml.safe_load((REPOSITORY_ROOT / meta["case"]).read_text(encoding="utf-8"))
    speed_of_sound, power = float(case["physics"]["speed_of_sound"]), float(case["physics"]["power"])
    timestep = float(meta["dt"])
    snapshots = [(int(path.stem[1:]) * timestep, path) for path in sorted((directory / "snapshots").glob("t*.npz"))]
    snapshots = [(instant, path) for instant, path in snapshots if WINDOW[0] <= instant <= WINDOW[1]]
    return {"name": name, "meta": meta, "case_path": REPOSITORY_ROOT / meta["case"], "snapshots": snapshots,
            "spacing": float(meta["spacing"]), "support": float(meta["support_radius"]),
            "eos_constant": speed_of_sound ** 2 * 1000.0 / power, "power": power}


def snapshot_particles(run: dict, path: pathlib.Path) -> dict:
    data = np.load(path)
    fluid = np.isin(data["material"], run["meta"]["fluid_groups"])
    volume = data["volumes"].astype(np.float64)
    wall_volume = float(np.median(volume[~fluid]))                  # a wall particle stores rho0
    density_ratio = wall_volume / volume
    pressure = run["eos_constant"] * (density_ratio ** run["power"] - 1.0)
    return {"positions": data["positions"].astype(np.float64), "velocities": data["velocities"].astype(np.float64),
            "pressure": pressure, "density_ratio": density_ratio, "fluid": fluid, "material": data["material"],
            "wall_volume": wall_volume}


def lattice_nodes(spacing: float) -> np.ndarray:
    count = int(round(1.0 / spacing)) + 1
    return -0.5 + spacing * np.arange(count)


def mean_fields(run: dict) -> dict:
    """(u, v, p') on the lattice nodes, [iy, ix], the time mean over the run's snapshots; and the levels."""
    spacing = run["spacing"]
    axis = lattice_nodes(spacing)
    grid_x, grid_y = np.meshgrid(axis, axis)
    nodes = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    node_tree = cKDTree(nodes)
    radius = KERNEL_RADIUS_SPACINGS * spacing
    weight_sum = np.zeros(nodes.shape[0])
    sums = np.zeros((nodes.shape[0], 3))
    levels, instants = [], []
    for instant, path in run["snapshots"]:
        particles = snapshot_particles(run, path)
        fluid = particles["fluid"]
        positions = particles["positions"][fluid]
        level = float(particles["pressure"][fluid].mean())
        values = np.stack([particles["velocities"][fluid, 0], particles["velocities"][fluid, 1],
                           (particles["pressure"][fluid] - level) / DYNAMIC_PRESSURE], axis=1)
        pairs = node_tree.sparse_distance_matrix(cKDTree(positions), radius, output_type="coo_matrix")
        ratio = pairs.data / radius
        weight = (1.0 - ratio) ** 4 * (1.0 + 4.0 * ratio)
        weight_sum += np.bincount(pairs.row, weights=weight, minlength=nodes.shape[0])
        for column in range(3):
            sums[:, column] += np.bincount(pairs.row, weights=weight * values[pairs.col, column],
                                           minlength=nodes.shape[0])
        levels.append(level / DYNAMIC_PRESSURE)
        instants.append(instant)
    shape = (axis.size, axis.size)
    mean = sums / weight_sum[:, None]
    return {"axis": axis, "u": mean[:, 0].reshape(shape), "v": mean[:, 1].reshape(shape),
            "p": mean[:, 2].reshape(shape), "levels": levels, "instants": instants,
            "empty_nodes": int((weight_sum == 0).sum())}


def region_masks(x: np.ndarray, y: np.ndarray, width: float) -> dict:
    """The nine regions of points (x, y) for a band of the given width along the walls of the box."""
    left, right = x + HALF_BOX < width, HALF_BOX - x < width
    bottom, top = y + HALF_BOX < width, HALF_BOX - y < width
    return {"top_left": top & left, "top_right": top & right, "bottom_left": bottom & left,
            "bottom_right": bottom & right, "lid": top & ~left & ~right, "bottom": bottom & ~left & ~right,
            "left": left & ~top & ~bottom, "right": right & ~top & ~bottom,
            "interior": ~(left | right | top | bottom)}


def shares(squared: np.ndarray, axis: np.ndarray, width: float) -> dict:
    """Per region: the share of the sum of a squared node field, the area share and the rms."""
    grid_x, grid_y = np.meshgrid(axis, axis)
    masks = region_masks(grid_x, grid_y, width)
    total = squared.sum()
    return {key: {"share": float(squared[mask].sum() / total), "area": float(mask.mean()),
                  "rms": float(np.sqrt(squared[mask].mean()))} for key, mask in masks.items()}


def fluctuations(run: dict, fields: dict) -> dict:
    """rms of the particles' velocity and level-free pressure about the time-mean field, per wall middle section
    and layer (distance to that wall in spacings, corners of 0.1 L left out) and in the interior beyond 5 dx."""
    axis = fields["axis"]
    interpolators = {key: RegularGridInterpolator((axis, axis), fields[key], bounds_error=False, fill_value=None)
                     for key in ("u", "v", "p")}
    spacing = run["spacing"]
    sums: dict = {}
    for _, path in run["snapshots"]:
        particles = snapshot_particles(run, path)
        fluid = particles["fluid"]
        positions = particles["positions"][fluid]
        clipped = np.clip(positions, axis[0], axis[-1])
        query = np.stack([clipped[:, 1], clipped[:, 0]], axis=1)            # (y, x): the fields are [iy, ix]
        level = particles["pressure"][fluid].mean()
        velocity_deviation = ((particles["velocities"][fluid, 0] - interpolators["u"](query)) ** 2
                              + (particles["velocities"][fluid, 1] - interpolators["v"](query)) ** 2)
        pressure_deviation = ((particles["pressure"][fluid] - level) / DYNAMIC_PRESSURE - interpolators["p"](query)) ** 2
        x, y = positions[:, 0], positions[:, 1]
        distances = {"lid": HALF_BOX - y, "bottom": y + HALF_BOX, "left": x + HALF_BOX, "right": HALF_BOX - x}
        along = {"lid": np.abs(x), "bottom": np.abs(x), "left": np.abs(y), "right": np.abs(y)}
        nearest = np.minimum.reduce(list(distances.values()))
        selections = {("interior", "≥ 5 dx"): nearest >= 5.0 * spacing}
        for wall, distance in distances.items():
            middle = along[wall] <= HALF_BOX - ZOOM_WIDTH
            for low, high, text in LAYERS:
                selections[(wall, text)] = middle & (distance >= low * spacing) & (distance < high * spacing)
        for key, mask in selections.items():
            entry = sums.setdefault(key, [0.0, 0.0, 0])
            entry[0] += float(velocity_deviation[mask].sum())
            entry[1] += float(pressure_deviation[mask].sum())
            entry[2] += int(mask.sum())
    return {f"{wall}|{layer}": {"velocity_rms": float(np.sqrt(entry[0] / entry[2])),
                                "pressure_rms": float(np.sqrt(entry[1] / entry[2])), "samples": entry[2]}
            for (wall, layer), entry in sums.items()}


def dropped_term(run: dict) -> dict:
    """The tangential part of the wall neighbours' drift on the last snapshots of a run, with the numpy
    reference: per wall middle section and layer the mean and rms of T / rho (whole vector minus normal only),
    of the normal-only wall part and of the fluid neighbours' part, all as velocity divergences in U / L."""
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing.reference import ReferenceConstants, evaluate_reference

    case = load_case_v7(run["case_path"])
    constants = ReferenceConstants.from_case(case)
    spacing, support = run["spacing"], run["support"]
    rest_density = 1000.0
    collected: dict = {}
    for _, path in run["snapshots"][-TERM_SNAPSHOTS:]:
        particles = snapshot_particles(run, path)
        positions2 = particles["positions"]
        # only the particles within 2 h + 2 dx of the box faces are needed: a fluid particle with a wall
        # neighbour lies within h of a face and its neighbours within 2 h
        nearest = HALF_BOX - np.abs(positions2).max(axis=1)
        keep = nearest < 2.0 * support + 2.0 * spacing
        positions = np.zeros((int(keep.sum()), 3))
        positions[:, 0:2] = positions2[keep]
        velocities = np.zeros_like(positions)
        velocities[:, 0:2] = particles["velocities"][keep]
        is_wall = ~particles["fluid"][keep]
        densities = rest_density * particles["density_ratio"][keep]
        mass = rest_density * particles["wall_volume"]
        masses = np.full(positions.shape[0], mass)
        pressures = np.zeros(positions.shape[0])
        results = {}
        for component in ("full", "normal"):
            results[component] = evaluate_reference(
                positions, velocities, densities, pressures, masses, is_wall,
                dataclasses.replace(constants, continuity="extrapolate", continuity_wall_component=component,
                                    continuity_extension="linear", viscous_extension="linear"))
        reference = results["normal"]
        fluid_rows = reference.fluid_index
        density = densities[fluid_rows]
        tangential = (results["full"].density_rate_drift_wall - reference.density_rate_drift_wall) / (-density)
        normal_part = reference.density_rate_drift_wall / (-density)
        fluid_part = reference.density_rate_drift_fluid / (-density)
        x, y = positions[fluid_rows, 0], positions[fluid_rows, 1]
        has_wall = reference.wall_neighbour_count > 0
        distances = {"lid": HALF_BOX - y, "bottom": y + HALF_BOX, "left": x + HALF_BOX, "right": HALF_BOX - x}
        along = {"lid": np.abs(x), "bottom": np.abs(x), "left": np.abs(y), "right": np.abs(y)}
        for wall, distance in distances.items():
            middle = has_wall & (along[wall] <= HALF_BOX - ZOOM_WIDTH)
            for low, high, text in LAYERS:
                mask = middle & (distance >= low * spacing) & (distance < high * spacing)
                entry = collected.setdefault(f"{wall}|{text}", {"tangential": [], "normal": [], "fluid": []})
                entry["tangential"].append(tangential[mask])
                entry["normal"].append(normal_part[mask])
                entry["fluid"].append(fluid_part[mask])
        # the four corners: the squares of 0.1 L and of h (every fluid particle there with a wall neighbour)
        for corner, horizontal, vertical in (("top_left", distances["left"], distances["lid"]),
                                             ("top_right", distances["right"], distances["lid"]),
                                             ("bottom_left", distances["left"], distances["bottom"]),
                                             ("bottom_right", distances["right"], distances["bottom"])):
            for width_text, width in (("0.1 L", ZOOM_WIDTH), ("h", support)):
                mask = has_wall & (horizontal < width) & (vertical < width)
                entry = collected.setdefault(f"{corner}|{width_text}", {"tangential": [], "normal": [], "fluid": []})
                entry["tangential"].append(tangential[mask])
                entry["normal"].append(normal_part[mask])
                entry["fluid"].append(fluid_part[mask])
    summary = {}
    for key, entry in collected.items():
        tangential = np.concatenate(entry["tangential"])
        normal_part = np.concatenate(entry["normal"])
        fluid_part = np.concatenate(entry["fluid"])
        kept = fluid_part + normal_part
        summary[key] = {"samples": int(tangential.size),
                        "tangential_mean": float(tangential.mean()), "tangential_rms": float(np.sqrt((tangential ** 2).mean())),
                        "kept_mean": float(kept.mean()), "kept_rms": float(np.sqrt((kept ** 2).mean())),
                        "whole_mean": float((kept + tangential).mean()),
                        "whole_rms": float(np.sqrt(((kept + tangential) ** 2).mean())),
                        "normal_rms": float(np.sqrt((normal_part ** 2).mean())),
                        "fluid_rms": float(np.sqrt((fluid_part ** 2).mean()))}
    return summary


def draw_walls(axis, layers_thickness: float) -> None:
    outer = HALF_BOX + layers_thickness
    shift = 0.5
    for x0, y0, width, height, colour in (
            (-outer, -outer, 2 * outer, outer - HALF_BOX, WALL_COLOR),                 # bottom
            (-outer, -HALF_BOX, outer - HALF_BOX, HALF_BOX + outer, WALL_COLOR),       # left (with the top corner)
            (HALF_BOX, -HALF_BOX, outer - HALF_BOX, HALF_BOX + outer, WALL_COLOR),     # right (with the top corner)
            (-HALF_BOX, HALF_BOX, 2 * HALF_BOX, outer - HALF_BOX, LID_COLOR)):         # the lid
        axis.add_patch(Rectangle((x0 + shift, y0 + shift), width, height, facecolor=colour, edgecolor="none", zorder=1))
    axis.add_patch(Rectangle((-HALF_BOX + shift, -HALF_BOX + shift), 2 * HALF_BOX, 2 * HALF_BOX, facecolor="none",
                             edgecolor="#ffffff", linewidth=0.6, zorder=4))


def field_figure(field: np.ndarray, axis_nodes: np.ndarray, spacing: float, title: str, colorbar_label: str,
                 colormap: str, signed: bool, path: pathlib.Path, arrows: tuple | None = None,
                 main_limit: float | None = None) -> dict:
    """Main panel and four corner zooms (each with its own scale and its raw range in the title)."""
    setup_style()
    plt.rcParams.update({"axes.grid": False})
    figure = plt.figure(figsize=(15.6, 7.6))
    grid = figure.add_gridspec(2, 4, width_ratios=(1.55, 1.55, 1.3, 1.3), left=0.04, right=0.985, top=0.86,
                               bottom=0.075, wspace=0.34, hspace=0.34)
    main = figure.add_subplot(grid[:, 0:2])
    edges = np.concatenate([axis_nodes - 0.5 * spacing, [axis_nodes[-1] + 0.5 * spacing]]) + 0.5
    limit = main_limit if main_limit is not None else float(np.percentile(np.abs(field), 99.5))
    norm = Normalize(-limit, limit) if signed else Normalize(0.0, limit)
    clipped = 100.0 * float((np.abs(field) > limit).mean())
    main.pcolormesh(edges, edges, field, cmap=colormap, norm=norm, shading="flat", zorder=2, rasterized=True)
    draw_walls(main, 11 * spacing)
    if arrows is not None:
        # the direction only (unit arrows; the colour carries the size), where the difference is above
        # ARROW_THRESHOLD (a few times the run-to-run difference of the mean fields)
        step = 12
        sample = (slice(step // 2, None, step), slice(step // 2, None, step))
        grid_x, grid_y = np.meshgrid(axis_nodes + 0.5, axis_nodes + 0.5)
        size = np.sqrt(arrows[0][sample] ** 2 + arrows[1][sample] ** 2)
        visible = size > ARROW_THRESHOLD
        main.quiver(grid_x[sample][visible], grid_y[sample][visible], (arrows[0][sample] / size)[visible],
                    (arrows[1][sample] / size)[visible], color="#ffffff", scale=42, scale_units="width",
                    width=0.0022, headwidth=3.6, zorder=3, alpha=0.9)
    for width, style in ((0.1, (0, (4, 3))),):
        for value in (width - 0.002, 1.002 - width):
            main.plot([value, value], [-0.002, 1.002], color=ANNOTATION_COLOR, linewidth=0.7, linestyle=style, zorder=5)
            main.plot([-0.002, 1.002], [value, value], color=ANNOTATION_COLOR, linewidth=0.7, linestyle=style, zorder=5)
    main.set_xlim(-0.05, 1.05)
    main.set_ylim(-0.05, 1.05)
    main.set_aspect("equal")
    main.set_xlabel("x / L")
    main.set_ylabel("y / L")
    main.set_title(f"整个方腔（色标到 99.5 % 分位 {limit:.4f}，超出的格点 {clipped:.2f} %；原始范围 "
                   f"{field.min():+.4f} … {field.max():+.4f}）", fontsize=9.5)
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=main, fraction=0.035, pad=0.015)
    colorbar.set_label(colorbar_label)
    colorbar.outline.set_visible(False)
    corners = (("左上角（盖的上游）", 0, 0, -0.002, 1.002 - ZOOM_WIDTH), ("右上角（盖的下游）", 0, 1, 1.002 - ZOOM_WIDTH, 1.002 - ZOOM_WIDTH),
               ("左下角", 1, 0, -0.002, -0.002), ("右下角", 1, 1, 1.002 - ZOOM_WIDTH, -0.002))
    centres = axis_nodes + 0.5
    ranges = {}
    for text, row, column, x0, y0 in corners:
        zoom = figure.add_subplot(grid[row, 2 + column])
        inside_x = (centres >= x0) & (centres <= x0 + ZOOM_WIDTH)
        inside_y = (centres >= y0) & (centres <= y0 + ZOOM_WIDTH)
        part = field[np.ix_(inside_y, inside_x)]
        zoom_limit = float(np.abs(part).max())
        zoom_norm = Normalize(-zoom_limit, zoom_limit) if signed else Normalize(0.0, zoom_limit)
        zoom.pcolormesh(edges, edges, field, cmap=colormap, norm=zoom_norm, shading="flat", zorder=2, rasterized=True)
        draw_walls(zoom, 11 * spacing)
        margin = 0.012
        zoom.set_xlim(x0 - (margin if column == 0 else 0.0), x0 + ZOOM_WIDTH + (margin if column == 1 else 0.0))
        zoom.set_ylim(y0 - (margin if row == 1 else 0.0), y0 + ZOOM_WIDTH + (margin if row == 0 else 0.0))
        zoom.set_aspect("equal")
        zoom.set_title(f"{text}\n{part.min():+.4f} … {part.max():+.4f}", fontsize=8.5)
        zoom.tick_params(labelsize=7)
        zoom_bar = figure.colorbar(ScalarMappable(norm=zoom_norm, cmap=colormap), ax=zoom, fraction=0.045, pad=0.03)
        zoom_bar.ax.tick_params(labelsize=7)
        zoom_bar.outline.set_visible(False)
        ranges[text] = [float(part.min()), float(part.max())]
    figure.suptitle(title, x=0.012, y=0.975, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.text(0.012, 0.925, "虚线 = 表里 0.1 L 的分区（四个角、四面壁的中段、内部）。灰 = 静止壁，黑 = 盖；右边四幅是四个角 0.1 L "
                              "的放大，各用自己的色标，标题里是该角的原始范围。", fontsize=8.5, color=INK_SECONDARY,
                ha="left", va="top")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return ranges


def main() -> int:
    tables = REPORT_ROOT / "tables"
    figures = REPORT_ROOT / "figures"
    out = REPORT_ROOT / "field_difference"
    for directory in (tables, figures, out):
        directory.mkdir(parents=True, exist_ok=True)
    with LoggedRun(kind="field_difference", case="none") as run_record:
        names = [name for name in (OLD, FIRST, OLD_REPEAT, FIRST_REPEAT, FIRST_NEGATIVE)
                 if (LOG_ROOT / "cavity" / name / "snapshots").is_dir()]
        runs = {name: load_run(name) for name in names}
        fields = {name: mean_fields(runs[name]) for name in names}
        spacing = runs[OLD]["spacing"]
        axis = fields[OLD]["axis"]
        summary = {"run_id": run_record.run_id, "window": [runs[OLD]["snapshots"][0][0], runs[OLD]["snapshots"][-1][0]],
                   "snapshots": {name: len(runs[name]["snapshots"]) for name in names},
                   "pressure_level": {name: [min(fields[name]["levels"]), max(fields[name]["levels"])] for name in names},
                   "empty_nodes": {name: fields[name]["empty_nodes"] for name in names}}

        def difference(first: str, second: str) -> dict:
            delta_u = fields[first]["u"] - fields[second]["u"]
            delta_v = fields[first]["v"] - fields[second]["v"]
            delta_p = fields[first]["p"] - fields[second]["p"]
            return {"u": delta_u, "v": delta_v, "p": delta_p, "speed": np.sqrt(delta_u ** 2 + delta_v ** 2)}

        main_difference = difference(OLD, FIRST)
        pairs = {"旧缺省 − 一阶": main_difference}
        if OLD_REPEAT in fields:
            pairs["旧缺省两次运行之差"] = difference(OLD, OLD_REPEAT)
        if FIRST_REPEAT in fields:
            pairs["一阶两次运行之差"] = difference(FIRST, FIRST_REPEAT)
        if FIRST_NEGATIVE in fields:
            pairs["一阶全程负压 − 一阶"] = difference(FIRST_NEGATIVE, FIRST)
            pairs["旧缺省 − 一阶全程负压"] = difference(OLD, FIRST_NEGATIVE)

        # the uniform scaling of the first-order field contained in the interior's difference
        grid_x, grid_y = np.meshgrid(axis, axis)
        interior = region_masks(grid_x, grid_y, 0.1)["interior"]
        first_u, first_v = fields[FIRST]["u"][interior], fields[FIRST]["v"][interior]
        old_u, old_v = fields[OLD]["u"][interior], fields[OLD]["v"][interior]
        alpha = float((old_u * first_u + old_v * first_v).sum() / (first_u ** 2 + first_v ** 2).sum() - 1.0)
        residual_squared = (old_u - (1.0 + alpha) * first_u) ** 2 + (old_v - (1.0 + alpha) * first_v) ** 2
        difference_squared = main_difference["speed"][interior] ** 2
        explained = 1.0 - float(residual_squared.sum() / difference_squared.sum())
        summary["scaling"] = {"alpha": alpha, "share_of_interior_squared_difference_explained": explained,
                              "interior_residual_rms": float(np.sqrt(residual_squared.mean())),
                              "interior_difference_rms": float(np.sqrt(difference_squared.mean())),
                              "interior_first_order_speed_rms": float(np.sqrt((first_u ** 2 + first_v ** 2).mean()))}

        # ---- the overall sizes of the differences
        rows = []
        summary["pairs"] = {}
        for text, delta in pairs.items():
            entry = {"speed_rms": float(np.sqrt((delta["speed"] ** 2).mean())), "speed_maximum": float(delta["speed"].max()),
                     "pressure_rms": float(np.sqrt((delta["p"] ** 2).mean())),
                     "pressure_maximum": float(np.abs(delta["p"]).max())}
            summary["pairs"][text] = entry
            rows.append([text, f"{entry['speed_rms']:.5f}", f"{entry['speed_maximum']:.4f}",
                         f"{entry['pressure_rms']:.5f}", f"{entry['pressure_maximum']:.4f}"])
        (tables / "field_difference_sizes.md").write_text(markdown_table(
            ["两个时间平均场之差", "速度差的 rms（盖速 = 1）", "速度差的最大值", "去电平压力差的 rms（ρ0U²）",
             "去电平压力差的最大值"], rows) + "\n", encoding="utf-8", newline="\n")

        # ---- the distribution over the regions
        summary["shares"] = {}
        for band_text, width in BANDS:
            quantities = (("速度差 |Δu|²", main_difference["speed"] ** 2),
                          ("去电平压力差 Δp'²", main_difference["p"] ** 2))
            computed = {text: shares(squared, axis, width) for text, squared in quantities}
            noise = noise_pressure = None
            if "旧缺省两次运行之差" in pairs:
                noise = shares(pairs["旧缺省两次运行之差"]["speed"] ** 2, axis, width)
                noise_pressure = shares(pairs["旧缺省两次运行之差"]["p"] ** 2, axis, width)
            rows = []
            for key, text in REGIONS:
                cells = [text, f"{100 * computed['速度差 |Δu|²'][key]['area']:.2f} %"]
                for quantity, _ in quantities:
                    entry = computed[quantity][key]
                    cells += [f"{100 * entry['share']:.1f} %", f"{entry['rms']:.5f}"]
                cells.append("—" if noise is None else f"{noise[key]['rms']:.5f}")
                cells.append("—" if noise_pressure is None else f"{noise_pressure[key]['rms']:.5f}")
                rows.append(cells)
            for group_text, members in GROUPS:
                cells = [f"**{group_text}合计**",
                         f"{100 * sum(computed['速度差 |Δu|²'][key]['area'] for key in members):.2f} %"]
                for quantity, _ in quantities:
                    cells += [f"**{100 * sum(computed[quantity][key]['share'] for key in members):.1f} %**", ""]
                cells += ["", ""]
                rows.append(cells)
            header = ["区域", "面积占比"]
            for quantity, _ in quantities:
                header += [f"{quantity} 的占比", "该区域内的 rms"]
            header += ["对照：旧缺省两次运行之差的速度 rms", "对照：同上，压力 rms"]
            name = "field_difference_shares_" + ("wide" if width == 0.1 else "support") + ".md"
            (tables / name).write_text(
                f"分区宽度 {band_text}：角 = 离两面壁都不到这个宽度的方块，壁的中段 = 离一面壁不到这个宽度、去掉角，内部 = 其余。"
                "占比 = 该区域内平方差之和 ÷ 全腔之和；rms 的单位是盖速（速度）和 ρ0U²（压力）。\n\n"
                + markdown_table(header, rows) + "\n", encoding="utf-8", newline="\n")
            summary["shares"][band_text] = {text: values for text, values in computed.items()}

        # ---- fluctuations about the mean field
        fluctuation = {name: fluctuations(runs[name], fields[name]) for name in names}
        summary["fluctuations"] = fluctuation
        keys = [f"{wall}|{text}" for wall in ("lid", "bottom", "left", "right") for _, _, text in LAYERS] \
            + ["interior|≥ 5 dx"]
        wall_text = {"lid": "盖", "bottom": "底壁", "left": "左壁", "right": "右壁", "interior": "内部"}
        shown = [name for name in (OLD, OLD_REPEAT, FIRST, FIRST_REPEAT, FIRST_NEGATIVE) if name in names]
        for quantity, label, file_name in (("velocity_rms", "速度（盖速 = 1）", "field_difference_fluctuation_velocity.md"),
                                           ("pressure_rms", "去电平压力（ρ0U²）", "field_difference_fluctuation_pressure.md")):
            rows = []
            for key in keys:
                wall, layer = key.split("|")
                rows.append([wall_text[wall], layer] + [f"{fluctuation[name][key][quantity]:.5f}" for name in shown])
            (tables / file_name).write_text(
                f"粒子的{label}相对时间平均场的 rms（{len(runs[OLD]['snapshots'])} 个快照的全部粒子；壁的中段，去掉 0.1 L 的角）。\n\n"
                + markdown_table(["壁", "层"] + [RUN_TEXT[name] for name in shown], rows) + "\n",
                encoding="utf-8", newline="\n")

        # ---- the term the first order drops, on the particles of the last snapshots
        term = {name: dropped_term(runs[name]) for name in (FIRST, OLD) if name in names}
        summary["dropped_term"] = term
        rows = []
        for key in keys[:-1]:
            wall, layer = key.split("|")
            cells = [wall_text[wall], layer]
            for name in (FIRST, OLD):
                entry = term[name][key]
                cells += [f"{entry['tangential_mean']:+.4f}", f"{entry['tangential_rms']:.4f}",
                          f"{entry['kept_mean']:+.4f}", f"{entry['kept_rms']:.4f}", f"{entry['whole_rms']:.4f}"]
            rows.append(cells)
        corner_text = {"top_left": "左上角", "top_right": "右上角", "bottom_left": "左下角", "bottom_right": "右下角"}
        for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
            for width_text in ("0.1 L", "h"):
                cells = [corner_text[corner], f"{width_text} 的方块"]
                for name in (FIRST, OLD):
                    entry = term[name][f"{corner}|{width_text}"]
                    cells += [f"{entry['tangential_mean']:+.4f}", f"{entry['tangential_rms']:.4f}",
                              f"{entry['kept_mean']:+.4f}", f"{entry['kept_rms']:.4f}", f"{entry['whole_rms']:.4f}"]
                rows.append(cells)
        header = ["壁", "层"]
        for name in (FIRST, OLD):
            short = "一阶的粒子" if name == FIRST else "旧缺省的粒子"
            header += [f"{short}：切向部分的平均", "切向部分的 rms", "只取法向时留下的散度的平均", "它的 rms",
                       "完整向量的散度的 rms"]
        (tables / "field_difference_dropped_term.md").write_text(
            f"壁邻居漂移项里的切向部分，用 numpy 参考解在各运行最后 {TERM_SNAPSHOTS} 个快照的粒子上算出，换算成速度散度"
            "（单位 U / L）。“切向部分” = 完整向量的壁项 − 只取法向的壁项，就是一阶规则丢掉的那一项；“只取法向时留下的散度” = "
            "流体邻居的部分 + 壁邻居的法向部分；“完整向量的散度” = 两者之和。只统计有壁邻居的流体粒子，壁的中段。\n\n"
            + markdown_table(header, rows) + "\n", encoding="utf-8", newline="\n")

        # ---- figures
        ranges = {}
        ranges["velocity"] = field_figure(
            main_difference["speed"], axis, spacing,
            "250² 方腔，t = 80 到 100 的平均：速度场之差的大小 |u(旧缺省) − u(一阶)|（盖速 = 1；箭头 = 差的方向）",
            "|Δu| / U", "viridis", False, figures / "field_difference_velocity.png",
            arrows=(main_difference["u"], main_difference["v"]))
        ranges["pressure"] = field_figure(
            main_difference["p"], axis, spacing,
            "250² 方腔，t = 80 到 100 的平均：去电平压力之差 p'(旧缺省) − p'(一阶)（ρ0U²；每个快照先减去全场平均压力）",
            "Δp' / (ρ0 U²)", "RdBu_r", True, figures / "field_difference_pressure.png")
        summary["figure_ranges"] = ranges
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
        run_record.extra["label"] = ("time-mean fields of the stored snapshots (t = 80 .. 100) of the 250^2 cavity runs: "
                                     "old default minus first order, by region")
        run_record.extra["status"] = "ok"
        print(f"[field_difference] interior: alpha {100 * alpha:+.3f} %, a uniform scaling explains "
              f"{100 * explained:.1f} % of the interior's squared velocity difference; record {run_record.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
