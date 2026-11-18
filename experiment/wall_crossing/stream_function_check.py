"""
stream_function_check.py - how far the post-processing of the cavity runs can be trusted for psi_min, the
velocity extrema and the vortex centre. No solver run: the stored snapshots (t = 80 to 100) only.

    .venv/Scripts/python.exe -m experiment.wall_crossing.stream_function_check grids --run n250_crossing_normal
    .venv/Scripts/python.exe -m experiment.wall_crossing.stream_function_check grids --run n250_... --analytic
    .venv/Scripts/python.exe -m experiment.wall_crossing.stream_function_check tables

The post-processing under test (experiment/validation/cavity_analysis.py, cavity_sampling.py):
    - the velocity of EVERY particle of a snapshot (fluid, walls, lid: a wall particle carries its stored
      velocity, 0 or the lid speed) is interpolated to a 513 x 513 grid that spans the centre lines of the
      innermost wall / lid rows, [-(0.5 + dx), 0.5 + dx]^2, by a linear moving-least-squares fit with the weight
      W(r / h) V_j (Wendland C4, support radius h = the case's, V_j = m_j / rho_j); the wall particles fill the
      support, so no kernel is cut off at a wall;
    - the grids of the snapshots of the averaging window are averaged;
    - psi(x, y) = integral of u from the bottom grid row upward (trapezoid); psi_min = the minimum over
      |x|, |y| < 0.3, refined by a quadratic surface over 5 x 5 grid points;
    - "wall frame": lengths are divided by L = 1 + 2 dx (the unit square = the innermost wall rows), so the table's
      psi_min is psi / L; "mid frame": L = 1 + dx (the unit square = half way between the innermost wall row and
      the outermost fluid row, the box of wall_boundary crossing);
    - the velocity extrema come from the centre lines: linear MLS at 1001 points per line (wall frame), averaged
      over the samples of the window, the discrete extremum refined by a parabola through 7 points.

grids: the time-mean u and v grids of one run (the snapshots of the window the run's analysis used), cached in
logs/wall_crossing/stream_function_check/. With --analytic the particles' velocities are replaced by an analytic
field before the same interpolation: psi = -A f(x / a) g(y / a), f(s) = (1 - s^2)^2 (1 + alpha s),
g(s) = (1 - s^2)^2 (1 + beta s), u = d psi / d y, v = -d psi / d x. psi and both velocity components vanish on
the walls |x| = a, |y| = a; alpha, beta put the minimum where the cavity's primary vortex sits, A makes it
-0.119. Fluid particles take the field at their own positions, wall and lid particles 0. Two wall positions:
a = 0.5 + dx / 2 ("mid": the crossing box) and a = 0.5 + dx ("wall": the innermost wall rows). The centre-line
samples of the same snapshots are taken too (the runner's sampling function).

tables: for the analytic fields, the pipeline's psi_min, vortex centre and velocity extrema against the exact
ones; for the runs, psi_min and the vortex centre by four integration paths:
    A  u, upward from the bottom row     (the analysis)
    B  u, downward from the top row      (the lid's row)
    C  v, rightward from the left row    psi = - integral of v dx
    D  v, leftward from the right row
and the vortex centres against Marchi et al. 2021. Output: claude/wall_crossing/tables/psi_check_*.md,
claude/wall_crossing/stream_function_check/summary.json; one runs.jsonl record per call.
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
from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                    # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
CACHE = LOG_ROOT / "stream_function_check"
RUNS = (("n250_reflection_full_volume", "250² 完整方案 + 体积约束"), ("n250_crossing_normal", "250² 一阶（现缺省）"),
        ("n250_adami", "250² adami"), ("n250_simple", "250² simple"),
        ("n500_reflection_full_volume", "500² 完整方案 + 体积约束"), ("n500_crossing_normal", "500² 一阶（现缺省）"),
        ("n500_adami", "500² adami"), ("n500_simple", "500² simple"))
ANALYTIC_RUNS = (("n250_reflection_full_volume", "250²"), ("n500_reflection_full_volume", "500²"))
# the analytic field: wall position label -> half width a as a function of the spacing
ANALYTIC_WALLS = {"mid": lambda spacing: 0.5 + 0.5 * spacing, "wall": lambda spacing: 0.5 + spacing}
ANALYTIC_WALL_TEXT = {"mid": "壁在 ±(0.5 + dx/2)（crossing 的盒面，mid 帧）", "wall": "壁在 ±(0.5 + dx)（最内层壁粒子行，wall 帧）"}
ANALYTIC_MINIMUM = -0.119
ANALYTIC_CENTRE = (0.0616, 0.1304)          # (x / a, y / a) of the minimum: Marchi's vortex centre (0.5308, 0.5652)
PATHS = (("A", "u，从底行向上（原算法）"), ("B", "u，从顶行（盖）向下"), ("C", "v，从左行向右"), ("D", "v，从右行向左"))
CHUNK = 40000


# ------------------------------------------------------------------------------------------- the analytic field
def analytic_coefficients() -> dict:
    """alpha, beta from the wanted position of the minimum (f'(s0) = 0 <=> alpha = 4 s0 / (1 - 5 s0^2)), and the
    amplitude that makes the minimum ANALYTIC_MINIMUM."""
    centre_x, centre_y = ANALYTIC_CENTRE
    alpha = 4.0 * centre_x / (1.0 - 5.0 * centre_x ** 2)
    beta = 4.0 * centre_y / (1.0 - 5.0 * centre_y ** 2)
    amplitude = -ANALYTIC_MINIMUM / (profile(centre_x, alpha)[0] * profile(centre_y, beta)[0])
    return {"alpha": alpha, "beta": beta, "amplitude": amplitude}


def profile(scaled: np.ndarray, skew: float) -> tuple:
    """f(s) = (1 - s^2)^2 (1 + skew s) and f'(s), zero outside |s| <= 1."""
    scaled = np.asarray(scaled, dtype=np.float64)
    inside = np.abs(scaled) <= 1.0
    bump = (1.0 - scaled ** 2)
    value = np.where(inside, bump ** 2 * (1.0 + skew * scaled), 0.0)
    derivative = np.where(inside, -4.0 * scaled * bump * (1.0 + skew * scaled) + skew * bump ** 2, 0.0)
    return value, derivative


def analytic_field(positions: np.ndarray, half_width: float) -> tuple:
    """(psi, u, v) at positions (N, 2): psi = -A f(x / a) g(y / a), u = d psi / d y, v = -d psi / d x."""
    coefficients = analytic_coefficients()
    value_x, derivative_x = profile(positions[:, 0] / half_width, coefficients["alpha"])
    value_y, derivative_y = profile(positions[:, 1] / half_width, coefficients["beta"])
    amplitude = coefficients["amplitude"]
    return (-amplitude * value_x * value_y, -amplitude * value_x * derivative_y / half_width,
            amplitude * derivative_x * value_y / half_width)


# ------------------------------------------------------------------------------------------------- the grids
def analysis_window(name: str) -> tuple:
    """The averaging window the run's analysis used (claude/wall_crossing/cavity/tables.json)."""
    tables = json.loads((REPORT_ROOT / "cavity" / "tables.json").read_text(encoding="utf-8"))["runs"]
    return tuple(tables[name]["window"])


def snapshot_files(run: dict, window: tuple) -> list:
    timestep = float(run["meta"]["dt"])
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    return [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * timestep <= window[1] + 1e-9]


def mean_grids(name: str, analytic_wall: str | None) -> dict:
    """The time mean of the MLS grid (both velocity components) of the run's snapshots, as the analysis takes it
    (cavity_analysis.stream_function_minimum's grid, points and interpolation); with analytic_wall the particles
    carry the analytic field instead, and the two centre lines are sampled as the runner samples them."""
    run = cavity_analysis.load_run(name, LOG_ROOT / "cavity")
    window = analysis_window(name)
    selected = snapshot_files(run, window)
    if not selected:
        raise RuntimeError(f"{name}: no snapshot in the window {window}")
    return grids_of_files(run, selected, analytic_wall)


def grids_of_files(run: dict, selected: list, analytic_wall: str | None = None) -> dict:
    """mean_grids for an explicit list of snapshot files of a run (cavity_analysis.load_run's dict)."""
    spacing = run["spacing"]
    support = float(run["meta"]["support_radius"])
    half = 0.5 + spacing
    axis = np.linspace(-half, half, cavity_analysis.GRID_POINTS)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    totals = np.zeros((points.shape[0], 2))
    fallbacks = 0
    point_sets = {"u_dense": ("u", run["points"]["u_dense"]), "v_dense": ("v", run["points"]["v_dense"])}
    lines = {"u_dense": [], "v_dense": []}
    half_width = ANALYTIC_WALLS[analytic_wall](spacing) if analytic_wall else None
    for path in selected:
        with np.load(path) as archive:
            positions = archive["positions"].astype(np.float64)
            velocities = archive["velocities"].astype(np.float64)
            volumes = archive["volumes"].astype(np.float64)
            fluid = np.isin(archive["material"], run["meta"]["fluid_groups"])
        if analytic_wall:
            _, velocity_x, velocity_y = analytic_field(positions, half_width)
            velocities = np.where(fluid[:, None], np.stack([velocity_x, velocity_y], axis=1), 0.0)
        particles = sampling.ParticleSet(positions, velocities, volumes)
        tree = sampling.cKDTree(positions)
        for start in range(0, points.shape[0], CHUNK):
            chunk = slice(start, start + CHUNK)
            sampled = sampling.interpolate(particles, points[chunk], support, tree=tree)
            totals[chunk] += sampled["mls"]
            fallbacks += int(sampled["mls_fallback"].sum())
        if analytic_wall:
            sampled_lines = sampling.sample_centerlines(particles, support, point_sets)
            for key in lines:
                lines[key].append(sampled_lines[key]["mls"])
    count = cavity_analysis.GRID_POINTS
    result = {"axis": axis, "u": (totals[:, 0] / len(selected)).reshape(count, count),
              "v": (totals[:, 1] / len(selected)).reshape(count, count),
              "snapshots": np.int64(len(selected)), "fallbacks": np.int64(fallbacks),
              "spacing": np.float64(spacing), "support": np.float64(support)}
    if analytic_wall:
        result["u_dense"] = np.mean(lines["u_dense"], axis=0)
        result["v_dense"] = np.mean(lines["v_dense"], axis=0)
        result["dense_reference"] = run["points"]["dense_reference"]
        result["half_width"] = np.float64(half_width)
    return result


def cache_path(name: str, analytic_wall: str | None) -> pathlib.Path:
    return CACHE / (name + (f"_analytic_{analytic_wall}" if analytic_wall else "") + ".npz")


def make_grids(name: str, analytic: bool) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    with LoggedRun(kind="analysis", case="none") as run:
        run.extra["label"] = f"stream_function_check grids {name}" + (" (analytic fields)" if analytic else "")
        for wall in (list(ANALYTIC_WALLS) if analytic else [None]):
            grids = mean_grids(name, wall)
            np.savez(cache_path(name, wall), **grids)
            print(f"[stream_function_check] {name}{' analytic ' + wall if wall else ''}: "
                  f"{int(grids['snapshots'])} snapshots, {int(grids['fallbacks'])} MLS fallbacks -> "
                  f"{cache_path(name, wall).name} ({run.run_id})", flush=True)
    return 0


# ----------------------------------------------------------------------------------------- the four paths
def stream_functions(u_grid: np.ndarray, v_grid: np.ndarray, axis: np.ndarray) -> dict:
    """psi on the grid ([iy, ix]) by the four paths."""
    offset = axis - axis[0]
    upward = sampling.stream_function(u_grid, offset)                           # A: psi = 0 on the bottom row
    rightward = -sampling.stream_function(v_grid.T, offset).T                   # C: psi = 0 on the left row
    return {"A": upward, "B": upward - upward[-1][None, :], "C": rightward,
            "D": rightward - rightward[:, -1][:, None]}


def refined_minimum(field: np.ndarray, axis: np.ndarray) -> tuple:
    """(x, y, psi) of the minimum over |x|, |y| < 0.3, as the analysis refines it."""
    grid_x, grid_y = np.meshgrid(axis, axis)
    masked = np.where((np.abs(grid_x) < 0.3) & (np.abs(grid_y) < 0.3), field, np.nan)
    return sampling.refine_grid_minimum(axis, axis, masked)


def frames(spacing: float) -> dict:
    return {"wall": 1.0 + 2.0 * spacing, "mid": 1.0 + spacing}


def write_table(name: str, text: str) -> None:
    path = REPORT_ROOT / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[stream_function_check] wrote {path.relative_to(REPOSITORY_ROOT)}")


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def tables() -> int:
    reference = cavity_reference.marchi2021()["extrema"]
    reference_psi = float(reference["psi_min"][0])
    reference_centre = (float(reference["x_at_psi_min"][0]), float(reference["y_at_psi_min"][0]))
    analysed = json.loads((REPORT_ROOT / "cavity" / "tables.json").read_text(encoding="utf-8"))["runs"]
    with LoggedRun(kind="analysis", case="none") as run:
        run.extra["label"] = "stream_function_check tables"
        summary = {"run_id": run.run_id, "analytic": {}, "runs": {}}

        # ---------------------------------------------------------------- the analytic fields
        coefficients = analytic_coefficients()
        summary["analytic_coefficients"] = coefficients
        psi_rows, extrema_rows = [], []
        for name, resolution_text in ANALYTIC_RUNS:
            for wall in ANALYTIC_WALLS:
                path = cache_path(name, wall)
                if not path.exists():
                    continue
                data = np.load(path)
                axis, spacing = data["axis"], float(data["spacing"])
                half_width = float(data["half_width"])
                exact_centre = (ANALYTIC_CENTRE[0] * half_width, ANALYTIC_CENTRE[1] * half_width)
                fields = stream_functions(data["u"], data["v"], axis)
                entry = {"half_width": half_width, "snapshots": int(data["snapshots"]),
                         "fallbacks": int(data["fallbacks"]), "paths": {}}
                cells = [resolution_text, ANALYTIC_WALL_TEXT[wall], f"{ANALYTIC_MINIMUM:.5f}"]
                for key, _ in PATHS:
                    x, y, value = refined_minimum(fields[key], axis)
                    entry["paths"][key] = {
                        "psi": value, "error_percent": 100.0 * (abs(value) / abs(ANALYTIC_MINIMUM) - 1.0),
                        "centre_error_over_spacing": [(x - exact_centre[0]) / spacing, (y - exact_centre[1]) / spacing]}
                path_a = entry["paths"]["A"]
                cells += [f"{path_a['psi']:.5f}（{path_a['error_percent']:+.3f} %）"]
                cells += [f"{entry['paths'][key]['error_percent']:+.3f} %" for key in ("B", "C", "D")]
                cells += [f"({path_a['centre_error_over_spacing'][0]:+.2f}, {path_a['centre_error_over_spacing'][1]:+.2f})"]
                # what the tables would show: psi / L of a frame against the exact psi / (2 a)
                true_value = ANALYTIC_MINIMUM / (2.0 * half_width)
                for frame_name, length in frames(spacing).items():
                    shown = path_a["psi"] / length
                    entry[f"{frame_name}_frame_error_percent"] = 100.0 * (abs(shown) / abs(true_value) - 1.0)
                    cells.append(f"{entry[f'{frame_name}_frame_error_percent']:+.3f} %")
                # closure: psi of path A on the top row, of path C on the right row (exactly 0 for the field)
                entry["closure_top_maximum"] = float(np.abs(fields["A"][-1]).max())
                entry["closure_right_maximum"] = float(np.abs(fields["C"][:, -1]).max())
                psi_rows.append(cells)

                # the velocity extrema on the centre lines: the pipeline's against the exact ones
                dense = data["dense_reference"]
                wall_length = 1.0 + 2.0 * spacing
                fine = np.linspace(0.0, 1.0, 400001)
                fine_physical = fine * wall_length - 0.5 * wall_length
                zeros = np.zeros_like(fine_physical)
                _, exact_u, _ = analytic_field(np.stack([zeros, fine_physical], axis=1), half_width)
                _, _, exact_v = analytic_field(np.stack([fine_physical, zeros], axis=1), half_width)
                entry["extrema"] = {}
                cells = [resolution_text, ANALYTIC_WALL_TEXT[wall]]
                for quantity, (line, kind, search) in cavity_analysis.EXTREMA_SEARCH.items():
                    sampled = data["u_dense"] if line == "u" else data["v_dense"]
                    position, value = sampling.refine_extremum(dense, sampled, kind, search=search)
                    exact_profile = exact_u if line == "u" else exact_v
                    inside = (fine >= search[0]) & (fine <= search[1])
                    index = np.flatnonzero(inside)[np.argmin(exact_profile[inside]) if kind == "min"
                                                   else np.argmax(exact_profile[inside])]
                    exact_value, exact_position = float(exact_profile[index]), float(fine[index])
                    entry["extrema"][quantity] = {
                        "value": value, "exact": exact_value,
                        "error_percent": 100.0 * (abs(value) / abs(exact_value) - 1.0),
                        "position_error": position - exact_position}
                    cells.append(f"{exact_value:+.5f} → {value:+.5f}（{entry['extrema'][quantity]['error_percent']:+.3f} %）")
                # the whole lines: the largest error of the sampled profile, in units of the largest speed
                entry["line_error_maximum"] = {
                    "u": float(np.abs(data["u_dense"] - np.interp(dense, fine, exact_u)).max() / np.abs(exact_u).max()),
                    "v": float(np.abs(data["v_dense"] - np.interp(dense, fine, exact_v)).max() / np.abs(exact_v).max())}
                cells.append(f"{100 * entry['line_error_maximum']['u']:.2f} % / "
                             f"{100 * entry['line_error_maximum']['v']:.2f} %")
                # where the tables' profile norms are taken: the 28 points of Marchi et al. per line (wall frame),
                # and the lines without their two ends
                marchi = cavity_reference.marchi2021()
                inner = (dense > 0.05) & (dense < 0.95)
                entry["profile_points"], entry["line_error_inner_maximum"] = {}, {}
                point_cells, inner_cells = [], []
                for line, sampled, exact_profile in (("u", data["u_dense"], exact_u), ("v", data["v_dense"], exact_v)):
                    exact_on_dense = np.interp(dense, fine, exact_profile)
                    coordinates = marchi[line][0]
                    point_error = np.interp(coordinates, dense, sampled) - np.interp(coordinates, dense, exact_on_dense)
                    entry["profile_points"][line] = {"L2": float(np.sqrt((point_error ** 2).mean())),
                                                     "Linf": float(np.abs(point_error).max())}
                    entry["line_error_inner_maximum"][line] = float(np.abs(sampled - exact_on_dense)[inner].max())
                    point_cells.append(f"{entry['profile_points'][line]['L2']:.5f} / "
                                       f"{entry['profile_points'][line]['Linf']:.5f}")
                    inner_cells.append(f"{entry['line_error_inner_maximum'][line]:.5f}")
                cells += [" ; ".join(point_cells), " / ".join(inner_cells)]
                extrema_rows.append(cells)
                summary["analytic"][f"{name}|{wall}"] = entry
        if psi_rows:
            write_table("psi_check_analytic_psi.md",
                        "解析场 ψ = −A f(x/a) g(y/a) 在快照的真实粒子位置上取值（流体粒子取场值，壁粒子和盖粒子取 0），走同一套"
                        "后处理。ψ 以物理单位给出（没有除以帧的长度）；括号内和后面各列是相对精确值的误差，负 = 后处理偏弱。"
                        "涡心误差以粒子间距 dx 为单位。最后两列：表里的 ψ_min（除以该帧的长度）相对真正的无量纲值 ψ / (2a) "
                        "的误差。\n\n"
                        + markdown_table(["分辨率", "解析场的壁", "精确的 ψ_min", "后处理（路径 A，原算法）",
                                          "路径 B 的误差", "路径 C 的误差", "路径 D 的误差", "涡心误差 (Δx, Δy) / dx",
                                          "按 wall 帧报出的误差", "按 mid 帧报出的误差"], psi_rows))
            write_table("psi_check_analytic_extrema.md",
                        "同一个解析场，两条中心线上的速度极值：精确值 → 后处理的值（括号内 = 相对误差，负 = 偏弱）。最后一列是"
                        "整条中心线上采样剖面与精确剖面之差的最大值，以该分量的最大速度为单位（u 线 / v 线；它出现在线的端点，即壁面处）。"
                        "后两列以速度为单位：28 个 Marchi 点上的误差 L2 / L∞（u 线 ; v 线），和去掉两端各 5 % 之后整条线上"
                        "的最大误差（u 线 / v 线）。\n\n"
                        + markdown_table(["分辨率", "解析场的壁", "u_min", "v_max", "v_min", "整条线上的最大误差",
                                          "28 个 Marchi 点上的 L2 / L∞", "去掉两端后的最大误差"], extrema_rows))

        # ---------------------------------------------------------------- the runs: four paths, vortex centres
        path_rows, centre_rows, closure_rows = [], [], []
        for name, text in RUNS:
            path = cache_path(name, None)
            if not path.exists() or name not in analysed:
                continue
            data = np.load(path)
            axis, spacing = data["axis"], float(data["spacing"])
            half = 0.5 + spacing
            lengths = frames(spacing)
            fields = stream_functions(data["u"], data["v"], axis)
            tabled = analysed[name]["extrema"]["psi_min"]
            entry = {"snapshots": int(data["snapshots"]), "fallbacks": int(data["fallbacks"]), "paths": {},
                     "tabled_wall": tabled["value_wall"]}
            cells = [text]
            for key, _ in PATHS:
                x, y, value = refined_minimum(fields[key], axis)
                wall_value = value / lengths["wall"]
                entry["paths"][key] = {
                    "psi_physical": value, "psi_wall": wall_value,
                    "deviation_wall_percent": 100.0 * (abs(wall_value) / abs(reference_psi) - 1.0),
                    "deviation_mid_percent": 100.0 * (abs(value / lengths["mid"]) / abs(reference_psi) - 1.0),
                    "centre_wall": [(x + half) / lengths["wall"], (y + half) / lengths["wall"]],
                    "centre_mid": [(x + 0.5 * lengths["mid"]) / lengths["mid"], (y + 0.5 * lengths["mid"]) / lengths["mid"]]}
            path_a = entry["paths"]["A"]
            cells.append(f"{path_a['psi_wall']:.5f}（{path_a['deviation_wall_percent']:+.2f} %）")
            for key in ("B", "C", "D"):
                other = entry["paths"][key]
                difference = other["deviation_wall_percent"] - path_a["deviation_wall_percent"]
                entry["paths"][key]["difference_from_A_points"] = difference
                cells.append(f"{other['deviation_wall_percent']:+.2f} %（{difference:+.2f}）")
            spread = [entry["paths"][key]["deviation_wall_percent"] for key in ("A", "C", "D")]
            entry["spread_without_lid_path"] = max(spread) - min(spread)
            cells.append(f"{entry['spread_without_lid_path']:.2f}")
            cells.append(f"{abs(path_a['psi_wall'] - tabled['value_wall']):.1e}")
            path_rows.append(cells)

            centre_cells = [text]
            for frame_name in ("wall", "mid"):
                centre = path_a[f"centre_{frame_name}"]
                centre_cells.append(f"({centre[0]:.4f}, {centre[1]:.4f})")
                centre_cells.append(f"({centre[0] - reference_centre[0]:+.4f}, {centre[1] - reference_centre[1]:+.4f})")
            mid = path_a["centre_mid"]
            entry["centre_mid_deviation_over_spacing"] = [(mid[0] - reference_centre[0]) * lengths["mid"] / spacing,
                                                          (mid[1] - reference_centre[1]) * lengths["mid"] / spacing]
            centre_cells.append(f"({entry['centre_mid_deviation_over_spacing'][0]:+.2f}, "
                                f"{entry['centre_mid_deviation_over_spacing'][1]:+.2f})")
            other = entry["paths"]["C"]["centre_mid"]
            centre_cells.append(f"({other[0] - mid[0]:+.4f}, {other[1] - mid[1]:+.4f})")
            centre_rows.append(centre_cells)

            # closure of the mean field: psi of path A on the top row and of path C on the right row
            column = int(np.argmin(np.abs(axis - (path_a["centre_wall"][0] * lengths["wall"] - half))))
            row = int(np.argmin(np.abs(axis - (path_a["centre_wall"][1] * lengths["wall"] - half))))
            entry["closure"] = {"top_at_vortex_column": float(fields["A"][-1, column]),
                                "top_maximum": float(np.abs(fields["A"][-1]).max()),
                                "right_at_vortex_row": float(fields["C"][row, -1]),
                                "right_maximum": float(np.abs(fields["C"][:, -1]).max())}
            closure_rows.append([text, f"{entry['closure']['top_at_vortex_column']:+.5f}",
                                 f"{entry['closure']['top_at_vortex_column'] / spacing:+.2f}",
                                 f"{entry['closure']['top_maximum']:.5f}",
                                 f"{entry['closure']['right_at_vortex_row']:+.5f}",
                                 f"{entry['closure']['right_maximum']:.5f}"])
            summary["runs"][name] = entry
        if path_rows:
            write_table("psi_check_paths.md",
                        f"ψ_min（wall 帧）按四条积分路径，同一批快照的同一个时间平均速度场。括号内 = 相对 Marchi 2021（"
                        f"{reference_psi:.5f}）的偏差；B、C、D 三列是该路径的偏差，括号内是它减去路径 A 的偏差（百分点）。"
                        "“A、C、D 的极差”是不经过盖的三条路径之间最大与最小偏差之差。最后一列：本工具的路径 A 与分析表里的值"
                        "之差（应为 0）。\n\n"
                        + markdown_table(["运行", "A：u，从底行向上（原算法）", "B：u，从顶行（盖）向下", "C：v，从左行向右",
                                          "D：v，从右行向左", "A、C、D 的极差（百分点）", "与分析表之差"], path_rows))
            write_table("psi_check_closure.md",
                        "时间平均速度场的闭合：路径 A 积到顶行（盖的第一排粒子）时的 ψ，和路径 C 积到右行时的 ψ。无散、壁面"
                        "无滑移的场这两个数都是 0。\n\n"
                        + markdown_table(["运行", "顶行的 ψ，涡心所在的列", "同，÷ (U dx)", "顶行 |ψ| 的最大值",
                                          "右行的 ψ，涡心所在的行", "右行 |ψ| 的最大值"], closure_rows))
            write_table("psi_check_centres.md",
                        f"主涡中心（路径 A，原算法），Marchi 2021 的位置是 ({reference_centre[0]:.4f}, "
                        f"{reference_centre[1]:.4f})。偏差 = 本次 − 参考。倒数第二列是 mid 帧的偏差换成粒子间距 dx；最后一列是"
                        "路径 C 的涡心减路径 A 的涡心（mid 帧）。\n\n"
                        + markdown_table(["运行", "wall 帧的位置", "wall 帧的偏差", "mid 帧的位置", "mid 帧的偏差",
                                          "mid 帧的偏差 / dx", "路径 C − 路径 A"], centre_rows))
        out = REPORT_ROOT / "stream_function_check"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8",
                                          newline="\n")
        print(f"[stream_function_check] tables written ({run.run_id}): {len(summary['analytic'])} analytic "
              f"fields, {len(summary['runs'])} runs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("grids", "tables"))
    parser.add_argument("--run", default=None, help="grids: the cavity run (logs/wall_crossing/cavity/<run>)")
    parser.add_argument("--analytic", action="store_true",
                        help="grids: the analytic fields on the run's particles instead of its own velocities")
    arguments = parser.parse_args()
    if arguments.action == "grids":
        if arguments.run is None:
            parser.error("grids needs --run")
        return make_grids(arguments.run, arguments.analytic)
    return tables()


if __name__ == "__main__":
    sys.exit(main())
