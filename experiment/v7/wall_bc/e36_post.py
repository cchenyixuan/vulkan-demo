"""e36_post.py - E36 post-processing (CPU only): coordinate frames, transport velocity, convergence.

Runs are cavity_runner run directories, cut at --t-stop and averaged over the last --average-span before it (the
window of e36_compare / cavity_analysis.analyse_run).

frames: the wall-row frame (unit square = centre lines of the innermost wall / lid rows, L = 1 + 2 dx: the frame of
    the main series and of e36_compare) against the interface frame (unit square = the midline between the outermost
    fluid row and the first wall row, L = 1 + dx; cavity_analysis 'mid'). psi_min and the two bottom secondary
    vortices from the time-averaged MLS grid of u (cavity_analysis.stream_function_minimum's grid), psi integrated
    upward from the bottom edge of each frame and divided by that frame's L; the centre-line extrema of the
    time-averaged dense lines (the values are physical velocities and do not change between frames, the positions
    are mapped); the L2 errors at the Marchi 2021 points (cavity_analysis: spline of the dense line in each frame).
transport: on the run's checkpoints with t in [t_stop - average_span, t_stop + 1] (full states, with the PST shift
    the next step applies), psi_min and the three centre-line extrema from the stored velocity u and from the
    transport velocity u + shift / dt (the particle displacement of a step over dt), the same states and sampling
    (wall-row frame); the difference is what the stored velocity's bias does to the comparison. Also the
    mass-weighted mean fluid velocity of both.
series: per series label::run250,run500,run1000 the observed order p = log2((f250 - f500) / (f500 - f1000)) and the
    Richardson limit f_inf = f1000 + (f1000 - f500) / (2^p - 1) of psi_min and the three extrema (wall-row frame),
    against Marchi 2021; with two runs only the values are listed.

Outputs <out>/post_summary.json and <out>/post_tables.md.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.e36_post --out docs/wall_bc/e36_post \\
        --run "v6 K=1::C:/.../logs/e36/runs/n250_k1_v6" ... --series "v6 K=2::run250,run500,run1000"
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_analysis, cavity_reference, cavity_sampling as sampling  # noqa: E402

GRID_POINTS = cavity_analysis.GRID_POINTS
EXTREMA_SEARCH = cavity_analysis.EXTREMA_SEARCH
SECONDARY_WINDOWS = {"BR1": ((0.6, 1.0), (0.0, 0.45)), "BL1": ((0.0, 0.35), (0.0, 0.35))}
REFERENCE_KEYS = {"psi_min": "psi_min", "u_min": "u_min", "v_max": "v_max", "v_min": "v_min"}
DENSE_POINTS = 1001


def frame_half_width(frame: str, spacing: float) -> float:
    return {"wall": 0.5 + spacing, "mid": 0.5 + 0.5 * spacing}[frame]


def load(run_dir: pathlib.Path, t_stop: float) -> dict:
    run = cavity_analysis.load_run(run_dir.name, root=run_dir.parent)
    if run is None:
        raise SystemExit(f"{run_dir}: no samples")
    keep = run["times"] <= t_stop + 1e-9
    run["rows"] = [row for row, flag in zip(run["rows"], keep) if flag]
    run["times"] = run["times"][keep]
    run["kinetic_energy"] = run["kinetic_energy"][keep]
    run["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    return run


def u_grid_from(particle_sets: list, support: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the MLS u on the 513^2 grid spanning the wall rows over ``particle_sets`` (ParticleSet list)."""
    half = 0.5 + spacing
    axis = np.linspace(-half, half, GRID_POINTS)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    total = np.zeros(points.shape[0])
    for particles in particle_sets:
        tree = sampling.cKDTree(particles.positions)
        for start in range(0, points.shape[0], 40000):
            chunk = slice(start, start + 40000)
            total[chunk] += sampling.interpolate(particles, points[chunk], support, tree=tree)["mls"][:, 0]
    return axis, (total / len(particle_sets)).reshape(GRID_POINTS, GRID_POINTS)


def vortices(axis: np.ndarray, u_grid: np.ndarray, spacing: float, frame: str) -> dict:
    """psi_min (primary, |x|, |y| < 0.3 physical) and the BR1 / BL1 maxima in ``frame``; psi from the frame's
    bottom edge, divided by the frame's L; positions in the frame's unit square."""
    half_wall = 0.5 + spacing
    psi_physical = sampling.stream_function(u_grid, axis + half_wall)        # from the bottom wall row
    half = frame_half_width(frame, spacing)
    if frame != "wall":
        bottom = np.array([np.interp(-half, axis, psi_physical[:, column]) for column in range(axis.size)])
        psi_physical = psi_physical - bottom[None, :]
    length = 2.0 * half
    xx, yy = np.meshgrid(axis, axis)
    interior = (np.abs(xx) < 0.3) & (np.abs(yy) < 0.3)
    result = {}
    x_min, y_min, psi_min = sampling.refine_grid_minimum(axis, axis, np.where(interior, psi_physical, np.nan))
    result["primary"] = {"psi": psi_min / length, "x": (x_min + half) / length, "y": (y_min + half) / length}
    reference = (axis + half) / length
    for name, ((x0, x1), (y0, y1)) in SECONDARY_WINDOWS.items():
        box = ((reference[None, :] > x0) & (reference[None, :] < x1)
               & (reference[:, None] > y0) & (reference[:, None] < y1))
        x_max, y_max, minus_psi = sampling.refine_grid_minimum(axis, axis, np.where(box, -psi_physical, np.nan))
        result[name] = {"psi": -minus_psi / length, "x": (x_max + half) / length, "y": (y_max + half) / length}
    return result


def snapshot_sets(run: dict, window: tuple[float, float]) -> list:
    dt = float(run["meta"]["dt"])
    sets = []
    for path in sorted((run["dir"] / "snapshots").glob("t*.npz")):
        if window[0] - 1e-9 <= int(path.stem[1:]) * dt <= window[1] + 1e-9:
            with np.load(path) as archive:
                sets.append(sampling.ParticleSet(archive["positions"].astype(np.float64),
                                                 archive["velocities"].astype(np.float64),
                                                 archive["volumes"].astype(np.float64)))
    return sets


def frames_analysis(run: dict, average_span: float) -> dict:
    analysis = cavity_analysis.analyse_run(run, average_span, 1e-3, 10.0, with_stream_function=False)
    window = analysis["window"]
    spacing, support = run["spacing"], float(run["meta"]["support_radius"])
    axis, u_grid = u_grid_from(snapshot_sets(run, window), support, spacing)
    wall_frame = sampling.Frame.make("wall", spacing)
    out = {"window": window, "frames": {}}
    for frame in ("wall", "mid"):
        half = frame_half_width(frame, spacing)
        extrema = {}
        for name, entry in analysis["extrema"].items():
            physical = float(wall_frame.to_physical(entry["position_wall"]))
            extrema[name] = {"value": entry["value"], "position": (physical + half) / (2.0 * half)}
        out["frames"][frame] = {
            "length": 2.0 * half, "vortices": vortices(axis, u_grid, spacing, frame), "extrema": extrema,
            "errors": {line: {key: analysis["errors"][(frame, "dense", line)][key] for key in ("L2", "L2_relative", "Linf")}
                       for line in ("u", "v")}}
    return out


def centreline_extrema(particles, support: float, spacing: float) -> dict:
    frame = sampling.Frame.make("wall", spacing)
    dense = np.linspace(0.0, 1.0, DENSE_POINTS)
    sampled = sampling.sample_centerlines(particles, support, {
        "u_dense": ("u", sampling.centerline_points(frame, dense, "u")),
        "v_dense": ("v", sampling.centerline_points(frame, dense, "v"))})
    result = {}
    for name, (line, kind, search) in EXTREMA_SEARCH.items():
        position, value = sampling.refine_extremum(dense, sampled[f"{line}_dense"]["mls"], kind, search=search)
        result[name] = {"value": value, "position": position}
    return result


def transport_analysis(run: dict, t_stop: float, average_span: float) -> dict:
    meta = run["meta"]
    dt = float(meta["dt"])
    spacing, support = run["spacing"], float(meta["support_radius"])
    fluid_groups = meta["fluid_groups"]
    states = []
    for path in sorted((run["dir"] / "checkpoints").glob("c*.npz")):
        time_value = int(path.stem[1:]) * dt
        if t_stop - average_span - 1e-9 <= time_value <= t_stop + 1.0 + 1e-9:
            states.append((time_value, path))
    per_state = []
    for time_value, path in states:
        with np.load(path) as archive:
            positions = archive["position_voxel_id"][:, :2].astype(np.float64)
            velocity_mass = archive["velocity_mass"].astype(np.float64)
            density = archive["density_pressure"][:, 0].astype(np.float64) + float(archive["_stored_density_offset"])
            shift = archive["shift"][:, :2].astype(np.float64)
            material = archive["material"]
        volumes = velocity_mass[:, 3] / density
        stored = velocity_mass[:, :2]
        transport = stored + shift / dt
        fluid = np.isin(material, fluid_groups)
        mass = velocity_mass[fluid, 3]
        entry = {"time": time_value}
        for name, velocities in (("stored", stored), ("transport", transport)):
            particles = sampling.ParticleSet(positions, velocities, volumes)
            axis, u_grid = u_grid_from([particles], support, spacing)
            mean_velocity = (mass[:, None] * velocities[fluid]).sum(axis=0) / mass.sum()
            entry[name] = {"psi_min": vortices(axis, u_grid, spacing, "wall")["primary"],
                           "extrema": centreline_extrema(particles, support, spacing),
                           "mean_velocity": [float(value) for value in mean_velocity]}
        per_state.append(entry)
    return {"states": per_state}


def mean_of(states: list, kind: str, quantity: str) -> float:
    if quantity == "psi_min":
        return float(np.mean([state[kind]["psi_min"]["psi"] for state in states]))
    return float(np.mean([state[kind]["extrema"][quantity]["value"] for state in states]))


def deficit(value: float, reference: float) -> float:
    return 100.0 * (abs(value) - abs(reference)) / abs(reference)


def series_analysis(values: dict) -> dict:
    """values: {resolution: {quantity: value}} for 2 or 3 resolutions (each double the last)."""
    reference = cavity_reference.marchi2021()["extrema"]
    resolutions = sorted(values)
    result = {}
    for quantity, key in REFERENCE_KEYS.items():
        series = [values[resolution][quantity] for resolution in resolutions]
        reference_value = reference[key][0]
        entry = {"values": dict(zip(map(str, resolutions), series)),
                 "deficits": dict(zip(map(str, resolutions), [deficit(value, reference_value) for value in series])),
                 "reference": reference_value}
        if len(series) == 3:
            first, second = series[0] - series[1], series[1] - series[2]
            if first != 0 and second != 0 and first / second > 0:
                order = math.log(first / second) / math.log(2.0)
                limit = series[2] + (series[2] - series[1]) / (2.0 ** order - 1.0)
                entry.update({"order": order, "limit": limit, "limit_deficit": deficit(limit, reference_value)})
            else:
                entry.update({"order": None, "limit": None, "limit_deficit": None,
                              "note": "non-monotone: differences change sign"})
        result[quantity] = entry
    return result


def markdown(summary: dict) -> str:
    reference = cavity_reference.marchi2021()["extrema"]
    lines = []
    if summary["frames"]:
        lines += ["| run | 坐标系(L) | ψ_min(相对 Marchi) | 涡心 (x, y) | BR1 ψ | BL1 ψ | u_min 位置 | v_max 位置 | v_min 位置 | 相对 L2 u / v |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for label, entry in summary["frames"].items():
            for frame, text in (("wall", "壁行"), ("mid", "界面")):
                item = entry["frames"][frame]
                primary = item["vortices"]["primary"]
                lines.append(
                    f"| {label} | {text}({item['length']:.4f}) | {primary['psi']:.5f}({deficit(primary['psi'], reference['psi_min'][0]):+.2f} %) | "
                    f"({primary['x']:.4f}, {primary['y']:.4f}) | {item['vortices']['BR1']['psi']:.3e} | {item['vortices']['BL1']['psi']:.3e} | "
                    f"{item['extrema']['u_min']['position']:.4f} | {item['extrema']['v_max']['position']:.4f} | "
                    f"{item['extrema']['v_min']['position']:.4f} | {100 * item['errors']['u']['L2_relative']:.2f} % / "
                    f"{100 * item['errors']['v']['L2_relative']:.2f} % |")
        lines.append("")
    if summary["transport"]:
        lines += ["| run | 检查点 t | 量 | 存储速度 | 输运速度 u + δr/dt | 差(百分点,相对 Marchi) |",
                  "|---|---|---|---|---|---|"]
        for label, entry in summary["transport"].items():
            states = entry["states"]
            times = ", ".join(f"{state['time']:.1f}" for state in states)
            for quantity in ("psi_min", "u_min", "v_max", "v_min"):
                reference_value = reference[REFERENCE_KEYS[quantity]][0]
                stored, transport = mean_of(states, "stored", quantity), mean_of(states, "transport", quantity)
                lines.append(f"| {label} | {times} | {quantity} | {stored:.5f}({deficit(stored, reference_value):+.2f} %) | "
                             f"{transport:.5f}({deficit(transport, reference_value):+.2f} %) | "
                             f"{deficit(transport, reference_value) - deficit(stored, reference_value):+.2f} |")
            stored_mean = np.mean([state["stored"]["mean_velocity"] for state in states], axis=0)
            transport_mean = np.mean([state["transport"]["mean_velocity"] for state in states], axis=0)
            lines.append(f"| {label} | {times} | 质量平均速度 (ū, v̄) | ({stored_mean[0]:+.2e}, {stored_mean[1]:+.2e}) | "
                         f"({transport_mean[0]:+.2e}, {transport_mean[1]:+.2e}) | |")
        lines.append("")
    if summary["series"]:
        lines += ["| 系列 | 量 | 250² | 500² | 1000² | 收敛阶 p | 外推极限(相对 Marchi) | Marchi |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, entry in summary["series"].items():
            for quantity, item in entry.items():
                cells = []
                for resolution in ("250", "500", "1000"):
                    if resolution in item["values"]:
                        cells.append(f"{item['values'][resolution]:.5f}({item['deficits'][resolution]:+.2f} %)")
                    else:
                        cells.append("–")
                order = f"{item['order']:.2f}" if item.get("order") is not None else ("非单调" if "note" in item else "–")
                limit = (f"{item['limit']:.5f}({item['limit_deficit']:+.2f} %)" if item.get("limit") is not None else "–")
                lines.append(f"| {label} | {quantity} | {' | '.join(cells)} | {order} | {limit} | {item['reference']:.5f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="label::run directory (frames + transport)")
    parser.add_argument("--series", action="append", default=[],
                        help="label::dir250,dir500[,dir1000] (convergence; values from the frames analysis)")
    parser.add_argument("--skip-transport", action="store_true")
    parser.add_argument("--t-stop", type=float, default=100.0)
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"frames": {}, "transport": {}, "series": {}}
    cache = {}

    def frames_of(path: str) -> dict:
        key = str(pathlib.Path(path).resolve())
        if key not in cache:
            print(f"[e36_post] frames {key}", flush=True)
            cache[key] = frames_analysis(load(pathlib.Path(key), arguments.t_stop), arguments.average_span)
        return cache[key]

    for entry in arguments.run:
        label, path = entry.rsplit("::", 1)
        summary["frames"][label] = frames_of(path)
        if not arguments.skip_transport:
            print(f"[e36_post] transport {label}", flush=True)
            summary["transport"][label] = transport_analysis(load(pathlib.Path(path).resolve(), arguments.t_stop),
                                                             arguments.t_stop, arguments.average_span)
    for entry in arguments.series:
        label, paths = entry.rsplit("::", 1)
        values = {}
        for resolution, path in zip((250, 500, 1000), paths.split(",")):
            frame = frames_of(path)["frames"]["wall"]
            values[resolution] = {"psi_min": frame["vortices"]["primary"]["psi"],
                                  **{name: frame["extrema"][name]["value"] for name in ("u_min", "v_max", "v_min")}}
        summary["series"][label] = series_analysis(values)
    (out / "post_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    text = markdown(summary)
    (out / "post_tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
