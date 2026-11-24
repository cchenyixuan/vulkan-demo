"""
cavity3d_smoke.py - claude/wall_store_report.md section 14.2 (round 8): the smoke tests of the 3-D cavity at
h/dx = 3.5 and 4 (runs S1-S6 of box_run.py --diagnostics --lines-time under stress.py queue).

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_smoke --root logs/wall_store/round8/smoke

For every run: the blow-up criteria of the brief (non-finite values; largest speed > 1.5 U; particle count, overflow,
far-migration and zombie counters; fluid outside the box; smallest pair distance < 0.25 dx; the specific kinetic
energy 1/2 <|v|^2> of h/dx = 3.5 against 4 at the same N, at t = 3 (and t = 10 for N = 50), more than 5 % apart), and
the correction matrix of the fluid particles near the corners, edges and faces.

Kinetic energy: the specific one, 1/2 <|v|^2> over the fluid particles (series.csv speed_root_mean_square^2 / 2),
because the particle mass is rest density x the calibrated volume, which depends on h/dx (1.1288 dx^3 at 3.5,
1.0830 dx^3 at 4): the total kinetic energy of the two differs by 4.2 % at the same flow.

Correction matrix (CPU, from the run's final.npz and from the initial lattice): M_i = sum_j V_j (x_j - x_i) (x) grad_i
W_ij = sum_j V_j (-W'(r) / r) (x_j - x_i) (x) (x_j - x_i), Wendland C4 with support h, V_j = m_j / rho_j; for crossing
over the fluid neighbours only (A_f, the matrix of the one-sided gradients, wall_boundary.glsl
wall_crossing_invert_fluid_matrix), for simple over the fluid and wall particles (the walls are in the particle pool).
The solver adds xi = 0.001 to the diagonal, falls back when |det| <= 1e-4 and scales the inverse down to a Frobenius
norm of 10. Classes by the number of box faces within h of the particle: 0 interior (a random 20,000), 1 face,
2 edge, 3 corner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

XI = 1.0e-3
DETERMINANT_THRESHOLD = 1.0e-4
FROBENIUS_MAXIMUM = 10.0
CLASS_NAMES = {0: "interior", 1: "face", 2: "edge", 3: "corner"}


def kernel_derivative_over_r(distance: np.ndarray, support: float) -> np.ndarray:
    """-W'(r) / r of the 3-D Wendland C4 kernel with support h (>= 0)."""
    q = np.clip(distance / support, 0.0, 1.0)
    constant = 495.0 / (32.0 * math.pi * support ** 3)
    derivative = constant / support * (-56.0 / 3.0) * q * (1.0 - q) ** 5 * (5.0 * q + 1.0)   # dW/dr
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(distance > 0.0, -derivative / distance, 0.0)


def correction_matrices(positions: np.ndarray, volumes: np.ndarray, chosen: np.ndarray, support: float,
                        neighbour_positions: np.ndarray, neighbour_volumes: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """M_i (len(chosen), 3, 3) of the particles positions[chosen] over the neighbour set."""
    tree = cKDTree(neighbour_positions)
    out = np.zeros((chosen.size, 3, 3))
    for start in range(0, chosen.size, chunk):
        block = chosen[start:start + chunk]
        lists = tree.query_ball_point(positions[block], r=support, workers=-1, return_sorted=False)
        lengths = np.fromiter((len(item) for item in lists), dtype=np.int64, count=len(lists))
        neighbour = np.concatenate([np.asarray(item, dtype=np.int64) for item in lists])
        owner = np.repeat(np.arange(block.size), lengths)
        offset = neighbour_positions[neighbour] - positions[block][owner]
        distance = np.sqrt((offset ** 2).sum(axis=1))
        weight = neighbour_volumes[neighbour] * kernel_derivative_over_r(distance, support)
        for row in range(3):
            for column in range(row, 3):
                entry = np.bincount(owner, weights=weight * offset[:, row] * offset[:, column], minlength=block.size)
                out[start:start + block.size, row, column] = entry
                out[start:start + block.size, column, row] = entry
    return out


def matrix_statistics(matrices: np.ndarray) -> dict:
    regularised = matrices + XI * np.eye(3)[None]
    eigenvalues = np.linalg.eigvalsh(regularised)
    determinant = np.linalg.det(regularised)
    fallback = np.abs(determinant) <= DETERMINANT_THRESHOLD
    inverse_norm = np.full(matrices.shape[0], np.inf)
    good = ~fallback
    inverse_norm[good] = np.linalg.norm(np.linalg.inv(regularised[good]), axis=(1, 2))
    smallest = eigenvalues[:, 0]
    condition = eigenvalues[:, 2] / np.maximum(eigenvalues[:, 0], 1e-300)
    return {"count": int(matrices.shape[0]),
            "smallest_eigenvalue": {"min": float(smallest.min()), "p01": float(np.percentile(smallest, 1)),
                                    "p50": float(np.median(smallest))},
            "condition": {"p50": float(np.median(condition)), "p99": float(np.percentile(condition, 99)),
                          "max": float(condition.max())},
            "determinant_min": float(determinant.min()),
            "fallback_count": int(fallback.sum()),
            "frobenius_clamped_count": int((inverse_norm > FROBENIUS_MAXIMUM).sum()),
            "inverse_norm_max": float(inverse_norm[good].max()) if good.any() else float("inf")}


def case_parts(case_yaml: pathlib.Path):
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing import cavity3d_lines
    from experiment.wall_crossing.hydrostatic import initial_box
    case = load_case_v7(case_yaml)
    lower, upper, radius, _dimension = initial_box(case)
    walls = cavity3d_lines.wall_particles_of_case(case_yaml)
    return case, lower, upper, 2.0 * radius, walls


def correction_report(case_yaml: pathlib.Path, state: dict, crossing: bool, seed: int = 1) -> dict:
    """Per class, matrix_statistics of the chosen fluid particles of state (positions, masses, densities)."""
    from experiment.wall_crossing.cavity3d_lines import wall_count_within
    case, lower, upper, spacing, (wall_positions, _wall_velocities, wall_volumes) = case_parts(case_yaml)
    support = float(case.physics.smoothing_length)
    positions = state["positions"]
    volumes = state["masses"] / state["densities"]
    classes = wall_count_within(positions, lower, upper, support)
    rng = np.random.default_rng(seed)
    interior = np.flatnonzero(classes == 0)
    chosen_by_class = {0: np.sort(rng.choice(interior, size=min(20000, interior.size), replace=False))}
    for count in (1, 2, 3):
        chosen_by_class[count] = np.flatnonzero(classes == count)
    if crossing:
        neighbour_positions, neighbour_volumes = positions, volumes
    else:
        neighbour_positions = np.concatenate([positions, wall_positions])
        neighbour_volumes = np.concatenate([volumes, wall_volumes])
    out = {}
    for count, chosen in chosen_by_class.items():
        matrices = correction_matrices(positions, volumes, chosen, support, neighbour_positions, neighbour_volumes)
        statistics = matrix_statistics(matrices)
        # where the worst one is
        smallest = np.linalg.eigvalsh(matrices + XI * np.eye(3)[None])[:, 0]
        worst = chosen[int(np.argmin(smallest))]
        statistics["worst_position"] = positions[worst].tolist()
        statistics["worst_distance_to_faces_over_dx"] = sorted(
            float(value) for value in np.minimum(positions[worst] - lower, upper - positions[worst]) / spacing)
        out[CLASS_NAMES[count]] = statistics
    return out


def initial_state(case_yaml: pathlib.Path) -> dict:
    case, *_ = case_parts(case_yaml)
    from experiment.v7.utils.case_v7 import KIND_FLUID
    kinds = np.asarray([material.kind for material in case.materials])
    groups = np.asarray(case.initial.material_group)
    fluid = kinds[groups] == KIND_FLUID
    material = next(material for material in case.materials if material.kind == KIND_FLUID)
    count = int(fluid.sum())
    return {"positions": np.asarray(case.initial.positions, dtype=np.float64)[fluid],
            "masses": np.full(count, float(material.rest_density) * float(material.volume)),
            "densities": np.full(count, float(material.rest_density))}


def read_series(path: pathlib.Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: (float(value) if value not in ("", None) else float("nan")) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def specific_energy_at(series: list, time_value: float) -> float:
    times = np.asarray([row["time"] for row in series])
    energy = 0.5 * np.asarray([row["speed_root_mean_square"] for row in series]) ** 2
    if time_value > times.max() + 1e-9:
        return float("nan")
    return float(np.interp(time_value, times, energy))


def run_summary(run_directory: pathlib.Path) -> dict:
    series = read_series(run_directory / "series.csv")
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8")) \
        if (run_directory / "summary.json").exists() else None
    stop = json.loads((run_directory / "watch_stop.json").read_text(encoding="utf-8")) \
        if (run_directory / "watch_stop.json").exists() else None
    defrag = read_series(run_directory / "defrag.csv") if (run_directory / "defrag.csv").exists() else []
    first = series[0]
    finite = all(math.isfinite(row["speed_maximum"]) and math.isfinite(row["density_minimum"]) for row in series)
    fastest = max(series, key=lambda row: row["speed_maximum"])
    accelerated = max(series, key=lambda row: row["acceleration_maximum"])
    after_start = [row for row in series if row["time"] > 0.0]
    record = {
        "run": run_directory.name, "completed": summary is not None, "stop": stop["reason"] if stop else None,
        "time_reached": series[-1]["time"], "samples": len(series),
        "finite": finite,
        "speed_maximum": fastest["speed_maximum"], "speed_maximum_time": fastest["time"],
        "speed_maximum_position": [fastest.get("fastest_x"), fastest.get("fastest_y"), fastest.get("fastest_z")],
        "alive_readback": sorted({int(row["alive_readback"]) for row in series}),
        "alive_gpu": sorted({int(row["alive_gpu"]) for row in series}),
        "zombie_samples": int(sum(row["alive_gpu"] != row["alive_readback"] for row in series)),
        "overflow_maximum": max(row["overflow"] for row in series),
        "overflow_defrag_maximum": max((row["overflow"] for row in defrag), default=float("nan")),
        "far_migrations": max(row["far_migrations"] for row in series),
        "far_migrations_defrag": max((row["far_migration"] for row in defrag), default=float("nan")),
        "stamp_errors": max(row["stamp_errors"] for row in series),
        "outside_box_maximum": max(row["outside_box_count"] for row in series),
        "below_floor_maximum": max(row["below_floor_count"] for row in series),
        "wall_distance_minimum_over_spacing": min(row["wall_distance_minimum_over_spacing"] for row in series),
        "pair_distance_minimum_over_spacing": min(row["pair_distance_minimum_over_spacing"] for row in after_start),
        "wall_particle_distance_minimum_over_spacing": min(row["wall_particle_distance_minimum_over_spacing"]
                                                           for row in after_start),
        "correction_fallbacks": series[-1]["correction_fallbacks"],
        "gradient_fallbacks": series[-1]["gradient_fallbacks"],
        "acceleration_maximum": accelerated["acceleration_maximum"],
        "acceleration_maximum_time": accelerated["time"],
        "acceleration_maximum_position": [accelerated["acceleration_maximum_x"], accelerated["acceleration_maximum_y"],
                                          accelerated["acceleration_maximum_z"]],
        "acceleration_maximum_faces_within_h": int(accelerated["acceleration_maximum_faces_within_h"]),
        "acceleration_faces_within_h_counts": {str(int(key)): int(sum(row["acceleration_maximum_faces_within_h"] == key
                                                                      for row in after_start)) for key in range(4)},
        "acceleration_p999_maximum": max(row["acceleration_99_9_percentile"] for row in after_start),
        "specific_energy": {str(t): specific_energy_at(series, t) for t in (1.0, 2.0, 3.0, 5.0, 10.0)},
        "steps_per_second_defrag_median": float(np.median([row["steps_per_second"] for row in defrag[1:]]))
        if len(defrag) > 1 else float("nan"),
        "initial_alive": int(first["alive_readback"]),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs/wall_store/round8/smoke")
    parser.add_argument("--out", default="logs/wall_store/round8/smoke_summary.json")
    parser.add_argument("--skip-correction", action="store_true")
    arguments = parser.parse_args()
    root = (REPOSITORY_ROOT / arguments.root).resolve()
    records = {}
    for run_directory in sorted(root.glob("gpu*/box_run/*_S*")):
        label = run_directory.name.split("_", 1)[1]
        record = run_summary(run_directory)
        record["directory"] = str(run_directory.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
        if not arguments.skip_correction:
            plan_case = json.loads((run_directory / "wrapper.jsonl").read_text(encoding="utf-8").splitlines()[0]) \
                if (run_directory / "wrapper.jsonl").exists() else None
            case_name = {"S1": "n50_hdx3p5_crossing", "S2": "n50_hdx4_crossing", "S3": "n100_hdx3p5_crossing",
                         "S4": "n100_hdx4_crossing", "S5": "n100_hdx3p5_simple", "S6": "n100_hdx4_simple"}[label[:2]]
            case_yaml = REPOSITORY_ROOT / "cases" / f"cavity3d_re1000_{case_name}" / "case.yaml"
            crossing = case_name.endswith("crossing")
            record["case"] = str(case_yaml.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
            record["correction_initial"] = correction_report(case_yaml, initial_state(case_yaml), crossing)
            state_file = run_directory / ("final.npz" if (run_directory / "final.npz").exists() else "stop_state.npz")
            if state_file.exists():
                with np.load(state_file) as archive:
                    state = {"positions": archive["positions"].astype(np.float64),
                             "densities": archive["densities"].astype(np.float64),
                             "masses": (archive["masses"].astype(np.float64) if "masses" in archive.files
                                        else np.full(archive["positions"].shape[0],
                                                     float(initial_state(case_yaml)["masses"][0])))}
                    if "accelerations" in archive.files:
                        acceleration = np.sqrt((archive["accelerations"].astype(np.float64) ** 2).sum(axis=1))
                        top = np.argsort(acceleration)[-20:][::-1]
                        from experiment.wall_crossing.cavity3d_lines import wall_count_within
                        _case, lower, upper, spacing, _walls = case_parts(case_yaml)
                        where = archive["acceleration_positions"].astype(np.float64)[top]
                        record["final_top_accelerations"] = {
                            "values": acceleration[top].tolist(),
                            "faces_within_h": wall_count_within(where, lower, upper,
                                                                float(_case.physics.smoothing_length)).tolist(),
                            "positions": where.tolist(),
                            "distance_to_nearest_face_over_dx": (np.minimum(where - lower, upper - where).min(axis=1)
                                                                 / spacing).tolist(),
                            "median": float(np.median(acceleration))}
                record["correction_final"] = correction_report(case_yaml, state, crossing)
                record["correction_final_state"] = state_file.name
        records[label] = record
        print(f"{label}: done", flush=True)
    # 3.5 against 4 at the same N
    pairs = {"N = 50 crossing": ("S1", "S2"), "N = 100 crossing": ("S3", "S4"), "N = 100 simple": ("S5", "S6")}
    comparison = {}
    for name, (first, second) in pairs.items():
        a = next((value for key, value in records.items() if key.startswith(first + "_")), None)
        b = next((value for key, value in records.items() if key.startswith(second + "_")), None)
        if a is None or b is None:
            continue
        comparison[name] = {t: (a["specific_energy"][t] - b["specific_energy"][t]) / b["specific_energy"][t]
                            for t in a["specific_energy"] if math.isfinite(a["specific_energy"][t])
                            and math.isfinite(b["specific_energy"][t])}
    output = {"runs": records, "energy_relative_difference_3p5_vs_4": comparison}
    out = REPOSITORY_ROOT / arguments.out
    out.write_text(json.dumps(output, indent=1), encoding="utf-8", newline="\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
