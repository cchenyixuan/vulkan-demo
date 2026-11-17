"""
hydrostatic.py - a closed box of fluid at rest under gravity, every wall mode: does the fluid stay at rest, and
does the pressure settle on the hydrostatic solution?

    # the cases (2-D: the n250 cavity geometry with the lid at rest; 3-D: a generated box)
    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic make-cases
    # one run
    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic run --case cases/wall_crossing/hydrostatic_2d_n250_gy_crossing/case.yaml \
        --name 2d_n250_gy_crossing --device 1 --uuid ae137c668a40f5acaab90a83f7cda175 --time 10
    # the table of every finished run
    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic table

The run: the solver's production K = 1 path (chain_runner.run_chain: a one-slab chain, pipelined at depth 2, a
defrag every defrag cadence), from uniform density and zero velocity. At every sample (a multiple of the defrag
cadence) the alive fluid particles are read back: kinetic energy, largest and rms speed, density range. At the end:
  - the pressure against the hydrostatic solution p = p_level + rho0 g . (r - r_centre) (a closed box fixes the
    level through the mass, so the level is fitted; both the raw error with the level of the box centre taken
    from the fit and the level-free rms are reported, in units of rho0 |g| H, H = the box extent along g);
  - the speed in layers of wall distance (phi / dx in [0,1), [1,2), [2,3), [3,5), >= 5), in units of sqrt(|g| H).
The box (wall planes) of every mode is the one wall_boundary crossing would take: the fluid particles' initial
bounding box widened by one particle radius.

Output: logs/wall_crossing/hydrostatic/<name>/series.csv, final.npz, summary.json; one runs.jsonl record per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                      # noqa: E402

CASE_ROOT = REPOSITORY_ROOT / "cases" / "wall_crossing"
MODES = {"simple": ("simple", None), "adami": ("adami", None), "crossing": ("crossing", "extrapolate"),
         "crossing_wall_velocity": ("crossing", "wall_velocity")}
# 2-D only, DIAGNOSTIC (numerics.wall_crossing.delta_wall, not part of the mode): mode name -> the value
DIAGNOSTIC_DELTA_WALL_2D = {"crossing_delta_wall_body_force": "body_force"}
# |g| H / c0^2 = 1e-3 with c0 = 100 and H = 1: |g| = 10
GRAVITY_MAGNITUDE = 10.0
DIRECTIONS_2D = {
    "gy": (0.0, -1.0, 0.0),                                             # along -y
    "g30": (-np.sin(np.radians(30.0)), -np.cos(np.radians(30.0)), 0.0),  # 30 degrees from -y
}
DIRECTIONS_3D = {
    "gy": (0.0, -1.0, 0.0),
    "gtilt": tuple(np.array([-0.4, -0.8, 0.45]) / np.linalg.norm([-0.4, -0.8, 0.45])),   # off two axes
}
LAYER_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, np.inf)


def make_cases(force: bool = False) -> list:
    """2-D: cases/wall_crossing/hydrostatic_2d_n250_<direction>_<mode>: the main-series n250 cavity case with the
    lid's material replaced by the static wall's, gravity added, and the wall option. 3-D:
    cases/wall_crossing/hydrostatic_3d_n48_<direction>_<mode>: a generated box of 48^3 fluid particles.
    Existing case files are left alone unless force (these directories hold only what this function writes)."""
    from experiment.wall_crossing.box_case import BoxCaseSpecification, write_box_case

    written = []
    base = REPOSITORY_ROOT / "cases" / "lid_driven_cavity_2d_n250_xi0p001_eps0p0025" / "case.yaml"
    base_text = base.read_text(encoding="utf-8")
    replacements = (
        ("  gravity: [0.0, 0.0, 0.0]\n", None),
        ("material_library: ../../materials/standard.yaml", "material_library: ../../../materials/standard.yaml"),
        ("    - {file: ../lid_driven_cavity_2d_n250/wall_top.obj, material: stick_lid}",
         "    - {file: ../../lid_driven_cavity_2d_n250/wall_top.obj, material: stick_wall}   # the lid at rest"),
        ("  frame: ../lid_driven_cavity_2d_n250/frame.obj", "  frame: ../../lid_driven_cavity_2d_n250/frame.obj"),
        ("    - {file: ../lid_driven_cavity_2d_n250/domain.obj,   material: stick_water}",
         "    - {file: ../../lid_driven_cavity_2d_n250/domain.obj,   material: stick_water}"),
        ("    - {file: ../lid_driven_cavity_2d_n250/wall.obj,     material: stick_wall}",
         "    - {file: ../../lid_driven_cavity_2d_n250/wall.obj,     material: stick_wall}"),
        ("  use_prefix_sum_defrag: false\n", None),
    )
    for old, _ in replacements:
        if base_text.count(old) != 1:
            raise ValueError(f"{base}: expected exactly one {old!r}")
    for direction, unit in DIRECTIONS_2D.items():
        gravity = [float(round(GRAVITY_MAGNITUDE * float(component), 9)) for component in unit]
        modes_2d = [(mode, wall_boundary, continuity, None) for mode, (wall_boundary, continuity) in MODES.items()]
        modes_2d += [(mode, "crossing", "extrapolate", delta_wall)
                     for mode, delta_wall in DIAGNOSTIC_DELTA_WALL_2D.items()]
        for mode, wall_boundary, continuity, delta_wall in modes_2d:
            directory = CASE_ROOT / f"hydrostatic_2d_n250_{direction}_{mode}"
            target = directory / "case.yaml"
            if target.exists() and not force:
                continue
            option = f"  use_prefix_sum_defrag: false\n  wall_boundary: {wall_boundary}\n"
            if continuity is not None:
                option += f"  wall_crossing:\n    continuity: {continuity}\n"
            if delta_wall is not None:
                option += f"    delta_wall: {delta_wall}   # DIAGNOSTIC, not part of the mode\n"
            text = base_text
            for old, new in replacements:
                if old.startswith("  gravity"):
                    new = f"  gravity: {gravity!r}\n"
                elif old.startswith("  use_prefix_sum_defrag"):
                    new = option
                text = text.replace(old, new)
            header = ("schema_version: 2\n\n"
                      f"# HYDROSTATIC BOX (experiment/wall_crossing/hydrostatic.py): the n250 cavity geometry of\n"
                      f"# cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025 with the lid at rest (stick_wall),\n"
                      f"# gravity {gravity} (|g| H / c0^2 = 1e-3, direction {direction}) and wall_boundary "
                      f"{wall_boundary}" + (f" (continuity {continuity})" if continuity else "")
                      + (f", DIAGNOSTIC delta_wall {delta_wall}" if delta_wall else "") + ".\n")
            if not text.startswith("schema_version: 2\n"):
                raise ValueError(f"{base} does not start with the schema line")
            directory.mkdir(parents=True, exist_ok=True)
            target.write_text(header + text[len("schema_version: 2\n"):], encoding="utf-8", newline="\n")
            written.append(target)
    for direction, unit in DIRECTIONS_3D.items():
        for mode, (wall_boundary, continuity) in MODES.items():
            directory = CASE_ROOT / f"hydrostatic_3d_n48_{direction}_{mode}"
            if (directory / "case.yaml").exists() and not force:
                continue
            # 48^3 fluid particles in the unit cube, h / dx = 4, c0 = 100, the cavity series' viscosity
            specification = BoxCaseSpecification(
                dimension=3, fluid_counts=(48, 48, 48), spacing=1.0 / 48.0, lower=(-0.5, -0.5, -0.5),
                support_ratio=4.0, speed_of_sound=100.0, viscosity=1.0e-3, rest_density=1000.0,
                lid_velocity=(0.0, 0.0, 0.0), wall_boundary=wall_boundary,
                continuity=continuity or "extrapolate", pst_anti=0.0005,
                gravity=tuple(float(round(GRAVITY_MAGNITUDE * float(component), 9)) for component in unit),
                # The lattice starts in uniform free fall, so a whole plane of particles (16 per voxel face at
                # h / dx = 4) enters a voxel in the same step; with gravity off two axes three faces do so at
                # once (48 arrivals): 32 incoming slots overflow there and predict kills the excess.
                max_per_voxel=128, max_incoming=64 if direction == "gtilt" else 32,
                comment=f"hydrostatic box, gravity direction {direction}")
            written.append(write_box_case(directory, specification))
    return written


def initial_box(case) -> tuple:
    """(lower, upper, particle radius, dimension): the fluid's initial bounding box widened by one radius."""
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY
    from experiment.v7.utils.wall_crossing_v7 import WallCrossingSettings, resolve_wall_crossing_box
    kinds = np.asarray([material.kind for material in case.materials])
    is_wall = kinds[case.initial.material_group] == KIND_BOUNDARY
    radius = float(next(material for material in case.materials if material.kind == 0).radius)
    dimension = int(case.physics.dimension)
    settings = resolve_wall_crossing_box(WallCrossingSettings(), case.initial.positions, is_wall, radius, dimension)
    return np.asarray(settings.box_lower), np.asarray(settings.box_upper), radius, dimension


def read_fluid_state(simulator, case) -> dict:
    """Alive fluid particles: positions, velocities, masses, density, pressure (float64 of the float32 values).
    The density is read as stored and the storage offset is added in float64 (0 with the default storage, where
    density_pressure.x is rho itself; rho_ref with V7_DELTA_DENSITY, where it is rho - rho_ref), so the value is
    the true density in either storage and no float32 addition rounds it."""
    from experiment.v7.utils.case_v7 import KIND_FLUID
    raw = simulator.readback_buffers_batch(["position_voxel_id", "velocity_mass", "density_pressure", "material"],
                                           density="stored")
    density_offset = float(simulator.stored_density_offset())
    capacity = case.capacities.total_pool_capacity()
    first = simulator.own_first_pid()
    rows = slice(first, first + case.capacities.own_pool_size)
    position = np.frombuffer(raw["position_voxel_id"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
    velocity = np.frombuffer(raw["velocity_mass"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
    density = np.frombuffer(raw["density_pressure"], dtype=np.float32)[:capacity * 2].reshape(capacity, 2)[rows]
    material = np.frombuffer(raw["material"], dtype=np.uint32)[:capacity][rows]
    kinds = np.asarray([entry.kind for entry in case.materials])
    alive = (velocity[:, 3] > 0.0) & (position[:, 3] > 0.5)
    fluid = alive & (kinds[np.minimum(material, kinds.size - 1)] == KIND_FLUID)
    return {"alive": int(alive.sum()),
            "positions": position[fluid, 0:3].astype(np.float64), "velocities": velocity[fluid, 0:3].astype(np.float64),
            "masses": velocity[fluid, 3].astype(np.float64),
            "densities": density[fluid, 0].astype(np.float64) + density_offset,
            "pressures": density[fluid, 1].astype(np.float64)}


def read_fluid_state_of_chain(simulators: list, slab_cases: list) -> dict:
    """read_fluid_state over every slab of a chain (own rows only), concatenated."""
    parts = [read_fluid_state(simulator, slab) for simulator, slab in zip(simulators, slab_cases)]
    merged = {"alive": sum(part["alive"] for part in parts)}
    for key in ("positions", "velocities", "masses", "densities", "pressures"):
        merged[key] = np.concatenate([part[key] for part in parts])
    return merged


def run_case(arguments) -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.wall_crossing_v7 import box_distance_to_wall
    from experiment.wall_crossing.chain_runner import run_chain

    case_path = (REPOSITORY_ROOT / arguments.case).resolve()
    out = LOG_ROOT / "hydrostatic" / arguments.name
    out.mkdir(parents=True, exist_ok=True)
    case = load_case_v7(case_path)
    lower, upper, radius, dimension = initial_box(case)
    spacing = 2.0 * radius
    gravity = np.asarray(case.physics.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm == 0.0:
        sys.exit(f"{arguments.case}: no gravity")
    unit = gravity / gravity_norm
    extent_along_gravity = float(np.abs(unit[:dimension] * (upper - lower)[:dimension]).sum())
    rest_density = float(next(material for material in case.materials if material.kind == 0).rest_density)
    timestep = float(case.physics.timestep)
    cadence = int(case.numerics.defrag_cadence)
    total_steps = int(np.ceil(arguments.time / timestep / cadence)) * cadence
    sample_steps = cadence * max(1, int(round(arguments.sample_time / timestep / cadence)))
    velocity_scale = float(np.sqrt(gravity_norm * extent_along_gravity))
    pressure_scale = rest_density * gravity_norm * extent_along_gravity
    expected_alive = int(case.initial.positions.shape[0])

    with LoggedRun(kind="hydrostatic", case=str(arguments.case).replace("\\", "/")) as run:
        run.extra["name"] = arguments.name
        series = []
        latest = {}

        def on_sample(step: int, simulators: list, slab_cases: list) -> None:
            state = read_fluid_state_of_chain(simulators, slab_cases)
            speed = np.sqrt((state["velocities"] ** 2).sum(axis=1))
            statuses = [simulator.readback_global_status() for simulator in simulators]
            distance = box_distance_to_wall(state["positions"], lower, upper, dimension)
            series.append({
                "step": step, "time": step * timestep,
                "kinetic_energy": float(0.5 * (state["masses"] * speed ** 2).sum()),
                "speed_maximum": float(speed.max()), "speed_root_mean_square": float(np.sqrt((speed ** 2).mean())),
                "density_minimum": float(state["densities"].min()),
                "density_maximum": float(state["densities"].max()),
                "wall_distance_minimum": float(distance.min()), "alive": state["alive"],
                "correction_fallbacks": int(sum(status["correction_fallback_count"] for status in statuses)),
                "gradient_fallbacks": int(sum(status.get("wall_crossing_gradient_fallback_count", 0)
                                              for status in statuses)),
                "overflow": int(sum(value for status in statuses for key, value in status.items()
                                    if key.startswith("overflow_")))})
            latest["state"] = state
            if state["alive"] != expected_alive or not np.isfinite(speed).all():
                overflow = series[-1]["overflow"]
                raise RuntimeError(
                    f"step {step}: alive {state['alive']} of {expected_alive}"
                    + (f" with {overflow} voxel-list overflow(s) (a capacity of the case, not the walls)"
                       if overflow else " with no voxel-list overflow (particles left through a wall)")
                    + ("" if np.isfinite(speed).all() else "; non-finite velocity"))

        # A run that loses a particle (or produces a non-finite velocity) stops at that sample; what it did until
        # then is still written (series, the state of that sample, the summary with "failed").
        failure = None
        try:
            outcome = run_chain(case, [arguments.device], total_steps, sample_steps, on_sample,
                                expected_uuids=[arguments.uuid] if arguments.uuid else None,
                                application_name="wall_crossing_hydrostatic")
        except RuntimeError as error:
            if "state" not in latest:
                raise
            failure = str(error)
            reached = series[-1]["step"]
            outcome = {"device_uuids": [arguments.uuid.replace("-", "").lower()] if arguments.uuid else [],
                       "elapsed_seconds": float("nan")}
            total_steps = reached
            run.extra["status"] = f"failed: {failure}"
        uuid = outcome["device_uuids"][0] if outcome["device_uuids"] else None
        run.record["gpu_uuids"] = outcome["device_uuids"]
        state = latest["state"]
        elapsed = outcome["elapsed_seconds"]

        # ---- the final state against the hydrostatic solution
        centre = 0.5 * (lower + upper)
        hydrostatic = rest_density * ((state["positions"] - centre) * gravity).sum(axis=1)
        difference = state["pressures"] - hydrostatic
        level = float(difference.mean())
        error = difference - level
        design = np.column_stack([np.ones(len(error)), (state["positions"] - centre)[:, :dimension]])
        coefficients, *_ = np.linalg.lstsq(design, state["pressures"], rcond=None)
        fitted_gradient = np.zeros(3)
        fitted_gradient[:dimension] = coefficients[1:]
        distance = box_distance_to_wall(state["positions"], lower, upper, dimension)
        speed = np.sqrt((state["velocities"] ** 2).sum(axis=1))
        layers = []
        for first_edge, last_edge in zip(LAYER_EDGES[:-1], LAYER_EDGES[1:]):
            mask = (distance >= first_edge * spacing) & (distance < last_edge * spacing)
            if mask.any():
                layers.append({"wall_distance_over_spacing": [first_edge, None if np.isinf(last_edge) else last_edge],
                               "count": int(mask.sum()),
                               "speed_maximum": float(speed[mask].max()),
                               "speed_root_mean_square": float(np.sqrt((speed[mask] ** 2).mean())),
                               "pressure_error_root_mean_square": float(np.sqrt((error[mask] ** 2).mean()))})
        tail = [row for row in series if row["time"] >= 0.8 * series[-1]["time"]]
        summary = {
            "name": arguments.name, "run_id": run.run_id, "case": str(arguments.case).replace("\\", "/"),
            "failed": failure, "alive_at_last_sample": series[-1]["alive"], "particles": expected_alive,
            "wall_boundary": case.numerics.wall_boundary,
            "continuity": case.numerics.wall_crossing.continuity if case.numerics.wall_crossing else None,
            "delta_wall": case.numerics.wall_crossing.delta_wall if case.numerics.wall_crossing else None,
            "dimension": dimension, "gravity": gravity.tolist(), "box_lower": lower.tolist(),
            "box_upper": upper.tolist(), "spacing": spacing, "fluid_particles": int(len(speed)),
            "steps": total_steps, "time": total_steps * timestep, "timestep": timestep,
            "steps_per_second": None if failure else total_steps / elapsed, "device_uuid": uuid,
            "scales": {"velocity_sqrt_g_extent": velocity_scale, "pressure_rho_g_extent": pressure_scale,
                       "speed_of_sound": float(case.physics.speed_of_sound)},
            "final": {
                "kinetic_energy": series[-1]["kinetic_energy"],
                "speed_maximum": float(speed.max()), "speed_root_mean_square": float(np.sqrt((speed ** 2).mean())),
                "speed_maximum_over_scale": float(speed.max() / velocity_scale),
                "speed_root_mean_square_over_scale": float(np.sqrt((speed ** 2).mean()) / velocity_scale),
                "pressure_level_at_centre": level,
                "pressure_level_over_scale": level / pressure_scale,
                "pressure_error_root_mean_square_over_scale": float(np.sqrt((error ** 2).mean()) / pressure_scale),
                "pressure_error_maximum_over_scale": float(np.abs(error).max() / pressure_scale),
                "pressure_raw_error_root_mean_square_over_scale": float(np.sqrt((difference ** 2).mean()) / pressure_scale),
                "fitted_pressure_gradient": fitted_gradient.tolist(),
                "fitted_gradient_error_over_rho_g": float(np.linalg.norm(fitted_gradient - rest_density * gravity)
                                                          / (rest_density * gravity_norm)),
                "density_minimum": series[-1]["density_minimum"], "density_maximum": series[-1]["density_maximum"],
                "wall_distance_minimum_over_spacing": series[-1]["wall_distance_minimum"] / spacing,
                "layers": layers},
            "last_fifth": {
                "speed_maximum_over_scale": float(max(row["speed_maximum"] for row in tail) / velocity_scale),
                "speed_root_mean_square_over_scale": float(np.mean([row["speed_root_mean_square"] for row in tail])
                                                           / velocity_scale),
                "kinetic_energy_mean": float(np.mean([row["kinetic_energy"] for row in tail]))},
            "whole_run": {
                "wall_distance_minimum_over_spacing": float(min(row["wall_distance_minimum"] for row in series) / spacing),
                "gradient_fallbacks": series[-1]["gradient_fallbacks"],
                "correction_fallbacks": series[-1]["correction_fallbacks"],
                "overflow": max(row["overflow"] for row in series),
                "density_deviation_maximum": float(max(max(abs(row["density_minimum"] - rest_density),
                                                           abs(row["density_maximum"] - rest_density))
                                                       for row in series))},
        }
        with open(out / "series.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(series[0].keys()))
            writer.writeheader()
            writer.writerows(series)
        np.savez_compressed(out / "final.npz", positions=state["positions"].astype(np.float32),
                            velocities=state["velocities"].astype(np.float32),
                            densities=state["densities"].astype(np.float32),
                            pressures=state["pressures"].astype(np.float32),
                            hydrostatic_pressure=(hydrostatic + level).astype(np.float32))
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
        run.extra["summary"] = {key: summary["final"][key] for key in
                                ("speed_maximum_over_scale", "speed_root_mean_square_over_scale",
                                 "pressure_error_root_mean_square_over_scale", "pressure_level_over_scale")}
        final = summary["final"]
        if failure:
            print(f"[hydrostatic] {arguments.name} ({run.run_id}): FAILED at step {total_steps:,} "
                  f"(t = {summary['time']:.2f}): {failure}", flush=True)
        print(f"[hydrostatic] {arguments.name} ({run.run_id}): {total_steps:,} steps to t = {summary['time']:.2f} at "
              f"{summary['steps_per_second'] or 0:.0f} steps/s; |v| max {final['speed_maximum_over_scale']:.2e}, rms "
              f"{final['speed_root_mean_square_over_scale']:.2e} sqrt(gH); pressure error rms "
              f"{100 * final['pressure_error_root_mean_square_over_scale']:.3f} %, max "
              f"{100 * final['pressure_error_maximum_over_scale']:.3f} % of rho g H (level "
              f"{100 * final['pressure_level_over_scale']:+.3f} %)", flush=True)
    return 0


def table(arguments) -> int:
    root = LOG_ROOT / "hydrostatic"
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/summary.json"))]
    if not summaries:
        print("[hydrostatic] no finished run")
        return 1
    lines = ["| 运行 | 壁面 | t | \\|v\\| 最大 / √(gH)（末 1/5 段） | \\|v\\| rms / √(gH)（末 1/5 段） | 末态 \\|v\\| 最大 | "
             "压力 rms 误差（去电平） | 压力最大误差（去电平） | 中心电平 | 拟合梯度误差 | 近壁第 1 层 \\|v\\| rms | "
             "第 2 层 | 第 3 层 | 内部（≥5 dx） | A_f 回退 | φ_min/dx | 记录 |",
             "|" + "---|" * 17]
    for summary in summaries:
        final = summary["final"]
        layers = {tuple(layer["wall_distance_over_spacing"]): layer for layer in final["layers"]}
        scale = summary["scales"]["velocity_sqrt_g_extent"]

        def layer_text(key: tuple) -> str:
            return "—" if key not in layers else f"{layers[key]['speed_root_mean_square'] / scale:.2e}"

        wall = summary["wall_boundary"] + (f" ({summary['continuity']})" if summary["continuity"] else "")
        if summary.get("delta_wall") not in (None, "extension"):
            wall += f"，诊断 delta_wall = {summary['delta_wall']}"
        lines.append(
            f"| {summary['name']}"
            + (f"（失败：t = {summary['time']:.2f} 时粒子数 {summary['alive_at_last_sample']:,} / "
               f"{summary['particles']:,}，以下为该时刻的值）" if summary.get("failed") else "")
            + f" | {wall} | {summary['time']:.1f} | "
            f"{summary['last_fifth']['speed_maximum_over_scale']:.2e} | "
            f"{summary['last_fifth']['speed_root_mean_square_over_scale']:.2e} | "
            f"{final['speed_maximum_over_scale']:.2e} | "
            f"{100 * final['pressure_error_root_mean_square_over_scale']:.3f} % | "
            f"{100 * final['pressure_error_maximum_over_scale']:.3f} % | "
            f"{100 * final['pressure_level_over_scale']:+.3f} % | "
            f"{100 * final['fitted_gradient_error_over_rho_g']:.3f} % | "
            f"{layer_text((0.0, 1.0))} | {layer_text((1.0, 2.0))} | {layer_text((2.0, 3.0))} | "
            f"{layer_text((5.0, None))} | {summary['whole_run']['gradient_fallbacks']} | "
            f"{summary['whole_run']['wall_distance_minimum_over_spacing']:.3f} | {summary['run_id']} |")
    text = ("静水盒：速度以 √(|g| H) 为单位，压力以 ρ₀ |g| H 为单位（H = 盒沿重力方向的尺寸）。"
            "“去电平” = 减去全场平均偏差后的误差。\n\n" + "\n".join(lines) + "\n")
    out = (REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "hydrostatic.md").write_text(text, encoding="utf-8", newline="\n")
    (out / "hydrostatic.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8", newline="\n")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make_parser = commands.add_parser("make-cases")
    make_parser.add_argument("--force", action="store_true", help="rewrite existing generated case files")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--device", type=int, required=True)
    run_parser.add_argument("--uuid", default=None)
    run_parser.add_argument("--time", type=float, default=10.0, help="simulated time")
    run_parser.add_argument("--sample-time", type=float, default=0.05)
    table_parser = commands.add_parser("table")
    table_parser.add_argument("--out", default="claude/wall_crossing/hydrostatic")
    arguments = parser.parse_args()
    if arguments.command == "make-cases":
        for path in make_cases(arguments.force):
            print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
        return 0
    if arguments.command == "run":
        return run_case(arguments)
    return table(arguments)


if __name__ == "__main__":
    sys.exit(main())
