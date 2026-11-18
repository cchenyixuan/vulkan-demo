"""
box_run.py - run a box case (2-D or 3-D, any wall mode) on one GPU with the v7 single-GPU step and record its
history: kinetic energy, speeds, density range, the fluid's smallest wall distance, the solver's fallback
counters, and (optionally) the velocity profiles along the two centre lines through the box centre.

    .venv/Scripts/python.exe -m experiment.wall_crossing.box_run --case cases/cavity3d_1m_crossing/case.yaml \
        --name cavity3d_1m_crossing --device 0 --uuid fb83f0dafd9e1d38dc05e7ae643e22a8 --time 20 \
        --sample-time 0.1 --profile-time 2.5

The run: the solver's production path (chain_runner.run_chain: a chain of K slabs, K = the number of --device
entries, pipelined at depth 2, a defrag every defrag cadence). Samples are taken at multiples of the defrag
cadence, over the own particles of every slab together.

Profiles (--profile-time > 0): at those times and at the end, the fluid velocity on 101 points of
  line "u_along_y": through the box centre along y (the lid's normal), component x (the lid's direction);
  line "v_along_x": through the box centre along x, component y;
by a Shepard average over the fluid particles within the support radius h of each point
(sum_j V_j v_j W / sum_j V_j W, Wendland C4, V_j = m_j / rho_j). Fluid particles only, so the value near a wall
is one-sided in every wall mode alike. In 2-D the lines lie in the plane.

--density-budget (DIAGNOSTIC; sets V7_DENSITY_BUDGET=1 before the solver is imported, one slab): the fused pass
runs its DENSITY_BUDGET build, which adds dt x (drift fluid, drift wall, diffusion fluid, diffusion wall) of
every fluid particle's density rate, every step, to a buffer binned by the particle's region (which walls of
the box lie within the support radius; shaders/density_budget.glsl). Every sample then also has
  budget_<region>_<term>   the buffer summed over all slots (float64): the cumulative sum over the fluid
                           particles and the steps so far of dt x term, kg/m^3 x particles;
  region_count_<region>    the fluid particles in the region at the sample.
The state the budget build computes is the state of the build without it (checked bit for bit over 700 steps).

Output: logs/wall_crossing/box_run/<name>/series.csv, profiles.json, final.npz (the fluid: positions,
velocities, densities, pressures), summary.json; one runs.jsonl record.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.hydrostatic import initial_box, read_fluid_state_of_chain      # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                    # noqa: E402

PROFILE_POINTS = 101


def wendland_c4(distance: np.ndarray, support: float) -> np.ndarray:
    """The kernel's profile (the normalisation cancels in a Shepard average)."""
    normalized = np.clip(distance / support, 0.0, 1.0)
    return (1.0 - normalized) ** 6 * ((35.0 / 3.0) * normalized ** 2 + 6.0 * normalized + 1.0)


def centre_line_profiles(state: dict, lower: np.ndarray, upper: np.ndarray, dimension: int, support: float) -> dict:
    """Shepard-averaged fluid velocity on the two centre lines (see the module docstring)."""
    centre = 0.5 * (lower + upper)
    tree = cKDTree(state["positions"])
    volumes = state["masses"] / state["densities"]
    profiles = {}
    for name, axis, component in (("u_along_y", 1, 0), ("v_along_x", 0, 1)):
        coordinate = np.linspace(lower[axis], upper[axis], PROFILE_POINTS)
        points = np.tile(centre, (PROFILE_POINTS, 1))
        points[:, axis] = coordinate
        if dimension == 2:
            points[:, 2] = 0.0
        values = np.full(PROFILE_POINTS, np.nan)
        neighbour_lists = tree.query_ball_point(points, r=support)
        for index, neighbours in enumerate(neighbour_lists):
            if not neighbours:
                continue
            neighbours = np.asarray(neighbours)
            distance = np.linalg.norm(state["positions"][neighbours] - points[index], axis=1)
            weight = volumes[neighbours] * wendland_c4(distance, support)
            total = weight.sum()
            if total > 0.0:
                values[index] = float((weight * state["velocities"][neighbours, component]).sum() / total)
        profiles[name] = {"coordinate": ((coordinate - lower[axis]) / (upper[axis] - lower[axis])).tolist(),
                          "velocity": values.tolist()}
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", required=True,
                        help="v7 device index, or one per slab separated by commas (K = the number of entries)")
    parser.add_argument("--uuid", action="append", default=[],
                        help="deviceUUID the device must have (repeat once per slab, in order)")
    parser.add_argument("--time", type=float, required=True, help="simulated time")
    parser.add_argument("--sample-time", type=float, default=0.1)
    parser.add_argument("--profile-time", type=float, default=0.0, help="interval of the centre-line profiles (0: none)")
    parser.add_argument("--density-budget", action="store_true",
                        help="DIAGNOSTIC: run with V7_DENSITY_BUDGET=1 and record the budget sums at every sample")
    arguments = parser.parse_args()
    if arguments.density_budget:
        os.environ["V7_DENSITY_BUDGET"] = "1"       # read once when simulator_v7 is imported (below)

    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.wall_crossing_v7 import box_distance_to_wall
    from experiment.wall_crossing.chain_runner import run_chain

    case = load_case_v7((REPOSITORY_ROOT / arguments.case).resolve())
    out = LOG_ROOT / "box_run" / arguments.name
    out.mkdir(parents=True, exist_ok=True)
    lower, upper, radius, dimension = initial_box(case)
    spacing = 2.0 * radius
    support = float(case.physics.smoothing_length)
    rest_density = float(next(material for material in case.materials if material.kind == 0).rest_density)
    timestep = float(case.physics.timestep)
    cadence = int(case.numerics.defrag_cadence)
    total_steps = int(np.ceil(arguments.time / timestep / cadence)) * cadence
    sample_steps = cadence * max(1, int(round(arguments.sample_time / timestep / cadence)))
    profile_steps = cadence * max(1, int(round(arguments.profile_time / timestep / cadence))) \
        if arguments.profile_time > 0 else 0
    expected_alive = int(case.initial.positions.shape[0])
    floor = case.numerics.wall_crossing.distance_floor(radius) if case.numerics.wall_crossing else None

    device_map = [int(item) for item in str(arguments.device).split(",")]
    with LoggedRun(kind="box_run", case=str(arguments.case).replace("\\", "/")) as run:
        run.extra["name"] = arguments.name
        run.extra["slabs"] = len(device_map)
        series = []
        profiles = []
        latest = {}
        series_file = open(out / "series.csv", "w", newline="", encoding="utf-8")
        writer = {"csv": None}
        clock = {"started": None}

        def on_sample(step: int, simulators: list, slab_cases: list) -> None:
            take_sample = step % sample_steps == 0 or step == total_steps
            take_profile = profile_steps and (step % profile_steps == 0 or step == total_steps)
            if not (take_sample or take_profile):
                return
            state = read_fluid_state_of_chain(simulators, slab_cases)
            latest["state"] = state
            speed = np.sqrt((state["velocities"] ** 2).sum(axis=1))
            # a run that loses a particle (or produces a non-finite velocity) stops at this sample; the sample's
            # row is still written first, so the series ends with the state that failed
            lost = state["alive"] != expected_alive or not np.isfinite(speed).all()
            if take_sample or lost:
                statuses = [simulator.readback_global_status() for simulator in simulators]
                distance = box_distance_to_wall(state["positions"], lower, upper, dimension)
                row = {"step": step, "time": step * timestep,
                       "kinetic_energy": float(0.5 * (state["masses"] * speed ** 2).sum()),
                       "speed_maximum": float(speed.max()),
                       "speed_root_mean_square": float(np.sqrt((speed ** 2).mean())),
                       "density_minimum": float(state["densities"].min()),
                       "density_maximum": float(state["densities"].max()),
                       # means over the alive fluid particles (no wall), float64 sums of the true density
                       # (read_fluid_state adds the storage offset) and of the stored pressure
                       "density_mean": float(state["densities"].mean()),
                       "pressure_mean": float(state["pressures"].mean()),
                       "pressure_negative_fraction": float((state["pressures"] < 0.0).mean()),
                       "alive": int(state["alive"]),
                       "density_deviation_maximum": float(np.abs(state["densities"] - rest_density).max()),
                       "wall_distance_minimum_over_spacing": float(distance.min() / spacing),
                       "below_floor_count": int((distance < floor).sum()) if floor is not None else 0,
                       "outside_box_count": int((distance <= 0.0).sum()),
                       "correction_fallbacks": int(sum(status["correction_fallback_count"] for status in statuses)),
                       "gradient_fallbacks": int(sum(status.get("wall_crossing_gradient_fallback_count", 0)
                                                     for status in statuses)),
                       "overflow": int(sum(value for status in statuses for key, value in status.items()
                                           if key.startswith("overflow_"))),
                       "steps_per_second": step / max(time.perf_counter() - clock["started"], 1e-9)}
                if arguments.density_budget:
                    from experiment.v7.utils.simulator_v7 import (DENSITY_BUDGET_REGION_NAMES,
                                                                  DENSITY_BUDGET_TERM_NAMES)
                    budget = sum(simulator.readback_density_budget() for simulator in simulators)
                    for region_index, region_name in enumerate(DENSITY_BUDGET_REGION_NAMES):
                        for term_index, term_name in enumerate(DENSITY_BUDGET_TERM_NAMES):
                            row[f"budget_{region_name}_{term_name}"] = float(budget[region_index, term_index])
                    # the region of every fluid particle now (the shader's rule)
                    from experiment.v7.utils.simulator_v7 import density_budget_region_index
                    budget_lower, budget_upper = simulators[0].density_budget_box()
                    counts = np.bincount(
                        density_budget_region_index(state["positions"], budget_lower, budget_upper, support),
                        minlength=len(DENSITY_BUDGET_REGION_NAMES))
                    for region_index, region_name in enumerate(DENSITY_BUDGET_REGION_NAMES):
                        row[f"region_count_{region_name}"] = int(counts[region_index])
                    row["fluid_particles"] = int(len(state["positions"]))
                series.append(row)
                if writer["csv"] is None:
                    writer["csv"] = csv.DictWriter(series_file, fieldnames=list(row.keys()))
                    writer["csv"].writeheader()
                writer["csv"].writerow(row)
                series_file.flush()
            if lost:
                raise RuntimeError(f"step {step} (t = {step * timestep:.3f}): alive {state['alive']} of "
                                   f"{expected_alive}"
                                   + ("" if np.isfinite(speed).all() else ", non-finite velocity"))
            if take_profile:
                profiles.append({"step": step, "time": step * timestep,
                                 **centre_line_profiles(state, lower, upper, dimension, support)})
                (out / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8", newline="\n")

        def on_ready(simulators: list, slab_cases: list) -> None:
            clock["started"] = time.perf_counter()
            on_sample(0, simulators, slab_cases)        # the state after the bootstrap, t = 0

        try:
            # the callback is asked at every defrag boundary and decides itself (samples and profiles may have
            # different intervals)
            outcome = run_chain(case, device_map, total_steps, cadence, on_sample,
                                expected_uuids=arguments.uuid or None,
                                application_name="wall_crossing_box_run", on_ready=on_ready)
        finally:
            series_file.close()
        uuid = ",".join(outcome["device_uuids"])
        run.record["gpu_uuids"] = outcome["device_uuids"]
        state = latest["state"]
        elapsed = outcome["elapsed_seconds"]

        np.savez_compressed(out / "final.npz", positions=state["positions"].astype(np.float32),
                            velocities=state["velocities"].astype(np.float32),
                            densities=state["densities"].astype(np.float32),
                            pressures=state["pressures"].astype(np.float32))
        summary = {
            "name": arguments.name, "run_id": run.run_id, "case": str(arguments.case).replace("\\", "/"),
            "wall_boundary": case.numerics.wall_boundary,
            "continuity": case.numerics.wall_crossing.continuity if case.numerics.wall_crossing else None,
            "continuity_wall_component": (case.numerics.wall_crossing.continuity_wall_component
                                          if case.numerics.wall_crossing else None),
            "delta_wall": case.numerics.wall_crossing.delta_wall if case.numerics.wall_crossing else None,
            "delta_density_storage": os.environ.get("V7_DELTA_DENSITY", "0") == "1",
            "dimension": dimension, "box_lower": lower.tolist(), "box_upper": upper.tolist(), "spacing": spacing,
            "fluid_particles": int(len(state["positions"])), "particles": expected_alive, "steps": total_steps,
            "time": total_steps * timestep, "timestep": timestep, "steps_per_second": total_steps / elapsed,
            "device_uuid": uuid, "samples": len(series), "profiles": len(profiles),
            "density_budget": bool(arguments.density_budget),
            "final": {key: series[-1][key] for key in ("kinetic_energy", "speed_maximum", "density_minimum",
                                                       "density_maximum")},
            "whole_run": {
                "density_deviation_maximum": max(row["density_deviation_maximum"] for row in series),
                "wall_distance_minimum_over_spacing": min(row["wall_distance_minimum_over_spacing"] for row in series),
                "below_floor_count_maximum": max(row["below_floor_count"] for row in series),
                "outside_box_count_maximum": max(row["outside_box_count"] for row in series),
                "gradient_fallbacks": series[-1]["gradient_fallbacks"],
                "correction_fallbacks": series[-1]["correction_fallbacks"],
                "overflow": max(row["overflow"] for row in series)},
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
        run.extra["summary"] = {**summary["final"], **summary["whole_run"],
                                "steps_per_second": summary["steps_per_second"]}
        print(f"[box_run] {arguments.name} ({run.run_id}): {total_steps:,} steps to t = {summary['time']:.2f} at "
              f"{summary['steps_per_second']:.0f} steps/s; KE {summary['final']['kinetic_energy']:.5g}, "
              f"max |rho - rho0| {summary['whole_run']['density_deviation_maximum']:.4g}, phi_min "
              f"{summary['whole_run']['wall_distance_minimum_over_spacing']:.3f} dx, A_f fallbacks "
              f"{summary['whole_run']['gradient_fallbacks']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
