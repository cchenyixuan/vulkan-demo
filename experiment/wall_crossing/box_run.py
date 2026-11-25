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

--density-budget: removed with the wall store (branch v7-crossing-wall-store); the diagnostic lives in the
particle-pool version (git tag crossing-pool). The option is refused.

Long runs (2026-11-19; box_run_supervised.py restarts a run that stops or stalls):
  --checkpoint-time T   every T (a multiple of the defrag cadence) the restart state of every slab
                        (cavity_runner.read_restart_state, merged) goes to checkpoints/c<step>.npz with the stored
                        density offset; the two newest are kept (manifest.jsonl lists them);
  --resume              start from the newest readable checkpoint instead of the case's initial state (none: from
                        the start): series.csv and profiles.json keep their rows up to that step, the volume
                        constraint's log its records before it (the log's steps restart at 0 with every segment;
                        a line {"segment", "segment_start_step"} marks each start); the constraint's target is
                        taken again from the case's initial particles, as the cavity runner does; with
                        numerics.volume_constraint.gpu_mean_volume (report section 22) the checkpoints also hold
                        v_ref, the rate and theta (_volume_constraint_*), and a resumed run takes them from there;
  heartbeat.json        written at every defrag boundary (at most every 2 s): step, time, unix time, segment;
  --stop-speed F        stop when the fluid's largest speed exceeds F x the lid speed (the largest initial
                        velocity of a boundary material);
  --stop-outside        stop when a fluid particle is outside the box (outside_box_count > 0).
  A particle loss or a non-finite velocity stops the run as well. A stop writes watch_stop.json (reason, step,
  time), keeps everything written so far and exits with code 3; segments.jsonl gets one line per segment.

Output: logs/wall_crossing/box_run/<name>/series.csv, profiles.json, final.npz (the fluid: positions,
velocities, densities, pressures), summary.json; one runs.jsonl record per segment.

3-D cavity benchmark (claude/wall_store_report.md section 14, 2026-11-24; all off by default, nothing else changes):
  --lines-time T        every T (a multiple of the defrag cadence) and at the end, the three centre lines (vertical
                        u, horizontal v, spanwise w; every dx / 8 and at the benchmark's table points) by linear MLS
                        (cavity3d_lines.py) go to lines.npz;
  --frame-time T, --frame-from T0
                        every T from T0 on (and at the end), the fluid state (float32 positions, velocities,
                        densities, pressures, masses) goes to frames/f<step>.npz;
  --diagnostics         every sample row also holds the largest fluid acceleration (its value, position and how many
                        box faces lie within h of it), the smallest fluid-fluid and fluid-to-wall-particle distances
                        (over the spacing), and the GPU's alive count (to compare with the readback); final.npz
                        also holds the masses and the accelerations.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
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


def cavity3d_refine_last_integral(path: pathlib.Path):
    from experiment.wall_crossing.cavity3d_refine import last_integral
    return last_integral(path)

PROFILE_POINTS = 101
STOP_EXIT_CODE = 3


class StopCondition(Exception):
    """A stop condition of the run (particle loss, speed, a fluid particle outside the box)."""


class Converged(Exception):
    """--converge: the level has converged (report section 15); the run ends like a complete one."""


def typed(value: str):
    """A csv cell back as int or float (an empty cell as None)."""
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def latest_checkpoint(directory: pathlib.Path, expected_rows: int, wall_store: bool = False) -> dict | None:
    """The newest readable checkpoint of directory (manifest.jsonl): {"state", "stored_offset", "step"}. wall_store (a
    crossing case of the wall-store version): a checkpoint without the _wall_store mark is one of the particle-pool
    version (git tag crossing-pool), whose walls are in the state; it cannot be resumed (SystemExit with the reason)."""
    manifest = directory / "manifest.jsonl"
    if not manifest.exists():
        return None
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for entry in reversed(entries):
        checkpoint = directory / entry["file"]
        if not checkpoint.exists():
            continue
        try:
            with np.load(checkpoint) as archive:
                if wall_store and "_wall_store" not in archive.files:
                    sys.exit(f"[box_run] checkpoint {checkpoint} is a wall_boundary crossing checkpoint of the "
                             "particle-pool version (git tag crossing-pool): its walls are in the particle pool. The "
                             "wall-store version cannot resume it; start the run anew or resume it with the "
                             "crossing-pool tag.")
                state = {name: archive[name] for name in archive.files if not name.startswith("_")}
                offset = float(archive["_stored_density_offset"])
                # numerics.volume_constraint.gpu_mean_volume (report section 22): v_ref, rate and theta when written
                volume_constraint = {name: float(archive[name]) for name in archive.files
                                     if name.startswith("_volume_constraint_")} or None
        except Exception as error:          # a partly written file: take an older one
            print(f"[box_run] checkpoint {checkpoint.name} unreadable ({error!r}); trying an older one", flush=True)
            continue
        if state["position_voxel_id"].shape[0] != expected_rows:
            print(f"[box_run] checkpoint {checkpoint.name} holds {state['position_voxel_id'].shape[0]} rows, "
                  f"expected {expected_rows}; trying an older one", flush=True)
            continue
        return {"state": state, "stored_offset": offset, "step": int(entry["step"]),
                "volume_constraint": volume_constraint}
    return None


def truncate_constraint_log(path: pathlib.Path, start_step: int, segment: int) -> None:
    """Keep the records of the volume constraint's log before start_step (absolute: the log's steps restart at 0
    with every segment, whose start a marker line gives) and mark the new segment's start."""
    kept, offset = [], 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "segment_start_step" in record:
                offset = int(record["segment_start_step"])
                if offset < start_step:
                    kept.append(line)
                continue
            if offset + int(record["step"]) < start_step:
                kept.append(line)
    kept.append(json.dumps({"segment": segment, "segment_start_step": start_step}))
    path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")


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
                        help="removed with the wall store (refused; the diagnostic is in the git tag crossing-pool)")
    parser.add_argument("--checkpoint-time", type=float, default=0.0,
                        help="interval of the restart checkpoints (0: none)")
    parser.add_argument("--resume", action="store_true", help="start from the newest checkpoint of the run")
    parser.add_argument("--stop-speed", type=float, default=0.0,
                        help="stop when the fluid's largest speed exceeds this x the lid speed (0: off)")
    parser.add_argument("--depth", type=int, default=2,
                        help="frames in flight of the pipelined chain (default 2, the production path)")
    parser.add_argument("--stop-outside", action="store_true",
                        help="stop when a fluid particle is outside the box")
    parser.add_argument("--lines-time", type=float, default=0.0,
                        help="interval of the MLS centre-line samples to lines.npz (0: none; report section 14)")
    parser.add_argument("--frame-time", type=float, default=0.0,
                        help="interval of the saved fluid frames (0: none; report section 14)")
    parser.add_argument("--frame-from", type=float, default=0.0, help="first time of the saved fluid frames")
    parser.add_argument("--diagnostics", action="store_true",
                        help="acceleration, pair distances and the GPU alive count in every sample row (section 14)")
    # report section 15 (round 9, cavity3d_refine.py): all off by default
    parser.add_argument("--initial-state", default="",
                        help="start from this fluid state (a previous level's final.npz) interpolated onto the case")
    parser.add_argument("--initial-support", type=float, default=0.0,
                        help="support of the --initial-state interpolation (the previous level's h)")
    parser.add_argument("--initial-integral", type=float, default=None,
                        help="the volume constraint's integral term I to start with (the previous level's)")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="added to every recorded time (the time of the previous levels)")
    parser.add_argument("--tube-time", type=float, default=0.0,
                        help="interval of the centre-line tube records to tubes/ (0: none)")
    parser.add_argument("--tube-half-width", type=float, default=1.5, help="tube half-width in units of h")
    parser.add_argument("--converge", action="store_true",
                        help="stop when converged by the rule of cavity3d_refine.ConvergenceMonitor (needs --lines-time)")
    parser.add_argument("--converge-minimum", type=float, default=6.0)
    parser.add_argument("--converge-block", type=float, default=2.0)
    parser.add_argument("--converge-tolerance", type=float, default=0.002)
    parser.add_argument("--converge-sigmas", type=float, default=3.0)
    parser.add_argument("--wall-deadline", type=float, default=0.0,
                        help="unix time: at the first line record after it the run ends like a converged one, marked "
                             "not converged (0: off)")
    # report section 16 (round 10): an outside monitor (cavity3d_round10.py) decides when the level is steady
    parser.add_argument("--stop-file", default="",
                        help="at the first line record after this file exists the run ends like a converged one, its "
                             "checkpoint written, end_reason stop_file (empty: off)")
    arguments = parser.parse_args()
    if arguments.density_budget:
        sys.exit("[box_run] --density-budget was removed with the wall store; use the git tag crossing-pool")
    if arguments.initial_state and arguments.resume:
        sys.exit("[box_run] --initial-state and --resume exclude each other")
    if arguments.initial_state and arguments.initial_support <= 0.0:
        sys.exit("[box_run] --initial-state needs --initial-support (the previous level's h)")
    if arguments.converge and arguments.lines_time <= 0.0:
        sys.exit("[box_run] --converge needs --lines-time (the records are the line samples)")

    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.wall_crossing_v7 import box_distance_to_wall
    from experiment.wall_crossing.chain_runner import run_chain

    case = load_case_v7((REPOSITORY_ROOT / arguments.case).resolve())
    out = LOG_ROOT / "box_run" / arguments.name
    out.mkdir(parents=True, exist_ok=True)
    expected_alive = int(case.initial.positions.shape[0])
    segments_path = out / "segments.jsonl"
    restart = (latest_checkpoint(out / "checkpoints", expected_alive, wall_store=case.wall_source is not None)
               if arguments.resume else None)
    if arguments.resume and restart is None:
        print("[box_run] --resume: no usable checkpoint; starting from the initial state", flush=True)
    # resuming: a restart from this run's own checkpoint (the files continue); an --initial-state start goes through
    # the same restart path at step 0 but is a new run (report section 15)
    resuming = restart is not None
    interpolation_report = None
    if arguments.initial_state:
        from experiment.wall_crossing import cavity3d_refine
        started_interpolation = time.perf_counter()
        restart = cavity3d_refine.interpolated_restart(case, pathlib.Path(arguments.initial_state).resolve(),
                                                       arguments.initial_support, arguments.initial_integral)
        interpolation_report = {**restart.pop("report"),
                                "seconds": round(time.perf_counter() - started_interpolation, 1)}
        (out / "initial_state.json").write_text(json.dumps(interpolation_report, indent=1), encoding="utf-8")
        print(f"[box_run] initial state interpolated: {json.dumps(interpolation_report)}", flush=True)
    start_step = restart["step"] if restart is not None else 0
    time_offset = float(arguments.time_offset)
    # the segment number: one more than any segment that left a trace (segments.jsonl is written at a segment's
    # end only, so a killed segment shows in the series rows and the constraint log's markers instead)
    segment = 0
    if resuming:
        traces = [-1]
        traces += [int(json.loads(line)["segment"]) for line in
                   (segments_path.read_text(encoding="utf-8").splitlines() if segments_path.exists() else [])
                   if line.strip()]
        if (out / "series.csv").exists():
            with open(out / "series.csv", newline="", encoding="utf-8") as handle:
                traces += [int(float(row["segment"])) for row in csv.DictReader(handle) if row.get("segment")]
        traces += [int(record["segment"]) for record in
                   ([json.loads(line) for line in (out / "volume_constraint.jsonl").read_text(encoding="utf-8")
                     .splitlines() if line.strip()] if (out / "volume_constraint.jsonl").exists() else [])
                   if "segment_start_step" in record]
        segment = max(traces) + 1
    if not resuming:
        segments_path.unlink(missing_ok=True)
        (out / "watch_stop.json").unlink(missing_ok=True)
    if case.numerics.volume_constraint.enabled:
        # every update of the volume constraint (time, theta, rate) goes to this run's directory; the file of an
        # earlier run of the same name goes the way of its series.csv
        constraint_log = out / "volume_constraint.jsonl"
        if not resuming:
            constraint_log.unlink(missing_ok=True)
        else:
            truncate_constraint_log(constraint_log, start_step, segment)
        os.environ["SPH_VOLUME_CONSTRAINT_LOG"] = str(constraint_log)
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
    floor = case.numerics.wall_crossing.distance_floor(radius) if case.numerics.wall_crossing else None
    checkpoint_steps = cadence * max(1, int(round(arguments.checkpoint_time / timestep / cadence))) \
        if arguments.checkpoint_time > 0 else 0
    # report section 14: MLS centre lines, saved frames, diagnostics (all off by default)
    lines_steps = cadence * max(1, int(round(arguments.lines_time / timestep / cadence))) \
        if arguments.lines_time > 0 else 0
    frame_steps = cadence * max(1, int(round(arguments.frame_time / timestep / cadence))) \
        if arguments.frame_time > 0 else 0
    frame_first_step = int(np.ceil(arguments.frame_from / timestep / cadence - 1e-9)) * cadence
    recorder = None
    wall_positions = None
    if lines_steps or arguments.diagnostics:
        from experiment.wall_crossing import cavity3d_lines
        from experiment.wall_crossing.cavity3d_benchmark import reference_points
        sampler = cavity3d_lines.sampler_of_case(case, lower, upper, reference_points(),
                                                  (REPOSITORY_ROOT / arguments.case).resolve())
        wall_positions = sampler.walls[0]
        if lines_steps:
            recorder = cavity3d_lines.LineRecorder(out / "lines.npz", sampler,
                                                   existing_until_step=start_step if resuming else None)
            if not resuming:
                (out / "lines.npz").unlink(missing_ok=True)
                recorder.rows = []
    if frame_steps:
        (out / "frames").mkdir(exist_ok=True)
    # report section 15: tubes around the centre lines and the convergence rule (cavity3d_refine.py)
    tube_steps = cadence * max(1, int(round(arguments.tube_time / timestep / cadence))) \
        if arguments.tube_time > 0 else 0
    tube_half_width = arguments.tube_half_width * support
    box_centre = 0.5 * (lower + upper)
    if tube_steps:
        (out / "tubes").mkdir(exist_ok=True)
        if not resuming:
            for old in (out / "tubes").glob("t*.npz"):
                old.unlink()
    monitor = None
    if arguments.converge:
        from experiment.wall_crossing import cavity3d_refine
        from experiment.wall_crossing.cavity3d_benchmark import extrema_of_profiles
        monitor = cavity3d_refine.ConvergenceMonitor(out / "convergence.jsonl", block=arguments.converge_block,
                                                     minimum=arguments.converge_minimum,
                                                     tolerance=arguments.converge_tolerance,
                                                     sigmas=arguments.converge_sigmas)
    lid_speed = max((float(np.linalg.norm(material.initial_velocity)) for material in case.materials
                     if material.kind != 0), default=0.0)
    if arguments.stop_speed > 0.0 and lid_speed <= 0.0:
        parser.error("--stop-speed needs a moving boundary material (the lid)")
    if start_step >= total_steps:
        print(f"[box_run] the newest checkpoint (step {start_step}) is at or past the end ({total_steps})", flush=True)
        return 0

    device_map = [int(item) for item in str(arguments.device).split(",")]
    with LoggedRun(kind="box_run", case=str(arguments.case).replace("\\", "/")) as run:
        run.extra["name"] = arguments.name
        run.extra["slabs"] = len(device_map)
        run.extra["depth"] = arguments.depth
        run.extra["segment"] = segment
        run.extra["start_step"] = start_step
        series = []
        profiles = []
        latest = {}
        writer = {"csv": None}
        if resuming and (out / "series.csv").exists():
            # the rows up to the checkpoint stay; the file is written again with them and continued
            import csv as csv_module
            with open(out / "series.csv", newline="", encoding="utf-8") as handle:
                reader = csv_module.DictReader(handle)
                fieldnames = reader.fieldnames
                series = [{key: typed(value) for key, value in row.items()} for row in reader
                          if int(row["step"]) <= start_step]
            series_file = open(out / "series.csv", "w", newline="", encoding="utf-8")
            writer["csv"] = csv.DictWriter(series_file, fieldnames=fieldnames, extrasaction="ignore")
            writer["csv"].writeheader()
            for row in series:
                writer["csv"].writerow({key: ("" if value is None else value) for key, value in row.items()})
            series_file.flush()
        else:
            series_file = open(out / "series.csv", "w", newline="", encoding="utf-8")
        if resuming and (out / "profiles.json").exists():
            profiles = [entry for entry in json.loads((out / "profiles.json").read_text(encoding="utf-8"))
                        if entry["step"] <= start_step]
        clock = {"started": None, "heartbeat": 0.0}
        checkpoint_directory = out / "checkpoints"
        if checkpoint_steps or arguments.converge or arguments.stop_file:
            checkpoint_directory.mkdir(exist_ok=True)
            if not resuming:
                for old in checkpoint_directory.glob("c*.npz"):
                    old.unlink()
                (checkpoint_directory / "manifest.jsonl").unlink(missing_ok=True)

        def save_checkpoint(step: int, simulators: list) -> None:
            from experiment.validation.cavity_runner import read_restart_state
            parts = [read_restart_state(simulator) for simulator in simulators]
            merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            if merged["position_voxel_id"].shape[0] != expected_alive:
                raise StopCondition(f"particle loss at the checkpoint of step {step}: "
                                    f"{merged['position_voxel_id'].shape[0]} rows of {expected_alive}")
            name = f"c{step:010d}.npz"
            temporary = checkpoint_directory / f"c{step:010d}.partial.npz"
            marker = {"_wall_store": np.int64(1)} if case.wall_source is not None else {}
            np.savez(temporary, _stored_density_offset=np.float64(simulators[0].stored_density_offset()),
                     **simulators[0].volume_constraint_checkpoint_entries(), **marker, **merged)
            os.replace(temporary, checkpoint_directory / name)
            with open(checkpoint_directory / "manifest.jsonl", "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"step": step, "time": step * timestep, "file": name}) + "\n")
            for old in sorted(checkpoint_directory.glob("c*.npz"))[:-2]:
                old.unlink()

        def on_sample(step: int, simulators: list, slab_cases: list) -> None:
            now = time.time()
            if now - clock["heartbeat"] >= 2.0 or step == total_steps:
                (out / "heartbeat.json").write_text(json.dumps({
                    "step": step, "time": time_offset + step * timestep, "level_time": step * timestep, "unix": now,
                    "segment": segment}), encoding="utf-8")
                clock["heartbeat"] = now
            if checkpoint_steps and step > start_step and step % checkpoint_steps == 0 and step < total_steps:
                save_checkpoint(step, simulators)
            if step == start_step and resuming:
                return                              # the checkpoint's own state: its row is already in series.csv
            take_sample = step % sample_steps == 0 or step == total_steps
            take_profile = profile_steps and (step % profile_steps == 0 or step == total_steps)
            take_lines = bool(lines_steps) and (step % lines_steps == 0 or step == total_steps)
            take_frame = bool(frame_steps) and step >= frame_first_step and (step % frame_steps == 0
                                                                             or step == total_steps)
            take_tube = bool(tube_steps) and (step % tube_steps == 0 or step == total_steps)
            if not (take_sample or take_profile or take_lines or take_frame or take_tube):
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
                row = {"step": step, "time": time_offset + step * timestep,
                       **({"level_time": step * timestep} if (time_offset or arguments.initial_state
                                                               or arguments.converge) else {}),
                       "kinetic_energy": float(0.5 * (state["masses"] * speed ** 2).sum()),
                       "speed_maximum": float(speed.max()),
                       # where the fastest fluid particle is (to locate a speed excess)
                       **{f"fastest_{axis_name}": float(state["positions"][int(np.argmax(speed)), axis_index])
                          for axis_index, axis_name in enumerate("xyz"[:dimension])},
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
                       # report section 22: each overflow counter, and the alive fluid particles
                       **{key: int(sum(status.get(key, 0) for status in statuses))
                          for key in sorted({key for status in statuses for key in status
                                             if key.startswith("overflow_")})},
                       "fluid_alive": int(len(state["positions"])),
                       "steps_per_second": (step - start_step) / max(time.perf_counter() - clock["started"], 1e-9),
                       "segment": segment}
                if arguments.diagnostics:
                    from experiment.wall_crossing import cavity3d_lines
                    accelerated = cavity3d_lines.read_acceleration_of_chain(simulators, slab_cases)
                    latest["accelerations"] = accelerated["accelerations"]
                    latest["acceleration_positions"] = accelerated["positions"]
                    magnitude = np.sqrt((accelerated["accelerations"] ** 2).sum(axis=1))
                    largest = int(np.argmax(magnitude)) if magnitude.size else 0
                    where = accelerated["positions"][largest] if magnitude.size else np.zeros(3)
                    fluid_fluid, fluid_wall = cavity3d_lines.pair_distance_minima(state["positions"], wall_positions)
                    row.update({
                        "acceleration_maximum": float(magnitude.max()) if magnitude.size else float("nan"),
                        **{f"acceleration_maximum_{axis_name}": float(where[axis_index])
                           for axis_index, axis_name in enumerate("xyz"[:dimension])},
                        "acceleration_maximum_faces_within_h": int(cavity3d_lines.wall_count_within(
                            where[None, :dimension], lower[:dimension], upper[:dimension], support)[0]),
                        "acceleration_99_9_percentile": float(np.percentile(magnitude, 99.9))
                        if magnitude.size else float("nan"),
                        "pair_distance_minimum_over_spacing": fluid_fluid / spacing,
                        "wall_particle_distance_minimum_over_spacing": fluid_wall / spacing,
                        # the GPU's alive count against the readback's (a zombie: counted, not readable, or the
                        # other way round), far migrations and stamp errors (must all be 0 / equal)
                        "alive_gpu": int(sum(status["alive_particle_count"] for status in statuses)),
                        "alive_readback": int(state["alive"]),
                        "far_migrations": int(sum(status["far_migration_count"] for status in statuses)),
                        "stamp_errors": int(sum(status["stamp_error_count"] for status in statuses))})
                series.append(row)
                if writer["csv"] is None:
                    writer["csv"] = csv.DictWriter(series_file, fieldnames=list(row.keys()), extrasaction="ignore")
                    writer["csv"].writeheader()
                writer["csv"].writerow(row)
                series_file.flush()
            if lost:
                raise StopCondition(f"step {step} (t = {step * timestep:.3f}): alive {state['alive']} of "
                                    f"{expected_alive}"
                                    + ("" if np.isfinite(speed).all() else ", non-finite velocity"))
            if take_sample and arguments.stop_speed > 0.0 and row["speed_maximum"] > arguments.stop_speed * lid_speed:
                raise StopCondition(f"step {step} (t = {step * timestep:.3f}): fluid speed {row['speed_maximum']:.4f} "
                                    f"> {arguments.stop_speed:g} x the lid speed {lid_speed:g}")
            if take_sample and arguments.stop_outside and row["outside_box_count"] > 0:
                raise StopCondition(f"step {step} (t = {step * timestep:.3f}): {row['outside_box_count']} fluid "
                                    "particle(s) outside the box")
            if take_profile:
                profiles.append({"step": step, "time": step * timestep,
                                 **centre_line_profiles(state, lower, upper, dimension, support)})
                (out / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8", newline="\n")
            tubes = None
            if take_tube or (take_lines and tube_half_width >= 1.05 * support):
                # the tubes hold every particle the line sampler reads (its square tubes have half-width 1.05 h), so
                # the lines come out the same from them, and faster
                from experiment.wall_crossing import cavity3d_refine
                tubes = cavity3d_refine.tube_state(state, box_centre, tube_half_width)
            if take_tube:
                cavity3d_refine.write_tubes(out / "tubes" / f"t{step:010d}.npz", step, time_offset + step * timestep,
                                            step * timestep, tubes, tube_half_width)
            if take_lines:
                sampled = recorder.add(step, time_offset + step * timestep, tubes if tubes is not None else state)
                if monitor is not None and step > start_step:
                    from experiment.wall_crossing.cavity3d_lines import LINES
                    fine = sampler.fine.size
                    extrema = extrema_of_profiles({name: (sampler.fine, sampled[name]["velocity"][:fine, component])
                                                   for name, _axis, component in LINES})
                    kinetic_energy = float(0.5 * (state["masses"] * speed ** 2).sum())
                    if monitor.add(step, time_offset + step * timestep, step * timestep, kinetic_energy, extrema):
                        save_checkpoint(step, simulators)
                        latest["converged_step"] = step
                        latest["end_reason"] = "converged"
                        raise Converged(f"converged at level time {step * timestep:.3f} (step {step})")
            if take_lines and arguments.wall_deadline > 0.0 and time.time() > arguments.wall_deadline                     and step > start_step:
                if checkpoint_steps or arguments.converge:
                    save_checkpoint(step, simulators)
                latest["converged_step"] = step
                latest["end_reason"] = "wall_deadline"
                raise Converged(f"wall-clock deadline passed at level time {step * timestep:.3f} (step {step}); "
                                "not converged")
            if take_lines and arguments.stop_file and step > start_step and os.path.exists(arguments.stop_file):
                save_checkpoint(step, simulators)
                latest["converged_step"] = step
                latest["end_reason"] = "stop_file"
                raise Converged(f"stop file found at level time {step * timestep:.3f} (step {step})")
            if take_frame:
                temporary = out / "frames" / f"f{step:010d}.partial.npz"
                np.savez_compressed(temporary, step=np.int64(step), time=np.float64(time_offset + step * timestep),
                                    positions=state["positions"].astype(np.float32),
                                    velocities=state["velocities"].astype(np.float32),
                                    densities=state["densities"].astype(np.float32),
                                    pressures=state["pressures"].astype(np.float32),
                                    masses=state["masses"].astype(np.float32))
                os.replace(temporary, out / "frames" / f"f{step:010d}.npz")

        def on_ready(simulators: list, slab_cases: list) -> None:
            clock["started"] = time.perf_counter()
            on_sample(start_step, simulators, slab_cases)     # the state after the bootstrap (t = 0) or restart

        def segment_line(status: str, end_step: int) -> None:
            with open(segments_path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"segment": segment, "start_step": start_step, "end_step": end_step,
                                         "status": status, "run_id": run.run_id, "unix": time.time()}) + "\n")

        try:
            # the callback is asked at every defrag boundary and decides itself (samples and profiles may have
            # different intervals)
            outcome = run_chain(case, device_map, total_steps, cadence, on_sample,
                                expected_uuids=arguments.uuid or None,
                                application_name="wall_crossing_box_run", on_ready=on_ready, restart=restart,
                                depth=arguments.depth)
        except Converged as converged:
            # report section 15: a converged level ends here, its checkpoint written; the rest as a complete run
            print(f"[box_run] {arguments.name}: {converged}", flush=True)
            outcome = {"device_uuids": [uuid.replace("-", "").lower() for uuid in arguments.uuid],
                       "elapsed_seconds": time.perf_counter() - clock["started"]}
        except StopCondition as stop:
            series_file.close()
            last_step = series[-1]["step"] if series else start_step
            (out / "watch_stop.json").write_text(json.dumps({"reason": str(stop), "step": last_step,
                                                             "time": last_step * timestep, "segment": segment,
                                                             "unix": time.time()}, indent=1), encoding="utf-8")
            if profiles:
                (out / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8", newline="\n")
            if "state" in latest:
                # the fluid state of the sample that stopped the run, to locate what triggered it
                stopped_state = latest["state"]
                np.savez_compressed(out / "stop_state.npz", step=last_step,
                                    positions=stopped_state["positions"].astype(np.float32),
                                    velocities=stopped_state["velocities"].astype(np.float32),
                                    densities=stopped_state["densities"].astype(np.float32),
                                    pressures=stopped_state["pressures"].astype(np.float32))
            segment_line(f"stopped: {stop}", last_step)
            run.extra["status"] = f"stopped: {stop}"
            print(f"[box_run] {arguments.name}: STOPPED ({stop})", flush=True)
            return STOP_EXIT_CODE
        finally:
            if not series_file.closed:
                series_file.close()
        end_step = latest.get("converged_step", total_steps)
        segment_line(latest.get("end_reason", "complete"), end_step)
        uuid = ",".join(outcome["device_uuids"])
        run.record["gpu_uuids"] = outcome["device_uuids"]
        state = latest["state"]
        elapsed = outcome["elapsed_seconds"]

        extra_final = {}
        if arguments.initial_state or arguments.converge or arguments.tube_time > 0:
            extra_final = {"masses": state["masses"].astype(np.float32)}       # report section 15: for the next level
        if arguments.diagnostics:
            extra_final = {"masses": state["masses"].astype(np.float32)}
            if "accelerations" in latest:
                extra_final["accelerations"] = latest["accelerations"].astype(np.float32)
                extra_final["acceleration_positions"] = latest["acceleration_positions"].astype(np.float32)
        np.savez_compressed(out / "final.npz", positions=state["positions"].astype(np.float32),
                            velocities=state["velocities"].astype(np.float32),
                            densities=state["densities"].astype(np.float32),
                            pressures=state["pressures"].astype(np.float32), **extra_final)
        summary = {
            "name": arguments.name, "run_id": run.run_id, "case": str(arguments.case).replace("\\", "/"),
            "wall_boundary": case.numerics.wall_boundary,
            # the wall store (branch v7-crossing-wall-store) has one scheme: the particle-pool version's names of it
            "continuity": "extrapolate" if case.numerics.wall_crossing else None,
            "continuity_wall_component": ((("normal" if case.numerics.wall_crossing.rule_set == "continuity_normal"
                                            else "full")) if case.numerics.wall_crossing else None),
            "delta_wall": "reflection" if case.numerics.wall_crossing else None,
            "viscous_extension": "linear" if case.numerics.wall_crossing else None,
            "continuity_extension": "linear" if case.numerics.wall_crossing else None,
            "rule_set": case.numerics.wall_crossing.rule_set if case.numerics.wall_crossing else None,
            "wall_store": case.wall_source is not None,
            "volume_constraint": (dataclasses.asdict(case.numerics.volume_constraint)
                                  if case.numerics.volume_constraint.enabled else None),
            "geometry": "pairwise" if case.numerics.wall_crossing else None,
            "pressure_wall": "reflection" if case.numerics.wall_crossing else None,
            "distance_floor_factor": (case.numerics.wall_crossing.distance_floor_factor
                                      if case.numerics.wall_crossing else None),
            "delta_density_storage": os.environ.get("V7_DELTA_DENSITY", "0") == "1",
            "dimension": dimension, "box_lower": lower.tolist(), "box_upper": upper.tolist(), "spacing": spacing,
            "fluid_particles": int(len(state["positions"])), "particles": expected_alive, "steps": end_step,
            "time": time_offset + end_step * timestep, "timestep": timestep,
            "level_time": end_step * timestep, "time_offset": time_offset,
            "converged": (latest.get("end_reason") == "converged") if monitor is not None else None,
            "end_reason": latest.get("end_reason", "complete"),
            "converged_level_time": monitor.converged_at if monitor is not None else None,
            "initial_state": interpolation_report,
            "volume_constraint_integral_final": (cavity3d_refine_last_integral(out / "volume_constraint.jsonl")
                                                 if case.numerics.volume_constraint.enabled else None),
            "steps_per_second": (end_step - start_step) / elapsed, "segments": segment + 1,
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
        print(f"[box_run] {arguments.name} ({run.run_id}): {end_step:,} steps to t = {summary['time']:.2f} at "
              f"{summary['steps_per_second']:.0f} steps/s; KE {summary['final']['kinetic_energy']:.5g}, "
              f"max |rho - rho0| {summary['whole_run']['density_deviation_maximum']:.4g}, phi_min "
              f"{summary['whole_run']['wall_distance_minimum_over_spacing']:.3f} dx, A_f fallbacks "
              f"{summary['whole_run']['gradient_fallbacks']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
