"""
cavity_runner_entry.py - experiment/validation/cavity_runner.py with three additions that need no change of the
runner: it starts on a machine without git, a wall_boundary crossing run records the mode's diagnostics, and
every run records the history of the fluid's mean density and pressure.

    python -m experiment.wall_crossing.cavity_runner_entry <cavity_runner arguments>

1. git. The runner records the repository state (git rev-parse / git status) in meta.json and segments.jsonl; on
   a deployed copy without git that call raises before the run starts. Here the runner's git_state falls back to
   the COMMIT file of the deployment (run_log.git_commit). With git present the runner's own function answers.

2. Diagnostics of wall_boundary crossing. The runner reads the solver's global status at every defrag boundary
   (frames drained) and at the end of a segment; for a crossing simulator that read also appends one line to
   <run-dir>/wall_crossing_status.jsonl:
       {"unix", "frame_stamp" (steps of this segment), "gradient_fallbacks" (cumulative count of fluid
        particles whose A_f fell back), "correction_fallbacks"}
   and, every POSITION_CHECK_INTERVAL-th read, the fluid's smallest wall distance from a position readback:
       "wall_distance_minimum", "below_floor_count", "outside_box_count", "fluid_count".
   The runner's own results are untouched (the hook returns the status it was asked for).

3. Field history (any wall mode). The runner reads every alive particle at each sample (every 0.2 time units)
   and at each snapshot (CachedStateReader.read: position, velocity, density, pressure, material in one staging
   copy) but keeps only the kinetic energy, the largest speed and the density range. Here that read also appends
   one line to <run-dir>/field_history.jsonl, over the alive fluid particles (no wall), float64 means:
       {"unix", "step", "time", "segment" (the heartbeat the runner wrote just before the read), "alive" (all
        particles), "fluid_particles", "density_mean", "density_minimum", "density_maximum" (the stored value
        plus the storage offset), "pressure_mean", "pressure_minimum", "pressure_maximum",
        "pressure_negative_fraction", "speed_maximum"}
   A step that is both a sample and a snapshot has two lines. The state the runner gets is the one it asked for.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.validation import cavity_runner                 # noqa: E402
from experiment.wall_crossing.run_log import git_commit         # noqa: E402

POSITION_CHECK_INTERVAL = 20        # status reads between two position readbacks (20 x 1000 steps)

_runner_git_state = cavity_runner.git_state


def git_state_or_commit_file(solver: str = "v6") -> dict:
    try:
        state = _runner_git_state(solver)
        if state.get("head"):
            return state
    except OSError:
        pass
    commit = git_commit(REPOSITORY_ROOT)
    return {"head": commit.replace("+dirty", ""), "dirty": commit.endswith("+dirty"), "source": "COMMIT file"}


cavity_runner.git_state = git_state_or_commit_file


def _argument(name: str):
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def install_wall_crossing_diagnostics(run_directory: pathlib.Path) -> None:
    """Wrap SphSimulatorV7.readback_global_status (see the module docstring, item 2)."""
    import numpy as np

    from experiment.v7.utils import simulator_v7
    from experiment.v7.utils.case_v7 import KIND_FLUID
    from experiment.v7.utils.wall_crossing_v7 import fluid_distance_statistics

    original = simulator_v7.SphSimulatorV7.readback_global_status
    counters = {"reads": 0}
    path = run_directory / "wall_crossing_status.jsonl"

    def readback_global_status_recording(self):
        status = original(self)
        try:
            if not self._wall_crossing:
                return status
            record = {"unix": time.time(), "frame_stamp": int(status.get("frame_stamp", 0)),
                      "gradient_fallbacks": int(status.get("wall_crossing_gradient_fallback_count", 0)),
                      "correction_fallbacks": int(status.get("correction_fallback_count", 0))}
            if counters["reads"] % POSITION_CHECK_INTERVAL == 0:
                case = self.case
                raw = self.readback_buffers_batch(["position_voxel_id", "material"])
                capacity = case.capacities.total_pool_capacity()
                first = self.own_first_pid()
                rows = slice(first, first + case.capacities.own_pool_size)
                position = np.frombuffer(raw["position_voxel_id"], dtype=np.float32)[:capacity * 4] \
                    .reshape(capacity, 4)[rows]
                material = np.frombuffer(raw["material"], dtype=np.uint32)[:capacity][rows]
                kinds = np.asarray([entry.kind for entry in case.materials])
                fluid = (position[:, 3] > 0.5) & (kinds[np.minimum(material, kinds.size - 1)] == KIND_FLUID)
                radius = self.wall_crossing_particle_radius()
                statistics = fluid_distance_statistics(position[:, 0:3], fluid, case.numerics.wall_crossing, radius,
                                                       int(case.physics.dimension))
                record.update({"wall_distance_minimum": statistics["distance_minimum"],
                               "below_floor_count": statistics["below_floor_count"],
                               "outside_box_count": statistics["outside_box_count"],
                               "fluid_count": statistics["fluid_count"]})
            counters["reads"] += 1
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as error:          # a diagnostic must never end a run
            print(f"[cavity_runner_entry] wall_crossing diagnostics skipped: {error!r}", flush=True)
        return status

    simulator_v7.SphSimulatorV7.readback_global_status = readback_global_status_recording


def install_field_history(run_directory: pathlib.Path) -> None:
    """Wrap cavity_runner.CachedStateReader.read (see the module docstring, item 3)."""
    import numpy as np

    from experiment.v7.utils.case_v7 import KIND_FLUID

    original = cavity_runner.CachedStateReader.read
    path = run_directory / "field_history.jsonl"
    heartbeat_path = run_directory / "heartbeat.json"

    def read_recording(self):
        state = original(self)
        try:
            # the staging copy the runner's read just filled
            count = self.count
            position = np.frombuffer(self.view, np.float32, 4 * count, self.offsets["position_voxel_id"]).reshape(count, 4)
            velocity = np.frombuffer(self.view, np.float32, 4 * count, self.offsets["velocity_mass"]).reshape(count, 4)
            density = np.frombuffer(self.view, np.float32, 2 * count, self.offsets["density_pressure"]).reshape(count, 2)
            material = np.frombuffer(self.view, np.uint32, count, self.offsets["material"])
            alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
            kinds = np.asarray([entry.kind for entry in self.sim.case.materials])
            fluid = alive & (kinds[np.minimum(material, kinds.size - 1)] == KIND_FLUID)
            true_density = density[fluid, 0].astype(np.float64) + float(self.sim.stored_density_offset())
            pressure = density[fluid, 1].astype(np.float64)
            speed_squared = (velocity[fluid, 0:3].astype(np.float64) ** 2).sum(axis=1)
            heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8")) if heartbeat_path.exists() else {}
            record = {"unix": time.time(), "step": heartbeat.get("step"), "time": heartbeat.get("time"),
                      "segment": heartbeat.get("segment"), "alive": int(alive.sum()),
                      "fluid_particles": int(fluid.sum()),
                      "density_mean": float(true_density.mean()), "density_minimum": float(true_density.min()),
                      "density_maximum": float(true_density.max()),
                      "pressure_mean": float(pressure.mean()), "pressure_minimum": float(pressure.min()),
                      "pressure_maximum": float(pressure.max()),
                      "pressure_negative_fraction": float((pressure < 0.0).mean()),
                      "speed_maximum": float(np.sqrt(speed_squared.max()))}
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as error:          # a diagnostic must never end a run
            print(f"[cavity_runner_entry] field history skipped: {error!r}", flush=True)
        return state

    cavity_runner.CachedStateReader.read = read_recording


if __name__ == "__main__":
    if _argument("--solver") == "v7" and _argument("--run-dir"):
        install_wall_crossing_diagnostics(pathlib.Path(_argument("--run-dir")).resolve())
        install_field_history(pathlib.Path(_argument("--run-dir")).resolve())
    sys.exit(cavity_runner.main())
