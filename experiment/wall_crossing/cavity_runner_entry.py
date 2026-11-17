"""
cavity_runner_entry.py - experiment/validation/cavity_runner.py with two additions that need no change of the
runner: it starts on a machine without git, and a wall_boundary crossing run records the mode's diagnostics.

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


if __name__ == "__main__":
    if _argument("--solver") == "v7" and _argument("--run-dir"):
        install_wall_crossing_diagnostics(pathlib.Path(_argument("--run-dir")).resolve())
    sys.exit(cavity_runner.main())
