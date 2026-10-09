"""
runner_single.py — the host loops of v7_single: the pipelined step loop with a drained defrag boundary,
and the chain-orchestrator-shaped wrapper the cavity validation runner drives.

SingleRunner wraps one SphSimulatorSingle with the methods experiment/validation/cavity_runner.py calls
on ChainOrchestratorV7 (bootstrap_all, restart_all, run_pipelined(max_steps, depth, on_defrag),
_frame_count, workers, destroy) so the same runner code serves both solvers. Its run_pipelined keeps
``depth`` steps in flight on the simulator's fence ring, drains at every defrag boundary, reports the
status block to ``on_defrag`` and runs the defrag alone (drained), as the chain's loop does.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from experiment.v7_single.utils.simulator_single import SphSimulatorSingle


def defrag_report(sim: SphSimulatorSingle) -> list[dict]:
    """One entry (one slab) with the status block: every overflow_* counter, alive, far_migration (always 0)."""
    status = sim.readback_global_status()
    entry = dict(status)
    entry["interval_migration"] = 0
    entry["far_migration"] = 0
    entry["alive"] = status["alive_particle_count"]
    return [entry]


def run_steps(sim: SphSimulatorSingle, max_steps: int, *, depth: int = 1, warmup: int = 0,
              defrag_cadence: Optional[int] = None, on_defrag: Optional[Callable[[int, list], None]] = None,
              drained_defrag: bool = True) -> dict:
    """Run ``max_steps`` steps with ``depth`` in flight. A defrag every ``defrag_cadence`` steps (None = the case's
    cadence; 0 = never): drained, reported to ``on_defrag(frame, report)`` and submitted alone (drained_defrag)
    or appended to the step's submission. Returns {frame_count, elapsed_s, fps, steady_*} measured like the
    chain's run_pipelined: the steady window opens after the warmup-th step is submitted and closes after the
    final drain."""
    if defrag_cadence is None:
        defrag_cadence = int(sim.case.numerics.defrag_cadence)
    if sim.in_flight != depth:
        sim.prepare_step(in_flight=depth)
    t_start = time.perf_counter()
    warmup_t = None
    warmup_frame = None
    for frame in range(1, max_steps + 1):
        boundary = bool(defrag_cadence) and frame % defrag_cadence == 0
        if boundary and not drained_defrag:
            sim.submit_step(with_defrag=True)
        else:
            sim.submit_step()
        if warmup_t is None and frame >= warmup:
            warmup_t = time.perf_counter()
            warmup_frame = frame
        if boundary and drained_defrag:
            sim.drain()
            if on_defrag is not None:
                on_defrag(frame, defrag_report(sim))
            sim.submit_defrag_and_wait()
    sim.drain()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    out = {"frame_count": max_steps, "elapsed_s": elapsed, "fps": max_steps / elapsed if elapsed > 0 else 0.0}
    if warmup_t is not None:
        steady_frames = max_steps - warmup_frame
        steady_s = t_end - warmup_t
        out.update({"steady_frames": steady_frames, "steady_s": steady_s,
                    "steady_fps": steady_frames / steady_s if steady_s > 0 else 0.0})
    return out


class SingleRunner:
    """The ChainOrchestrator-shaped driver of one SphSimulatorSingle (module docstring)."""

    def __init__(self, sim: SphSimulatorSingle, defrag_cadence: Optional[int] = None) -> None:
        self.sim = sim
        self.sims = [sim]
        self.defrag_cadence = int(sim.case.numerics.defrag_cadence if defrag_cadence is None else defrag_cadence)
        self.workers: tuple = ()
        self._frame_count = 0

    def bootstrap_all(self) -> None:
        """Bootstrap, record the step command buffers and (as the chain does) defrag once before the first step."""
        self.sim.bootstrap()
        self.sim.prepare_step(in_flight=1)
        self.sim.submit_defrag_and_wait()

    def restart_all(self, states: list, stored_density_offset: float = 0.0) -> None:
        if len(states) != 1:
            raise ValueError(f"{len(states)} restart states for one simulator")
        self.sim.restart_init(states[0], stored_density_offset)
        self.sim.prepare_step(in_flight=1)
        self.sim.submit_defrag_and_wait()

    def run_pipelined(self, max_steps: int, *, depth: int = 2, warmup: int = 0,
                      on_defrag: Optional[Callable[[int, list], None]] = None, **_ignored) -> dict:
        self._frame_count = 0

        def counted(frame: int, report: list) -> None:
            self._frame_count = frame
            if on_defrag is not None:
                on_defrag(frame, report)
        result = run_steps(self.sim, max_steps, depth=depth, warmup=warmup, defrag_cadence=self.defrag_cadence,
                           on_defrag=counted, drained_defrag=True)
        self._frame_count = max_steps
        return result

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def destroy(self) -> None:
        pass

    def __enter__(self) -> "SingleRunner":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
