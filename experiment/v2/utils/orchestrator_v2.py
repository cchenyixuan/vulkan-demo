"""
orchestrator_v2.py — Cross-GPU orchestrator for V2's 3-submit-per-frame pattern.

Owns 2 SphSimulatorV2 (borrows, does not destroy) + 2 GhostMigrationWorker
(owns). Drives frame loop, collects instrumentation timestamps, dispatches
defrag at cadence boundaries. LoadBalancer hook is reserved for v1.x
(docs/sph_v2_design.md §11.3) — v1.0 uses static partition only.

Frame loop (synchronous, depth=1; depth>1 frame pipelining is §5.4 future):

    n = self.frame_count
    for worker in self.workers:  worker.notify(n)
    for sim in self.sims:        sim.submit_phase_a(n)
    for sim in self.sims:        sim.submit_phase_b(n)
    for sim in self.sims:        sim.submit_phase_c(n)
    for sim in self.sims:        sim.wait_frame_done(n)
    self._collect_instrumentation(n)
    if defrag_due(n):
        for sim in self.sims:    sim.submit_defrag_and_wait()
    self.frame_count += 1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2
    from experiment.v2.utils.transport_v2 import GhostMigrationWorker


class DualGpuOrchestratorV2:
    """V2 dual-GPU orchestrator.

    Lifecycle:
        sim_a = SphSimulatorV2(...); sim_b = SphSimulatorV2(...)
        with DualGpuOrchestratorV2(sim_a, sim_b) as orch:
            orch.bootstrap_all()
            orch.run_until(max_steps=1000)
        sim_a.destroy(); sim_b.destroy()
    """

    # ========================================================================
    # Construction / destruction
    # ========================================================================

    def __init__(
        self,
        sim_a: "SphSimulatorV2",
        sim_b: "SphSimulatorV2",
        *,
        defrag_cadence: int = 1000,
    ) -> None:
        # Phase 4 fills in: build 2 GhostMigrationWorker (a→b, b→a), start them.
        raise NotImplementedError("Phase 4")

    def destroy(self) -> None:
        """Stop workers, vkDeviceWaitIdle on both sims. Does NOT destroy sims —
        caller's responsibility (sims pre-exist orchestrator, may outlive it)."""
        raise NotImplementedError("Phase 4")

    def __enter__(self) -> "DualGpuOrchestratorV2":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Frame loop
    # ========================================================================

    def bootstrap_all(self) -> None:
        """Bootstrap both sims sequentially. Each sim's bootstrap is fence-
        waited (synchronous); cross-GPU ghost on bootstrap step is handled by
        an explicit single-shot worker run after both sims complete."""
        raise NotImplementedError("Phase 4")

    def step(self) -> dict:
        """One frame. Returns per-frame instrumentation record:
            {
                "frame_n": int,
                "t_frame_start_ns": int,
                "t_a_phase_a_done_ns": int / None,   (None if not measured)
                "t_b_phase_a_done_ns": int / None,
                "t_a_cpu_sync_done_ns": int / None,
                "t_b_cpu_sync_done_ns": int / None,
                "t_a_frame_done_ns": int,
                "t_b_frame_done_ns": int,
            }

        Most timestamps come from worker dicts; the frame_done ones are taken
        by the orchestrator after vkWaitSemaphores returns. See §10.
        """
        raise NotImplementedError("Phase 4")

    def run_until(
        self,
        total_time: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Drive step() until either condition is met. v1.0: caller must
        provide at least one bound (no time-budget abstraction)."""
        raise NotImplementedError("Phase 4")

    # ========================================================================
    # Inspection
    # ========================================================================

    @property
    def frame_count(self) -> int:
        raise NotImplementedError("Phase 4")

    def instrumentation_records(self) -> list:
        """All collected per-frame records since construction. Caller can
        feed this to pandas / matplotlib for offline analysis."""
        raise NotImplementedError("Phase 4")

    # ========================================================================
    # LoadBalancer hook (v1.x; v1.0 is no-op)
    # ========================================================================

    def _maybe_adjust_partition(self) -> None:
        """Called before defrag every defrag_cadence frames. v1.0: no-op.
        v1.x will accumulate wait_X from instrumentation and adjust partition.x
        per docs/sph_v2_design.md §11.3."""
        # v1.0 intentional no-op; keep the call site so v1.x just fills the body.
        pass
