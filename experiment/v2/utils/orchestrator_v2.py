"""
orchestrator_v2.py — Cross-GPU orchestrator for V2's 3-submit-per-frame pattern.

Owns 2 SphSimulatorV2 (borrows, does not destroy) + 2 GhostMigrationWorker
(owns). Drives frame loop, collects instrumentation timestamps, dispatches
defrag at cadence boundaries. LoadBalancer hook is reserved for v1.x
(docs/sph_v2_design.md §11.3) — v1.0 uses static partition only.

Frame loop (synchronous depth=1; depth>1 frame pipelining is §5.4 future):

    n = self.frame_count
    for w in self.workers:  w.notify(n)
    for sim in self.sims:   sim.submit_phase_a(n)
    for sim in self.sims:   sim.submit_phase_b(n)
    for sim in self.sims:   sim.submit_phase_c(n)
    for sim in self.sims:   sim.wait_frame_done(n)
    if defrag_due(n):
        for sim in self.sims:    sim.submit_defrag_and_wait()
    self.frame_count += 1
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from experiment.v2.utils.transport_v2 import GhostMigrationWorker

if TYPE_CHECKING:
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2


class DualGpuOrchestratorV2:
    """V2 dual-GPU orchestrator. Sim ownership: borrows only."""

    def __init__(
        self,
        sim_a: "SphSimulatorV2",
        sim_b: "SphSimulatorV2",
        *,
        defrag_cadence: int = 1000,
    ) -> None:
        self.sim_a = sim_a
        self.sim_b = sim_b
        self.sims = (sim_a, sim_b)
        self.defrag_cadence = defrag_cadence

        # Pathway A→B: sim_a sends in its peer direction; sim_b receives in
        # the opposite direction. For the standard 2-GPU 1D layout:
        #   sim_a = leftmost  → has trailing peer (sim_b)
        #   sim_b = rightmost → has leading peer  (sim_a)
        # So worker_a_to_b: source_dir=trailing, dest_dir=leading
        #    worker_b_to_a: source_dir=leading,  dest_dir=trailing
        # Validate before constructing workers.
        if not sim_a.case.transport.has_trailing_peer:
            raise ValueError("sim_a must have trailing peer (leftmost slot)")
        if not sim_b.case.transport.has_leading_peer:
            raise ValueError("sim_b must have leading peer (rightmost slot)")

        self.worker_a_to_b = GhostMigrationWorker(
            source_sim=sim_a, dest_sim=sim_b,
            source_direction="trailing", dest_direction="leading",
            label="a_to_b")
        self.worker_b_to_a = GhostMigrationWorker(
            source_sim=sim_b, dest_sim=sim_a,
            source_direction="leading", dest_direction="trailing",
            label="b_to_a")
        self.workers = (self.worker_a_to_b, self.worker_b_to_a)

        self._frame_count = 0
        self._records: list[dict] = []
        self._destroyed = False

        for w in self.workers:
            w.start()
        print(f"[OrchV2] workers started ({self.worker_a_to_b.label}, "
              f"{self.worker_b_to_a.label})")

    def destroy(self) -> None:
        if self._destroyed:
            return
        for w in self.workers:
            w.stop()
        self._destroyed = True

    def __enter__(self) -> "DualGpuOrchestratorV2":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Frame loop
    # ========================================================================

    def bootstrap_all(self) -> None:
        """Dual-GPU bootstrap with one synchronous ghost round.

        Without the ghost round, step 1's correction/density/force on boundary
        particles would see EMPTY ghost neighbors → density underestimate →
        negative pressure → wild attractive forces → particles drift across
        the slab and overflow corner voxels by step 10–20. Symptom: vid=1
        overflow_incoming spikes. Fix matches V1's _record_bootstrap_pre_sync
        / _record_bootstrap_post_sync split.

        Sequence:
          1. Per sim: bootstrap_init (init_voxelization + ghost_send + readback)
          2. Inline CPU memcpy: sim0.sender_staging_trailing → sim1.receiver_staging_leading,
                                 sim1.sender_staging_leading  → sim0.receiver_staging_trailing
             (replicas only — no particles have moved yet, so migrations are empty)
          3. Per sim: bootstrap_compute (upload + install_migrations + correction +
             density + force + bootstrap_half_kick)
          4. Per sim: prepare_step_cmd_buffers (record phase A/B/C for run loop)
        """
        # Stage 1: init + ghost_send on both GPUs
        for sim in self.sims:
            sim.bootstrap_init()

        # Stage 2: host bridge — exchange replicas between the two staging pairs
        # Use the same view shapes the workers use during steady-state to
        # validate the staging size match.
        self.sim_b.receiver_staging_view("leading")[:] = self.sim_a.sender_staging_view("trailing")
        self.sim_a.receiver_staging_view("trailing")[:] = self.sim_b.sender_staging_view("leading")
        print(f"[OrchV2] bootstrap ghost sync: bridged "
              f"{self.sim_a.sender_staging_view('trailing').nbytes / 1024:.1f} KB each way")

        # Stage 3: install + correction + density + force + half_kick
        for sim in self.sims:
            sim.bootstrap_compute()

        # Stage 4: record per-frame cmd buffers (phase A/B/C)
        for sim in self.sims:
            sim.prepare_step_cmd_buffers()
        print(f"[OrchV2] both sims bootstrapped + step cmds ready")

    def step(self) -> dict:
        """Run one frame, return per-frame instrumentation record."""
        n = self._frame_count
        t_start = time.perf_counter_ns()

        # 1. Notify workers (they wait for phase_a_done internally)
        for w in self.workers:
            w.notify(n)

        # 2. Submit phase A on both sims (signal 3N+1 each)
        for sim in self.sims:
            sim.submit_phase_a(n)

        # 3. Submit phase B on both sims (queue-ordered after A; no sem ops)
        for sim in self.sims:
            sim.submit_phase_b(n)

        # 4. Submit phase C on both sims (wait 3N+2 = signaled by worker,
        #    signal 3N+3 at end)
        for sim in self.sims:
            sim.submit_phase_c(n)

        # 5. Wait for both sims to reach 3N+3 (frame done).
        # Use a watchdog poll instead of INFINITE wait so a dead worker thread
        # surfaces as a clear error instead of an opaque hang. Each poll is
        # 1s; we re-check worker health between polls.
        #
        # TODO(refactor): this block uses for-else + inline magic constants +
        # a long f-string print. Split into helpers:
        #   _wait_frame_done_with_watchdog(sim_idx, sim, frame_n)
        #   _raise_if_worker_died(frame_n)
        #   _log_watchdog(frame_n, sim_idx, target, poll)
        # Hoist constants to class level. Drops step() body from ~35 lines to
        # ~3 lines for this section. Functional behavior unchanged.
        WATCHDOG_TIMEOUT_S = 1.0
        MAX_POLL_ITERS = 10   # → 10s total before giving up
        for sim_idx, sim in enumerate(self.sims):
            target_value = sim.value_frame_done(n)
            for poll_iter in range(MAX_POLL_ITERS):
                done = sim.wait_timeline(
                    target_value, timeout_ns=int(WATCHDOG_TIMEOUT_S * 1e9))
                if done:
                    break
                # Check workers
                for w in self.workers:
                    if w.last_error is not None:
                        raise RuntimeError(
                            f"worker {w.label} died during frame {n}: "
                            f"{w.last_error}") from w.last_error
                # Workers alive but timeline stuck → likely GPU TDR or kernel hang
                cur_a = self.sim_a.current_timeline_value()
                cur_b = self.sim_b.current_timeline_value()
                wa = self.worker_a_to_b
                wb = self.worker_b_to_a
                print(f"[OrchV2 WATCHDOG] frame {n}: waiting sim_{('a','b')[sim_idx]} "
                      f"to reach {target_value}; cur a={cur_a} b={cur_b}; "
                      f"a→b iters={wa.iteration_count} last_done={wa.last_completed_frame} "
                      f"phase={wa.last_activity[0]}@frame{wa.last_activity[1]}; "
                      f"b→a iters={wb.iteration_count} last_done={wb.last_completed_frame} "
                      f"phase={wb.last_activity[0]}@frame{wb.last_activity[1]}; "
                      f"poll {poll_iter+1}/{MAX_POLL_ITERS}", flush=True)
            else:
                # Loop exhausted without success
                raise RuntimeError(
                    f"frame {n}: sim_{('a','b')[sim_idx]} timeline stuck at "
                    f"{self.sims[sim_idx].current_timeline_value()} "
                    f"(target {target_value}) after {MAX_POLL_ITERS * WATCHDOG_TIMEOUT_S}s. "
                    f"Likely GPU device-lost / kernel hang.")
        t_end = time.perf_counter_ns()

        # Check worker health one more time (catches workers that died right
        # at the end of the frame).
        for w in self.workers:
            if w.last_error is not None:
                raise RuntimeError(
                    f"worker {w.label} died: {w.last_error}") from w.last_error

        record = {
            "frame_n": n,
            "frame_start_ns": t_start,
            "frame_end_ns": t_end,
            "frame_time_us": (t_end - t_start) / 1000.0,
            "worker_a_to_b": self.worker_a_to_b.timestamps_for_frame(n),
            "worker_b_to_a": self.worker_b_to_a.timestamps_for_frame(n),
        }
        self._records.append(record)
        self._frame_count += 1

        # 6. Defrag if due (independent fence-wait pass; not in timeline)
        if (self._frame_count > 0
                and self._frame_count % self.defrag_cadence == 0):
            for sim in self.sims:
                sim.submit_defrag_and_wait()

        return record

    def run_until(
        self,
        max_steps: Optional[int] = None,
    ) -> None:
        if max_steps is None:
            raise ValueError("v1.0 requires max_steps (no time budget yet)")
        while self._frame_count < max_steps:
            self.step()

    # ========================================================================
    # Inspection
    # ========================================================================

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def instrumentation_records(self) -> list[dict]:
        return list(self._records)
