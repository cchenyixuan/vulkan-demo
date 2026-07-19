"""
orchestrator_v5.py — Cross-GPU orchestrator for V5's 3-submit-per-frame pattern.

Owns 2 SphSimulatorV5 (borrows, does not destroy) + 2 GhostMigrationWorker
(owns). Drives frame loop, collects instrumentation timestamps, dispatches
defrag at cadence boundaries. LoadBalancer hook is reserved for v1.x
(docs/sph_v5_design.md §11.3) — v1.0 uses static partition only.

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

from experiment.v5.utils.transport_v5 import GhostMigrationWorker

if TYPE_CHECKING:
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5


class DualGpuOrchestratorV5:
    """V5 dual-GPU orchestrator. Sim ownership: borrows only."""

    def __init__(
        self,
        sim_a: "SphSimulatorV5",
        sim_b: "SphSimulatorV5",
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
        print(f"[OrchV5] workers started ({self.worker_a_to_b.label}, "
              f"{self.worker_b_to_a.label})")

    def destroy(self) -> None:
        if self._destroyed:
            return
        for w in self.workers:
            w.stop()
        self._destroyed = True

    def __enter__(self) -> "DualGpuOrchestratorV5":
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
        print(f"[OrchV5] bootstrap ghost sync: bridged "
              f"{self.sim_a.sender_staging_view('trailing').nbytes / 1024:.1f} KB each way")

        # Stage 3: install + correction + density + force + half_kick
        for sim in self.sims:
            sim.bootstrap_compute()

        # Stage 4: record per-frame cmd buffers (phase A/B/C)
        for sim in self.sims:
            sim.prepare_step_cmd_buffers()
        print(f"[OrchV5] both sims bootstrapped + step cmds ready")

    def step(self) -> dict:
        """Run one frame, return per-frame instrumentation record."""
        n = self._frame_count
        t_start = time.perf_counter_ns()

        # 1. Notify workers (they wait for readback_done(n) internally)
        for w in self.workers:
            w.notify(n)

        # 2. Compute Q: submit phase A on both sims (signal phase_a_done)
        for sim in self.sims:
            sim.submit_phase_a(n)

        # 3. Transfer Q: submit readback DMAs (wait phase_a_done, signal
        #    readback_done). Runs in parallel with Phase B on the compute queue.
        for sim in self.sims:
            sim.submit_transfer_readback(n)

        # 4. Compute Q: submit phase B (queue-ordered after A on compute Q;
        #    no semaphore ops). Hides transfer chain behind correction_
        #    interior + density_deep_interior.
        for sim in self.sims:
            sim.submit_phase_b(n)

        # 5. Transfer Q: submit upload DMAs (wait worker_done = worker
        #    host-signal, signal upload_done). Submitted ahead of time;
        #    transfer queue blocks until worker memcpy completes.
        for sim in self.sims:
            sim.submit_transfer_upload(n)

        # 6. Compute Q: submit phase C (wait upload_done, signal frame_done)
        for sim in self.sims:
            sim.submit_phase_c(n)

        # 7. Wait for both sims to reach frame_done(n).
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
            frame_semaphore, target_value = sim.sync.frame_done_op(n)
            for poll_iter in range(MAX_POLL_ITERS):
                done = sim.wait_semaphore(
                    frame_semaphore, target_value,
                    timeout_ns=int(WATCHDOG_TIMEOUT_S * 1e9))
                if done:
                    break
                # Check workers
                for w in self.workers:
                    if w.last_error is not None:
                        raise RuntimeError(
                            f"worker {w.label} died during frame {n}: "
                            f"{w.last_error}") from w.last_error
                # Workers alive but timeline stuck → likely GPU TDR or kernel hang
                state_a = self.sim_a.sync_state()
                state_b = self.sim_b.sync_state()
                wa = self.worker_a_to_b
                wb = self.worker_b_to_a
                print(f"[OrchV5 WATCHDOG] frame {n}: waiting sim_{('a','b')[sim_idx]} "
                      f"to reach {target_value}; cur a={state_a} b={state_b}; "
                      f"a→b iters={wa.iteration_count} last_done={wa.last_completed_frame} "
                      f"phase={wa.last_activity[0]}@frame{wa.last_activity[1]}; "
                      f"b→a iters={wb.iteration_count} last_done={wb.last_completed_frame} "
                      f"phase={wb.last_activity[0]}@frame{wb.last_activity[1]}; "
                      f"poll {poll_iter+1}/{MAX_POLL_ITERS}", flush=True)
            else:
                # Loop exhausted without success
                raise RuntimeError(
                    f"frame {n}: sim_{('a','b')[sim_idx]} timeline stuck at "
                    f"{self.sims[sim_idx].sync_state()} "
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

        # 6. Defrag if due (independent fence-wait pass; not in timeline).
        if (self._frame_count > 0
                and self._frame_count % self.defrag_cadence == 0):
            record["defrag_frame"] = self._frame_count
            record["defrag_report"] = self._collect_defrag_report()
            for sim in self.sims:
                sim.submit_defrag_and_wait()

        return record

    def _collect_defrag_report(self) -> list[dict]:
        """Snapshot BEFORE defrag resets migration_install_count: per-interval
        migration total (particles installed at the own-pool tail since the last
        defrag) + the never-reset pool-health peaks. 2 tiny readbacks per defrag
        (once per defrag_cadence frames). Callers print the migration series."""
        report = []
        for sim in self.sims:
            status = sim.readback_global_status()
            health = sim.readback_pool_health()
            report.append({
                "interval_migration":     status["migration_install_count"],
                "alive":                  status["alive_particle_count"],
                "overflow_install_tail":  status["overflow_install_tail"],
                "peak_migration":         health["peak_migration_count"],
                "peak_tail":              health["peak_tail_high_water"],
                "own_pool":               health["own_pool_size"],
                "used_fraction":          health["used_fraction"],
            })
        return report

    def run_until(
        self,
        max_steps: Optional[int] = None,
    ) -> None:
        if max_steps is None:
            raise ValueError("v1.0 requires max_steps (no time budget yet)")
        while self._frame_count < max_steps:
            self.step()

    # ========================================================================
    # Submit-ahead pipelined run (V5: depth>1 frame pipelining, §5.4)
    # ========================================================================

    def _submit_frame(self, n: int) -> None:
        """Submit all of frame n's work (no wait). The sync-scheme timelines
        enforce the GPU-side ordering — phase_a(n) waits frame_done(n-1) etc. —
        so it is safe to queue a frame before the previous one finished on the
        GPU."""
        for w in self.workers:
            w.notify(n)
        for sim in self.sims:
            sim.submit_phase_a(n)
        for sim in self.sims:
            sim.submit_transfer_readback(n)
        for sim in self.sims:
            sim.submit_phase_b(n)
        for sim in self.sims:
            sim.submit_transfer_upload(n)
        for sim in self.sims:
            sim.submit_phase_c(n)

    def _wait_frame(self, n: int) -> None:
        """Block until both sims signal frame_done(n); fail fast if a worker died."""
        for sim in self.sims:
            sim.wait_frame_done(n)
        for w in self.workers:
            if w.last_error is not None:
                raise RuntimeError(
                    f"worker {w.label} died (frame {n}): {w.last_error}"
                ) from w.last_error

    def run_pipelined(
        self,
        max_steps: int,
        *,
        depth: int = 2,
        warmup: int = 0,
        on_defrag=None,
    ) -> dict:
        """Submit-ahead pipelined run loop. Keeps ``depth`` frames in flight so
        the GPU never idles on CPU submit latency / the inter-frame bubble that
        the synchronous depth-1 ``step()`` pays every frame.

        SAFETY (single-buffered state, validated drift=0 at depth 2 and 3):
        the sync-scheme timelines make worker(n)'s host-signal a prerequisite
        for frame n+1's readback: readback(n+1) ← phase_a(n+1) ←
        frame_done(n) ← phase_c(n) ← upload_done(n) ← worker_done(n). The
        chain holds in both schemes (aggregated: all on one timeline;
        per-direction: through main + the direction's transport timeline).
        So no staging buffer is reused before its reader finishes, and no
        host-signal ever goes backwards, at any depth.

        NOTE: does NOT collect per-kernel GPU timestamps — the in-flight next
        frame would overwrite the query-pool slots. Use the synchronous ``step()``
        path (depth-1) when per-stage timing is needed. This loop measures
        wall-clock fps only.

        ``on_defrag(frame_n, report)`` fires at each defrag boundary (after the
        pipeline drains, before the counter resets) for migration-series logging.

        Returns {frame_count, elapsed_s, fps, steady_frames, steady_s, steady_fps}.
        """
        depth = max(1, depth)
        t_start = time.perf_counter()
        warmup_t = None
        warmup_frame = None
        n = 0
        next_wait = 0
        while n < max_steps:
            self._submit_frame(n)
            n += 1
            self._frame_count = n
            # Bound in-flight depth.
            while n - next_wait >= depth:
                self._wait_frame(next_wait)
                next_wait += 1
            if warmup_t is None and n >= warmup:
                warmup_t = time.perf_counter()
                warmup_frame = n
            # Defrag boundary: drain the pipeline, snapshot migration, defrag.
            if n % self.defrag_cadence == 0:
                while next_wait < n:
                    self._wait_frame(next_wait)
                    next_wait += 1
                report = self._collect_defrag_report()
                if on_defrag is not None:
                    on_defrag(n, report)
                for sim in self.sims:
                    sim.submit_defrag_and_wait()
        # Drain remaining in-flight frames.
        while next_wait < max_steps:
            self._wait_frame(next_wait)
            next_wait += 1
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        out = {
            "frame_count": max_steps,
            "elapsed_s": elapsed,
            "fps": max_steps / elapsed if elapsed > 0 else 0.0,
        }
        if warmup_t is not None:
            steady_frames = max_steps - warmup_frame
            steady_s = t_end - warmup_t
            out.update({
                "steady_frames": steady_frames,
                "steady_s": steady_s,
                "steady_fps": steady_frames / steady_s if steady_s > 0 else 0.0,
            })
        return out

    # ========================================================================
    # Inspection
    # ========================================================================

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def instrumentation_records(self) -> list[dict]:
        return list(self._records)


# ============================================================================
# M3: N-slab chain orchestrator (docs/sph_v5_design.md §3.3)
#
# DualGpuOrchestratorV5 above stays UNTOUCHED as the K=2 reference for the
# M3.3 A/B regression (same philosophy as legacy_dual_gpu_partition in M2).
# This class is the general implementation used going forward: N sims,
# 2(N-1) ghost workers built from the chain topology, a generic bootstrap
# bridge, and notify-last submission with queue_depth>1 so a slow link
# cannot head-of-line-block the submit loop.
# ============================================================================


class ChainOrchestratorV5:
    """N-slab 1D chain orchestrator. Sim ownership: borrows only.

    sims[i] must be slab i of a ChainPartition (left to right): its
    transport flags must be has_leading_peer == (i > 0) and
    has_trailing_peer == (i < N-1). Interior sims require
    sync_scheme='per-direction' (the aggregated scheme raises at sim
    construction — shared worker_done slot)."""

    WATCHDOG_TIMEOUT_S = 1.0
    MAX_POLL_ITERS = 10

    def __init__(
        self,
        sims: list,
        *,
        defrag_cadence: int = 1000,
        worker_queue_depth: int = 4,
    ) -> None:
        self.sims = list(sims)
        slab_count = len(self.sims)
        if slab_count < 1:
            raise ValueError("need at least one sim")
        self.defrag_cadence = defrag_cadence

        for index, sim in enumerate(self.sims):
            transport = sim.case.transport
            if transport.has_leading_peer != (index > 0):
                raise ValueError(
                    f"sim {index}: has_leading_peer="
                    f"{transport.has_leading_peer} does not match chain "
                    f"position (expected {index > 0})")
            if transport.has_trailing_peer != (index < slab_count - 1):
                raise ValueError(
                    f"sim {index}: has_trailing_peer="
                    f"{transport.has_trailing_peer} does not match chain "
                    f"position (expected {index < slab_count - 1})")

        # 2(N-1) directed ghost workers, one pair per adjacent link.
        workers = []
        for index in range(slab_count - 1):
            workers.append(GhostMigrationWorker(
                source_sim=self.sims[index], dest_sim=self.sims[index + 1],
                source_direction="trailing", dest_direction="leading",
                label=f"s{index}_to_s{index + 1}",
                queue_depth=worker_queue_depth))
            workers.append(GhostMigrationWorker(
                source_sim=self.sims[index + 1], dest_sim=self.sims[index],
                source_direction="leading", dest_direction="trailing",
                label=f"s{index + 1}_to_s{index}",
                queue_depth=worker_queue_depth))
        self.workers = tuple(workers)

        self._frame_count = 0
        self._records: list[dict] = []
        self._destroyed = False

        for worker in self.workers:
            worker.start()
        print(f"[ChainOrchV5] N={slab_count}, {len(self.workers)} workers "
              f"started ({', '.join(w.label for w in self.workers)})")

    def destroy(self) -> None:
        if self._destroyed:
            return
        for worker in self.workers:
            worker.stop()
        self._destroyed = True

    def __enter__(self) -> "ChainOrchestratorV5":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Bootstrap
    # ========================================================================

    def bootstrap_all(self) -> None:
        """Chain bootstrap with one synchronous ghost round per link.

        Same 4 stages as the dual orchestrator (see its docstring for why
        the ghost round is required), with stage 2 generalized to a bridge
        loop over the N-1 links (replicas only; no migrations at t=0)."""
        for sim in self.sims:
            sim.bootstrap_init()

        total_bridged_bytes = 0
        for index in range(len(self.sims) - 1):
            left, right = self.sims[index], self.sims[index + 1]
            right.receiver_staging_view("leading")[:] = (
                left.sender_staging_view("trailing"))
            left.receiver_staging_view("trailing")[:] = (
                right.sender_staging_view("leading"))
            total_bridged_bytes += (
                left.sender_staging_view("trailing").nbytes
                + right.sender_staging_view("leading").nbytes)
        if len(self.sims) > 1:
            print(f"[ChainOrchV5] bootstrap ghost sync: bridged "
                  f"{total_bridged_bytes / 1024:.1f} KB over "
                  f"{len(self.sims) - 1} links")

        for sim in self.sims:
            sim.bootstrap_compute()
        for sim in self.sims:
            sim.prepare_step_cmd_buffers()
        print(f"[ChainOrchV5] all {len(self.sims)} sims bootstrapped "
              f"+ step cmds ready")

    # ========================================================================
    # Frame loop
    # ========================================================================

    def _submit_frame(self, frame_n: int) -> None:
        """Submit all of frame n's work (no wait). Submits go FIRST and
        worker notifies LAST — with queue_depth>1 on the notify queues, a
        slow link's backpressure lands after the frame's GPU work is
        already queued instead of stalling every sim's submission."""
        for sim in self.sims:
            sim.submit_phase_a(frame_n)
        for sim in self.sims:
            sim.submit_transfer_readback(frame_n)
        for sim in self.sims:
            sim.submit_phase_b(frame_n)
        for sim in self.sims:
            sim.submit_transfer_upload(frame_n)
        for sim in self.sims:
            sim.submit_phase_c(frame_n)
        for worker in self.workers:
            worker.notify(frame_n)

    def _wait_frame(self, frame_n: int, stall_timeout_s: float = 120.0) -> None:
        """Bounded wait on every sim's frame_done. Polls in 1 s slices so a
        worker death surfaces immediately; a stall past ``stall_timeout_s``
        dumps the autopsy and raises instead of hanging forever."""
        deadline = time.perf_counter() + stall_timeout_s
        for sim_index, sim in enumerate(self.sims):
            frame_semaphore, target_value = sim.sync.frame_done_op(frame_n)
            while True:
                if sim.wait_semaphore(frame_semaphore, target_value,
                                      timeout_ns=int(1e9)):
                    break
                self._raise_if_worker_died(frame_n)
                if time.perf_counter() > deadline:
                    autopsy = self.dump_stall_autopsy(frame_n)
                    print(autopsy, flush=True)
                    raise RuntimeError(
                        f"frame {frame_n}: sim{sim_index} frame_done stalled "
                        f"> {stall_timeout_s}s (GPU pipeline stall or lost "
                        f"signal). Autopsy above.")
        self._raise_if_worker_died(frame_n)

    def _raise_if_worker_died(self, frame_n: int) -> None:
        for worker in self.workers:
            if worker.last_error is not None:
                raise RuntimeError(
                    f"worker {worker.label} died (frame {frame_n}): "
                    f"{worker.last_error}") from worker.last_error

    def _wait_frame_with_watchdog(self, frame_n: int) -> None:
        for sim_index, sim in enumerate(self.sims):
            frame_semaphore, target_value = sim.sync.frame_done_op(frame_n)
            for poll_iter in range(self.MAX_POLL_ITERS):
                done = sim.wait_semaphore(
                    frame_semaphore, target_value,
                    timeout_ns=int(self.WATCHDOG_TIMEOUT_S * 1e9))
                if done:
                    break
                self._raise_if_worker_died(frame_n)
                states = " | ".join(
                    f"s{i}={s.sync_state()}" for i, s in enumerate(self.sims))
                worker_states = " | ".join(
                    f"{w.label}:{w.last_activity[0]}@{w.last_activity[1]}"
                    f"(done={w.last_completed_frame})" for w in self.workers)
                print(f"[ChainOrchV5 WATCHDOG] frame {frame_n}: waiting "
                      f"sim{sim_index} -> {target_value}; {states}; "
                      f"{worker_states}; poll {poll_iter + 1}/"
                      f"{self.MAX_POLL_ITERS}", flush=True)
            else:
                raise RuntimeError(
                    f"frame {frame_n}: sim{sim_index} timeline stuck at "
                    f"{sim.sync_state()} (target {target_value}) after "
                    f"{self.MAX_POLL_ITERS * self.WATCHDOG_TIMEOUT_S}s. "
                    f"Likely GPU device-lost / kernel hang.")

    def step(self) -> dict:
        """Run one frame synchronously, return the per-frame record."""
        frame_n = self._frame_count
        t_start = time.perf_counter_ns()
        self._submit_frame(frame_n)
        self._wait_frame_with_watchdog(frame_n)
        t_end = time.perf_counter_ns()
        self._raise_if_worker_died(frame_n)

        record = {
            "frame_n": frame_n,
            "frame_start_ns": t_start,
            "frame_end_ns": t_end,
            "frame_time_us": (t_end - t_start) / 1000.0,
            "workers": {w.label: w.timestamps_for_frame(frame_n)
                        for w in self.workers},
        }
        self._records.append(record)
        self._frame_count += 1

        if self._frame_count > 0 and self._frame_count % self.defrag_cadence == 0:
            record["defrag_frame"] = self._frame_count
            record["defrag_report"] = self._collect_defrag_report()
            for sim in self.sims:
                sim.submit_defrag_and_wait()
        return record

    def _collect_defrag_report(self) -> list[dict]:
        report = []
        for sim in self.sims:
            status = sim.readback_global_status()
            health = sim.readback_pool_health()
            report.append({
                "interval_migration":     status["migration_install_count"],
                "alive":                  status["alive_particle_count"],
                "overflow_install_tail":  status["overflow_install_tail"],
                # Ghost-pool per-voxel-slot drops — the counters the K=8
                # 60h soak's drift=-85 was invisible in (only install_tail
                # was logged). Cumulative since last defrag reset.
                "overflow_inside":        status["overflow_inside_count"],
                "overflow_incoming":      status["overflow_incoming_count"],
                "overflow_ghost":         status["overflow_ghost_count"],
                "overflow_install_inside": status["overflow_install_inside"],
                "peak_migration":         health["peak_migration_count"],
                "peak_tail":              health["peak_tail_high_water"],
                "own_pool":               health["own_pool_size"],
                "used_fraction":          health["used_fraction"],
            })
        return report

    def run_until(self, max_steps: Optional[int] = None) -> None:
        if max_steps is None:
            raise ValueError("requires max_steps (no time budget)")
        while self._frame_count < max_steps:
            self.step()

    def dump_stall_autopsy(self, frame_n: int) -> str:
        """Host-side-only diagnostic snapshot (safe on a stalled device:
        vkGetSemaphoreCounterValue never blocks). Returns a multi-line
        string with every sim's full semaphore state and every worker's
        last activity — the data the 2026-07-18 K=8 hang lacked."""
        lines = [f"=== STALL AUTOPSY (waiting frame {frame_n}) ==="]
        for index, sim in enumerate(self.sims):
            frame_semaphore, target = sim.sync.frame_done_op(frame_n)
            lines.append(f"sim{index}: sync_state={sim.sync_state()} "
                         f"frame_done_target={target}")
        for worker in self.workers:
            activity, activity_frame, stamp_ns = worker.last_activity
            lines.append(
                f"worker {worker.label}: {activity}@frame{activity_frame} "
                f"iters={worker.iteration_count} "
                f"last_done={worker.last_completed_frame} "
                f"age={(time.perf_counter_ns() - stamp_ns) / 1e9:.1f}s")
        return "\n".join(lines)

    def run_pipelined(
        self,
        max_steps: int,
        *,
        depth: int = 2,
        warmup: int = 0,
        on_defrag=None,
        stall_timeout_s: float = 120.0,
    ) -> dict:
        """Submit-ahead pipelined run loop — same protocol-safety argument
        as DualGpuOrchestratorV5.run_pipelined (see its docstring); the
        chain adds nothing new because all cross-sim dependencies remain
        nearest-neighbor worker-mediated edges.

        Unlike the dual version, _wait_frame here is BOUNDED: a frame that
        fails to complete within ``stall_timeout_s`` triggers a full
        semaphore/worker autopsy dump and raises. The 2026-07-18 K=8 soak
        hang sat invisible for 1.7 days behind an INFINITE vkWaitSemaphores
        — never again."""
        depth = max(1, depth)
        t_start = time.perf_counter()
        warmup_t = None
        warmup_frame = None
        n = 0
        next_wait = 0
        while n < max_steps:
            self._submit_frame(n)
            n += 1
            self._frame_count = n
            while n - next_wait >= depth:
                self._wait_frame(next_wait, stall_timeout_s)
                next_wait += 1
            if warmup_t is None and n >= warmup:
                warmup_t = time.perf_counter()
                warmup_frame = n
            if n % self.defrag_cadence == 0:
                while next_wait < n:
                    self._wait_frame(next_wait, stall_timeout_s)
                    next_wait += 1
                report = self._collect_defrag_report()
                if on_defrag is not None:
                    on_defrag(n, report)
                for sim in self.sims:
                    sim.submit_defrag_and_wait()
        while next_wait < max_steps:
            self._wait_frame(next_wait, stall_timeout_s)
            next_wait += 1
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        out = {
            "frame_count": max_steps,
            "elapsed_s": elapsed,
            "fps": max_steps / elapsed if elapsed > 0 else 0.0,
        }
        if warmup_t is not None:
            steady_frames = max_steps - warmup_frame
            steady_s = t_end - warmup_t
            out.update({
                "steady_frames": steady_frames,
                "steady_s": steady_s,
                "steady_fps": steady_frames / steady_s if steady_s > 0 else 0.0,
            })
        return out

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def instrumentation_records(self) -> list[dict]:
        return list(self._records)
