"""
orchestrator_v3.py — V3.3 cross-GPU orchestrator (shared host RAM,
no CPU worker thread).

Owns 2 SphSimulatorV3 (borrows, does not destroy) + 1 SharedHostTransport
(owns). Drives frame loop, collects instrumentation timestamps, dispatches
defrag at cadence boundaries. LoadBalancer hook is reserved for V4
(docs/sph_v3_design.md §11.3) — V3.3 uses static partition only.

Frame loop (synchronous depth=1):

    n = self.frame_count
    for sim in self.sims:  sim.submit_phase_a(n)
    for sim in self.sims:  sim.submit_phase_b(n)
    for sim in self.sims:  sim.submit_phase_c(n)
    for sim in self.sims:  sim.wait_frame_done(n)
    if defrag_due(n):
        for sim in self.sims: sim.submit_defrag_and_wait()
    self.frame_count += 1

Cross-GPU sync between phase A and the peer's phase C is a cross-device
binary semaphore (VK_KHR_external_semaphore_win32 OPAQUE_WIN32), signaled
on submission of phase A and waited on submission of phase C. No CPU
worker thread, no host memcpy, no host_signal_timeline — these all
collapsed when SharedHostTransport let both VkDevices import the same
host pointer (see memory/project_cross_vendor_shared_host_breakthrough.md).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from experiment.v3.utils.shared_host_transport_v3 import SharedHostTransport

if TYPE_CHECKING:
    from experiment.v3.utils.simulator_v3 import SphSimulatorV3


class DualGpuOrchestratorV3:
    """V3.3 dual-GPU orchestrator. Sim ownership: borrows only. Owns the
    SharedHostTransport (created on construction unless one is injected)."""

    def __init__(
        self,
        sim_a: "SphSimulatorV3",
        sim_b: "SphSimulatorV3",
        *,
        defrag_cadence: int = 1000,
        shared_host_transport: Optional[SharedHostTransport] = None,
    ) -> None:
        self.sim_a = sim_a
        self.sim_b = sim_b
        self.sims = (sim_a, sim_b)
        self.defrag_cadence = defrag_cadence

        # Pathway naming convention (matches SharedHostTransport):
        #   sim_a = leftmost  → has trailing peer (sim_b)
        #   sim_b = rightmost → has leading peer  (sim_a)
        # a_to_b region carries sim_a's trailing boundary to sim_b's leading
        # b_to_a region carries sim_b's leading boundary to sim_a's trailing
        if not sim_a.case.transport.has_trailing_peer:
            raise ValueError("sim_a must have trailing peer (leftmost slot)")
        if not sim_b.case.transport.has_leading_peer:
            raise ValueError("sim_b must have leading peer (rightmost slot)")

        if shared_host_transport is None:
            a_to_b_bytes = sim_a.transport_bytes_for_direction("trailing")
            b_to_a_bytes = sim_b.transport_bytes_for_direction("leading")
            if a_to_b_bytes == 0 or b_to_a_bytes == 0:
                raise RuntimeError(
                    f"transport sizes computed as zero (a_to_b={a_to_b_bytes}, "
                    f"b_to_a={b_to_a_bytes}); sims may not have completed "
                    f"_allocate_staging_buffers yet")
            shared_host_transport = SharedHostTransport.create(
                ctx_a=sim_a.ctx,
                ctx_b=sim_b.ctx,
                a_to_b_bytes=a_to_b_bytes,
                b_to_a_bytes=b_to_a_bytes,
            )
            self._owns_transport = True
        else:
            self._owns_transport = False
        self.shared_host_transport = shared_host_transport

        sim_a.attach_shared_transport(shared_host_transport, side="a")
        sim_b.attach_shared_transport(shared_host_transport, side="b")

        self._frame_count = 0
        self._records: list[dict] = []
        self._destroyed = False

        print(f"[OrchV3] V3.3 SharedHostTransport attached to both sims; "
              f"no worker threads (compute-queue cross-device semaphores).")

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self._owns_transport and self.shared_host_transport is not None:
            self.shared_host_transport.destroy()
            self.shared_host_transport = None
        self._destroyed = True

    def __enter__(self) -> "DualGpuOrchestratorV3":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Frame loop
    # ========================================================================

    def bootstrap_all(self) -> None:
        """Dual-GPU bootstrap with one synchronous ghost round.

        Without the ghost round, step 1's correction/density/force on
        boundary particles would see EMPTY ghost neighbors → density
        underestimate → negative pressure → wild attractive forces →
        particles drift across the slab and overflow corner voxels by
        step 10-20.

        V3.3 sequence:
          1. Per sim: bootstrap_init (init_voxelization + ghost_send +
             readback into shared host RAM)
          2. vkDeviceWaitIdle on both compute queues — synchronous
             barrier so both sims' phase-A writes to shared host RAM
             are visible to the peer before phase-C reads them.
             (Replaces V3.0/Path A+ inline memcpy bridge: with shared
             host RAM, sim_a's sender == sim_b's receiver physically,
             so no memcpy is needed — only a flush.)
          3. Per sim: bootstrap_compute (upload + install_migrations +
             correction + density + force + bootstrap_half_kick)
          4. Per sim: prepare_step_cmd_buffers (record phase A/B/C for
             run loop)
        """
        from vulkan import vkDeviceWaitIdle

        # Stage 1: init + ghost_send + readback into shared host RAM
        for sim in self.sims:
            sim.bootstrap_init()

        # Stage 2: ensure both readback DMAs have completed on host RAM
        # before either sim's bootstrap_compute issues an upload from
        # the same memory.
        for sim in self.sims:
            vkDeviceWaitIdle(sim.ctx.device)
        print(f"[OrchV3] bootstrap ghost sync: both compute queues idle "
              f"(shared host RAM has both sims' ghost data)")

        # Stage 3: install + correction + density + force + half_kick
        for sim in self.sims:
            sim.bootstrap_compute()

        # Stage 4: record per-frame cmd buffers (phase A/B/C)
        for sim in self.sims:
            sim.prepare_step_cmd_buffers()
        print(f"[OrchV3] both sims bootstrapped + step cmds ready")

    def step(self) -> dict:
        """Run one frame, return per-frame instrumentation record."""
        n = self._frame_count
        t_start = time.perf_counter_ns()

        # 1. Compute Q: submit phase A on both sims (predict + update_voxel +
        #    ghost_send + inline readback DMA to shared host RAM). Phase A's
        #    submission signals each outgoing cross-device binary semaphore.
        for sim in self.sims:
            sim.submit_phase_a(n)

        # 2. Compute Q: submit phase B (correction_interior +
        #    density_deep_interior). Queue-ordered after phase A; no
        #    cross-queue sync.
        for sim in self.sims:
            sim.submit_phase_b(n)

        # 3. Compute Q: submit phase C (inline upload from shared host RAM +
        #    install_migrations + correction_boundary + density + force).
        #    Each sim's phase C waits on the peer's cross-device binary
        #    semaphore signaled by step 1's phase A submission.
        for sim in self.sims:
            sim.submit_phase_c(n)

        # 4. Wait both sims to reach frame_done on their timeline. Watchdog
        #    poll surfaces stuck GPUs (TDR / device-lost) instead of
        #    hanging forever. No worker threads to check — they don't exist.
        WATCHDOG_TIMEOUT_S = 1.0
        MAX_POLL_ITERS = 10   # → 10s total before giving up
        for sim_idx, sim in enumerate(self.sims):
            target_value = sim.value_frame_done(n)
            for poll_iter in range(MAX_POLL_ITERS):
                done = sim.wait_timeline(
                    target_value, timeout_ns=int(WATCHDOG_TIMEOUT_S * 1e9))
                if done:
                    break
                cur_a = self.sim_a.current_timeline_value()
                cur_b = self.sim_b.current_timeline_value()
                print(f"[OrchV3 WATCHDOG] frame {n}: waiting "
                      f"sim_{('a','b')[sim_idx]} to reach {target_value}; "
                      f"cur a={cur_a} b={cur_b}; "
                      f"poll {poll_iter+1}/{MAX_POLL_ITERS}", flush=True)
            else:
                raise RuntimeError(
                    f"frame {n}: sim_{('a','b')[sim_idx]} timeline stuck at "
                    f"{self.sims[sim_idx].current_timeline_value()} "
                    f"(target {target_value}) after "
                    f"{MAX_POLL_ITERS * WATCHDOG_TIMEOUT_S}s. "
                    f"Likely GPU device-lost / kernel hang.")
        t_end = time.perf_counter_ns()

        record = {
            "frame_n": n,
            "frame_start_ns": t_start,
            "frame_end_ns": t_end,
            "frame_time_us": (t_end - t_start) / 1000.0,
        }
        self._records.append(record)
        self._frame_count += 1

        # 5. Defrag if due (independent fence-wait pass; not in timeline).
        if (self._frame_count > 0
                and self._frame_count % self.defrag_cadence == 0):
            # Snapshot BEFORE defrag resets migration_install_count: this gives
            # the per-interval migration total (how many particles installed at
            # the own-pool tail since the last defrag) plus the never-reset
            # pool-health peaks. Cheap: 2 tiny readbacks once per defrag_cadence
            # frames. Callers (bench) print this as the migration time series.
            defrag_report = []
            for sim in self.sims:
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                defrag_report.append({
                    "interval_migration":     status["migration_install_count"],
                    "alive":                  status["alive_particle_count"],
                    "overflow_install_tail":  status["overflow_install_tail"],
                    "peak_migration":         health["peak_migration_count"],
                    "peak_tail":              health["peak_tail_high_water"],
                    "own_pool":               health["own_pool_size"],
                    "used_fraction":          health["used_fraction"],
                })
            record["defrag_frame"] = self._frame_count
            record["defrag_report"] = defrag_report
            for sim in self.sims:
                sim.submit_defrag_and_wait()

        return record

    def run_until(
        self,
        max_steps: Optional[int] = None,
    ) -> None:
        if max_steps is None:
            raise ValueError("V3.3 requires max_steps (no time budget yet)")
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
