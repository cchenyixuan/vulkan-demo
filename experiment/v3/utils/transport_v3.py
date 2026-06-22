"""
transport_v3.py — V3 cross-GPU ghost transport: per-pathway worker thread.

V3 v1.0 backend = CPU-staged 3-hop (see docs/sph_v3_design.md §6 + §14.5):

    sender VRAM → sender host_staging → memcpy → receiver host_staging → receiver VRAM

The two vkCmdCopyBuffer hops are folded into SphSimulatorV3's phase_a (readback)
and phase_c (upload) cmd buffers (E-5 = option a). What lives here is the
*middle hop*: a persistent worker thread that bridges two devices' host stagings
via numpy uint8 slice copy.

No CPU-side remap: sender's ghost_send.comp pre-encodes packets in receiver's
voxel_id / pid coordinates via spec consts GHOST_VOXEL_ID_OFFSET_TO_RECEIVER
and GHOST_PID_OFFSET_TO_RECEIVER. Worker only does byte memcpy.

V3 v1.0 spawns 2 GhostMigrationWorker instances (one per pathway: A→B and
B→A). They share no mutable state and never contend on a lock — pathway A→B
touches sim_a.sender_staging + sim_b.receiver_staging + sim_b.timeline; B→A
is disjoint.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experiment.v3.utils.simulator_v3 import SphSimulatorV3


_STOP_SENTINEL = -1   # frame_n value that means "stop the worker thread"


class GhostMigrationWorker:
    """One pathway (source → dest) persistent worker thread.

    Per-frame main loop:
        1. wait source.timeline >= source.value_readback_done(n)
        2. memcpy source.sender_staging_view(source_dir) →
                  dest.receiver_staging_view(dest_dir)
        3. host_signal dest.timeline = dest.value_worker_done(n)
        4. record per-frame timestamps for instrumentation

    The (source_dir, dest_dir) pair is asymmetric: GPU 0's trailing send goes
    to GPU 1's leading receive. Caller (orchestrator) sets these at construction.

    Owns: 1 daemon=False threading.Thread, 1 queue.Queue(maxsize=1) for frame_n
    notify, per-frame timestamp dict, last_error state for non-silent failures.
    """

    def __init__(
        self,
        source_sim: "SphSimulatorV3",
        dest_sim: "SphSimulatorV3",
        source_direction: str,         # "leading" or "trailing"
        dest_direction: str,
        label: str,
    ) -> None:
        self.source = source_sim
        self.dest = dest_sim
        self.source_direction = source_direction
        self.dest_direction = dest_direction
        self.label = label

        # Pre-fetch numpy views over the persistent-mapped stagings so the
        # hot loop doesn't reach into sim internals each frame.
        self._source_view = source_sim.sender_staging_view(source_direction)
        self._dest_view = dest_sim.receiver_staging_view(dest_direction)
        if self._source_view.nbytes != self._dest_view.nbytes:
            raise ValueError(
                f"worker {label}: source/dest staging sizes mismatch — "
                f"{self._source_view.nbytes} vs {self._dest_view.nbytes}. "
                f"Likely partition / GhostTransportConfig misconfigured.")

        # Notify channel: main thread puts frame_n; worker takes it.
        # maxsize=1 = backpressure (main thread blocks if worker is behind).
        self.work_queue: queue.Queue = queue.Queue(maxsize=1)

        # Instrumentation: per-frame timestamps populated inside _run().
        self.timestamps: dict[int, dict] = {}

        # Error state: worker thread stashes exception here; main thread
        # checks each frame to fail-fast instead of deadlocking.
        self.last_error: Optional[BaseException] = None

        # Initialized in _run(); seed here so orchestrator can peek before start.
        self.last_activity: tuple = ("not_started", -1, 0)
        self.iteration_count = 0
        self.last_completed_frame = -1

        self.thread = threading.Thread(
            target=self._run, name=f"ghost-{label}", daemon=False)
        self._started = False

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def start(self) -> None:
        if self._started:
            return
        self.thread.start()
        self._started = True

    def stop(self) -> None:
        """Best-effort stop. Safe on happy-path close (worker idle in
        work_queue.get); UNSAFE on exception paths.

        TODO(v1.x): not robust against worker stuck in vkWaitSemaphores.
        Current behavior on failure:
          - queue.Full silently swallowed → sentinel never delivered
          - thread.join(timeout=10) returns regardless → leaked daemon=False
            thread blocks process exit
          - subsequent stop() returns early (_started=False set anyway),
            masking the leak
        Fix sketch: (a) host_signal_timeline(POISON) on source+dest to wake
        blocked vkWaitSemaphores; (b) log timeout cases; (c) return bool so
        orchestrator can react. Deferred per docs/sph_v3_design.md §14.4
        ("watchdog = v1.x task, not v1.0").
        """
        if not self._started:
            return
        try:
            self.work_queue.put(_STOP_SENTINEL, timeout=5.0)
        except queue.Full:
            pass
        self.thread.join(timeout=10.0)
        self._started = False

    # ========================================================================
    # Per-frame interface (orchestrator main thread)
    # ========================================================================

    def notify(self, frame_n: int) -> None:
        """Push frame_n; blocks if worker hasn't consumed previous frame yet
        (queue maxsize=1). Fail-fast if worker died last frame."""
        if self.last_error is not None:
            raise RuntimeError(
                f"worker {self.label} died: {self.last_error}") from self.last_error
        self.work_queue.put(frame_n)

    def timestamps_for_frame(self, frame_n: int) -> dict:
        return self.timestamps.get(frame_n, {})

    # ========================================================================
    # Thread body (internal)
    # ========================================================================

    def _run(self) -> None:
        import sys as _sys
        # Last activity timestamp + phase, for orchestrator watchdog introspection.
        self.last_activity: tuple = ("init", 0, time.perf_counter_ns())
        self.iteration_count = 0
        self.last_completed_frame = -1
        try:
            while True:
                self.last_activity = ("wait_queue", -1, time.perf_counter_ns())
                frame_n = self.work_queue.get()
                if frame_n == _STOP_SENTINEL:
                    return
                self.iteration_count += 1

                # 1a. Wait for source GPU's transfer queue to signal
                #     readback_done(n) = 5N+2 — sender_staging is now fully
                #     populated and CPU-visible (host coherence barrier ran).
                #     Path A+: previously was phase_a_done (3N+1) which included
                #     the readback DMA inside phase A; now readback is on its
                #     own queue and signals 5N+2 independently.
                self.last_activity = ("wait_source_timeline", frame_n, time.perf_counter_ns())
                self.source.wait_timeline(self.source.value_readback_done(frame_n))
                # 1b. Wait for DEST sim's readback to also be done (5N+2).
                #     Critical for timeline monotonicity: our host_signal of
                #     dest.worker_done (5N+3) must come AFTER dest's transfer
                #     queue signals 5N+2, otherwise dest's signal would be
                #     "backwards" relative to host's. Same hazard as V3.0's
                #     wait-for-dest-phase-a-done; just shifted by 1 value.
                self.last_activity = ("wait_dest_timeline", frame_n, time.perf_counter_ns())
                dest_readback = self.dest.value_readback_done(frame_n)
                self.dest.wait_timeline(dest_readback)
                t_wait = time.perf_counter_ns()

                # 2. Byte memcpy (CPU → CPU)
                self.last_activity = ("memcpy", frame_n, time.perf_counter_ns())
                self._dest_view[:] = self._source_view
                t_copy = time.perf_counter_ns()

                # 3. Host-signal dest's worker_done(n) = 5N+3. Dest's transfer
                #    queue's upload cmd waits on this and then signals 5N+4
                #    (upload_done), which Phase C's submit waits on.
                self.last_activity = ("signal_dest_timeline", frame_n, time.perf_counter_ns())
                signal_value = self.dest.value_worker_done(frame_n)
                # Safety net: if a future refactor removes the dest.wait_timeline
                # above, this assert will trip instead of silently deadlocking
                # via AMD driver's backwards-signal corruption.
                current_dest = self.dest.current_timeline_value()
                assert current_dest >= dest_readback, (
                    f"worker {self.label} about to host_signal({signal_value}) on "
                    f"dest, but dest.timeline={current_dest} < readback_done"
                    f"={dest_readback}. Without waiting dest's transfer-queue "
                    f"readback signal first, the host signal would race ahead "
                    f"and corrupt the timeline (Vulkan backwards-signal hazard).")
                self.dest.host_signal_timeline(signal_value)
                t_signal = time.perf_counter_ns()
                self.last_activity = ("done_frame", frame_n, time.perf_counter_ns())
                self.last_completed_frame = frame_n

                self.timestamps[frame_n] = {
                    "wait_ns": t_wait,
                    "copy_ns": t_copy,
                    "signal_ns": t_signal,
                }
        except BaseException as e:  # noqa: BLE001 — capture everything for diagnostics
            self.last_error = e
            import traceback as _tb
            print(f"[worker {self.label}] DIED at {self.last_activity}: {e!r}",
                  file=_sys.stderr, flush=True)
            _tb.print_exc(file=_sys.stderr)
