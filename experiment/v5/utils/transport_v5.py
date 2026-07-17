"""
transport_v5.py — V5 cross-GPU ghost transport: per-pathway worker thread.

V5 v1.0 backend = CPU-staged 3-hop (see docs/sph_v5_design.md §6 + §14.5):

    sender VRAM → sender host_staging → memcpy → receiver host_staging → receiver VRAM

The two vkCmdCopyBuffer hops are folded into SphSimulatorV5's phase_a (readback)
and phase_c (upload) cmd buffers (E-5 = option a). What lives here is the
*middle hop*: a persistent worker thread that bridges two devices' host stagings
via numpy uint8 slice copy.

No CPU-side remap: sender's ghost_send.comp pre-encodes packets in receiver's
voxel_id / pid coordinates via spec consts GHOST_VOXEL_ID_OFFSET_TO_RECEIVER
and GHOST_PID_OFFSET_TO_RECEIVER. Worker only does byte memcpy.

V5 v1.0 spawns 2 GhostMigrationWorker instances (one per pathway: A→B and
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
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5


_STOP_SENTINEL = -1   # frame_n value that means "stop the worker thread"


class GhostMigrationWorker:
    """One pathway (source → dest) persistent worker thread.

    Per-frame main loop (semaphores/values come from each sim's sync scheme —
    see sync_scheme_v5.py; aggregated = shared 5N timeline, per-direction =
    the direction's own transport timeline):
        1. wait source readback_done(n)  [sender_staging fully populated]
        2. wait dest readback_done(n)    [backwards-signal guard, see _run]
        3. memcpy source.sender_staging_view(source_dir) →
                  dest.receiver_staging_view(dest_dir)
        4. host_signal dest worker_done(n)
        5. record per-frame timestamps for instrumentation

    The (source_dir, dest_dir) pair is asymmetric: GPU 0's trailing send goes
    to GPU 1's leading receive. Caller (orchestrator) sets these at construction.

    Owns: 1 daemon=False threading.Thread, 1 queue.Queue(maxsize=1) for frame_n
    notify, per-frame timestamp dict, last_error state for non-silent failures.
    """

    def __init__(
        self,
        source_sim: "SphSimulatorV5",
        dest_sim: "SphSimulatorV5",
        source_direction: str,         # "leading" or "trailing"
        dest_direction: str,
        label: str,
        queue_depth: int = 1,          # notify backpressure depth; the chain
                                       # orchestrator passes >1 so one slow
                                       # link cannot stall the submit loop
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
        # Bounded = backpressure (main thread blocks if worker falls
        # queue_depth frames behind; the historical default is 1).
        self.work_queue: queue.Queue = queue.Queue(maxsize=max(1, queue_depth))

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
        orchestrator can react. Deferred per docs/sph_v5_design.md §14.4
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
        """Push frame_n; blocks if the worker is queue_depth frames behind.
        Fail-fast if the worker died — including WHILE blocked in put(), so
        a dead worker can never hang the orchestrator inside notify()."""
        while True:
            if self.last_error is not None:
                raise RuntimeError(
                    f"worker {self.label} died: "
                    f"{self.last_error}") from self.last_error
            try:
                self.work_queue.put(frame_n, timeout=1.0)
                return
            except queue.Full:
                continue

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
                #     readback_done(n) — sender_staging is now fully
                #     populated and CPU-visible (host coherence barrier ran).
                self.last_activity = ("wait_source_timeline", frame_n, time.perf_counter_ns())
                source_semaphore, source_value = self.source.sync.source_readback_op(
                    self.source_direction, frame_n)
                self.source.wait_semaphore(source_semaphore, source_value)
                # 1b. Wait for DEST sim's readback_done(n) on the SAME
                #     semaphore we are about to host-signal. Critical for
                #     timeline monotonicity: our host_signal of worker_done
                #     must come AFTER the pending GPU signal below it
                #     (dest's own readback), otherwise the GPU signal would
                #     land "backwards" relative to ours. This is the
                #     sync-scheme safety invariant: before host-signaling
                #     value v on semaphore S, wait S >= v-1.
                self.last_activity = ("wait_dest_timeline", frame_n, time.perf_counter_ns())
                guard_semaphore, guard_value = self.dest.sync.dest_guard_op(
                    self.dest_direction, frame_n)
                self.dest.wait_semaphore(guard_semaphore, guard_value)
                t_wait = time.perf_counter_ns()

                # 2. Byte memcpy (CPU → CPU)
                self.last_activity = ("memcpy", frame_n, time.perf_counter_ns())
                self._dest_view[:] = self._source_view
                t_copy = time.perf_counter_ns()

                # 3. Host-signal dest's worker_done(n). Dest's transfer
                #    queue's upload cmd for our direction waits on this and
                #    then signals upload_done, which Phase C's submit waits on.
                self.last_activity = ("signal_dest_timeline", frame_n, time.perf_counter_ns())
                signal_semaphore, signal_value = self.dest.sync.worker_signal_op(
                    self.dest_direction, frame_n)
                # Safety net: if a future refactor removes the dest guard wait
                # above, this assert will trip instead of silently deadlocking
                # via AMD driver's backwards-signal corruption. (worker_signal
                # and dest_guard target the same semaphore in both schemes.)
                current_dest = self.dest.semaphore_value(signal_semaphore)
                assert current_dest >= guard_value, (
                    f"worker {self.label} about to host_signal({signal_value}) on "
                    f"dest, but dest semaphore={current_dest} < readback_done"
                    f"={guard_value}. Without waiting dest's transfer-queue "
                    f"readback signal first, the host signal would race ahead "
                    f"and corrupt the timeline (Vulkan backwards-signal hazard).")
                self.dest.host_signal_semaphore(signal_semaphore, signal_value)
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
