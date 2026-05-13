"""
transport_v2.py — V2 cross-GPU ghost transport: per-pathway worker thread.

V2 v1.0 backend = CPU-staged 3-hop (see docs/sph_v2_design.md §6 + §14.5):

    sender VRAM → sender host_staging → memcpy → receiver host_staging → receiver VRAM

The two vkCmdCopyBuffer hops are folded into SphSimulatorV2's phase_a (readback)
and phase_c (upload) cmd buffers (E-5 = option a). What lives here is the
*middle hop*: a persistent worker thread that bridges two devices' host stagings
via numpy uint8 slice copy.

No CPU-side remap: sender's ghost_send.comp pre-encodes packets in receiver's
voxel_id / pid coordinates via spec consts GHOST_VOXEL_ID_OFFSET_TO_RECEIVER
and GHOST_PID_OFFSET_TO_RECEIVER. Worker only does byte memcpy.

V2 v1.0 spawns 2 GhostMigrationWorker instances (one per pathway: A→B and
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
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2


_STOP_SENTINEL = -1   # frame_n value that means "stop the worker thread"


class GhostMigrationWorker:
    """One pathway (source → dest) persistent worker thread.

    Per-frame main loop:
        1. wait source.timeline >= source.value_phase_a_done(n)
        2. memcpy source.sender_staging_view(source_dir) →
                  dest.receiver_staging_view(dest_dir)
        3. host_signal dest.timeline = dest.value_cpu_sync_done(n)
        4. record per-frame timestamps for instrumentation

    The (source_dir, dest_dir) pair is asymmetric: GPU 0's trailing send goes
    to GPU 1's leading receive. Caller (orchestrator) sets these at construction.

    Owns: 1 daemon=False threading.Thread, 1 queue.Queue(maxsize=1) for frame_n
    notify, per-frame timestamp dict, last_error state for non-silent failures.
    """

    def __init__(
        self,
        source_sim: "SphSimulatorV2",
        dest_sim: "SphSimulatorV2",
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
        try:
            while True:
                frame_n = self.work_queue.get()
                if frame_n == _STOP_SENTINEL:
                    return

                # 1. Wait for source GPU to signal phase_a_done(n)
                wait_value = self.source.value_phase_a_done(frame_n)
                self.source.wait_timeline(wait_value)
                t_wait = time.perf_counter_ns()

                # 2. Byte memcpy (numpy slice assignment; releases GIL during
                #    the underlying memmove for HOST_VISIBLE memory).
                self._dest_view[:] = self._source_view
                t_copy = time.perf_counter_ns()

                # 3. Signal dest GPU's cpu_sync_done(n); phase C wait clears.
                signal_value = self.dest.value_cpu_sync_done(frame_n)
                self.dest.host_signal_timeline(signal_value)
                t_signal = time.perf_counter_ns()

                self.timestamps[frame_n] = {
                    "wait_ns": t_wait,
                    "copy_ns": t_copy,
                    "signal_ns": t_signal,
                }
        except BaseException as e:  # noqa: BLE001 — capture everything for diagnostics
            self.last_error = e
