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

Future backends (P2P, shared host via VK_EXT_external_memory_host) plug in
here as alternate worker / pathway implementations; the simulator's phase_a /
phase_c cmd buffers may shrink (no readback / upload) but the orchestrator-
facing interface (notify / start / stop + signals 3N+2) stays the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2


class GhostMigrationWorker:
    """One pathway (source → dest) persistent worker thread.

    Per-frame main loop:
        1. wait source.timeline >= source.value_phase_a_done(n)
        2. memcpy source.sender_staging_view() → dest.receiver_staging_view()
        3. signal dest.timeline = dest.value_cpu_sync_done(n)
        4. record per-frame timestamps for instrumentation

    Owns: 1 daemon=False threading.Thread, 1 queue.Queue(maxsize=1) for frame_n
    notify, per-frame timestamp dict, last_error state for non-silent failures.

    Borrows (does not own): source_sim, dest_sim. Caller must keep them alive
    for the worker's lifetime and stop() the worker before destroying them.
    """

    def __init__(
        self,
        source_sim: "SphSimulatorV2",
        dest_sim: "SphSimulatorV2",
        label: str,                      # "a_to_b" / "b_to_a"; debug log + thread name
    ) -> None:
        # Phase 4 fills in: take numpy views, create work_queue, create thread
        # (not started yet — call .start() after both sims constructed).
        raise NotImplementedError("Phase 4")

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def start(self) -> None:
        """Launch the worker thread. Must be called after both sims have
        finished construction (timeline semaphore + staging buffer ready)."""
        raise NotImplementedError("Phase 4")

    def stop(self) -> None:
        """Push sentinel + join. Idempotent. Must be called before either
        sim's destroy() to avoid the worker dereferencing freed handles."""
        raise NotImplementedError("Phase 4")

    # ========================================================================
    # Per-frame interface (orchestrator main thread)
    # ========================================================================

    def notify(self, frame_n: int) -> None:
        """Main thread tells worker frame N is ready to process. Non-blocking
        unless previous frame's notify hasn't been consumed yet (queue maxsize=1
        provides backpressure → main thread waits, prevents N drift)."""
        raise NotImplementedError("Phase 4")

    @property
    def last_error(self) -> Optional[Exception]:
        """If worker raised, this holds the exception. Orchestrator checks
        every frame; non-None means worker is dead and pipeline cannot continue."""
        raise NotImplementedError("Phase 4")

    def timestamps_for_frame(self, frame_n: int) -> dict:
        """Return {wait_done_ns, copy_done_ns, signal_done_ns} for frame N.
        Used by orchestrator's instrumentation collection."""
        raise NotImplementedError("Phase 4")

    # ========================================================================
    # Thread body (internal)
    # ========================================================================

    def _run(self) -> None:
        """Worker thread main loop. Internal; do not call from main thread."""
        raise NotImplementedError("Phase 4")
