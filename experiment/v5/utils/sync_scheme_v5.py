"""
sync_scheme_v5.py — frame synchronization schemes for the V5 multi-GPU step.

A FrameSyncScheme owns the timeline semaphore(s) of ONE sim and answers, for
every submit site in the frame protocol, "which (semaphore, value) pairs do I
wait on / signal here". Command buffers, kernels, and barriers are identical
across schemes — timeline wait/signal parameters exist only at vkQueueSubmit2 /
vkWaitSemaphores / vkSignalSemaphore time, so switching schemes changes zero
recorded GPU work. See docs/sph_v5_design.md §3.1.

Two implementations:

AggregatedTimelineScheme — the historical Path A+ layout. ONE timeline per sim
advancing 5 values per frame:

    5N+1  phase_a_done   compute Q signals at end of Phase A
    5N+2  readback_done  transfer Q signals after the LAST direction's
                         device→sender_staging DMA (directions aggregated)
    5N+3  worker_done    the inbound worker host-signals after its CPU memcpy
    5N+4  upload_done    transfer Q signals after the LAST direction's
                         receiver_staging→device DMA
    5N+5  frame_done     compute Q signals at end of Phase C

Only valid for sims with AT MOST ONE peer direction (chain endpoints — i.e.
the classic 2-GPU setup): an interior chain node has TWO inbound workers and
both would have to host-signal the same 5N+3 slot, which is a Vulkan
non-monotonic-signal violation AND releases both directions' uploads on the
first worker's signal (silent data race). Construction with two peers raises.

PerDirectionTimelineScheme — the N-GPU chain layout. One MAIN timeline
(3 values/frame) plus one TRANSPORT timeline per peer direction
(2 values/frame):

    main:            3N+1 phase_a_done   3N+2 upload_done   3N+3 frame_done
    transport[dir]:  2N+1 readback_done  2N+2 worker_done

Every direction's readback signals its OWN transport timeline (no
last-signals aggregation); each direction's upload waits its OWN direction's
worker_done; the LAST upload signals main.upload_done (Phase C needs all
directions installed, so a single aggregated upload_done is semantically
right). Interior nodes are safe: the two inbound workers host-signal two
DIFFERENT semaphores.

Host-signal safety invariant (both schemes, enforced by the worker's guard
wait + assert): before host-signaling value v on semaphore S, wait S >= v-1.
For the per-direction scheme the pending GPU signal below v is that
direction's own readback (2N+1); the next frame's readback (2(N+1)+1) is
fenced off by the frame_done -> phase_a chain, so no forward hazard exists.
"""

from __future__ import annotations

from typing import Sequence

from vulkan import *  # noqa: F401, F403


SemaphoreOp = tuple  # (VkSemaphore, int value)

_VALID_DIRECTIONS = ("leading", "trailing")


def _create_timeline_semaphore(device):
    type_info = VkSemaphoreTypeCreateInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,
        initialValue=0,
    )
    ci = VkSemaphoreCreateInfo(pNext=type_info)
    return vkCreateSemaphore(device, ci, None)


class FrameSyncScheme:
    """Interface. All wait/signal getters return lists of (semaphore, value);
    the worker-facing *_op getters return a single (semaphore, value) pair."""

    name: str = "abstract"

    def __init__(self, peer_directions: Sequence[str]) -> None:
        for direction in peer_directions:
            if direction not in _VALID_DIRECTIONS:
                raise ValueError(f"unknown direction {direction!r}")
        self.peer_directions: tuple[str, ...] = tuple(peer_directions)

    # -- lifecycle -----------------------------------------------------------
    def create(self, device) -> None:
        raise NotImplementedError

    def destroy(self, device) -> None:
        raise NotImplementedError

    def primary_semaphore(self):
        """The semaphore carrying frame_done — used for coarse progress
        introspection (watchdog prints, legacy accessors)."""
        raise NotImplementedError

    # -- compute queue submit sites -------------------------------------------
    def phase_a_waits(self, frame_n: int) -> list:
        raise NotImplementedError

    def phase_a_signals(self, frame_n: int) -> list:
        raise NotImplementedError

    def phase_c_waits(self, frame_n: int) -> list:
        raise NotImplementedError

    def phase_c_signals(self, frame_n: int) -> list:
        raise NotImplementedError

    # -- transfer queue submit sites -------------------------------------------
    def readback_waits(self, direction: str, frame_n: int) -> list:
        raise NotImplementedError

    def readback_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        raise NotImplementedError

    def upload_waits(self, direction: str, frame_n: int) -> list:
        raise NotImplementedError

    def upload_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        raise NotImplementedError

    # -- host-side ops (orchestrator + ghost worker) ---------------------------
    def frame_done_op(self, frame_n: int) -> SemaphoreOp:
        raise NotImplementedError

    def source_readback_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        """Worker waits this on the SOURCE sim: sender_staging fully populated."""
        raise NotImplementedError

    def dest_guard_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        """Worker waits this on the DEST sim before host-signaling, so the
        host signal never lands behind a pending lower GPU signal on the same
        semaphore (Vulkan backwards-signal hazard)."""
        raise NotImplementedError

    def worker_signal_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        """Worker host-signals this on the DEST sim after its memcpy."""
        raise NotImplementedError

    # -- diagnostics -----------------------------------------------------------
    def state(self, device) -> dict:
        """{semaphore_name: current counter value} for watchdog prints."""
        raise NotImplementedError


class AggregatedTimelineScheme(FrameSyncScheme):
    """Historical Path A+ 5N single-timeline layout (module docstring)."""

    name = "aggregated"

    def __init__(self, peer_directions: Sequence[str]) -> None:
        super().__init__(peer_directions)
        if len(self.peer_directions) > 1:
            raise ValueError(
                "aggregated 5N scheme cannot serve an interior chain node "
                "(two inbound workers would collide on the shared worker_done "
                "slot 5N+3) — use sync_scheme='per-direction'")
        self.timeline = None

    # -- lifecycle
    def create(self, device) -> None:
        self.timeline = _create_timeline_semaphore(device)

    def destroy(self, device) -> None:
        if self.timeline is not None:
            vkDestroySemaphore(device, self.timeline, None)
            self.timeline = None

    def primary_semaphore(self):
        return self.timeline

    # -- values
    def value_phase_a_done(self, frame_n: int) -> int:
        return 5 * frame_n + 1

    def value_readback_done(self, frame_n: int) -> int:
        return 5 * frame_n + 2

    def value_worker_done(self, frame_n: int) -> int:
        return 5 * frame_n + 3

    def value_upload_done(self, frame_n: int) -> int:
        return 5 * frame_n + 4

    def value_frame_done(self, frame_n: int) -> int:
        return 5 * frame_n + 5

    # -- compute queue
    def phase_a_waits(self, frame_n: int) -> list:
        wait_value = self.value_frame_done(frame_n - 1) if frame_n > 0 else 0
        return [(self.timeline, wait_value)]

    def phase_a_signals(self, frame_n: int) -> list:
        return [(self.timeline, self.value_phase_a_done(frame_n))]

    def phase_c_waits(self, frame_n: int) -> list:
        # No peer -> no upload to wait for; wait phase_a_done instead (queue
        # order guarantees Phase B is done; we just need a monotonically
        # advancing valid wait value).
        if self.peer_directions:
            return [(self.timeline, self.value_upload_done(frame_n))]
        return [(self.timeline, self.value_phase_a_done(frame_n))]

    def phase_c_signals(self, frame_n: int) -> list:
        return [(self.timeline, self.value_frame_done(frame_n))]

    # -- transfer queue: directions share one value; only the LAST submitted
    # direction signals, so the value advances once all DMAs completed
    # (transfer queue is FIFO).
    def readback_waits(self, direction: str, frame_n: int) -> list:
        return [(self.timeline, self.value_phase_a_done(frame_n))]

    def readback_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        if not is_last:
            return []
        return [(self.timeline, self.value_readback_done(frame_n))]

    def upload_waits(self, direction: str, frame_n: int) -> list:
        return [(self.timeline, self.value_worker_done(frame_n))]

    def upload_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        if not is_last:
            return []
        return [(self.timeline, self.value_upload_done(frame_n))]

    # -- host-side
    def frame_done_op(self, frame_n: int) -> SemaphoreOp:
        return (self.timeline, self.value_frame_done(frame_n))

    def source_readback_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.timeline, self.value_readback_done(frame_n))

    def dest_guard_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.timeline, self.value_readback_done(frame_n))

    def worker_signal_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.timeline, self.value_worker_done(frame_n))

    # -- diagnostics
    def state(self, device) -> dict:
        return {"timeline": vkGetSemaphoreCounterValue(device, self.timeline)}


class PerDirectionTimelineScheme(FrameSyncScheme):
    """M1 N-GPU chain layout: main (3 values/frame) + one transport timeline
    per peer direction (2 values/frame). See module docstring."""

    name = "per-direction"

    def __init__(self, peer_directions: Sequence[str]) -> None:
        super().__init__(peer_directions)
        self.main = None
        self.transport: dict[str, object] = {}

    # -- lifecycle
    def create(self, device) -> None:
        self.main = _create_timeline_semaphore(device)
        for direction in self.peer_directions:
            self.transport[direction] = _create_timeline_semaphore(device)

    def destroy(self, device) -> None:
        if self.main is not None:
            vkDestroySemaphore(device, self.main, None)
            self.main = None
        for direction, semaphore in self.transport.items():
            vkDestroySemaphore(device, semaphore, None)
        self.transport = {}

    def primary_semaphore(self):
        return self.main

    # -- values
    def value_phase_a_done(self, frame_n: int) -> int:
        return 3 * frame_n + 1

    def value_upload_done(self, frame_n: int) -> int:
        return 3 * frame_n + 2

    def value_frame_done(self, frame_n: int) -> int:
        return 3 * frame_n + 3

    def value_readback_done(self, frame_n: int) -> int:
        return 2 * frame_n + 1

    def value_worker_done(self, frame_n: int) -> int:
        return 2 * frame_n + 2

    # -- compute queue
    def phase_a_waits(self, frame_n: int) -> list:
        wait_value = self.value_frame_done(frame_n - 1) if frame_n > 0 else 0
        return [(self.main, wait_value)]

    def phase_a_signals(self, frame_n: int) -> list:
        return [(self.main, self.value_phase_a_done(frame_n))]

    def phase_c_waits(self, frame_n: int) -> list:
        if self.peer_directions:
            return [(self.main, self.value_upload_done(frame_n))]
        return [(self.main, self.value_phase_a_done(frame_n))]

    def phase_c_signals(self, frame_n: int) -> list:
        return [(self.main, self.value_frame_done(frame_n))]

    # -- transfer queue: every direction signals its OWN transport timeline
    # (no aggregation — this is what frees the inbound workers from each
    # other); the LAST upload signals main.upload_done for Phase C.
    def readback_waits(self, direction: str, frame_n: int) -> list:
        return [(self.main, self.value_phase_a_done(frame_n))]

    def readback_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        return [(self.transport[direction], self.value_readback_done(frame_n))]

    def upload_waits(self, direction: str, frame_n: int) -> list:
        return [(self.transport[direction], self.value_worker_done(frame_n))]

    def upload_signals(self, direction: str, frame_n: int, is_last: bool) -> list:
        if not is_last:
            return []
        return [(self.main, self.value_upload_done(frame_n))]

    # -- host-side
    def frame_done_op(self, frame_n: int) -> SemaphoreOp:
        return (self.main, self.value_frame_done(frame_n))

    def source_readback_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.transport[direction], self.value_readback_done(frame_n))

    def dest_guard_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.transport[direction], self.value_readback_done(frame_n))

    def worker_signal_op(self, direction: str, frame_n: int) -> SemaphoreOp:
        return (self.transport[direction], self.value_worker_done(frame_n))

    # -- diagnostics
    def state(self, device) -> dict:
        result = {"main": vkGetSemaphoreCounterValue(device, self.main)}
        for direction, semaphore in self.transport.items():
            result[f"transport_{direction}"] = vkGetSemaphoreCounterValue(
                device, semaphore)
        return result


_SCHEME_CLASSES = {
    AggregatedTimelineScheme.name: AggregatedTimelineScheme,
    PerDirectionTimelineScheme.name: PerDirectionTimelineScheme,
}

SCHEME_NAMES = tuple(_SCHEME_CLASSES)


def make_sync_scheme(name: str, peer_directions: Sequence[str]) -> FrameSyncScheme:
    try:
        scheme_class = _SCHEME_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown sync scheme {name!r}; valid: {sorted(_SCHEME_CLASSES)}"
        ) from None
    return scheme_class(peer_directions)
