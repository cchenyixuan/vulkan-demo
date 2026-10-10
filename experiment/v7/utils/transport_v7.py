"""
transport_v7.py — V5 cross-GPU ghost transport: per-pathway worker thread.

V5 v1.0 backend = CPU-staged 3-hop (see docs/sph_v5_design.md §6 + §14.5):

    sender VRAM → sender host_staging → memcpy → receiver host_staging → receiver VRAM

The two vkCmdCopyBuffer hops are folded into SphSimulatorV7's phase_a (readback)
and phase_c (upload) cmd buffers (E-5 = option a). What lives here is the
*middle hop*: a persistent worker thread that bridges two devices' host stagings
via numpy uint8 slice copy.

No CPU-side remap: sender's ghost_send.comp pre-encodes packets in receiver's
voxel_id / pid coordinates via spec consts GHOST_VOXEL_ID_OFFSET_TO_RECEIVER
and GHOST_PID_OFFSET_TO_RECEIVER. Worker only does byte memcpy.

V5 v1.0 spawns 2 GhostMigrationWorker instances (one per pathway: A→B and
B→A). Pathway A→B touches sim_a.sender_staging + sim_b.receiver_staging +
sim_b.timeline; B→A is disjoint. Since E7 B2 (2026-10-10) the two workers of a
link share two DestGuardRelay objects, one per endpoint (sim, direction): each
worker publishes its source observation to one and its dest guard reads the
other (see DestGuardRelay); nothing else is shared.
"""

from __future__ import annotations

import queue
import os
import struct
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7


_STOP_SENTINEL = -1   # frame_n value that means "stop the worker thread"

# E29 step trace: host clock of the per-frame time points below.
# perf_counter_ns = QueryPerformanceCounter (Windows) / CLOCK_MONOTONIC (Linux),
# the domains VK_KHR_calibrated_timestamps maps GPU ticks onto;
# phase_trace_v7 switches it to CLOCK_MONOTONIC_RAW when the driver offers
# only that domain.
_host_clock_ns = time.perf_counter_ns


def set_host_clock(clock_ns) -> None:
    global _host_clock_ns
    _host_clock_ns = clock_ns


# E7 B2 (2026-10-10): the dest guard without a second host sleeper. The worker
# of link a -> b host-signals worker_done(n) = 2n+2 on b's transport timeline of
# its receive direction only after that timeline has reached b's own
# readback_done(n) = 2n+1 (the dest guard, see _run). That (semaphore, value)
# is also the SOURCE wait of the reverse worker b -> a. E7 batch 1 (N56, Linux
# driver 580.82.07): when both threads slept in vkWaitSemaphores on it, one of
# them sometimes woke ~10 ms late; every late wake in 16 traces had a second
# sleeper on the value, and 318 lone waits were never late. With the relay the
# reverse worker's source wait is the value's only blocking host waiter: it
# publishes the frame to the link's DestGuardRelay right after its wait
# returns, and the dest guard completes on a zero-timeout vkWaitSemaphores or
# on that relay. The GPU protocol, semaphore values, command buffers and
# numerics are rc1's.
# V7_DEST_GUARD: relay (default) = the B2 guard above; wait = rc1's blocking
# vkWaitSemaphores (INFINITE) in the dest guard, no relays.
# V7_DEST_GUARD_PRECHECK (relay only), what the dest guard does on arrival:
#   counter (default) = vkGetSemaphoreCounterValue (a query, not a wait):
#     reached -> a zero-timeout vkWaitSemaphores as the formal host wait, done;
#     else the relay, then that zero-timeout wait. The guard calls
#     vkWaitSemaphores only once the value is reached, so never while the
#     reverse worker sleeps on a value that is still pending;
#   zero_wait = a zero-timeout vkWaitSemaphores on arrival (itself a Vulkan
#     host wait operation): reached -> done; else the relay, then the
#     zero-timeout wait. The arrival call can fall inside the reverse
#     worker's sleep on the pending value;
#   none = the relay, then the zero-timeout wait.
# (The E7 wake reproduction, docs/cluster_v6/scripts/e7_wake_repro.py on the
# e7-cluster branch, measures whether a non-blocking call next to a sleeper
# delays its wake: its zeropoll / counterpoll cases.)
# Read once at import; any other value is refused, like the E39 switches.
_DEST_GUARD_ACCEPTED = ("relay", "wait")
_DEST_GUARD_PRECHECK_ACCEPTED = ("zero_wait", "counter", "none")


def _parse_dest_guard(text: str) -> str:
    """V7_DEST_GUARD: one of _DEST_GUARD_ACCEPTED (case and surrounding blanks ignored)."""
    value = text.strip().lower()
    if value not in _DEST_GUARD_ACCEPTED:
        raise ValueError(f"V7_DEST_GUARD={text!r}: accepted values are {', '.join(_DEST_GUARD_ACCEPTED)} "
                         "(relay = E7 B2, the dest guard completes on a zero-timeout wait or on the reverse "
                         "worker's relay; wait = rc1's blocking vkWaitSemaphores)")
    return value


def _parse_dest_guard_precheck(text: str) -> str:
    """V7_DEST_GUARD_PRECHECK: one of _DEST_GUARD_PRECHECK_ACCEPTED (case and surrounding blanks ignored)."""
    value = text.strip().lower()
    if value not in _DEST_GUARD_PRECHECK_ACCEPTED:
        raise ValueError(f"V7_DEST_GUARD_PRECHECK={text!r}: accepted values are "
                         f"{', '.join(_DEST_GUARD_PRECHECK_ACCEPTED)} (counter = vkGetSemaphoreCounterValue on "
                         "arrival, reached -> a zero-timeout wait; zero_wait = a zero-timeout vkWaitSemaphores on "
                         "arrival; none = no arrival check, always the relay)")
    return value


_DEST_GUARD = _parse_dest_guard(os.environ.get("V7_DEST_GUARD", "relay"))
_DEST_GUARD_PRECHECK = _parse_dest_guard_precheck(os.environ.get("V7_DEST_GUARD_PRECHECK", "counter"))
# Upper bound of one Condition wait inside a relay wait: each round re-checks
# the abort and release flags, so a relay never holds a thread past them for
# longer.
_RELAY_ROUND_TIMEOUT_S = 0.1
# GhostMigrationWorker.stop(): how long the worker may drain its queued frames
# (rc1's join timeout), and how long it gets after its relays are released.
_STOP_JOIN_TIMEOUT_S = 10.0
_STOP_RELEASE_JOIN_TIMEOUT_S = 5.0


def configured_dest_guard_switches() -> dict[str, str]:
    """E7 B2 switches with the values this process runs (read at import), for
    simulator_v7.configured_v7_switches (run headers, step-trace run_meta).
    The pre-check reads "n/a" under V7_DEST_GUARD=wait, where it has no
    effect."""
    return {"V7_DEST_GUARD": _DEST_GUARD,
            "V7_DEST_GUARD_PRECHECK": _DEST_GUARD_PRECHECK if _DEST_GUARD == "relay" else "n/a"}


class DestGuardRelayAborted(RuntimeError):
    """A dest guard's relay was aborted: a worker of the link died."""


class DestGuardRelay:
    """E7 B2: one (sim, peer direction) of a link. published_frame is the last
    frame n for which the PUBLISHER - the worker whose source is this (sim,
    direction) - saw its source wait on the sim's readback_done(n) return
    (the sim's transport timeline of that direction >= its readback_done(n)).
    The CONSUMER - the worker whose dest is this (sim, direction) - waits that
    same (semaphore, value) in its dest guard and reads it here instead of
    sleeping a second time in vkWaitSemaphores.

    published_frame starts at -1, not 0: with 0, frame 0 would pass before the
    sim's readback(0), and the host signal of worker_done(0) could precede the
    GPU signal of readback_done(0). It only grows (max), and the consumer
    compares with '>=': when the consumer reaches frame n the counter holds
    n-1 or n, never more (publishing m needs the sim's phase A(m), which
    follows the consumer's worker_done(m-1)), so a counter, never per-frame
    events that get cleared. Deadlock-free because every worker publishes
    BEFORE its own dest guard: a publication of frame n never waits on the
    consumer's frame n.

    abort() (a worker of the link died: its exception path) wakes the
    consumer, which then raises DestGuardRelayAborted. release() (stop() of a
    worker that did not drain within its join timeout) wakes it too, and the
    consumer then takes rc1's blocking vkWaitSemaphores for that frame, so a
    stop is never worse than rc1's. Each Condition wait is also bounded
    (_RELAY_ROUND_TIMEOUT_S)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.condition = threading.Condition()
        self.published_frame = -1
        self.aborted = False
        self.abort_reason: Optional[str] = None
        self.released = False
        self.release_reason: Optional[str] = None
        self.publisher_label: Optional[str] = None
        self.consumer_label: Optional[str] = None

    def publish(self, frame_n: int) -> None:
        with self.condition:
            if frame_n > self.published_frame:
                self.published_frame = frame_n
            self.condition.notify_all()

    def abort(self, reason: str) -> None:
        with self.condition:
            if not self.aborted:
                self.aborted = True
                self.abort_reason = reason
            self.condition.notify_all()

    def release(self, reason: str) -> None:
        with self.condition:
            if not self.released:
                self.released = True
                self.release_reason = reason
            self.condition.notify_all()

    def wait_published(self, frame_n: int, round_timeout_s: Optional[float] = None) -> tuple[bool, bool]:
        """Block until published_frame >= frame_n. Returns (published, slept):
        published is False only when the relay was released first (the
        consumer then waits the semaphore itself); slept is True if it had to
        sleep on the Condition. Raises DestGuardRelayAborted when the relay is
        aborted first. A frame already published passes either way."""
        round_timeout = _RELAY_ROUND_TIMEOUT_S if round_timeout_s is None else round_timeout_s
        slept = False
        with self.condition:
            while self.published_frame < frame_n:
                if self.aborted:
                    raise DestGuardRelayAborted(
                        f"dest guard relay {self.label} aborted at frame {frame_n} (published "
                        f"{self.published_frame}): {self.abort_reason}")
                if self.released:
                    return False, slept
                self.condition.wait(round_timeout)
                slept = True
        return True, slept


def connect_dest_guard_relays(workers) -> dict:
    """E7 B2: give the workers of every link their DestGuardRelay pair, before
    the workers start. One relay per (sim, peer direction) that has both a
    publisher (the worker whose source it is) and a consumer (the worker whose
    dest it is); the chain and dual orchestrators build both workers of every
    link, so every (sim, direction) with a peer gets one. A consumer without a
    publisher gets none and keeps rc1's blocking wait (it says so at its first
    dest guard); so does a pair whose source op and dest guard op are not the
    same (semaphore, value) (a sync scheme where the relay would not apply).
    V7_DEST_GUARD=wait (read by each worker at construction): no relays.
    Returns {(sim, direction) key: relay}; K = 1 (no workers): {} silently."""
    if not workers:
        return {}
    if any(worker._dest_guard_mode != "relay" for worker in workers):
        print(f"[transport_v7] dest guard: V7_DEST_GUARD=wait (rc1 blocking vkWaitSemaphores), "
              f"no relays for {len(workers)} workers", flush=True)
        return {}
    publishers = {}
    for worker in workers:
        key = (id(worker.source), worker.source_direction)
        if key in publishers:
            raise ValueError(f"workers {publishers[key].label} and {worker.label} both send from one "
                             f"(sim, {worker.source_direction})")
        publishers[key] = worker
    consumers = {}
    relays = {}
    for worker in workers:
        key = (id(worker.dest), worker.dest_direction)
        if key in consumers:
            raise ValueError(f"workers {consumers[key].label} and {worker.label} both receive into one "
                             f"(sim, {worker.dest_direction})")
        consumers[key] = worker
        publisher = publishers.get(key)
        if publisher is None:
            continue
        same_operation = all(
            publisher.source.sync.source_readback_op(publisher.source_direction, frame_n)
            == worker.dest.sync.dest_guard_op(worker.dest_direction, frame_n) for frame_n in (0, 1))
        if not same_operation:
            print(f"[transport_v7] dest guard of {worker.label}: its dest guard op is not {publisher.label}'s "
                  f"source op, no relay (rc1 blocking wait)", flush=True)
            continue
        relay = DestGuardRelay(f"{publisher.label}->{worker.label}")
        relay.publisher_label, relay.consumer_label = publisher.label, worker.label
        publisher.attach_publish_relay(relay)
        worker.attach_consume_relay(relay)
        relays[key] = relay
    print(f"[transport_v7] dest guard: V7_DEST_GUARD=relay V7_DEST_GUARD_PRECHECK={_DEST_GUARD_PRECHECK}, "
          f"{len(relays)} relays for {len(workers)} workers", flush=True)
    return relays


class GhostMigrationWorker:
    """One pathway (source → dest) persistent worker thread.

    Per-frame main loop (semaphores/values come from each sim's sync scheme —
    see sync_scheme_v7.py; aggregated = shared 5N timeline, per-direction =
    the direction's own transport timeline):
        1. wait source readback_done(n)  [sender_staging fully populated];
           publish n to the source's DestGuardRelay (E7 B2)
        2. dest guard: dest readback_done(n) reached [backwards-signal guard,
           see _run and _complete_dest_guard: the counter or the relay of the
           reverse worker, then a zero-timeout wait; never a second sleeper]
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
        source_sim: "SphSimulatorV7",
        dest_sim: "SphSimulatorV7",
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
        # Frame-stamp host check state (segment 13 of the staging layout).
        self._stamp_base: Optional[int] = None
        self.stamp_error_count = 0
        if self._source_view.nbytes != self._dest_view.nbytes:
            raise ValueError(
                f"worker {label}: source/dest staging sizes mismatch — "
                f"{self._source_view.nbytes} vs {self._dest_view.nbytes}. "
                f"Likely partition / GhostTransportConfig misconfigured.")

        # Count-aware copy plan (2026-09-15, N56 8-GPU diagnosis): every
        # per-particle segment is sized stride x ghost_pool_size, so the
        # legacy whole-view copy moves the full POOL capacity every frame
        # (25 MB per direction at 64M vs ~2.6 MB of live ghosts -> 1.5-2.8 ms
        # of memcpy per link). With V7_WORKER_COUNT_AWARE=1 the worker reads
        # the sender's ghost count (segment 12) and copies only count*stride
        # bytes of each SoA segment; the voxel-indexed segments (10, 11) and
        # the count/stamp words (12, 13) are copied in full.
        self._count_aware = os.environ.get("V7_WORKER_COUNT_AWARE", "1") == "1"   # default on since E6b
        # V6: (staging_offset, size, stride, count_staging_offset or None).
        # Each per-particle segment names the staging word that holds its live
        # slot count (the sender's allocation counter for that region): the V5
        # layout's nine segments all use the ghost send count; the two-layer
        # layout's replica regions use their own replica counters. Segments
        # without a count (voxel lists, count words, stamp) are copied in full.
        self._copy_plan: list[tuple[int, int, int, Optional[int]]] = []
        self.last_copy_bytes = 0
        # Per-frame byte accounting for the seam report: host copy bytes per
        # frame (live bytes when count-aware) and the DMA size per frame.
        self.total_copy_bytes = 0
        self.copy_frame_count = 0
        self.staging_bytes = self._source_view.nbytes
        if self._count_aware:
            segments = source_sim._transport_segments[source_direction]
            for segment in segments:
                if segment.count_staging_offset is not None:
                    self._copy_plan.append((segment.staging_offset, segment.size,
                                            segment.stride, segment.count_staging_offset))
                else:
                    self._copy_plan.append((segment.staging_offset, segment.size, 0, None))

        # V7_POOL_PEAKS=1: record every frame's live slot count of each pool
        # region (the sender's allocation counter = demand, also when it
        # exceeds the region) for the pool-capacity study. Off by default:
        # the reads sit on the worker's critical path.
        self.region_capacity: dict[str, int] = {}
        self.region_counts: dict[str, list] = {}
        self._region_words: list[tuple[str, int]] = []
        if os.environ.get("V7_POOL_PEAKS", "0") == "1":
            for segment in source_sim._transport_segments[source_direction]:
                region = getattr(segment, "region", None)
                if region is None or region in self.region_capacity:
                    continue
                self.region_capacity[region] = segment.size // segment.stride
                self.region_counts[region] = []
                self._region_words.append((region, segment.count_staging_offset))

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

        # E7 B2 dest guard (see DestGuardRelay): the relay this worker
        # publishes its source observations to and the one its dest guard
        # reads; both attached by connect_dest_guard_relays before start(), so
        # without a relay (V7_DEST_GUARD=wait, or no reverse worker) the dest
        # guard is rc1's blocking wait. How each guard completed, per worker:
        # precheck (the value was reached on arrival: counter, then the
        # zero-timeout wait; or zero_wait's zero-timeout wait), relay (relay,
        # then the zero-timeout wait), fallback (blocking wait: that wait
        # unexpectedly not reached, or the relay released by stop()),
        # blocking (no relay: rc1's wait).
        self._dest_guard_mode = _DEST_GUARD
        self._dest_guard_precheck = _DEST_GUARD_PRECHECK
        self._publish_relay: Optional[DestGuardRelay] = None
        self._consume_relay: Optional[DestGuardRelay] = None
        self._missing_relay_reported = False
        self.dest_guard_counts = {"precheck": 0, "relay": 0, "fallback": 0, "blocking": 0}
        self.dest_guard_relay_slept = 0         # relay completions that slept on the Condition
        self.dest_guard_released = 0            # fallbacks because stop() released the relay

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

    def attach_publish_relay(self, relay: DestGuardRelay) -> None:
        """E7 B2: publish every source observation to ``relay`` (before start)."""
        if self._started:
            raise RuntimeError(f"worker {self.label}: relays must be attached before start()")
        self._publish_relay = relay

    def attach_consume_relay(self, relay: DestGuardRelay) -> None:
        """E7 B2: complete the dest guard through ``relay`` (before start)."""
        if self._started:
            raise RuntimeError(f"worker {self.label}: relays must be attached before start()")
        self._consume_relay = relay

    def _abort_relays(self, reason: str) -> None:
        """Wake every thread waiting on a relay this worker publishes to or
        reads; a consumer that has not got its frame then raises."""
        for relay in (self._publish_relay, self._consume_relay):
            if relay is not None:
                relay.abort(reason)

    def _release_relays(self, reason: str) -> None:
        """Wake every thread waiting on a relay this worker publishes to or
        reads; a consumer that has not got its frame then takes rc1's
        blocking wait for it."""
        for relay in (self._publish_relay, self._consume_relay):
            if relay is not None:
                relay.release(reason)

    def _dest_guard_mode_text(self) -> str:
        if self._consume_relay is not None:
            return f"V7_DEST_GUARD=relay, V7_DEST_GUARD_PRECHECK={self._dest_guard_precheck}"
        return f"no relay, V7_DEST_GUARD={self._dest_guard_mode}"

    def dest_guard_record(self) -> dict:
        """How this worker's dest guards completed so far (E7 B2), for the
        step-trace run_meta (ChainOrchestratorV7.dest_guard_record)."""
        return {**self.dest_guard_counts, "relay_slept": self.dest_guard_relay_slept,
                "released": self.dest_guard_released, "relay_attached": self._consume_relay is not None,
                "mode": self._dest_guard_mode_text()}

    def dest_guard_summary(self) -> str:
        """One line: how this worker's dest guards completed (E7 B2); the E7
        harness (parse_run_v6.py) reads it."""
        counts = self.dest_guard_counts
        return (f"[worker {self.label}] dest guard: precheck {counts['precheck']}, relay {counts['relay']}, "
                f"fallback {counts['fallback']}, blocking {counts['blocking']} (relay slept "
                f"{self.dest_guard_relay_slept}, released {self.dest_guard_released}; "
                f"{self._dest_guard_mode_text()})")

    def stop(self) -> None:
        """Best-effort stop. Safe on happy-path close (worker idle in
        work_queue.get); UNSAFE on exception paths. Prints the dest guard
        summary line (E7 B2).

        E7 B2: like rc1, the worker first drains the frames queued before the
        sentinel (their dest guards still complete through the relay, whose
        publisher keeps running until its own stop()). Only if it has not
        ended within _STOP_JOIN_TIMEOUT_S are its relays released: a worker of
        this link held by a relay then takes rc1's blocking vkWaitSemaphores
        for that frame (counted as fallback, released), so stop() is never
        worse than rc1's. A worker inside vkWaitSemaphores (its source wait,
        the upload guard, a blocking dest guard) still cannot be woken:
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
        self.thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        if self.thread.is_alive():
            self._release_relays(f"worker {self.label} stop(): not ended after {_STOP_JOIN_TIMEOUT_S:g} s")
            self.thread.join(timeout=_STOP_RELEASE_JOIN_TIMEOUT_S)
        self._started = False
        print(self.dest_guard_summary(), flush=True)

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
        # 2026-09-15 NUMA pinning experiment: V7_WORKER_AFFINITY="cpulist0;cpulist1;..."
        # indexed by the DEST sim's physical device index (job script builds it
        # from sysfs numa_node -> node cpulist). Pins this worker thread.
        affinity = os.environ.get("V7_WORKER_AFFINITY")
        if affinity:
            try:
                ctx = getattr(self.dest, "ctx", None) or getattr(self.dest, "context", None)
                device_index = int(getattr(ctx, "physical_device_index"))
                cpulist = affinity.split(";")[device_index]
                cpus = set()
                for part in cpulist.split(","):
                    if "-" in part:
                        lo, hi = part.split("-"); cpus.update(range(int(lo), int(hi) + 1))
                    elif part.strip():
                        cpus.add(int(part))
                os.sched_setaffinity(0, cpus)
                print(f"[worker {self.label}] pinned to dest device {device_index} cpus {cpulist}", flush=True)
            except Exception as error:      # noqa: BLE001
                print(f"[worker {self.label}] affinity pin skipped: {error!r}", flush=True)
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
                t_dequeue = _host_clock_ns()
                self.last_activity = ("wait_source_timeline", frame_n, time.perf_counter_ns())
                source_semaphore, source_value = self.source.sync.source_readback_op(
                    self.source_direction, frame_n)
                self.source.wait_semaphore(source_semaphore, source_value)
                t_source_wait = _host_clock_ns()
                # E7 B2: this wait returned, so the source's transport
                # timeline of source_direction reached readback_done(n) -
                # the value the reverse worker's dest guard needs. Publish
                # BEFORE our own dest guard (deadlock freedom, DestGuardRelay).
                if self._publish_relay is not None:
                    self._publish_relay.publish(frame_n)
                # 1b. Wait for DEST sim's readback_done(n) on the SAME
                #     semaphore we are about to host-signal. Critical for
                #     timeline monotonicity: our host_signal of worker_done
                #     must come AFTER the pending GPU signal below it
                #     (dest's own readback), otherwise the GPU signal would
                #     land "backwards" relative to ours. This is the
                #     sync-scheme safety invariant: before host-signaling
                #     value v on semaphore S, wait S >= v-1. Through it the
                #     upload(n) of the dest also follows the dest's own
                #     ghost_send(n) / readback(n) (same device range) and, at
                #     depth 2, phase C(n-1). E7 B2: completed without a
                #     second sleeper on the value (_complete_dest_guard).
                self.last_activity = ("wait_dest_timeline", frame_n, time.perf_counter_ns())
                guard_semaphore, guard_value = self.dest.sync.dest_guard_op(
                    self.dest_direction, frame_n)
                self._complete_dest_guard(frame_n, guard_semaphore, guard_value)
                t_dest_guard = _host_clock_ns()
                # 1c. Wait until dest's upload of frame n-1 has finished
                #     READING receiver_staging before we overwrite it.
                #     Transfer-queue FIFO completion order is NOT a spec
                #     guarantee — relying on it caused reproducible drift
                #     on the 3090 cluster (see dest_upload_guard_ops).
                for upload_guard_semaphore, upload_guard_value in (
                        self.dest.sync.dest_upload_guard_ops(
                            self.dest_direction, frame_n)):
                    self.dest.wait_semaphore(upload_guard_semaphore,
                                             upload_guard_value)
                t_wait = _host_clock_ns()

                # 1c-bis. Host-side frame-stamp check: the LAST 4 bytes of the
                # sender staging carry the sender GPU's frame_stamp (segment
                # 13). It must advance by exactly 1 per frame; anything else
                # means the readback delivered stale/torn bytes DESPITE the
                # semaphore wait — the exact failure the cluster residual-race
                # hunt is trying to localize.
                stamp = struct.unpack_from(
                    "<I", self._source_view, len(self._source_view) - 4)[0]
                if self._stamp_base is None:
                    self._stamp_base = stamp - frame_n
                elif stamp != self._stamp_base + frame_n:
                    self.stamp_error_count += 1
                    if self.stamp_error_count <= 5:
                        print(f"[worker {self.label}] *** STALE READBACK at "
                              f"frame {frame_n}: stamp={stamp} expected="
                              f"{self._stamp_base + frame_n} ***", flush=True)
                t_stamp = _host_clock_ns()

                for region, count_offset in self._region_words:
                    self.region_counts[region].append(struct.unpack_from(
                        "<I", self._source_view, count_offset)[0])

                # 2. Byte memcpy (CPU → CPU)
                self.last_activity = ("memcpy", frame_n, time.perf_counter_ns())
                if self._count_aware:
                    copied = 0
                    for staging_offset, size, stride, count_offset in self._copy_plan:
                        if count_offset is not None:
                            live_count = struct.unpack_from(
                                "<I", self._source_view, count_offset)[0]
                            n = min(size, live_count * stride)
                        else:
                            n = size
                        if n:
                            self._dest_view[staging_offset:staging_offset + n] = \
                                self._source_view[staging_offset:staging_offset + n]
                            copied += n
                    self.last_copy_bytes = copied
                else:
                    self._dest_view[:] = self._source_view
                    self.last_copy_bytes = self._source_view.nbytes
                self.total_copy_bytes += self.last_copy_bytes
                self.copy_frame_count += 1
                t_copy = _host_clock_ns()

                # 2b. Consumed-ack on the SOURCE: sender_staging(frame_n) has
                # been fully read — the source's readback(frame_n+1) waits
                # this before overwriting it. THE fix for the stale-readback
                # race the frame stamps caught (see consumed_signal_op).
                consumed_semaphore, consumed_value = (
                    self.source.sync.consumed_signal_op(
                        self.source_direction, frame_n))
                self.source.host_signal_semaphore(consumed_semaphore,
                                                  consumed_value)

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
                t_dest_signal = _host_clock_ns()
                self.dest.host_signal_semaphore(signal_semaphore, signal_value)
                t_signal = _host_clock_ns()
                self.last_activity = ("done_frame", frame_n, time.perf_counter_ns())
                self.last_completed_frame = frame_n

                self.timestamps[frame_n] = {
                    # Segment boundaries (host clock, ns) — diff neighbours
                    # for per-exchange accounting: dequeue -> source-readback
                    # wait -> dest guard wait -> upload guard wait -> frame
                    # stamp check -> count words + memcpy -> host signals.
                    "dequeue_ns": t_dequeue,
                    "source_wait_ns": t_source_wait,
                    "dest_guard_ns": t_dest_guard,
                    "wait_ns": t_wait,
                    "stamp_ns": t_stamp,
                    "copy_ns": t_copy,
                    "dest_signal_ns": t_dest_signal,      # right before the worker_done host signal
                    "signal_ns": t_signal,                # both host signals returned
                    "copy_bytes": self.last_copy_bytes,
                }
        except BaseException as e:  # noqa: BLE001 — capture everything for diagnostics
            self.last_error = e
            # E7 B2: the partner may wait on a relay this worker will never
            # publish again; wake it (it raises, the orchestrator reports).
            self._abort_relays(f"worker {self.label} died: {e!r}")
            import traceback as _tb
            print(f"[worker {self.label}] DIED at {self.last_activity}: {e!r}",
                  file=_sys.stderr, flush=True)
            _tb.print_exc(file=_sys.stderr)

    def _complete_dest_guard(self, frame_n: int, guard_semaphore, guard_value: int) -> None:
        """The dest guard of frame n: return once the dest's transport
        timeline of dest_direction has reached guard_value = its
        readback_done(n), observed through a Vulkan host wait operation.

        Without a relay (V7_DEST_GUARD=wait, or no worker publishes this
        (sim, direction)): rc1's blocking vkWaitSemaphores. With one (E7 B2),
        never a sleep in the driver:
          (a) pre-check on arrival (V7_DEST_GUARD_PRECHECK): counter =
              vkGetSemaphoreCounterValue, reached -> the zero-timeout wait of
              (c), done; zero_wait = a zero-timeout vkWaitSemaphores, reached
              -> done; none = nothing;
          (b) the relay: published >= n means the reverse worker's source wait
              on this very (semaphore, value) returned (Condition rounds;
              abort raises, release by stop() goes to the blocking wait);
          (c) a zero-timeout vkWaitSemaphores as the formal host wait; it
              cannot miss after (a) or (b) (timeline values only grow), and if
              it ever did, the blocking wait follows and is counted as
              fallback.
        The assert before the worker_done signal (_run) still reads the
        counter itself."""
        relay = self._consume_relay
        if relay is None:
            if self._dest_guard_mode == "relay" and not self._missing_relay_reported:
                self._missing_relay_reported = True
                print(f"[worker {self.label}] dest guard: no relay for the dest's {self.dest_direction} "
                      f"direction (no reverse worker registered) -> rc1 blocking vkWaitSemaphores",
                      flush=True)
            self.dest.wait_semaphore(guard_semaphore, guard_value)
            self.dest_guard_counts["blocking"] += 1
            return
        reached_on_arrival = False
        if self._dest_guard_precheck == "zero_wait":
            if self.dest.wait_semaphore(guard_semaphore, guard_value, timeout_ns=0):
                self.dest_guard_counts["precheck"] += 1
                return
        elif self._dest_guard_precheck == "counter":
            reached_on_arrival = self.dest.semaphore_value(guard_semaphore) >= guard_value
        published = True
        if not reached_on_arrival:
            self.last_activity = ("wait_dest_relay", frame_n, time.perf_counter_ns())
            published, slept = relay.wait_published(frame_n)
            if slept:
                self.dest_guard_relay_slept += 1
        if published and self.dest.wait_semaphore(guard_semaphore, guard_value, timeout_ns=0):
            self.dest_guard_counts["precheck" if reached_on_arrival else "relay"] += 1
            return
        self.dest_guard_counts["fallback"] += 1
        if not published:
            self.dest_guard_released += 1
            reason = f"relay released ({relay.release_reason})"
        else:
            reason = (f"{'counter' if reached_on_arrival else 'relay'} reached but the zero-timeout wait "
                      f"missed (counter {self.dest.semaphore_value(guard_semaphore)})")
        if self.dest_guard_counts["fallback"] <= 5:
            print(f"[worker {self.label}] *** dest guard frame {frame_n} (value {guard_value}): {reason}; "
                  f"blocking wait ***", flush=True)
        self.last_activity = ("wait_dest_timeline", frame_n, time.perf_counter_ns())
        self.dest.wait_semaphore(guard_semaphore, guard_value)
