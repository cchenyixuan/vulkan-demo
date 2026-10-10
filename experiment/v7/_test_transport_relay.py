"""
_test_transport_relay.py — E7 B2 tests of the transport worker's dest guard relay (transport_v7: DestGuardRelay,
connect_dest_guard_relays, GhostMigrationWorker._complete_dest_guard). Zero-dependency runnable script (assert +
non-zero exit on failure), CPU only: fake timeline semaphores and fake simulators, no Vulkan device, no case files.

  1  relay: starts at -1 (frame 0 does not pass), publish keeps the maximum, a wait returns at published >= n and
     sleeps until the publish, an abort wakes a sleeper at once and raises, a release wakes it at once and returns
     not-published, an abort or release flag set without a notify is seen within one bounded round, a frame already
     published passes an aborted or released relay.
  2  switches: V7_DEST_GUARD / V7_DEST_GUARD_PRECHECK parsing (accepted values, case and blanks, refusals), the
     defaults in the source (relay / counter), configured_dest_guard_switches (pre-check "n/a" under wait) and
     simulator_v7.configured_v7_switches report both.
  3  wiring: connect_dest_guard_relays on fake chains K = 2 / 3 / 4: one relay per (sim, peer direction), publisher =
     the worker whose source it is, consumer = the worker whose dest it is; K = 1 (no workers) prints nothing;
     V7_DEST_GUARD=wait attaches none; a consumer without a publisher keeps the blocking wait and says so once; two
     senders from one (sim, direction) raise; relays only before start().
  4  protocol: the per-direction frame protocol (sync_scheme_v7.PerDirectionTimelineScheme, its values) on fake
     timeline semaphores. Every sim has one thread per GPU queue (compute: phase A / C; transfer: readback; transfer:
     upload) that moves a (frame, origin) packet through a device range send and receive share, as the solver's
     ghost range: phase A writes the outbound packet, readback(n) copies it out, upload(n) overwrites it with the
     neighbour's packet, phase C(n) reads that. The real GhostMigrationWorker threads (sync_scheme ops, stamp check,
     memcpy, assert, host signals) and the chain orchestrator's depth-2 loop (all submits, then the notifies; worker
     queue depth 4) drive it: K = 2 with 10,000 frames per link for every pre-check (zero_wait, counter, none) and
     for V7_DEST_GUARD=wait (rc1), K = 3 (an interior sim, two relays) with 3,000 frames. Checks: every timeline
     signal raises the value (a host signal of 2n+2 ahead of the GPU's 2n+1 would not), upload(n) starts after the
     receiver's own readback(n), every readback copies its own frame's packet and every phase C reads the
     neighbour's frame-n packet, no stale stamp, no deadlock (bounded run time), the guard counts add up with no
     fallback, at most one sleeping host waiter per (semaphore, value) and no dest guard asleep in a semaphore wait
     (relay), the upload guard never blocks; with V7_DEST_GUARD=wait the shared sleepers of rc1 are seen. Zero-timeout
     calls are recorded: with counter and none the guard never makes one on a pending value (zero_wait does, the
     co-call the cluster reproduction measures; its count is printed). The K = 3 runs cover all three pre-checks.
  5  negative controls: a relay that starts at 0 lets frame 0 past the relay before the receiver's readback(0), and
     the formal zero-timeout wait catches it (fallback); a counter that reads "reached" too early is caught the same
     way; a worker without any dest guard dies on the assert before its host signal.
  6  abort and stop: a worker that dies at frame k wakes its partner asleep on the relay (DestGuardRelayAborted, the
     orchestrator loop sees the death, no thread left); stop() with a frame in flight drains it like rc1 (both
     workers finish the frame through the relay, no fallback); stop() of a worker held by a relay that is never
     published releases it after the join timeout and the worker takes rc1's blocking wait (fallback, released),
     finishing the frame once the value arrives.

Usage:
    .venv/Scripts/python.exe experiment/v7/_test_transport_relay.py [--frames 10000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import queue
import random
import re
import sys
import threading
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _name in ("V7_WORKER_AFFINITY", "V7_POOL_PEAKS", "V7_WORKER_COUNT_AWARE", "V7_DEST_GUARD",
              "V7_DEST_GUARD_PRECHECK"):
    os.environ.pop(_name, None)

import experiment.v7.utils.transport_v7 as transport_v7  # noqa: E402
from experiment.v7.utils.sync_scheme_v7 import PerDirectionTimelineScheme  # noqa: E402
from experiment.v7.utils.transport_v7 import (  # noqa: E402
    DestGuardRelay,
    DestGuardRelayAborted,
    GhostMigrationWorker,
    connect_dest_guard_relays,
)

INFINITE_TIMEOUT = 0xFFFFFFFFFFFFFFFF
PACKET_WORDS = 2                                    # (frame, origin sim)
STAGING_BYTES = 4 * (PACKET_WORDS + 1)              # the packet + the frame stamp word (the last 4 bytes)
STAMP_BASE = 7000
WORKER_QUEUE_DEPTH = 4                              # ChainOrchestratorV7's default
PIPELINE_DEPTH = 2
STALL_SECONDS = 20.0

_passed = 0
_failed: list[str] = []


def check(condition: bool, message: str) -> None:
    global _passed
    if condition:
        _passed += 1
    else:
        _failed.append(message)
        print(f"  FAIL: {message}")


@contextlib.contextmanager
def dest_guard_switches(mode: str, precheck: str):
    """The module constants read at import (V7_DEST_GUARD / V7_DEST_GUARD_PRECHECK), set for one test."""
    saved = transport_v7._DEST_GUARD, transport_v7._DEST_GUARD_PRECHECK
    transport_v7._DEST_GUARD, transport_v7._DEST_GUARD_PRECHECK = mode, precheck
    try:
        yield
    finally:
        transport_v7._DEST_GUARD, transport_v7._DEST_GUARD_PRECHECK = saved


# ----------------------------------------------------------------------------- fakes


class Shutdown(Exception):
    """The harness ended a run: every fake wait still blocked raises this."""


class Harness:
    """Shared state of one protocol run: the shutdown flag, protocol violations, the event sequence, wait records."""

    def __init__(self, seed: int) -> None:
        self.shutdown = threading.Event()
        self.violations: list[str] = []
        self.sequence_lock = threading.Lock()
        self.sequence = 0
        self.wait_records: list[tuple] = []         # (thread, semaphore, value, blocked) for every timed wait
        self.zero_wait_records: list[tuple] = []    # (thread, semaphore, value, reached) for every zero-timeout wait
        self.seed = seed
        self.local = threading.local()
        self.inject = None                          # callable(thread name, semaphore, value) before a fake wait

    def violation(self, message: str) -> None:
        if len(self.violations) < 50:
            self.violations.append(message)
        else:
            self.violations[-1] = f"... more violations, last: {message}"

    def next_sequence(self) -> int:
        with self.sequence_lock:
            self.sequence += 1
            return self.sequence

    def generator(self) -> random.Random:
        generator = getattr(self.local, "generator", None)
        if generator is None:
            generator = self.local.generator = random.Random(f"{self.seed}:{threading.current_thread().name}")
        return generator

    def jitter(self) -> None:
        """Random timing: nothing, a yield, or a short spin (holding the GIL like Python work)."""
        draw = self.generator().random()
        if draw < 0.55:
            return
        if draw < 0.85:
            time.sleep(0)
            return
        end = time.perf_counter() + self.generator().random() * 60e-6
        while time.perf_counter() < end:
            pass


class FakeTimeline:
    """A timeline semaphore: monotonic value, blocking waits; a signal that does not raise the value is a
    protocol violation (the Vulkan backwards-signal hazard)."""

    def __init__(self, name: str, harness: Harness) -> None:
        self.name = name
        self.value = 0
        self.condition = threading.Condition()
        self.harness = harness

    def signal(self, value: int, source: str) -> None:
        with self.condition:
            if value <= self.value:
                self.harness.violation(f"{self.name}: {source} signals {value}, value already {self.value}")
            else:
                self.value = value
            self.condition.notify_all()

    def reached(self, value: int) -> bool:
        with self.condition:
            return self.value >= value

    def wait(self, value: int, timeout_s=None) -> bool:
        """Until value is reached (timeout_s None: until the harness shuts down, then Shutdown)."""
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        with self.condition:
            while self.value < value:
                if self.harness.shutdown.is_set():
                    raise Shutdown(f"{self.name} >= {value}")
                remaining = 0.05 if deadline is None else min(0.05, deadline - time.perf_counter())
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True


class FakeSegment:
    """What GhostMigrationWorker's count-aware copy plan reads of a transport segment (copied in full)."""

    def __init__(self, staging_offset: int, size: int) -> None:
        self.staging_offset = staging_offset
        self.size = size
        self.stride = 0
        self.count_staging_offset = None
        self.region = None


class FakeSimulator:
    """The worker-facing surface of SphSimulatorV7 (sync scheme ops, waits, signals, staging views) over fake
    semaphores, plus the GPU side of the frame protocol as three queue threads."""

    def __init__(self, index: int, slab_count: int, harness: Harness) -> None:
        self.index = index
        self.harness = harness
        self.directions = tuple(direction for direction, has_peer in (("leading", index > 0),
                                                                      ("trailing", index < slab_count - 1))
                                if has_peer)
        self.sync = PerDirectionTimelineScheme(self.directions)
        self.sync.main = FakeTimeline(f"s{index}.main", harness)
        self.sync.transport = {direction: FakeTimeline(f"s{index}.transport_{direction}", harness)
                               for direction in self.directions}
        self.sync.consumed = {direction: FakeTimeline(f"s{index}.consumed_{direction}", harness)
                              for direction in self.directions}
        self.sender = {direction: np.zeros(STAGING_BYTES, dtype=np.uint8) for direction in self.directions}
        self.receiver = {direction: np.zeros(STAGING_BYTES, dtype=np.uint8) for direction in self.directions}
        self.device_range = {direction: np.zeros(PACKET_WORDS, dtype=np.uint32) for direction in self.directions}
        self._transport_segments = {direction: [FakeSegment(0, STAGING_BYTES)] for direction in self.directions}
        self.readback_sequence: dict = {}           # (direction, frame) -> event sequence of readback done
        self.queues = {name: queue.Queue() for name in ("compute", "readback", "upload")}
        self.delays: dict = {}                      # (queue name, frame) -> seconds before that frame's work
        self.threads: list[threading.Thread] = []

    # -- worker-facing surface (simulator_v7 names)
    def sender_staging_view(self, direction: str):
        return self.sender[direction]

    def receiver_staging_view(self, direction: str):
        return self.receiver[direction]

    def wait_semaphore(self, semaphore: FakeTimeline, value: int, timeout_ns: int = INFINITE_TIMEOUT) -> bool:
        name = threading.current_thread().name
        if self.harness.inject is not None:
            self.harness.inject(name, semaphore, value)
        if name.startswith("ghost-"):
            self.harness.jitter()
        if timeout_ns == 0:
            reached = semaphore.reached(value)
            self.harness.zero_wait_records.append((name, semaphore.name, value, reached))
            return reached
        self.harness.wait_records.append((name, semaphore.name, value, not semaphore.reached(value)))
        if timeout_ns == INFINITE_TIMEOUT:
            return semaphore.wait(value)
        return semaphore.wait(value, timeout_ns / 1e9)

    def semaphore_value(self, semaphore: FakeTimeline) -> int:
        return semaphore.value

    def host_signal_semaphore(self, semaphore: FakeTimeline, value: int) -> None:
        self.harness.jitter()
        semaphore.signal(value, f"host ({threading.current_thread().name})")

    # -- GPU side
    def neighbour(self, direction: str) -> int:
        return self.index - 1 if direction == "leading" else self.index + 1

    def start_queues(self) -> None:
        for name, body in (("compute", self._compute_queue), ("readback", self._readback_queue),
                           ("upload", self._upload_queue)):
            thread = threading.Thread(target=self._queue_thread, args=(name, body), name=f"gpu-s{self.index}-{name}",
                                      daemon=True)
            self.threads.append(thread)
            thread.start()

    def submit(self, frame_n: int) -> None:
        for submit_queue in self.queues.values():
            submit_queue.put(frame_n)

    def stop_queues(self) -> None:
        for submit_queue in self.queues.values():
            submit_queue.put(None)

    def _queue_thread(self, name: str, body) -> None:
        try:
            while True:
                frame_n = self.queues[name].get()
                if frame_n is None:
                    return
                delay = self.delays.get((name, frame_n))
                if delay:
                    time.sleep(delay)
                body(frame_n)
        except Shutdown:
            return

    def _compute_queue(self, frame_n: int) -> None:
        main = self.sync.main
        # phase A(n): after C(n-1) by queue order (this thread); ghost_send writes the outbound packets into the
        # range the inbound data later lands in
        self.harness.jitter()
        for direction in self.directions:
            self.device_range[direction][:] = (frame_n, self.index)
        main.signal(self.sync.value_phase_a_done(frame_n), f"gpu s{self.index} phase A")
        # phase C(n): waits upload_done(n), reads every inbound packet
        main.wait(self.sync.phase_c_waits(frame_n)[0][1])
        self.harness.jitter()
        for direction in self.directions:
            packet = tuple(int(word) for word in self.device_range[direction])
            if packet != (frame_n, self.neighbour(direction)):
                self.harness.violation(f"s{self.index} phase C({frame_n}) {direction}: packet {packet}, expected "
                                       f"{(frame_n, self.neighbour(direction))}")
        main.signal(self.sync.value_frame_done(frame_n), f"gpu s{self.index} phase C")

    def _readback_queue(self, frame_n: int) -> None:
        for direction in self.directions:
            for semaphore, value in self.sync.readback_waits(direction, frame_n):
                semaphore.wait(value)
            self.harness.jitter()
            packet = self.device_range[direction].copy()
            if tuple(int(word) for word in packet) != (frame_n, self.index):
                self.harness.violation(f"s{self.index} readback({frame_n}) {direction}: copied {tuple(packet)}, "
                                       f"expected its own {(frame_n, self.index)}")
            self.sender[direction][:4 * PACKET_WORDS] = packet.view(np.uint8)
            self.sender[direction][-4:] = np.array([STAMP_BASE + frame_n], dtype=np.uint32).view(np.uint8)
            self.readback_sequence[(direction, frame_n)] = self.harness.next_sequence()
            for semaphore, value in self.sync.readback_signals(direction, frame_n, True):
                semaphore.signal(value, f"gpu s{self.index} readback")

    def _upload_queue(self, frame_n: int) -> None:
        for position, direction in enumerate(self.directions):
            for semaphore, value in self.sync.upload_waits(direction, frame_n):
                semaphore.wait(value)
            started = self.harness.next_sequence()
            readback_done = self.readback_sequence.get((direction, frame_n))
            if readback_done is None or readback_done > started:
                self.harness.violation(f"s{self.index} upload({frame_n}) {direction} started before its own "
                                       f"readback({frame_n})")
            self.harness.jitter()
            packet = self.receiver[direction][:4 * PACKET_WORDS].view(np.uint32).copy()
            expected = (frame_n, self.neighbour(direction))
            if tuple(int(word) for word in packet) != expected:
                self.harness.violation(f"s{self.index} upload({frame_n}) {direction}: receiver staging holds "
                                       f"{tuple(packet)}, expected {expected}")
            self.device_range[direction][:] = packet
            is_last = position == len(self.directions) - 1
            for semaphore, value in self.sync.upload_signals(direction, frame_n, is_last):
                semaphore.signal(value, f"gpu s{self.index} upload")


def build_chain(slab_count: int, harness: Harness, queue_depth: int = WORKER_QUEUE_DEPTH):
    """Fake sims + the workers of ChainOrchestratorV7.__init__ (same order and labels)."""
    sims = [FakeSimulator(index, slab_count, harness) for index in range(slab_count)]
    workers = []
    for index in range(slab_count - 1):
        workers.append(GhostMigrationWorker(source_sim=sims[index], dest_sim=sims[index + 1],
                                            source_direction="trailing", dest_direction="leading",
                                            label=f"s{index}_to_s{index + 1}", queue_depth=queue_depth))
        workers.append(GhostMigrationWorker(source_sim=sims[index + 1], dest_sim=sims[index],
                                            source_direction="leading", dest_direction="trailing",
                                            label=f"s{index + 1}_to_s{index}", queue_depth=queue_depth))
    return sims, workers


class WorkerDied(Exception):
    pass


def reports_dead_worker(error) -> bool:
    """The loop saw a dead worker: in its frame wait (WorkerDied) or in GhostMigrationWorker.notify (RuntimeError
    'worker ... died', the orchestrators' own fail-fast path)."""
    return isinstance(error, WorkerDied) or (isinstance(error, RuntimeError) and "died" in str(error))


class Deadlock(Exception):
    pass


def run_protocol(slab_count: int, frames: int, mode: str, precheck: str, seed: int,
                 prepare=None, inject=None) -> dict:
    """One run of the frame protocol; returns the observations (raises nothing: errors are in the result).
    stdout / stderr of every thread are captured for the whole run (output / stderr)."""
    harness = Harness(seed)
    harness.inject = inject
    output, error_output = io.StringIO(), io.StringIO()
    result = {"error": None, "frames_done": 0}
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
        with dest_guard_switches(mode, precheck):      # read at worker construction / connection
            sims, workers = build_chain(slab_count, harness)
            relays = connect_dest_guard_relays(workers)
        if prepare is not None:
            prepare(sims, workers, relays)
        started_at = time.perf_counter()
        for sim in sims:
            sim.start_queues()
        for worker in workers:
            worker.start()
        next_wait = 0
        last_progress = time.perf_counter()

        def wait_frame(frame_n: int) -> None:
            nonlocal last_progress
            for sim in sims:
                semaphore, value = sim.sync.frame_done_op(frame_n)
                while not sim.wait_semaphore(semaphore, value, timeout_ns=int(0.25e9)):
                    for worker in workers:
                        if worker.last_error is not None:
                            raise WorkerDied(f"worker {worker.label} died (frame {frame_n}): "
                                             f"{worker.last_error!r}")
                    if time.perf_counter() - last_progress > STALL_SECONDS:
                        raise Deadlock(f"frame {frame_n} not done after {STALL_SECONDS} s: "
                                       + "; ".join(f"{worker.label} {worker.last_activity[:2]}"
                                                   for worker in workers))
            last_progress = time.perf_counter()
            result["frames_done"] = frame_n + 1

        try:
            # ChainOrchestratorV7.run_pipelined: every submit of the frame, then the notifies; depth 2
            for frame_n in range(frames):
                for sim in sims:
                    sim.submit(frame_n)
                for worker in workers:
                    worker.notify(frame_n)
                while frame_n + 1 - next_wait >= PIPELINE_DEPTH:
                    wait_frame(next_wait)
                    next_wait += 1
            while next_wait < frames:
                wait_frame(next_wait)
                next_wait += 1
        except (WorkerDied, Deadlock, RuntimeError) as error:
            result["error"] = error
            harness.shutdown.set()
        result["seconds"] = time.perf_counter() - started_at
        if result["error"] is None:
            for sim in sims:
                sim.stop_queues()
            for sim in sims:
                for thread in sim.threads:
                    thread.join(timeout=10.0)
        for worker in workers:                      # ChainOrchestratorV7.destroy order
            worker.stop()
        harness.shutdown.set()
        for sim in sims:
            sim.stop_queues()
            for thread in sim.threads:
                thread.join(timeout=10.0)
    result.update({"harness": harness, "sims": sims, "workers": workers, "relays": relays,
                   "output": output.getvalue(), "stderr": error_output.getvalue(),
                   "alive_threads": [thread.name for thread in threading.enumerate()
                                     if thread.name.startswith(("ghost-", "gpu-")) and thread.is_alive()]})
    return result


def sleeping_waits(result: dict) -> dict:
    """{(semaphore, value): set of threads that called a timed wait before the value was reached}."""
    sleepers: dict = {}
    for thread_name, semaphore_name, value, blocked in result["harness"].wait_records:
        if blocked:
            sleepers.setdefault((semaphore_name, value), set()).add(thread_name)
    return sleepers


# ----------------------------------------------------------------------------- 1 relay


def test_relay_unit() -> None:
    print("[1] relay: initial -1, monotonic publish, waits, bounded rounds, abort")
    relay = DestGuardRelay("unit")
    check(relay.published_frame == -1, f"initial published_frame {relay.published_frame}, expected -1")
    outcome = {}

    def wait_frame_zero():
        try:
            outcome["result"] = relay.wait_published(0, round_timeout_s=0.02)
        except DestGuardRelayAborted as error:
            outcome["error"] = error

    waiter = threading.Thread(target=wait_frame_zero)
    waiter.start()
    time.sleep(0.15)
    check(waiter.is_alive() and not outcome, "frame 0 passed a relay that published nothing")
    relay.publish(0)
    waiter.join(timeout=1.0)
    check(not waiter.is_alive() and outcome.get("result") == (True, True), f"frame 0 after publish(0): {outcome}")

    relay.publish(5)
    relay.publish(3)
    check(relay.published_frame == 5, f"publish(5) then publish(3): {relay.published_frame}, expected 5")
    check(relay.wait_published(4) == (True, False) and relay.wait_published(5) == (True, False),
          "frames <= 5 should pass without sleeping")

    woken = {}

    def wait_frame_six():
        woken["result"] = relay.wait_published(6)
        woken["at"] = time.perf_counter()

    waiter = threading.Thread(target=wait_frame_six)
    waiter.start()
    time.sleep(0.25)                                 # a few rounds without the frame
    check(waiter.is_alive(), "frame 6 passed before publish(6)")
    published_at = time.perf_counter()
    relay.publish(6)
    waiter.join(timeout=1.0)
    check(woken.get("result") == (True, True) and woken.get("at", 1e9) - published_at < 0.05,
          f"frame 6 should pass right after publish(6): {woken}, {woken.get('at', 0) - published_at:.4f} s")

    # abort wakes a sleeper at once; it raises
    relay = DestGuardRelay("abort")
    caught = {}

    def wait_aborted():
        try:
            relay.wait_published(0)
        except DestGuardRelayAborted as error:
            caught["error"] = error
            caught["at"] = time.perf_counter()

    waiter = threading.Thread(target=wait_aborted)
    waiter.start()
    time.sleep(0.05)
    aborted_at = time.perf_counter()
    relay.abort("test abort")
    waiter.join(timeout=1.0)
    check(isinstance(caught.get("error"), DestGuardRelayAborted) and "test abort" in str(caught.get("error")),
          f"abort should raise DestGuardRelayAborted with the reason: {caught}")
    check(caught.get("at", 1e9) - aborted_at < 0.05, "abort should wake the sleeper at once")
    check(not waiter.is_alive(), "the aborted waiter is still alive")
    try:
        relay.wait_published(0)
        check(False, "a wait on an aborted relay without the frame should raise")
    except DestGuardRelayAborted:
        check(True, "")

    # an abort flag without a notify is seen within one bounded round
    relay = DestGuardRelay("round")
    caught = {}
    waiter = threading.Thread(target=wait_aborted)
    waiter.start()
    time.sleep(0.05)
    flagged_at = time.perf_counter()
    with relay.condition:
        relay.aborted = True
        relay.abort_reason = "flag only"
    waiter.join(timeout=2.0)
    elapsed = caught.get("at", 1e9) - flagged_at
    check(isinstance(caught.get("error"), DestGuardRelayAborted) and elapsed < 3 * transport_v7._RELAY_ROUND_TIMEOUT_S,
          f"an abort flag without notify should be seen within a round ({elapsed:.3f} s)")

    # a frame already published passes an aborted relay
    relay = DestGuardRelay("published")
    relay.publish(2)
    relay.abort("after the publish")
    check(relay.wait_published(2) == (True, False), "a published frame should pass an aborted relay")

    # release wakes a sleeper at once; it returns not-published (the consumer then waits the semaphore itself)
    relay = DestGuardRelay("release")
    returned = {}

    def wait_released():
        returned["result"] = relay.wait_published(0)
        returned["at"] = time.perf_counter()

    waiter = threading.Thread(target=wait_released)
    waiter.start()
    time.sleep(0.05)
    released_at = time.perf_counter()
    relay.release("test release")
    waiter.join(timeout=1.0)
    check(returned.get("result") == (False, True) and returned.get("at", 1e9) - released_at < 0.05,
          f"release should wake the sleeper at once, not published: {returned}")
    check(relay.wait_published(0) == (False, False) and relay.release_reason == "test release",
          "a wait on a released relay without the frame should return not-published at once")
    relay.publish(1)
    check(relay.wait_published(1) == (True, False), "a published frame should pass a released relay")

    # a release flag without a notify is seen within one bounded round
    relay = DestGuardRelay("release round")
    returned = {}
    waiter = threading.Thread(target=wait_released)
    waiter.start()
    time.sleep(0.05)
    flagged_at = time.perf_counter()
    with relay.condition:
        relay.released = True
        relay.release_reason = "flag only"
    waiter.join(timeout=2.0)
    elapsed = returned.get("at", 1e9) - flagged_at
    check(returned.get("result") == (False, True) and elapsed < 3 * transport_v7._RELAY_ROUND_TIMEOUT_S,
          f"a release flag without notify should be seen within a round ({elapsed:.3f} s)")


# ----------------------------------------------------------------------------- 2 switches


def test_switches() -> None:
    print("[2] switches: parsing, defaults, registry")
    for text, expected in (("relay", "relay"), ("wait", "wait"), (" Relay ", "relay"), ("WAIT", "wait")):
        try:
            check(transport_v7._parse_dest_guard(text) == expected, f"V7_DEST_GUARD={text!r}")
        except ValueError as error:
            check(False, f"V7_DEST_GUARD={text!r} refused: {error}")
    for text in ("", "relays", "0", "1", "on", "blocking", "zero_wait"):
        try:
            transport_v7._parse_dest_guard(text)
            check(False, f"V7_DEST_GUARD={text!r} accepted")
        except ValueError:
            check(True, "")
    for text, expected in (("zero_wait", "zero_wait"), ("counter", "counter"), ("none", "none"),
                           (" NONE ", "none"), ("Counter", "counter")):
        try:
            check(transport_v7._parse_dest_guard_precheck(text) == expected, f"V7_DEST_GUARD_PRECHECK={text!r}")
        except ValueError as error:
            check(False, f"V7_DEST_GUARD_PRECHECK={text!r} refused: {error}")
    for text in ("", "zero", "zerowait", "zero-wait", "0", "off", "relay", "wait"):
        try:
            transport_v7._parse_dest_guard_precheck(text)
            check(False, f"V7_DEST_GUARD_PRECHECK={text!r} accepted")
        except ValueError:
            check(True, "")
    source = pathlib.Path(transport_v7.__file__).read_text(encoding="utf-8")
    for pattern in (r'_parse_dest_guard\(os\.environ\.get\("V7_DEST_GUARD", "relay"\)\)',
                    r'_parse_dest_guard_precheck\(os\.environ\.get\("V7_DEST_GUARD_PRECHECK", "counter"\)\)'):
        check(re.search(pattern, source) is not None, f"transport_v7.py: default {pattern} not found")
    check(transport_v7.configured_dest_guard_switches() == {"V7_DEST_GUARD": transport_v7._DEST_GUARD,
                                                            "V7_DEST_GUARD_PRECHECK": transport_v7._DEST_GUARD_PRECHECK},
          f"configured_dest_guard_switches {transport_v7.configured_dest_guard_switches()}")
    check((transport_v7._DEST_GUARD, transport_v7._DEST_GUARD_PRECHECK) == ("relay", "counter"),
          "with neither variable set the process should run relay / counter")
    with dest_guard_switches("wait", "counter"):
        check(transport_v7.configured_dest_guard_switches() == {"V7_DEST_GUARD": "wait",
                                                                "V7_DEST_GUARD_PRECHECK": "n/a"},
              f"under wait the pre-check should read n/a: {transport_v7.configured_dest_guard_switches()}")
    try:
        import experiment.v7.utils.simulator_v7 as simulator_v7
    except Exception as error:                       # noqa: BLE001 - the vulkan package may be missing
        print(f"  simulator_v7 not importable ({error!r}): registry check SKIPPED")
        return
    registry = simulator_v7.configured_v7_switches()
    check(registry.get("V7_DEST_GUARD") == transport_v7._DEST_GUARD
          and registry.get("V7_DEST_GUARD_PRECHECK") == transport_v7._DEST_GUARD_PRECHECK,
          f"configured_v7_switches() does not report the dest guard switches: {registry}")


# ----------------------------------------------------------------------------- 3 wiring


def test_wiring() -> None:
    print("[3] wiring: one relay per (sim, direction), publisher / consumer, wait mode, missing publisher")
    for slab_count in (2, 3, 4):
        harness = Harness(seed=slab_count)
        with dest_guard_switches("relay", "counter"), contextlib.redirect_stdout(io.StringIO()):
            sims, workers = build_chain(slab_count, harness)
            relays = connect_dest_guard_relays(workers)
        check(len(relays) == 2 * (slab_count - 1), f"K={slab_count}: {len(relays)} relays")
        for worker in workers:
            publish = relays.get((id(worker.source), worker.source_direction))
            consume = relays.get((id(worker.dest), worker.dest_direction))
            check(worker._publish_relay is publish and publish is not None,
                  f"K={slab_count} {worker.label}: publish relay")
            check(worker._consume_relay is consume and consume is not None,
                  f"K={slab_count} {worker.label}: consume relay")
            check(publish.publisher_label == worker.label and consume.consumer_label == worker.label,
                  f"K={slab_count} {worker.label}: relay labels {publish.label} / {consume.label}")
            check(publish is not consume, f"K={slab_count} {worker.label}: publishes to the relay it reads")
            check(consume.published_frame == -1, f"K={slab_count} {worker.label}: relay starts at "
                                                 f"{consume.published_frame}")
        # the reverse worker of a link reads the relay this worker publishes to
        for index in range(0, len(workers), 2):
            check(workers[index]._publish_relay is workers[index + 1]._consume_relay
                  and workers[index + 1]._publish_relay is workers[index]._consume_relay,
                  f"K={slab_count} link {index // 2}: the two workers do not share their relays")

    output = io.StringIO()
    with dest_guard_switches("relay", "counter"), contextlib.redirect_stdout(output):
        relays = connect_dest_guard_relays([])
    check(relays == {} and output.getvalue() == "", f"K = 1 (no workers): {relays}, {output.getvalue()!r}")

    harness = Harness(seed=9)
    with dest_guard_switches("wait", "counter"):
        sims, workers = build_chain(3, harness)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            relays = connect_dest_guard_relays(workers)
    check(relays == {} and all(worker._publish_relay is None and worker._consume_relay is None for worker in workers),
          "V7_DEST_GUARD=wait should attach no relay")
    check("V7_DEST_GUARD=wait" in output.getvalue(), f"wait mode should say so: {output.getvalue()!r}")

    # a consumer without a publisher (only the rightward worker of a link) keeps rc1's wait, says so once
    harness = Harness(seed=10)
    with dest_guard_switches("relay", "counter"):
        sims, workers = build_chain(2, harness)
        with contextlib.redirect_stdout(io.StringIO()):
            relays = connect_dest_guard_relays(workers[:1])
        lone = workers[0]
        check(relays == {} and lone._consume_relay is None and lone._publish_relay is None,
              "a worker without its reverse worker should get no relay")
        semaphore, value = sims[1].sync.dest_guard_op("leading", 0)
        semaphore.signal(value, "test")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            lone._complete_dest_guard(0, semaphore, value)
            lone._complete_dest_guard(0, semaphore, value)
        check(output.getvalue().count("no relay") == 1, f"the missing relay should be reported once: "
                                                        f"{output.getvalue()!r}")
        check(lone.dest_guard_counts["blocking"] == 2, f"blocking fallback counts {lone.dest_guard_counts}")
        check(any(record[0] == threading.current_thread().name and record[1] == semaphore.name
                  for record in harness.wait_records), "the missing-relay guard should use a timed (blocking) wait")

    # two senders from one (sim, direction) raise
    harness = Harness(seed=11)
    with dest_guard_switches("relay", "counter"):
        sims, workers = build_chain(2, harness)
        duplicate = GhostMigrationWorker(source_sim=sims[0], dest_sim=sims[1], source_direction="trailing",
                                         dest_direction="leading", label="duplicate")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                connect_dest_guard_relays(workers + [duplicate])
            check(False, "two workers sending from one (sim, direction) should raise")
        except ValueError:
            check(True, "")

    # relays only before start()
    harness = Harness(seed=12)
    with dest_guard_switches("relay", "counter"):
        sims, workers = build_chain(2, harness)
        workers[0].start()
        try:
            workers[0].attach_consume_relay(DestGuardRelay("late"))
            check(False, "attaching a relay after start() should raise")
        except RuntimeError:
            check(True, "")
        with contextlib.redirect_stdout(io.StringIO()):
            workers[0].stop()
        check(not workers[0].thread.is_alive(), "an idle worker should stop")


# ----------------------------------------------------------------------------- 4 protocol


def check_protocol_run(result: dict, label: str, frames: int, mode: str, precheck: str) -> None:
    harness, workers = result["harness"], result["workers"]
    check(result["error"] is None, f"{label}: run failed: {result['error']!r}")
    check(result["frames_done"] == frames, f"{label}: {result['frames_done']} of {frames} frames done")
    check(not harness.violations, f"{label}: protocol violations {harness.violations[:5]}")
    check(not result["alive_threads"], f"{label}: threads left alive {result['alive_threads']}")
    for worker in workers:
        counts = worker.dest_guard_counts
        check(worker.last_error is None, f"{label} {worker.label}: died {worker.last_error!r}")
        check(worker.stamp_error_count == 0, f"{label} {worker.label}: {worker.stamp_error_count} stale stamps")
        check(worker.last_completed_frame == frames - 1, f"{label} {worker.label}: last frame "
                                                         f"{worker.last_completed_frame}")
        check(sum(counts.values()) == frames, f"{label} {worker.label}: guard counts {counts} != {frames}")
        check(counts["fallback"] == 0 and worker.dest_guard_released == 0,
              f"{label} {worker.label}: {counts['fallback']} fallbacks, {worker.dest_guard_released} released")
        if mode == "wait":
            check(counts["blocking"] == frames, f"{label} {worker.label}: rc1 mode counts {counts}")
        else:
            check(counts["blocking"] == 0, f"{label} {worker.label}: blocking guards {counts}")
            if precheck == "none":
                check(counts["precheck"] == 0, f"{label} {worker.label}: none cannot complete on arrival")
        record = worker.dest_guard_record()
        check(all(record[key] == counts[key] for key in counts) and record["released"] == 0
              and record["relay_attached"] == (mode == "relay"),
              f"{label} {worker.label}: dest_guard_record {record}")
        line = re.search(rf"\[worker {worker.label}\] dest guard: precheck (\d+), relay (\d+), fallback (\d+), "
                         rf"blocking (\d+) \(relay slept (\d+)", result["output"])
        check(line is not None and tuple(int(group) for group in line.groups()[:4])
              == (counts["precheck"], counts["relay"], counts["fallback"], counts["blocking"]),
              f"{label} {worker.label}: stop() summary line missing or wrong")
    if mode == "relay":
        for relay in result["relays"].values():
            check(relay.published_frame == frames - 1, f"{label} relay {relay.label}: published "
                                                       f"{relay.published_frame}")
        if precheck == "counter":
            check(sum(worker.dest_guard_counts["precheck"] for worker in workers) > 0,
                  f"{label}: the counter pre-check never completed a guard on arrival")
    # zero-timeout calls of the dest guards on a value still pending (the zero_wait co-call)
    pending_zero_waits = sum(1 for thread_name, semaphore_name, _, reached in harness.zero_wait_records
                             if thread_name.startswith("ghost-") and ".transport_" in semaphore_name and not reached)
    if mode == "relay" and precheck in ("counter", "none"):
        check(pending_zero_waits == 0, f"{label}: {pending_zero_waits} zero-timeout calls on a pending value")
    sleepers = sleeping_waits(result)
    transport_shared = {key: threads for key, threads in sleepers.items()
                        if ".transport_" in key[0] and len(threads) > 1}
    guard_sleeps = 0
    upload_guard_sleeps = 0
    for worker in workers:
        guard_name = worker.dest.sync.transport[worker.dest_direction].name
        main_name = worker.dest.sync.main.name
        for (semaphore_name, _), threads in sleepers.items():
            if f"ghost-{worker.label}" in threads:
                guard_sleeps += semaphore_name == guard_name
                upload_guard_sleeps += semaphore_name == main_name
    check(upload_guard_sleeps == 0, f"{label}: the upload guard blocked {upload_guard_sleeps} times")
    if mode == "relay":
        check(guard_sleeps == 0, f"{label}: {guard_sleeps} dest guards slept in a semaphore wait")
        check(not transport_shared, f"{label}: {len(transport_shared)} transport values with two sleepers, e.g. "
                                    f"{next(iter(transport_shared.items()), None)}")
    else:
        # rc1: the dest guard and the reverse source wait sleep on one value (what E7 batch 1 saw)
        check(guard_sleeps > 0 and transport_shared, f"{label}: rc1 mode should show sleeping dest guards "
                                                     f"({guard_sleeps}) and shared values ({len(transport_shared)})")
    summary = ", ".join(f"{worker.label} {worker.dest_guard_counts['precheck']}/{worker.dest_guard_counts['relay']}/"
                        f"{worker.dest_guard_counts['fallback']}/{worker.dest_guard_counts['blocking']} slept "
                        f"{worker.dest_guard_relay_slept}" for worker in workers)
    print(f"  {label}: {result['frames_done']} frames in {result['seconds']:.1f} s; guard precheck/relay/fallback/"
          f"blocking: {summary}; dest guards asleep {guard_sleeps}, shared transport values "
          f"{len(transport_shared)}, zero-timeout calls on a pending value {pending_zero_waits}")


def test_protocol(frames: int) -> None:
    print(f"[4] protocol: K = 2 x {frames} frames per pre-check and rc1, K = 3 interior")
    for seed, (mode, precheck) in enumerate((("relay", "counter"), ("relay", "zero_wait"), ("relay", "none"),
                                             ("wait", "counter"))):
        label = f"K=2 {mode}" + (f"/{precheck}" if mode == "relay" else " (rc1)")
        result = run_protocol(2, frames, mode, precheck, seed=100 + seed)
        check_protocol_run(result, label, frames, mode, precheck)
    interior_frames = max(300, frames * 3 // 10)
    for precheck in ("counter", "zero_wait", "none"):
        result = run_protocol(3, interior_frames, "relay", precheck, seed=31)
        check_protocol_run(result, f"K=3 relay/{precheck}", interior_frames, "relay", precheck)


# ----------------------------------------------------------------------------- 5 negative controls


def test_negative_controls() -> None:
    print("[5] negative controls: relay starting at 0; no dest guard at all")

    def start_relays_at_zero(sims, workers, relays):
        for relay in relays.values():
            relay.published_frame = 0
        sims[1].delays[("readback", 0)] = 0.2        # sim 1's readback(0) lands well after s0_to_s1's guard

    result = run_protocol(2, 50, "relay", "none", seed=5, prepare=start_relays_at_zero)
    counts = result["workers"][0].dest_guard_counts
    check(result["error"] is None and not result["harness"].violations,
          f"relay at 0: the formal wait should keep the run correct: {result['error']!r} "
          f"{result['harness'].violations[:3]}")
    check(counts["fallback"] >= 1, f"relay at 0: frame 0 should pass the relay early and fall back ({counts})")
    check("missed" in result["output"], "relay at 0: the fallback should be reported")

    def counter_reads_reached(sims, workers, relays):
        sims[1].semaphore_value = lambda semaphore: 1 << 40      # every counter read says "reached"
        sims[1].delays[("readback", 0)] = 0.2

    result = run_protocol(2, 50, "relay", "counter", seed=4, prepare=counter_reads_reached)
    counts = result["workers"][0].dest_guard_counts
    check(result["error"] is None and not result["harness"].violations,
          f"early counter: the formal wait should keep the run correct: {result['error']!r} "
          f"{result['harness'].violations[:3]}")
    check(counts["fallback"] >= 1, f"early counter: frame 0 should fail the formal wait and fall back ({counts})")
    check("counter reached but the zero-timeout wait missed" in result["output"],
          "early counter: the fallback should be reported")

    def remove_guard(sims, workers, relays):
        for worker in workers:
            worker._complete_dest_guard = lambda frame_n, semaphore, value: None
        sims[1].delays[("readback", 0)] = 0.2

    result = run_protocol(2, 50, "relay", "counter", seed=6, prepare=remove_guard)
    errors = [worker.last_error for worker in result["workers"] if worker.last_error is not None]
    check(reports_dead_worker(result["error"]) and any(isinstance(error, AssertionError) for error in errors),
          f"no dest guard: the worker's assert should fire before its host signal ({result['error']!r}, {errors})")
    check(not result["harness"].violations, f"no dest guard: the assert should stop the host signal before it "
                                            f"goes backwards: {result['harness'].violations[:3]}")
    check(not result["alive_threads"], f"no dest guard: threads left alive {result['alive_threads']}")


# ----------------------------------------------------------------------------- 6 abort


def test_abort() -> None:
    print("[6] abort and stop: a dying worker wakes its partner; stop() drains like rc1; release after the join "
          "timeout")
    target_label, target_frame = "s1_to_s0", 37
    died = {}

    def inject(thread_name, semaphore, value):
        if thread_name == f"ghost-{target_label}" and semaphore.name == "s1.transport_leading" \
                and value == 2 * target_frame + 1:
            died["at"] = time.perf_counter()
            raise RuntimeError(f"injected death at frame {target_frame}")

    result = run_protocol(2, 200, "relay", "none", seed=7, inject=inject)
    workers = {worker.label: worker for worker in result["workers"]}
    partner = workers["s0_to_s1"]
    check(reports_dead_worker(result["error"]), f"the loop should report the dead worker: {result['error']!r}")
    check(isinstance(workers[target_label].last_error, RuntimeError)
          and "injected" in str(workers[target_label].last_error),
          f"target error {workers[target_label].last_error!r}")
    check(isinstance(partner.last_error, DestGuardRelayAborted) and target_label in str(partner.last_error),
          f"the partner should raise DestGuardRelayAborted naming {target_label}: {partner.last_error!r}")
    check(partner.last_completed_frame == target_frame - 1,
          f"the partner should stop at frame {target_frame}: last done {partner.last_completed_frame}")
    check(not result["alive_threads"], f"threads left alive after the abort: {result['alive_threads']}")
    check("DIED" in result["stderr"], "the deaths should be printed")

    # stop() with a frame in flight drains it like rc1: sim 1's readback(0) lands 0.3 s after the stop() calls
    harness = Harness(seed=8)
    with dest_guard_switches("relay", "counter"):
        sims, workers = build_chain(2, harness)
        with contextlib.redirect_stdout(io.StringIO()):
            connect_dest_guard_relays(workers)
    sims[1].delays[("readback", 0)] = 0.3
    for sim in sims:
        sim.start_queues()
    for worker in workers:
        worker.start()
    for sim in sims:
        sim.submit(0)
    for worker in workers:
        worker.notify(0)
    stop_started = time.perf_counter()
    output, error_output = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
        for worker in workers:                       # ChainOrchestratorV7.destroy order
            worker.stop()
    elapsed = time.perf_counter() - stop_started
    for worker in workers:
        check(worker.last_error is None and worker.last_completed_frame == 0 and not worker.thread.is_alive(),
              f"drain: {worker.label} should finish frame 0 and end (error {worker.last_error!r}, last frame "
              f"{worker.last_completed_frame}, alive {worker.thread.is_alive()})")
        check(worker.dest_guard_counts["fallback"] == 0 and worker.dest_guard_released == 0,
              f"drain: {worker.label} should complete through the relay or on arrival: {worker.dest_guard_counts}")
    check(elapsed < transport_v7._STOP_JOIN_TIMEOUT_S, f"drain: stop() took {elapsed:.2f} s")
    check("DIED" not in error_output.getvalue(), f"drain: no death expected: {error_output.getvalue()[:300]!r}")
    harness.shutdown.set()
    for sim in sims:
        sim.stop_queues()
        for thread in sim.threads:
            thread.join(timeout=5.0)

    # stop() of a worker held by a relay that is never published: released after the join timeout, the worker takes
    # rc1's blocking wait (fallback, released) and finishes the frame once the value arrives
    saved_timeouts = transport_v7._STOP_JOIN_TIMEOUT_S, transport_v7._STOP_RELEASE_JOIN_TIMEOUT_S
    transport_v7._STOP_JOIN_TIMEOUT_S, transport_v7._STOP_RELEASE_JOIN_TIMEOUT_S = 0.3, 0.3
    try:
        harness = Harness(seed=9)
        with dest_guard_switches("relay", "counter"):
            sims, workers = build_chain(2, harness)
            with contextlib.redirect_stdout(io.StringIO()):
                connect_dest_guard_relays(workers)
        sleeper = workers[0]                         # s0_to_s1: its source (s0 trailing) is ready, s1 never reads back
        sims[0].sync.transport["trailing"].signal(1, "test")
        sleeper.start()
        sleeper.notify(0)
        deadline = time.perf_counter() + 2.0
        while sleeper.last_activity[0] != "wait_dest_relay" and time.perf_counter() < deadline:
            time.sleep(0.005)
        check(sleeper.last_activity[0] == "wait_dest_relay", f"the worker should sleep on the relay: "
                                                             f"{sleeper.last_activity[:2]}")
        stop_started = time.perf_counter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            sleeper.stop()
            workers[1].stop()                        # never started: a no-op
        elapsed = time.perf_counter() - stop_started
        check(0.3 <= elapsed < 1.5, f"release: stop() should return after the two join timeouts ({elapsed:.3f} s)")
        check(sleeper.last_error is None and sleeper.last_activity[0] == "wait_dest_timeline",
              f"release: the worker should be in rc1's blocking wait, not dead: {sleeper.last_activity[:2]}, "
              f"{sleeper.last_error!r}")
        check(sleeper.dest_guard_counts["fallback"] == 1 and sleeper.dest_guard_released == 1,
              f"release: counts {sleeper.dest_guard_counts}, released {sleeper.dest_guard_released}")
        check("released 1;" in output.getvalue() and "relay released" in output.getvalue(),
              f"release: stop() and fallback lines: {output.getvalue()!r}")
        semaphore, value = sims[1].sync.dest_guard_op("leading", 0)
        semaphore.signal(value, "test")              # the receiver's readback(0) arrives: rc1's wait returns
        deadline = time.perf_counter() + 2.0
        while sleeper.last_completed_frame < 0 and sleeper.last_error is None and time.perf_counter() < deadline:
            time.sleep(0.005)
        check(sleeper.last_completed_frame == 0 and sleeper.last_error is None,
              f"release: the worker should finish frame 0 after the value arrives: last frame "
              f"{sleeper.last_completed_frame}, error {sleeper.last_error!r}")
        sleeper.thread.join(timeout=2.0)
        check(not sleeper.thread.is_alive(), "release: the worker should end on its queued sentinel")
    finally:
        transport_v7._STOP_JOIN_TIMEOUT_S, transport_v7._STOP_RELEASE_JOIN_TIMEOUT_S = saved_timeouts
        harness.shutdown.set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=10000, help="frames per link of the K = 2 protocol runs")
    arguments = parser.parse_args()
    sys.setswitchinterval(0.0002)                    # the chain bench's default GIL switch interval
    started = time.perf_counter()
    test_relay_unit()
    test_switches()
    test_wiring()
    test_protocol(arguments.frames)
    test_negative_controls()
    test_abort()
    print(f"\n{_passed} checks passed, {len(_failed)} failed ({time.perf_counter() - started:.0f} s)")
    if _failed:
        for message in _failed[:30]:
            print(f"  FAILED: {message}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
