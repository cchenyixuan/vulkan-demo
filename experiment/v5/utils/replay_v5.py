"""
replay_v5.py — discrete-event replay of the V5 chain frame protocol (M5b).

Reconstructs the steady-state frame period of an N-slab chain from
per-segment durations, by replaying the EXACT dependency graph of the
per-direction sync scheme (docs/sph_v5_design.md §3.1):

    A(i,n)   waits frame_done(i,n-1)            [compute engine]
    RB(i,d,n) waits A(i,n) + sched_gap          [transfer engine, FIFO]
    B(i,n)   queue-ordered after A (no sem)     [compute engine]
    worker(src->dst,d): waits RB(src) AND RB(dst) [guard], then memcpy [host]
    UP(dst,d,n) waits worker_done                [transfer engine, FIFO]
    C(i,n)   waits all UP(i,*,n) + sem_latency   [compute engine]

Virtual-GPU emulation: sims mapped to the same physical device share that
device's compute/transfer ENGINES (execution serializes; semaphore waits do
not occupy an engine — WDDM runs other contexts while one waits). Engines
are modeled as ready-time-FCFS resources.

Host-side costs the GPU graph cannot see (submit latency, worker thread
wakeup, notify) are lumped into two disclosed parameters:
    host_overhead_us  — added to each frame's critical path (fit on K=2)
    worker_wakeup_us  — added before each worker memcpy

Usage: build PerSim/Link tables (see _run_v5_replay.py), call
``replay(sims, links, device_map, params, frames=...)`` → ReplayResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerSimDurations:
    """One sim's per-frame segment durations (µs)."""
    phase_a: float
    phase_b: float
    phase_c: float
    readback_dma: dict          # direction -> µs
    upload_dma: dict            # direction -> µs


@dataclass
class LinkDurations:
    """One directed ghost pathway."""
    source_index: int
    destination_index: int
    source_direction: str       # side of the SOURCE sim that sends
    destination_direction: str  # side of the DEST sim that receives
    memcpy: float               # µs (0 for P2P what-if)


@dataclass
class ReplayParams:
    sched_gap_us: float = 44.8       # measured, size-independent (M5a)
    sem_latency_us: float = 35.8     # upload landed -> phase C start (M5a)
    host_overhead_us: float = 0.0    # fit on K=2 calibration
    worker_wakeup_us: float = 0.0    # folded into host_overhead by default
    memcpy_channels: int = 0         # >0: worker memcpys contend for this
                                     # many host-RAM channels (the 1M/8M
                                     # K-sweep shows period ~linear in K —
                                     # memcpys serialize on host memory
                                     # bandwidth, NOT kernel cache thrash);
                                     # 0 = unlimited (legacy parallel)
    context_switch_us: float = 0.0   # engine cost when consecutive batches
                                     # belong to DIFFERENT VkDevices on one
                                     # physical device (WDDM context switch;
                                     # fit on ONE oversubscribed point,
                                     # validated blind on the rest). Zero
                                     # for 1-sim/GPU cluster predictions.


@dataclass
class ReplayResult:
    steady_period_us: float
    steady_fps: float
    per_sim_gap_us: list        # per sim: median b_to_c-equivalent gap
    critical_sim: int           # sim whose chain binds the period


class _Engine:
    """Ready-time-FCFS execution resource (one compute or transfer engine
    per PHYSICAL device). A batch runs for `duration` once both the engine
    is free and the batch's dependencies are met. Switching between
    OWNERS (different VkDevices timesliced on one physical device) costs
    `switch_cost` engine time — the WDDM context-switch model."""

    def __init__(self, switch_cost: float = 0.0):
        self.free_at = 0.0
        self.switch_cost = switch_cost
        self.last_owner = None

    def run(self, ready_time: float, duration: float, owner=None) -> tuple:
        start = max(ready_time, self.free_at)
        if (self.switch_cost > 0.0 and owner is not None
                and self.last_owner is not None and owner != self.last_owner):
            start += self.switch_cost
        self.last_owner = owner
        end = start + duration
        self.free_at = end
        return start, end


def replay(sims: list, links: list, device_map: list,
           params: ReplayParams, frames: int = 200,
           warmup_frames: int = 50) -> ReplayResult:
    """Replay `frames` frames of the chain protocol; return steady stats."""
    sim_count = len(sims)
    physical_devices = sorted(set(device_map))
    compute_engine = {d: _Engine(params.context_switch_us)
                      for d in physical_devices}
    transfer_engine = {d: _Engine(params.context_switch_us)
                       for d in physical_devices}

    links_into = [[] for _ in range(sim_count)]
    links_from = [[] for _ in range(sim_count)]
    for link in links:
        links_into[link.destination_index].append(link)
        links_from[link.source_index].append(link)

    frame_done = [0.0] * sim_count           # done(C[i, n-1])
    periods = []
    gaps = [[] for _ in range(sim_count)]
    last_done = [0.0] * sim_count
    critical_counts = [0] * sim_count

    for frame in range(frames):
        # Phase A on every sim (submission order = sim order, matching the
        # orchestrator's round-robin submit loops).
        a_done = [0.0] * sim_count
        for i, sim in enumerate(sims):
            ready = frame_done[i] + params.host_overhead_us
            _, a_done[i] = compute_engine[device_map[i]].run(
                ready, sim.phase_a, owner=i)

        # Readbacks (per sim, per direction; transfer queue FIFO per device).
        rb_done = {}
        for i, sim in enumerate(sims):
            for direction, duration in sim.readback_dma.items():
                ready = a_done[i] + params.sched_gap_us
                _, done = transfer_engine[device_map[i]].run(
                    ready, duration, owner=i)
                rb_done[(i, direction)] = done

        # Phase B (queue-ordered after A on the same compute engine).
        b_done = [0.0] * sim_count
        for i, sim in enumerate(sims):
            _, b_done[i] = compute_engine[device_map[i]].run(
                a_done[i], sim.phase_b, owner=i)

        # Workers: memcpy after BOTH endpoint readbacks (guard included).
        # With memcpy_channels > 0, memcpys contend for host RAM channels:
        # earliest-ready first, each assigned to the earliest-free channel.
        worker_done = {}
        ready_list = []
        for link in links:
            src_rb = rb_done[(link.source_index, link.source_direction)]
            dst_rb = rb_done[(link.destination_index,
                              link.destination_direction)]
            ready = max(src_rb, dst_rb) + params.worker_wakeup_us
            ready_list.append((ready, link))
        if params.memcpy_channels > 0:
            channel_free = [0.0] * params.memcpy_channels
            for ready, link in sorted(ready_list, key=lambda item: item[0]):
                channel = min(range(len(channel_free)),
                              key=lambda c: channel_free[c])
                start = max(ready, channel_free[channel])
                channel_free[channel] = start + link.memcpy
                worker_done[(link.destination_index,
                             link.destination_direction)] = start + link.memcpy
        else:
            for ready, link in ready_list:
                worker_done[(link.destination_index,
                             link.destination_direction)] = ready + link.memcpy

        # Uploads (dest transfer queue, FIFO behind that device's readbacks).
        upload_done = [0.0] * sim_count
        for i, sim in enumerate(sims):
            for direction, duration in sim.upload_dma.items():
                ready = worker_done[(i, direction)]
                _, done = transfer_engine[device_map[i]].run(
                    ready, duration, owner=i)
                upload_done[i] = max(upload_done[i], done)

        # Phase C: after B (queue order) and all uploads (+ semaphore hop).
        for i, sim in enumerate(sims):
            if sim.upload_dma:
                sem_ready = upload_done[i] + params.sem_latency_us
            else:
                sem_ready = a_done[i]
            gap = max(0.0, sem_ready - b_done[i])
            start, done = compute_engine[device_map[i]].run(
                max(b_done[i], sem_ready), sim.phase_c, owner=i)
            if frame >= warmup_frames:
                gaps[i].append(gap)
            frame_done[i] = done

        if frame >= warmup_frames:
            period = max(frame_done) - max(last_done)
            periods.append(period)
            critical_counts[frame_done.index(max(frame_done))] += 1
        last_done = list(frame_done)

    steady = sorted(periods)[len(periods) // 2]
    return ReplayResult(
        steady_period_us=steady,
        steady_fps=1e6 / steady if steady > 0 else 0.0,
        per_sim_gap_us=[sorted(g)[len(g) // 2] if g else 0.0 for g in gaps],
        critical_sim=critical_counts.index(max(critical_counts)),
    )


def make_chain(per_sim_table: list, memcpy_table: dict) -> tuple:
    """Convenience: build (sims, links) for an N-chain from a per-sim table
    and {(src,dst): memcpy_us}. Endpoint sims get one direction, interior
    sims two, mirroring compute_chain_partition topology."""
    sim_count = len(per_sim_table)
    sims = []
    for index, entry in enumerate(per_sim_table):
        directions_rb, directions_up = {}, {}
        if index > 0:                      # leading side faces sim index-1
            directions_rb["leading"] = entry["readback_dma"]
            directions_up["leading"] = entry["upload_dma"]
        if index < sim_count - 1:          # trailing side faces sim index+1
            directions_rb["trailing"] = entry["readback_dma"]
            directions_up["trailing"] = entry["upload_dma"]
        sims.append(PerSimDurations(
            phase_a=entry["phase_a"], phase_b=entry["phase_b"],
            phase_c=entry["phase_c"],
            readback_dma=directions_rb, upload_dma=directions_up))
    links = []
    for index in range(sim_count - 1):
        links.append(LinkDurations(
            source_index=index, destination_index=index + 1,
            source_direction="trailing", destination_direction="leading",
            memcpy=memcpy_table.get((index, index + 1), 0.0)))
        links.append(LinkDurations(
            source_index=index + 1, destination_index=index,
            source_direction="leading", destination_direction="trailing",
            memcpy=memcpy_table.get((index + 1, index), 0.0)))
    return sims, links
