"""
link_inventory.py - transport bytes per link per frame, segment by segment and
field by field, for one case and one seam configuration (V6_KEEP_DEPARTED /
V6_GHOST_LAYERS from the environment).

Builds the K=2 chain with the production transport switches (count-aware
worker, ghost pool factor 0.25 in 2-D / 1.0 in 3-D, split transfer queues),
runs a warmup, then samples the sender stagings: after a drained frame the
staging of each direction holds that frame's allocation counters, so the live
prefix of every per-particle segment, min(size, count x stride), is exactly
what the count-aware worker copies; voxel lists, count words and the frame
stamp are copied whole. DMA = the whole segment (readback = upload). The
worker's own byte counter is recorded as a cross-check.

Usage (GPU, both 5090s):
  V6_KEEP_DEPARTED=1 V6_GHOST_LAYERS=2 .venv/Scripts/python.exe -m \\
      experiment.seam_audit.link_inventory --case cases/lid_driven_cavity_2d_gen/case.yaml \\
      --dimension 2 --warmup 1000 --frames 2000 --sample-every 20 --out inventory.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

PRODUCTION_SWITCHES = {"V6_WORKER_COUNT_AWARE": "1", "V6_SPLIT_TRANSFER_QUEUES": "1",
                       "V6_CASCADE_FORCE": "1", "V6_BAND_VOXEL_DISPATCH": "1"}
POOL_FACTOR_BY_DIMENSION = {2: "0.25", 3: "1.0"}


def classify_segments(segments: list, ghost_layers: int) -> list[dict]:
    """Name every segment of one direction's staging layout (simulator
    _compute_transport_segments / _compute_two_layer_transport_segments)."""
    count_offsets = []
    for segment in segments:
        if segment.count_staging_offset is not None and segment.count_staging_offset not in count_offsets:
            count_offsets.append(segment.count_staging_offset)
    if ghost_layers == 1:
        region_of_count = {offset: "replica+migrant" for offset in count_offsets}
    else:
        region_of_count = dict(zip(count_offsets, ("G1 replica", "G2 replica", "migrant")))
    described = []
    status_index = 0
    status_total = sum(1 for segment in segments if segment.buffer_name == "global_status")
    for segment in segments:
        entry = {"buffer": segment.buffer_name, "size": int(segment.size),
                 "stride": int(segment.stride or 0),
                 "staging_offset": int(segment.staging_offset),
                 "count_staging_offset": segment.count_staging_offset}
        if segment.count_staging_offset is not None:
            entry["region"] = region_of_count[segment.count_staging_offset]
            entry["slots"] = int(segment.size // segment.stride)
        elif segment.buffer_name == "inside_particle_count":
            entry["region"] = "slot counts"
        elif segment.buffer_name == "inside_particle_index":
            entry["region"] = "slot index"
        else:
            status_index += 1
            entry["region"] = "frame stamp" if status_index == status_total else "count word"
        described.append(entry)
    return described


def sample_live_bytes(sim, direction: str, segments: list) -> tuple[list[int], list[int]]:
    view = sim.sender_staging_view(direction)
    live, counts = [], []
    for segment in segments:
        if segment.count_staging_offset is not None:
            count = struct.unpack_from("<I", view, segment.count_staging_offset)[0]
            counts.append(int(count))
            live.append(int(min(segment.size, count * segment.stride)))
        else:
            counts.append(-1)
            live.append(int(segment.size))
    return live, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    for key, value in PRODUCTION_SWITCHES.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("V6_GHOST_POOL_FACTOR", POOL_FACTOR_BY_DIMENSION[arguments.dimension])
    # then the pre-E6b defaults for every other switch (caller > production > legacy)
    from experiment.v6.utils.partition_v6 import LEGACY_DEFAULTS
    for key, value in LEGACY_DEFAULTS.items():
        os.environ.setdefault(key, value)

    from experiment.seam_audit.dump_state import run_frames
    from experiment.seam_audit.solver_adapter import load_solver
    solver = load_solver("v6")
    global_case = solver.load_case(arguments.case)
    chain = solver.compute_chain_partition(global_case, [1.0, 1.0], arguments.pool_safety)
    device_map = [int(item) for item in arguments.device_map.split(",")]
    contexts, sims = [], []
    for slab_index in range(2):
        contexts.append(solver.Context.create(device_index=device_map[slab_index],
                                              enable_validation=False,
                                              application_name=f"link_inventory_s{slab_index}"))
        sims.append(solver.Simulator(contexts[-1], chain.slabs[slab_index],
                                     sync_scheme="per-direction"))
    defrag_cadence = int(global_case.numerics.defrag_cadence) or 10 ** 12
    orchestrator = solver.Orchestrator(sims, defrag_cadence=defrag_cadence)
    orchestrator.bootstrap_all()
    ghost_layers = sims[0].ghost_layers()

    defrag_log: list = []
    start = time.perf_counter()
    run_frames(orchestrator, 0, arguments.warmup, 2, defrag_cadence, defrag_log)
    current = arguments.warmup
    layouts = {}
    sums = {}
    count_sums = {}
    count_maxima = {}
    departed_samples = [[] for _ in sims]
    samples = 0
    for sim_index, sim in enumerate(sims):
        for direction, segments in sim._transport_segments.items():
            key = f"s{sim_index}_{direction}"
            layouts[key] = classify_segments(segments, ghost_layers)
            sums[key] = np.zeros(len(segments))
            count_sums[key] = np.zeros(len(segments))
            count_maxima[key] = np.zeros(len(segments))
    worker_bytes_before = {worker.label: (worker.total_copy_bytes, worker.copy_frame_count)
                           for worker in orchestrator.workers}
    while current < arguments.warmup + arguments.frames:
        stop = min(current + arguments.sample_every, arguments.warmup + arguments.frames)
        run_frames(orchestrator, current, stop, 2, defrag_cadence, defrag_log)
        current = stop
        for sim_index, sim in enumerate(sims):
            for direction, segments in sim._transport_segments.items():
                key = f"s{sim_index}_{direction}"
                live, counts = sample_live_bytes(sim, direction, segments)
                sums[key] += live
                count_sums[key] += counts
                count_maxima[key] = np.maximum(count_maxima[key], counts)
            # departed_count = this frame's outgoing migrants of this sim (one
            # peer side in a K=2 chain); 0 when V6_KEEP_DEPARTED is off.
            departed_samples[sim_index].append(int(sim.readback_global_status()["departed_count"]))
        samples += 1
    worker_bytes = {}
    for worker in orchestrator.workers:
        before_bytes, before_frames = worker_bytes_before[worker.label]
        frames = worker.copy_frame_count - before_frames
        worker_bytes[worker.label] = ((worker.total_copy_bytes - before_bytes) / frames
                                      if frames else None)
    statuses = [dict(sim.readback_global_status()) for sim in sims]
    healths = [dict(sim.readback_pool_health()) for sim in sims]
    for key, layout in layouts.items():
        for entry, live_sum, count_sum, count_maximum in zip(layout, sums[key], count_sums[key],
                                                             count_maxima[key]):
            entry["host_bytes_mean"] = float(live_sum / samples)
            counted = entry["count_staging_offset"] is not None
            entry["count_mean"] = float(count_sum / samples) if counted else None
            entry["count_max"] = int(count_maximum) if counted else None
    capacities = sims[0].case.capacities
    document = {
        "case": arguments.case, "dimension": arguments.dimension,
        "switches": {key: os.environ.get(key, "") for key in sorted(os.environ) if key.startswith("V6_")},
        "ghost_layers": ghost_layers,
        "keep_departed": int(capacities.departed_pool_size > 0),
        "cuts": [int(cut) for cut in chain.cuts],
        "face_voxels": int(global_case.grid.grid_dimension_y * global_case.grid.grid_dimension_z),
        "max_particles_per_voxel": int(capacities.max_particles_per_voxel),
        "max_incoming_per_voxel": int(capacities.max_incoming_per_voxel),
        "ghost_pool_per_direction": int(capacities.trailing_ghost_pool_size
                                        or capacities.leading_ghost_pool_size),
        "replica_region_size": int(capacities.replica_region_size),
        "departed_pool_size": int(capacities.departed_pool_size),
        "warmup": arguments.warmup, "frames": arguments.frames,
        "sample_every": arguments.sample_every, "samples": samples,
        "layouts": layouts, "worker_bytes_per_frame": worker_bytes,
        "departed_per_frame": [{"mean": float(np.mean(values)) if values else None,
                                "max": int(max(values)) if values else None}
                               for values in departed_samples],
        "staging_bytes": {f"s{index}": sim.transport_staging_bytes() for index, sim in enumerate(sims)},
        "global_status": statuses, "pool_health": healths,
        "wall_time_s": time.perf_counter() - start,
    }
    pathlib.Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(arguments.out).write_text(json.dumps(document, indent=1), encoding="utf-8")
    overflow = {key: value for status in statuses for key, value in status.items()
                if key.startswith("overflow_") and value}
    print(f"[link_inventory] {arguments.case} layers={ghost_layers} keep={document['keep_departed']}: "
          f"{samples} samples, worker bytes/frame {worker_bytes}, overflow {overflow or 'none'}",
          flush=True)
    orchestrator.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()
    return 0 if not overflow else 3


if __name__ == "__main__":
    sys.exit(main())
