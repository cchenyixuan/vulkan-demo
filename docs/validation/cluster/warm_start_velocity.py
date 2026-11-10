"""warm_start_velocity.py (2026-10-20): the fluid particles' initial velocities of a finer cavity case interpolated from a
coarser run's final snapshot (linear MLS with the coarser run's support, Shepard where the MLS is ill-conditioned; every
particle of the snapshot - fluid, wall and lid - contributes, so the no-slip walls and the lid are respected), written as
the file cavity_runner --initial-velocity reads: rows = the fine case's fluid rows in the loader's order, velocities (N, 2),
case_sha256 of the fine case.yaml, provenance (source run, snapshot step / time / sha256, support, statistics).

    python docs/validation/cluster/warm_start_velocity.py --source-run <run dir> --case <fine case.yaml> --out <npz>
        [--snapshot <t<step>.npz, default the newest>] [--chunk 200000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--chunk", type=int, default=200000)
    arguments = parser.parse_args()
    from scipy.spatial import cKDTree
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_FLUID
    from experiment.validation import cavity_sampling as sampling

    source = pathlib.Path(arguments.source_run).resolve()
    source_meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    snapshot = pathlib.Path(arguments.snapshot).resolve() if arguments.snapshot else \
        sorted((source / "snapshots").glob("t*.npz"))[-1]
    started = time.perf_counter()
    with np.load(snapshot) as archive:
        particles = sampling.ParticleSet(archive["positions"].astype(np.float64), archive["velocities"].astype(np.float64),
                                         archive["volumes"].astype(np.float64))
    support = float(source_meta["support_radius"])
    step = int(snapshot.stem[1:])
    print(f"[warm_start] source {source.name} snapshot {snapshot.name} (t = {step * float(source_meta['dt']):.4f}), "
          f"{particles.positions.shape[0]:,} particles, support {support}", flush=True)
    case_path = pathlib.Path(arguments.case).resolve()
    case = load_case_v7(case_path)
    fluid_groups = np.array([index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID],
                            dtype=case.initial.material_group.dtype)
    rows = np.flatnonzero(np.isin(case.initial.material_group, fluid_groups))
    points = case.initial.positions[rows, :2].astype(np.float64)
    print(f"[warm_start] target {case_path.parent.name}: {rows.size:,} fluid particles of {case.initial.positions.shape[0]:,}", flush=True)
    tree = cKDTree(particles.positions)
    velocities = np.zeros((rows.size, 2), dtype=np.float64)
    neighbours = np.zeros(rows.size, dtype=np.int64)
    fallback = np.zeros(rows.size, dtype=bool)
    for start in range(0, rows.size, arguments.chunk):
        stop = min(start + arguments.chunk, rows.size)
        result = sampling.interpolate(particles, points[start:stop], support, tree=tree)
        velocities[start:stop] = result["mls"]
        neighbours[start:stop] = result["neighbours"]
        fallback[start:stop] = result["mls_fallback"]
        print(f"[warm_start] {stop:,} / {rows.size:,} points, {time.perf_counter() - started:.0f} s", flush=True)
    missing = ~np.isfinite(velocities).all(axis=1)
    if missing.any():
        sys.exit(f"[warm_start] {int(missing.sum())} points without neighbours")
    speed = np.sqrt((velocities ** 2).sum(axis=1))
    provenance = {"source_run": source.name, "source_case": source_meta.get("case"), "snapshot": snapshot.name, "snapshot_step": step,
                  "snapshot_time": step * float(source_meta["dt"]), "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  "support": support, "method": "linear MLS (Wendland C4, volume weights), Shepard fallback",
                  "mls_fallback_points": int(fallback.sum()), "neighbours_min": int(neighbours.min()),
                  "neighbours_median": float(np.median(neighbours)), "max_speed": float(speed.max()),
                  "source_max_speed": float(np.sqrt((particles.velocities ** 2).sum(axis=1)).max()),
                  "mean_u": float(velocities[:, 0].mean()), "mean_v": float(velocities[:, 1].mean()), "seconds": time.perf_counter() - started}
    np.savez(arguments.out, rows=rows, velocities=velocities.astype(np.float32),
             case_sha256=np.array(hashlib.sha256(case_path.read_bytes()).hexdigest()), provenance=np.array(json.dumps(provenance)))
    print(json.dumps(provenance, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
