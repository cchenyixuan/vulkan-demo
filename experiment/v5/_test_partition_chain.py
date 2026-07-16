"""
_test_partition_chain.py — M2 acceptance tests for the N-way chain partition.

Zero-dependency runnable script (assert + non-zero exit on failure).

Layer 1  GOLDEN:      chain(N=2, minimum_own_columns=1) field-by-field equals
                      legacy_dual_gpu_partition on real cases x weights x
                      pool_safety (grid includes production values 1.2 and the
                      reversed/extreme weight orders).
Layer 2  INVARIANTS:  symmetric AND asymmetric weight chains at N=3..8 —
                      particle coverage, geometry self-consistency, per-link
                      pid/vid offset round-trip, INDEPENDENT pool-tiling
                      oracle (from Capacities alone, not the production
                      PidLayout helper — breaks test/production tautology),
                      ghost_voxel_x_local spec assertions, aliasing checks,
                      min-width rejection paths.
Layer 3  DEGENERATE:  N=1 == the untouched global case; isolate_slab
                      consistency (the eta_weak helper); global case verified
                      unmutated at the end.
Layer 4  GPU SMOKE:   (--gpu) instantiate SphSimulatorV5 on an N=4 INTERIOR
                      slab case — allocation + pipelines + spec-const packing,
                      no stepping.

Usage:
    .venv/Scripts/python.exe experiment/v5/_test_partition_chain.py [--quick] [--gpu]
    --quick   skip the 8M golden cases (1M only)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v5.utils.case_loader_v5 import load_case_v5
from experiment.v5.utils.case_v5 import CaseV5
from experiment.v5.utils.partition_v5 import (
    ChainPartition,
    _ghost_pool_size,
    _sized_pool,
    compute_chain_partition,
    compute_dual_gpu_partition,
    isolate_slab,
    legacy_dual_gpu_partition,
)

CASE_1M = "cases/lid_driven_cavity_2d/case.yaml"
CASE_8M = "cases/lid_driven_cavity_2d_8m/case.yaml"

_passed = 0
_failed: list[str] = []


def check(condition: bool, message: str) -> None:
    global _passed
    if condition:
        _passed += 1
    else:
        _failed.append(message)
        print(f"  FAIL: {message}")


# ---------------------------------------------------------------------------
# Deep CaseV5 comparison
# ---------------------------------------------------------------------------

def deep_diff(a, b, path: str = "case") -> list[str]:
    """Recursive field-by-field diff of dataclasses / lists / numpy arrays."""
    diffs: list[str] = []
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            diffs.append(f"{path}: array meta {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        elif not np.array_equal(a, b):
            diffs.append(f"{path}: array values differ "
                         f"({(a != b).sum()} of {a.size} elements)")
        return diffs
    if dataclasses.is_dataclass(a):
        for field_info in dataclasses.fields(a):
            diffs.extend(deep_diff(getattr(a, field_info.name),
                                   getattr(b, field_info.name),
                                   f"{path}.{field_info.name}"))
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(item_a, item_b, f"{path}[{index}]"))
        return diffs
    if a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")
    return diffs


# ---------------------------------------------------------------------------
# Layer 1: golden equality
# ---------------------------------------------------------------------------

def test_golden(global_case: CaseV5, tag: str, weight_sets, pool_safeties) -> None:
    print(f"[layer 1] golden N=2 equality — {tag}")
    for weights in weight_sets:
        for pool_safety in pool_safeties:
            legacy = legacy_dual_gpu_partition(global_case, list(weights),
                                               pool_safety)
            wrapped = compute_dual_gpu_partition(global_case, list(weights),
                                                 pool_safety)
            label = f"{tag} w={weights} safety={pool_safety}"
            check(legacy[2] == wrapped[2], f"{label}: k_split differs "
                                           f"{legacy[2]} vs {wrapped[2]}")
            for slot in (0, 1):
                diffs = deep_diff(legacy[slot], wrapped[slot], f"slab{slot}")
                check(not diffs, f"{label}: slab{slot} diffs: {diffs[:5]}")


# ---------------------------------------------------------------------------
# Layer 2: chain invariants (weight-agnostic verifier, reused everywhere)
# ---------------------------------------------------------------------------

def _all_particle_column_histogram(global_case: CaseV5) -> np.ndarray:
    grid_nx = global_case.grid.grid_dimension_x
    h = global_case.physics.smoothing_length
    x_indices = np.floor(
        (global_case.initial.positions[:, 0] - global_case.grid.origin_x) / h
    ).astype(np.int64)
    np.clip(x_indices, 0, grid_nx - 1, out=x_indices)
    return np.bincount(x_indices, minlength=grid_nx)


def verify_chain(global_case: CaseV5, chain: ChainPartition, label: str,
                 pool_safety) -> None:
    """All structural invariants for one constructed chain."""
    h = global_case.physics.smoothing_length
    voxel_per_x = (global_case.grid.grid_dimension_y
                   * global_case.grid.grid_dimension_z)
    grid_nx = global_case.grid.grid_dimension_x
    slab_count = len(chain.slabs)
    column_histogram = _all_particle_column_histogram(global_case)
    total_particles = int(global_case.initial.positions.shape[0])
    pool_per_direction = _ghost_pool_size(global_case)
    workgroup = global_case.capacities.workgroup_size
    global_pool = global_case.capacities.own_pool_size

    # --- coverage ---------------------------------------------------------
    check(len(chain.cuts) == slab_count - 1, f"{label}: cut count")
    check(all(chain.cuts[i] < chain.cuts[i + 1]
              for i in range(len(chain.cuts) - 1)),
          f"{label}: cuts not strictly monotonic: {chain.cuts}")
    slab_total = sum(g.own_particle_count for g in chain.geometry)
    check(slab_total == total_particles,
          f"{label}: particle coverage {slab_total} != {total_particles}")
    boundaries = [0] + chain.cuts + [grid_nx]
    for index, geometry in enumerate(chain.geometry):
        expected = int(column_histogram[
            boundaries[index]:boundaries[index + 1]].sum())
        check(geometry.own_particle_count == expected,
              f"{label}: slab{index} count vs histogram")
        check(chain.slabs[index].initial.positions.shape[0]
              == geometry.own_particle_count,
              f"{label}: slab{index} initial array count")

    # --- per-slab geometry + aliasing + particle-membership oracle ---------
    for index, (geometry, case) in enumerate(zip(chain.geometry, chain.slabs)):
        check(geometry.has_leading_peer == (index > 0)
              and geometry.has_trailing_peer == (index < slab_count - 1),
              f"{label}: slab{index} peer flags")
        check(case.grid.grid_dimension_x
              == geometry.own_column_count
              + geometry.leading_thickness + geometry.trailing_thickness,
              f"{label}: slab{index} extended nx")
        expected_origin = (global_case.grid.origin_x
                           + (geometry.own_global_first_column
                              - geometry.leading_thickness) * h)
        check(abs(case.grid.origin_x - expected_origin) < 1e-12,
              f"{label}: slab{index} origin_x")
        check(case.ghost_grid.leading_ghost_voxel_count
              == geometry.leading_thickness * voxel_per_x
              and case.ghost_grid.trailing_ghost_voxel_count
              == geometry.trailing_thickness * voxel_per_x,
              f"{label}: slab{index} ghost voxel counts")
        check(case.capacities.leading_ghost_pool_size
              == (pool_per_direction if index > 0 else 0)
              and case.capacities.trailing_ghost_pool_size
              == (pool_per_direction if index < slab_count - 1 else 0),
              f"{label}: slab{index} ghost pool sizes")
        if 0 < index < slab_count - 1:
            check(case.transport.leading is not None
                  and case.transport.trailing is not None,
                  f"{label}: slab{index} interior must have both specs")
        # Aliasing: slab arrays must be independent copies of the global.
        check(not np.shares_memory(case.initial.positions,
                                   global_case.initial.positions),
              f"{label}: slab{index} positions alias the global case")
        if index > 0:
            check(not np.shares_memory(case.initial.positions,
                                       chain.slabs[index - 1].initial.positions),
                  f"{label}: slab{index} positions alias slab{index-1}")
        # Particle-membership oracle: recompute each particle's LOCAL column
        # from the slab's own origin — every own particle must land inside
        # [leading_thickness, leading_thickness + own_column_count).
        local_columns = np.floor(
            (case.initial.positions[:, 0] - case.grid.origin_x) / h
        ).astype(np.int64)
        np.clip(local_columns, 0, case.grid.grid_dimension_x - 1,
                out=local_columns)
        low = geometry.leading_thickness
        high = geometry.leading_thickness + geometry.own_column_count
        check(bool(((local_columns >= low) & (local_columns < high)).all()),
              f"{label}: slab{index} has particles outside its own columns")
        # Pool sizing rule (exact, not just lower-bounded).
        if pool_safety is not None:
            check(geometry.own_pool_size == _sized_pool(
                      geometry.own_particle_count, pool_safety, workgroup,
                      global_pool),
                  f"{label}: slab{index} pool != _sized_pool rule")
        else:
            check(geometry.own_pool_size == global_pool,
                  f"{label}: slab{index} pool != global pool (safety None)")

    # --- links: uniqueness + INDEPENDENT pool-tiling oracle -----------------
    triples = [(link.sender_index, link.receiver_index, link.direction)
               for link in chain.links]
    check(len(triples) == 2 * (slab_count - 1),
          f"{label}: link count {len(triples)} != 2(N-1)")
    check(len(set(triples)) == len(triples), f"{label}: duplicate links")
    for index in range(slab_count - 1):
        check((index, index + 1, "trailing") in triples
              and (index + 1, index, "leading") in triples,
              f"{label}: link pair {index}<->{index+1} missing")

    for link in chain.links:
        sender_case = chain.slabs[link.sender_index]
        receiver_case = chain.slabs[link.receiver_index]
        receive_side = ("leading" if link.direction == "trailing"
                        else "trailing")
        # INDEPENDENT oracle — receiver ghost range start computed from the
        # receiver's Capacities ALONE (never via PidLayout, which production
        # also uses). Pool layout must tile: [0][1..L][L+1..L+O][L+O+1..L+O+T].
        receiver_capacities = receiver_case.capacities
        receiver_pool_total = (1
                               + receiver_capacities.leading_ghost_pool_size
                               + receiver_capacities.own_pool_size
                               + receiver_capacities.trailing_ghost_pool_size)
        if receive_side == "leading":
            independent_receiver_first = 1
        else:
            independent_receiver_first = (
                1 + receiver_capacities.leading_ghost_pool_size
                + receiver_capacities.own_pool_size)
            check(independent_receiver_first
                  == receiver_pool_total
                  - receiver_capacities.trailing_ghost_pool_size,
                  f"{label}: receiver pool does not tile")
        sender_capacities = sender_case.capacities
        if link.direction == "leading":
            independent_sender_first = 1
        else:
            independent_sender_first = (
                1 + sender_capacities.leading_ghost_pool_size
                + sender_capacities.own_pool_size)
        offset = link.ghost_pid_offset_to_receiver
        check(independent_sender_first + offset == independent_receiver_first,
              f"{label}: link {link.sender_index}->{link.receiver_index} "
              f"pid offset fails the independent capacities oracle")
        check(independent_receiver_first >= 1
              and independent_receiver_first + pool_per_direction - 1
              < receiver_pool_total,
              f"{label}: receiver ghost range escapes the pool")
        # Send/receive pool width must match (worker staging assert, statically).
        sender_side_pool = (sender_capacities.leading_ghost_pool_size
                            if link.direction == "leading"
                            else sender_capacities.trailing_ghost_pool_size)
        receiver_side_pool = (receiver_capacities.leading_ghost_pool_size
                              if receive_side == "leading"
                              else receiver_capacities.trailing_ghost_pool_size)
        check(sender_side_pool == receiver_side_pool == pool_per_direction,
              f"{label}: link pool width mismatch")

        # --- vid offset + spec fields ------------------------------------
        spec = (sender_case.transport.trailing if link.direction == "trailing"
                else sender_case.transport.leading)
        check(spec is not None
              and spec.ghost_pid_offset_to_receiver == offset,
              f"{label}: LinkSpec/DirectionalTransportSpec offset mismatch")
        sender_geometry = chain.geometry[link.sender_index]
        receiver_geometry = chain.geometry[link.receiver_index]
        # ghost_voxel_x_local (spec id 92): my ghost column for this side.
        expected_ghost_voxel_x = (
            0 if link.direction == "leading"
            else sender_case.grid.grid_dimension_x - 1)
        check(spec.ghost_voxel_x_local == expected_ghost_voxel_x,
              f"{label}: link {link.sender_index} {link.direction} "
              f"ghost_voxel_x_local {spec.ghost_voxel_x_local} "
              f"!= {expected_ghost_voxel_x}")
        # boundary_voxel_x_local: first/last own column depending on side.
        expected_boundary_x = (
            sender_geometry.leading_thickness
            if link.direction == "leading"
            else sender_geometry.leading_thickness
            + sender_geometry.own_column_count - 1)
        check(spec.boundary_voxel_x_local == expected_boundary_x,
              f"{label}: boundary_voxel_x_local")
        # vid offset: column-multiple + receiver-local target + WORLD-space.
        column_delta = spec.ghost_voxel_id_offset_to_receiver // voxel_per_x
        check(spec.ghost_voxel_id_offset_to_receiver
              == column_delta * voxel_per_x,
              f"{label}: vid offset not a column multiple")
        receiver_ghost_local = spec.boundary_voxel_x_local + column_delta
        expected_receiver_ghost_local = (
            0 if link.direction == "trailing"
            else receiver_geometry.leading_thickness
            + receiver_geometry.own_column_count)
        check(receiver_ghost_local == expected_receiver_ghost_local,
              f"{label}: vid target column")
        sender_world = (sender_case.grid.origin_x
                        + spec.boundary_voxel_x_local * h)
        receiver_world = (receiver_case.grid.origin_x
                          + receiver_ghost_local * h)
        check(abs(sender_world - receiver_world) < 1e-9,
              f"{label}: boundary/ghost world-x mismatch")


def test_chain_invariants(global_case: CaseV5, tag: str,
                          weight_sets, pool_safeties) -> None:
    print(f"[layer 2] chain invariants — {tag}")
    for weights in weight_sets:
        for pool_safety in pool_safeties:
            label = f"{tag} w={list(weights)} safety={pool_safety}"
            chain = compute_chain_partition(global_case, list(weights),
                                            pool_safety)
            verify_chain(global_case, chain, label, pool_safety)


def test_min_width_paths(global_case: CaseV5, tag: str) -> None:
    print(f"[layer 2] min-width rejection — {tag}")
    grid_nx = global_case.grid.grid_dimension_x
    too_many = grid_nx // 12 + 1
    try:
        compute_chain_partition(global_case, [1.0] * too_many)
        check(False, f"{tag}: N={too_many} should have raised (grid too small)")
    except ValueError:
        check(True, "")
    # Extreme weights: cut clamped to the minimum width — run the FULL
    # invariant body on the clamped chain, not just the width assert.
    chain = compute_chain_partition(global_case, [1000.0, 1.0, 1.0],
                                    pool_safety=1.2)
    widths = [g.own_column_count for g in chain.geometry]
    check(all(width >= 12 for width in widths),
          f"{tag}: extreme weights violated min width: {widths}")
    verify_chain(global_case, chain, f"{tag} clamped[1000,1,1]", 1.2)


# ---------------------------------------------------------------------------
# Layer 3: degenerate chain + isolate_slab
# ---------------------------------------------------------------------------

def test_degenerate_and_isolate(global_case: CaseV5, tag: str) -> None:
    print(f"[layer 3] N=1 degenerate + isolate_slab — {tag}")
    chain_single = compute_chain_partition(global_case, [1.0])
    diffs = deep_diff(global_case, chain_single.slabs[0], "N1")
    check(not diffs, f"{tag}: N=1 != global case: {diffs[:5]}")

    chain = compute_chain_partition(global_case, [1.0] * 4, pool_safety=1.05)
    h = global_case.physics.smoothing_length
    for index in range(4):
        isolated = isolate_slab(global_case, chain, index)
        geometry = chain.geometry[index]
        check(isolated.transport.leading is None
              and isolated.transport.trailing is None,
              f"{tag}: isolated slab{index} has transport")
        check(isolated.capacities.leading_ghost_pool_size == 0
              and isolated.capacities.trailing_ghost_pool_size == 0,
              f"{tag}: isolated slab{index} has ghost pools")
        check(isolated.grid.grid_dimension_x == geometry.own_column_count,
              f"{tag}: isolated slab{index} grid width")
        expected_origin = (global_case.grid.origin_x
                           + geometry.own_global_first_column * h)
        check(abs(isolated.grid.origin_x - expected_origin) < 1e-12,
              f"{tag}: isolated slab{index} origin_x")
        check(isolated.capacities.own_pool_size == geometry.own_pool_size,
              f"{tag}: isolated slab{index} pool size differs from chain slab")
        check(np.array_equal(isolated.initial.positions,
                             chain.slabs[index].initial.positions),
              f"{tag}: isolated slab{index} particle set differs")


# ---------------------------------------------------------------------------
# Layer 4: GPU construction smoke (--gpu)
# ---------------------------------------------------------------------------

def test_gpu_interior_construction(global_case: CaseV5) -> None:
    print("[layer 4] GPU construction smoke on an interior slab")
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    chain = compute_chain_partition(global_case, [1.0] * 4, pool_safety=1.05)
    interior_case = chain.slabs[1]      # both-sided slab
    ctx = VulkanContextV5.create(device_index=0,
                                 application_name="m2_interior_smoke")
    try:
        sim = SphSimulatorV5(ctx, interior_case, sync_scheme="per-direction")
        check(sim.case.transport.leading is not None
              and sim.case.transport.trailing is not None,
              "interior sim lost a transport side")
        check(len(sim._transport_segments) == 2,
              f"interior sim transport segments: "
              f"{list(sim._transport_segments)} != 2")
        state = sim.sync_state()
        check("transport_leading" in state and "transport_trailing" in state,
              f"per-direction scheme semaphores missing: {state}")
        sim.destroy()
        check(True, "")
        print("  interior slab: alloc + pipelines + both-sided staging OK")
    finally:
        ctx.destroy()


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="skip 8M golden (1M only)")
    parser.add_argument("--gpu", action="store_true",
                        help="also run the GPU construction smoke")
    args = parser.parse_args()

    print("loading 1M case ...")
    case_1m = load_case_v5(CASE_1M)
    case_1m_snapshot = copy.deepcopy(case_1m)

    test_golden(case_1m, "1M",
                weight_sets=[(1.0, 1.0), (3.2, 1.0), (2.6, 1.0),
                             (1.0, 3.2), (1000.0, 1.0)],
                pool_safeties=[None, 1.05, 1.2, 1.5])
    test_chain_invariants(
        case_1m, "1M",
        weight_sets=[[1.0] * 3, [1.0] * 4, [1.0] * 5, [1.0] * 8,
                     [2.0, 1.0, 3.0], [1.0, 4.0, 1.5, 1.0]],
        pool_safeties=(None, 1.05, 1.2))
    test_min_width_paths(case_1m, "1M")
    test_degenerate_and_isolate(case_1m, "1M")

    if not args.quick:
        print("loading 8M case ...")
        case_8m = load_case_v5(CASE_8M)
        test_golden(case_8m, "8M",
                    weight_sets=[(1.0, 1.0), (1.0, 3.2)],
                    pool_safeties=[None, 1.2])
        test_chain_invariants(case_8m, "8M",
                              weight_sets=[[1.0] * 4, [1.0, 4.0, 1.5, 1.0]],
                              pool_safeties=(1.05,))

    # Partition must never mutate the global case.
    mutation = deep_diff(case_1m, case_1m_snapshot, "global_1m")
    check(not mutation, f"global case mutated by partitioning: {mutation[:5]}")

    if args.gpu:
        test_gpu_interior_construction(case_1m)

    print(f"\n{_passed} checks passed, {len(_failed)} failed")
    if _failed:
        for message in _failed:
            print(f"  FAILED: {message}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
