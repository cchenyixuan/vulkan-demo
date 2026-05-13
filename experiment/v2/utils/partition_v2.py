"""
partition_v2.py — V2 dual-GPU static 1D X-axis partition.

Takes a global CaseV2 (loaded by case_loader_v2.load_case_v2) and computes:
  - K_split_voxel_x        (the column where domain boundary sits)
  - per-GPU slab CaseV2    (own column range + leading/trailing ghost)
  - per-direction transport spec (offsets that match ghost_send.comp's
    spec const semantics — see shaders/ghost_send.comp Option B header)

V2 v1.0: 2 GPUs only, static partition (V3+ generalizes to N + dynamic).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from experiment.v2.utils.case_v2 import (
    CaseV2,
    Capacities,
    DirectionalTransportSpec,
    GhostGridParams,
    GridLayout,
    InitialParticles,
    KIND_FLUID,
    TransportConfig,
)


GHOST_THICKNESS = 1   # V2 v1.0: 1-voxel-thick ghost on the interior side


def _ghost_pool_size(case: CaseV2) -> int:
    """Worst-case per-direction ghost pool capacity. One yz-face's worth of
    replicas (one full boundary voxel) + migrations (one full incoming list)."""
    voxel_per_x = case.grid.grid_dimension_y * case.grid.grid_dimension_z
    return voxel_per_x * (case.capacities.max_particles_per_voxel
                          + case.capacities.max_incoming_per_voxel)


def compute_k_split(global_case: CaseV2, weights: list[float]) -> int:
    """Pick K_split_voxel_x so fluid particle count is split per weights.

    Same algorithm as V1's partition.compute_partition: bin fluid particles
    by global x_index, cumsum, searchsorted for target fraction. Returns
    the voxel column where GPU 0's own range ends (and GPU 1's own begins).
    """
    if len(weights) != 2:
        raise NotImplementedError("V2 v1.0 supports exactly 2 GPUs")
    if any(w <= 0 for w in weights):
        raise ValueError(f"weights must be positive, got {weights}")

    grid_nx = global_case.grid.grid_dimension_x
    h = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x

    # Bin fluid particles by global x_index
    fluid_counts = np.zeros(grid_nx, dtype=np.int64)
    positions = global_case.initial.positions
    materials = global_case.initial.material_group
    for i in range(positions.shape[0]):
        if global_case.materials[int(materials[i])].kind != KIND_FLUID:
            continue
        x_idx = int(np.floor((positions[i, 0] - origin_x) / h))
        x_idx = max(0, min(x_idx, grid_nx - 1))
        fluid_counts[x_idx] += 1

    fluid_total = int(fluid_counts.sum())
    if fluid_total == 0:
        raise ValueError("global case has no fluid particles")

    fraction_gpu0 = weights[0] / sum(weights)
    target = max(1, int(fluid_total * fraction_gpu0))
    cumsum = np.cumsum(fluid_counts)
    k = int(np.searchsorted(cumsum, target, side="left"))
    # Clamp so each side owns ≥ 1 column
    return max(1, min(k, grid_nx - 1))


def _filter_particles_by_x_range(
    global_case: CaseV2,
    x_lo_inclusive: int,
    x_hi_exclusive: int,
) -> InitialParticles:
    """Keep particles whose global voxel x-index ∈ [lo, hi)."""
    positions = global_case.initial.positions
    velocities = global_case.initial.velocities
    material_group = global_case.initial.material_group
    h = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x
    x_indices = np.floor((positions[:, 0] - origin_x) / h).astype(np.int64)
    grid_nx = global_case.grid.grid_dimension_x
    np.clip(x_indices, 0, grid_nx - 1, out=x_indices)
    mask = (x_indices >= x_lo_inclusive) & (x_indices < x_hi_exclusive)
    return InitialParticles(
        positions=positions[mask].copy(),
        velocities=velocities[mask].copy(),
        material_group=material_group[mask].copy(),
    )


def _build_slab_case(
    global_case: CaseV2,
    slot_index: int,
    k_split: int,
    grid_nx: int,
) -> CaseV2:
    """Build a per-GPU CaseV2 for slot 0 (leftmost) or slot 1 (rightmost).

    Geometry:
      Slot 0: own = [0, k_split); trailing peer = GPU 1
              ghost column on TRAILING side at global x = k_split
              extended_nx_0 = k_split + GHOST_THICKNESS
              origin shift: unchanged (own starts at global x=0)
      Slot 1: own = [k_split, grid_nx); leading peer = GPU 0
              ghost column on LEADING side at global x = k_split - 1
              extended_nx_1 = (grid_nx - k_split) + GHOST_THICKNESS
              origin shift: origin_x += (k_split - GHOST_THICKNESS) * h
                            so extended grid voxel 0 = global x (k_split - GHOST_THICKNESS)

    Transport spec (per docs §6 + shaders/ghost_send.comp Option B):
      offset_in_voxel_id_space = (peer_ghost_first_x_local - my_own_boundary_first_x_local) × NY × NZ
      For 2-GPU symmetric: same numeric offset for both directions due to
      cancellation, but opposite sign (since "my boundary" and "peer ghost"
      swap roles).
    """
    h = global_case.physics.smoothing_length
    ny = global_case.grid.grid_dimension_y
    nz = global_case.grid.grid_dimension_z
    voxel_per_x = ny * nz

    leading_thickness = GHOST_THICKNESS if slot_index == 1 else 0
    trailing_thickness = GHOST_THICKNESS if slot_index == 0 else 0

    if slot_index == 0:
        own_x_count = k_split
        own_global_first = 0
        own_global_last = k_split - 1                  # inclusive
    else:
        own_x_count = grid_nx - k_split
        own_global_first = k_split
        own_global_last = grid_nx - 1                  # inclusive

    extended_nx = own_x_count + leading_thickness + trailing_thickness

    # Origin: extended grid voxel 0 in world coords
    new_origin_x = global_case.grid.origin_x + (own_global_first - leading_thickness) * h

    grid = GridLayout(
        origin_x=new_origin_x,
        origin_y=global_case.grid.origin_y,
        origin_z=global_case.grid.origin_z,
        grid_dimension_x=extended_nx,
        grid_dimension_y=ny,
        grid_dimension_z=nz,
        voxel_order=global_case.grid.voxel_order,
    )

    leading_voxel_count = leading_thickness * voxel_per_x
    trailing_voxel_count = trailing_thickness * voxel_per_x
    ghost_grid = GhostGridParams(
        leading_ghost_voxel_count=leading_voxel_count,
        trailing_ghost_voxel_count=trailing_voxel_count,
    )

    pool_per_dir = _ghost_pool_size(global_case)
    leading_pool = pool_per_dir if leading_thickness > 0 else 0
    trailing_pool = pool_per_dir if trailing_thickness > 0 else 0

    # Per-direction transport specs
    transport = TransportConfig()
    if leading_thickness > 0:
        # Slot 1's leading send: sends to slot 0's trailing ghost
        # my.own_boundary_first_x_local = leading_thickness (own_first_x in extended)
        # peer (slot 0) ghost is at slot 0's trailing column = slot 0 extended_nx - 1
        # peer.ghost_first_x_local = (k_split + GHOST_THICKNESS) - 1   = k_split
        # In local-to-local terms: offset = (peer_ghost_x_local_in_peer_grid - my_own_boundary_x_local_in_my_grid) * NY * NZ
        peer_extended_nx = k_split + GHOST_THICKNESS  # slot 0's extended_nx
        peer_ghost_first_x_local = peer_extended_nx - 1   # trailing ghost
        my_own_boundary_first_x_local = leading_thickness
        voxel_id_offset = (peer_ghost_first_x_local - my_own_boundary_first_x_local) * voxel_per_x
        # PID offset: both GPUs have equal own_pool_size; peer's leading_ghost_pool_size
        # == my trailing_ghost_pool_size (== pool_per_dir). For now keep 0; refined below
        # based on the actual pid layout (slot 0's trailing-ghost-pid range vs slot 1's leading-ghost-pid range).
        pid_offset = _compute_pid_offset(global_case, slot_index=1, direction="leading", peer_slot=0)
        transport.leading = DirectionalTransportSpec(
            direction=0,
            boundary_voxel_x_local=leading_thickness,
            ghost_voxel_x_local=0,
            ghost_pid_offset_to_receiver=pid_offset,
            ghost_voxel_id_offset_to_receiver=voxel_id_offset,
            pool_size=leading_pool,
        )
    if trailing_thickness > 0:
        # Slot 0's trailing send: sends to slot 1's leading ghost
        my_extended_nx = own_x_count + trailing_thickness   # k_split + 1
        my_own_boundary_first_x_local = my_extended_nx - 1 - trailing_thickness  # = own_last_x_local
        peer_ghost_first_x_local = 0  # slot 1's leading ghost
        voxel_id_offset = (peer_ghost_first_x_local - my_own_boundary_first_x_local) * voxel_per_x
        pid_offset = _compute_pid_offset(global_case, slot_index=0, direction="trailing", peer_slot=1)
        transport.trailing = DirectionalTransportSpec(
            direction=1,
            boundary_voxel_x_local=my_own_boundary_first_x_local,
            ghost_voxel_x_local=my_extended_nx - 1,
            ghost_pid_offset_to_receiver=pid_offset,
            ghost_voxel_id_offset_to_receiver=voxel_id_offset,
            pool_size=trailing_pool,
        )

    capacities = Capacities(
        max_particles_per_voxel=global_case.capacities.max_particles_per_voxel,
        workgroup_size=global_case.capacities.workgroup_size,
        max_incoming_per_voxel=global_case.capacities.max_incoming_per_voxel,
        own_pool_size=global_case.capacities.own_pool_size,
        leading_ghost_pool_size=leading_pool,
        trailing_ghost_pool_size=trailing_pool,
    )

    initial = _filter_particles_by_x_range(
        global_case,
        x_lo_inclusive=own_global_first,
        x_hi_exclusive=own_global_last + 1,
    )

    return CaseV2(
        physics=global_case.physics,
        numerics=global_case.numerics,
        capacities=capacities,
        grid=grid,
        ghost_grid=ghost_grid,
        transport=transport,
        materials=list(global_case.materials),       # shallow copy ok; immutable per-run
        initial=initial,
    )


def _compute_pid_offset(
    global_case: CaseV2,
    *,
    slot_index: int,        # the sender
    direction: str,         # "leading" or "trailing"
    peer_slot: int,         # the receiver
) -> int:
    """Compute GHOST_PID_OFFSET_TO_RECEIVER.

    From shaders/ghost_send.comp:
      peer_dst_pid = my_dst_pid + GHOST_PID_OFFSET_TO_RECEIVER

    where my_dst_pid is the sender's ghost-pid slot (in sender's pid layout)
    and peer_dst_pid is the corresponding slot in receiver's pid layout
    (where receiver's install_migrations reads from).

    Sender's ghost-pid range for `direction`:
      direction=trailing → [own_first_pid + own_pool_size, ... + trailing_pool] (trailing range)
      direction=leading  → [1, leading_pool]

    Receiver's ghost-pid range for the corresponding receive direction:
      sender direction=trailing → receiver's LEADING range = [1, leading_pool]
      sender direction=leading  → receiver's TRAILING range

    For symmetric 2-GPU (equal own_pool_size, equal ghost pools):
      offset = receiver_first_dst_pid - sender_first_dst_pid

    Sender first dst pid (direction=trailing) = leading_ghost_pool + own_pool + 1
                                              = (leading + own) + 1
    Receiver first dst pid (its leading)      = 1
    offset = 1 - (leading + own + 1) = -(leading + own)

    For 2-GPU where both have same own + both have one peer (one direction
    only), sender's leading = 0 if direction=trailing-send. So:
      offset = 1 - (0 + own + 1) = -own_pool_size  for trailing-send→receiver-leading
              = (leading + own + 1) - 1            for leading-send→receiver-trailing
              = own_pool_size + leading_ghost_pool_size  (positive)

    Conservatively compute both endpoints from layout.
    """
    own_pool = global_case.capacities.own_pool_size
    ghost_pool = _ghost_pool_size(global_case)

    if slot_index == 0 and direction == "trailing":
        # sender = slot 0; its trailing ghost-pid first slot:
        # slot 0 has leading_ghost = 0; its own range = [1, own_pool]
        # its trailing range = [own_pool + 1, own_pool + ghost_pool]
        sender_first = own_pool + 1
        # receiver = slot 1; its leading-pid range = [1, ghost_pool]
        receiver_first = 1
    elif slot_index == 1 and direction == "leading":
        # sender = slot 1; its leading ghost-pid first slot = 1
        sender_first = 1
        # receiver = slot 0; its trailing range = [own_pool + 1, own_pool + ghost_pool]
        receiver_first = own_pool + 1
    else:
        return 0   # endpoint with no peer in this direction
    return receiver_first - sender_first


def compute_dual_gpu_partition(
    global_case: CaseV2,
    weights: list[float],
) -> tuple[CaseV2, CaseV2, int]:
    """Top-level entry: split global_case into two slab CaseV2 + return K_split.

    Returns (slab_case_gpu0, slab_case_gpu1, k_split_voxel_x).
    """
    grid_nx = global_case.grid.grid_dimension_x
    k_split = compute_k_split(global_case, weights)
    print(f"[partition_v2] K_split = {k_split} / {grid_nx} "
          f"(GPU 0 owns {k_split} cols, GPU 1 owns {grid_nx - k_split})")
    slab0 = _build_slab_case(global_case, slot_index=0, k_split=k_split, grid_nx=grid_nx)
    slab1 = _build_slab_case(global_case, slot_index=1, k_split=k_split, grid_nx=grid_nx)
    print(f"  slab 0: own_x [0, {k_split}) + trailing ghost; "
          f"{slab0.initial.positions.shape[0]:,} particles")
    print(f"  slab 1: own_x [{k_split}, {grid_nx}) + leading ghost; "
          f"{slab1.initial.positions.shape[0]:,} particles")
    return slab0, slab1, k_split
