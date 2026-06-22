"""
partition_v5.py — V5 dual-GPU static 1D X-axis partition.

Input contract: a **degenerate** CaseV5 from ``case_loader_v5.load_case_v5``
— owns the whole domain, no peer, ghost_grid = (0, 0), transport = empty.
``compute_dual_gpu_partition`` asserts this at entry to keep the loader →
partition seam honest.

Given that input, computes:
  - K_split_voxel_x        (the column where domain boundary sits)
  - per-GPU slab CaseV5    (own column range + leading/trailing ghost +
                            per-direction transport spec)

Per-direction transport spec offsets match ``shaders/ghost_send.comp``'s
spec const semantics (see Option B header in that file).

V5 v1.0: 2 GPUs only, static partition (V3+ generalizes to N + dynamic).
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np

from experiment.v5.utils.case_v5 import (
    CaseV5,
    Capacities,
    DirectionalTransportSpec,
    GhostGridParams,
    GridLayout,
    InitialParticles,
    KIND_FLUID,
    TransportConfig,
)


GHOST_THICKNESS = 1   # V5 v1.0: 1-voxel-thick ghost on the interior side


def _ghost_pool_size(case: CaseV5) -> int:
    """Worst-case pid-slot count for ONE direction's ghost pool.

    V5 v1.0 ghost is 1 voxel thick (``GHOST_THICKNESS``), so the ghost zone
    is one full x-column = NY × NZ voxels. Each ghost voxel reserves slots
    for two kinds of particles that **share the same pid pool**:

      - REPLICAS  : peer's boundary-column particles copied every step by
                    ghost_send.comp (live for one step, overwritten next).
                    Up to ``max_particles_per_voxel`` per voxel.
      - MIGRATIONS: peer particles that crossed the boundary; install_migrations
                    promotes them into own pid the same step. Up to
                    ``max_incoming_per_voxel`` per voxel.

    Total = (NY · NZ) · (max_particles_per_voxel + max_incoming_per_voxel).

    "Worst-case" assumes every ghost voxel saturates simultaneously. Real
    distributions are uneven (boundary-adjacent voxels fill, distant ones
    stay sparse), so typical occupancy is far below this. Over-allocating
    is cheap (~16 B / slot in set 0 SoA); the alternative — overflow — silently
    drops particles via ``overflow_inside_count`` / ``overflow_incoming_count``.
    """
    voxel_per_x = case.grid.grid_dimension_y * case.grid.grid_dimension_z
    return voxel_per_x * (case.capacities.max_particles_per_voxel
                          + case.capacities.max_incoming_per_voxel)


def compute_k_split(global_case: CaseV5, weights: list[float]) -> int:
    """Pick K_split_voxel_x so fluid particle count is split per weights.

    Same algorithm as V1's partition.compute_partition: bin fluid particles
    by global x_index, cumsum, searchsorted for target fraction. Returns
    the voxel column where GPU 0's own range ends (and GPU 1's own begins).
    """
    if len(weights) != 2:
        raise NotImplementedError("V5 v1.0 supports exactly 2 GPUs")
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
    global_case: CaseV5,
    x_lo_inclusive: int,
    x_hi_exclusive: int,
) -> InitialParticles:
    """Slice the global particle set down to one slab's OWN x-column range.

    For each global particle, compute its voxel x-index via
    ``floor((position.x - origin_x) / h)`` (matches ``_compute_grid``'s
    convention that voxel (0,…) is centered on bbox_min). Keep particles
    whose voxel-x falls in ``[lo, hi)``.

    The clamp ``np.clip(x_indices, 0, grid_nx - 1)`` covers two edge cases:
      - Particles exactly on the +x bbox face floor() to ``grid_nx`` (one
        past the last valid column) — clamp pulls them back into the last
        column where they geometrically belong.
      - Particles at ``position.x == origin_x`` floor() to 0 already; the
        ``max(0, ...)`` half is defensive against slight float drift
        producing -1 for particles at or just below the bbox-min face.

    Returns INDEPENDENT arrays (``.copy()``) so each slab's InitialParticles
    can be mutated downstream without aliasing the global case.

    Note: ghost-column particles are NOT included here. Ghost data is
    populated at runtime by ``ghost_send.comp`` from the peer GPU, not at
    load time. This filter is strictly for OWN particles.
    """
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
    global_case: CaseV5,
    slot_index: int,
    k_split: int,
    grid_nx: int,
    own_pool_size: int,
    slot0_own_pool: int,
) -> CaseV5:
    """Build a per-GPU CaseV5 for slot 0 (leftmost) or slot 1 (rightmost).

    ``own_pool_size`` is THIS slab's own particle-pool capacity (may be < the
    global pool when per-slab shrinking is enabled). ``slot0_own_pool`` is slot
    0's own_pool_size, which is the ONLY pool the cross-GPU pid offset depends on
    (slot 0's trailing-ghost range starts at slot0_own_pool+1; slot 1's leading-
    ghost range is [1,G], independent of slot 1's pool). Both must be threaded in
    so the offsets stay correct when slot 0 is shrunk.

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
        pid_offset = _compute_pid_offset(slot0_own_pool, slot_index=1, direction="leading")
        transport.leading = DirectionalTransportSpec(
            direction=0,
            boundary_voxel_x_local=leading_thickness,
            ghost_voxel_x_local=0,
            ghost_pid_offset_to_receiver=pid_offset,
            ghost_voxel_id_offset_to_receiver=voxel_id_offset,
        )
    if trailing_thickness > 0:
        # Slot 0's trailing send: sends to slot 1's leading ghost
        my_extended_nx = own_x_count + trailing_thickness   # k_split + 1
        my_own_boundary_first_x_local = my_extended_nx - 1 - trailing_thickness  # = own_last_x_local
        peer_ghost_first_x_local = 0  # slot 1's leading ghost
        voxel_id_offset = (peer_ghost_first_x_local - my_own_boundary_first_x_local) * voxel_per_x
        pid_offset = _compute_pid_offset(slot0_own_pool, slot_index=0, direction="trailing")
        transport.trailing = DirectionalTransportSpec(
            direction=1,
            boundary_voxel_x_local=my_own_boundary_first_x_local,
            ghost_voxel_x_local=my_extended_nx - 1,
            ghost_pid_offset_to_receiver=pid_offset,
            ghost_voxel_id_offset_to_receiver=voxel_id_offset,
        )

    capacities = Capacities(
        max_particles_per_voxel=global_case.capacities.max_particles_per_voxel,
        workgroup_size=global_case.capacities.workgroup_size,
        max_incoming_per_voxel=global_case.capacities.max_incoming_per_voxel,
        # Per-slab pool: own_pool_size = this slab's share + migration headroom
        # (see compute_dual_gpu_partition pool_safety). Falls back to the global
        # whole-domain size when pool_safety is None (legacy behaviour).
        own_pool_size=own_pool_size,
        leading_ghost_pool_size=leading_pool,
        trailing_ghost_pool_size=trailing_pool,
    )

    initial = _filter_particles_by_x_range(
        global_case,
        x_lo_inclusive=own_global_first,
        x_hi_exclusive=own_global_last + 1,
    )

    return CaseV5(
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
    slot0_own_pool: int,    # slot 0's own_pool_size — the ONLY pool the offset depends on
    *,
    slot_index: int,        # the sender
    direction: str,         # "leading" or "trailing"
) -> int:
    """Compute ``GHOST_PID_OFFSET_TO_RECEIVER`` for one send direction.

    ``slot0_own_pool`` is P below. The offset depends ONLY on slot 0's pool: slot
    0's trailing-ghost range starts at P+1, and slot 1's leading-ghost range is
    [1,G] regardless of slot 1's pool. So shrinking slot 1 needs no offset change;
    shrinking slot 0 requires passing the shrunk P here (the caller does).

    ``ghost_send.comp`` uses this to pre-encode each ghost replica's pid in
    the receiver's coordinate system, so receiver's ``install_migrations.comp``
    sees ready-to-install bytes without any CPU remap:

        peer_dst_pid = my_dst_pid + GHOST_PID_OFFSET_TO_RECEIVER

    where ``my_dst_pid`` is the slot sender allocated in its own ghost-pid
    range, and ``peer_dst_pid`` is the same slot expressed in the receiver's
    pid layout.

    Per-GPU pid layout (P = own_pool_size, G = ghost_pool_size):

        slot 0  (trailing peer = slot 1, no leading peer):
            0           : sentinel
            1 .. P      : own particles
            P+1 .. P+G  : trailing-ghost-pid range
                            ↳ sender writes here when sending to slot 1
                            ↳ receives here when slot 1 sends back

        slot 1  (leading peer = slot 0, no trailing peer):
            0           : sentinel
            1 .. G      : leading-ghost-pid range
                            ↳ sender writes here when sending to slot 0
                            ↳ receives here when slot 0 sends back
            G+1 .. G+P  : own particles

    For the k-th slot of a send, sender allocates pid = ``sender_first + k``
    and wants the receiver to interpret it as pid = ``receiver_first + k``:

        offset = (receiver_first + k) - (sender_first + k)
               = receiver_first - sender_first       ← k drops out

    Per-direction derivation:

      (a) slot 0, trailing-send  →  slot 1's leading-receive
          sender_first   = P + 1   (slot 0's trailing range start)
          receiver_first = 1       (slot 1's leading  range start)
          offset = 1 - (P + 1)     = -P

      (b) slot 1, leading-send   →  slot 0's trailing-receive
          sender_first   = 1       (slot 1's leading  range start)
          receiver_first = P + 1   (slot 0's trailing range start)
          offset = (P + 1) - 1     = +P

    Note: the offset depends only on P, not on G. The two ghost ranges have
    the same width G by construction (symmetric 2-GPU), but their starting
    positions differ by exactly P slots — that's all the formula needs.

    Endpoint GPUs with no peer in this direction return 0 (caller drops it).
    """
    own_pool = slot0_own_pool

    if slot_index == 0 and direction == "trailing":
        sender_first   = own_pool + 1   # slot 0's trailing range starts after its own range
        receiver_first = 1              # slot 1's leading  range starts at pid 1
    elif slot_index == 1 and direction == "leading":
        sender_first   = 1              # slot 1's leading  range starts at pid 1
        receiver_first = own_pool + 1   # slot 0's trailing range starts after its own range
    else:
        return 0   # endpoint with no peer in this direction
    return receiver_first - sender_first


def compute_dual_gpu_partition(
    global_case: CaseV5,
    weights: list[float],
    pool_safety: Optional[float] = None,
) -> tuple[CaseV5, CaseV5, int]:
    """Top-level entry: split global_case into two slab CaseV5 + return K_split.

    Returns (slab_case_gpu0, slab_case_gpu1, k_split_voxel_x).

    ``pool_safety``:
      - None (default): legacy behaviour — both slabs get the global whole-domain
        own_pool_size. Maximally conservative, wastes empty-slot dispatch on the
        GPU owning the smaller share (NV scans ~944k dead slots per kernel).
      - float (e.g. 1.2): size each slab own_pool_size = ceil(slab_particles *
        pool_safety), rounded up to a workgroup multiple, capped at the global
        pool. The headroom above the slab's particle count covers cross-GPU
        migrants installed at the own-pool TAIL between defrags. Size this from
        the PoolHealthBuffer watermark (readback_pool_health) — for cavity 1M the
        measured peak migrant tail is only ~80-83/defrag-interval, so 1.1-1.2x is
        ample. The install overflow guard + pool_health WARN catch undersizing.
    """
    # Contract check — input must be a degenerate slab from load_case_v5.
    # Re-partitioning an already-partitioned case (ghost / transport populated)
    # would silently double-count ghost capacity and corrupt offsets.
    assert global_case.ghost_grid.leading_ghost_voxel_count == 0, (
        "global_case must be degenerate (leading_ghost_voxel_count == 0); "
        "got an already-partitioned slab")
    assert global_case.ghost_grid.trailing_ghost_voxel_count == 0, (
        "global_case must be degenerate (trailing_ghost_voxel_count == 0); "
        "got an already-partitioned slab")
    assert global_case.transport.leading is None, (
        "global_case.transport.leading must be None for a degenerate slab")
    assert global_case.transport.trailing is None, (
        "global_case.transport.trailing must be None for a degenerate slab")
    assert global_case.capacities.leading_ghost_pool_size == 0, (
        "global_case.capacities.leading_ghost_pool_size must be 0 for a degenerate slab")
    assert global_case.capacities.trailing_ghost_pool_size == 0, (
        "global_case.capacities.trailing_ghost_pool_size must be 0 for a degenerate slab")

    grid_nx = global_case.grid.grid_dimension_x
    k_split = compute_k_split(global_case, weights)
    print(f"[partition_v5] K_split = {k_split} / {grid_nx} "
          f"(GPU 0 owns {k_split} cols, GPU 1 owns {grid_nx - k_split})")

    # --- Per-slab own_pool_size ----------------------------------------------
    global_pool = global_case.capacities.own_pool_size
    if pool_safety is None:
        own_pool_0 = global_pool
        own_pool_1 = global_pool
    else:
        if pool_safety <= 1.0:
            raise ValueError(f"pool_safety must be > 1.0, got {pool_safety}")
        wg = global_case.capacities.workgroup_size
        # Per-slab OWN particle count (same filter the slab build uses).
        n0 = _filter_particles_by_x_range(global_case, 0, k_split).positions.shape[0]
        n1 = _filter_particles_by_x_range(global_case, k_split, grid_nx).positions.shape[0]

        def _sized(n: int) -> int:
            v = int(math.ceil(n * pool_safety))
            v = ((v + wg - 1) // wg) * wg          # round up to workgroup multiple
            return min(v, global_pool)             # never exceed the global pool
        own_pool_0 = _sized(n0)
        own_pool_1 = _sized(n1)
        print(f"[partition_v5] pool_safety={pool_safety}: "
              f"slot0 own_pool {global_pool:,}->{own_pool_0:,} (n={n0:,}); "
              f"slot1 own_pool {global_pool:,}->{own_pool_1:,} (n={n1:,})")

    # Both slabs' transport pid offsets depend ONLY on slot 0's pool (own_pool_0).
    slab0 = _build_slab_case(global_case, slot_index=0, k_split=k_split, grid_nx=grid_nx,
                             own_pool_size=own_pool_0, slot0_own_pool=own_pool_0)
    slab1 = _build_slab_case(global_case, slot_index=1, k_split=k_split, grid_nx=grid_nx,
                             own_pool_size=own_pool_1, slot0_own_pool=own_pool_0)
    print(f"  slab 0: own_x [0, {k_split}) + trailing ghost; "
          f"{slab0.initial.positions.shape[0]:,} particles, pool={own_pool_0:,}")
    print(f"  slab 1: own_x [{k_split}, {grid_nx}) + leading ghost; "
          f"{slab1.initial.positions.shape[0]:,} particles, pool={own_pool_1:,}")
    return slab0, slab1, k_split
