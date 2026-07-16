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


def legacy_dual_gpu_partition(
    global_case: CaseV5,
    weights: list[float],
    pool_safety: Optional[float] = None,
) -> tuple[CaseV5, CaseV5, int]:
    """LEGACY 2-GPU implementation, kept verbatim as the golden reference for
    ``_test_partition_chain.py``. Production entry points go through
    ``compute_chain_partition`` / ``compute_dual_gpu_partition`` (below); this
    body is the pre-M2 code that months of GPU runs validated (50k drift=0,
    the 12 h soak). Do not modify.

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


# ============================================================================
# M2: N-way chain partition (docs/sph_v5_design.md §3.2)
#
# Generalizes the 2-slot logic above to an N-slab 1D chain. Interior slabs
# have ghost + transport on BOTH sides. The legacy dual implementation is
# kept verbatim above as the golden reference; `compute_dual_gpu_partition`
# is now a thin N=2 wrapper over `compute_chain_partition`.
#
# Per-GPU pid layout (general; L/O/T = leading ghost / own / trailing ghost
# pool sizes, slot 0 of the buffer is the sentinel):
#
#     0                     sentinel
#     1 .. L                leading-ghost pid range
#     L+1 .. L+O            own particles
#     L+O+1 .. L+O+T        trailing-ghost pid range
#
# Link offset algebra (the sender pre-encodes receiver pids, so the offset
# is receiver_ghost_range_first - sender_ghost_range_first):
#
#     trailing send (slab i -> slab i+1's leading ghost):
#         sender_first   = L_i + O_i + 1
#         receiver_first = 1
#         offset         = -(L_i + O_i)          <- depends on SENDER layout
#     leading send  (slab i -> slab i-1's trailing ghost):
#         sender_first   = 1
#         receiver_first = L_prev + O_prev + 1
#         offset         = +(L_prev + O_prev)    <- depends on RECEIVER layout
#
# The dual case (L_0 = 0) collapses both to -/+ slot0_own_pool — the legacy
# "offsets depend only on slot 0's pool" rule is the special case of this.
# ============================================================================

import sys as _sys
from dataclasses import dataclass, field


MINIMUM_OWN_COLUMNS_HARD = 12   # < 8 -> force deep-interior empty; 12 = margin
MINIMUM_OWN_COLUMNS_WARN = 20   # below this Phase B's hiding budget is thin


@dataclass
class PidLayout:
    """One slab's pid-pool triple. Offsets are pure functions of these."""
    leading_ghost_pool_size: int
    own_pool_size: int
    trailing_ghost_pool_size: int

    def ghost_range_first(self, direction: str) -> int:
        if direction == "leading":
            return 1
        if direction == "trailing":
            return self.leading_ghost_pool_size + self.own_pool_size + 1
        raise ValueError(f"unknown direction {direction!r}")


@dataclass
class SlabGeometry:
    """Per-slab scalar geometry, computed in pass 1 before any CaseV5 exists."""
    slot_index: int
    own_global_first_column: int        # inclusive, global voxel-x
    own_global_last_column: int         # inclusive
    has_leading_peer: bool
    has_trailing_peer: bool
    own_pool_size: int
    own_particle_count: int

    @property
    def own_column_count(self) -> int:
        return self.own_global_last_column - self.own_global_first_column + 1

    @property
    def leading_thickness(self) -> int:
        return GHOST_THICKNESS if self.has_leading_peer else 0

    @property
    def trailing_thickness(self) -> int:
        return GHOST_THICKNESS if self.has_trailing_peer else 0

    @property
    def extended_column_count(self) -> int:
        return self.own_column_count + self.leading_thickness + self.trailing_thickness

    def pid_layout(self, ghost_pool_per_direction: int) -> "PidLayout":
        return PidLayout(
            leading_ghost_pool_size=(ghost_pool_per_direction
                                     if self.has_leading_peer else 0),
            own_pool_size=self.own_pool_size,
            trailing_ghost_pool_size=(ghost_pool_per_direction
                                      if self.has_trailing_peer else 0),
        )


@dataclass
class LinkSpec:
    """One directed ghost pathway (metadata mirror of the spec constants)."""
    sender_index: int
    receiver_index: int
    direction: str                      # sender-side direction name
    ghost_pid_offset_to_receiver: int
    ghost_voxel_id_offset_to_receiver: int


@dataclass
class ChainPartition:
    slabs: list                         # list[CaseV5], left -> right
    geometry: list                      # list[SlabGeometry], same order
    cuts: list                          # N-1 global voxel-x cut columns
    links: list = field(default_factory=list)   # list[LinkSpec], both directions


def derive_link_pid_offset(sender_layout: PidLayout,
                           receiver_layout: PidLayout,
                           direction: str) -> int:
    """GHOST_PID_OFFSET_TO_RECEIVER for one directed link (header algebra)."""
    receive_side = "leading" if direction == "trailing" else "trailing"
    return (receiver_layout.ghost_range_first(receive_side)
            - sender_layout.ghost_range_first(direction))


def derive_link_voxel_id_offset(sender_geometry: SlabGeometry,
                                receiver_geometry: SlabGeometry,
                                direction: str,
                                voxel_per_x: int) -> int:
    """GHOST_VOXEL_ID_OFFSET_TO_RECEIVER for one directed link.

    Local-x of the sender's boundary column (in ITS extended grid) vs local-x
    of the receiver's ghost column (in ITS extended grid); both grids share
    NY x NZ so the column delta x voxel_per_x is exact in voxel_id space.
    """
    if direction == "trailing":
        sender_boundary_x_local = (sender_geometry.leading_thickness
                                   + sender_geometry.own_column_count - 1)
        receiver_ghost_x_local = 0
    elif direction == "leading":
        sender_boundary_x_local = sender_geometry.leading_thickness
        receiver_ghost_x_local = (receiver_geometry.leading_thickness
                                  + receiver_geometry.own_column_count)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return (receiver_ghost_x_local - sender_boundary_x_local) * voxel_per_x


def _bin_fluid_counts(global_case: CaseV5) -> np.ndarray:
    """Vectorized per-column fluid particle histogram (same result as the
    legacy per-particle loop in compute_k_split, minus the Python time)."""
    grid_nx = global_case.grid.grid_dimension_x
    h = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x
    positions = global_case.initial.positions
    material_group = global_case.initial.material_group

    fluid_groups = np.array(
        [index for index, material in enumerate(global_case.materials)
         if material.kind == KIND_FLUID], dtype=material_group.dtype)
    fluid_mask = np.isin(material_group, fluid_groups)
    x_indices = np.floor(
        (positions[fluid_mask, 0] - origin_x) / h).astype(np.int64)
    np.clip(x_indices, 0, grid_nx - 1, out=x_indices)
    return np.bincount(x_indices, minlength=grid_nx).astype(np.int64)


def compute_chain_cuts(global_case: CaseV5, weights: list[float],
                       minimum_own_columns: int) -> list[int]:
    """N-1 monotonic cut columns from N weights.

    Degenerates EXACTLY to the legacy compute_k_split for N=2 with
    minimum_own_columns=1: same target formula, same searchsorted side,
    same clamp.
    """
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"weights must be positive, got {weights}")
    slab_count = len(weights)
    grid_nx = global_case.grid.grid_dimension_x
    if grid_nx < slab_count * minimum_own_columns:
        raise ValueError(
            f"grid has {grid_nx} columns; {slab_count} slabs need at least "
            f"{slab_count * minimum_own_columns} (minimum_own_columns="
            f"{minimum_own_columns})")

    fluid_counts = _bin_fluid_counts(global_case)
    fluid_total = int(fluid_counts.sum())
    if fluid_total == 0:
        raise ValueError("global case has no fluid particles")
    cumulative = np.cumsum(fluid_counts)
    weight_total = sum(weights)

    cuts: list[int] = []
    cumulative_weight = 0.0
    for weight in weights[:-1]:
        cumulative_weight += weight
        target = max(1, int(fluid_total * (cumulative_weight / weight_total)))
        cuts.append(int(np.searchsorted(cumulative, target, side="left")))

    # Enforce monotonicity + per-slab minimum width (leaving room for the
    # slabs still to come on the right).
    for j in range(len(cuts)):
        low = (cuts[j - 1] if j > 0 else 0) + minimum_own_columns
        high = grid_nx - minimum_own_columns * (slab_count - 1 - j)
        if low > high:
            raise ValueError(
                f"cannot place cut {j}: need [{low}, {high}] with "
                f"minimum_own_columns={minimum_own_columns}")
        cuts[j] = max(low, min(cuts[j], high))
    return cuts


def _sized_pool(particle_count: int, pool_safety: float, workgroup: int,
                global_pool: int) -> int:
    value = int(math.ceil(particle_count * pool_safety))
    value = ((value + workgroup - 1) // workgroup) * workgroup
    return min(value, global_pool)


def _build_chain_slab_case(
    global_case: CaseV5,
    geometry: SlabGeometry,
    left_neighbor: Optional[SlabGeometry],
    right_neighbor: Optional[SlabGeometry],
    ghost_pool_per_direction: int,
) -> CaseV5:
    """General slab builder: endpoint OR interior (both-sided) slabs."""
    h = global_case.physics.smoothing_length
    ny = global_case.grid.grid_dimension_y
    nz = global_case.grid.grid_dimension_z
    voxel_per_x = ny * nz

    grid = GridLayout(
        origin_x=(global_case.grid.origin_x
                  + (geometry.own_global_first_column
                     - geometry.leading_thickness) * h),
        origin_y=global_case.grid.origin_y,
        origin_z=global_case.grid.origin_z,
        grid_dimension_x=geometry.extended_column_count,
        grid_dimension_y=ny,
        grid_dimension_z=nz,
        voxel_order=global_case.grid.voxel_order,
    )
    ghost_grid = GhostGridParams(
        leading_ghost_voxel_count=geometry.leading_thickness * voxel_per_x,
        trailing_ghost_voxel_count=geometry.trailing_thickness * voxel_per_x,
    )

    my_layout = geometry.pid_layout(ghost_pool_per_direction)
    transport = TransportConfig()
    if geometry.has_leading_peer:
        assert left_neighbor is not None
        transport.leading = DirectionalTransportSpec(
            direction=0,
            boundary_voxel_x_local=geometry.leading_thickness,
            ghost_voxel_x_local=0,
            ghost_pid_offset_to_receiver=derive_link_pid_offset(
                my_layout,
                left_neighbor.pid_layout(ghost_pool_per_direction),
                "leading"),
            ghost_voxel_id_offset_to_receiver=derive_link_voxel_id_offset(
                geometry, left_neighbor, "leading", voxel_per_x),
        )
    if geometry.has_trailing_peer:
        assert right_neighbor is not None
        transport.trailing = DirectionalTransportSpec(
            direction=1,
            boundary_voxel_x_local=(geometry.leading_thickness
                                    + geometry.own_column_count - 1),
            ghost_voxel_x_local=geometry.extended_column_count - 1,
            ghost_pid_offset_to_receiver=derive_link_pid_offset(
                my_layout,
                right_neighbor.pid_layout(ghost_pool_per_direction),
                "trailing"),
            ghost_voxel_id_offset_to_receiver=derive_link_voxel_id_offset(
                geometry, right_neighbor, "trailing", voxel_per_x),
        )

    capacities = Capacities(
        max_particles_per_voxel=global_case.capacities.max_particles_per_voxel,
        workgroup_size=global_case.capacities.workgroup_size,
        max_incoming_per_voxel=global_case.capacities.max_incoming_per_voxel,
        own_pool_size=geometry.own_pool_size,
        leading_ghost_pool_size=my_layout.leading_ghost_pool_size,
        trailing_ghost_pool_size=my_layout.trailing_ghost_pool_size,
    )
    initial = _filter_particles_by_x_range(
        global_case,
        x_lo_inclusive=geometry.own_global_first_column,
        x_hi_exclusive=geometry.own_global_last_column + 1,
    )
    return CaseV5(
        physics=global_case.physics,
        numerics=global_case.numerics,
        capacities=capacities,
        grid=grid,
        ghost_grid=ghost_grid,
        transport=transport,
        materials=list(global_case.materials),
        initial=initial,
    )


def _assert_degenerate_global(global_case: CaseV5) -> None:
    assert global_case.ghost_grid.leading_ghost_voxel_count == 0, (
        "global_case must be degenerate; got an already-partitioned slab")
    assert global_case.ghost_grid.trailing_ghost_voxel_count == 0, (
        "global_case must be degenerate; got an already-partitioned slab")
    assert global_case.transport.leading is None
    assert global_case.transport.trailing is None
    assert global_case.capacities.leading_ghost_pool_size == 0
    assert global_case.capacities.trailing_ghost_pool_size == 0


def compute_chain_partition(
    global_case: CaseV5,
    weights: list[float],
    pool_safety: Optional[float] = None,
    *,
    minimum_own_columns: int = MINIMUM_OWN_COLUMNS_HARD,
) -> ChainPartition:
    """Split a degenerate global case into an N-slab 1D chain.

    ``weights[i]`` is slab i's share of the fluid particle count (left ->
    right). Interior slabs carry ghost pools + transport specs on BOTH
    sides. ``pool_safety`` sizes each slab's own pool as in the dual path
    (None = every slab gets the global pool). ``minimum_own_columns``
    guards the cascading-band floor (interior deep-interior work vanishes
    below 2 x force_band = 8 own columns; the N=2 compatibility wrapper
    passes 1 to reproduce legacy clamping).
    """
    _assert_degenerate_global(global_case)
    slab_count = len(weights)
    if slab_count < 1:
        raise ValueError("need at least one weight")
    grid_nx = global_case.grid.grid_dimension_x

    cuts = compute_chain_cuts(global_case, weights, minimum_own_columns)
    boundaries = [0] + cuts + [grid_nx]   # slab i owns [boundaries[i], boundaries[i+1])

    # Pass 1: per-slab scalar geometry (pools need particle counts first).
    global_pool = global_case.capacities.own_pool_size
    workgroup = global_case.capacities.workgroup_size
    geometry: list[SlabGeometry] = []
    for index in range(slab_count):
        first, last = boundaries[index], boundaries[index + 1] - 1
        own_particles = _filter_particles_by_x_range(
            global_case, first, last + 1).positions.shape[0]
        if pool_safety is None:
            own_pool = global_pool
        else:
            if pool_safety <= 1.0:
                raise ValueError(f"pool_safety must be > 1.0, got {pool_safety}")
            own_pool = _sized_pool(own_particles, pool_safety, workgroup,
                                   global_pool)
        geometry.append(SlabGeometry(
            slot_index=index,
            own_global_first_column=first,
            own_global_last_column=last,
            has_leading_peer=index > 0,
            has_trailing_peer=index < slab_count - 1,
            own_pool_size=own_pool,
            own_particle_count=int(own_particles),
        ))
        if geometry[-1].own_column_count < MINIMUM_OWN_COLUMNS_WARN:
            print(f"[partition_v5] WARN slab {index}: only "
                  f"{geometry[-1].own_column_count} own columns — Phase B "
                  f"hiding budget is thin (warn threshold "
                  f"{MINIMUM_OWN_COLUMNS_WARN})", file=_sys.stderr)

    # Pass 2: build cases with neighbor geometry in hand.
    ghost_pool_per_direction = _ghost_pool_size(global_case)
    slabs = []
    links: list[LinkSpec] = []
    for index, slab_geometry in enumerate(geometry):
        left = geometry[index - 1] if index > 0 else None
        right = geometry[index + 1] if index < slab_count - 1 else None
        case = _build_chain_slab_case(
            global_case, slab_geometry, left, right, ghost_pool_per_direction)
        slabs.append(case)
        if case.transport.trailing is not None:
            links.append(LinkSpec(
                sender_index=index, receiver_index=index + 1,
                direction="trailing",
                ghost_pid_offset_to_receiver=(
                    case.transport.trailing.ghost_pid_offset_to_receiver),
                ghost_voxel_id_offset_to_receiver=(
                    case.transport.trailing.ghost_voxel_id_offset_to_receiver)))
        if case.transport.leading is not None:
            links.append(LinkSpec(
                sender_index=index, receiver_index=index - 1,
                direction="leading",
                ghost_pid_offset_to_receiver=(
                    case.transport.leading.ghost_pid_offset_to_receiver),
                ghost_voxel_id_offset_to_receiver=(
                    case.transport.leading.ghost_voxel_id_offset_to_receiver)))

    column_spans = ", ".join(
        f"[{g.own_global_first_column},{g.own_global_last_column + 1})"
        for g in geometry)
    print(f"[partition_v5] chain N={slab_count}: columns {column_spans} "
          f"of {grid_nx}; particles "
          + ", ".join(f"{g.own_particle_count:,}" for g in geometry))
    return ChainPartition(slabs=slabs, geometry=geometry, cuts=cuts,
                          links=links)


def compute_dual_gpu_partition(
    global_case: CaseV5,
    weights: list[float],
    pool_safety: Optional[float] = None,
) -> tuple[CaseV5, CaseV5, int]:
    """N=2 compatibility wrapper over ``compute_chain_partition``.

    Same signature/return as the legacy entry point (all runners keep
    working unchanged). ``minimum_own_columns=1`` reproduces the legacy
    clamp semantics exactly; ``_test_partition_chain.py`` asserts
    field-by-field equality against ``legacy_dual_gpu_partition``.
    """
    if len(weights) != 2:
        raise NotImplementedError(
            "compute_dual_gpu_partition is the 2-GPU wrapper; use "
            "compute_chain_partition for N != 2")
    chain = compute_chain_partition(global_case, weights, pool_safety,
                                    minimum_own_columns=1)
    return chain.slabs[0], chain.slabs[1], chain.cuts[0]


def isolate_slab(global_case: CaseV5, chain: ChainPartition,
                 slab_index: int) -> CaseV5:
    """The η_weak helper: slab ``slab_index``'s own subdomain as a standalone
    no-peer case (ghost pools 0, transport empty, grid = own columns only).

    Keeps the chain slab's own_pool_size so per-kernel dispatch counts match
    the in-chain slab — the single-run reference then isolates pure
    coordination overhead. See docs/sph_v5_design.md §1.3 / roadmap η_weak."""
    source = chain.geometry[slab_index]
    isolated = SlabGeometry(
        slot_index=0,
        own_global_first_column=source.own_global_first_column,
        own_global_last_column=source.own_global_last_column,
        has_leading_peer=False,
        has_trailing_peer=False,
        own_pool_size=source.own_pool_size,
        own_particle_count=source.own_particle_count,
    )
    return _build_chain_slab_case(global_case, isolated, None, None,
                                  ghost_pool_per_direction=0)
