"""
partition_v6.py — V5 dual-GPU static 1D X-axis partition.

Input contract: a **degenerate** CaseV6 from ``case_loader_v6.load_case_v6``
— owns the whole domain, no peer, ghost_grid = (0, 0), transport = empty.
``compute_dual_gpu_partition`` asserts this at entry to keep the loader →
partition seam honest.

Given that input, computes:
  - K_split_voxel_x        (the column where domain boundary sits)
  - per-GPU slab CaseV6    (own column range + leading/trailing ghost +
                            per-direction transport spec)

Per-direction transport spec offsets match ``shaders/ghost_send.comp``'s
spec const semantics (see Option B header in that file).

V5 v1.0: 2 GPUs only, static partition (V3+ generalizes to N + dynamic).
"""

from __future__ import annotations

import copy
import math
import os
import sys
from typing import Optional

import numpy as np

from experiment.v6.utils.case_v6 import (
    CaseV6,
    Capacities,
    DirectionalTransportSpec,
    GhostGridParams,
    GridLayout,
    InitialParticles,
    KIND_FLUID,
    TransportConfig,
)


GHOST_THICKNESS = 1   # V5 v1.0: 1-voxel-thick ghost on the interior side (legacy dual path)


# ============================================================================
# V6 seam switches (read at partition time; the simulator takes everything
# from the per-slab CaseV6 these functions build).
#
#   V6_GHOST_LAYERS=1|2      ghost thickness in voxel columns per peer side.
#                            1 = V5. 2 = the peer's seam column (inner layer,
#                            "G1") + the column behind it (outer layer, "G2"):
#                            phase C then recomputes correction + density for
#                            G1 locally so the seam column's force reads the
#                            neighbours' rho/P of THIS step. Requires
#                            V6_KEEP_DEPARTED=1 and band-voxel dispatch.
#   V6_KEEP_DEPARTED=0|1     keep a copy of this frame's outgoing migrants in a
#                            local departed pool so the sender's seam column
#                            still sees them as ghost neighbours this step.
#   V6_DEPARTED_CAPACITY=<n> departed pool slots per slab (overrides the
#                            face-fraction default below).
#   V6_DEPARTED_FACE_FRACTION=<f>  default capacity = max(64, ceil(f x face
#                            voxels x peer sides)); f = 0.25 by default. The
#                            seam crossing rate per frame scales with the face
#                            area, not with NY*NZ*MAX_INCOMING_PER_VOXEL.
#   V6_LEAN_TRANSPORT=1      every ghost packet carries 4 fields (position +
#                            vid, velocity + mass, rho / P, material) instead
#                            of 9; see common.glsl LEAN_TRANSPORT.
#   V6_TRANSPORT_EXTENSION=1 lean packets also carry extension_fields (seam
#                            audit ids; forced on by experiment/seam_audit).
#   V6_MIGRANT_POOL_FACTOR=<f> V6_GHOST_LAYERS=2: migrant region = max(64,
#                            ceil(face x MAX_INCOMING x f)) slots per direction;
#                            default f = V6_GHOST_POOL_FACTOR (the old sizing).
#   V6_COMPACT_GHOST_LISTS=1 ghost voxel lists travel as (count, first pid) and
#                            are rebuilt on the receiver (expand_ghost_lists).
# ============================================================================

DEPARTED_CAPACITY_FLOOR = 64
MIGRANT_REGION_FLOOR = 64      # V6_GHOST_LAYERS = 2 migrant region, slots per direction

# E6b (2026-10-05): the code defaults are the recommended release set of
# docs/seam_audit/v6_opt.md; every switch keeps its old value as an option.
# Pool factors per dimension (2-D: C = 96 / C_inc = 16, h/dx = 5; 3-D: 128 / 32,
# h/dx = 4; other capacities need their own factors, see v6_opt.md (e)).
RELEASE_POOL_DEFAULTS = {
    2: {"V6_GHOST_POOL_FACTOR": 0.29, "V6_MIGRANT_POOL_FACTOR": 0.05, "V6_DEPARTED_FACE_FRACTION": 0.8},
    3: {"V6_GHOST_POOL_FACTOR": 0.5, "V6_MIGRANT_POOL_FACTOR": 0.02, "V6_DEPARTED_FACE_FRACTION": 0.64},
}
# The pre-E6b defaults, for tools that pin an old baseline (the seam_audit tools
# name their switches on top of these). An unset V6_MIGRANT_POOL_FACTOR follows an
# explicitly set V6_GHOST_POOL_FACTOR (the old rule), so it needs no entry.
LEGACY_DEFAULTS = {
    "V6_KEEP_DEPARTED": "0", "V6_GHOST_LAYERS": "1", "V6_LEAN_TRANSPORT": "0",
    "V6_GHOST_POOL_FACTOR": "1", "V6_DEPARTED_FACE_FRACTION": "0.25",
    "V6_COMPACT_GHOST_LISTS": "0", "V6_PACKED_REPLICAS": "0", "V6_BAND_SLOT_LANES": "0",
    "V6_INIT_SEAM_CLAMP": "0", "V6_BAND_WIDTHS": "2,3,4",
    "V6_WORKER_COUNT_AWARE": "0", "V6_SPLIT_TRANSFER_QUEUES": "0",
}


def _release_pool_default(global_case: CaseV6, name: str) -> float:
    dimension = 3 if int(global_case.physics.dimension) == 3 else 2
    return RELEASE_POOL_DEFAULTS[dimension][name]


def configured_ghost_pool_factor(global_case: CaseV6) -> float:
    """V6_GHOST_POOL_FACTOR (default 0.29 in 2-D, 0.5 in 3-D; before E6b 1)."""
    text = os.environ.get("V6_GHOST_POOL_FACTOR")
    return float(text) if text is not None else _release_pool_default(global_case, "V6_GHOST_POOL_FACTOR")


def configured_migrant_pool_factor(global_case: CaseV6) -> float:
    """V6_MIGRANT_POOL_FACTOR (default 0.05 in 2-D, 0.02 in 3-D). Unset while
    V6_GHOST_POOL_FACTOR is set: the ghost factor, as before E6b."""
    text = os.environ.get("V6_MIGRANT_POOL_FACTOR")
    if text is not None:
        return float(text)
    if os.environ.get("V6_GHOST_POOL_FACTOR") is not None:
        return configured_ghost_pool_factor(global_case)
    return _release_pool_default(global_case, "V6_MIGRANT_POOL_FACTOR")


def configured_departed_face_fraction(global_case: CaseV6) -> float:
    """V6_DEPARTED_FACE_FRACTION (default 0.8 in 2-D, 0.64 in 3-D; before E6b 0.25)."""
    text = os.environ.get("V6_DEPARTED_FACE_FRACTION")
    return float(text) if text is not None else _release_pool_default(global_case, "V6_DEPARTED_FACE_FRACTION")


def configured_ghost_layers() -> int:
    """V6_GHOST_LAYERS (default 2; 1 when V6_KEEP_DEPARTED=0 or V6_BAND_VOXEL_DISPATCH=0
    is set without a layer count: the v5 seam (0,1), or a run without the band-voxel
    dispatch that two layers need)."""
    two_layers = configured_keep_departed() and os.environ.get("V6_BAND_VOXEL_DISPATCH", "1") == "1"
    layers = int(os.environ.get("V6_GHOST_LAYERS", "2" if two_layers else "1"))
    if layers not in (1, 2):
        raise ValueError(f"V6_GHOST_LAYERS={layers}: only 1 or 2 are supported")
    if layers == 2 and not configured_keep_departed():
        raise ValueError("V6_GHOST_LAYERS=2 requires V6_KEEP_DEPARTED=1 (the inner ghost "
                         "layer is recomputed as self; without the departed migrants its "
                         "neighbour set misses this frame's arrivals)")
    return layers


def configured_keep_departed() -> bool:
    """V6_KEEP_DEPARTED (default 1)."""
    return os.environ.get("V6_KEEP_DEPARTED", "1") == "1"


def configured_lean_transport() -> bool:
    """V6_LEAN_TRANSPORT=1 (default): ghost packets carry 4 fields (common.glsl id 87)."""
    return os.environ.get("V6_LEAN_TRANSPORT", "1") == "1"


def configured_transport_extension() -> bool:
    """V6_TRANSPORT_EXTENSION=1: lean packets also carry extension_fields
    (common.glsl id 88). The seam-audit workers force it on (global ids)."""
    return os.environ.get("V6_TRANSPORT_EXTENSION", "0") == "1"


def configured_init_seam_clamp() -> bool:
    """V6_INIT_SEAM_CLAMP=1 (default): initialize_voxelization keeps an own particle that
    lands one column into a ghost column in the adjacent own column (common.glsl
    id 97) instead of losing it at the bootstrap."""
    return os.environ.get("V6_INIT_SEAM_CLAMP", "1") == "1"


def configured_compact_ghost_lists() -> bool:
    """V6_COMPACT_GHOST_LISTS=1 (default): the transport ships one first-pid word per
    ghost voxel instead of the inside_particle_index rows (common.glsl id 89)."""
    return os.environ.get("V6_COMPACT_GHOST_LISTS", "1") == "1"


def configured_ghost_self_kernels() -> tuple[str, ...]:
    """V6_DIAG_GHOST_SELF (diagnostic, default "correction,density"): the band
    kernels that recompute the inner ghost column (G1) as self."""
    return tuple(name.strip() for name in
                 os.environ.get("V6_DIAG_GHOST_SELF", "correction,density").split(",")
                 if name.strip())


def configured_packed_replicas() -> bool:
    """V6_PACKED_REPLICAS=1: two-layer replicas travel packed, G1 and G2 the
    same 32 B record (common.glsl id 98). Needs V6_GHOST_LAYERS=2 and
    V6_COMPACT_GHOST_LISTS=1: with one ghost layer there is no packed region and
    expand_ghost_lists would unpack garbage over every inbound replica, so the
    combination is rejected here (every reader of the switch goes through this
    function). No pressure travels: G1's P is rebuilt by the density band's
    G1-as-self pass before force reads it, so V6_DIAG_GHOST_SELF must keep
    'density' (without it force would read P = 0 at the seam).
    Unset (E6b): on wherever it is valid, i.e. with two ghost layers, compact
    lists and the density G1 pass; off otherwise."""
    text = os.environ.get("V6_PACKED_REPLICAS")
    if text is None:
        return (configured_ghost_layers() == 2 and configured_compact_ghost_lists()
                and "density" in configured_ghost_self_kernels())
    packed = text == "1"
    if packed and (configured_ghost_layers() != 2 or not configured_compact_ghost_lists()):
        raise ValueError("V6_PACKED_REPLICAS=1 needs V6_GHOST_LAYERS=2 and V6_COMPACT_GHOST_LISTS=1")
    if packed and "density" not in configured_ghost_self_kernels():
        raise ValueError("V6_PACKED_REPLICAS=1 ships no G1 pressure; V6_DIAG_GHOST_SELF must include "
                         "'density' (the G1-as-self density pass rebuilds it before force)")
    return packed


DIAGNOSTIC_POISON_VALUES = {"": 0, "0": 0, "off": 0, "pressure": 1, "density": 2}


def configured_diagnostic_poison_inner_replica() -> int:
    """V6_DIAG_POISON_G1=off|pressure|density (diagnostic only, spec 99, declared in
    expand_ghost_lists.comp): the unpack writes a quiet NaN into the inner (G1) replica's
    pressure (nothing may read it) or density (negative control: it is read). Only the
    packed unpack path can poison, so a non-off value needs V6_PACKED_REPLICAS=1 (without
    it the poison would be a silent no-op)."""
    text = os.environ.get("V6_DIAG_POISON_G1", "off").strip().lower()
    if text not in DIAGNOSTIC_POISON_VALUES:
        raise ValueError(f"V6_DIAG_POISON_G1={text!r}: expected off, pressure or density")
    value = DIAGNOSTIC_POISON_VALUES[text]
    if value and not configured_packed_replicas():
        raise ValueError("V6_DIAG_POISON_G1 needs V6_PACKED_REPLICAS=1 (it poisons the packed unpack)")
    return value


def configured_delta_density() -> bool:
    """V6_DELTA_DENSITY=1 (evaluation): density_pressure.x stores rho - rho_ref
    (common.glsl ids 95 / 96)."""
    return os.environ.get("V6_DELTA_DENSITY", "0") == "1"


RELEASE_BAND_WIDTHS = (2, 2, 3)            # the code default (E6b)
COMPACT_DISPATCH_BAND_WIDTHS = (2, 3, 4)   # the only widths V6_BAND_COMPACT_DISPATCH supports


def configured_band_widths() -> tuple[int, int, int]:
    """V6_BAND_WIDTHS="c,d,f" (default "2,2,3"; "2,3,4" with V6_BAND_COMPACT_DISPATCH=1,
    the only widths its band list supports): boundary band widths in own voxel
    columns of correction / density / force = NEIGHBOR_X_RANGE (spec 82) of each
    kernel's split pipelines. Phase B runs the interior variants (every column from
    the band on), phase C the bands. In phase B own columns 0 and 1 lack inputs
    (this frame's migrants land in column 0 and the ghosts arrive in phase C), so a
    column's correction is final from column 2 on: c >= 2. Density reads its own L
    and the neighbours' r, v, m, rho_n and material only: d >= c. Force reads the
    neighbours' rho_{n+1} / P_{n+1}: f >= d + 1. Anything else is rejected."""
    default = (COMPACT_DISPATCH_BAND_WIDTHS if os.environ.get("V6_BAND_COMPACT_DISPATCH", "0") == "1"
               else RELEASE_BAND_WIDTHS)
    text = os.environ.get("V6_BAND_WIDTHS", ",".join(map(str, default)))
    try:
        widths = tuple(int(part) for part in text.split(","))
    except ValueError:
        widths = ()
    if len(widths) != 3:
        raise ValueError(f"V6_BAND_WIDTHS={text!r}: expected three integers 'correction,density,force'")
    correction, density, force = widths
    if correction < 2 or density < correction or force < density + 1:
        raise ValueError(f"V6_BAND_WIDTHS={text!r}: needs correction >= 2, density >= correction and "
                         "force >= density + 1")
    return widths


def transported_particle_fields() -> tuple[str, ...]:
    """SoA fields of a migrant packet (and of every slot of the V5 mixed pool).
    V5 / lean off: the nine defrag fields. Lean: the four fields the receiver
    reads, + extension_fields when V6_TRANSPORT_EXTENSION=1."""
    if not configured_lean_transport():
        return ("position_voxel_id", "density_pressure", "velocity_mass",
                "acceleration", "shift", "material", "correction_inverse",
                "density_gradient_kernel_sum", "extension_fields")
    # same relative order as the full layout (set 0 binding order), so the
    # lean staging is the full staging with the dead segments removed
    fields = ("position_voxel_id", "density_pressure", "velocity_mass", "material")
    if configured_transport_extension():
        fields += ("extension_fields",)
    return fields


def _departed_pool_size(global_case: CaseV6, peer_side_count: int) -> int:
    """Departed pool slots for one slab (0 when V6_KEEP_DEPARTED is off or the
    slab has no peer). Sized from the seam face, not from the worst case."""
    if not configured_keep_departed() or peer_side_count == 0:
        return 0
    override = os.environ.get("V6_DEPARTED_CAPACITY")
    if override:
        return max(1, int(override))
    face_voxel_count = (global_case.grid.grid_dimension_y
                        * global_case.grid.grid_dimension_z)
    fraction = configured_departed_face_fraction(global_case)
    return max(DEPARTED_CAPACITY_FLOOR,
               int(math.ceil(fraction * face_voxel_count * peer_side_count)))


def _ghost_pool_layout(global_case: CaseV6, ghost_layers: int) -> tuple[int, int]:
    """(pool slots per direction, replica region size) for one ghost direction.

    ghost_layers = 1: the V5 mixed pool (_ghost_pool_size), region size 0.
    ghost_layers = 2: [inner replicas R | outer replicas R | migrants M] with
      R = the V5 per-direction pool (one column's replicas + its migrant share,
          scaled by V6_GHOST_POOL_FACTOR exactly like V5), M = face x
          MAX_INCOMING_PER_VOXEL x the same factor. Replicas only carry 4 fields
          over the link, migrants all 9 (simulator transport segments).
    """
    if ghost_layers == 1:
        return _ghost_pool_size(global_case), 0
    voxel_per_x = global_case.grid.grid_dimension_y * global_case.grid.grid_dimension_z
    factor = configured_ghost_pool_factor(global_case)
    # V6_MIGRANT_POOL_FACTOR: the migrant region's own scale. A frame's migrants
    # are a few per seam row (measured in docs/seam_audit/v6_opt.md), far below
    # one column's replicas, so the ghost factor over-sizes this region ~100x;
    # every slot is DMA'd.
    migrant_factor = configured_migrant_pool_factor(global_case)
    replica_region = int(math.ceil(
        voxel_per_x * (global_case.capacities.max_particles_per_voxel
                       + global_case.capacities.max_incoming_per_voxel) * factor))
    migrant_region = max(MIGRANT_REGION_FLOOR, int(math.ceil(
        voxel_per_x * global_case.capacities.max_incoming_per_voxel * migrant_factor)))
    _warn_if_replica_region_tight(global_case, replica_region / voxel_per_x, factor)
    return 2 * replica_region + migrant_region, replica_region


def _warn_if_replica_region_tight(global_case: CaseV6, slots_per_voxel: float, factor: float) -> None:
    """A ghost column holds ~(h/dx)^d particles per voxel; the replica region
    reserves (C + C_inc) * f slots per voxel. Measured peaks reached 0.2250 (2-D,
    h/dx 5, C + C_inc = 112) and 0.3891 (3-D, h/dx 4, 160) of the f = 1 slots,
    i.e. 0.97-1.01 x (h/dx)^d; below 1.2 x (h/dx)^d the region has < 20 % headroom
    (the release factors were derived for those capacities; other C / h/dx need
    their own factor). Warning only: an overflow is counted, never silent."""
    radii = [float(material.radius) for material in global_case.materials if float(material.radius) > 0]
    if not radii:
        return
    spacing = 2.0 * min(radii)
    ratio = global_case.physics.smoothing_length / spacing
    expected = ratio ** global_case.physics.dimension
    if slots_per_voxel < 1.2 * expected:
        print(f"[partition_v6] WARNING: replica region {slots_per_voxel:.1f} slots per voxel "
              f"(V6_GHOST_POOL_FACTOR={factor}) < 1.2 x (h/dx)^d = {1.2 * expected:.1f}; "
              f"expect overflow_ghost_count > 0", file=sys.stderr, flush=True)


def _ghost_pool_size(case: CaseV6) -> int:
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
    pool = voxel_per_x * (case.capacities.max_particles_per_voxel
                          + case.capacities.max_incoming_per_voxel)
    # 2026-09-15 experiment (N56 64M K=8 utilization): the worst-case pool is
    # ~10x the live ghost count and every readback/upload DMA moves the full
    # pool. V6_GHOST_POOL_FACTOR=<0..1> scales the pool (overflow_ghost is
    # reported per interval if the bound is ever exceeded).
    factor = configured_ghost_pool_factor(case)
    if factor != 1.0:
        pool = int(math.ceil(pool * factor))
        print(f"[partition_v6] V6_GHOST_POOL_FACTOR={factor}: ghost pool per direction {pool:,}")
    return pool


def compute_k_split(global_case: CaseV6, weights: list[float]) -> int:
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
    global_case: CaseV6,
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
    global_case: CaseV6,
    slot_index: int,
    k_split: int,
    grid_nx: int,
    own_pool_size: int,
    slot0_own_pool: int,
) -> CaseV6:
    """Build a per-GPU CaseV6 for slot 0 (leftmost) or slot 1 (rightmost).

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

    return CaseV6(
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
    global_case: CaseV6,
    weights: list[float],
    pool_safety: Optional[float] = None,
) -> tuple[CaseV6, CaseV6, int]:
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
    # Contract check — input must be a degenerate slab from load_case_v6.
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
    print(f"[partition_v6] K_split = {k_split} / {grid_nx} "
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
        print(f"[partition_v6] pool_safety={pool_safety}: "
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


MINIMUM_OWN_COLUMNS_HARD = 12   # < 2 x force band (8 at the default V6_BAND_WIDTHS) -> force deep-interior empty; 12 = margin
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
    """Per-slab scalar geometry, computed in pass 1 before any CaseV6 exists."""
    slot_index: int
    own_global_first_column: int        # inclusive, global voxel-x
    own_global_last_column: int         # inclusive
    has_leading_peer: bool
    has_trailing_peer: bool
    own_pool_size: int
    own_particle_count: int
    ghost_layers: int = GHOST_THICKNESS     # V6_GHOST_LAYERS

    @property
    def own_column_count(self) -> int:
        return self.own_global_last_column - self.own_global_first_column + 1

    @property
    def leading_thickness(self) -> int:
        return self.ghost_layers if self.has_leading_peer else 0

    @property
    def trailing_thickness(self) -> int:
        return self.ghost_layers if self.has_trailing_peer else 0

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
    slabs: list                         # list[CaseV6], left -> right
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
    of the receiver's INNERMOST ghost column (the one adjacent to its own
    columns, in ITS extended grid); both grids share NY x NZ so the column
    delta x voxel_per_x is exact in voxel_id space. The offset is a pure
    translation of the x index, so the same value maps every sender column
    near the seam (own boundary -> receiver inner ghost, own boundary-1 ->
    outer ghost, sender inner ghost -> receiver own boundary column, ...).
    With one ghost column (V5) the innermost leading ghost is local x 0.
    """
    if direction == "trailing":
        sender_boundary_x_local = (sender_geometry.leading_thickness
                                   + sender_geometry.own_column_count - 1)
        receiver_ghost_x_local = receiver_geometry.leading_thickness - 1
    elif direction == "leading":
        sender_boundary_x_local = sender_geometry.leading_thickness
        receiver_ghost_x_local = (receiver_geometry.leading_thickness
                                  + receiver_geometry.own_column_count)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return (receiver_ghost_x_local - sender_boundary_x_local) * voxel_per_x


def _bin_fluid_counts(global_case: CaseV6) -> np.ndarray:
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


def compute_chain_cuts(global_case: CaseV6, weights: list[float],
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
    global_case: CaseV6,
    geometry: SlabGeometry,
    left_neighbor: Optional[SlabGeometry],
    right_neighbor: Optional[SlabGeometry],
    ghost_pool_per_direction: int,
    replica_region_size: int = 0,
    departed_pool_size: int = 0,
) -> CaseV6:
    """General slab builder: endpoint OR interior (both-sided) slabs.

    V6: ``replica_region_size`` > 0 selects the split ghost-pool layout of
    V6_GHOST_LAYERS = 2; ``departed_pool_size`` sizes the V6_KEEP_DEPARTED
    pool. Both 0 reproduce V5 exactly."""
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
        ghost_layers=geometry.ghost_layers,
    )

    my_layout = geometry.pid_layout(ghost_pool_per_direction)
    transport = TransportConfig()
    # ghost_voxel_x_local = the INNERMOST ghost column on the send side (the
    # one adjacent to the boundary column); with one ghost column this is
    # local x 0 (leading) / extended_nx - 1 (trailing), exactly as in V5.
    if geometry.has_leading_peer:
        assert left_neighbor is not None
        transport.leading = DirectionalTransportSpec(
            direction=0,
            boundary_voxel_x_local=geometry.leading_thickness,
            ghost_voxel_x_local=geometry.leading_thickness - 1,
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
            ghost_voxel_x_local=(geometry.leading_thickness
                                 + geometry.own_column_count),
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
        departed_pool_size=departed_pool_size,
        replica_region_size=replica_region_size,
    )
    initial = _filter_particles_by_x_range(
        global_case,
        x_lo_inclusive=geometry.own_global_first_column,
        x_hi_exclusive=geometry.own_global_last_column + 1,
    )
    return CaseV6(
        physics=global_case.physics,
        numerics=global_case.numerics,
        capacities=capacities,
        grid=grid,
        ghost_grid=ghost_grid,
        transport=transport,
        materials=list(global_case.materials),
        initial=initial,
    )


def _assert_degenerate_global(global_case: CaseV6) -> None:
    assert global_case.ghost_grid.leading_ghost_voxel_count == 0, (
        "global_case must be degenerate; got an already-partitioned slab")
    assert global_case.ghost_grid.trailing_ghost_voxel_count == 0, (
        "global_case must be degenerate; got an already-partitioned slab")
    assert global_case.transport.leading is None
    assert global_case.transport.trailing is None
    assert global_case.capacities.leading_ghost_pool_size == 0
    assert global_case.capacities.trailing_ghost_pool_size == 0


def compute_chain_partition(
    global_case: CaseV6,
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
    ghost_layers = configured_ghost_layers()

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
            ghost_layers=ghost_layers,
        ))
        if geometry[-1].own_column_count < MINIMUM_OWN_COLUMNS_WARN:
            print(f"[partition_v6] WARN slab {index}: only "
                  f"{geometry[-1].own_column_count} own columns — Phase B "
                  f"hiding budget is thin (warn threshold "
                  f"{MINIMUM_OWN_COLUMNS_WARN})", file=_sys.stderr)

    # Pass 2: build cases with neighbor geometry in hand.
    ghost_pool_per_direction, replica_region_size = _ghost_pool_layout(
        global_case, ghost_layers)
    slabs = []
    links: list[LinkSpec] = []
    for index, slab_geometry in enumerate(geometry):
        left = geometry[index - 1] if index > 0 else None
        right = geometry[index + 1] if index < slab_count - 1 else None
        peer_side_count = int(slab_geometry.has_leading_peer) + int(slab_geometry.has_trailing_peer)
        case = _build_chain_slab_case(
            global_case, slab_geometry, left, right, ghost_pool_per_direction,
            replica_region_size=replica_region_size,
            departed_pool_size=_departed_pool_size(global_case, peer_side_count))
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
    print(f"[partition_v6] chain N={slab_count}: columns {column_spans} "
          f"of {grid_nx}; particles "
          + ", ".join(f"{g.own_particle_count:,}" for g in geometry))
    if slab_count > 1:
        print(f"[partition_v6] seam: ghost_layers={ghost_layers} "
              f"keep_departed={int(configured_keep_departed())} ghost pool/direction="
              f"{ghost_pool_per_direction:,}"
              + (f" (replica regions 2 x {replica_region_size:,} + migrants "
                 f"{ghost_pool_per_direction - 2 * replica_region_size:,})"
                 if replica_region_size else " (V5 mixed layout)")
              + " departed pool/slab="
              + "/".join(str(s.capacities.departed_pool_size) for s in slabs))
    return ChainPartition(slabs=slabs, geometry=geometry, cuts=cuts,
                          links=links)


def restart_slab_rows(global_case: CaseV6, chain: ChainPartition,
                      positions_x: np.ndarray) -> list[np.ndarray]:
    """Row indices of a saved step-boundary state owned by each slab of
    ``chain``: the x-column test of _filter_particles_by_x_range (same
    expression and clamp), so a restart partitions a saved state exactly
    the way the case loader partitions the initial condition. Every row is
    owned by exactly one slab."""
    h = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x
    x_indices = np.floor((np.asarray(positions_x) - origin_x) / h).astype(np.int64)
    np.clip(x_indices, 0, global_case.grid.grid_dimension_x - 1, out=x_indices)
    rows = []
    for geometry in chain.geometry:
        mask = ((x_indices >= geometry.own_global_first_column)
                & (x_indices < geometry.own_global_last_column + 1))
        rows.append(np.flatnonzero(mask))
    if sum(part.size for part in rows) != x_indices.size:
        raise AssertionError("restart_slab_rows: the slabs do not cover every row")
    return rows


def compute_dual_gpu_partition(
    global_case: CaseV6,
    weights: list[float],
    pool_safety: Optional[float] = None,
) -> tuple[CaseV6, CaseV6, int]:
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


def isolate_slab(global_case: CaseV6, chain: ChainPartition,
                 slab_index: int) -> CaseV6:
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
