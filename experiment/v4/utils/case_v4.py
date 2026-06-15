"""
case_v4.py — V4 simulation case dataclasses + minimal test factory.

V4 isolation: no import from utils/sph/ or experiment/v1/. The dataclasses
here mirror what V1's Case carries but with the structure flattened for V4's
needs (per-GPU view; orchestrator builds two CaseV4 objects, one per slab).

For Phase 2 bring-up we only need *enough* state to:
  (1) size buffers (Capacities + GridLayout + GhostGridParams)
  (2) build pipelines (PhysicsConstants + NumericsConstants + Capacities + TransportConfig)
  (3) build descriptors (numbers of buffers per set)

Initial particle state (positions / velocities / materials) is collected here
too but only consumed in Phase 3 bootstrap. The minimal test case factory
generates an empty cavity (no particles yet) — Phase 2 doesn't run bootstrap,
so it doesn't need particles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================================
# Material kinds (mirror common.glsl)
# ============================================================================

KIND_FLUID    = 0
KIND_BOUNDARY = 1
KIND_INLET    = 2
KIND_ROTOR    = 3


# ============================================================================
# Per-shader parameter bundles
# ============================================================================

@dataclass
class PhysicsConstants:
    """Spec const ids 0-2, 4-6, 17-19, 30-33 in common.glsl. (id=3 reserved.)

    CPU-only fields (NOT packed into spec consts): ``lattice``,
    ``calibrate_volume``. Consumed at load time by
    ``case_loader_v4._calibrate_particle_volume`` to derive each material's
    ``volume`` so that interior ``Σ V·W(r) = 1`` in the bulk.
    """
    smoothing_length: float          # h
    speed_of_sound: float            # c0
    delta_coefficient: float         # δ
    power_parameter: float           # γ
    cfl_number: float
    timestep: float                  # dt
    gravity: tuple[float, float, float]
    dimension: int                   # 2 or 3
    neighbor_z_range: int            # 0 for 2D, 1 for 3D
    kernel_coefficient: float
    kernel_gradient_coefficient: float
    lattice: str = "grid"            # CPU-only; {"grid", "hex"}; preprocessor's particle layout
    calibrate_volume: bool = True    # CPU-only; True → partition-of-unity calibrated V; False → (2·r)^dim


@dataclass
class NumericsConstants:
    """Spec const ids 14-16, 40-46 + ablation toggles."""
    regularization_xi: float
    regularization_determinant_threshold: float
    regularization_max_frobenius_norm: float
    eps_h_squared: float
    pst_main_shift_coefficient: float
    pst_anti_shift_coefficient: float
    use_kcg_correction: bool = True
    use_density_diffusion: bool = True
    use_pst: bool = True
    use_prefix_sum_defrag: bool = False
    defrag_cadence: int = 1000


@dataclass
class Capacities:
    """Particle pool sizing (spec const ids 50-55).

    Pool sizes are particle slot counts (not bytes). The pid layout is:
      0              — sentinel (unused, contributes the +1 in total_pool_capacity)
      [1, L]         — leading ghost  (L = leading_ghost_pool_size)
      [L+1, L+O]     — own particles  (O = own_pool_size)
      [L+O+1, L+O+T] — trailing ghost (T = trailing_ghost_pool_size)
    End-of-chain GPUs set the non-peer side's ghost pool size to 0.
    """
    max_particles_per_voxel: int
    workgroup_size: int
    max_incoming_per_voxel: int
    own_pool_size: int
    leading_ghost_pool_size: int
    trailing_ghost_pool_size: int

    def total_pool_capacity(self) -> int:
        """Total particle slot count = slot 0 unused + leading + own + trailing."""
        return (1
                + self.leading_ghost_pool_size
                + self.own_pool_size
                + self.trailing_ghost_pool_size)


@dataclass
class GridLayout:
    """Per-GPU extended grid layout (spec const ids 7-9, 11-13, 20).

    All fields are per-GPU; orchestrator builds a different GridLayout per slab.

    ``grid_dimension_x`` is the EXTENDED nx = own_x_count + leading_x_thickness
    + trailing_x_thickness. Voxel-x 0 is THIS GPU's leftmost extended column —
    a leading ghost column when a leading peer exists, otherwise own's first
    column. ``origin_x/y/z`` is the world-space position of voxel (0,0,0); see
    ``partition_v4._build_slab_case`` for the per-slab shift formula.

    Voxel_id encoding (helpers.glsl) is 1-based over the extended grid:
        [1, M]                          — leading ghost voxels  (M = LEADING_GHOST_VOXEL_COUNT)
        [M+1, EXTENDED_TOTAL - N]       — own voxels
        [EXTENDED_TOTAL - N + 1, TOTAL] — trailing ghost voxels (N = TRAILING_GHOST_VOXEL_COUNT)
    """
    origin_x: float
    origin_y: float
    origin_z: float
    grid_dimension_x: int            # extended (own + ghost columns)
    grid_dimension_y: int
    grid_dimension_z: int
    voxel_order: int = 0             # 0 = linear z-major

    def total_voxel_count(self) -> int:
        """Extended voxel count (matches helpers.glsl extended_voxel_count())."""
        return self.grid_dimension_x * self.grid_dimension_y * self.grid_dimension_z


@dataclass
class GhostGridParams:
    """Spec const ids 80-81. THIS GPU's ghost voxel counts (per direction).

    ghost_voxel_count = ghost_x_thickness * grid_y * grid_z.
    End-of-chain GPUs have the non-peer side count = 0.
    """
    leading_ghost_voxel_count: int
    trailing_ghost_voxel_count: int


@dataclass
class DirectionalTransportSpec:
    """Per-direction (leading / trailing) ghost_send + install_migrations
    spec constants (ids 90-94 in common.glsl).

    Each direction holds an independent instance — the same GPU's leading and
    trailing specs carry different values (the offsets typically have opposite
    signs in the symmetric 2-GPU case). End-of-chain GPUs leave the non-peer
    direction as None.

    Ghost pool capacity is NOT stored here; it lives in
    ``Capacities.leading_ghost_pool_size`` / ``trailing_ghost_pool_size``
    (spec const ids 54/55).
    """
    direction: int                            # id=90; 0 = leading send, 1 = trailing send
    boundary_voxel_x_local: int               # id=91; this GPU's outermost own column (extended-grid local x)
    ghost_voxel_x_local: int                  # id=92; this GPU's ghost column adjacent to that boundary
    ghost_pid_offset_to_receiver: int         # id=93; signed; peer_dst_pid = my_dst_pid + this
    ghost_voxel_id_offset_to_receiver: int    # id=94; signed; peer_vid     = my_vid     + this


@dataclass
class TransportConfig:
    """Per-GPU bundle of directional specs. has_*_peer property simplifies
    end-of-chain checks."""
    leading: Optional[DirectionalTransportSpec] = None
    trailing: Optional[DirectionalTransportSpec] = None

    @property
    def has_leading_peer(self) -> bool:
        return self.leading is not None

    @property
    def has_trailing_peer(self) -> bool:
        return self.trailing is not None


@dataclass
class MaterialParameter:
    """One row of MaterialParametersBuffer (set 3 binding 7). 48 B (struct layout)."""
    kind: int                        # KIND_FLUID / BOUNDARY / INLET / ROTOR
    rest_density: float
    viscosity: float
    eos_constant: float              # c0² · rest_density / γ
    smoothing_length: float          # per-group; V0/V1 = global h
    radius: float
    volume: float
    rotor_angular_velocity: float = 0.0
    viscosity_transfer: float = 0.0  # reserved (micropolar)
    viscosity_rotation: float = 0.0  # reserved
    reserved_material_0: int = 0
    reserved_material_1: int = 0

    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Not part of the GPU struct — used CPU-side at bootstrap to fill
    velocity_mass for fresh particles."""


@dataclass
class InitialParticles:
    """Pre-filtered initial state for THIS GPU's slab. Sized (N_alive,);
    simulator pads up to own_pool_size during upload."""
    positions: np.ndarray              # (N, 3) float32; world coords
    velocities: np.ndarray             # (N, 3) float32
    material_group: np.ndarray         # (N,) uint32, indexes CaseV4.materials


# ============================================================================
# Top-level case
# ============================================================================

@dataclass
class CaseV4:
    """Per-GPU V4 simulation case. Orchestrator builds one CaseV4 per slab."""
    physics: PhysicsConstants
    numerics: NumericsConstants
    capacities: Capacities
    grid: GridLayout
    ghost_grid: GhostGridParams
    transport: TransportConfig
    materials: list[MaterialParameter]
    initial: InitialParticles


