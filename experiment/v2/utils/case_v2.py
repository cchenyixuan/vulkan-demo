"""
case_v2.py — V2 simulation case dataclasses + minimal test factory.

V2 isolation: no import from utils/sph/ or experiment/v1/. The dataclasses
here mirror what V1's Case carries but with the structure flattened for V2's
needs (per-GPU view; orchestrator builds two CaseV2 objects, one per slab).

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
    """Spec const ids 0-9, 17-19, 30-33 in common.glsl."""
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
    """Spec const ids 50-55."""
    max_particles_per_voxel: int
    workgroup_size: int
    max_incoming_per_voxel: int
    own_pool_size: int
    leading_ghost_pool_size: int     # = 0 if no leading peer (end of chain)
    trailing_ghost_pool_size: int    # = 0 if no trailing peer

    def total_pool_capacity(self) -> int:
        """Total particle slot count = slot 0 unused + leading + own + trailing."""
        return (1
                + self.leading_ghost_pool_size
                + self.own_pool_size
                + self.trailing_ghost_pool_size)


@dataclass
class GridLayout:
    """Spec const ids 7-9, 11-13, 20.

    For V1's merged-buffer scheme, grid_dimension_x is the EXTENDED nx covering
    own columns PLUS leading + trailing ghost columns. origin_x is the world-
    space x of voxel column 0 on THIS GPU (= world domain origin shifted left
    by leading ghost columns × voxel_size for end-of-chain-rightmost / middle).
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
    """Per-direction (one of leading / trailing) ghost_send + install_migrations
    spec constants. Spec const ids 90-94.

    Created only for directions where THIS GPU has a peer (end-of-chain GPUs
    have None for the non-peer direction).
    """
    direction: int                            # 0 = leading send, 1 = trailing send
    boundary_voxel_x_local: int               # this GPU's outermost own column (extended-grid local)
    ghost_voxel_x_local: int                  # this GPU's ghost column adjacent to boundary
    ghost_pid_offset_to_receiver: int         # signed; for equal-pool 2-GPU = ±OWN_POOL_SIZE
    ghost_voxel_id_offset_to_receiver: int    # signed; sender→receiver coord shift
    pool_size: int                            # this direction's ghost pool (own side, capacity)


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
    simulator pads up to own_pool_size during upload. N_alive may be 0 for
    Phase 2 test cases (buffer allocation doesn't need particles)."""
    positions: np.ndarray              # (N, 3) float32; world coords
    velocities: np.ndarray             # (N, 3) float32
    material_group: np.ndarray         # (N,) uint32, indexes CaseV2.materials


# ============================================================================
# Top-level case
# ============================================================================

@dataclass
class CaseV2:
    """Per-GPU V2 simulation case. Orchestrator builds one CaseV2 per slab."""
    physics: PhysicsConstants
    numerics: NumericsConstants
    capacities: Capacities
    grid: GridLayout
    ghost_grid: GhostGridParams
    transport: TransportConfig
    materials: list[MaterialParameter]
    initial: InitialParticles


# ============================================================================
# Minimal test factory — for Phase 2 / Phase 3 bring-up without real case data
# ============================================================================

def make_minimal_test_case(
    own_pool_size: int = 1024,
    grid_xyz: tuple[int, int, int] = (16, 16, 1),
    ghost_x_thickness: int = 1,
    *,
    has_leading_peer: bool = False,
    has_trailing_peer: bool = False,
) -> CaseV2:
    """Build a tiny CaseV2 sized for Phase 2/3 bring-up. No particles
    (InitialParticles is empty arrays); buffer sizing covers the requested
    pool_size + ghost capacity. Physical params are V1-cavity-ish defaults.

    has_leading_peer / has_trailing_peer toggle which directions have ghost
    pool + transport spec populated. Default = endpoint GPU (no peers; ghost
    counts all zero) for single-GPU smoke tests.
    """
    nx_own, ny, nz = grid_xyz
    leading_thickness = ghost_x_thickness if has_leading_peer else 0
    trailing_thickness = ghost_x_thickness if has_trailing_peer else 0
    extended_nx = nx_own + leading_thickness + trailing_thickness
    voxel_per_x_column = ny * nz

    leading_voxel_count = leading_thickness * voxel_per_x_column
    trailing_voxel_count = trailing_thickness * voxel_per_x_column
    # Conservative ghost pool size: 1 yz-face × (replicas + migrations) per slot
    cap_inside = 96
    cap_incoming = 16
    leading_pool = leading_voxel_count * (cap_inside + cap_incoming) if has_leading_peer else 0
    trailing_pool = trailing_voxel_count * (cap_inside + cap_incoming) if has_trailing_peer else 0

    h = 0.009  # V1 default smoothing length
    voxel_size = h
    physics = PhysicsConstants(
        smoothing_length=h,
        speed_of_sound=300.0,
        delta_coefficient=0.1,
        power_parameter=7.0,
        cfl_number=0.15,
        timestep=0.15 * h / 300.0,
        gravity=(0.0, 0.0, 0.0),
        dimension=2 if nz == 1 else 3,
        neighbor_z_range=0 if nz == 1 else 1,
        kernel_coefficient=9.0 / (np.pi * h * h),               # 2D Wendland C4
        kernel_gradient_coefficient=9.0 / (np.pi * h * h * h),
    )
    numerics = NumericsConstants(
        regularization_xi=0.1,
        regularization_determinant_threshold=1e-4,
        regularization_max_frobenius_norm=10.0,
        eps_h_squared=0.01 * h * h,
        pst_main_shift_coefficient=0.1,
        pst_anti_shift_coefficient=0.005,
    )
    capacities = Capacities(
        max_particles_per_voxel=cap_inside,
        workgroup_size=128,
        max_incoming_per_voxel=cap_incoming,
        own_pool_size=own_pool_size,
        leading_ghost_pool_size=leading_pool,
        trailing_ghost_pool_size=trailing_pool,
    )
    grid = GridLayout(
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        grid_dimension_x=extended_nx,
        grid_dimension_y=ny,
        grid_dimension_z=nz,
    )
    ghost_grid = GhostGridParams(
        leading_ghost_voxel_count=leading_voxel_count,
        trailing_ghost_voxel_count=trailing_voxel_count,
    )

    transport = TransportConfig(
        leading=DirectionalTransportSpec(
            direction=0,
            boundary_voxel_x_local=leading_thickness,
            ghost_voxel_x_local=0,
            ghost_pid_offset_to_receiver=0,   # placeholder; orchestrator fills real value
            ghost_voxel_id_offset_to_receiver=0,
            pool_size=leading_pool,
        ) if has_leading_peer else None,
        trailing=DirectionalTransportSpec(
            direction=1,
            boundary_voxel_x_local=extended_nx - 1 - trailing_thickness,
            ghost_voxel_x_local=extended_nx - 1,
            ghost_pid_offset_to_receiver=0,
            ghost_voxel_id_offset_to_receiver=0,
            pool_size=trailing_pool,
        ) if has_trailing_peer else None,
    )

    fluid = MaterialParameter(
        kind=KIND_FLUID,
        rest_density=1000.0,
        viscosity=1e-3,
        eos_constant=300.0 ** 2 * 1000.0 / 7.0,
        smoothing_length=h,
        radius=h * 0.5,
        volume=h * h,                # 2D
    )
    materials = [fluid]

    initial = InitialParticles(
        positions=np.zeros((0, 3), dtype=np.float32),
        velocities=np.zeros((0, 3), dtype=np.float32),
        material_group=np.zeros(0, dtype=np.uint32),
    )

    return CaseV2(
        physics=physics,
        numerics=numerics,
        capacities=capacities,
        grid=grid,
        ghost_grid=ghost_grid,
        transport=transport,
        materials=materials,
        initial=initial,
    )
