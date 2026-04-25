"""
SimulationConfig — pure-parameter dataclass mirroring case.yaml's block structure.

Layered design:
  - PhysicsConfig / NumericsConfig / CapacitiesConfig / TimeConfig — leaf blocks,
    1:1 with the corresponding case.yaml block. Each owns its own validation
    in __post_init__.
  - SimulationConfig — root; holds the four leaves. Exposes derived quantities
    that depend ONLY on intrinsic parameters (timestep, Wendland C4 coefficients,
    eps_h², neighbor_z_range) as @property — geometry-derived values
    (grid origin / dimension) are NOT stored here, they are passed separately
    to build_specialization_info().

Vulkan-free by design. No PyYAML dependency, no NumPy, no python-vulkan.
yaml parsing happens in case.py; this module is purely a typed parameter
container + a spec-constant blob assembler.

The _SPEC_CONSTANT_MAPPING table at the bottom MUST stay in lockstep with
shaders/sph/common.glsl. Adding a new spec constant requires editing both
files in the same commit.
"""

import math
import struct
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional


# ============================================================================
# Leaf config blocks (1:1 with case.yaml)
# ============================================================================


@dataclass
class PhysicsConfig:
    dimension: int                                  # 2 or 3
    smoothing_length: float                         # h (Wendland C4 support radius)
    particle_spacing: float                         # dx; per-material radius = dx/2, volume = dx^DIM
    speed_of_sound: float                           # c0
    power: float                                    # γ in Tait EOS
    cfl: float                                      # for timestep derivation
    gravity: tuple[float, float, float]

    def __post_init__(self):
        # YAML parses [0, -9.81, 0] as a list; coerce to tuple of floats.
        self.gravity = tuple(float(component) for component in self.gravity)
        if len(self.gravity) != 3:
            raise ValueError(
                f"physics.gravity must have 3 components, got {len(self.gravity)}")
        if self.dimension not in (2, 3):
            raise ValueError(
                f"physics.dimension must be 2 or 3, got {self.dimension}")
        if self.smoothing_length <= 0:
            raise ValueError(
                f"physics.smoothing_length must be > 0, got {self.smoothing_length}")
        if self.particle_spacing <= 0:
            raise ValueError(
                f"physics.particle_spacing must be > 0, got {self.particle_spacing}")
        if self.speed_of_sound <= 0:
            raise ValueError(
                f"physics.speed_of_sound must be > 0, got {self.speed_of_sound}")
        if self.power <= 0:
            raise ValueError(f"physics.power must be > 0, got {self.power}")
        if self.cfl <= 0:
            raise ValueError(f"physics.cfl must be > 0, got {self.cfl}")
        # Hard math contradiction: kernel support radius < particle spacing means
        # the nearest neighbor is outside the kernel — SPH cannot operate.
        if self.particle_spacing > self.smoothing_length:
            raise ValueError(
                f"physics.particle_spacing ({self.particle_spacing}) must be "
                f"<= smoothing_length ({self.smoothing_length}); otherwise the "
                f"kernel support contains no neighbors and SPH is degenerate. "
                f"Soft-margin warnings (e.g. dx > h/2) are reported in case.py.")


@dataclass
class RegularizationConfig:
    xi: float                                       # Tikhonov diagonal add
    det_threshold: float                            # below → fall back to identity M⁻¹
    frobenius_max: float                            # cap |M⁻¹|_F

    def __post_init__(self):
        if self.xi <= 0:
            raise ValueError(f"regularization.xi must be > 0, got {self.xi}")
        if self.det_threshold <= 0:
            raise ValueError(
                f"regularization.det_threshold must be > 0, got {self.det_threshold}")
        if self.frobenius_max <= 0:
            raise ValueError(
                f"regularization.frobenius_max must be > 0, got {self.frobenius_max}")


@dataclass
class NumericsConfig:
    delta_coefficient: float                        # δ in δ-SPH density diffusion
    pst_main: float                                 # PST main-shift scale (Sun 2017 δ-plus)
    pst_anti: float                                 # PST anti-shift (cohesion) multiplier
    regularization: RegularizationConfig

    def __post_init__(self):
        if self.delta_coefficient < 0:
            raise ValueError(
                f"numerics.delta_coefficient must be >= 0, got {self.delta_coefficient}")
        if self.pst_main < 0:
            raise ValueError(f"numerics.pst_main must be >= 0, got {self.pst_main}")
        if self.pst_anti < 0:
            raise ValueError(f"numerics.pst_anti must be >= 0, got {self.pst_anti}")


@dataclass
class CapacitiesConfig:
    pool_size: int                                  # max active particles
    max_per_voxel: int                              # MAX_PARTICLES_PER_VOXEL
    max_incoming: int                               # MAX_INCOMING_PER_VOXEL
    workgroup: int                                  # WORKGROUP_SIZE

    def __post_init__(self):
        if self.pool_size <= 0:
            raise ValueError(f"capacities.pool_size must be > 0, got {self.pool_size}")
        if self.max_per_voxel <= 0:
            raise ValueError(
                f"capacities.max_per_voxel must be > 0, got {self.max_per_voxel}")
        if self.max_incoming <= 0:
            raise ValueError(
                f"capacities.max_incoming must be > 0, got {self.max_incoming}")
        if self.workgroup <= 0:
            raise ValueError(f"capacities.workgroup must be > 0, got {self.workgroup}")


@dataclass
class TimeConfig:
    """All three budgets are independent; None on a budget disables it."""
    total: Optional[float]                          # physical-time cap (s); None = unlimited
    max_steps: Optional[int]                        # step-count cap; None = unlimited
    output_cadence: Optional[float]                 # snapshot interval (s); None = no snapshots

    def __post_init__(self):
        if self.total is not None and self.total <= 0:
            raise ValueError(f"time.total must be > 0 or None, got {self.total}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(
                f"time.max_steps must be > 0 or None, got {self.max_steps}")
        if self.output_cadence is not None and self.output_cadence <= 0:
            raise ValueError(
                f"time.output_cadence must be > 0 or None, got {self.output_cadence}")

    def is_time_exceeded(self, current_time: float) -> bool:
        return self.total is not None and current_time >= self.total

    def is_step_exceeded(self, current_step: int) -> bool:
        return self.max_steps is not None and current_step >= self.max_steps


# ============================================================================
# Root config + derived quantities
# ============================================================================


@dataclass
class SimulationConfig:
    physics: PhysicsConfig
    numerics: NumericsConfig
    capacities: CapacitiesConfig
    time: TimeConfig

    # ---- derived quantities (depend only on intrinsic params, not geometry) ----

    @property
    def timestep(self) -> float:
        """dt = CFL · h / c0  (informational on shader side; CPU computes for clock)."""
        return (self.physics.cfl * self.physics.smoothing_length
                / self.physics.speed_of_sound)

    @property
    def kernel_coefficient(self) -> float:
        """Wendland C4 normalization. Support radius = h (NOT 2h).

        2D:  9   / (π · h²)
        3D:  495 / (32 · π · h³)
        """
        h = self.physics.smoothing_length
        if self.physics.dimension == 2:
            return 9.0 / (math.pi * h * h)
        return 495.0 / (32.0 * math.pi * h * h * h)

    @property
    def kernel_gradient_coefficient(self) -> float:
        """∇W coefficient = W coefficient / h  (chain-rule 1/h factor)."""
        return self.kernel_coefficient / self.physics.smoothing_length

    @property
    def eps_h_squared(self) -> float:
        """Antuono δ-SPH division-by-zero guard for 1/(r² + ε_h²) terms."""
        h = self.physics.smoothing_length
        return 0.01 * h * h

    @property
    def neighbor_z_range(self) -> int:
        """27-voxel neighbor loop in 3D; collapses to 9 in 2D."""
        return 1 if self.physics.dimension == 3 else 0

    # ---- capacity diagnostics (uniform-spacing sanity estimates) ----
    # Both quantities assume uniform particle packing at `particle_spacing`.
    # They are LOWER-BOUND expectations for a flat fluid region; clumping under
    # gravity / vortices can spike these by 2–3×, so capacity headroom matters.

    @property
    def particles_per_voxel_estimate(self) -> float:
        """Expected particle count in a single voxel (side = h, spacing = dx).

        2D: (h / dx)²        3D: (h / dx)³

        Compare against capacities.max_per_voxel; flat fluid at uniform density
        should leave 1.5–2× headroom for transient clumping.
        """
        ratio = self.physics.smoothing_length / self.physics.particle_spacing
        return ratio ** self.physics.dimension

    @property
    def neighbors_in_support_estimate(self) -> float:
        """Expected neighbor count inside the Wendland C4 support (radius = h).

        2D: π · (h/dx)²              3D: (4/3) · π · (h/dx)³

        Stable WCSPH typically wants >= ~30 in 2D, >= ~50 in 3D. Below that,
        density / gradient / KCG estimators get noisy.
        """
        ratio = self.physics.smoothing_length / self.physics.particle_spacing
        if self.physics.dimension == 2:
            return math.pi * ratio * ratio
        return (4.0 / 3.0) * math.pi * ratio ** 3


# ============================================================================
# Specialization constant assembly
# ----------------------------------------------------------------------------
# Hand-maintained mapping. MUST be kept in lockstep with shaders/sph/common.glsl
# spec constant declarations (matching IDs, types, and intent).
#
# Each row: (constant_id, getter(config, grid) -> python value, struct format).
# Format characters: 'f' = float32, 'I' = uint32, 'i' = int32.
# Boolean spec constants are encoded as 'I' (0 / 1) per VkBool32 convention.
#
# `grid` is a dict produced by utils/sph/grid.py:
#     {'origin': (ox, oy, oz),     # tuple[float, float, float]
#      'dimension': (dx, dy, dz)}  # tuple[int,   int,   int]
# ============================================================================


_SpecGetter = Callable[["SimulationConfig", dict], object]
_SpecRow = tuple[int, _SpecGetter, str]


_SPEC_CONSTANT_MAPPING: list[_SpecRow] = [
    # id  getter                                                    fmt
    (0,   lambda c, g: c.physics.smoothing_length,                  'f'),
    (1,   lambda c, g: c.physics.speed_of_sound,                    'f'),
    (2,   lambda c, g: c.numerics.delta_coefficient,                'f'),
    # id=3 reserved (was EPSILON_SHIFT, removed)
    (4,   lambda c, g: c.physics.power,                             'f'),
    (5,   lambda c, g: c.physics.cfl,                               'f'),
    (6,   lambda c, g: c.timestep,                                  'f'),  # derived
    (7,   lambda c, g: g['origin'][0],                              'f'),
    (8,   lambda c, g: g['origin'][1],                              'f'),
    (9,   lambda c, g: g['origin'][2],                              'f'),
    (10,  lambda c, g: 0,                                           'I'),  # STRICT_BIT_EXACT (V0: false)
    (11,  lambda c, g: g['dimension'][0],                           'I'),
    (12,  lambda c, g: g['dimension'][1],                           'I'),
    (13,  lambda c, g: g['dimension'][2],                           'I'),
    (14,  lambda c, g: c.numerics.regularization.xi,                'f'),
    (15,  lambda c, g: c.numerics.regularization.det_threshold,     'f'),
    (16,  lambda c, g: c.numerics.regularization.frobenius_max,     'f'),
    (17,  lambda c, g: c.physics.gravity[0],                        'f'),
    (18,  lambda c, g: c.physics.gravity[1],                        'f'),
    (19,  lambda c, g: c.physics.gravity[2],                        'f'),
    (20,  lambda c, g: 0,                                           'I'),  # VOXEL_ORDER (V0: linear)
    (21,  lambda c, g: 2.0,                                         'f'),  # MICROPOLAR_THETA (V0 unused)
    (30,  lambda c, g: c.physics.dimension,                         'I'),
    (31,  lambda c, g: c.neighbor_z_range,                          'I'),
    (32,  lambda c, g: c.kernel_coefficient,                        'f'),
    (33,  lambda c, g: c.kernel_gradient_coefficient,               'f'),
    (40,  lambda c, g: c.eps_h_squared,                             'f'),
    (41,  lambda c, g: c.numerics.pst_main,                         'f'),
    (42,  lambda c, g: c.numerics.pst_anti,                         'f'),
    (50,  lambda c, g: c.capacities.max_per_voxel,                  'I'),
    (51,  lambda c, g: c.capacities.workgroup,                      'I'),
    (52,  lambda c, g: c.capacities.max_incoming,                   'I'),
    (53,  lambda c, g: c.capacities.pool_size,                      'I'),
    # V0-a ghost grid: all disabled (GHOST_DIMENSION_* = 0 dead-code-eliminates branches)
    (80,  lambda c, g: 0,                                           'I'),
    (81,  lambda c, g: 0,                                           'I'),
    (82,  lambda c, g: 0,                                           'I'),
    (83,  lambda c, g: 0.0,                                         'f'),
    (84,  lambda c, g: 0.0,                                         'f'),
    (85,  lambda c, g: 0.0,                                         'f'),
    (86,  lambda c, g: 0,                                           'i'),
    (87,  lambda c, g: 0,                                           'i'),
    (88,  lambda c, g: 0,                                           'i'),
]


class SpecializationInfo(NamedTuple):
    """Pure-Python spec info; pipelines.py converts to VkSpecializationInfo."""
    map_entries: list[tuple[int, int, int]]         # (constant_id, offset, size)
    data: bytes                                     # packed blob


def _validate_grid(grid: dict) -> None:
    if 'origin' not in grid or 'dimension' not in grid:
        raise ValueError(
            f"grid dict must have 'origin' and 'dimension' keys, got {list(grid.keys())}")
    if len(grid['origin']) != 3:
        raise ValueError(
            f"grid['origin'] must have 3 components, got {len(grid['origin'])}")
    if len(grid['dimension']) != 3:
        raise ValueError(
            f"grid['dimension'] must have 3 components, got {len(grid['dimension'])}")
    for axis_index, dimension_value in enumerate(grid['dimension']):
        if not isinstance(dimension_value, int) or dimension_value <= 0:
            raise ValueError(
                f"grid['dimension'][{axis_index}] must be a positive int, "
                f"got {dimension_value!r}")


def build_specialization_info(
    config: SimulationConfig,
    grid: dict,
) -> SpecializationInfo:
    """
    Pack all spec constants into a contiguous data blob and return alongside
    map entries (constant_id, offset, size). Caller (utils/sph/pipelines.py)
    converts to VkSpecializationMapEntry + VkSpecializationInfo at pipeline
    creation time.

    Constant ordering and types must match shaders/sph/common.glsl exactly.
    """
    _validate_grid(grid)

    map_entries: list[tuple[int, int, int]] = []
    data = bytearray()
    for constant_id, getter, fmt in _SPEC_CONSTANT_MAPPING:
        value = getter(config, grid)
        size = struct.calcsize(fmt)
        map_entries.append((constant_id, len(data), size))
        data.extend(struct.pack(fmt, value))

    return SpecializationInfo(map_entries=map_entries, data=bytes(data))
