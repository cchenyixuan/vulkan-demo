"""
case_loader_v3.py — load a YAML case + OBJ vertex files into CaseV3.

V3 isolation: minimal self-contained loader; does NOT import utils/sph/.
Replicates the bare-minimum subset of utils/sph/case.py that V3 needs:

  - Parse case.yaml (physics / numerics / capacities / geometry / materials)
  - Parse material YAML library (resolve material names → MaterialParameter)
  - Parse OBJ vertex lines (`v x y z`) for each particle source + frame
  - Compute grid origin / dimension from frame.obj bbox via the same anchoring
    rule as V0 (bbox.min at voxel (0,0,0) center; dim = floor(span/h+0.5)+1)
  - Compose CaseV3 ready to hand to SphSimulatorV3

What is NOT replicated:
  - Closest-packing self-check (V0 case.py has it; V3 trusts caller)
  - Per-source mask filtering (Phase 4 introduces this for multi-GPU)
  - Schema version validation (V3 reads schema_version: 2 directly)
  - INLET / ROTOR kind handling (V1 cavity has none)
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import yaml

from experiment.v3.utils.case_v3 import (
    CaseV3,
    Capacities,
    GhostGridParams,
    GridLayout,
    InitialParticles,
    KIND_BOUNDARY,
    KIND_FLUID,
    KIND_INLET,
    KIND_ROTOR,
    MaterialParameter,
    NumericsConstants,
    PhysicsConstants,
    TransportConfig,
)


_KIND_NAME_TO_ID = {
    "fluid": KIND_FLUID,
    "boundary": KIND_BOUNDARY,
    "inlet": KIND_INLET,
    "rotor": KIND_ROTOR,
}


# ============================================================================
# OBJ vertex parser
# ============================================================================

def _parse_obj_vertices(path: pathlib.Path) -> np.ndarray:
    """Return (N, 3) float32 array of `v x y z` lines, ignoring everything else."""
    vertices: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            hash_pos = raw_line.find("#")
            if hash_pos != -1:
                raw_line = raw_line[:hash_pos]
            tokens = raw_line.split()
            if not tokens or tokens[0] != "v":
                continue
            if len(tokens) < 4:
                continue
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
    if not vertices:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)


# ============================================================================
# Grid derivation (matches utils/sph/grid.py)
# ============================================================================

def _compute_grid(bbox_min: np.ndarray, bbox_max: np.ndarray,
                  h: float, dimension: int) -> tuple[tuple[float, float, float],
                                                    tuple[int, int, int]]:
    """Anchor voxel (0,0,0) at bbox_min and size the grid to enclose bbox_max.

    Conventions:
      - origin = bbox_min - 0.5·h shifts origin half a voxel "below" so that
        bbox_min sits at the CENTER of voxel (0,0,0) (not its corner).
      - dims uses round-then-+1: round(span/h) gives the number of voxel-
        spacings between min and max centers; +1 converts spacings → voxel
        count (e.g. span=0 → 1 voxel, span=h → 2 voxels).
      - 2D: collapse z to a single voxel layer so dim_z = 1.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    if dimension == 2:
        bbox_min = bbox_min.copy()
        bbox_max = bbox_max.copy()
        bbox_min[2] = 0.0
        bbox_max[2] = 0.0
    origin = bbox_min - 0.5 * h
    span = bbox_max - bbox_min
    dims = np.floor(span / h + 0.5).astype(int) + 1
    return (tuple(float(o) for o in origin),
            tuple(int(d) for d in dims))


# ============================================================================
# Particle volume calibration (ported from utils/sph/case.py:74-193)
# ============================================================================

def _calibrate_particle_volume(
    h: float,
    particle_radius: float,
    dimension: int,
    lattice: str,
) -> float:
    """SPH partition-of-unity volume: V = 1 / Σ_{j≠i} W(r_ij).

    ``correction.comp`` accumulates ``Σ_{j != i} V_j · W(r_ij)`` (self excluded
    via ``if (neighbor == self) continue``). For an interior particle on a
    regular lattice with spacing dx = 2·particle_radius and uniform V_j = V,
    setting V = 1 / Σ W(r_j) makes interior ``kernel_sum`` land on 1.0 in the
    bulk — the partition-of-unity property the KGC correction and δ-SPH
    diffusion both assume.

    ``lattice`` must match the preprocessor's particle layout:
      - ``"grid"``: Cartesian (square 2D / simple cubic 3D), spacing dx.
                    Blender uniform-mesh export defaults to this.
      - ``"hex"`` : 2D hex close packing / 3D FCC, nearest-neighbor distance dx.
    """
    if lattice not in ("grid", "hex"):
        raise ValueError(f"lattice must be 'grid' or 'hex', got {lattice!r}")

    diameter = 2.0 * particle_radius
    if dimension == 2:
        coefficient = 9.0 / (math.pi * h * h)
    else:
        coefficient = 495.0 / (32.0 * math.pi * h * h * h)

    def kernel_W(distance: float) -> float:
        q = distance / h
        if q >= 1.0:
            return 0.0
        one_minus_q = 1.0 - q
        return (
            coefficient
            * (one_minus_q ** 6)
            * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)
        )

    kernel_sum = 0.0

    if lattice == "grid":
        n_max = int(math.ceil(h / diameter))
        if dimension == 2:
            for i in range(-n_max, n_max + 1):
                for j in range(-n_max, n_max + 1):
                    if i == 0 and j == 0:
                        continue                       # self excluded
                    r = math.sqrt((i * diameter) ** 2 + (j * diameter) ** 2)
                    kernel_sum += kernel_W(r)
        else:
            for i in range(-n_max, n_max + 1):
                for j in range(-n_max, n_max + 1):
                    for k in range(-n_max, n_max + 1):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        r = math.sqrt(
                            (i * diameter) ** 2
                            + (j * diameter) ** 2
                            + (k * diameter) ** 2
                        )
                        kernel_sum += kernel_W(r)

    else:  # lattice == "hex"
        if dimension == 2:
            # Hex 2D primitive vectors:
            #   a1 = (dx, 0)
            #   a2 = (dx/2, dx·√3/2)
            # Each interior particle sees 6 nearest neighbors at dx.
            n_iter = int(math.ceil(h / diameter)) + 2     # +2 buffer for skew
            sqrt3_half = math.sqrt(3.0) * 0.5
            for i in range(-2 * n_iter, 2 * n_iter + 1):
                for j in range(-2 * n_iter, 2 * n_iter + 1):
                    if i == 0 and j == 0:
                        continue
                    x = i * diameter + j * diameter * 0.5
                    y = j * diameter * sqrt3_half
                    r = math.sqrt(x * x + y * y)
                    kernel_sum += kernel_W(r)
        else:
            # FCC 3D: 4 atoms per cubic supercell of side a = dx·√2 — gives
            # 12 nearest neighbors at distance dx for each atom.
            a = diameter * math.sqrt(2.0)
            n_iter = int(math.ceil(h / a)) + 2
            basis = (
                (0.0,     0.0,     0.0),
                (a * 0.5, a * 0.5, 0.0),
                (a * 0.5, 0.0,     a * 0.5),
                (0.0,     a * 0.5, a * 0.5),
            )
            for ci in range(-n_iter, n_iter + 1):
                for cj in range(-n_iter, n_iter + 1):
                    for ck in range(-n_iter, n_iter + 1):
                        for (bx, by, bz) in basis:
                            if (ci == 0 and cj == 0 and ck == 0
                                    and bx == 0.0 and by == 0.0 and bz == 0.0):
                                continue
                            x = ci * a + bx
                            y = cj * a + by
                            z = ck * a + bz
                            r = math.sqrt(x * x + y * y + z * z)
                            kernel_sum += kernel_W(r)

    if kernel_sum <= 0:
        raise ValueError(
            f"calibration failed: kernel_sum={kernel_sum} for "
            f"h={h}, dx={diameter}, dim={dimension}, lattice={lattice!r}")
    return 1.0 / kernel_sum


# ============================================================================
# Material library resolution
# ============================================================================

def _resolve_materials(
    library: dict,
    used_names: list[str],            # in order of first appearance in case.yaml
    physics_h: float,
    speed_of_sound: float,
    power: float,
    particle_radius: float,
    dimension: int,
    lattice: str,
    calibrate_volume: bool,
) -> list[MaterialParameter]:
    """For each name in used_names, build a MaterialParameter at group_id = index.

    Derived fields (not in YAML):
      eos_constant     = c0² · rest_density / power
      smoothing_length = global h (per-material spacing is a future feature;
                                   version TBD)
      radius           = global particle_radius
      volume           = partition-of-unity calibrated (1 / Σ W on the
                         requested lattice) when calibrate_volume=True; else
                         naive (2·radius)^dimension. See
                         _calibrate_particle_volume.
    """
    result: list[MaterialParameter] = []
    for name in used_names:
        if name not in library:
            raise KeyError(f"material {name!r} not in library")
        spec = library[name]
        kind_name = spec["kind"]
        if kind_name not in _KIND_NAME_TO_ID:
            raise ValueError(f"material {name!r} has unknown kind: {kind_name}")
        kind = _KIND_NAME_TO_ID[kind_name]
        rest_density = float(spec["rest_density"])
        viscosity = float(spec.get("viscosity", 0.0))
        eos_constant = speed_of_sound ** 2 * rest_density / power
        if calibrate_volume:
            volume = _calibrate_particle_volume(
                physics_h, particle_radius, dimension, lattice)
        else:
            diameter = 2.0 * particle_radius
            volume = diameter ** dimension
        initial_velocity = tuple(spec.get("initial_velocity", [0.0, 0.0, 0.0]))
        result.append(MaterialParameter(
            kind=kind,
            rest_density=rest_density,
            viscosity=viscosity,
            eos_constant=eos_constant,
            smoothing_length=physics_h,
            radius=particle_radius,
            volume=volume,
            rotor_angular_velocity=float(spec.get("rotor_angular_velocity", 0.0)),
            initial_velocity=(float(initial_velocity[0]),
                              float(initial_velocity[1]),
                              float(initial_velocity[2])),
        ))
    return result


# ============================================================================
# Public loader
# ============================================================================

def load_case_v3(case_yaml_path: str | pathlib.Path) -> CaseV3:
    """Parse a V0/V1-style case.yaml into a CaseV3.

    Output contract: the returned CaseV3 is a **degenerate slab** that owns
    the whole simulation domain — no peer, ghost pools = 0, ghost voxel
    counts = 0, transport.leading / transport.trailing both None.

    Two consumption modes downstream:

      single-GPU: feed the returned CaseV3 directly into SphSimulatorV3;
                  the simulator's per-direction ghost flow auto-skips because
                  ``case.transport.has_*_peer`` is False on both sides.

      dual-GPU:   hand it to ``partition_v3.compute_dual_gpu_partition`` which
                  splits this global case into per-slab CaseV3s with populated
                  ghost / transport fields.

    Ghost column synthesis is partition_v3's job, not this loader's.
    """
    case_path = pathlib.Path(case_yaml_path).resolve()
    case_dir = case_path.parent
    case_data = yaml.safe_load(case_path.read_text(encoding="utf-8"))

    if case_data.get("schema_version") != 2:
        raise ValueError(
            f"unsupported schema_version: got {case_data.get('schema_version')}; "
            f"V3 loader expects 2")

    # --- Physics ----------------------------------------------------------
    phys = case_data["physics"]
    h = float(phys["h"])
    speed_of_sound = float(phys["speed_of_sound"])
    power = float(phys["power"])
    cfl = float(phys["cfl"])
    timestep = cfl * h / speed_of_sound
    dimension = int(phys["dimension"])
    gravity = tuple(float(g) for g in phys.get("gravity", [0.0, 0.0, 0.0]))
    particle_radius = float(phys["particle_radius"])
    lattice = str(phys.get("lattice", "grid"))
    if lattice not in ("grid", "hex"):
        raise ValueError(
            f"physics.lattice must be 'grid' or 'hex', got {lattice!r}")
    calibrate_volume = bool(phys.get("calibrate_volume", True))

    physics = PhysicsConstants(
        smoothing_length=h,
        speed_of_sound=speed_of_sound,
        delta_coefficient=float(case_data["numerics"].get("delta_coefficient", 0.1)),
        power_parameter=power,
        cfl_number=cfl,
        timestep=timestep,
        gravity=gravity,
        dimension=dimension,
        neighbor_z_range=0 if dimension == 2 else 1,
        # Wendland C4 normalization (matches utils/sph/case.py)
        kernel_coefficient=(9.0 / (np.pi * h * h) if dimension == 2
                            else 495.0 / (32.0 * np.pi * h ** 3)),
        kernel_gradient_coefficient=(9.0 / (np.pi * h ** 3) if dimension == 2
                                     else 495.0 / (32.0 * np.pi * h ** 4)),
        lattice=lattice,
        calibrate_volume=calibrate_volume,
    )

    # --- Numerics ---------------------------------------------------------
    nm = case_data["numerics"]
    reg = nm["regularization"]
    numerics = NumericsConstants(
        regularization_xi=float(reg["xi"]),
        regularization_determinant_threshold=float(reg["det_threshold"]),
        regularization_max_frobenius_norm=float(reg["frobenius_max"]),
        eps_h_squared=0.01 * h * h,                # V0/V1 default (Antuono δ-SPH)
        pst_main_shift_coefficient=float(nm.get("pst_main", 0.1)),
        pst_anti_shift_coefficient=float(nm.get("pst_anti", 0.005)),
        use_kcg_correction=bool(nm.get("use_kcg_correction", True)),
        use_density_diffusion=bool(nm.get("use_density_diffusion", True)),
        use_pst=bool(nm.get("use_pst", True)),
        use_prefix_sum_defrag=bool(nm.get("use_prefix_sum_defrag", False)),
        defrag_cadence=int(nm.get("defrag_cadence", 1000)),
    )

    # --- Capacities -------------------------------------------------------
    caps = case_data["capacities"]
    cap_inside = int(caps["max_per_voxel"])
    cap_incoming = int(caps["max_incoming"])
    own_pool_size = int(caps["pool_size"])

    # --- Geometry: materials + particle sources ---------------------------
    library_path = case_data["material_library"]
    library_full = (case_dir / library_path).resolve()
    library_yaml = yaml.safe_load(library_full.read_text(encoding="utf-8"))

    geometry = case_data["geometry"]
    used_material_names: list[str] = []
    for entry in geometry["particles"]:
        name = entry["material"]
        if name not in used_material_names:
            used_material_names.append(name)
    materials = _resolve_materials(
        library_yaml, used_material_names,
        physics_h=h, speed_of_sound=speed_of_sound, power=power,
        particle_radius=particle_radius, dimension=dimension,
        lattice=lattice, calibrate_volume=calibrate_volume,
    )
    material_name_to_group = {name: idx for idx, name in enumerate(used_material_names)}

    # Load each source's OBJ; concatenate with material group tagging.
    all_positions_list: list[np.ndarray] = []
    all_velocities_list: list[np.ndarray] = []
    all_material_group_list: list[np.ndarray] = []
    total_loaded = 0
    print(f"[case_loader_v3] loading {case_dir.name}")
    for entry in geometry["particles"]:
        obj_path = (case_dir / entry["file"]).resolve()
        verts = _parse_obj_vertices(obj_path)
        if dimension == 2:
            verts = verts.copy()
            verts[:, 2] = 0.0
        n = verts.shape[0]
        group = material_name_to_group[entry["material"]]
        # Initial velocity from material spec (e.g. lid moves with U=1)
        init_vel = np.broadcast_to(
            np.asarray(materials[group].initial_velocity, dtype=np.float32),
            (n, 3)).copy()
        groups = np.full(n, group, dtype=np.uint32)
        all_positions_list.append(verts)
        all_velocities_list.append(init_vel)
        all_material_group_list.append(groups)
        total_loaded += n
        print(f"  - {entry['file']}: {n:,} particles ({entry['material']})")
    positions = np.concatenate(all_positions_list).astype(np.float32)
    velocities = np.concatenate(all_velocities_list).astype(np.float32)
    material_group = np.concatenate(all_material_group_list).astype(np.uint32)
    print(f"[case_loader_v3] total loaded: {total_loaded:,} particles")

    if total_loaded > own_pool_size:
        raise ValueError(
            f"loaded {total_loaded} particles > pool_size {own_pool_size}; "
            f"increase capacities.pool_size in case.yaml")

    # --- Grid: bbox from frame.obj (single-GPU = full domain) -------------
    frame_obj = (case_dir / geometry["frame"]).resolve()
    frame_verts = _parse_obj_vertices(frame_obj)
    if frame_verts.shape[0] == 0:
        raise ValueError(f"frame OBJ {frame_obj} contains no vertices")
    bbox_min = frame_verts.min(axis=0)
    bbox_max = frame_verts.max(axis=0)
    origin_tuple, dim_tuple = _compute_grid(bbox_min, bbox_max, h, dimension)

    # Degenerate slab: owns the whole domain, no ghost columns, no peer.
    # partition_v3 is responsible for synthesizing per-slab ghost / transport.
    grid = GridLayout(
        origin_x=origin_tuple[0],
        origin_y=origin_tuple[1],
        origin_z=origin_tuple[2],
        grid_dimension_x=dim_tuple[0],
        grid_dimension_y=dim_tuple[1],
        grid_dimension_z=dim_tuple[2],
    )
    capacities = Capacities(
        max_particles_per_voxel=cap_inside,
        workgroup_size=int(caps["workgroup"]),
        max_incoming_per_voxel=cap_incoming,
        own_pool_size=own_pool_size,
        leading_ghost_pool_size=0,
        trailing_ghost_pool_size=0,
    )
    ghost_grid = GhostGridParams(
        leading_ghost_voxel_count=0,
        trailing_ghost_voxel_count=0,
    )
    transport = TransportConfig()

    initial = InitialParticles(
        positions=positions, velocities=velocities, material_group=material_group)

    print(f"[case_loader_v3] grid: dim={dim_tuple} origin={origin_tuple}")

    return CaseV3(
        physics=physics, numerics=numerics, capacities=capacities,
        grid=grid, ghost_grid=ghost_grid, transport=transport,
        materials=materials, initial=initial,
    )
