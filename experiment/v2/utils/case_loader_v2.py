"""
case_loader_v2.py — load a YAML case + OBJ vertex files into CaseV2.

V2 isolation: minimal self-contained loader; does NOT import utils/sph/.
Replicates the bare-minimum subset of utils/sph/case.py that V2 needs:

  - Parse case.yaml (physics / numerics / capacities / geometry / materials)
  - Parse material YAML library (resolve material names → MaterialParameter)
  - Parse OBJ vertex lines (`v x y z`) for each particle source + frame
  - Compute grid origin / dimension from frame.obj bbox via the same anchoring
    rule as V0 (bbox.min at voxel (0,0,0) center; dim = floor(span/h+0.5)+1)
  - Compose CaseV2 ready to hand to SphSimulatorV2

What is NOT replicated:
  - Closest-packing self-check (V0 case.py has it; V2 trusts caller)
  - Per-source mask filtering (Phase 4 introduces this for multi-GPU)
  - Schema version validation (V2 reads schema_version: 2 directly)
  - INLET / ROTOR kind handling (V1 cavity has none)
"""

from __future__ import annotations

import pathlib

import numpy as np
import yaml

from experiment.v2.utils.case_v2 import (
    CaseV2,
    Capacities,
    DirectionalTransportSpec,
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
) -> list[MaterialParameter]:
    """For each name in used_names, build a MaterialParameter at group_id = index.

    Derived fields (not in YAML):
      eos_constant     = c0² · rest_density / power
      smoothing_length = global h (V0/V1 spec; per-material spacing is V0+)
      radius           = global particle_radius
      volume           = radius^2 · 4 (2D) or radius^3 · 8 (3D)  // matches V1
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
        # Volume convention: V0's case.py uses (2r)^dim → particle "cell" size.
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

def load_case_v2(case_yaml_path: str | pathlib.Path,
                 *,
                 leading_ghost_x_thickness: int = 0,
                 trailing_ghost_x_thickness: int = 0) -> CaseV2:
    """Load a V0/V1-style case.yaml into a CaseV2.

    Single-GPU mode (Phase 3 testing): both ghost thicknesses = 0 → end-of-chain
    layout, no peer. Phase 4 will introduce per-slab loading where each GPU
    sees a sub-range of the global grid plus the appropriate ghost columns.
    """
    case_path = pathlib.Path(case_yaml_path).resolve()
    case_dir = case_path.parent
    case_data = yaml.safe_load(case_path.read_text(encoding="utf-8"))

    if case_data.get("schema_version") != 2:
        raise ValueError(
            f"unsupported schema_version: got {case_data.get('schema_version')}; "
            f"V2 loader expects 2")

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
    )
    material_name_to_group = {name: idx for idx, name in enumerate(used_material_names)}

    # Load each source's OBJ; concatenate with material group tagging.
    all_positions_list: list[np.ndarray] = []
    all_velocities_list: list[np.ndarray] = []
    all_material_group_list: list[np.ndarray] = []
    total_loaded = 0
    print(f"[case_loader_v2] loading {case_dir.name}")
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
    print(f"[case_loader_v2] total loaded: {total_loaded:,} particles")

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

    # V2 single-GPU = end-of-chain on both sides → no ghost columns
    grid = GridLayout(
        origin_x=origin_tuple[0],
        origin_y=origin_tuple[1],
        origin_z=origin_tuple[2],
        grid_dimension_x=dim_tuple[0] + leading_ghost_x_thickness + trailing_ghost_x_thickness,
        grid_dimension_y=dim_tuple[1],
        grid_dimension_z=dim_tuple[2],
    )
    voxel_per_x_column = grid.grid_dimension_y * grid.grid_dimension_z
    leading_voxel_count = leading_ghost_x_thickness * voxel_per_x_column
    trailing_voxel_count = trailing_ghost_x_thickness * voxel_per_x_column
    leading_pool = leading_voxel_count * (cap_inside + cap_incoming) if leading_ghost_x_thickness > 0 else 0
    trailing_pool = trailing_voxel_count * (cap_inside + cap_incoming) if trailing_ghost_x_thickness > 0 else 0

    capacities = Capacities(
        max_particles_per_voxel=cap_inside,
        workgroup_size=int(caps["workgroup"]),
        max_incoming_per_voxel=cap_incoming,
        own_pool_size=own_pool_size,
        leading_ghost_pool_size=leading_pool,
        trailing_ghost_pool_size=trailing_pool,
    )
    ghost_grid = GhostGridParams(
        leading_ghost_voxel_count=leading_voxel_count,
        trailing_ghost_voxel_count=trailing_voxel_count,
    )
    transport = TransportConfig(
        leading=DirectionalTransportSpec(
            direction=0, boundary_voxel_x_local=leading_ghost_x_thickness,
            ghost_voxel_x_local=0,
            ghost_pid_offset_to_receiver=0,
            ghost_voxel_id_offset_to_receiver=0,
            pool_size=leading_pool,
        ) if leading_ghost_x_thickness > 0 else None,
        trailing=DirectionalTransportSpec(
            direction=1,
            boundary_voxel_x_local=grid.grid_dimension_x - 1 - trailing_ghost_x_thickness,
            ghost_voxel_x_local=grid.grid_dimension_x - 1,
            ghost_pid_offset_to_receiver=0,
            ghost_voxel_id_offset_to_receiver=0,
            pool_size=trailing_pool,
        ) if trailing_ghost_x_thickness > 0 else None,
    )

    initial = InitialParticles(
        positions=positions, velocities=velocities, material_group=material_group)

    print(f"[case_loader_v2] grid: dim={dim_tuple} origin={origin_tuple}")

    return CaseV2(
        physics=physics, numerics=numerics, capacities=capacities,
        grid=grid, ghost_grid=ghost_grid, transport=transport,
        materials=materials, initial=initial,
    )
