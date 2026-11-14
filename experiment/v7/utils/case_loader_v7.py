"""
case_loader_v7.py — load a YAML case + OBJ vertex files into CaseV7.

V5 isolation: minimal self-contained loader; does NOT import utils/sph/.
Replicates the bare-minimum subset of utils/sph/case.py that V5 needs:

  - Parse case.yaml (physics / numerics / capacities / geometry / materials)
  - Parse material YAML library (resolve material names → MaterialParameter)
  - Parse OBJ vertex lines (`v x y z`) for each particle source + frame
  - Compute grid origin / dimension from frame.obj bbox via the same anchoring
    rule as V0 (bbox.min at voxel (0,0,0) center; dim = floor(span/h+0.5)+1)
  - Compose CaseV7 ready to hand to SphSimulatorV7

What is NOT replicated:
  - Closest-packing self-check (V0 case.py has it; V5 trusts caller)
  - Per-source mask filtering (Phase 4 introduces this for multi-GPU)
  - Schema version validation (V5 reads schema_version: 2 directly)
  - INLET / ROTOR kind handling (V1 cavity has none)
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import yaml

from experiment.v7.utils.case_v7 import (
    CaseV7,
    Capacities,
    GhostGridParams,
    GridLayout,
    InitialParticles,
    KIND_BOUNDARY,
    KIND_FLUID,
    VISCOUS_FORM_CHOICES,
    WALL_DENSITY_CHOICES,
    KIND_INLET,
    KIND_ROTOR,
    MaterialParameter,
    NumericsConstants,
    PhysicsConstants,
    TransportConfig,
    WALL_BOUNDARY_CHOICES,
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
        rigid_rotation = spec.get("rigid_rotation") or {}
        if set(rigid_rotation) - {"angular_velocity", "centre"}:
            raise ValueError(f"material {name!r}: rigid_rotation takes angular_velocity and centre")
        rigid_rotation_rate = float(rigid_rotation.get("angular_velocity", 0.0))
        rigid_rotation_centre = tuple(float(value) for value in rigid_rotation.get("centre", [0.0, 0.0, 0.0]))
        if len(rigid_rotation_centre) != 3:
            raise ValueError(f"material {name!r}: rigid_rotation.centre needs three components")
        if rigid_rotation_rate != 0.0 and kind_name != "boundary":
            raise ValueError(f"material {name!r}: rigid_rotation is the velocity field of a boundary material")
        if rigid_rotation_rate != 0.0 and any(initial_velocity):
            raise ValueError(f"material {name!r}: give rigid_rotation or initial_velocity, not both")
        wall_angular_velocity = tuple(float(value) for value in spec.get("wall_angular_velocity", [0.0, 0.0, 0.0]))
        if len(wall_angular_velocity) != 3:
            raise ValueError(f"material {name!r}: wall_angular_velocity needs three components")
        if any(wall_angular_velocity) and kind_name != "boundary":
            raise ValueError(f"material {name!r}: wall_angular_velocity is the prescribed micro-rotation of a "
                             f"boundary material, not of kind {kind_name!r}")
        result.append(MaterialParameter(
            kind=kind,
            rest_density=rest_density,
            viscosity=viscosity,
            eos_constant=eos_constant,
            smoothing_length=physics_h,
            radius=particle_radius,
            volume=volume,
            rotor_angular_velocity=float(spec.get("rotor_angular_velocity", 0.0)),
            viscosity_transfer=float(spec.get("viscosity_transfer", 0.0)),
            viscosity_rotation=float(spec.get("viscosity_rotation", 0.0)),
            initial_velocity=(float(initial_velocity[0]),
                              float(initial_velocity[1]),
                              float(initial_velocity[2])),
            wall_angular_velocity=wall_angular_velocity,
            rigid_rotation_rate=rigid_rotation_rate,
            rigid_rotation_centre=rigid_rotation_centre,
        ))
    return result


MICROPOLAR_STEP_NUMBER_WARNING = 0.1


def micropolar_coefficient_report(material_name: str, viscosity: float, viscosity_transfer: float,
                                  viscosity_rotation: float, micro_inertia: float, timestep: float,
                                  particle_spacing: float) -> dict:
    """The micropolar coefficients of one fluid material in float64: c = nu + nu_t, Eringen's groups, the
    admissible range 0 <= c <= 2 nu and the four explicit step numbers (docs/micropolar_design.md). Raises on
    c < 0, xi < 0, Theta <= 0. dt is never changed here; a step number above MICROPOLAR_STEP_NUMBER_WARNING is
    listed in "warnings"."""
    coupling = viscosity + viscosity_transfer
    if micro_inertia <= 0.0:
        raise ValueError(f"physics.micropolar.micro_inertia = {micro_inertia}: Theta must be > 0")
    if coupling < 0.0:
        raise ValueError(f"material {material_name!r}: c = viscosity + viscosity_transfer = {coupling} < 0")
    if viscosity_rotation < 0.0:
        raise ValueError(f"material {material_name!r}: viscosity_rotation = {viscosity_rotation} < 0")
    report = {
        "material": material_name, "viscosity": viscosity, "viscosity_transfer": viscosity_transfer,
        "coupling_coefficient": coupling, "viscosity_rotation": viscosity_rotation, "micro_inertia": micro_inertia,
        # Eringen: mu + kappa = rho nu, kappa = rho c, gamma = rho xi, j = Theta
        "eringen_kappa_over_mu": coupling / (viscosity - coupling) if viscosity != coupling else None,
        "coupling_number_squared": (coupling / (2.0 * viscosity - coupling)
                                    if 2.0 * viscosity != coupling else None),
        "characteristic_length_squared": (viscosity_rotation * viscosity / (coupling * (2.0 * viscosity - coupling))
                                          if coupling * (2.0 * viscosity - coupling) != 0.0 else None),
        "admissible": 0.0 <= coupling <= 2.0 * viscosity,
        # c > 2 nu: transverse modes with k^2 < c (c - 2 nu) / (nu xi) grow in the continuous model
        "shortest_growing_wavelength": None,
        "step_numbers": {
            "viscous": viscosity * timestep / particle_spacing ** 2,
            "angular_diffusion": viscosity_rotation / micro_inertia * timestep / particle_spacing ** 2,
            "relaxation": 2.0 * coupling * timestep / micro_inertia,
            "coupling": coupling * timestep / (micro_inertia ** 0.5 * particle_spacing),
        },
    }
    if coupling > 2.0 * viscosity:
        report["shortest_growing_wavelength"] = (
            2.0 * np.pi * (viscosity * viscosity_rotation / (coupling * (coupling - 2.0 * viscosity))) ** 0.5)
    report["warnings"] = [name for name, value in report["step_numbers"].items()
                          if value > MICROPOLAR_STEP_NUMBER_WARNING]
    return report


def _print_micropolar_report(report: dict) -> None:
    def text(value) -> str:
        return "undefined" if value is None else f"{value:.6g}"
    print(f"[case_loader_v7] micropolar, material {report['material']}: nu = {report['viscosity']:.6g}, "
          f"nu_t = {report['viscosity_transfer']:.6g}, c = nu + nu_t = {report['coupling_coefficient']:.6g} m^2/s, "
          f"xi = {report['viscosity_rotation']:.6g} m^4/s, Theta = {report['micro_inertia']:.6g} m^2")
    print(f"[case_loader_v7]   Eringen: K = kappa/mu = c/(nu - c) = {text(report['eringen_kappa_over_mu'])}, "
          f"N_c^2 = c/(2 nu - c) = {text(report['coupling_number_squared'])}, "
          f"l^2 = xi nu/(c (2 nu - c)) = {text(report['characteristic_length_squared'])} m^2")
    if report["admissible"]:
        print("[case_loader_v7]   0 <= c <= 2 nu: inside the thermodynamically admissible range")
    else:
        wavelength = report["shortest_growing_wavelength"]
        growing = ("every wavelength grows (xi = 0)" if wavelength == 0.0
                   else f"transverse modes longer than {wavelength:.6g} m grow")
        print(f"[case_loader_v7]   c > 2 nu: OUTSIDE the thermodynamically admissible range (run allowed); in the "
              f"continuous model {growing}")
    numbers = report["step_numbers"]
    print(f"[case_loader_v7]   step numbers: nu dt/dx^2 = {numbers['viscous']:.4g}, (xi/Theta) dt/dx^2 = "
          f"{numbers['angular_diffusion']:.4g}, 2 c dt/Theta = {numbers['relaxation']:.4g}, "
          f"c dt/(sqrt(Theta) dx) = {numbers['coupling']:.4g}")
    for name in report["warnings"]:
        print(f"[case_loader_v7]   WARNING: the {name} step number {numbers[name]:.4g} is above "
              f"{MICROPOLAR_STEP_NUMBER_WARNING} (dt is the acoustic step and is not changed)")


# ============================================================================
# Public loader
# ============================================================================

DEFRAG_ORDER_CHOICES = ("prefix", "atomic")     # case.yaml numerics.defrag_order (E9, 2026-10-27)


def _defrag_order_is_prefix(numerics_mapping: dict, case_path: pathlib.Path) -> bool:
    """numerics.defrag_order: "prefix" (default since E9: the host scan of utils/defrag_prefix.py + defrag.comp's
    prefix branch, pid order = voxel-id order) or "atomic" (the pre-E9 atomicAdd arrival order, kept for A/B runs).
    The legacy key use_prefix_sum_defrag is ignored: the generators wrote 'false' into every case before 2026-10-27
    without anybody choosing it, and 'true' never had a producer of voxel_base_offset (the switch alone leaves the
    buffer unwritten and the alive count 0)."""
    order = str(numerics_mapping.get("defrag_order", "prefix"))
    if order not in DEFRAG_ORDER_CHOICES:
        raise ValueError(f"numerics.defrag_order: {order!r}, expected one of {', '.join(DEFRAG_ORDER_CHOICES)}")
    if "use_prefix_sum_defrag" in numerics_mapping:
        print(f"[case_loader_v7] {case_path.parent.name}: numerics.use_prefix_sum_defrag is a legacy key and is ignored "
              f"(defrag_order = {order}; set numerics.defrag_order to choose)", flush=True)
    return order == "prefix"


def load_case_v7(case_yaml_path: str | pathlib.Path) -> CaseV7:
    """Parse a V0/V1-style case.yaml into a CaseV7.

    Output contract: the returned CaseV7 is a **degenerate slab** that owns
    the whole simulation domain — no peer, ghost pools = 0, ghost voxel
    counts = 0, transport.leading / transport.trailing both None.

    Two consumption modes downstream:

      single-GPU: feed the returned CaseV7 directly into SphSimulatorV7;
                  the simulator's per-direction ghost flow auto-skips because
                  ``case.transport.has_*_peer`` is False on both sides.

      dual-GPU:   hand it to ``partition_v7.compute_dual_gpu_partition`` which
                  splits this global case into per-slab CaseV5s with populated
                  ghost / transport fields.

    Ghost column synthesis is partition_v7's job, not this loader's.
    """
    case_path = pathlib.Path(case_yaml_path).resolve()
    case_dir = case_path.parent
    case_data = yaml.safe_load(case_path.read_text(encoding="utf-8"))

    if case_data.get("schema_version") != 2:
        raise ValueError(
            f"unsupported schema_version: got {case_data.get('schema_version')}; "
            f"V5 loader expects 2")

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
    # Micropolar terms (optional block; absent = off, the Newtonian solver): physics.micropolar: {enabled,
    # micro_inertia}. The coefficients nu_t / xi and the wall value are material keys.
    micropolar_block = phys.get("micropolar") or {}
    unknown_micropolar_keys = set(micropolar_block) - {"enabled", "micro_inertia"}
    if unknown_micropolar_keys:
        raise ValueError(f"physics.micropolar: unknown keys {sorted(unknown_micropolar_keys)}")
    micropolar_enabled = bool(micropolar_block.get("enabled", False))
    # Stage 8b: physics.periodic: [x, y] (absent = [false, false]); a third entry for z must be false.
    raw_periodic = list(phys.get("periodic", [False, False]))
    if len(raw_periodic) == 3:
        if raw_periodic[2]:
            raise ValueError("physics.periodic: z cannot be periodic")
        raw_periodic = raw_periodic[:2]
    if len(raw_periodic) != 2:
        raise ValueError(f"physics.periodic must be [x, y] booleans, got {raw_periodic!r}")
    periodic = (bool(raw_periodic[0]), bool(raw_periodic[1]))
    micro_inertia = float(micropolar_block.get("micro_inertia", 0.0)) if micropolar_enabled else 0.0
    if micropolar_enabled and "micro_inertia" not in micropolar_block:
        raise ValueError("physics.micropolar.enabled needs physics.micropolar.micro_inertia (Theta, m^2)")

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
        micropolar_enabled=micropolar_enabled,
        micro_inertia=micro_inertia,
        periodic=periodic,
    )

    # --- Numerics ---------------------------------------------------------
    nm = case_data["numerics"]
    reg = nm["regularization"]
    numerics = NumericsConstants(
        # KCG Tikhonov ξ (added to the diagonal of M). Optional; default 0.001 since 2026-10-05 (was 0.1,
        # which scaled every corrected operator by M (M + ξ I)⁻¹ ≈ 0.92, docs/validation/cavity_re1000.md).
        regularization_xi=float(reg.get("xi", 0.001)),
        regularization_determinant_threshold=float(reg["det_threshold"]),
        regularization_max_frobenius_norm=float(reg["frobenius_max"]),
        # ε² in the r² + ε² denominators (viscosity, δ-diffusion): epsilon_squared_factor · h², h = support
        # radius. Optional; default 0.0025 since 2026-10-05 = Antuono's 0.01 H² with the smoothing length
        # H = h / 2 (was 0.01, the V0/V1 value, which lowered the viscous operator by ~7 %).
        eps_h_squared=float(nm.get("epsilon_squared_factor", 0.0025)) * h * h,
        pst_main_shift_coefficient=float(nm.get("pst_main", 0.1)),
        pst_anti_shift_coefficient=float(nm.get("pst_anti", 0.005)),
        use_kcg_correction=bool(nm.get("use_kcg_correction", True)),
        use_density_diffusion=bool(nm.get("use_density_diffusion", True)),
        use_pst=bool(nm.get("use_pst", True)),
        use_prefix_sum_defrag=_defrag_order_is_prefix(nm, case_path),
        defrag_cadence=int(nm.get("defrag_cadence", 1000)),
        # E37 wall boundary option: simple (the v6 walls, default) or adami (Adami et al. 2012 wall pressure +
        # no-slip, walls store rho0; one GPU only). Optional key.
        wall_boundary=str(nm.get("wall_boundary", "simple")),
        # the viscous term's discretization (optional key; default = the Monaghan-Gingold form of paper 2)
        viscous_form=str(nm.get("viscous_form", "monaghan_gingold")),
        # the density diffusion between fluid pairs only (optional key; default = with wall neighbours too)
        delta_fluid_only=nm.get("delta_fluid_only", False),
        # what an adami wall stores as its density (optional key; default rho0)
        wall_density=str(nm.get("wall_density", "rho0")),
    )
    if numerics.wall_density not in WALL_DENSITY_CHOICES:
        raise ValueError(f"numerics.wall_density: {numerics.wall_density!r}, expected one of "
                         f"{', '.join(WALL_DENSITY_CHOICES)}")
    if numerics.wall_density == "eos" and numerics.wall_boundary != "adami":
        raise ValueError("numerics.wall_density: eos needs numerics.wall_boundary: adami (a simple wall integrates "
                         "its own density)")
    if not isinstance(numerics.delta_fluid_only, bool):
        raise ValueError(f"numerics.delta_fluid_only: {numerics.delta_fluid_only!r}, expected true or false")
    if numerics.viscous_form not in VISCOUS_FORM_CHOICES:
        raise ValueError(f"numerics.viscous_form: {numerics.viscous_form!r}, expected one of "
                         f"{', '.join(VISCOUS_FORM_CHOICES)}")
    if numerics.wall_boundary not in WALL_BOUNDARY_CHOICES:
        raise ValueError(f"numerics.wall_boundary: {numerics.wall_boundary!r}, expected one of "
                         f"{', '.join(WALL_BOUNDARY_CHOICES)}")

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
    if micropolar_enabled:
        for name, material in zip(used_material_names, materials):
            if dimension == 2 and (material.wall_angular_velocity[0] != 0.0
                                   or material.wall_angular_velocity[1] != 0.0):
                raise ValueError(f"material {name!r}: in a 2-D case the micro-rotation is along z; "
                                 f"wall_angular_velocity = {material.wall_angular_velocity}")
            if material.kind != KIND_FLUID:
                continue
            _print_micropolar_report(micropolar_coefficient_report(
                name, material.viscosity, material.viscosity_transfer, material.viscosity_rotation,
                micro_inertia, timestep, 2.0 * particle_radius))
    # E39 B1: in 2-D every particle stays on z = 0 (positions are flattened below); the fused correction + density
    # kernel drops the z terms of its 2-D accumulators, which is exact only if nothing ever moves a particle in z.
    if dimension == 2:
        moving_in_z = [name for name, material in zip(used_material_names, materials)
                       if material.initial_velocity[2] != 0.0]
        if gravity[2] != 0.0 or moving_in_z:
            raise ValueError(f"2-D case with motion in z: physics.gravity z = {gravity[2]}, initial_velocity z != 0 "
                             f"for {moving_in_z}; a 2-D case keeps every particle on z = 0")

    # Load each source's OBJ; concatenate with material group tagging.
    all_positions_list: list[np.ndarray] = []
    all_velocities_list: list[np.ndarray] = []
    all_material_group_list: list[np.ndarray] = []
    total_loaded = 0
    print(f"[case_loader_v7] loading {case_dir.name}")
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
        if materials[group].rigid_rotation_rate != 0.0:
            # a wall in rigid rotation about z: v = Omega z^ x (r - centre) per particle (float64, then float32)
            rate = materials[group].rigid_rotation_rate
            centre = np.asarray(materials[group].rigid_rotation_centre, dtype=np.float64)
            offset = verts.astype(np.float64) - centre[None, :]
            init_vel = np.column_stack([-rate * offset[:, 1], rate * offset[:, 0],
                                        np.zeros(n)]).astype(np.float32)
        groups = np.full(n, group, dtype=np.uint32)
        all_positions_list.append(verts)
        all_velocities_list.append(init_vel)
        all_material_group_list.append(groups)
        total_loaded += n
        print(f"  - {entry['file']}: {n:,} particles ({entry['material']})")
    positions = np.concatenate(all_positions_list).astype(np.float32)
    velocities = np.concatenate(all_velocities_list).astype(np.float32)
    material_group = np.concatenate(all_material_group_list).astype(np.uint32)
    print(f"[case_loader_v7] total loaded: {total_loaded:,} particles")

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
    if periodic[0] or periodic[1]:
        # Stage 8b: on a periodic axis the bbox is the box: origin = bbox min, dimension = span / h (an integer >= 3,
        # so the three neighbour columns of the sweep are distinct voxels), DOMAIN_LENGTH = span
        origin_list, dim_list, lengths = list(origin_tuple), list(dim_tuple), [1.0, 1.0]
        for axis in (0, 1):
            if not periodic[axis]:
                continue
            span = float(bbox_max[axis]) - float(bbox_min[axis])
            count = int(round(span / h))
            if count < 3 or abs(span / h - count) > 1e-6 * count:
                raise ValueError(f"periodic axis {'xy'[axis]}: span {span!r} / h {h!r} = {span / h!r} must be an "
                                 "integer >= 3")
            origin_list[axis], dim_list[axis], lengths[axis] = float(bbox_min[axis]), count, span
            outside = int(((positions[:, axis] < float(bbox_min[axis]))
                           | (positions[:, axis] >= float(bbox_min[axis]) + span)).sum())
            if outside:
                raise ValueError(f"{outside} initial particles outside the periodic box on axis {'xy'[axis]}")
        origin_tuple, dim_tuple = tuple(origin_list), tuple(dim_list)
        import dataclasses as _dataclasses
        physics = _dataclasses.replace(physics, domain_length=(lengths[0], lengths[1]))
        print(f"[case_loader_v7] periodic {periodic}: grid dim={dim_tuple} origin={origin_tuple} lengths={lengths}")

    # Degenerate slab: owns the whole domain, no ghost columns, no peer.
    # partition_v7 is responsible for synthesizing per-slab ghost / transport.
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

    print(f"[case_loader_v7] grid: dim={dim_tuple} origin={origin_tuple}")

    return CaseV7(
        physics=physics, numerics=numerics, capacities=capacities,
        grid=grid, ghost_grid=ghost_grid, transport=transport,
        materials=materials, initial=initial,
    )
