"""
case_loader_v8.py — v7's case loader plus the two things experiment/v8 adds to a case.yaml:

  physics.periodic: [bool, bool]      the x / y axes are periodic (z never: 2-D cases keep z = 0). On a periodic
                                      axis the frame.obj bounding box IS the box: origin = bbox min, grid
                                      dimension = span / h (asserted integer and >= 3), DOMAIN_LENGTH = span;
                                      the other axes keep v7's rule (bbox min at the centre of voxel 0).
  initial_field:                      a per-particle initial state for the FLUID particles:
    type: taylor_green                experiment/v8/utils/taylor_green.py's (u, v, p) at t = 0
    amplitude: U0                     velocity amplitude
    vortex_pairs: 1                   k = 2 pi n / L (L = the periodic span in x; the box must be square)
    origin_offset: [0.25, 0.25]       the field's origin at bbox min + f L (0 = vortex-cell edges on the box edges,
                                      where nothing crosses; 0.25 = the fastest streams cross the box edges)
    density_from_pressure: true       initial stored density rho = rho0 (1 + p / B)^(1 / gamma) (Tait inverted,
                                      B = c0^2 rho0 / gamma), so the pressure the bootstrap computes is the
                                      analytic p(x, 0) instead of 0; false = rho0 everywhere

Everything else is v7's (experiment.v7.utils.case_loader_v7.load_case_v7, called first). The result wraps the
CaseV7 in a CaseV8 with the periodic flags, the box lengths (spec constants 116-119) and the initial density.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from experiment.v7.utils.case_loader_v7 import _parse_obj_vertices, load_case_v7
from experiment.v7.utils.case_v7 import CaseV7, GridLayout, KIND_FLUID
from experiment.v8.utils import taylor_green

MINIMUM_PERIODIC_VOXELS = 3        # the 3^d sweep's -1 / 0 / +1 columns must be distinct voxels


@dataclass
class CaseV8:
    case: CaseV7
    periodic: tuple[bool, bool]
    domain_length: tuple[float, float]         # spec 118 / 119 (the span on a periodic axis, 1.0 otherwise)
    initial_density: Optional[np.ndarray]      # (N,) float32 stored density at upload, None = every material's rho0
    field: dict                                # the initial field's description (run metadata)

    @property
    def any_periodic(self) -> bool:
        return self.periodic[0] or self.periodic[1]


def _periodic_grid(case: CaseV7, periodic: tuple[bool, bool], bbox_min: np.ndarray, bbox_max: np.ndarray
                   ) -> tuple[GridLayout, tuple[float, float]]:
    h = float(case.physics.smoothing_length)
    origin = [case.grid.origin_x, case.grid.origin_y, case.grid.origin_z]
    dims = [case.grid.grid_dimension_x, case.grid.grid_dimension_y, case.grid.grid_dimension_z]
    lengths = [1.0, 1.0]
    for axis in (0, 1):
        if not periodic[axis]:
            continue
        span = float(bbox_max[axis]) - float(bbox_min[axis])
        count = int(round(span / h))
        if count < MINIMUM_PERIODIC_VOXELS or abs(span / h - count) > 1e-6 * count:
            raise ValueError(f"periodic axis {'xy'[axis]}: span {span!r} / h {h!r} = {span / h!r} must be an integer "
                             f">= {MINIMUM_PERIODIC_VOXELS}")
        origin[axis] = float(bbox_min[axis])
        dims[axis] = count
        lengths[axis] = span
    grid = GridLayout(origin_x=origin[0], origin_y=origin[1], origin_z=origin[2],
                      grid_dimension_x=dims[0], grid_dimension_y=dims[1], grid_dimension_z=dims[2],
                      voxel_order=case.grid.voxel_order)
    return grid, (lengths[0], lengths[1])


def _taylor_green_initial_state(case: CaseV7, spec: dict, periodic: tuple[bool, bool], bbox_min: np.ndarray,
                                lengths: tuple[float, float]) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    if not (periodic[0] and periodic[1]):
        raise ValueError("initial_field taylor_green needs physics.periodic: [true, true]")
    if abs(lengths[0] - lengths[1]) > 1e-9 * lengths[0]:
        raise ValueError(f"initial_field taylor_green needs a square box, got {lengths}")
    amplitude = float(spec.get("amplitude", 1.0))
    vortex_pairs = int(spec.get("vortex_pairs", 1))
    density_from_pressure = bool(spec.get("density_from_pressure", True))
    # origin_offset [fx, fy] (fractions of L): the field's origin sits at bbox min + f L. With the default 0 the
    # vortex-cell edges (u = 0 lines) coincide with the periodic box edges and no particle ever crosses them; 0.25
    # puts the fastest streams on the box edges, so the periodic wrap is exercised by every run.
    offset = [float(value) for value in spec.get("origin_offset", [0.0, 0.0])]
    length = lengths[0]
    k = taylor_green.wavenumber(length, vortex_pairs)
    kinds = np.array([material.kind for material in case.materials])[np.asarray(case.initial.material_group)]
    fluid = kinds == KIND_FLUID
    fluid_materials = [material for material in case.materials if material.kind == KIND_FLUID]
    if not fluid_materials:
        raise ValueError("initial_field taylor_green: no fluid material")
    rest_density = float(fluid_materials[0].rest_density)
    eos_constant = float(fluid_materials[0].eos_constant)
    viscosity = float(fluid_materials[0].viscosity)
    gamma = float(case.physics.power_parameter)
    x0, y0 = float(bbox_min[0]) + offset[0] * length, float(bbox_min[1]) + offset[1] * length
    positions = np.asarray(case.initial.positions, dtype=np.float64)
    velocities = np.asarray(case.initial.velocities, dtype=np.float64).copy()
    field_velocity = taylor_green.velocity(positions[fluid, 0], positions[fluid, 1], 0.0, amplitude=amplitude,
                                           viscosity=viscosity, k=k, x0=x0, y0=y0)
    velocities[fluid, 0:2] = field_velocity
    velocities[fluid, 2] = 0.0
    initial_density = None
    if density_from_pressure:
        field_pressure = taylor_green.pressure(positions[fluid, 0], positions[fluid, 1], 0.0, amplitude=amplitude,
                                               viscosity=viscosity, k=k, rest_density=rest_density, x0=x0, y0=y0)
        rest_of_each = np.array([material.rest_density for material in case.materials], dtype=np.float64)
        initial_density = rest_of_each[np.asarray(case.initial.material_group)].copy()
        # Tait: p = B ((rho / rho0)^gamma - 1)  ->  rho = rho0 (1 + p / B)^(1 / gamma)
        initial_density[fluid] = rest_density * np.power(1.0 + field_pressure / eos_constant, 1.0 / gamma)
        initial_density = initial_density.astype(np.float32)
    speed_of_sound = float(case.physics.speed_of_sound)
    description = {
        "type": "taylor_green", "amplitude": amplitude, "vortex_pairs": vortex_pairs, "wavenumber": k,
        "length": length, "origin": [x0, y0], "origin_offset": offset, "box_min": [float(bbox_min[0]), float(bbox_min[1])],
        "rest_density": rest_density, "viscosity": viscosity,
        "reynolds": amplitude * length / viscosity if viscosity > 0 else math.inf,
        "mach": amplitude / speed_of_sound, "speed_of_sound": speed_of_sound, "eos_constant": eos_constant,
        "density_from_pressure": density_from_pressure,
        "decay_rate": 2.0 * viscosity * k * k,            # F(t) = exp(-decay_rate t)
        "kinetic_energy_0": taylor_green.kinetic_energy(0.0, amplitude=amplitude, viscosity=viscosity, k=k,
                                                        rest_density=rest_density, length=length),
        "fluid_particles": int(fluid.sum()),
        "momentum_balance_residual": taylor_green.momentum_balance_residual(amplitude=amplitude, k=k,
                                                                             rest_density=rest_density),
    }
    return velocities.astype(np.float32), initial_density, description


def load_case_v8(case_yaml_path: str | pathlib.Path) -> CaseV8:
    """v7's loader, then the periodic grid and the initial field (module docstring)."""
    case_path = pathlib.Path(case_yaml_path).resolve()
    case = load_case_v7(case_path)
    data = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    raw_periodic = list(data["physics"].get("periodic", [False, False]))
    if len(raw_periodic) == 3:
        if raw_periodic[2]:
            raise ValueError("physics.periodic: z cannot be periodic (2-D cases keep z = 0; no 3-D periodic axis in v8)")
        raw_periodic = raw_periodic[:2]
    if len(raw_periodic) != 2:
        raise ValueError(f"physics.periodic must be [x, y] booleans, got {raw_periodic!r}")
    periodic = (bool(raw_periodic[0]), bool(raw_periodic[1]))
    lengths = (1.0, 1.0)
    frame_vertices = _parse_obj_vertices((case_path.parent / data["geometry"]["frame"]).resolve())
    bbox_min = frame_vertices.min(axis=0).astype(np.float64)
    bbox_max = frame_vertices.max(axis=0).astype(np.float64)
    if periodic[0] or periodic[1]:
        grid, lengths = _periodic_grid(case, periodic, bbox_min, bbox_max)
        case = dataclasses.replace(case, grid=grid)
        positions = np.asarray(case.initial.positions, dtype=np.float64)
        for axis in (0, 1):
            if periodic[axis]:
                low, high = float(bbox_min[axis]), float(bbox_min[axis]) + lengths[axis]
                outside = int(((positions[:, axis] < low) | (positions[:, axis] >= high)).sum())
                if outside:
                    raise ValueError(f"{outside} initial particles outside the periodic box on axis {'xy'[axis]} "
                                     f"[{low}, {high})")
        print(f"[case_loader_v8] periodic {periodic}: grid dim=({grid.grid_dimension_x}, {grid.grid_dimension_y}, "
              f"{grid.grid_dimension_z}) origin=({grid.origin_x}, {grid.origin_y}, {grid.origin_z}) "
              f"lengths={lengths}")

    initial_density = None
    field: dict = {"type": "none"}
    spec = data.get("initial_field")
    if spec:
        kind = str(spec.get("type", ""))
        if kind != "taylor_green":
            raise ValueError(f"initial_field.type {kind!r}: only taylor_green is implemented")
        velocities, initial_density, field = _taylor_green_initial_state(case, spec, periodic, bbox_min, lengths)
        case = dataclasses.replace(case, initial=dataclasses.replace(case.initial, velocities=velocities))
        print(f"[case_loader_v8] initial field taylor_green: U0={field['amplitude']} k={field['wavenumber']:.6f} "
              f"Re={field['reynolds']:.6g} Ma={field['mach']:.4g} density_from_pressure={field['density_from_pressure']} "
              f"(momentum balance residual {field['momentum_balance_residual']:.2e})")
    return CaseV8(case=case, periodic=periodic, domain_length=lengths, initial_density=initial_density, field=field)
