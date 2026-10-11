"""
simulator_v8.py — SphSimulatorSingle (experiment/v7_single) running experiment/v8's SPIR-V with the periodic
domain's four specialization constants and a per-particle initial density.

Nothing else differs from v7_single: the same buffers, descriptor layouts, recordings, fence ring and
readbacks. With physics.periodic [false, false] the v8 kernels specialize to the v7 code (the periodic branches
are folded away), which experiment/v8/_test_v8_equivalence.py checks on a cavity case.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import numpy as np

from experiment.v7_single.utils.simulator_single import SphSimulatorSingle
from experiment.v7_single.utils.switches import Switches
from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle
from experiment.v8.utils.case_loader_v8 import CaseV8

V8_SPV_DIRECTORY = pathlib.Path(__file__).resolve().parents[1] / "shaders" / "spv"


class SphSimulatorV8(SphSimulatorSingle):
    """v7_single's simulator on the v8 kernels (module docstring)."""

    def __init__(self, ctx: VulkanContextSingle, case_v8: CaseV8, *, switches: Optional[Switches] = None,
                 spirv_directory: Optional[pathlib.Path] = None) -> None:
        self.case_v8 = case_v8
        super().__init__(ctx, case_v8.case, switches=switches, spirv_directory=spirv_directory or V8_SPV_DIRECTORY)
        if case_v8.any_periodic and self.deep_wall_active:
            raise ValueError("periodic axes with V7_DEEP_WALL_SKIP on: deep_wall_marker.comp does not wrap "
                             "(no walls in a periodic run; set V7_DEEP_WALL_SKIP=0)")
        print(f"[SimV8] periodic={case_v8.periodic} domain_length={case_v8.domain_length} "
              f"initial_density={'per particle' if case_v8.initial_density is not None else 'rest density'} "
              f"delta_density={case_v8.delta_density} (rho_ref {self.reference_density():g})")

    def stored_density_offset(self) -> float:
        """What to add to a stored density to get rho: rho_ref with numerics.delta_density, 0 otherwise."""
        return self.reference_density() if self.case_v8.delta_density else 0.0

    def global_entries(self) -> list[tuple[int, str, Any]]:
        entries = [entry for entry in super().global_entries() if entry[0] != 95]
        entries += [(95, "B", int(self.case_v8.delta_density)),
                    (116, "B", int(self.case_v8.periodic[0])), (117, "B", int(self.case_v8.periodic[1])),
                    (118, "f", float(self.case_v8.domain_length[0])), (119, "f", float(self.case_v8.domain_length[1]))]
        return entries

    def _build_initial_data(self) -> dict[str, bytes]:
        """v7_single's payload, then the stored density column: the per-particle initial density (or every material's
        rho0) minus the stored offset, computed in float64 and cast once."""
        data = super()._build_initial_data()
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        count = int(case.initial.positions.shape[0])
        initial_density = self.case_v8.initial_density
        if initial_density is None:
            rest = np.array([material.rest_density for material in case.materials], dtype=np.float64)
            initial_density = rest[np.asarray(case.initial.material_group)]
        if initial_density.shape != (count,):
            raise ValueError(f"initial_density shape {initial_density.shape}, expected ({count},)")
        density_pressure = np.frombuffer(data["density_pressure"], dtype=np.float32).reshape(pool_capacity, 2).copy()
        density_pressure[own_first:own_first + count, 0] = (
            np.asarray(initial_density, dtype=np.float64) - self.stored_density_offset()).astype(np.float32)
        data["density_pressure"] = density_pressure.tobytes()
        return data

    def resolution_record(self) -> dict:
        record = super().resolution_record()
        record.update({"periodic": list(self.case_v8.periodic), "domain_length": list(self.case_v8.domain_length),
                       "initial_field": self.case_v8.field, "build": "v8", "delta_density": self.case_v8.delta_density,
                       "stored_density_offset": self.stored_density_offset()})
        return record
