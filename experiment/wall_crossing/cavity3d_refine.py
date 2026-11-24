"""
cavity3d_refine.py - claude/wall_store_report.md section 15 (round 9): the 3-D cavity at Re = 1000 refined level by
level (N = 100 -> 140 -> 170 and 100 -> 200 -> ...), every level started from the previous level's converged state
and stopped as soon as it has converged. Used by box_run.py (--initial-state, --tube-time, --converge); nothing here
changes the solver.

Interpolated start (box_run --initial-state <previous final.npz>):
  the new level's particles are its own case's (the generator's lattice and walls, not the old particles refined);
  velocity and pressure come from the previous level's fluid by linear moving least squares (Wendland C4 of support
  --initial-support, the previous level's h; weight W V_j; basis [1, (x - p) / support]; fluid particles only, so the
  fit is one-sided near a wall); the density from the pressure by the inverse of the solver's Tait law,
      rho = rho0 (1 + p / B)^(1 / gamma),   B = c0^2 rho0 / gamma (the material's eos_constant);
  the mass rho0 V (the material's calibrated volume, as the case's initial payload). The state goes to the solver's
  restart path (SphSimulatorV7.restart_init via chain_runner.run_chain(restart=...)) as a step-0 state:
  acceleration, shift, the correction fields and the extension fields zero (the correction fields are recomputed
  before every read; with a = 0 the first step only drifts), the voxel ids rebuilt by initialize_voxelization.
  The volume constraint: only the integral term I of the previous level is handed over
  (_volume_constraint_integral); v_ref is computed from the new case's initial particles as in a fresh start (the
  solver prints that the "checkpoint" has no v_ref, and the rate is 0 until the first update).

Tubes (box_run --tube-time T): every T the fluid particles within a square tube of half-width
--tube-half-width x h around each of the three centre lines (vertical along y, horizontal along x, spanwise along
z, through the box centre) go to tubes/t<step>.npz: positions, velocities, pressures, densities, masses (float32) and
a membership bit per line (1 vertical, 2 horizontal, 4 spanwise).

Convergence (box_run --converge; report section 15, the brief's rule 3.2): every record (the --lines-time samples,
~0.5) gives the kinetic energy and the three in-plane extrema of the centre lines (the round-8 MLS method; only their
relative changes matter here) and |w|max (recorded, not part of the rule). The level time (steps x dt of this level)
is cut into blocks of --converge-block (2); when a block is complete its mean is compared with the previous block's:
    |mean_k - mean_(k-1)| / |mean_k| < max(tolerance, sigmas x standard error of block k's mean / |mean_k|)
for the energy and the three extrema (tolerance 0.002, sigmas 3). Two consecutive passes, and a level time of at
least --converge-minimum (6), stop the level: the restart checkpoint and final.npz are written and box_run exits 0.
A level that has not converged by --time (25) ends there (summary.json says so). convergence.jsonl keeps every
record and every comparison.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from scipy.spatial import cKDTree

LINE_BITS = {"vertical": 1, "horizontal": 2, "spanwise": 4}
LINE_AXES = {"vertical": 1, "horizontal": 0, "spanwise": 2}
CRITERION_QUANTITIES = ("kinetic_energy", "u_min", "v_max", "v_min")


def wendland_c4(normalized_distance: np.ndarray) -> np.ndarray:
    """The solver's Wendland C4 profile without its constant (it cancels in moving least squares)."""
    one_minus = np.clip(1.0 - normalized_distance, 0.0, None)
    return one_minus ** 6 * ((35.0 / 3.0) * normalized_distance ** 2 + 6.0 * normalized_distance + 1.0)


def linear_moving_least_squares(source_positions: np.ndarray, source_volumes: np.ndarray, source_values: np.ndarray,
                                targets: np.ndarray, support: float, neighbour_limit: int = 400,
                                chunk_size: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Linear MLS of source_values (M, C) at targets (P, 3) with the neighbours within support (at most
    neighbour_limit nearest; the farthest of a fuller support would carry weights near 0). Returns (values (P, C),
    neighbour counts (P,)). A target with fewer than 4 neighbours or a singular moment matrix gets the Shepard
    average (or NaN without neighbours)."""
    tree = cKDTree(source_positions)
    component_count = source_values.shape[1]
    values = np.full((targets.shape[0], component_count), np.nan)
    counts = np.zeros(targets.shape[0], dtype=np.int64)
    for first in range(0, targets.shape[0], chunk_size):
        block = targets[first:first + chunk_size]
        distance, index = tree.query(block, k=neighbour_limit, distance_upper_bound=support, workers=-1)
        valid = np.isfinite(distance)
        index = np.where(valid, index, 0)
        weight = np.where(valid, wendland_c4(np.where(valid, distance, support) / support), 0.0) * \
            source_volumes[index]
        offset = (source_positions[index] - block[:, None, :]) / support
        basis = np.concatenate([np.ones(offset.shape[:2] + (1,)), offset], axis=2)        # (block, k, 4)
        weighted_basis = basis * weight[:, :, None]
        moment = np.einsum("bki,bkj->bij", weighted_basis, basis)
        right = np.einsum("bki,bkc->bic", weighted_basis, source_values[index])
        counts[first:first + block.shape[0]] = valid.sum(axis=1)
        determinant = np.linalg.det(moment)
        scale = moment[:, 0, 0] ** 4
        good = (valid.sum(axis=1) >= 4) & (np.abs(determinant) > 1e-12 * np.maximum(scale, 1e-300))
        result = np.full((block.shape[0], component_count), np.nan)
        if good.any():
            result[good] = np.linalg.solve(moment[good], right[good])[:, 0, :]
        shepard = ~good & (moment[:, 0, 0] > 0.0)
        result[shepard] = right[shepard, 0, :] / moment[shepard, 0, 0][:, None]
        values[first:first + block.shape[0]] = result
    return values, counts


def interpolated_restart(case, previous_path: pathlib.Path, support: float, integral: float | None) -> dict:
    """The step-0 restart state of case (its own fluid particles) with velocity and pressure interpolated from the
    previous level's fluid state previous_path (final.npz: positions, velocities, pressures, densities, masses), as
    run_chain(restart=...) takes it: {"state", "stored_offset", "step", "volume_constraint", "report"}."""
    from experiment.v7.utils.case_v7 import KIND_FLUID
    with np.load(previous_path) as archive:
        previous = {key: archive[key].astype(np.float64) for key in ("positions", "velocities", "pressures",
                                                                      "densities")}
        # every fluid particle has the same mass rho0 V; a constant mass cancels in the weights W m_j / rho_j, so a
        # state without masses (box_run's final.npz before report section 15) is read with mass 1
        previous["masses"] = archive["masses"].astype(np.float64) if "masses" in archive.files \
            else np.ones(previous["positions"].shape[0])
    positions = np.asarray(case.initial.positions, dtype=np.float64)
    groups = np.asarray(case.initial.material_group).astype(np.int64)
    kinds = np.asarray([material.kind for material in case.materials])
    if not np.all(kinds[groups] == KIND_FLUID):
        raise ValueError("interpolated_restart: the case's initial particles are not all fluid (a crossing case "
                         "holds its walls in the wall store; simple / adami cases are not supported here)")
    fluid = case.materials[int(groups[0])]
    rest_density, eos_constant = float(fluid.rest_density), float(fluid.eos_constant)
    power = float(case.physics.power_parameter)
    source_values = np.column_stack([previous["velocities"], previous["pressures"]])
    # the neighbours within the support: about 4/3 pi (support / spacing)^3 of the previous lattice; twice that (and 20
    # more) as the limit of the nearest-neighbour query (a compressed region has more), at most 400
    previous_spacing = (np.prod(previous["positions"].max(axis=0) - previous["positions"].min(axis=0))
                        / previous["positions"].shape[0]) ** (1.0 / 3.0)
    expected = 4.0 / 3.0 * np.pi * (support / previous_spacing) ** 3
    neighbour_limit = int(min(400, 2.0 * expected + 20))
    values, counts = linear_moving_least_squares(previous["positions"], previous["masses"] / previous["densities"],
                                                 source_values, positions, support, neighbour_limit=neighbour_limit)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"interpolated_restart: {int((~np.isfinite(values)).any(axis=1).sum())} particles got no "
                         "value (no previous particle within the support)")
    velocities, pressures = values[:, :3], values[:, 3]
    densities = rest_density * np.power(1.0 + pressures / eos_constant, 1.0 / power)
    tait = eos_constant * (np.power(densities / rest_density, power) - 1.0)
    masses = np.array([case.materials[group].rest_density * case.materials[group].volume for group in groups])
    count = positions.shape[0]
    state = {
        "position_voxel_id": np.column_stack([positions, np.zeros(count)]).astype(np.float32),
        "velocity_mass": np.column_stack([velocities, masses]).astype(np.float32),
        "density_pressure": np.column_stack([densities, tait]).astype(np.float32),
        "acceleration": np.zeros((count, 4), dtype=np.float32),
        "shift": np.zeros((count, 4), dtype=np.float32),
        "material": groups.astype(np.uint32),
        "correction_inverse": np.zeros((count, 8), dtype=np.float32),
        "density_gradient_kernel_sum": np.zeros((count, 4), dtype=np.float32),
        "extension_fields": np.zeros((count, 4), dtype=np.float32),
    }
    speed = np.sqrt((velocities ** 2).sum(axis=1))
    previous_speed = np.sqrt((previous["velocities"] ** 2).sum(axis=1))
    report = {
        "previous": str(previous_path), "support": float(support), "particles": int(count),
        "previous_particles": int(previous["positions"].shape[0]),
        "neighbours_minimum": int(counts.min()), "neighbours_median": float(np.median(counts)),
        "speed_maximum": float(speed.max()), "previous_speed_maximum": float(previous_speed.max()),
        "specific_energy": float(0.5 * (speed ** 2).mean()),
        "previous_specific_energy": float(0.5 * (previous_speed ** 2).mean()),
        "density_minimum": float(densities.min()), "density_maximum": float(densities.max()),
        "previous_density_minimum": float(previous["densities"].min()),
        "previous_density_maximum": float(previous["densities"].max()),
        "pressure_mean": float(pressures.mean()), "previous_pressure_mean": float(previous["pressures"].mean()),
        "neighbour_limit": neighbour_limit, "neighbours_maximum": int(counts.max()),
        "integral": integral,
    }
    volume_constraint = {"_volume_constraint_integral": float(integral)} if integral is not None else None
    return {"state": state, "stored_offset": 0.0, "step": 0, "volume_constraint": volume_constraint,
            "report": report}


def last_integral(volume_constraint_log: pathlib.Path) -> float | None:
    """I of the integral term after the last update of a run's volume_constraint.jsonl (None without records)."""
    value = None
    if volume_constraint_log.exists():
        for line in volume_constraint_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "integral" in record:
                    value = float(record["integral"])
    return value


def tube_membership(positions: np.ndarray, centre: np.ndarray, half_width: float) -> np.ndarray:
    """Bit mask per particle: LINE_BITS of the square tubes (half-width around each centre line) it lies in."""
    bits = np.zeros(positions.shape[0], dtype=np.uint8)
    for name, axis in LINE_AXES.items():
        inside = np.ones(positions.shape[0], dtype=bool)
        for other in range(3):
            if other != axis:
                inside &= np.abs(positions[:, other] - centre[other]) <= half_width
        bits[inside] |= LINE_BITS[name]
    return bits


def tube_state(state: dict, centre: np.ndarray, half_width: float) -> dict:
    """The fluid particles of state inside any of the three tubes (same keys, plus "membership")."""
    bits = tube_membership(state["positions"], centre, half_width)
    keep = bits > 0
    out = {key: np.asarray(state[key])[keep] for key in ("positions", "velocities", "pressures", "densities", "masses")}
    out["membership"] = bits[keep]
    return out


def write_tubes(path: pathlib.Path, step: int, time_value: float, level_time: float, tubes: dict,
                half_width: float) -> None:
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, step=np.int64(step), time=np.float64(time_value), level_time=np.float64(level_time),
             half_width=np.float64(half_width),
             positions=tubes["positions"].astype(np.float32), velocities=tubes["velocities"].astype(np.float32),
             pressures=tubes["pressures"].astype(np.float32), densities=tubes["densities"].astype(np.float32),
             masses=tubes["masses"].astype(np.float32), membership=tubes["membership"])
    temporary.replace(path)


class ConvergenceMonitor:
    """The brief's rule 3.2 on the records of one level (see the module docstring)."""

    def __init__(self, path: pathlib.Path, block: float = 2.0, minimum: float = 6.0, tolerance: float = 0.002,
                 sigmas: float = 3.0):
        self.path = path
        self.block, self.minimum, self.tolerance, self.sigmas = float(block), float(minimum), float(tolerance), \
            float(sigmas)
        self.records: list = []
        self.comparisons: list = []
        self.converged_at: float | None = None
        path.unlink(missing_ok=True)

    def _write(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")

    def block_of(self, level_time: float) -> int:
        return int(math.floor(level_time / self.block + 1e-9))

    def block_statistics(self, index: int) -> dict:
        rows = [record for record in self.records if self.block_of(record["level_time"]) == index]
        out = {"block": index, "records": len(rows)}
        for name in CRITERION_QUANTITIES + ("w_abs_max",):
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            mean = float(values.mean())
            error = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")
            out[name] = {"mean": mean, "standard_error": error}
        return out

    def add(self, step: int, time_value: float, level_time: float, kinetic_energy: float, extrema: dict) -> bool:
        """One record (level_time > 0; the start state is not recorded). Returns True when the level has converged."""
        record = {"kind": "record", "step": int(step), "time": float(time_value), "level_time": float(level_time),
                  "kinetic_energy": float(kinetic_energy),
                  **{name: float(extrema[name]["value"]) for name in ("u_min", "v_max", "v_min", "w_abs_max")}}
        self._write(record)
        completed = self.block_of(level_time) - 1          # this record opens block k + 1: block k is complete
        if self.records and self.block_of(self.records[-1]["level_time"]) < self.block_of(level_time) \
                and completed >= 1:
            current, before = self.block_statistics(completed), self.block_statistics(completed - 1)
            passes, detail = True, {}
            for name in CRITERION_QUANTITIES:
                mean = current[name]["mean"]
                change = abs(mean - before[name]["mean"]) / abs(mean)
                noise = self.sigmas * current[name]["standard_error"] / abs(mean)
                limit = max(self.tolerance, noise if math.isfinite(noise) else 0.0)
                detail[name] = {"change": change, "limit": limit, "pass": bool(change < limit)}
                passes &= change < limit
            consecutive = bool(passes and self.comparisons and self.comparisons[-1]["pass"])
            comparison = {"kind": "comparison", "block": completed, "level_time": float(level_time),
                          "pass": bool(passes), "consecutive": consecutive, "detail": detail,
                          "current": current, "before": before}
            self.comparisons.append(comparison)
            self._write(comparison)
            if consecutive and level_time >= self.minimum - 1e-9 and self.converged_at is None:
                self.converged_at = float(level_time)
                self._write({"kind": "converged", "level_time": float(level_time), "step": int(step),
                             "time": float(time_value)})
        self.records.append(record)
        return self.converged_at is not None


def remove_level_files(run_directory: pathlib.Path) -> list:
    """Delete a finished level's large files (final.npz and the checkpoints); returns what was deleted."""
    removed = []
    for path in [run_directory / "final.npz", *sorted((run_directory / "checkpoints").glob("c*.npz"))]:
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            removed.append((str(path), size))
    return removed
