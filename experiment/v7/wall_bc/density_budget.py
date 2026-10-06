"""density_budget.py - where the fluid's mean density changes: density.comp's right-hand side recomputed in float64
from a k1_dump.py state (v6, or v7 with either wall condition) and split by term and neighbour kind.

For every fluid particle i (the state's positions, stored half-step velocities, densities, masses and the dumped
KCG inverse M_i^-1 of the last correction pass), over its neighbours j within the support radius h:
    drift_i     = - rho_i sum_j V_j (v_j - v_i) . M_i^-1 grad W_ij              V_j = m_j / rho_j
    diffusion_i =   delta h c0 sum_j 2 (rho_j - rho_i) ((r_j - r_i) . M_i^-1 grad W_ij) / (r^2 + eps^2) V_j
(the linear psi of density.comp), each split into j = fluid and j = wall / lid. Reports the fluid mean of every part
(the mean density rate, kg/m^3 per time unit) and its split by region (within h of the lid, of the side or bottom
walls, both = the two top corners, interior), next to the mean-density rate the monitor measured over the run.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.density_budget logs/e36/t5/v6_t5.npz logs/e36/t5/v7_bc1_t5.npz
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def kernel_gradient_2d(relative_position: np.ndarray, distance: np.ndarray, support: float) -> np.ndarray:
    """grad_i W(r_i - r_j), 2-D Wendland C4 with support radius ``support`` (helpers.glsl)."""
    q = distance / support
    scalar = 9.0 / (np.pi * support ** 3) * (1.0 - q) ** 5 * q * ((-280.0 / 3.0) * q - 56.0 / 3.0)
    return scalar[:, None] * relative_position / distance[:, None]


def budget(path: pathlib.Path) -> dict:
    with np.load(path) as archive:
        state = {name: archive[name] for name in archive.files if name != "meta"}
        meta = json.loads(str(archive["meta"]))
    case_module = f"experiment.{meta['solver']}.utils"
    import importlib
    case = getattr(importlib.import_module(f"{case_module}.case_loader_{meta['solver']}"),
                   f"load_case_{meta['solver']}")(str(_REPO_ROOT / meta["case"]))
    kinds = np.array([material.kind for material in case.materials])
    fluid_kind = importlib.import_module(f"{case_module}.case_{meta['solver']}").KIND_FLUID
    support = float(case.physics.smoothing_length)
    epsilon_squared = float(case.numerics.eps_h_squared)
    delta = float(case.physics.delta_coefficient)
    sound = float(case.physics.speed_of_sound)
    spacing = float(meta["spacing"])

    material = state["material"].astype(np.int64)
    positions = state["position_voxel_id"][:, :2].astype(np.float64)
    velocities = state["velocity_mass"][:, :2].astype(np.float64)
    mass = state["velocity_mass"][:, 3].astype(np.float64)
    density = state["density_pressure"][:, 0].astype(np.float64) + float(meta["stored_density_offset"])
    volume = mass / density
    inverse = state["correction_inverse"].astype(np.float64)              # (m00, m11, m22, m01, m02, m12, _, _)
    fluid = np.flatnonzero(kinds[material] == fluid_kind)
    is_fluid = kinds[material] == fluid_kind

    tree = cKDTree(positions)
    pairs = tree.query_pairs(support, output_type="ndarray")
    i_index = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j_index = np.concatenate([pairs[:, 1], pairs[:, 0]])
    keep = is_fluid[i_index]
    i_index, j_index = i_index[keep], j_index[keep]
    relative = positions[i_index] - positions[j_index]                     # x_ij = r_i - r_j
    distance = np.linalg.norm(relative, axis=1)
    ok = (distance < support) & (distance >= 1e-12)
    i_index, j_index, relative, distance = i_index[ok], j_index[ok], relative[ok], distance[ok]
    gradient = kernel_gradient_2d(relative, distance, support)
    m00, m11, m01 = inverse[i_index, 0], inverse[i_index, 1], inverse[i_index, 3]
    corrected = np.stack([m00 * gradient[:, 0] + m01 * gradient[:, 1], m01 * gradient[:, 0] + m11 * gradient[:, 1]], axis=1)
    drift = -density[i_index] * volume[j_index] * np.einsum(
        "ni,ni->n", velocities[j_index] - velocities[i_index], corrected)
    diffusion = (delta * support * sound * 2.0 * (density[j_index] - density[i_index])
                 * np.einsum("ni,ni->n", -relative, corrected) / (distance ** 2 + epsilon_squared) * volume[j_index])
    wall_neighbour = ~is_fluid[j_index]

    fluid_count = fluid.size
    parts = {}
    for name, values in (("drift", drift), ("diffusion", diffusion)):
        for neighbour, mask in (("fluid", ~wall_neighbour), ("wall", wall_neighbour)):
            per_particle = np.bincount(i_index[mask], weights=values[mask], minlength=positions.shape[0])[fluid]
            parts[f"{name}_{neighbour}"] = per_particle
    total = sum(parts.values())
    # regions by the distance to the wall rows (innermost rows at |x|, y = 0.5 + dx, lid at y = 0.5 + dx)
    x, y = positions[fluid, 0], positions[fluid, 1]
    near_lid = (0.5 + spacing - y) < support
    near_side_or_bottom = ((0.5 + spacing - np.abs(x)) < support) | ((y + 0.5 + spacing) < support)
    regions = {"top corners (lid and side wall)": near_lid & near_side_or_bottom,
               "under the lid": near_lid & ~near_side_or_bottom,
               "side / bottom walls": near_side_or_bottom & ~near_lid,
               "interior": ~near_lid & ~near_side_or_bottom}
    rates = {name: float(values.mean()) for name, values in parts.items()}
    rates["total"] = float(total.mean())
    by_region = {}
    for region, mask in regions.items():
        by_region[region] = {"particles": int(mask.sum()),
                             **{name: float(values[mask].sum() / fluid_count) for name, values in parts.items()},
                             "total": float(total[mask].sum() / fluid_count)}
    monitor = path.with_suffix(".monitor.jsonl")
    measured = None
    if monitor.exists():
        rows = [json.loads(line) for line in monitor.read_text(encoding="utf-8").splitlines() if line]
        steps = np.array([row["step"] for row in rows], dtype=np.float64)
        means = np.array([row["fluid_density_mean"] for row in rows])
        late = steps * meta["dt"] >= 1.0
        measured = float(np.polyfit(steps[late] * meta["dt"], means[late], 1)[0])
    return {"dump": path.name, "solver": meta["solver"], "wall_bc": meta["wall_bc"], "time": meta["steps"] * meta["dt"],
            "fluid_particles": int(fluid_count), "fluid_density_mean": float(density[fluid].mean()),
            "mean_rate_by_term": rates, "mean_rate_by_region": by_region,
            "measured_mean_density_rate_t_ge_1": measured}


def main() -> int:
    for argument in sys.argv[1:]:
        report = budget(pathlib.Path(argument).resolve())
        print(json.dumps(report, indent=1))
        pathlib.Path(argument).with_suffix(".budget.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
