"""effective_viscosity.py - the effective viscosity of the solver's discrete viscous operator, on the cavity runs' particles.

force.comp computes the viscous acceleration (Monaghan form with the KCG-corrected kernel gradient)
    a_i = sum_{j != i} 2 (d + 2) nu V_j ((v_i - v_j) . x_ij) / (r_ij^2 + eps^2) (M_i + xi I)^-1 grad_i W_ij
with V_j = m / rho_j, eps^2 = numerics.eps_h_squared (= 0.01 h^2, h = support radius), xi = numerics.regularization_xi
and M_i = sum_{j != i} V_j (r_j - r_i) (x) grad_i W_ij (correction.comp). The loader calibrates the particle volume
as V = 1 / sum_{j != i} W on the lattice (1.1293 dx^2 at h = 5 dx), so in a full support M ~ 1.129 I and the
regularised inverse leaves the factor F = M (M + xi I)^-1 ~ 0.919 on every corrected operator.

For each run this script takes the last particle snapshot and
  * evaluates M on a random subset of fluid particles (interior / within one support radius of a wall row) and F;
  * applies the viscous operator, with the snapshot's positions and volumes, to the divergence-free fields
    v = (y^2, 0) and v = (0, x^2), for which nu lap v = (2 nu, 0) and (0, 2 nu): the ratio of the discrete
    result to the exact one is nu_eff / nu. The operator is evaluated as released (xi, eps) and with xi = 0 and / or
    eps = 0 to separate the two factors.
Outputs: docs/validation/data/kcg_matrix.csv and docs/validation/data/effective_viscosity.csv.

    .venv/Scripts/python.exe -m experiment.validation.effective_viscosity [--runs n250_k2_float32,...]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v6.utils.case_loader_v6 import load_case_v6  # noqa: E402
from experiment.validation.cavity_analysis import DATA, LOGS, write_csv  # noqa: E402

MATRIX_SUBSET = 60000
OPERATOR_SUBSET = 20000
FIELDS = {"(y^2, 0)": (lambda position: np.stack([position[:, 1] ** 2, np.zeros(len(position))], axis=1), 0),
          "(0, x^2)": (lambda position: np.stack([np.zeros(len(position)), position[:, 0] ** 2], axis=1), 1)}


def kernel_value(normalized_distance: np.ndarray, coefficient: float) -> np.ndarray:
    return (coefficient * (1.0 - normalized_distance) ** 6
            * ((35.0 / 3.0) * normalized_distance ** 2 + 6.0 * normalized_distance + 1.0))


def kernel_gradient(relative_position: np.ndarray, distance: np.ndarray, coefficient: float, support: float) -> np.ndarray:
    """grad_i W(r_i - r_j) of the 2-D Wendland C4 kernel with support radius ``support`` (helpers.glsl)."""
    normalized_distance = distance / support
    scalar = (coefficient / support * (1.0 - normalized_distance) ** 5 * normalized_distance
              * ((-280.0 / 3.0) * normalized_distance - 56.0 / 3.0))
    return scalar[:, None] * relative_position / distance[:, None]


def neighbourhood(positions: np.ndarray, particle: int, neighbours: list[int], support: float):
    neighbours = np.asarray([neighbour for neighbour in neighbours if neighbour != particle])
    relative_position = positions[particle] - positions[neighbours]            # x_ij = r_i - r_j
    distance = np.linalg.norm(relative_position, axis=1)
    keep = (distance < support) & (distance > 1e-12)
    return neighbours[keep], relative_position[keep], distance[keep]


def evaluate(run_id: str) -> tuple[list[list], list[list]]:
    run_dir = LOGS / run_id
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    case = load_case_v6(_REPO_ROOT / meta["case"])
    regularization, epsilon_squared = float(case.numerics.regularization_xi), float(case.numerics.eps_h_squared)
    spacing, support = float(meta["spacing"]), float(meta["support_radius"])
    snapshot = sorted((run_dir / "snapshots").glob("t*.npz"))[-1]
    with np.load(snapshot) as archive:
        positions = archive["positions"][:, :2].astype(np.float64)
        volumes = archive["volumes"].astype(np.float64)
        material = archive["material"].astype(np.int64)
    fluid = np.flatnonzero(material == np.bincount(material).argmax())
    generator = np.random.default_rng(0)
    coefficient = 9.0 / (np.pi * support ** 2)
    tree = cKDTree(positions)
    wall_distance = (0.5 + spacing) - np.abs(positions).max(axis=1)              # to the nearest wall / lid row

    # ---- correction matrix and the factor F ------------------------------------------------------------------
    chosen = fluid if fluid.size <= MATRIX_SUBSET else generator.choice(fluid, MATRIX_SUBSET, replace=False)
    matrices, kernel_sums = np.zeros((chosen.size, 2, 2)), np.zeros(chosen.size)
    for row_index, (particle, found) in enumerate(zip(chosen, tree.query_ball_point(positions[chosen], support))):
        neighbours, relative_position, distance = neighbourhood(positions, particle, found, support)
        gradient = kernel_gradient(relative_position, distance, coefficient, support)
        matrices[row_index] = np.einsum("n,ni,nj->ij", volumes[neighbours], -relative_position, gradient)
        kernel_sums[row_index] = np.sum(volumes[neighbours] * kernel_value(distance / support, coefficient))
    factors = np.einsum("nij,njk->nik", matrices, np.linalg.inv(matrices + regularization * np.eye(2)))
    matrix_rows = []
    for region, mask in (("interior", wall_distance[chosen] >= support), ("near_wall", wall_distance[chosen] < support)):
        diagonal = matrices[mask][:, [0, 1], [0, 1]].mean(axis=1)
        factor = factors[mask][:, [0, 1], [0, 1]].mean(axis=1)
        matrix_rows.append([run_id, int(round(1.0 / spacing)), snapshot.name, region, int(mask.sum()), f"{regularization:g}",
                            f"{np.median(volumes[fluid]) / spacing ** 2:.5f}", f"{diagonal.mean():.5f}",
                            f"{np.percentile(diagonal, 1):.5f}", f"{np.percentile(diagonal, 99):.5f}",
                            f"{np.abs(matrices[mask][:, 0, 1]).mean():.2e}", f"{kernel_sums[mask].mean():.5f}",
                            f"{factor.mean():.5f}", f"{np.percentile(factor, 1):.5f}", f"{np.percentile(factor, 99):.5f}"])
        print(f"{run_id} {region}: n = {mask.sum()}, M diagonal {diagonal.mean():.4f}, kernel sum "
              f"{kernel_sums[mask].mean():.4f}, F {factor.mean():.4f}", flush=True)

    # ---- the viscous operator on quadratic divergence-free fields ----------------------------------------------
    interior = fluid[wall_distance[fluid] >= 1.5 * support]
    chosen = interior if interior.size <= OPERATOR_SUBSET else generator.choice(interior, OPERATOR_SUBSET, replace=False)
    # released; smaller xi; eps^2 = 0.01 (h / 2)^2 (the 0.01 factor applied to half the support radius); both off
    quarter = 0.25 * epsilon_squared
    variants = [("released", regularization, epsilon_squared), ("xi = 0.01", 0.01, epsilon_squared),
                ("xi = 0.001", 0.001, epsilon_squared), ("xi = 0", 0.0, epsilon_squared),
                ("eps^2 / 4", regularization, quarter), ("xi = 0.01, eps^2 / 4", 0.01, quarter),
                ("xi = 0.001, eps^2 / 4", 0.001, quarter), ("eps = 0", regularization, 0.0),
                ("xi = 0, eps = 0", 0.0, 0.0)]
    ratios = {(variant, field): np.zeros(chosen.size) for variant, _, _ in variants for field in FIELDS}
    for row_index, (particle, found) in enumerate(zip(chosen, tree.query_ball_point(positions[chosen], support))):
        neighbours, relative_position, distance = neighbourhood(positions, particle, found, support)
        gradient = kernel_gradient(relative_position, distance, coefficient, support)
        matrix = np.einsum("n,ni,nj->ij", volumes[neighbours], -relative_position, gradient)
        for variant, xi, epsilon in variants:
            corrected = gradient @ np.linalg.inv(matrix + xi * np.eye(2)).T
            for field, (function, component) in FIELDS.items():
                difference = function(positions[particle][None, :])[0] - function(positions[neighbours])
                projection = np.einsum("ni,ni->n", difference, relative_position)
                acceleration = np.sum((2.0 * (2 + 2) * volumes[neighbours] * projection
                                       / (distance ** 2 + epsilon))[:, None] * corrected, axis=0)
                ratios[(variant, field)][row_index] = acceleration[component] / 2.0
    operator_rows = []
    for variant, xi, epsilon in variants:
        for field in FIELDS:
            values = ratios[(variant, field)]
            operator_rows.append([run_id, int(round(1.0 / spacing)), snapshot.name, variant, f"{xi:g}", f"{epsilon:.3e}",
                                  f"{epsilon / spacing ** 2:.4f}", field, values.size, f"{values.mean():.5f}",
                                  f"{np.median(values):.5f}", f"{np.percentile(values, 5):.5f}",
                                  f"{np.percentile(values, 95):.5f}"])
        both = np.concatenate([ratios[(variant, field)] for field in FIELDS])
        print(f"{run_id} {variant}: nu_eff / nu median {np.median(both):.4f}", flush=True)
    return matrix_rows, operator_rows


SETTING_LABELS = {"released": "ξ = 0.1,ε² = 0.01h²(发布配置)", "xi = 0.01": "ξ = 0.01,ε² = 0.01h²",
                  "xi = 0.001": "ξ = 0.001,ε² = 0.01h²", "xi = 0": "ξ = 0,ε² = 0.01h²",
                  "eps^2 / 4": "ξ = 0.1,ε² = 0.0025h²", "xi = 0.01, eps^2 / 4": "ξ = 0.01,ε² = 0.0025h²",
                  "xi = 0.001, eps^2 / 4": "ξ = 0.001,ε² = 0.0025h²(主系列)", "eps = 0": "ξ = 0.1,ε = 0",
                  "xi = 0, eps = 0": "ξ = 0,ε = 0"}


def read_rows(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def write_markdown() -> None:
    """tables_viscosity.md from the two CSVs: nu_eff / nu per setting and resolution (mean of the two fields'
    medians), and the correction matrix statistics."""
    operator_rows = read_rows(DATA / "effective_viscosity.csv")
    resolutions = sorted({int(row["resolution"]) for row in operator_rows})
    rows = []
    for setting, label in SETTING_LABELS.items():
        cells = [label]
        for resolution in resolutions:
            values = [float(row["nu_eff_over_nu_median"]) for row in operator_rows
                      if row["variant"] == setting and int(row["resolution"]) == resolution]
            cells.append(f"{np.mean(values):.3f}" if values else "–")
        rows.append(cells)
    sections = {"VISCOSITY": md_table(["设置"] + [f"{resolution}²" for resolution in resolutions], rows)}
    matrix_rows = read_rows(DATA / "kcg_matrix.csv")
    sections["KCG_MATRIX"] = md_table(
        ["分辨率", "区域", "粒子数", "M 对角线均值(1 %–99 %)", "Σ_{j≠i} V W", "F = M (M + ξI)⁻¹ 均值(1 %–99 %)"],
        [[f"{row['resolution']}²", "内部" if row["region"] == "interior" else "离壁面一个支撑半径以内", row["particles"],
          f"{float(row['M_diagonal_mean']):.4f}({float(row['M_diagonal_p01']):.4f}–{float(row['M_diagonal_p99']):.4f})",
          f"{float(row['kernel_sum_mean']):.4f}",
          f"{float(row['factor_mean']):.4f}({float(row['factor_p01']):.4f}–{float(row['factor_p99']):.4f})"]
         for row in matrix_rows])
    text = "\n\n".join(f"<!-- {name} -->\n{table}" for name, table in sections.items()) + "\n"
    (DATA / "tables_viscosity.md").write_text(text, encoding="utf-8")
    print("wrote", DATA / "tables_viscosity.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="n250_k2_float32,n500_k2_float32,n1000_k2_float32")
    parser.add_argument("--tables-only", action="store_true", help="only rewrite tables_viscosity.md from the CSVs")
    arguments = parser.parse_args()
    if arguments.tables_only:
        write_markdown()
        return 0
    matrix_rows, operator_rows = [], []
    for run_id in arguments.runs.split(","):
        matrices, operators = evaluate(run_id)
        matrix_rows += matrices
        operator_rows += operators
    write_csv(DATA / "kcg_matrix.csv",
              ["run", "resolution", "snapshot", "region", "particles", "xi", "volume_over_dx2", "M_diagonal_mean",
               "M_diagonal_p01", "M_diagonal_p99", "M_offdiagonal_mean_abs", "kernel_sum_mean", "factor_mean",
               "factor_p01", "factor_p99"], matrix_rows,
              "KCG correction matrix M = sum_{j != i} V_j (r_j - r_i) (x) grad W_ij on the last snapshot of each run (random\n"
              f"subset of up to {MATRIX_SUBSET} fluid particles; interior = at least one support radius from the wall / lid rows)\n"
              "and the factor F = M (M + xi I)^-1 that the regularised correction leaves on every corrected operator (diagonal mean).")
    write_csv(DATA / "effective_viscosity.csv",
              ["run", "resolution", "snapshot", "variant", "xi", "eps_squared", "eps_squared_over_dx2", "field", "particles",
               "nu_eff_over_nu_mean", "nu_eff_over_nu_median", "nu_eff_over_nu_p05", "nu_eff_over_nu_p95"], operator_rows,
              "The solver's viscous operator (force.comp: Monaghan form, KCG-corrected gradient, eps in the denominator) applied\n"
              "with the snapshot's positions and volumes to v = (y^2, 0) and (0, x^2) (exact nu lap v = 2 nu), at up to\n"
              f"{OPERATOR_SUBSET} fluid particles at least 1.5 support radii from the wall rows: discrete / exact = nu_eff / nu.\n"
              "'released' uses the case's xi and eps^2 = 0.01 h^2 (h = support radius); the other rows switch them off.")
    print("wrote", DATA / "kcg_matrix.csv", "and", DATA / "effective_viscosity.csv")
    write_markdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
