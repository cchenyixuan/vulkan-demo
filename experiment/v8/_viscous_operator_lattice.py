"""
_viscous_operator_lattice.py — the solver's viscous operator on the exact square lattice, by hand (CPU, float64):
how much of the Laplacian does it return for a sinusoidal velocity field, and what the eps^2 regularisation costs.

force.comp's term (Cleary–Monaghan form with the KCG-corrected gradient):

    a_i = 2 (d + 2) nu sum_j V_j (v_i - v_j) . x_ij / (r_ij^2 + eps^2) L_i grad W_ij,   x_ij = r_i - r_j

evaluated for v = (sin(k y), 0) on the periodic lattice at ((i + 1/2) dx, (j + 1/2) dx), V_j = the loader's calibrated
volume (1 / sum_j W_ij), L_i = (M_i + xi I)^-1 with M_i = sum_j V_j (r_j - r_i) (x) grad W_ij. The exact value is
nu lap v = -nu k^2 sin(k y). The ratio a_x / (-nu k^2 sin(k y)) is the operator's effective viscosity on the lattice
(uniform over particles up to the kernel's truncation), at eps^2 = 0 and at the case values 0.0025 h^2 / 0.01 h^2.

    .venv/Scripts/python.exe experiment/v8/_viscous_operator_lattice.py [--hdx 5 --side 200] [--out docs/tgv/data/viscous_operator_lattice.md]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def wendland_c4(q: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """W and dW/dr of the 2-D Wendland C4 with support h (helpers.glsl's profile and coefficients)."""
    inside = q < 1.0
    one_minus = np.where(inside, 1.0 - q, 0.0)
    coefficient = 9.0 / (math.pi * h * h)
    value = coefficient * one_minus ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0) * inside
    derivative = (coefficient / h) * one_minus ** 5 * q * ((-280.0 / 3.0) * q - 56.0 / 3.0) * inside
    return value, derivative


def lattice_operator(hdx: float, side: int, epsilon_factors: list[float], xi: float, vortex_pairs: int = 1) -> dict:
    from scipy.spatial import cKDTree
    length = 1.0
    dx = length / side
    h = hdx * dx
    k = 2.0 * math.pi * vortex_pairs / length
    index = np.arange(side) + 0.5
    x, y = np.meshgrid(index * dx, index * dx, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel()])
    tree = cKDTree(points, boxsize=length)
    pairs = tree.query_pairs(h * (1 + 1e-9), output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]])                 # both directions: i -> j and j -> i
    i, j = pairs[:, 0], pairs[:, 1]
    difference = points[i] - points[j]                             # x_ij = r_i - r_j
    difference -= length * np.round(difference / length)
    distance = np.sqrt((difference ** 2).sum(axis=1))
    keep = distance > 1e-12
    i, j, difference, distance = i[keep], j[keep], difference[keep], distance[keep]
    value, derivative = wendland_c4(distance / h, h)
    gradient = (derivative / distance)[:, None] * difference        # grad_i W_ij
    # calibrated volume: 1 / sum_j W_ij at an interior particle (all are interior on the periodic lattice)
    kernel_sum = np.zeros(points.shape[0])
    np.add.at(kernel_sum, i, value)
    volume = 1.0 / kernel_sum[0]
    # KCG matrix M_i = sum_j V (r_j - r_i) (x) grad W_ij, per particle
    outer = (-difference)[:, :, None] * gradient[:, None, :] * volume
    matrix = np.zeros((points.shape[0], 2, 2))
    np.add.at(matrix, i, outer)
    matrix += xi * np.eye(2)[None]
    inverse = np.linalg.inv(matrix)
    corrected = np.einsum("nab,nb->na", inverse[i], gradient)
    velocity = np.column_stack([np.sin(k * points[:, 1]), np.zeros(points.shape[0])])
    relative_velocity = velocity[i] - velocity[j]
    exact = -k * k * np.sin(k * points[:, 1])
    results = {"hdx": hdx, "side": side, "h": h, "dx": dx, "volume_over_dx2": volume / (dx * dx),
               "correction_diagonal": float(inverse[:, 0, 0].mean()), "pairs_per_particle": float(keep.sum() / points.shape[0]),
               "epsilon": {}}
    for factor in epsilon_factors:
        epsilon_squared = factor * h * h
        weight = volume * (relative_velocity * difference).sum(axis=1) / (distance ** 2 + epsilon_squared)
        acceleration = np.zeros(points.shape[0])
        np.add.at(acceleration, i, 2.0 * (2 + 2) * weight * corrected[:, 0])
        mask = np.abs(exact) > 0.3 * k * k
        ratio = acceleration[mask] / exact[mask]
        results["epsilon"][f"{factor:g}"] = {"ratio_mean": float(ratio.mean()), "ratio_min": float(ratio.min()),
                                             "ratio_max": float(ratio.max())}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=int, default=200)
    parser.add_argument("--hdx", type=float, nargs="+", default=[5.0, 4.0, 3.5])
    parser.add_argument("--xi", type=float, default=0.001)
    parser.add_argument("--epsilon-factors", type=float, nargs="+", default=[0.0, 0.0025, 0.01])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    lines = ["# The viscous operator on the exact lattice (experiment/v8/_viscous_operator_lattice.py)", "",
             f"v = (sin(2 pi y), 0) on the periodic {arguments.side}² lattice, calibrated volume, KCG with xi = {arguments.xi:g}; "
             "the table gives a_x / (nu lap v) = the operator's nu_eff / nu (mean over the particles where |lap v| > 0.3 k²; "
             "min–max in brackets).", "",
             "| h/dx | V / dx² | L diagonal | pairs / particle | " + " | ".join(f"ε² = {f:g} h²" for f in arguments.epsilon_factors) + " |",
             "|---|---|---|---|" + "---|" * len(arguments.epsilon_factors)]
    for hdx in arguments.hdx:
        side = arguments.side
        if abs(side / hdx - round(side / hdx)) > 1e-9:
            side = int(round(side / hdx) * hdx) if (round(side / hdx) * hdx) % 1 == 0 else int(math.ceil(side / hdx) * hdx)
        result = lattice_operator(hdx, side, arguments.epsilon_factors, arguments.xi)
        cells = " | ".join(f"{e['ratio_mean']:.4f} [{e['ratio_min']:.4f}, {e['ratio_max']:.4f}]" for e in result["epsilon"].values())
        lines.append(f"| {hdx:g} ({side}²) | {result['volume_over_dx2']:.4f} | {result['correction_diagonal']:.4f} | "
                     f"{result['pairs_per_particle']:.1f} | {cells} |")
        print(f"h/dx {hdx:g}: " + ", ".join(f"eps2 {f} h^2 -> {e['ratio_mean']:.4f}" for f, e in result["epsilon"].items()))
    text = "\n".join(lines) + "\n"
    if arguments.out:
        out = _REPOSITORY_ROOT / arguments.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[lattice] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
