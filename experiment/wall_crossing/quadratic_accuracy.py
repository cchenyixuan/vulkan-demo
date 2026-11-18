"""
quadratic_accuracy.py - numpy (float64) check of the second-order continuation of the velocity on a single flat
wall: is it exact for a quadratic velocity field on a lattice, and what is left of that on a jittered lattice?

    .venv/Scripts/python.exe -m experiment.wall_crossing.quadratic_accuracy

Setting (2-D): the wall is the plane y = 0; fluid particles at y = (k + 1/2) dx, k = 0, 1, ... and wall particles
at y = -(k + 1/2) dx on the same square lattice, V_j = dx^2 for every particle, Wendland C4 with support radius
h = 5 dx (the cavity cases' ratio), no regularisation (xi = 0), the viscous denominators' eps^2 = 0.0025 h^2
(the cases' value). The wall is at rest (v_w = 0). phi_i = y_i, d_j = -y_j, a_j = d_j / phi_i.

Fields (gamma, kappa, a: constants of order one, in units of dx and one velocity unit):
    (a) v = (gamma y, 0)                        linear
    (b) v = (gamma y + kappa y^2 / 2, 0)        quadratic, no slip
    (c) v = (2 a x y, -a y^2)                   quadratic, no slip, divergence free

Rules for a wall neighbour's velocity difference (shaders/wall_boundary.glsl):
    first order    (v_w - v_i) (1 + a_j)
    second order   (v_w - v_i) (1 + a_j)^2 - a_j grad v_i (r_j - r_i),  grad v_i = M_0 L (I - K_raw L)^-1
    truth          v(r_j) - v_i with the polynomial evaluated at the wall particle (what the rule should give)

Per fluid particle of the first three layers:
    gradient       grad v_i against the analytic gradient (relative, Frobenius norms); the first-order row is the
                   corrected gradient with the first-order wall values, the truth row the same with the truth
    wall velocity  the largest |rule's v_j - v(r_j)| over the wall neighbours, over the largest |v(r_j) - v_i|
    viscous        the Morris viscous acceleration against the same operator with the truth in the wall
                   particles: |difference| / (sum of the absolute pair terms), and the ratio of the two
                   accelerations' components along the truth's direction where the truth is not zero
    divergence     field (c): the velocity divergence (zero in truth) over |d v_y / d y| = 2 a phi_i, for
                   first order normal only / first order whole vector / second order whole vector /
                   second order normal only (normal only: the wall neighbour's velocity difference projected
                   on the wall normal before it enters the sum; the second-order one uses the same solved
                   grad v_i) and for the truth in the wall particles

The strip is centred on x = 0 (240 columns, x from -120 dx to 120 dx); field (c) is not invariant along x: its
tangential velocity at a particle is 2 a x_i phi_i, i.e. x_i / dx times the scale 2 a phi_i dx of the divergence.

Lattice: one particle per layer, at x = 20.5 dx (the operator is exact there, the numbers are rounding). Jitter:
every FLUID particle displaced by a uniform random vector in [-0.05, 0.05]^2 dx (the wall stays on the lattice),
10 seeds; statistics over the particles of each layer with |x| <= 100 dx, and for the divergence of field (c)
also apart for |x| <= 10 dx and for 50 dx <= |x| <= 100 dx (the errors of the normal-only variants grow with
the tangential velocity, the others do not). Output: claude/wall_crossing/tables/
quadratic_accuracy_*.md and claude/wall_crossing/quadratic_extension/accuracy.json; one runs.jsonl record.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LoggedRun     # noqa: E402

SPACING = 1.0
SUPPORT = 5.0 * SPACING
EPSILON_SQUARED = 0.0025 * SUPPORT * SUPPORT
COLUMNS = 240
FLUID_LAYERS = 16
WALL_LAYERS = 7
EDGE_COLUMNS = 20                   # columns left out at either lateral end (support = 5 dx)
JITTER_FRACTION = 0.05
SEED = 20261118
SEEDS = 10                          # jittered clouds (SEED, SEED + 1, ...)
LATTICE_COLUMN_OFFSET = 20          # the lattice particle: this many columns right of the centre
DIVERGENCE_SUBSETS = (("全部（|x| ≤ 100 dx）", 0.0, 100.0), ("|x| ≤ 10 dx", 0.0, 10.0),
                      ("50 dx ≤ |x| ≤ 100 dx", 50.0, 100.0))
GAMMA, KAPPA, STRAIN = 1.3, 0.7, 0.9
VISCOSITY = 1.0
LAYERS = (1, 2, 3)
FIELDS = ("a", "b", "c")
FIELD_TEXT = {"a": "(a) v = (γy, 0)", "b": "(b) v = (γy + κy²/2, 0)", "c": "(c) v = (2a·x·y, −a·y²)"}
WALL_NORMAL = np.array([0.0, 1.0])


def field_velocity(name: str, positions: np.ndarray) -> np.ndarray:
    x, y = positions[:, 0], positions[:, 1]
    if name == "a":
        return np.stack([GAMMA * y, np.zeros_like(y)], axis=1)
    if name == "b":
        return np.stack([GAMMA * y + 0.5 * KAPPA * y * y, np.zeros_like(y)], axis=1)
    return np.stack([2.0 * STRAIN * x * y, -STRAIN * y * y], axis=1)


def field_gradient(name: str, position: np.ndarray) -> np.ndarray:
    """[a, b] = d v_a / d x_b at one position."""
    x, y = position
    if name == "a":
        return np.array([[0.0, GAMMA], [0.0, 0.0]])
    if name == "b":
        return np.array([[0.0, GAMMA + KAPPA * y], [0.0, 0.0]])
    return np.array([[2.0 * STRAIN * y, 2.0 * STRAIN * x], [0.0, -2.0 * STRAIN * y]])


def kernel_gradient(relative_position: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """grad_i W_ij of the Wendland C4 kernel (support radius SUPPORT, any normalisation) for r_i - r_j."""
    normalized = distance / SUPPORT
    magnitude = (1.0 - normalized) ** 5 * normalized * ((-280.0 / 3.0) * normalized - 56.0 / 3.0) / SUPPORT
    return (magnitude / distance)[:, None] * relative_position


def particles(jitter: bool, seed: int = SEED) -> dict:
    columns = (np.arange(COLUMNS) + 0.5 - COLUMNS // 2) * SPACING
    fluid_rows = (np.arange(FLUID_LAYERS) + 0.5) * SPACING
    wall_rows = -(np.arange(WALL_LAYERS) + 0.5) * SPACING
    fluid = np.stack(np.meshgrid(columns, fluid_rows, indexing="ij"), axis=-1).reshape(-1, 2)
    fluid_layer = np.tile(np.arange(FLUID_LAYERS) + 1, COLUMNS)
    fluid_column = np.repeat(np.arange(COLUMNS), FLUID_LAYERS)
    wall = np.stack(np.meshgrid(columns, wall_rows, indexing="ij"), axis=-1).reshape(-1, 2)
    if jitter:
        generator = np.random.default_rng(seed)
        fluid = fluid + generator.uniform(-JITTER_FRACTION, JITTER_FRACTION, size=fluid.shape) * SPACING
    return {"fluid": fluid, "wall": wall, "layer": fluid_layer, "column": fluid_column}


def evaluate_particle(index: int, fluid: np.ndarray, wall: np.ndarray, name: str) -> dict:
    """Everything of one fluid particle and one field."""
    position = fluid[index]
    volume = SPACING * SPACING

    def neighbours(points: np.ndarray, skip_self: bool) -> tuple:
        displacement = points - position                      # r_j - r_i
        distance = np.sqrt((displacement ** 2).sum(axis=1))
        keep = distance < SUPPORT
        if skip_self:
            keep &= distance > 1e-12
        return displacement[keep], distance[keep], points[keep]

    fluid_displacement, fluid_distance, fluid_points = neighbours(fluid, True)
    wall_displacement, wall_distance, wall_points = neighbours(wall, False)
    fluid_gradient = kernel_gradient(-fluid_displacement, fluid_distance)
    wall_gradient = kernel_gradient(-wall_displacement, wall_distance)
    correction_matrix = volume * (np.einsum("pa,pb->ab", fluid_displacement, fluid_gradient)
                                  + np.einsum("pa,pb->ab", wall_displacement, wall_gradient))
    correction = np.linalg.inv(correction_matrix)             # L, xi = 0

    velocity = field_velocity(name, position[None, :])[0]
    fluid_difference = field_velocity(name, fluid_points) - velocity
    truth_difference = field_velocity(name, wall_points) - velocity      # the polynomial in the wall particles
    wall_distance_to_plane = position[1]                                  # phi_i
    depth_ratio = (-wall_points[:, 1]) / wall_distance_to_plane          # a_j
    wall_value_difference = -velocity                                     # v_w - v_i, v_w = 0

    # grad v_i of the second-order rule: M_0 L (I - K_raw L)^-1
    fluid_matrix = volume * np.einsum("pa,pb->ab", fluid_difference, fluid_gradient)
    wall_matrix = volume * np.einsum("p,a,pb->ab", (1.0 + depth_ratio) ** 2, wall_value_difference, wall_gradient)
    displacement_matrix = volume * np.einsum("p,pa,pb->ab", -depth_ratio, wall_displacement, wall_gradient)
    system = np.eye(2) - displacement_matrix @ correction
    gradient_second = (fluid_matrix + wall_matrix) @ correction @ np.linalg.inv(system)

    differences = {
        "first": wall_value_difference[None, :] * (1.0 + depth_ratio)[:, None],
        "second": wall_value_difference[None, :] * ((1.0 + depth_ratio) ** 2)[:, None]
        - depth_ratio[:, None] * (wall_displacement @ gradient_second.T),
        "truth": truth_difference,
    }
    analytic = field_gradient(name, position)
    result = {"layer_distance": float(wall_distance_to_plane), "system_determinant": float(np.linalg.det(system)),
              "x": float(position[0])}

    corrected_fluid = fluid_gradient @ correction.T            # L grad W of every fluid pair (L symmetric)
    corrected_wall = wall_gradient @ correction.T
    softened_fluid = fluid_distance ** 2 + EPSILON_SQUARED
    softened_wall = wall_distance ** 2 + EPSILON_SQUARED
    morris = 2.0 * (2 + 2.0) * VISCOSITY * volume

    def viscous(wall_difference: np.ndarray) -> tuple:
        # a = sum_j 2 (d + 2) nu V_j (v_i - v_j) . (r_i - r_j) / (r^2 + eps^2) L grad W_ij
        fluid_pairs = (morris * (fluid_difference * fluid_displacement).sum(axis=1) / softened_fluid)[:, None] \
            * corrected_fluid
        wall_pairs = (morris * (wall_difference * wall_displacement).sum(axis=1) / softened_wall)[:, None] \
            * corrected_wall
        scale = np.sqrt((fluid_pairs ** 2).sum(axis=1)).sum() + np.sqrt((wall_pairs ** 2).sum(axis=1)).sum()
        return fluid_pairs.sum(axis=0) + wall_pairs.sum(axis=0), scale

    def divergence(wall_difference: np.ndarray, normal_only: bool) -> float:
        if normal_only:
            wall_difference = (wall_difference @ WALL_NORMAL)[:, None] * WALL_NORMAL[None, :]
        return float(volume * ((fluid_difference * corrected_fluid).sum() + (wall_difference * corrected_wall).sum()))

    def corrected_gradient(wall_difference: np.ndarray) -> np.ndarray:
        return (fluid_matrix + volume * np.einsum("pa,pb->ab", wall_difference, wall_gradient)) @ correction

    viscous_truth, viscous_scale = viscous(differences["truth"])
    truth_norm = float(np.sqrt((viscous_truth ** 2).sum()))
    difference_scale = float(np.sqrt((truth_difference ** 2).sum(axis=1)).max())
    gradient_norm = float(np.sqrt((analytic ** 2).sum()))
    for rule in ("first", "second", "truth"):
        gradient_of_rule = gradient_second if rule == "second" else corrected_gradient(differences[rule])
        acceleration, _ = viscous(differences[rule])
        result[rule] = {
            "gradient_error": float(np.sqrt(((gradient_of_rule - analytic) ** 2).sum()) / gradient_norm),
            "wall_velocity_error": float(np.sqrt(((differences[rule] - truth_difference) ** 2).sum(axis=1)).max()
                                         / difference_scale),
            "viscous_error_over_pair_terms": float(np.sqrt(((acceleration - viscous_truth) ** 2).sum())
                                                   / viscous_scale),
            # the component along the truth's acceleration over the truth's magnitude (1 = exact)
            "viscous_ratio": (float((acceleration @ viscous_truth) / truth_norm ** 2)
                              if truth_norm > 1e-9 * viscous_scale else None),
            "divergence_whole": divergence(differences[rule], False),
            "divergence_normal": divergence(differences[rule], True),
        }
    result["divergence_scale"] = float(2.0 * STRAIN * wall_distance_to_plane)
    return result


def collect(jitter: bool) -> dict:
    """field -> layer -> list of per-particle results."""
    results = {name: {layer: [] for layer in LAYERS} for name in FIELDS}
    for seed in range(SEED, SEED + (SEEDS if jitter else 1)):
        cloud = particles(jitter, seed)
        interior = (cloud["column"] >= EDGE_COLUMNS) & (cloud["column"] < COLUMNS - EDGE_COLUMNS)
        for name in FIELDS:
            for layer in LAYERS:
                rows = np.flatnonzero(interior & (cloud["layer"] == layer))
                if not jitter:
                    # one particle of the lattice layer (x = 20.5 dx: field (c) depends on x)
                    rows = rows[[rows.size // 2 + LATTICE_COLUMN_OFFSET]]
                results[name][layer] += [evaluate_particle(int(row), cloud["fluid"], cloud["wall"], name)
                                         for row in rows]
    return results


def scientific(value: float) -> str:
    return f"{value:.1e}"


def lattice_tables(results: dict) -> tuple:
    """(the exactness table, the divergence table of field (c), the summary numbers)."""
    lines = ["| 场 | 层 | φ_i / dx | ∇v_i 对解析梯度 | 壁邻居速度对多项式真值 | 粘性加速度对“填真值”（÷ 逐对项绝对值之和） | "
             "一阶规则：粘性加速度 ÷ 填真值的 | det(I − K L) |", "|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    summary = {}
    for name in FIELDS:
        for layer in LAYERS:
            entry = results[name][layer][0]
            second, first = entry["second"], entry["first"]
            worst = max(worst, second["gradient_error"], second["wall_velocity_error"],
                        second["viscous_error_over_pair_terms"])
            ratio = "（填真值的加速度为零）" if first["viscous_ratio"] is None else f"{first['viscous_ratio']:.3f}"
            lines.append(f"| {FIELD_TEXT[name]} | {layer} | {entry['layer_distance'] / SPACING:.1f} | "
                         f"{scientific(second['gradient_error'])} | {scientific(second['wall_velocity_error'])} | "
                         f"{scientific(second['viscous_error_over_pair_terms'])} | {ratio} | "
                         f"{entry['system_determinant']:.3f} |")
            summary[f"{name}_{layer}"] = {"second": second, "first": first, "truth": entry["truth"],
                                          "system_determinant": entry["system_determinant"]}
    divergence_lines = ["| 层 | φ_i / dx | 一阶，只取法向 | 一阶，完整向量 | 二阶，完整向量 | 二阶，只取法向 | 壁粒子填真值 |",
                        "|---|---|---|---|---|---|---|"]
    for layer in LAYERS:
        entry = results["c"][layer][0]
        scale = entry["divergence_scale"]
        divergence_lines.append(
            f"| {layer} | {entry['layer_distance'] / SPACING:.1f} | "
            f"{100 * entry['first']['divergence_normal'] / scale:+.2f} % | "
            f"{100 * entry['first']['divergence_whole'] / scale:+.2f} % | "
            f"{entry['second']['divergence_whole'] / scale:+.1e} | "
            f"{100 * entry['second']['divergence_normal'] / scale:+.2f} % | "
            f"{entry['truth']['divergence_whole'] / scale:+.1e} |")
    return "\n".join(lines) + "\n", "\n".join(divergence_lines) + "\n", {"worst_second_order_error": worst,
                                                                         "particles": summary}


def jitter_tables(results: dict) -> tuple:
    def statistics(values: list) -> tuple:
        values = np.abs(np.asarray(values, dtype=np.float64))
        return float(np.median(values)), float(values.max())

    def cell(values: list) -> str:
        median, largest = statistics(values)
        return f"{100 * median:.2f} % / {100 * largest:.2f} %"

    lines = ["| 场 | 层 | 量 | 一阶规则 | 二阶规则 | 壁粒子填真值 |", "|---|---|---|---|---|---|"]
    summary = {}
    for name in FIELDS:
        for layer in LAYERS:
            entries = results[name][layer]
            summary[f"{name}_{layer}"] = {}
            quantities = [("∇v_i 对解析梯度", "gradient_error"), ("壁邻居速度对真值", "wall_velocity_error"),
                          ("粘性加速度对“填真值”（÷ 逐对项绝对值之和）", "viscous_error_over_pair_terms")]
            for text, key in quantities:
                cells = []
                for rule in ("first", "second", "truth"):
                    if rule == "truth" and key != "gradient_error":
                        cells.append("—")
                        continue
                    values = [entry[rule][key] for entry in entries]
                    cells.append(cell(values))
                    summary[f"{name}_{layer}"][f"{key}_{rule}"] = statistics(values)
                lines.append(f"| {FIELD_TEXT[name]} | {layer} | {text} | " + " | ".join(cells) + " |")
            if entries[0]["first"]["viscous_ratio"] is not None:
                cells = []
                for rule in ("first", "second"):
                    ratios = np.asarray([entry[rule]["viscous_ratio"] for entry in entries])
                    cells.append(f"{np.median(ratios):.3f}（{ratios.min():.3f} 到 {ratios.max():.3f}）")
                    summary[f"{name}_{layer}"][f"viscous_ratio_{rule}"] = [float(np.median(ratios)),
                                                                           float(ratios.min()), float(ratios.max())]
                lines.append(f"| {FIELD_TEXT[name]} | {layer} | 粘性加速度 ÷ 填真值的（沿真值方向；中位数和范围） | "
                             + " | ".join(cells) + " | 1 |")
    divergence_lines = ["| 粒子 | 层 | 粒子数 | 一阶，只取法向 | 一阶，完整向量 | 二阶，完整向量 | 二阶，只取法向 | 壁粒子填真值 |",
                        "|---|---|---|---|---|---|---|---|"]
    for subset_text, lower, upper in DIVERGENCE_SUBSETS:
        for layer in LAYERS:
            entries = [entry for entry in results["c"][layer] if lower <= abs(entry["x"]) / SPACING <= upper]
            cells = []
            for rule, key in (("first", "divergence_normal"), ("first", "divergence_whole"),
                              ("second", "divergence_whole"), ("second", "divergence_normal"),
                              ("truth", "divergence_whole")):
                values = [entry[rule][key] / entry["divergence_scale"] for entry in entries]
                cells.append(cell(values))
                summary[f"c_{layer}"][f"{key}_{rule}_{lower:g}_{upper:g}"] = statistics(values)
            divergence_lines.append(f"| {subset_text} | {layer} | {len(entries)} | " + " | ".join(cells) + " |")
    count = len(results["a"][1])
    return "\n".join(lines) + "\n", "\n".join(divergence_lines) + "\n", {"particles_per_layer": count,
                                                                         "statistics": summary}


def main() -> int:
    tables = REPOSITORY_ROOT / "claude" / "wall_crossing" / "tables"
    out = REPOSITORY_ROOT / "claude" / "wall_crossing" / "quadratic_extension"
    tables.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with LoggedRun(kind="quadratic_accuracy", case="none") as run:
        lattice = collect(False)
        jittered = collect(True)
        exact_text, divergence_text, lattice_summary = lattice_tables(lattice)
        jitter_text, jitter_divergence_text, jitter_summary = jitter_tables(jittered)
        note = f"记录 {run.run_id}。"
        (tables / "quadratic_accuracy_lattice.md").write_text(exact_text + "\n" + note + "\n", encoding="utf-8",
                                                              newline="\n")
        (tables / "quadratic_accuracy_divergence.md").write_text(divergence_text + "\n" + note + "\n",
                                                                 encoding="utf-8", newline="\n")
        (tables / "quadratic_accuracy_jitter.md").write_text(jitter_text + "\n" + note + "\n", encoding="utf-8",
                                                             newline="\n")
        (tables / "quadratic_accuracy_jitter_divergence.md").write_text(jitter_divergence_text + "\n" + note + "\n",
                                                                        encoding="utf-8", newline="\n")
        summary = {"run_id": run.run_id, "spacing": SPACING, "support": SUPPORT, "jitter_fraction": JITTER_FRACTION,
                   "seed": SEED, "gamma": GAMMA, "kappa": KAPPA, "strain": STRAIN,
                   "lattice": lattice_summary, "jitter": jitter_summary}
        (out / "accuracy.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
        passed = lattice_summary["worst_second_order_error"] < 1.0e-10
        run.extra["label"] = "numpy check of the second-order continuation on a flat wall (lattice and 5 % jitter)"
        run.extra["worst_second_order_error_on_the_lattice"] = lattice_summary["worst_second_order_error"]
        run.extra["status"] = "ok" if passed else "failed: a lattice error above 1e-10"
        print(exact_text)
        print(divergence_text)
        print(jitter_text)
        print(jitter_divergence_text)
        print(f"[quadratic_accuracy] worst second-order error on the lattice "
              f"{lattice_summary['worst_second_order_error']:.2e} ({'PASS' if passed else 'FAIL'}: 1e-10); "
              f"{jitter_summary['particles_per_layer']} particles per layer with jitter; record {run.run_id}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
