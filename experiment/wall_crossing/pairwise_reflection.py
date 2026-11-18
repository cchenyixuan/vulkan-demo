"""
pairwise_reflection.py - wall_boundary crossing with the pair-by-pair geometry and the reflection of pressure and
density at the wall (numerics.wall_crossing.geometry pairwise, pressure_wall reflection, delta_wall reflection;
shaders/wall_boundary.glsl): its cases, the numpy check and the tables of section 12 of
claude/wall_crossing_report.md.

    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection make-cases [--all]
    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection numpy-check
    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection tables

Configurations (the names used in case directories, run names and tables):
    reflection                    geometry pairwise, pressure_wall reflection, delta_wall reflection ("反射版")
    pairwise_delta_reflection     geometry pairwise, delta_wall reflection only        (--all: the three variants
    pairwise_pressure_reflection  geometry pairwise, pressure_wall reflection only      that locate a failure of
    pairwise                      geometry pairwise only                                the screening)
make-cases derives, from an existing crossing case, the same case with the keys added: the cavity at 250^2 and the
two 2-D hydrostatic boxes. Existing case files are left alone.

numpy-check (float64, xi = 0, particles on a lattice, a box with its four corners; no GPU):
  1. the worked example: a fluid particle half a spacing below the lid and three spacings from a side wall, a wall
     particle half a spacing deep in that side wall at the same height: 1 / t_ij = 3.5 / 3 with the side wall's
     normal (the nearest_wall geometry gives 2);
  2. on a box with non-trivial smooth fields, the density rate and the acceleration with geometry pairwise are
     those of geometry nearest_wall for every fluid particle farther than the support radius from any second
     wall plane;
  3. a hydrostatic field (uniform rho0, p = p_0 + rho0 g . x, both signs of p, gravity along -y and tilted by
     30 degrees), the reflection configuration: |pressure acceleration + g| / |g| over every fluid particle.
Output: claude/wall_crossing/pairwise_reflection/numpy_check.json, tables/pairwise_numpy_*.md; one runs.jsonl
record.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# configuration -> (geometry, pressure_wall, delta_wall)
CONFIGURATIONS = {
    "reflection": ("pairwise", "reflection", "reflection"),
    "pairwise_delta_reflection": ("pairwise", "extension", "reflection"),
    "pairwise_pressure_reflection": ("pairwise", "reflection", "extension"),
    "pairwise": ("pairwise", "extension", "extension"),
}
CONFIGURATION_TEXT = {
    "reflection": "反射版（逐对几何 + 压力反射 + 密度反射）",
    "pairwise_delta_reflection": "逐对几何 + 只有密度反射",
    "pairwise_pressure_reflection": "逐对几何 + 只有压力反射",
    "pairwise": "只有逐对几何",
}
MAIN_CONFIGURATION = "reflection"
SOURCE_CASES = (
    "lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
    "wall_crossing/hydrostatic_2d_n250_gy_crossing",
    "wall_crossing/hydrostatic_2d_n250_g30_crossing",
)
LOCATING_SOURCES = ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",)
REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"


# ===================================================================================================== cases
def variant_text(source_text: str, source_name: str, configuration: str) -> str:
    """The source case with the configuration's keys in numerics.wall_crossing."""
    geometry, pressure_wall, delta_wall = CONFIGURATIONS[configuration]
    lines = source_text.split("\n")
    mapping_rows = [index for index, line in enumerate(lines) if line.startswith("  wall_crossing:")]
    if len(mapping_rows) != 1:
        raise ValueError(f"{source_name}: expected exactly one numerics.wall_crossing mapping")
    # (the case's own top-level "geometry:" section, the particle files, is not indented)
    if any(line.startswith(("    geometry:", "    pressure_wall:", "    delta_wall:")) for line in lines):
        raise ValueError(f"{source_name}: the source already sets geometry, pressure_wall or delta_wall")
    added = [f"    geometry: {geometry}             # nearest_wall | pairwise"]
    if pressure_wall != "extension":
        added.append(f"    pressure_wall: {pressure_wall}      # extension | reflection")
    if delta_wall != "extension":
        added.append(f"    delta_wall: {delta_wall}         # extension | reflection")
    lines[mapping_rows[0] + 1:mapping_rows[0] + 1] = added
    if lines[0] != "schema_version: 2":
        raise ValueError(f"{source_name}: does not start with the schema line")
    header = [f"# VARIANT of cases/{source_name} (experiment/wall_crossing/pairwise_reflection.py): geometry "
              f"{geometry},",
              f"# pressure_wall {pressure_wall}, delta_wall {delta_wall}; everything else identical."]
    lines[2:2] = header + [""]
    return "\n".join(lines)


def variant_name(source_name: str, configuration: str) -> str:
    return f"{source_name}_{configuration}"


def make_cases(every_configuration: bool) -> list:
    written = []
    for source_name in SOURCE_CASES:
        source_text = (REPOSITORY_ROOT / "cases" / source_name / "case.yaml").read_text(encoding="utf-8")
        configurations = [MAIN_CONFIGURATION]
        if every_configuration and source_name in LOCATING_SOURCES:
            configurations = list(CONFIGURATIONS)
        for configuration in configurations:
            target = REPOSITORY_ROOT / "cases" / variant_name(source_name, configuration) / "case.yaml"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(variant_text(source_text, source_name, configuration), encoding="utf-8",
                              newline="\n")
            written.append(target)
    return written


# =============================================================================================== numpy check
GRAVITY_MAGNITUDE = 9.81
NUMPY_BOXES = {2: dict(dimension=2, fluid_counts=(50, 40), spacing=0.02, support_ratio=5.0),
               3: dict(dimension=3, fluid_counts=(24, 20, 16), spacing=0.025, support_ratio=4.0)}
HYDROSTATIC_LIMIT = 1.0e-10


def write_table(name: str, header: list, rows: list) -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    path = REPORT_ROOT / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def lattice_state(dimension: int) -> dict:
    """A lattice box (fluid, static wall, lid; float64) and the constants of the numpy reference with xi = 0."""
    from experiment.wall_crossing.box_case import BoxCaseSpecification, box_particles
    from experiment.wall_crossing.reference import ReferenceConstants

    specification = BoxCaseSpecification(wall_boundary="crossing", speed_of_sound=10.0, viscosity=1.0e-2,
                                         **NUMPY_BOXES[dimension])
    particles = box_particles(specification)
    positions = np.concatenate([particles["fluid"], particles["wall"], particles["lid"]])
    fluid_count, wall_count, lid_count = (particles[key].shape[0] for key in ("fluid", "wall", "lid"))
    is_wall = np.arange(positions.shape[0]) >= fluid_count
    is_lid = np.arange(positions.shape[0]) >= fluid_count + wall_count
    support = specification.support_radius()
    spacing = specification.spacing
    # Wendland C4 with support radius h: (1 - q)^6 (35/3 q^2 + 6 q + 1) x 9 / (pi h^2) in 2-D, 495 / (32 pi h^3) in 3-D
    normalisation = 9.0 / np.pi if dimension == 2 else 495.0 / (32.0 * np.pi)
    rest_density = specification.rest_density
    constants = ReferenceConstants(
        dimension=dimension, smoothing_length=support,
        kernel_coefficient=normalisation / support ** dimension,
        kernel_gradient_coefficient=normalisation / support ** (dimension + 1),
        speed_of_sound=specification.speed_of_sound, delta_coefficient=specification.delta_coefficient,
        timestep=specification.cfl * support / specification.speed_of_sound, cfl_number=specification.cfl,
        eps_h_squared=specification.epsilon_squared_factor * support * support,
        regularization_xi=0.0, regularization_determinant_threshold=1.0e-4,
        regularization_max_frobenius_norm=10.0, use_density_diffusion=True, use_pst=False,
        pst_main_shift_coefficient=0.0, pst_anti_shift_coefficient=0.0, gravity=(0.0, 0.0, 0.0),
        viscosity=specification.viscosity, particle_radius=0.5 * spacing, particle_volume=spacing ** dimension,
        distance_floor=0.25 * spacing, box_lower=tuple(particles["box_lower"]),
        box_upper=tuple(particles["box_upper"]), continuity="extrapolate", delta_wall="extension",
        power_parameter=specification.power,
        eos_constant=specification.speed_of_sound ** 2 * rest_density / specification.power,
        rest_density=rest_density)
    gap = np.minimum(particles["fluid"] - particles["box_lower"],
                     particles["box_upper"] - particles["fluid"])[:, :dimension]
    return {"specification": specification, "positions": positions, "is_wall": is_wall, "is_lid": is_lid,
            "fluid_count": fluid_count, "constants": constants, "support": support, "spacing": spacing,
            "masses": np.full(positions.shape[0], rest_density * spacing ** dimension),
            "rest_density": rest_density, "gap_sorted": np.sort(gap, axis=1),
            "box_lower": particles["box_lower"], "box_upper": particles["box_upper"]}


def numpy_check() -> int:
    from experiment.v7.utils.wall_crossing_v7 import (box_depth_behind_wall, box_distance_to_wall,
                                                      box_pair_geometry)
    from experiment.wall_crossing.reference import evaluate_reference
    from experiment.wall_crossing.run_log import LoggedRun
    from experiment.wall_crossing.single_step import prescribed_fields

    with LoggedRun(kind="numpy_check", case="none") as run:
        run.extra["label"] = "pairwise geometry and reflection: numpy check (float64, xi = 0, lattice)"
        summary = {"run_id": run.run_id}

        # ---- 1. the worked example (spacing 1; the box [0, 40] x [0, 30], the lid is the face y = 30) --------
        lower, upper = (0.0, 0.0, 0.0), (40.0, 30.0, 0.0)
        floor = 0.25
        example_rows = []
        examples = (
            ("任务书的例子：i 离盖 0.5、离侧壁 3；j 在侧壁里深 0.5，与 i 同高", (3.0, 29.5, 0.0), (-0.5, 29.5, 0.0)),
            ("同一个 i；j 在盖里深 0.5，正上方", (3.0, 29.5, 0.0), (3.0, 30.5, 0.0)),
            ("同一个 i；j 在左上角块里（两个轴各出界 0.5）", (3.0, 29.5, 0.0), (-0.5, 30.5, 0.0)),
            ("同一个 i；j 在左上角块里（x 出界 0.5，y 出界 2.5）", (3.0, 29.5, 0.0), (-0.5, 32.5, 0.0)),
        )
        summary["examples"] = []
        for text, fluid_position, wall_position in examples:
            fluid_array = np.asarray([fluid_position])
            wall_array = np.asarray([wall_position])
            normal, depth_ratio, mirror = box_pair_geometry(fluid_array, wall_array, lower, upper, 2, floor)
            nearest_factor = 1.0 + float(box_depth_behind_wall(wall_array, lower, upper, 2)[0]
                                         / max(box_distance_to_wall(fluid_array, lower, upper, 2)[0], floor))
            mirror_point = wall_array[0] - mirror[0]
            entry = {"text": text, "fluid": fluid_position, "wall": wall_position,
                     "pairwise_factor": 1.0 + float(depth_ratio[0]), "normal": normal[0].tolist(),
                     "mirror_point": mirror_point.tolist(), "nearest_wall_factor": nearest_factor}
            summary["examples"].append(entry)
            example_rows.append([text, f"({fluid_position[0]:g}, {fluid_position[1]:g})",
                                 f"({wall_position[0]:g}, {wall_position[1]:g})",
                                 f"{entry['pairwise_factor']:.6f}",
                                 f"({normal[0, 0]:+.0f}, {normal[0, 1]:+.0f})",
                                 f"({mirror_point[0]:g}, {mirror_point[1]:g})", f"{nearest_factor:.6f}"])
        example_error = abs(summary["examples"][0]["pairwise_factor"] - 3.5 / 3.0)
        summary["example_error"] = example_error
        summary["example_passed"] = bool(example_error < 1e-15
                                         and summary["examples"][0]["normal"] == [1.0, 0.0, 0.0])
        write_table("pairwise_numpy_examples.md",
                    ["情形（间距 = 1，盒 [0, 40] × [0, 30]，盖在 y = 30）", "r_i", "r_j", "逐对 1/t_ij", "逐对法向 n_ij",
                     "镜像点 r_j′", "旧写法 1 + d_j/φ_i"], example_rows)

        equality_rows, hydrostatic_rows = [], []
        summary["equality"], summary["hydrostatic"] = {}, {}
        for dimension in (2, 3):
            state = lattice_state(dimension)
            constants = state["constants"]
            specification = state["specification"]
            fluid_positions = state["positions"][:state["fluid_count"]]
            velocities = np.zeros_like(state["positions"])
            densities = np.full(state["positions"].shape[0], state["rest_density"])
            pressures = np.zeros(state["positions"].shape[0])

            # ---- 2. pairwise against nearest_wall on smooth fields with a moving lid and a gravity ------------
            specification.lid_velocity = (1.0, 0.0, 0.0)
            fields = prescribed_fields("smooth", specification, fluid_positions)
            velocities[:state["fluid_count"]] = fields["velocity"]
            velocities[state["is_lid"]] = np.asarray(specification.lid_velocity)
            densities[:state["fluid_count"]] = fields["density"]
            pressures[:state["fluid_count"]] = fields["pressure"]
            gravity = (3.5, -8.0, 0.0) if dimension == 2 else (3.5, -8.0, 4.5)
            results = {}
            for geometry in ("nearest_wall", "pairwise"):
                results[geometry] = evaluate_reference(
                    state["positions"], velocities, densities, pressures, state["masses"], state["is_wall"],
                    dataclasses.replace(constants, geometry=geometry, gravity=gravity))
            # farther than the support radius from any second wall plane: the second smallest gap exceeds h
            one_wall = state["gap_sorted"][:, 1] > state["support"]
            near_one_wall = one_wall & (state["gap_sorted"][:, 0] < state["support"])
            rate_difference = np.abs(results["pairwise"].density_rate - results["nearest_wall"].density_rate)
            acceleration_difference = np.sqrt(((results["pairwise"].acceleration
                                                - results["nearest_wall"].acceleration) ** 2).sum(axis=1))
            rate_scale = float(np.abs(results["nearest_wall"].density_rate).max())
            acceleration_scale = float(np.sqrt((results["nearest_wall"].acceleration ** 2).sum(axis=1)).max())
            kernel_sum = results["nearest_wall"].kernel_sum
            entry = {
                "fluid": int(state["fluid_count"]), "one_wall_or_none": int(one_wall.sum()),
                "within_h_of_exactly_one_wall": int(near_one_wall.sum()),
                "within_h_of_two_or_more_walls": int((~one_wall).sum()),
                "rate_difference_maximum_one_wall": float(rate_difference[one_wall].max()),
                "acceleration_difference_maximum_one_wall": float(acceleration_difference[one_wall].max()),
                "rate_difference_maximum_corner_over_field": float(rate_difference[~one_wall].max() / rate_scale),
                "acceleration_difference_maximum_corner_over_field": float(
                    acceleration_difference[~one_wall].max() / acceleration_scale),
                "kernel_sum_range": [float(kernel_sum.min()), float(kernel_sum.max())],
                "passed": bool(rate_difference[one_wall].max() == 0.0
                               and acceleration_difference[one_wall].max() == 0.0),
            }
            summary["equality"][f"{dimension}d"] = entry
            equality_rows.append([
                f"{dimension}-D（{' × '.join(str(count) for count in specification.active_counts())}，h/dx = "
                f"{specification.support_ratio:g}）",
                f"{entry['fluid']:,}", f"{entry['within_h_of_exactly_one_wall']:,}",
                f"{entry['rate_difference_maximum_one_wall']:.1e}",
                f"{entry['acceleration_difference_maximum_one_wall']:.1e}",
                f"{entry['within_h_of_two_or_more_walls']:,}",
                f"{100 * entry['rate_difference_maximum_corner_over_field']:.1f} %",
                f"{100 * entry['acceleration_difference_maximum_corner_over_field']:.1f} %"])

            # ---- 3. hydrostatic field, the reflection configuration ----------------------------------------
            velocities[:] = 0.0
            densities[:] = state["rest_density"]
            centre = 0.5 * (state["box_lower"] + state["box_upper"])
            tilt = np.deg2rad(30.0)
            directions = (("重力沿 −y", np.array([0.0, -1.0, 0.0])),
                          ("重力偏离 −y 30°", np.array([np.sin(tilt), -np.cos(tilt), 0.0])))
            corner = ~one_wall
            for direction_text, direction in directions:
                gravity_vector = GRAVITY_MAGNITUDE * direction
                hydrostatic = state["rest_density"] * ((fluid_positions - centre) * gravity_vector).sum(axis=1)
                swing = float(np.abs(hydrostatic).max())
                for sign_text, offset in (("p > 0", 3.0 * swing), ("p < 0", -3.0 * swing)):
                    pressures[:] = 0.0
                    pressures[:state["fluid_count"]] = offset + hydrostatic
                    for configuration in ("reflection", "nearest_wall"):
                        if configuration == "reflection":
                            variant = dataclasses.replace(constants, geometry="pairwise", pressure_wall="reflection",
                                                          delta_wall="reflection", gravity=tuple(gravity_vector))
                        else:
                            variant = dataclasses.replace(constants, gravity=tuple(gravity_vector))
                        result = evaluate_reference(state["positions"], velocities, densities, pressures,
                                                    state["masses"], state["is_wall"], variant)
                        residual = np.sqrt((result.acceleration ** 2).sum(axis=1)) / GRAVITY_MAGNITUDE
                        key = f"{dimension}d|{direction_text}|{sign_text}|{configuration}"
                        entry = {
                            "residual_maximum": float(residual.max()),
                            "residual_maximum_corner": float(residual[corner].max()),
                            "fluid_inverse_fallbacks": int((~result.fluid_inverse_valid).sum()),
                            "fluid_inverse_capped": int(result.fluid_inverse_capped.sum()),
                            "correction_capped": int(result.correction_capped.sum()),
                            "density_rate_maximum": float(np.abs(result.density_rate).max()),
                            "passed": bool(residual.max() <= HYDROSTATIC_LIMIT),
                        }
                        summary["hydrostatic"][key] = entry
                        hydrostatic_rows.append([
                            f"{dimension}-D", direction_text, sign_text,
                            "反射版" if configuration == "reflection" else "一阶（现缺省）",
                            f"{entry['residual_maximum']:.1e}", f"{entry['residual_maximum_corner']:.1e}",
                            f"{entry['fluid_inverse_fallbacks']} / {entry['fluid_inverse_capped']}",
                            "通过" if entry["passed"] else "**不通过**"])
        write_table("pairwise_numpy_equality.md",
                    ["盒", "流体粒子", "离恰好一面壁不到 h 的粒子", "这些粒子上密度变化率之差的最大值", "加速度之差的最大值",
                     "离两面以上壁不到 h 的粒子（角）", "角里密度变化率之差 / 全场最大值", "角里加速度之差 / 全场最大值"],
                    equality_rows)
        write_table("pairwise_numpy_hydrostatic.md",
                    ["维数", "重力方向", "压力符号", "配置", "全部流体粒子上 |a_p + g| / |g| 的最大值", "其中角里的最大值",
                     "A_f 回退 / 被 Frobenius 上限缩小的粒子数", f"≤ {HYDROSTATIC_LIMIT:.0e}"], hydrostatic_rows)
        summary["passed"] = bool(
            summary["example_passed"]
            and all(entry["passed"] for entry in summary["equality"].values())
            and all(entry["passed"] for key, entry in summary["hydrostatic"].items()
                    if key.startswith("2d") and key.endswith("|reflection")))
        summary["passed_three_dimensions"] = bool(
            all(entry["passed"] for key, entry in summary["hydrostatic"].items()
                if key.startswith("3d") and key.endswith("|reflection")))
        out = REPORT_ROOT / "pairwise_reflection"
        out.mkdir(parents=True, exist_ok=True)
        (out / "numpy_check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8",
                                              newline="\n")
        run.extra["passed"] = summary["passed"]
        run.extra["status"] = "ok" if summary["passed"] else "failed"
        print(f"[pairwise_reflection] numpy check {run.run_id}: example 1 / t = "
              f"{summary['examples'][0]['pairwise_factor']:.12f} (3.5 / 3 = {3.5 / 3.0:.12f}), old rule "
              f"{summary['examples'][0]['nearest_wall_factor']:.3f}")
        for key, entry in summary["equality"].items():
            print(f"  {key}: pairwise = nearest_wall away from corners: rate diff "
                  f"{entry['rate_difference_maximum_one_wall']:.1e}, acceleration diff "
                  f"{entry['acceleration_difference_maximum_one_wall']:.1e} on "
                  f"{entry['within_h_of_exactly_one_wall']:,} particles; corners differ by "
                  f"{100 * entry['rate_difference_maximum_corner_over_field']:.1f} % / "
                  f"{100 * entry['acceleration_difference_maximum_corner_over_field']:.1f} % of the field; kernel "
                  f"sum {entry['kernel_sum_range'][0]:.4f} .. {entry['kernel_sum_range'][1]:.4f}")
        for key, entry in summary["hydrostatic"].items():
            print(f"  {key}: |a_p + g| / |g| max {entry['residual_maximum']:.2e} (corners "
                  f"{entry['residual_maximum_corner']:.2e}), density rate max {entry['density_rate_maximum']:.3e}, "
                  f"A_f fallbacks {entry['fluid_inverse_fallbacks']}, capped {entry['fluid_inverse_capped']}")
        print(f"[pairwise_reflection] numpy check: {'PASS' if summary['passed'] else 'FAIL'} (2-D); 3-D hydrostatic "
              f"{'PASS' if summary['passed_three_dimensions'] else 'FAIL'}")
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("make-cases", "numpy-check", "tables"))
    parser.add_argument("--all", action="store_true",
                        help="make-cases: also the three variants that locate a failure of the screening")
    arguments = parser.parse_args()
    if arguments.action == "make-cases":
        written = make_cases(arguments.all)
        for path in written:
            print(f"[pairwise_reflection] wrote {path.relative_to(REPOSITORY_ROOT)}")
        print(f"[pairwise_reflection] {len(written)} case file(s) written")
        return 0
    if arguments.action == "numpy-check":
        return numpy_check()
    return tables()


def tables() -> int:
    raise SystemExit("tables: not written yet")


if __name__ == "__main__":
    sys.exit(main())
