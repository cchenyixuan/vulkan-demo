"""
pairwise_reflection.py - wall_boundary crossing with the pair-by-pair geometry and the reflection of pressure and
density at the wall (numerics.wall_crossing.geometry pairwise, pressure_wall reflection, delta_wall reflection;
shaders/wall_boundary.glsl): its cases, the numpy check and the tables of section 12 of
claude/wall_crossing_report.md.

    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection make-cases [--all]
    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection numpy-check
    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection tables
    .venv/Scripts/python.exe -m experiment.wall_crossing.pairwise_reflection long-runs

tables = section 12 (the screening; the long run of the reflection configuration as it was when it was stopped at
t = 18.06: its directory was copied to cavity/n250_reflection_stopped_t18 before the run was resumed).
long-runs = section 13: the two runs to t = 100 (n250_reflection resumed from its checkpoint at t = 10.08, and
n250_pairwise_full) beside the existing ones, their histories, and the state of the runs at t = 18.
full-scheme = section 14: the complete scheme (reflection_full: pairwise geometry, whole vector, both
reflections) run to t = 100 from rho0 (n250_reflection_full) and from rho0 + 0.3 kg/m3
(n250_reflection_full_rho_plus_0p3): the cavity metrics, the mean density at fixed instants and its rate over
t in [60, 100], the mean pressure, the share of negative pressures and the fluctuation of the mean pressure;
and, for the runs without a field history (the old default), the same fluctuation from the stored snapshots.
volume-constraint = section 15: the complete scheme with numerics.volume_constraint (reflection_full_volume) at
250^2 and 500^2: the regression, the single step, the short run, the cavity metrics and the two-point order, the
histories of density, theta, pressure, the time per step against the first order, the hydrostatic boxes.

Configurations (the names used in case directories, run names and tables):
    reflection                    geometry pairwise, pressure_wall reflection, delta_wall reflection ("反射版")
    pairwise_delta_reflection     geometry pairwise, delta_wall reflection only        (--all: the three variants
    pairwise_pressure_reflection  geometry pairwise, pressure_wall reflection only      that locate a failure of
    pairwise                      geometry pairwise only                                the screening)
    pairwise_full                 geometry pairwise with continuity_wall_component full: the default until
                                  2026-11-18 (the whole vector in the continuity equation) on the pairwise geometry
    reflection_full               the reflection configuration with continuity_wall_component full ("完整方案":
                                  pairwise geometry, whole vector, both reflections)
    reflection_full_volume        the complete scheme with numerics.volume_constraint enabled (relaxation time 1,
                                  the default); reflection_full_volume_tau5: the same with relaxation time 5
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
    "pairwise_full": ("pairwise", "extension", "extension"),
    "reflection_full": ("pairwise", "reflection", "reflection"),
    "reflection_full_volume": ("pairwise", "reflection", "reflection"),
    "reflection_full_volume_tau5": ("pairwise", "reflection", "reflection"),
}
# numerics.wall_crossing.continuity_wall_component of the configurations whose name says full; the others were
# made under the default of 2026-11-18 (normal) and write it out since the defaults changed on 2026-11-19
CONTINUITY_WALL_COMPONENT = {"pairwise_full": "full", "reflection_full": "full", "reflection_full_volume": "full",
                             "reflection_full_volume_tau5": "full"}
CONTINUITY_WALL_COMPONENT_OTHERWISE = "normal"
# configurations with numerics.volume_constraint enabled: the keys written beside "enabled: true"
VOLUME_CONSTRAINT = {"reflection_full_volume": {}, "reflection_full_volume_tau5": {"relaxation_time": 5.0}}
# (source case, configuration) made by make-cases beside the main configuration of SOURCE_CASES
EXTRA_CASES = (
    ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing", "reflection_full_volume"),
    ("lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing", "reflection_full_volume"),
    ("wall_crossing/hydrostatic_2d_n250_gy_crossing", "reflection_full_volume"),
    ("wall_crossing/hydrostatic_2d_n250_g30_crossing", "reflection_full_volume"),
)
# made only with --tau5 (the fallback of the task when the constraint oscillates with relaxation time 1)
TAU5_CASES = (("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing", "reflection_full_volume_tau5"),)
CONFIGURATION_TEXT = {
    "reflection": "反射版（逐对几何 + 压力反射 + 密度反射）",
    "pairwise_delta_reflection": "逐对几何 + 只有密度反射",
    "pairwise_pressure_reflection": "逐对几何 + 只有压力反射",
    "pairwise": "只有逐对几何",
    "pairwise_full": "逐对几何 + 连续性取完整向量（旧缺省换成逐对几何）",
    "reflection_full": "完整方案（逐对几何 + 完整向量 + 压力反射 + 密度反射）",
    "reflection_full_volume": "完整方案 + 体积约束",
    "reflection_full_volume_tau5": "完整方案 + 体积约束（松弛时间 5）",
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
    """The source case with the configuration's keys in numerics.wall_crossing: all four switches of the complete
    scheme written out (whatever the solver's defaults are), the source's numerics.volume_constraint replaced by
    the configuration's (none, or VOLUME_CONSTRAINT)."""
    geometry, pressure_wall, delta_wall = CONFIGURATIONS[configuration]
    continuity_wall_component = CONTINUITY_WALL_COMPONENT.get(configuration, CONTINUITY_WALL_COMPONENT_OTHERWISE)
    lines = without_volume_constraint(source_text.split("\n"))
    mapping_rows = [index for index, line in enumerate(lines) if line.startswith("  wall_crossing:")]
    if len(mapping_rows) != 1:
        raise ValueError(f"{source_name}: expected exactly one numerics.wall_crossing mapping")
    # (the case's own top-level "geometry:" section, the particle files, is not indented)
    if any(line.startswith(("    geometry:", "    pressure_wall:", "    delta_wall:")) for line in lines):
        raise ValueError(f"{source_name}: the source already sets geometry, pressure_wall or delta_wall")
    if any(line.startswith("    continuity_wall_component:") for line in lines):
        raise ValueError(f"{source_name}: the source already sets continuity_wall_component")
    added = [f"    geometry: {geometry}             # pairwise | nearest_wall",
             f"    pressure_wall: {pressure_wall}      # reflection | extension",
             f"    delta_wall: {delta_wall}         # reflection | extension",
             f"    continuity_wall_component: {continuity_wall_component}   # full | normal"]
    lines[mapping_rows[0] + 1:mapping_rows[0] + 1] = added
    if configuration in VOLUME_CONSTRAINT:
        constraint = ["  volume_constraint:        # the fluid's total volume is held (closed container)",
                      "    enabled: true"]
        constraint += [f"    {key}: {value!r}" for key, value in VOLUME_CONSTRAINT[configuration].items()]
        lines[mapping_rows[0]:mapping_rows[0]] = constraint
    if lines[0] != "schema_version: 2":
        raise ValueError(f"{source_name}: does not start with the schema line")
    header = [f"# VARIANT of cases/{source_name} (experiment/wall_crossing/pairwise_reflection.py): geometry "
              f"{geometry},",
              f"# pressure_wall {pressure_wall}, delta_wall {delta_wall}"
              + f", continuity_wall_component {continuity_wall_component}"
              + (", numerics.volume_constraint enabled" + "".join(
                  f", {key} {value!r}" for key, value in VOLUME_CONSTRAINT[configuration].items())
                 if configuration in VOLUME_CONSTRAINT else "") + "; everything else identical."]
    lines[2:2] = header + [""]
    return "\n".join(lines)


def without_volume_constraint(lines: list) -> list:
    """The lines of a case without its numerics.volume_constraint block (the default crossing cases carry one
    since 2026-11-19; the variants made before had none unless they asked for it)."""
    kept, inside = [], False
    for line in lines:
        if line.startswith("  volume_constraint:"):
            inside = True
            continue
        if inside and line.startswith("    "):
            continue
        inside = False
        kept.append(line)
    return kept


def variant_name(source_name: str, configuration: str) -> str:
    return f"{source_name}_{configuration}"


def make_cases(every_configuration: bool, relaxation_time_five: bool = False) -> list:
    written = []
    wanted = {}
    for source_name in SOURCE_CASES:
        wanted[source_name] = [MAIN_CONFIGURATION]
        if every_configuration and source_name in LOCATING_SOURCES:
            wanted[source_name] = [configuration for configuration in CONFIGURATIONS
                                   if configuration not in VOLUME_CONSTRAINT]
    for source_name, configuration in EXTRA_CASES + (TAU5_CASES if relaxation_time_five else ()):
        wanted.setdefault(source_name, []).append(configuration)
    for source_name, configurations in wanted.items():
        source_text = (REPOSITORY_ROOT / "cases" / source_name / "case.yaml").read_text(encoding="utf-8")
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
    parser.add_argument("action", choices=("make-cases", "numpy-check", "tables", "long-runs", "full-scheme",
                                           "volume-constraint"))
    parser.add_argument("--all", action="store_true",
                        help="make-cases: also the three variants that locate a failure of the screening")
    parser.add_argument("--tau5", action="store_true",
                        help="make-cases: also the 250^2 complete scheme with the volume constraint at "
                             "relaxation time 5")
    arguments = parser.parse_args()
    if arguments.action == "make-cases":
        written = make_cases(arguments.all, arguments.tau5)
        for path in written:
            print(f"[pairwise_reflection] wrote {path.relative_to(REPOSITORY_ROOT)}")
        print(f"[pairwise_reflection] {len(written)} case file(s) written")
        return 0
    if arguments.action == "numpy-check":
        return numpy_check()
    if arguments.action == "long-runs":
        return long_run_tables()
    if arguments.action == "full-scheme":
        return full_scheme_tables()
    if arguments.action == "volume-constraint":
        return volume_constraint_tables()
    return tables()


# ======================================================================================================== tables
SCREENING_LIMITS = {"speed_maximum": 1.05, "interior_continuity": 0.005}
# (run name, text, whether the run is one of this task's)
SCREENING_RUNS = (
    ("budget_n250_crossing_reflection", "反射版", True),
    ("budget_n250_crossing_pairwise_delta_reflection", "逐对几何 + 密度反射（关压力反射）", True),
    ("budget_n250_crossing_pairwise_pressure_reflection", "逐对几何 + 压力反射（关密度反射）", True),
    ("budget_n250_crossing_pairwise", "只有逐对几何（两个反射都关）", True),
    ("budget_n250_crossing_normal_r1", "一阶（现缺省），第 1 次", False),
    ("budget_n250_crossing_normal_r2", "一阶（现缺省），第 2 次", False),
    ("budget_n250_crossing", "旧缺省（完整向量）", False),
)
HYDROSTATIC_DIRECTIONS = {"gy": "重力沿 −y", "g30": "重力偏离 −y 30°"}
LONG_RUN_INSTANTS = (0.2, 1.0, 5.0, 10.0, 18.0, 20.0, 40.0, 60.0, 80.0, 90.0, 100.0)
WALLS = (("左壁", 0, -1), ("右壁", 0, +1), ("底壁", 1, -1), ("盖（上壁）", 1, +1))
HALF_BOX = 0.502
# the reflection run as it was when it was stopped at t = 18.06 (a copy made before the run was resumed)
STOPPED_RUN = "n250_reflection_stopped_t18"
STOPPED_RECORD = "wc-20261118-165529-7df1"          # the launcher's record of that first attempt


def tables() -> int:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing import delta_wall_experiment, density_budget, normal_default
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, run_records
    from experiment.wall_crossing.run_log import LOG_ROOT

    write_text = normal_default.write_table
    record_text = normal_default.record_text
    new_colour, first_colour, old_colour = INK_PRIMARY, normal_default.NEW_COLOUR, normal_default.OLD_COLOUR
    records = run_records()
    records_by_id = {record["run_id"]: record for record in records}
    records_by_name = {record["name"]: record for record in records if record.get("name")}
    summary = {}

    # ---------------------------------------------------------------- 4.1 regression
    rows = []
    manifest_now = normal_default.manifest_rows((REPOSITORY_ROOT / normal_default.MANIFEST).read_text(encoding="utf-8"))
    manifest_base = normal_default.manifest_rows(normal_default.git_show(normal_default.BASE_COMMIT,
                                                                         normal_default.MANIFEST))
    manifest_before = normal_default.manifest_rows(normal_default.git_show("b18597b", normal_default.MANIFEST))
    same_as_base = sorted(name for name in manifest_base if manifest_now.get(name) == manifest_base[name])
    changed_since = sorted(name for name in manifest_now if manifest_before.get(name) != manifest_now[name])
    rows.append(["基点清单里的 SPIR-V 文件", f"基点（{normal_default.BASE_COMMIT}）清单里的 {len(manifest_base)} 个文件",
                 f"{len(same_as_base)} 个逐字节相同", "通过" if len(same_as_base) == len(manifest_base) else "**不通过**",
                 "MANIFEST.txt"])
    rows.append(["相对 b18597b 变了的 SPIR-V", f"全部 {len(manifest_now)} 个文件",
                 "、".join(name.replace(".comp.spv", "") for name in changed_since) or "无", "—", "MANIFEST.txt"])
    summary["regression"] = {"manifest_same_as_base": len(same_as_base), "manifest_base": len(manifest_base),
                             "changed_since_b18597b": changed_since}
    classes_path = LOG_ROOT / "outcome_classes" / "defaults_against_b18597b_s700" / "classes.json"
    if classes_path.exists():
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        members = "；".join(f"类 {index}：参照 {group['reference_runs']} 次，本次 {group['case_runs']} 次"
                           for index, group in enumerate(classes["classes"]))
        rows.append(["crossing，三个开关取缺省，250²，700 步",
                     "b18597b 的着色器的 6 个转储（第 10 节 5.1 的那 6 次；db3d609 之后到 b18597b 着色器没有改动）",
                     f"{len(classes['classes'])} 个结果类。{members}",
                     "逐位相同" if classes["case_within_reference_classes"] else "**不同**", classes["run_id"]])
        summary["regression"]["crossing_defaults"] = {"classes": len(classes["classes"]),
                                                      "identical": classes["case_within_reference_classes"],
                                                      "run_id": classes["run_id"]}
    failures = LOG_ROOT / "tests" / "failures_pairwise_reflection.txt"
    base_failures = LOG_ROOT / "tests" / "failures_base.txt"
    if failures.exists() and base_failures.exists():
        now = sorted(line.strip() for line in failures.read_text(encoding="utf-8").splitlines() if line.strip())
        base = sorted(line.strip() for line in base_failures.read_text(encoding="utf-8").splitlines() if line.strip())
        seam = [record for record in records if record.get("kind") == "seam_layout_test"]
        rows.append(["`_test_seam_layout.py` 的失败清单", "基点的清单（858 条，基点自身就有）",
                     f"{len(now)} 条，" + ("与基点的清单逐行相同" if now == base else "**与基点不同**"),
                     "通过" if now == base else "**不通过**", seam[-1]["run_id"] if seam else "?"])
        summary["regression"]["seam_failures"] = {"count": len(now), "same_as_base": now == base}
    write_text("pairwise_regression.md", markdown_table(["检查", "参照", "结果", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 4.3 GPU single step against numpy
    rows = []
    summary["single_step"] = {}
    suffix = "_delta_wall_reflection_wall_component_normal_geometry_pairwise_pressure_reflection"
    for dimension in (2, 3):
        for build_text, build_suffix in (("缺省构建", ""), ("收支构建", "_budget"),
                                         ("缺省构建，回退测试（L 和 A_f 的行列式阈值 1.27）", "_det_threshold_1p27")):
            budget_build = build_suffix == "_budget"
            path = LOG_ROOT / "single_step" / f"summary_{dimension}d{suffix}{build_suffix}.json"
            if not path.exists():
                continue
            for outcome in json.loads(path.read_text(encoding="utf-8")):
                if outcome["configuration"] != "smooth" or outcome["wall_boundary"] != "crossing":
                    continue
                reference = outcome["reference"]
                comparisons = reference["comparisons"]
                terms = reference["reflection_terms"]
                budget = outcome.get("budget")
                rows.append([
                    f"{dimension}-D", build_text,
                    f"{outcome['particles']['fluid']:,} / {reference['counts']['fluid_with_wall_neighbours']:,}",
                    f"{reference['counts']['fluid_inverse_fallbacks']:,}",
                    f"{reference['counts']['fluid_below_distance_floor']}；{terms['extrapolation_factor_maximum']:.0f}",
                    f"{terms['pressure_reflection_acceleration_maximum']:.1f} / "
                    f"{terms['pressure_acceleration_maximum']:.1f}",
                    f"{terms['density_reflection_rate_maximum']:.0f} / {terms['density_rate_maximum']:.0f}",
                    f"{comparisons['acceleration']['error_over_pair_term_sum_maximum']:.1e}",
                    f"{comparisons['density_rate']['relative_error_beyond_one_ulp_maximum']:.1e}",
                    "—" if not budget else f"{budget['worst_term_error_over_pair_terms_times_10']:.1e}",
                    f"{reference['worst_relative_error']:.1e}",
                    "通过" if outcome["passed"] else "**不通过**", outcome["run_id"]])
                summary["single_step"][outcome["label"]] = {
                    "passed": outcome["passed"], "worst": reference["worst_relative_error"],
                    "run_id": outcome["run_id"], "reflection_terms": terms,
                    "derivation_step": reference["derivation_step"]}
    if rows:
        write_text("pairwise_single_step.md", markdown_table(
            ["维数", "构建", "流体粒子 / 其中有壁邻居的", "A_f 回退的粒子（反射项取零）", "低于距离下限的粒子；最大的 1/t_ij",
             "压力反射项的最大值 / 压力加速度的最大值", "密度反射项的最大值 / 密度变化率的最大值",
             "加速度误差 ÷ 逐对项绝对值之和（最大）", "密度变化率的相对误差（超出 1 ulp 的部分，最大）",
             "收支四项：10 × 误差 ÷ 逐对项之和（最大）", "判据值（容差 1e-4）", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 4.4 hydrostatic boxes
    hydrostatic_runs = {}
    for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
        entries = ((f"2d_n250_{direction}_crossing_normal_r1", "一阶（现缺省），第 1 次", 0),
                   (f"2d_n250_{direction}_crossing_normal_r2", "一阶（现缺省），第 2 次", 0),
                   (f"2d_n250_{direction}_crossing_reflection", "反射版", 1))
        present = tuple(entry for entry in entries if (LOG_ROOT / "hydrostatic" / entry[0] / "summary.json").exists())
        if any(form == 1 for _, _, form in present):
            hydrostatic_runs[direction] = (direction_text, present)
    if len(hydrostatic_runs) == len(HYDROSTATIC_DIRECTIONS):
        introduction = (
            "2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始，到 t = 10.02）。速度以 "
            "√(|g| H) 为单位，压力以 ρ0 |g| H 为单位。“去电平” = 减去全场平均偏差后的误差；“电平” = 压力减去 "
            "ρ0 g·(r − r_中心) 之后对全部流体粒子的平均。一阶的两次是第 9 节的运行，反射版每个方向一次。")
        hydrostatic = delta_wall_experiment.hydrostatic_comparison(
            records_by_id, runs_by_direction=hydrostatic_runs,
            forms=(("一阶（现缺省）", first_colour, "solid"), ("反射版", new_colour, "solid")),
            introduction=introduction, table_name="pairwise_hydrostatic.md", figure_name="pairwise_hydrostatic.png",
            figure_title="2-D 静水盒：一阶（现缺省）与反射版（压力以 ρ0 g H 为单位）")
        ratio_rows = []
        summary["hydrostatic"] = {"runs": hydrostatic, "ratios": {}}
        for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
            entries = hydrostatic.get(direction, {})
            first = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                     if "normal_r" in name]
            second = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                      if name.endswith("_reflection")]
            if not first or not second:
                continue
            ratio = second[0] / (sum(first) / len(first))
            summary["hydrostatic"]["ratios"][direction] = {"first_order": first, "reflection": second[0],
                                                          "ratio_to_first_order_mean": ratio}
            ratio_rows.append([direction_text, " / ".join(f"{value:.2e}" for value in first), f"{second[0]:.2e}",
                               f"{ratio:.2f}", "通过" if ratio <= 2.0 else "**不通过**"])
        write_text("pairwise_hydrostatic_ratio.md", markdown_table(
            ["方向", "一阶的残余速度 rms / √(gH)（末 1/5 段；两次）", "反射版", "反射版 ÷ 一阶两次的平均", "不超过 2 倍"],
            ratio_rows))

    # ---------------------------------------------------------------- 4.5 screening and the locating runs
    def load_series(name: str) -> dict:
        with open(LOG_ROOT / "box_run" / name / "series.csv", encoding="utf-8", newline="") as handle:
            series_rows = list(csv.DictReader(handle))
        return {key: np.array([float(row[key]) if row[key] != "" else np.nan for row in series_rows])
                for key in series_rows[0]}

    rows, term_rows, region_rows, diagnostic_rows = [], [], [], []
    region_groups = (("静止壁带（左、右、下）", ("left", "right", "bottom")), ("盖带", ("top",)),
                     ("下面两个角", ("bottom_left", "bottom_right")), ("盖的左角（上游）", ("top_left",)),
                     ("盖的右角（下游）", ("top_right",)))
    summary["screening"] = {}
    for name, text, screened in SCREENING_RUNS:
        if not (LOG_ROOT / "box_run" / name / "series.csv").exists():
            continue
        result = density_budget.analyse(name)
        series = load_series(name)
        groups = result["groups"]
        everything = groups["all"]
        interior = groups["interior"]["rate"][density_budget.ROW_DRIFT]
        interior_error = groups["interior"]["standard_error"][density_budget.ROW_DRIFT]
        conserved = bool(series["alive"].min() == series["alive"].max())
        outside = int(series["outside_box_count"].max())
        below_floor = int(series["below_floor_count"].max())
        speed = float(series["speed_maximum"].max())
        checks = {"conserved": conserved, "inside": outside == 0,
                  "speed": speed <= SCREENING_LIMITS["speed_maximum"],
                  "interior": abs(interior) <= SCREENING_LIMITS["interior_continuity"]}
        passed = all(checks.values())
        record = records_by_name.get(name, {})
        # sample-to-sample fluctuation of the mean pressure over t in [1, 5]: the rms of (a sample minus the mean
        # of its two neighbours), samples every 0.03 time units
        windowed = series["pressure_mean"][series["time"] >= 1.0] / normal_default.DYNAMIC_PRESSURE
        pressure_fluctuation = float(np.sqrt(((windowed[1:-1] - 0.5 * (windowed[:-2] + windowed[2:])) ** 2).mean()))
        entry = {
            "interior_continuity": interior, "interior_continuity_error": interior_error,
            "density_mean_fit_slope": result["closure"]["density_mean_fit_slope"],
            "terms_all": [float(value) for value in everything["rate"]],
            "terms_all_error": [float(value) for value in everything["standard_error"]],
            "checks": checks, "passed": passed if screened else None, "speed_maximum": speed,
            "below_floor_maximum": below_floor, "outside_maximum": outside,
            "wall_distance_minimum_over_spacing": float(series["wall_distance_minimum_over_spacing"].min()),
            "density_mean_minus_rest_at_1": float(np.interp(1.0, series["time"], series["density_mean"])
                                                  - normal_default.REST_DENSITY),
            "density_mean_minus_rest_last": float(series["density_mean"][-1] - normal_default.REST_DENSITY),
            "density_deviation_maximum": float(series["density_deviation_maximum"].max()),
            "density_deviation_maximum_time": float(series["time"][int(np.nanargmax(
                series["density_deviation_maximum"]))]),
            "density_deviation_maximum_after_1": float(
                series["density_deviation_maximum"][series["time"] >= 1.0].max()),
            "pressure_mean_last": float(series["pressure_mean"][-1]),
            "pressure_negative_fraction_last": float(series["pressure_negative_fraction"][-1]),
            "kinetic_energy_last": float(series["kinetic_energy"][-1]),
            "pressure_mean_fluctuation": pressure_fluctuation,
            "gradient_fallbacks": int(series["gradient_fallbacks"][-1]),
            "window": [float(result["window"][0]), float(result["window"][1])] if "window" in result else None,
            "run_id": record.get("run_id")}
        summary["screening"][name] = entry
        rows.append([
            f"**{text}**" if screened else text,
            f"{int(series['alive'][0]):,}，" + ("守恒" if conserved else "**不守恒**") + f"；越出盒 {outside}",
            f"{speed:.3f}",
            f"{interior:+.5f} ± {interior_error:.5f}" + ("" if checks["interior"] else "（**超过 0.005**）"),
            f"{everything['rate'][density_budget.ROW_DIFFUSION]:+.5f}（流体 {everything['rate'][2]:+.5f}，壁 "
            f"{everything['rate'][3]:+.5f}）",
            f"{entry['density_mean_fit_slope']:+.5f}",
            f"{below_floor}；{entry['wall_distance_minimum_over_spacing']:.3f}",
            ("**通过**" if passed else "**不通过**") if screened else "（参照）",
            record_text(record)])
        term_rows.append([
            f"**{text}**" if screened else text,
            f"{everything['rate'][0]:+.5f}", f"{everything['rate'][1]:+.5f}",
            f"{everything['rate'][density_budget.ROW_DRIFT]:+.5f}",
            f"{everything['rate'][density_budget.ROW_DIFFUSION]:+.5f}",
            f"{everything['rate'][density_budget.ROW_SUM]:+.5f} ± "
            f"{everything['standard_error'][density_budget.ROW_SUM]:.5f}",
            f"{everything['rate'][density_budget.ROW_ROUNDING]:+.5f}",
            f"{everything['rate'][density_budget.ROW_APPLIED]:+.5f}",
            f"{entry['density_mean_minus_rest_at_1']:+.4f} / {entry['density_mean_minus_rest_last']:+.4f}",
            f"{entry['pressure_mean_last'] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²；"
            f"{100 * entry['pressure_negative_fraction_last']:.2f} %",
            f"{pressure_fluctuation:.4f}",
            f"{entry['density_deviation_maximum']:.3f}（t = {entry['density_deviation_maximum_time']:.2f}）；"
            f"{entry['density_deviation_maximum_after_1']:.3f}",
            f"{entry['kinetic_energy_last']:.3f}"])
        if "extension_left_wall_steps" in series:
            entry["diagnostics"] = {}
            for group_text, regions in region_groups:
                factor = max(series[f"extension_{region}_factor_maximum"][-1] for region in regions)
                number = max(series[f"extension_{region}_diffusion_maximum"][-1] for region in regions)
                entry["diagnostics"][group_text] = {"extrapolation_factor_maximum": float(np.sqrt(factor)),
                                                    "diffusion_maximum": float(number)}
            diagnostic_rows.append([f"**{text}**" if screened else text] + [
                f"{entry['diagnostics'][group_text]['extrapolation_factor_maximum']:.1f} / "
                f"{entry['diagnostics'][group_text]['diffusion_maximum']:.3f}"
                for group_text, _ in region_groups])
        region_rows.append([f"**{text}**" if screened else text] + [
            f"{groups[key]['rate'][density_budget.ROW_DRIFT]:+.5f} / "
            f"{groups[key]['rate'][density_budget.ROW_DIFFUSION]:+.5f}"
            for key in ("lid", "static", "lid_corner_left", "lid_corner_right", "bottom_corners", "interior")])
        if screened:
            write_text(f"density_{name}.md", density_budget.budget_table(
                result, f"crossing，{text}",
                ("连续性项，流体邻居", "连续性项，壁邻居（一阶，只取法向）", "δ 项，流体邻居", "δ 项，壁邻居"), record))
    if rows:
        write_text("pairwise_screening.md", markdown_table(
            ["运行", "粒子数（含壁）", "流体最大速度（盖速 = 1；线 1.05）", "内部的连续性项（线 ±0.005）",
             "δ 项全场合计", "平均密度的变化率（线性拟合斜率）", "低于距离下限的粒子数（各样本的最大值）；φ_min / dx",
             "结论", "记录（提交）"], rows))
        write_text("pairwise_screening_terms.md", markdown_table(
            ["运行", "连续性项，流体邻居", "连续性项，壁邻居", "连续性项合计", "δ 项合计", "四项之和", "float32 存储舍入",
             "实际写入", "平均密度 − ρ0：t = 1 / t = 5", "t = 5 的平均压力；负压粒子比例",
             "平均压力的逐样本涨落，t ∈ [1, 5]（ρ0U²）", "|ρ − ρ0| 的最大值：整个运行（出现的时刻）；t ≥ 1", "t = 5 的动能"], term_rows))
    if diagnostic_rows:
        write_text("pairwise_diagnostics.md", markdown_table(
            ["运行（每格 = 最大的 1/t_ij / 壁面粘性扩散数的最大值，整个运行）"] + [group_text for group_text, _ in region_groups],
            diagnostic_rows))
        write_text("pairwise_screening_regions.md", markdown_table(
            ["运行", "盖带", "静止壁带", "盖的左角（上游）", "盖的右角（下游）", "下面两个角", "内部"], region_rows))

    # ---------------------------------------------------------------- 4.6 cavity against Marchi 2021
    tables_path = REPORT_ROOT / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}
    # no configuration passed the screening, so section 12 has no run to t = 100 (the two runs made afterwards
    # at the user's request are section 13's: long_run_tables below)
    cavity_configuration = None
    summary["cavity_configuration"] = cavity_configuration
    if cavity_configuration:
        cavity = normal_default.cavity(
            modes=((cavity_configuration, f"crossing，{CONFIGURATION_TEXT[cavity_configuration]}", new_colour, "P"),
                   ("crossing_normal", "crossing，一阶（现缺省）", first_colour, "o"),
                   ("quadratic_viscous", "crossing，二阶粘性", INK_SECONDARY, "X"),
                   ("crossing", "crossing，旧缺省（一阶，完整向量）", old_colour, "s"),
                   ("simple", "simple", normal_default.SIMPLE_COLOUR, "^"),
                   ("adami", "adami", normal_default.ADAMI_COLOUR, "D")),
            extra_runs=(), prefix="pairwise", required_mode=cavity_configuration,
            labelled_modes=(cavity_configuration, "crossing_normal"), labels_beside=True,
            figure_title="Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧；数字 = "
                         f"{CONFIGURATION_TEXT[cavity_configuration].split('（')[0]}和一阶）")
        if cavity:
            summary["cavity"] = cavity

    # ---------------------------------------------------------------- 4.6 histories of the long runs
    long_runs = [(STOPPED_RUN, f"250² {CONFIGURATION_TEXT['reflection']}（停在中途）", new_colour, (0, (1, 1.5)))]
    long_runs.append(("n250_crossing_normal", "250² 一阶（现缺省）", first_colour, "solid"))
    histories = {name: normal_default.load_field_history(name) for name, _, _, _ in long_runs}
    histories = {name: history for name, history in histories.items() if history and len(history["time"]) > 5}
    summary["long_runs"] = {}
    rows, summary_rows = [], []
    for name, text, _, _ in long_runs:
        history = histories.get(name)
        if not history:
            continue
        time = history["time"]
        entry = {"t_last": float(time[-1]), "at_times": {}}
        for instant in LONG_RUN_INSTANTS:
            if instant > time[-1] + 0.3:
                continue
            index = int(np.argmin(np.abs(time - instant)))
            entry["at_times"][f"{instant:g}"] = {
                "time": float(time[index]),
                "density_mean_minus_rest": float(history["density_mean"][index] - normal_default.REST_DENSITY),
                "pressure_mean": float(history["pressure_mean"][index]),
                "pressure_negative_fraction": float(history["pressure_negative_fraction"][index])}
            rows.append([text, f"{time[index]:.2f}",
                         f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.5f}",
                         f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²",
                         f"{100.0 * history['pressure_negative_fraction'][index]:.3f} %",
                         f"{history['speed_maximum'][index]:.3f}", f"{int(history['alive'][index]):,}"])
        early = (time >= 1.0) & (time <= 5.011)
        last = time >= time[-1] - 20.0
        crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
        entry.update({
            "alive_constant": bool((history["alive"] == history["alive"][0]).all()),
            "speed_maximum": float(history["speed_maximum"].max()),
            "density_mean_rate_1_to_5": float(np.polyfit(time[early], history["density_mean"][early], 1)[0]),
            "density_mean_rate_last_20": float(np.polyfit(time[last], history["density_mean"][last], 1)[0])
            if time[-1] >= 40.0 else None,
            "pressure_negative_fraction_maximum_after_5": float(
                history["pressure_negative_fraction"][time >= 5.0].max()) if time[-1] > 5.0 else None,
            "crossing_status": crossing})
        summary["long_runs"][name] = entry
        summary_rows.append([
            text, f"t = {time[-1]:.2f}",
            f"{int(history['alive'][0]):,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
            f"{entry['speed_maximum']:.3f}", f"{entry['density_mean_rate_1_to_5']:+.5f}",
            "—" if entry["density_mean_rate_last_20"] is None else f"{entry['density_mean_rate_last_20']:+.6f}",
            "—" if entry["pressure_negative_fraction_maximum_after_5"] is None
            else f"{100.0 * entry['pressure_negative_fraction_maximum_after_5']:.3f} %",
            "—" if not crossing else str(crossing["gradient_fallbacks"]),
            "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
            else f"{crossing['wall_distance_minimum_over_spacing']:.3f}",
            "—" if not crossing or crossing.get("below_floor_maximum") is None
            else f"{crossing['below_floor_maximum']} / {crossing['outside_box_maximum']}",
            record_text(records_by_id.get(STOPPED_RECORD, {}) if name == STOPPED_RUN
                        else records_by_name.get(name, {}))])
    if rows:
        write_text("pairwise_long_runs.md", markdown_table(
            ["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "负压粒子比例", "流体最大速度", "粒子数（含壁）"], rows))
        write_text("pairwise_long_runs_summary.md", markdown_table(
            ["运行", "到达", "粒子数（含壁）", "流体最大速度（整个运行）", "平均密度变化率，t ∈ [1, 5]", "同，最后 20 个时间单位",
             "负压粒子比例的最大值（t ≥ 5）", "A_f 回退（累计）", "φ_min / dx", "低于下限 / 越出盒（最大粒子数）",
             "记录（提交）"], summary_rows))
        setup_style()
        figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
        panels = (("density_mean", lambda values: values - normal_default.REST_DENSITY, "流体平均密度 − ρ0（kg/m³）"),
                  ("pressure_mean", lambda values: values / normal_default.DYNAMIC_PRESSURE, "流体平均压力 / (ρ0 U²)"),
                  ("pressure_negative_fraction", lambda values: 100.0 * values, "负压粒子比例（%），t ≥ 1"))
        handles, labels = [], []
        for axis, (key, transform, title) in zip(axes, panels):
            for name, text, colour, style in long_runs:
                history = histories.get(name)
                if not history:
                    continue
                shown = history["time"] >= (1.0 if key == "pressure_negative_fraction" else 0.0)
                line, = axis.plot(history["time"][shown], transform(history[key][shown]), color=colour,
                                  linestyle=style, zorder=3, linewidth=1.6)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            if key != "pressure_negative_fraction":
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            axis.set_xlim(0, 100)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔 250²：统计运行全程的密度与压力", x=0.01, y=1.0 - 0.05 / height,
                        ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=min(len(handles), 3), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        rows_of_legend = (len(handles) + 2) // 3
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - (0.40 + 0.26 * rows_of_legend) / height), w_pad=2.6)
        (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
        figure.savefig(REPORT_ROOT / "figures" / "pairwise_long_runs.png")
        plt.close(figure)

    # ---------------------------------------------------------------- 4.6 first fluid layer at the four walls
    layer_runs = [(f"n250_{cavity_configuration}", CONFIGURATION_TEXT[cavity_configuration])] \
        if cavity_configuration else []
    layer_runs.append(("n250_crossing_normal", "一阶（现缺省）"))
    layer_data = {}
    for name, text in layer_runs:
        directory = LOG_ROOT / "cavity" / name
        if not (directory / "meta.json").exists() or not (directory / "snapshots").is_dir():
            continue
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        spacing, support, timestep = float(meta["spacing"]), float(meta["support_radius"]), float(meta["dt"])
        snapshots = [(int(path.stem[1:]) * timestep, path) for path in sorted((directory / "snapshots").glob("t*.npz"))]
        snapshots = [(instant, path) for instant, path in snapshots if 89.5 <= instant <= 100.5]
        if not snapshots:
            continue
        per_wall = {wall_text: {"first": [], "second": [], "counts": []} for wall_text, _, _ in WALLS}
        for _, path in snapshots:
            data = np.load(path)
            fluid = np.isin(data["material"], meta["fluid_groups"])
            positions = data["positions"][fluid].astype(np.float64)
            for wall_text, axis_index, side in WALLS:
                distance = (HALF_BOX - side * positions[:, axis_index]) / spacing
                along = np.abs(positions[:, 1 - axis_index]) <= HALF_BOX - support
                selected = distance[along]
                per_wall[wall_text]["first"].append(selected[selected < 1.0])
                per_wall[wall_text]["second"].append(selected[(selected >= 1.0) & (selected < 2.0)])
                per_wall[wall_text]["counts"].append(int((selected < 1.0).sum()))
        lattice_count = 2.0 * (HALF_BOX - support) / spacing
        layer_data[name] = {"text": text, "snapshots": len(snapshots), "times": [snapshots[0][0], snapshots[-1][0]],
                            "walls": {}}
        for wall_text, _, _ in WALLS:
            first = np.concatenate(per_wall[wall_text]["first"])
            second = np.concatenate(per_wall[wall_text]["second"])
            layer_data[name]["walls"][wall_text] = {
                "count_per_snapshot": float(np.mean(per_wall[wall_text]["counts"])),
                "count_over_lattice": float(np.mean(per_wall[wall_text]["counts"]) / lattice_count),
                "mean": float(np.abs(first).mean()), "standard_deviation": float(first.std()),
                "percentiles_5_50_95": [float(value) for value in np.percentile(first, (5, 50, 95))],
                "minimum": float(first.min()), "second_layer_mean": float(second.mean())}
    if layer_data:
        rows = []
        for wall_text, _, _ in WALLS:
            for name, text in layer_runs:
                if name not in layer_data:
                    continue
                entry = layer_data[name]["walls"][wall_text]
                low, median, high = entry["percentiles_5_50_95"]
                rows.append([wall_text, text,
                             f"{entry['count_per_snapshot']:.1f}（{100 * entry['count_over_lattice']:.1f} %）",
                             f"{entry['mean']:.3f}", f"{entry['standard_deviation']:.3f}",
                             f"{low:.3f} / {median:.3f} / {high:.3f}", f"{entry['minimum']:.3f}",
                             f"{entry['second_layer_mean']:.3f}"])
        first_entry = next(iter(layer_data.values()))
        write_text("pairwise_wall_layer.md",
                   f"250² 方腔，t ≈ {first_entry['times'][0]:.0f} 到 {first_entry['times'][1]:.0f} 的 "
                   f"{first_entry['snapshots']} 个快照合在一起：流体粒子到各壁面（crossing 的盒面，±0.502）的距离，以粒子间距 "
                   "dx 为单位；只取离另外两面壁超过核半径 h 的那一段（每面壁中间 0.964 的长度，点阵上是 241 个粒子）。"
                   "第一层 = 距离小于 1 dx 的粒子；点阵上第一层在 0.5 dx，第二层在 1.5 dx。\n\n"
                   + markdown_table(["壁", "运行", "第一层的粒子数（每个快照；相对点阵）", "第一层到壁面距离的平均（绝对值，dx）",
                                     "标准差", "5 % / 中位 / 95 % 分位", "最小", "第二层（1 到 2 dx）的平均"], rows))
        summary["wall_layer"] = layer_data

    # ---------------------------------------------------------------- the stopped long run: how noisy it is
    # per quantity: the rms over t in [5, 18.06] of (a sample minus the mean of its two neighbours), samples
    # every 1000 steps (0.03 time units)
    fluctuation_rows = []
    summary["fluctuations"] = {}
    stopped, reference_run = histories.get(STOPPED_RUN), histories.get("n250_crossing_normal")
    if stopped and reference_run:
        window_end = float(stopped["time"][-1])
        quantities = (("density_mean", "平均密度（kg/m³）", 1.0), ("pressure_mean", "平均压力（ρ0U²）",
                                                                   normal_default.DYNAMIC_PRESSURE),
                      ("pressure_minimum", "最小压力（ρ0U²）", normal_default.DYNAMIC_PRESSURE),
                      ("pressure_maximum", "最大压力（ρ0U²）", normal_default.DYNAMIC_PRESSURE),
                      ("density_maximum", "最大密度（kg/m³）", 1.0), ("speed_maximum", "最大速度（盖速）", 1.0))
        for key, key_text, scale in quantities:
            cells = [key_text]
            for name, history in ((STOPPED_RUN, stopped), ("n250_crossing_normal", reference_run)):
                inside = (history["time"] >= 5.0) & (history["time"] <= window_end)
                values = history[key][inside] / scale
                high_pass = values[1:-1] - 0.5 * (values[:-2] + values[2:])
                offset = normal_default.REST_DENSITY if key.startswith("density") else 0.0
                entry = {"mean": float(values.mean() - offset),
                         "fluctuation": float(np.sqrt((high_pass ** 2).mean()))}
                summary["fluctuations"].setdefault(name, {})[key] = entry
                cells += [f"{entry['mean']:+.4f}", f"{entry['fluctuation']:.4f}"]
            first_order = summary["fluctuations"]["n250_crossing_normal"][key]["fluctuation"]
            cells.append(f"{summary['fluctuations'][STOPPED_RUN][key]['fluctuation'] / first_order:.1f}")
            fluctuation_rows.append(cells)
        write_text("pairwise_fluctuations.md",
                   f"250² 方腔的两次统计运行，t 从 5 到 {window_end:.2f}（反射版停下的时刻），每 7000 步（0.21 个时间单位）一个样本。"
                   "“涨落” = 一个样本减去前后两个样本平均值之后的 rms；密度的平均值已减去 ρ0。\n\n"
                   + markdown_table(["量", "反射版：平均", "反射版：涨落", "一阶（现缺省）：平均", "一阶：涨落", "涨落之比"],
                                    fluctuation_rows))

    # ---------------------------------------------------------------- the first fluid layer at t = 5
    layer_rows = []
    summary["wall_layer_at_5"] = {}
    for name, text, _ in SCREENING_RUNS:
        final = LOG_ROOT / "box_run" / name / "final.npz"
        box_summary = LOG_ROOT / "box_run" / name / "summary.json"
        if not final.exists() or not box_summary.exists():
            continue
        box = json.loads(box_summary.read_text(encoding="utf-8"))
        spacing = float(box["spacing"])
        positions = np.load(final)["positions"].astype(np.float64)
        support = 5.0 * spacing
        cells, entry = [text], {}
        for wall_text, axis_index, side in WALLS:
            distance = (HALF_BOX - side * positions[:, axis_index]) / spacing
            along = np.abs(positions[:, 1 - axis_index]) <= HALF_BOX - support
            first = distance[along & (distance < 1.0)]
            entry[wall_text] = {"mean": float(np.abs(first).mean()), "count": int(first.size)}
            cells.append(f"{entry[wall_text]['mean']:.3f}（{first.size}）")
        summary["wall_layer_at_5"][name] = entry
        layer_rows.append(cells)
    if layer_rows:
        write_text("pairwise_wall_layer_at_5.md",
                   "250² 方腔的筛查短跑，各自 t = 5.01 的终态（一个快照）：四面壁各自贴壁第一层流体粒子（距离小于 1 dx，"
                   "只取离另外两面壁超过核半径 h 的那一段）到壁面距离的平均，单位 dx；括号内是粒子数（点阵上是 241）。\n\n"
                   + markdown_table(["运行", "左壁", "右壁", "底壁", "盖（上壁）"], layer_rows))

    out = REPORT_ROOT / "pairwise_reflection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                      encoding="utf-8", newline="\n")
    print(f"[pairwise_reflection] tables written; screening: "
          + "; ".join(f"{name.replace('budget_n250_crossing_', '')} interior {entry['interior_continuity']:+.5f} "
                      f"({'PASS' if entry['passed'] else 'FAIL' if entry['passed'] is not None else 'reference'})"
                      for name, entry in summary["screening"].items())
          + f"; cavity configuration {cavity_configuration}")
    return 0


# ============================================================================================ runs to t = 100
T100_RUNS = (("n250_reflection", "反射版（从 t = 10.08 的检查点续跑）"),
             ("n250_pairwise_full", "逐对几何 + 完整向量"))
EARLY_INSTANT = 18.06               # the instant the first reflection run was stopped at
EARLY_SPAN = 2.0                    # the samples of (EARLY_INSTANT - EARLY_SPAN, EARLY_INSTANT] are averaged
HISTORY_INSTANTS = (1.05, 5.04, 10.08, 18.06, 30.03, 39.9, 60.06, 80.01, 90.09, 99.96)
NOISE_BIN = 10.0


def run_samples(name: str) -> list:
    """The rows of samples.jsonl of a cavity run (one per step; a resumed run's superseded rows are gone)."""
    from experiment.wall_crossing.run_log import LOG_ROOT
    path = LOG_ROOT / "cavity" / name / "samples.jsonl"
    if not path.exists():
        return []
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def centre_line_state(name: str, first: float, last: float) -> dict | None:
    """What the stored samples of a run say about the flow over first < t <= last: the fluid's kinetic energy
    (0.5 sum m |v|^2, as the runner records it) and the centre-line profiles (MLS, 1001 points), averaged over
    the samples of the window. From the u profile on the vertical centre line the stream function on that line,
    psi(y) = integral of u from the bottom, and its minimum, divided by 1 + 2 dx (the "wall frame" of the
    tables). That is an ESTIMATE of psi_min: the vortex centre lies beside the line. None without samples."""
    from experiment.wall_crossing.run_log import LOG_ROOT
    directory = LOG_ROOT / "cavity" / name
    rows = [row for row in run_samples(name) if first < row["time"] <= last + 1e-9]
    if not rows:
        return None
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    points = np.load(directory / "points.npz")
    height, width = points["u_dense"][:, 1], points["v_dense"][:, 0]
    u_profiles, v_profiles = [], []
    for row in rows:
        with np.load(directory / "samples" / f"s{row['step']:010d}.npz") as archive:
            u_profiles.append(archive["u_dense_mls"])
            v_profiles.append(archive["v_dense_mls"])
    u_profile, v_profile = np.mean(u_profiles, axis=0), np.mean(v_profiles, axis=0)
    stream = np.concatenate([[0.0], np.cumsum(0.5 * (u_profile[1:] + u_profile[:-1]) * np.diff(height))])
    length = 1.0 + 2.0 * float(meta["spacing"])
    energies = np.array([row["kinetic_energy"] for row in rows])
    return {"samples": len(rows), "times": [float(rows[0]["time"]), float(rows[-1]["time"])],
            "kinetic_energy_mean": float(energies.mean()), "kinetic_energy_last": float(energies[-1]),
            "psi_centre_line_wall": float(stream.min() / length),
            "y_at_psi": float(height[int(np.argmin(stream))] + 0.5),
            "u_min": float(u_profile.min()), "v_max": float(v_profile.max()), "v_min": float(v_profile.min())}


def long_run_tables() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.validation import cavity_reference
    from experiment.wall_crossing import normal_default
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, run_records
    from experiment.wall_crossing.run_log import LOG_ROOT

    write_text = normal_default.write_table
    record_text = normal_default.record_text
    records = run_records()
    records_by_name = {}
    for record in records:
        if record.get("name"):
            records_by_name[record["name"]] = record          # the newest record of a name (a resumed run)
    first_colour, old_colour = normal_default.NEW_COLOUR, normal_default.OLD_COLOUR
    colours = {"n250_reflection": (INK_PRIMARY, "solid"), "n250_pairwise_full": (INK_PRIMARY, (0, (4, 2))),
               "n250_crossing_normal": (first_colour, "solid"), "n250_crossing": (old_colour, "solid")}
    summary = {}
    tables_path = REPORT_ROOT / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}
    reference_psi = float(cavity_reference.marchi2021()["extrema"]["psi_min"][0])

    # ---------------------------------------------------------------- the cavity metrics beside the existing rows
    if all(analysed.get(name, {}).get("complete") for name, _ in T100_RUNS):
        cavity = normal_default.cavity(
            modes=(("reflection", "crossing，反射版", INK_PRIMARY, "P"),
                   ("pairwise_full", "crossing，逐对几何 + 完整向量", INK_SECONDARY, "X"),
                   ("crossing_normal", "crossing，一阶（现缺省）", first_colour, "o"),
                   ("crossing", "crossing，旧缺省（一阶，完整向量）", old_colour, "s"),
                   ("simple", "simple", normal_default.SIMPLE_COLOUR, "^"),
                   ("adami", "adami", normal_default.ADAMI_COLOUR, "D")),
            extra_runs=(("n250_quadratic_viscous", "crossing，二阶粘性", "crossing_normal"),
                        ("n500_quadratic_viscous", "crossing，二阶粘性", "crossing_normal"),
                        ("n250_crossing_r2", "crossing，旧缺省，第二次运行", "crossing")),
            prefix="pairwise_t100", required_mode="reflection", labelled_modes=("reflection", "pairwise_full"),
            labels_beside=True,
            figure_title="Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧；数字 = 反射版和逐对几何 + 完整向量）")
        if cavity:
            summary["cavity"] = cavity

    # ---------------------------------------------------------------- the interior continuity term (reported only)
    # from the budget short runs (250^2, t <= 5): the two configurations of the runs to t = 100 beside the
    # first order and the old default
    from experiment.wall_crossing import density_budget
    rows = []
    summary["continuity"] = {}
    for name, text in (("budget_n250_crossing_reflection", "反射版"),
                       ("budget_n250_crossing_pairwise_full", "逐对几何 + 完整向量"),
                       ("budget_n250_crossing_pairwise", "只有逐对几何（第 12 节）"),
                       ("budget_n250_crossing_normal_r1", "一阶（现缺省），第 1 次"),
                       ("budget_n250_crossing_normal_r2", "一阶（现缺省），第 2 次"),
                       ("budget_n250_crossing", "旧缺省（完整向量）")):
        if not (LOG_ROOT / "box_run" / name / "summary.json").exists():
            continue
        result = density_budget.analyse(name)
        groups = result["groups"]
        entry = {"interior_continuity": groups["interior"]["rate"][density_budget.ROW_DRIFT],
                 "interior_continuity_error": groups["interior"]["standard_error"][density_budget.ROW_DRIFT],
                 "terms_all": [float(value) for value in groups["all"]["rate"]],
                 "density_mean_fit_slope": result["closure"]["density_mean_fit_slope"]}
        summary["continuity"][name] = entry
        everything = groups["all"]["rate"]
        rows.append([text, f"{entry['interior_continuity']:+.5f} ± {entry['interior_continuity_error']:.5f}",
                     f"{everything[0]:+.5f} / {everything[1]:+.5f}",
                     f"{everything[density_budget.ROW_DIFFUSION]:+.5f}",
                     f"{everything[density_budget.ROW_SUM]:+.5f}", f"{everything[density_budget.ROW_ROUNDING]:+.5f}",
                     f"{everything[density_budget.ROW_APPLIED]:+.5f}",
                     record_text(records_by_name.get(name, {}))])
    if rows:
        write_text("pairwise_t100_continuity.md",
                   "收支短跑（250²，t = 0 到 5.01，收支构建），t ∈ [1, 5] 的时间平均，kg/m³ 每时间单位。\n\n"
                   + markdown_table(["配置", "内部的连续性项", "连续性项全场：流体邻居 / 壁邻居", "δ 项全场合计", "四项之和",
                                     "float32 存储舍入", "实际写入", "记录（提交）"], rows))

    # ---------------------------------------------------------------- how the runs went (the stop conditions)
    rows = []
    summary["runs"] = {}
    for name, text in T100_RUNS + (("n250_crossing_normal", "一阶（现缺省）"),):
        directory = LOG_ROOT / "cavity" / name
        history = normal_default.load_field_history(name)
        if not history:
            continue
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
        stop_path = directory / "watch_stop.json"
        entry = {
            "t_last": float(history["time"][-1]), "complete": bool(result),
            "alive_constant": bool((history["alive"] == history["alive"][0]).all()),
            "speed_maximum": float(history["speed_maximum"].max()),
            "watch_stop": json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None,
            "crossing_status": crossing,
            "invariant_segments": len(result.get("segments_with_invariant_violation", [])) if result else None}
        summary["runs"][name] = entry
        rows.append([
            text, f"t = {entry['t_last']:.2f}" + ("" if entry["complete"] else "（没有跑完）"),
            f"{int(history['alive'][0]):,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
            "—" if not crossing or crossing.get("outside_box_maximum") is None else str(crossing["outside_box_maximum"]),
            f"{entry['speed_maximum']:.3f}",
            "没有触发" if entry["watch_stop"] is None else f"**停止：{entry['watch_stop']['reason']}**",
            "—" if not crossing else str(crossing["gradient_fallbacks"]),
            "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
            else f"{crossing['wall_distance_minimum_over_spacing']:.3f}；{crossing['below_floor_maximum']}",
            record_text(records_by_name.get(name, {}))])
    if rows:
        write_text("pairwise_t100_runs.md", markdown_table(
            ["运行", "到达", "粒子数（含壁）", "越出盒的流体粒子（各次位置读回的最大值）", "流体最大速度（整个运行；线 1.05）",
             "停止条件", "A_f 回退（累计）", "φ_min / dx；低于距离下限的粒子（最大）", "记录（提交）"], rows))

    # ---------------------------------------------------------------- histories: density, pressure, noise
    history_runs = T100_RUNS + (("n250_crossing_normal", "一阶（现缺省）"),)
    histories = {name: normal_default.load_field_history(name) for name, _ in history_runs}
    histories = {name: history for name, history in histories.items() if history and len(history["time"]) > 5}
    rows, noise_rows = [], []
    summary["histories"], summary["noise"] = {}, {}
    edges = np.arange(0.0, 100.0 + 1e-9, NOISE_BIN)
    for name, text in history_runs:
        history = histories.get(name)
        if not history:
            continue
        time = history["time"]
        summary["histories"][name] = {}
        for instant in HISTORY_INSTANTS:
            if instant > time[-1] + 0.3:
                continue
            index = int(np.argmin(np.abs(time - instant)))
            summary["histories"][name][f"{instant:g}"] = {
                "time": float(time[index]),
                "density_mean_minus_rest": float(history["density_mean"][index] - normal_default.REST_DENSITY),
                "pressure_mean": float(history["pressure_mean"][index]),
                "pressure_negative_fraction": float(history["pressure_negative_fraction"][index])}
            rows.append([text, f"{time[index]:.2f}",
                         f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.5f}",
                         f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²",
                         f"{100.0 * history['pressure_negative_fraction'][index]:.3f} %",
                         f"{history['speed_maximum'][index]:.3f}"])
        # the noise of the mean pressure: per bin of NOISE_BIN time units, the rms of (a sample minus the mean of
        # its two neighbours), samples every 0.21 time units
        pressure = history["pressure_mean"] / normal_default.DYNAMIC_PRESSURE
        high_pass = pressure[1:-1] - 0.5 * (pressure[:-2] + pressure[2:])
        centre_time = time[1:-1]
        cells, values = [text], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (centre_time >= low) & (centre_time < high)
            value = float(np.sqrt((high_pass[inside] ** 2).mean())) if inside.sum() >= 5 else None
            values.append(value)
            cells.append("—" if value is None else f"{value:.4f}")
        late = (centre_time >= 80.0)
        cells.append(f"{100.0 * history['pressure_negative_fraction'][time >= 5.0].max():.3f} %")
        summary["noise"][name] = {"bins": values, "edges": edges.tolist(),
                                  "late": float(np.sqrt((high_pass[late] ** 2).mean())) if late.sum() >= 5 else None,
                                  "density_mean_rate_last_20": float(np.polyfit(
                                      time[time >= time[-1] - 20.0],
                                      history["density_mean"][time >= time[-1] - 20.0], 1)[0])}
        noise_rows.append(cells)
    if rows:
        write_text("pairwise_t100_history.md", markdown_table(
            ["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "负压粒子比例", "流体最大速度"], rows))
        write_text("pairwise_t100_noise.md",
                   "平均压力的逐样本涨落（ρ0U²）：一个样本减去前后两个样本平均值之后的 rms，样本间隔 0.21 个时间单位，"
                   f"按每 {NOISE_BIN:g} 个时间单位一段给出。最后一列是 t ≥ 5 里负压粒子比例的最大值。\n\n"
                   + markdown_table(["运行"] + [f"t {low:g}–{high:g}" for low, high in zip(edges[:-1], edges[1:])]
                                    + ["负压粒子比例的最大值（t ≥ 5）"], noise_rows))
        setup_style()
        figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
        panels = (("density_mean", "流体平均密度 − ρ0（kg/m³）"), ("pressure_mean", "流体平均压力 / (ρ0 U²)"),
                  ("noise", f"平均压力的逐样本涨落 / (ρ0 U²)，每 {NOISE_BIN:g} 个时间单位一段"),
                  ("pressure_negative_fraction", "负压粒子比例（%），t ≥ 1"))
        handles, labels = [], []
        for axis, (key, title) in zip(axes, panels):
            for name, text in history_runs:
                history = histories.get(name)
                if not history:
                    continue
                colour, style = colours[name]
                if key == "noise":
                    values = summary["noise"][name]["bins"]
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    known = [index for index, value in enumerate(values) if value is not None]
                    axis.plot(centres[known], [values[index] for index in known], color=colour, linestyle=style,
                              marker="o", markersize=4, zorder=3, linewidth=1.6)
                    continue
                shown = history["time"] >= (1.0 if key == "pressure_negative_fraction" else 0.0)
                values = history[key][shown]
                values = {"density_mean": values - normal_default.REST_DENSITY,
                          "pressure_mean": values / normal_default.DYNAMIC_PRESSURE,
                          "pressure_negative_fraction": 100.0 * values}[key]
                line, = axis.plot(history["time"][shown], values, color=colour, linestyle=style, zorder=3,
                                  linewidth=1.6)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            if key in ("density_mean", "pressure_mean"):
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            if key == "noise":
                axis.set_ylim(bottom=0.0)
            axis.set_xlim(0, 100)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔 250²：两次到 t = 100 的运行与一阶，密度、压力和压力噪声的历史", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
        figure.savefig(REPORT_ROOT / "figures" / "pairwise_t100_history.png")
        plt.close(figure)

    # ---------------------------------------------------------------- kinetic energy over the run, every mode
    energy_runs = T100_RUNS + (("n250_crossing_normal", "一阶（现缺省）"), ("n250_crossing", "旧缺省（完整向量）"),
                               ("n250_quadratic_viscous", "二阶粘性"), ("n250_simple", "simple"),
                               ("n250_adami", "adami"))
    energy_instants = (5.04, 10.08, 18.06, 30.03, 39.9, 60.06, 80.01, 99.96)
    rows = []
    summary["kinetic_energy"] = {}
    for name, text in energy_runs:
        samples = run_samples(name)
        if not samples:
            continue
        times = np.array([row["time"] for row in samples])
        energies = np.array([row["kinetic_energy"] for row in samples])
        cells, entry = [text], {}
        for instant in energy_instants:
            if instant > times[-1] + 0.3:
                cells.append("—")
                continue
            index = int(np.argmin(np.abs(times - instant)))
            entry[f"{instant:g}"] = float(energies[index])
            cells.append(f"{energies[index]:.3f}")
        summary["kinetic_energy"][name] = entry
        rows.append(cells)
    if rows:
        write_text("pairwise_t100_kinetic_energy.md",
                   "流体的动能 ½ Σ m |v|²（运行器在每个样本记录的值），250²，各时刻最近的那个样本。\n\n"
                   + markdown_table(["运行"] + [f"t = {instant:g}" for instant in energy_instants], rows))

    # ---------------------------------------------------------------- the runs at t = 18
    early_runs = (("n250_crossing_normal", "一阶（现缺省）"), ("n250_crossing", "旧缺省（完整向量）"),
                  (STOPPED_RUN, "反射版，第一次（停在 t = 18.06 的那次）"),
                  ("n250_reflection", "反射版，续跑（从 t = 10.08 的检查点）"),
                  ("n250_pairwise_full", "逐对几何 + 完整向量"), ("n250_quadratic_viscous", "二阶粘性"),
                  ("n250_simple", "simple"), ("n250_adami", "adami"))
    rows, check_rows = [], []
    summary["at_18"], summary["estimate_check"] = {}, {}
    for name, text in early_runs:
        state = centre_line_state(name, EARLY_INSTANT - EARLY_SPAN, EARLY_INSTANT)
        if state is None or state["times"][1] < EARLY_INSTANT - 0.3:
            continue
        summary["at_18"][name] = state
        deviation = 100.0 * (abs(state["psi_centre_line_wall"]) / abs(reference_psi) - 1.0)
        rows.append([text, f"{state['kinetic_energy_last']:.3f}", f"{state['kinetic_energy_mean']:.3f}",
                     f"{state['psi_centre_line_wall']:.5f}（{deviation:+.2f} %）", f"{state['y_at_psi']:.3f}",
                     f"{state['u_min']:.4f}", f"{state['v_max']:.4f}", f"{state['v_min']:.4f}"])
        late = centre_line_state(name, 80.0, 100.0)
        true = (analysed.get(name, {}).get("extrema") or {}).get("psi_min")
        if late and true and analysed[name].get("complete"):
            ratio = late["psi_centre_line_wall"] / true["value_wall"]
            summary["estimate_check"][name] = {"estimate": late["psi_centre_line_wall"], "true": true["value_wall"],
                                               "ratio": ratio}
            check_rows.append([text, f"{late['psi_centre_line_wall']:.5f}", f"{true['value_wall']:.5f}",
                               f"{ratio:.4f}"])
    if rows:
        write_text("pairwise_t100_at_18.md",
                   f"250²，t = {EARLY_INSTANT:g}。动能是运行器记录的流体动能；“窗口平均”和剖面量是 t ∈ "
                   f"({EARLY_INSTANT - EARLY_SPAN:g}, {EARLY_INSTANT:g}] 的 {int(EARLY_SPAN / 0.21) + 1} 个样本的平均。"
                   "ψ 的估计 = 竖直中心线上 u 剖面从底向上积分的最小值，除以 1 + 2 dx（wall 帧）；括号内是相对 Marchi 2021 "
                   f"定常值 {reference_psi:.5f} 的偏差（流动在 t = 18 还没有定常）。\n\n"
                   + markdown_table(["运行", f"动能，t = {EARLY_INSTANT:g}", "动能，窗口平均", "ψ_min 的中心线估计（wall 帧）",
                                     "估计的最小值所在高度 y", "中心线上的 u_min", "v_max", "v_min"], rows))
    if check_rows:
        write_text("pairwise_t100_estimate_check.md",
                   "中心线估计与真正的 ψ_min 的关系，在两者都有的地方（t = 80 到 100 的窗口；真值由粒子快照在整个腔上求）。\n\n"
                   + markdown_table(["运行", "中心线估计（wall 帧）", "ψ_min（wall 帧，快照）", "估计 ÷ 真值"], check_rows))

    # ---------------------------------------------------------------- the first fluid layer, t = 90 to 100
    layer_runs = T100_RUNS + (("n250_crossing_normal", "一阶（现缺省）"), ("n250_crossing", "旧缺省（完整向量）"))
    rows = []
    summary["wall_layer"] = {}
    for wall_text, axis_index, side in WALLS:
        for name, text in layer_runs:
            directory = LOG_ROOT / "cavity" / name
            if not (directory / "meta.json").exists() or not (directory / "snapshots").is_dir():
                continue
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            spacing, support, timestep = float(meta["spacing"]), float(meta["support_radius"]), float(meta["dt"])
            snapshots = [path for path in sorted((directory / "snapshots").glob("t*.npz"))
                         if 89.5 <= int(path.stem[1:]) * timestep <= 100.5]
            if not snapshots:
                continue
            first, second, counts = [], [], []
            for path in snapshots:
                data = np.load(path)
                positions = data["positions"][np.isin(data["material"], meta["fluid_groups"])].astype(np.float64)
                distance = (HALF_BOX - side * positions[:, axis_index]) / spacing
                selected = distance[np.abs(positions[:, 1 - axis_index]) <= HALF_BOX - support]
                first.append(selected[selected < 1.0])
                second.append(selected[(selected >= 1.0) & (selected < 2.0)])
                counts.append(int((selected < 1.0).sum()))
            first, second = np.concatenate(first), np.concatenate(second)
            entry = {"mean": float(np.abs(first).mean()), "standard_deviation": float(first.std()),
                     "count_per_snapshot": float(np.mean(counts)), "minimum": float(first.min()),
                     "second_layer_mean": float(second.mean()), "snapshots": len(snapshots)}
            summary["wall_layer"].setdefault(name, {})[wall_text] = entry
            rows.append([wall_text, text, f"{entry['count_per_snapshot']:.1f}", f"{entry['mean']:.3f}",
                         f"{entry['standard_deviation']:.3f}", f"{entry['minimum']:.3f}",
                         f"{entry['second_layer_mean']:.3f}"])
    if rows:
        write_text("pairwise_t100_wall_layer.md",
                   "250² 方腔，t ≈ 90 到 100 的 11 个快照合在一起：流体粒子到各壁面（盒面，±0.502）的距离，单位 dx；只取离"
                   "另外两面壁超过核半径 h 的那一段（点阵上是 241 个粒子）。第一层 = 距离小于 1 dx 的粒子。\n\n"
                   + markdown_table(["壁", "运行", "第一层的粒子数（每个快照）", "第一层到壁面距离的平均（绝对值，dx）", "标准差",
                                     "最小", "第二层（1 到 2 dx）的平均"], rows))

    out = REPORT_ROOT / "pairwise_reflection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "long_runs.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                        encoding="utf-8", newline="\n")
    print("[pairwise_reflection] long-run tables written; runs: "
          + "; ".join(f"{name} t = {entry['t_last']:.2f}" + (" complete" if entry["complete"] else "")
                      for name, entry in summary["runs"].items()))
    return 0


# ====================================================================================== the complete scheme
FULL_RUNS = (("n250_reflection_full", "C1：完整方案，初始密度 ρ0"),
             ("n250_reflection_full_rho_plus_0p3", "C2：完整方案，初始密度 ρ0 + 0.3"))
FULL_CONTEXT = (("n250_pairwise_full", "B：逐对几何 + 完整向量"), ("n250_reflection", "A：反射版（只取法向）"),
                ("n250_crossing_normal", "一阶（现缺省）"))
DENSITY_INSTANTS = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
RATE_WINDOW = (60.0, 100.0)
SNAPSHOT_RUNS = (("n250_crossing", "旧缺省（完整向量）"), ("n250_crossing_r2", "旧缺省，第二次运行"),
                 ("n250_crossing_normal", "一阶（现缺省）"), ("n250_reflection", "A：反射版（只取法向）"),
                 ("n250_pairwise_full", "B：逐对几何 + 完整向量"),
                 ("n250_reflection_full", "C1：完整方案，初始密度 ρ0"),
                 ("n250_reflection_full_rho_plus_0p3", "C2：完整方案，初始密度 ρ0 + 0.3"))
SHORT_RUNS = (("budget_n250_crossing", "旧缺省（完整向量）"), ("budget_n250_crossing_normal_r1", "一阶（现缺省），第 1 次"),
              ("budget_n250_crossing_normal_r2", "一阶（现缺省），第 2 次"),
              ("budget_n250_crossing_pairwise_full", "B：逐对几何 + 完整向量"),
              ("budget_n250_crossing_reflection", "A：反射版（只取法向）"))


def high_pass_fluctuation(values: np.ndarray) -> float | None:
    """The rms of (a sample minus the mean of its two neighbours); None with fewer than five samples."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 5:
        return None
    return float(np.sqrt(((values[1:-1] - 0.5 * (values[:-2] + values[2:])) ** 2).mean()))


def full_scheme_tables() -> int:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing import field_difference, normal_default
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, run_records
    from experiment.wall_crossing.run_log import LOG_ROOT

    write_text = normal_default.write_table
    record_text = normal_default.record_text
    records_by_name = {}
    for record in run_records():
        if record.get("name"):
            records_by_name[record["name"]] = record
    first_colour, old_colour = normal_default.NEW_COLOUR, normal_default.OLD_COLOUR
    summary = {}
    tables_path = REPORT_ROOT / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}

    # ---------------------------------------------------------------- the cavity metrics beside the existing rows
    if all(analysed.get(name, {}).get("complete") for name, _ in FULL_RUNS):
        cavity = normal_default.cavity(
            modes=(("reflection_full", "crossing，完整方案（C1）", INK_PRIMARY, "P"),
                   ("crossing_normal", "crossing，一阶（现缺省）", first_colour, "o"),
                   ("crossing", "crossing，旧缺省（一阶，完整向量）", old_colour, "s"),
                   ("simple", "simple", normal_default.SIMPLE_COLOUR, "^"),
                   ("adami", "adami", normal_default.ADAMI_COLOUR, "D")),
            extra_runs=(("n250_reflection_full_rho_plus_0p3", "crossing，完整方案，初始密度 ρ0 + 0.3（C2）",
                         "reflection_full"),
                        ("n250_pairwise_full", "crossing，逐对几何 + 完整向量（B）", "reflection_full"),
                        ("n250_reflection", "crossing，反射版，只取法向（A）", "reflection_full"),
                        ("n250_quadratic_viscous", "crossing，二阶粘性", "crossing_normal"),
                        ("n500_quadratic_viscous", "crossing，二阶粘性", "crossing_normal"),
                        ("n250_crossing_r2", "crossing，旧缺省，第二次运行", "crossing")),
            prefix="full_scheme", required_mode="reflection_full", labelled_modes=("reflection_full",),
            labels_beside=True,
            figure_title="Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧；数字 = 完整方案 C1）")
        if cavity:
            summary["cavity"] = cavity

    # ---------------------------------------------------------------- how the runs went (the stop conditions)
    rows = []
    summary["runs"] = {}
    for name, text in FULL_RUNS:
        directory = LOG_ROOT / "cavity" / name
        history = normal_default.load_field_history(name)
        if not history:
            continue
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
        stop_path = directory / "watch_stop.json"
        entry = {"t_last": float(history["time"][-1]), "complete": bool(result),
                 "alive_constant": bool((history["alive"] == history["alive"][0]).all()),
                 "speed_maximum": float(history["speed_maximum"].max()),
                 "watch_stop": json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None,
                 "crossing_status": crossing}
        summary["runs"][name] = entry
        rows.append([
            text, f"t = {entry['t_last']:.2f}" + ("" if entry["complete"] else "（没有跑完）"),
            f"{int(history['alive'][0]):,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
            "—" if not crossing or crossing.get("outside_box_maximum") is None else str(crossing["outside_box_maximum"]),
            f"{entry['speed_maximum']:.3f}",
            "没有触发" if entry["watch_stop"] is None else f"**停止：{entry['watch_stop']['reason']}**",
            "—" if not crossing else str(crossing["gradient_fallbacks"]),
            "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
            else f"{crossing['wall_distance_minimum_over_spacing']:.3f}；{crossing['below_floor_maximum']}",
            record_text(records_by_name.get(name, {}))])
    if rows:
        write_text("full_scheme_runs.md", markdown_table(
            ["运行", "到达", "粒子数（含壁）", "越出盒的流体粒子（各次位置读回的最大值）", "流体最大速度（整个运行；线 1.05）",
             "停止条件", "A_f 回退（累计）", "φ_min / dx；低于距离下限的粒子（最大）", "记录（提交）"], rows))

    # ---------------------------------------------------------------- mean density, pressure, negative share
    history_runs = FULL_RUNS + FULL_CONTEXT
    histories = {name: normal_default.load_field_history(name) for name, _ in history_runs}
    histories = {name: history for name, history in histories.items() if history and len(history["time"]) > 5}
    edges = np.arange(0.0, 100.0 + 1e-9, NOISE_BIN)
    density_rows, pressure_rows, negative_rows, noise_rows = [], [], [], []
    summary["histories"] = {}
    for name, text in history_runs:
        history = histories.get(name)
        if not history:
            continue
        time = history["time"]
        entry = {"instants": {}}
        density_cells, pressure_cells, negative_cells = [text], [text], [text]
        for instant in DENSITY_INSTANTS:
            if instant > time[-1] + 0.3:
                density_cells.append("—")
                pressure_cells.append("—")
                negative_cells.append("—")
                continue
            index = int(np.argmin(np.abs(time - instant)))
            entry["instants"][f"{instant:g}"] = {
                "time": float(time[index]),
                "density_mean_minus_rest": float(history["density_mean"][index] - normal_default.REST_DENSITY),
                "pressure_mean": float(history["pressure_mean"][index]),
                "pressure_negative_fraction": float(history["pressure_negative_fraction"][index])}
            density_cells.append(f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.4f}")
            pressure_cells.append(f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.3f}")
            negative_cells.append(f"{100.0 * history['pressure_negative_fraction'][index]:.2f} %")
        inside = (time >= RATE_WINDOW[0]) & (time <= RATE_WINDOW[1] + 0.3)
        slope, intercept = np.polyfit(time[inside], history["density_mean"][inside], 1) if inside.sum() >= 5 \
            else (float("nan"), float("nan"))
        residual = float(np.std(history["density_mean"][inside] - (slope * time[inside] + intercept))) \
            if inside.sum() >= 5 else float("nan")
        pressure_slope = float(np.polyfit(time[inside], history["pressure_mean"][inside], 1)[0]
                               / normal_default.DYNAMIC_PRESSURE) if inside.sum() >= 5 else float("nan")
        entry.update({"density_rate_60_100": float(slope), "density_fit_residual": residual,
                      "pressure_rate_60_100": pressure_slope})
        density_cells += [f"{slope:+.5f}", f"{residual:.4f}"]
        pressure_cells.append(f"{pressure_slope:+.4f}")
        negative_cells.append(f"{100.0 * history['pressure_negative_fraction'][time >= 5.0].max():.2f} %")
        # the fluctuation of the mean pressure, per bin of NOISE_BIN time units and over t >= 80
        pressure = history["pressure_mean"] / normal_default.DYNAMIC_PRESSURE
        high_pass = pressure[1:-1] - 0.5 * (pressure[:-2] + pressure[2:])
        centre_time = time[1:-1]
        cells, bins = [text], []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = (centre_time >= low) & (centre_time < high)
            value = float(np.sqrt((high_pass[chosen] ** 2).mean())) if chosen.sum() >= 5 else None
            bins.append(value)
            cells.append("—" if value is None else f"{value:.4f}")
        late = centre_time >= 80.0
        entry.update({"noise_bins": bins,
                      "noise_late": float(np.sqrt((high_pass[late] ** 2).mean())) if late.sum() >= 5 else None})
        cells.append("—" if entry["noise_late"] is None else f"{entry['noise_late']:.4f}")
        summary["histories"][name] = entry
        density_rows.append(density_cells)
        pressure_rows.append(pressure_cells)
        negative_rows.append(negative_cells)
        noise_rows.append(cells)
    if density_rows:
        instant_header = [f"t = {instant:g}" for instant in DENSITY_INSTANTS]
        write_text("full_scheme_density.md",
                   "流体平均密度 − ρ0（kg/m³），各时刻最近的那个样本（样本间隔 0.21）。变化率 = t ∈ [60, 100] 的线性拟合斜率，"
                   "kg/m³ 每时间单位；最后一列是拟合残差的标准差。\n\n"
                   + markdown_table(["运行"] + instant_header + ["变化率，t ∈ [60, 100]", "拟合残差"], density_rows))
        write_text("full_scheme_pressure.md",
                   "流体平均压力（ρ0U²），同样的时刻；最后一列是 t ∈ [60, 100] 的线性拟合斜率（ρ0U² 每时间单位）。\n\n"
                   + markdown_table(["运行"] + instant_header + ["变化率，t ∈ [60, 100]"], pressure_rows))
        write_text("full_scheme_negative.md",
                   "压力为负的流体粒子比例，同样的时刻；最后一列是 t ≥ 5 里的最大值。\n\n"
                   + markdown_table(["运行"] + instant_header + ["最大值（t ≥ 5）"], negative_rows))
        write_text("full_scheme_noise.md",
                   "平均压力的逐样本涨落（ρ0U²）：一个样本减去前后两个样本平均值之后的 rms，样本间隔 0.21 个时间单位，"
                   f"按每 {NOISE_BIN:g} 个时间单位一段给出；最后一列是 t ≥ 80。\n\n"
                   + markdown_table(["运行"] + [f"t {low:g}–{high:g}" for low, high in zip(edges[:-1], edges[1:])]
                                    + ["t ≥ 80"], noise_rows))
        setup_style()
        styles = {"n250_reflection_full": (INK_PRIMARY, "solid"),
                  "n250_reflection_full_rho_plus_0p3": (INK_PRIMARY, (0, (4, 2))),
                  "n250_pairwise_full": (INK_SECONDARY, (0, (6, 2, 1, 2))), "n250_reflection": (INK_SECONDARY, (0, (1, 1.5))),
                  "n250_crossing_normal": (first_colour, "solid")}
        figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.4))
        panels = (("density_mean", "流体平均密度 − ρ0（kg/m³）"), ("pressure_mean", "流体平均压力 / (ρ0 U²)"),
                  ("noise", f"平均压力的逐样本涨落 / (ρ0 U²)，每 {NOISE_BIN:g} 个时间单位一段"),
                  ("pressure_negative_fraction", "负压粒子比例（%），t ≥ 1"))
        handles, labels = [], []
        for axis, (key, title) in zip(axes, panels):
            for name, text in history_runs:
                history = histories.get(name)
                if not history:
                    continue
                colour, style = styles[name]
                if key == "noise":
                    values = summary["histories"][name]["noise_bins"]
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    known = [index for index, value in enumerate(values) if value is not None]
                    axis.plot(centres[known], [values[index] for index in known], color=colour, linestyle=style,
                              marker="o", markersize=4, zorder=3, linewidth=1.6)
                    continue
                shown = history["time"] >= (1.0 if key == "pressure_negative_fraction" else 0.0)
                values = history[key][shown]
                values = {"density_mean": values - normal_default.REST_DENSITY,
                          "pressure_mean": values / normal_default.DYNAMIC_PRESSURE,
                          "pressure_negative_fraction": 100.0 * values}[key]
                line, = axis.plot(history["time"][shown], values, color=colour, linestyle=style, zorder=3,
                                  linewidth=1.6)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            if key in ("density_mean", "pressure_mean"):
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            if key == "noise":
                axis.set_ylim(bottom=0.0)
            axis.set_xlim(0, 100)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔 250²：完整方案的两次运行（黑）与 B、A（灰）、一阶（红），密度、压力和压力涨落的历史", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.0)
        figure.savefig(REPORT_ROOT / "figures" / "full_scheme_history.png")
        plt.close(figure)

    # ---------------------------------------------------------------- the fluctuation where no history exists
    # (a) from the particle snapshots of t = 80 to 100 (one per 0.99 time units): the mean pressure of the fluid
    #     per snapshot (from the stored volumes through the EOS), then the same high-pass measure
    rows = []
    summary["snapshot_fluctuation"] = {}
    for name, text in SNAPSHOT_RUNS:
        if not (LOG_ROOT / "cavity" / name / "snapshots").is_dir():
            continue
        run = field_difference.load_run(name)
        if len(run["snapshots"]) < 5:
            continue
        instants, levels = [], []
        for instant, path in run["snapshots"]:
            particles = field_difference.snapshot_particles(run, path)
            instants.append(instant)
            levels.append(float(particles["pressure"][particles["fluid"]].mean()) / normal_default.DYNAMIC_PRESSURE)
        levels = np.array(levels)
        entry = {"snapshots": len(levels), "times": [instants[0], instants[-1]],
                 "level_first": float(levels[0]), "level_last": float(levels[-1]),
                 "fluctuation": high_pass_fluctuation(levels)}
        history = normal_default.load_field_history(name)
        if history:
            # the same instants from the run's own history (the pressure the solver stored): a check of the
            # reconstruction from the volumes, and the same measure on it
            recorded = np.array([history["pressure_mean"][int(np.argmin(np.abs(history["time"] - instant)))]
                                 for instant in instants]) / normal_default.DYNAMIC_PRESSURE
            entry["recorded_fluctuation"] = high_pass_fluctuation(recorded)
            entry["reconstruction_error_maximum"] = float(np.abs(recorded - levels).max())
            late = history["time"] >= 80.0
            fine = history["pressure_mean"][late] / normal_default.DYNAMIC_PRESSURE
            entry["fine_fluctuation"] = high_pass_fluctuation(fine)
        summary["snapshot_fluctuation"][name] = entry
        rows.append([text, f"{entry['snapshots']}（t = {instants[0]:.1f} 到 {instants[-1]:.1f}）",
                     f"{entry['level_first']:+.3f} → {entry['level_last']:+.3f}", f"{entry['fluctuation']:.4f}",
                     "—" if "recorded_fluctuation" not in entry else f"{entry['recorded_fluctuation']:.4f}",
                     "—" if "fine_fluctuation" not in entry else f"{entry['fine_fluctuation']:.4f}"])
    if rows:
        write_text("full_scheme_snapshot_fluctuation.md",
                   "由 t = 80 到 100 的粒子快照（每 0.99 个时间单位一个）得到的平均压力（ρ0U²；由存下的粒子体积经状态方程还原）"
                   "和它的涨落：一个快照减去前后两个快照平均值之后的 rms。有历史文件的运行另给两个数：同样这些时刻取历史文件里"
                   "记录的平均压力算出的涨落（检验还原），和 0.21 间隔的涨落（13.4、14.4 用的量度）。\n\n"
                   + markdown_table(["运行", "快照数", "平均压力：第一个 → 最后一个快照", "涨落（快照，0.99 间隔）",
                                     "同样时刻，用历史文件的压力", "涨落（历史文件，0.21 间隔，t ≥ 80）"], rows))
    # (b) from the budget short runs (t <= 5, a sample every 0.03 time units)
    rows = []
    summary["short_run_fluctuation"] = {}
    for name, text in SHORT_RUNS:
        series_path = LOG_ROOT / "box_run" / name / "series.csv"
        if not series_path.exists():
            continue
        with open(series_path, encoding="utf-8", newline="") as handle:
            series_rows = list(csv.DictReader(handle))
        time = np.array([float(row["time"]) for row in series_rows])
        pressure = np.array([float(row["pressure_mean"]) for row in series_rows]) / normal_default.DYNAMIC_PRESSURE
        value = high_pass_fluctuation(pressure[time >= 1.0])
        summary["short_run_fluctuation"][name] = value
        rows.append([text, f"{value:.4f}", record_text(records_by_name.get(name, {}))])
    if rows:
        write_text("full_scheme_short_run_fluctuation.md",
                   "收支短跑（250²，t ≤ 5.01），t ∈ [1, 5]，样本间隔 0.03 个时间单位：平均压力的逐样本涨落（ρ0U²）。\n\n"
                   + markdown_table(["运行", "涨落", "记录（提交）"], rows))

    out = REPORT_ROOT / "pairwise_reflection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_scheme.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                          encoding="utf-8", newline="\n")
    print("[pairwise_reflection] full-scheme tables written; runs: "
          + "; ".join(f"{name} t = {entry['t_last']:.2f}" + (" complete" if entry["complete"] else "")
                      for name, entry in summary["runs"].items()))
    return 0


# ====================================================================================== the volume constraint
VOLUME_RUNS = (("n250_reflection_full_volume", "250² 完整方案 + 体积约束"),
               ("n500_reflection_full_volume", "500² 完整方案 + 体积约束"))
VOLUME_CONTEXT = (("n250_reflection_full", "250² 完整方案，没有约束（C1）"),
                  ("n250_crossing_normal", "250² 一阶（现缺省）"), ("n500_crossing_normal", "500² 一阶（现缺省）"))
VOLUME_SHORT_RUN = "short_n250_reflection_full_volume"
TIMING_WINDOW = (20.0, 80.0)
TIMING_PAIRS = (("n250_reflection_full_volume", "n250_crossing_normal", "250²"),
                ("n500_reflection_full_volume", "n500_crossing_normal", "500²"),
                ("n250_reflection_full_volume", "n250_reflection_full_rho_plus_0p3",
                 "250²，对没有约束的完整方案（C2，同一张卡）"))


def constraint_log(directory: pathlib.Path) -> dict:
    """volume_constraint.jsonl of a run directory as arrays (empty dict without the file)."""
    path = directory / "volume_constraint.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    return {key: np.array([row[key] for row in rows], dtype=float)
            for key in ("unix", "step", "time", "volume", "target_volume", "theta", "rate")}


def volume_constraint_tables() -> int:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.wall_crossing import delta_wall_experiment, normal_default
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, run_records
    from experiment.wall_crossing.run_log import LOG_ROOT

    write_text = normal_default.write_table
    record_text = normal_default.record_text
    records = run_records()
    records_by_id = {record["run_id"]: record for record in records}
    records_by_name = {}
    for record in records:
        if record.get("name") and str(record.get("status", "")).startswith("ok"):
            records_by_name[record["name"]] = record              # the newest successful record of a name
    first_colour, old_colour = normal_default.NEW_COLOUR, normal_default.OLD_COLOUR
    summary = {}
    tables_path = REPORT_ROOT / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}

    # ---------------------------------------------------------------- 3.1 regression
    rows = []
    manifest_now = normal_default.manifest_rows((REPOSITORY_ROOT / normal_default.MANIFEST).read_text(encoding="utf-8"))
    manifest_base = normal_default.manifest_rows(normal_default.git_show(normal_default.BASE_COMMIT,
                                                                         normal_default.MANIFEST))
    manifest_before = normal_default.manifest_rows(normal_default.git_show("2186032", normal_default.MANIFEST))
    same_as_base = sorted(name for name in manifest_base if manifest_now.get(name) == manifest_base[name])
    changed_since = sorted(name for name in manifest_now if manifest_before.get(name) != manifest_now[name])
    rows.append(["simple、adami 用的 SPIR-V", f"基点（{normal_default.BASE_COMMIT}）清单里的 {len(manifest_base)} 个文件",
                 f"{len(same_as_base)} 个逐字节相同", "通过" if len(same_as_base) == len(manifest_base) else "**不通过**",
                 "MANIFEST.txt"])
    rows.append(["相对 2186032 变了的 SPIR-V", f"全部 {len(manifest_now)} 个文件",
                 "、".join(name.replace(".comp.spv", "") for name in changed_since) or "无", "—", "MANIFEST.txt"])
    summary["regression"] = {"manifest_same_as_base": len(same_as_base), "manifest_base": len(manifest_base),
                             "changed_since_2186032": changed_since}
    classes_path = LOG_ROOT / "outcome_classes" / "volume_constraint_off_against_2186032_s700" / "classes.json"
    if classes_path.exists():
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        members = "；".join(f"类 {index}：参照 {group['reference_runs']} 次，本次 {group['case_runs']} 次"
                           for index, group in enumerate(classes["classes"]))
        rows.append(["完整方案，约束关闭，250²，700 步",
                     "2186032 的求解器（冻结检出 3176d57，求解器文件相同）的 24 个转储",
                     f"{len(classes['classes'])} 个结果类。{members}",
                     "逐位相同" if classes["case_within_reference_classes"] else "**不同**", classes["run_id"]])
        summary["regression"]["constraint_off"] = {"classes": len(classes["classes"]),
                                                   "identical": classes["case_within_reference_classes"],
                                                   "run_id": classes["run_id"]}
    failures = LOG_ROOT / "tests" / "failures_volume_constraint.txt"
    base_failures = LOG_ROOT / "tests" / "failures_base.txt"
    if failures.exists() and base_failures.exists():
        now = sorted(line.strip() for line in failures.read_text(encoding="utf-8").splitlines() if line.strip())
        base = sorted(line.strip() for line in base_failures.read_text(encoding="utf-8").splitlines() if line.strip())
        seam = [record for record in records if record.get("kind") == "seam_layout_test"]
        rows.append(["`_test_seam_layout.py` 的失败清单", "基点的清单（858 条，基点自身就有）",
                     f"{len(now)} 条，" + ("与基点的清单逐行相同" if now == base else "**与基点不同**"),
                     "通过" if now == base else "**不通过**", seam[-1]["run_id"] if seam else "?"])
    write_text("volume_regression.md", markdown_table(["检查", "参照", "结果", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 3.2 single step
    rows = []
    summary["single_step"] = {}
    path = LOG_ROOT / "single_step" / ("summary_2d_delta_wall_reflection_geometry_pairwise_pressure_reflection"
                                       "_volume_constraint.json")
    if path.exists():
        for outcome in json.loads(path.read_text(encoding="utf-8")):
            if outcome["configuration"] != "smooth" or outcome["wall_boundary"] != "crossing":
                continue
            reference = outcome["reference"]
            comparisons = reference["comparisons"]
            terms = reference["reflection_terms"]
            rows.append([
                "2-D", f"{outcome['particles']['fluid']:,} / {reference['counts']['fluid_with_wall_neighbours']:,}",
                f"{terms['volume_constraint_rate_maximum']:.0f} / {terms['density_rate_maximum']:.0f}",
                f"{comparisons['density_rate']['relative_error_beyond_one_ulp_maximum']:.1e}",
                f"{comparisons['density_rate']['new_density_error_ulp_maximum']:.2f}",
                f"{comparisons['acceleration']['error_over_pair_term_sum_maximum']:.1e}",
                f"{reference['worst_relative_error']:.1e}", "通过" if outcome["passed"] else "**不通过**",
                outcome["run_id"]])
            summary["single_step"][outcome["label"]] = {"passed": outcome["passed"],
                                                        "worst": reference["worst_relative_error"],
                                                        "run_id": outcome["run_id"], "terms": terms}
    if rows:
        write_text("volume_single_step.md", markdown_table(
            ["维数", "流体粒子 / 其中有壁邻居的", "约束项 ρ_i·rate 的最大值 / 密度变化率的最大值",
             "密度变化率的相对误差（超出 1 ulp 的部分，最大）", "新密度与参考解之差（ulp，最大）",
             "加速度误差 ÷ 逐对项绝对值之和（最大）", "判据值（容差 1e-4）", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 3.3 short run
    short_directory = LOG_ROOT / "box_run" / VOLUME_SHORT_RUN
    if (short_directory / "series.csv").exists():
        with open(short_directory / "series.csv", encoding="utf-8", newline="") as handle:
            series_rows = list(csv.DictReader(handle))
        series = {key: np.array([float(row[key]) if row[key] != "" else np.nan for row in series_rows])
                  for key in series_rows[0]}
        log = constraint_log(short_directory)
        unconstrained = normal_default.load_field_history("n250_reflection_full")
        rows = []
        for instant in (0.18, 0.36, 0.54, 1.08, 1.98, 3.06, 3.96, 4.86):
            index = int(np.argmin(np.abs(log["time"] - instant)))
            sample = int(np.argmin(np.abs(series["time"] - log["time"][index])))
            other = int(np.argmin(np.abs(unconstrained["time"] - log["time"][index]))) if unconstrained else None
            rows.append([f"{log['time'][index]:.2f}", f"{log['theta'][index]:+.2e}", f"{log['rate'][index]:+.2e}",
                         f"{series['density_mean'][sample] - normal_default.REST_DENSITY:+.5f}",
                         "—" if other is None else
                         f"{unconstrained['density_mean'][other] - normal_default.REST_DENSITY:+.5f}",
                         f"{series['pressure_mean'][sample] / normal_default.DYNAMIC_PRESSURE:+.4f}",
                         f"{100 * series['pressure_negative_fraction'][sample]:.1f} %"])
        late = log["time"] >= 2.0
        late_series = series["time"] >= 2.0
        entry = {
            "updates": int(log["time"].size), "update_interval": float(np.diff(log["time"]).mean()),
            "theta_maximum": float(np.abs(log["theta"]).max()),
            "theta_late_mean": float(log["theta"][late].mean()), "theta_late_std": float(log["theta"][late].std()),
            "theta_late_range": [float(log["theta"][late].min()), float(log["theta"][late].max())],
            "density_late_mean": float(series["density_mean"][late_series].mean() - normal_default.REST_DENSITY),
            "density_late_range": [float(series["density_mean"][late_series].min() - normal_default.REST_DENSITY),
                                   float(series["density_mean"][late_series].max() - normal_default.REST_DENSITY)],
            "alive_constant": bool(series["alive"].min() == series["alive"].max()),
            "outside_maximum": int(series["outside_box_count"].max()),
            "speed_maximum": float(series["speed_maximum"].max()),
            "below_floor_maximum": int(series["below_floor_count"].max()),
            "kinetic_energy_last": float(series["kinetic_energy"][-1]),
            "run_id": records_by_name.get(VOLUME_SHORT_RUN, {}).get("run_id")}
        summary["short_run"] = entry
        write_text("volume_short_run.md",
                   f"250² 方腔短跑，完整方案 + 体积约束，t = 0 到 5.01（记录 {entry['run_id']}）。θ 和 rate 是约束在那一刻"
                   "更新出的值；“没有约束”一列是 C1（第 14 节）在同一时刻的平均密度。\n\n"
                   + markdown_table(["t", "θ", "rate（每时间单位）", "平均密度 − ρ0（kg/m³）", "同，没有约束（C1）",
                                     "平均压力（ρ0U²）", "负压粒子比例"], rows))

    # ---------------------------------------------------------------- 3.4 cavity metrics
    if analysed.get("n250_reflection_full_volume", {}).get("complete"):
        cavity = normal_default.cavity(
            modes=(("reflection_full_volume", "crossing，完整方案 + 体积约束", INK_PRIMARY, "P"),
                   ("crossing", "crossing，旧缺省（一阶，完整向量）", old_colour, "s"),
                   ("adami", "adami", normal_default.ADAMI_COLOUR, "D"),
                   ("crossing_normal", "crossing，一阶（现缺省）", first_colour, "o"),
                   ("simple", "simple", normal_default.SIMPLE_COLOUR, "^")),
            extra_runs=(("n250_reflection_full", "crossing，完整方案，没有约束（C1）", "reflection_full_volume"),
                        ("n250_reflection_full_rho_plus_0p3", "crossing，完整方案，没有约束，初始密度 ρ0 + 0.3（C2）",
                         "reflection_full_volume"),
                        ("n250_pairwise_full", "crossing，逐对几何 + 完整向量（B）", "reflection_full_volume"),
                        ("n250_reflection", "crossing，反射版，只取法向（A）", "reflection_full_volume"),
                        ("n250_quadratic_viscous", "crossing，二阶粘性", "crossing_normal"),
                        ("n500_quadratic_viscous", "crossing，二阶粘性", "crossing_normal")),
            prefix="volume", required_mode="reflection_full_volume", labelled_modes=("reflection_full_volume",),
            labels_beside=True,
            figure_title="Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧；数字 = 完整方案 + 体积约束）")
        if cavity:
            summary["cavity"] = cavity

    # ---------------------------------------------------------------- how the runs went
    rows = []
    summary["runs"] = {}
    for name, text in VOLUME_RUNS:
        directory = LOG_ROOT / "cavity" / name
        history = normal_default.load_field_history(name)
        if not history:
            continue
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
        stop_path = directory / "watch_stop.json"
        entry = {"t_last": float(history["time"][-1]), "complete": bool(result),
                 "alive_constant": bool((history["alive"] == history["alive"][0]).all()),
                 "speed_maximum": float(history["speed_maximum"].max()),
                 "watch_stop": json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None,
                 "crossing_status": crossing}
        summary["runs"][name] = entry
        rows.append([
            text, f"t = {entry['t_last']:.2f}" + ("" if entry["complete"] else "（运行中）"),
            f"{int(history['alive'][0]):,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
            "—" if not crossing or crossing.get("outside_box_maximum") is None else str(crossing["outside_box_maximum"]),
            f"{entry['speed_maximum']:.3f}",
            "没有触发" if entry["watch_stop"] is None else f"**停止：{entry['watch_stop']['reason']}**",
            "—" if not crossing else str(crossing["gradient_fallbacks"]),
            "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
            else f"{crossing['wall_distance_minimum_over_spacing']:.3f}；{crossing['below_floor_maximum']}",
            record_text(records_by_name.get(name, {}))])
    if rows:
        write_text("volume_runs.md", markdown_table(
            ["运行", "到达", "粒子数（含壁）", "越出盒的流体粒子（各次位置读回的最大值）", "流体最大速度（线 1.05）",
             "停止条件", "A_f 回退（累计）", "φ_min / dx；低于距离下限的粒子（最大）", "记录（提交）"], rows))

    # ---------------------------------------------------------------- density, theta, pressure, noise
    history_runs = VOLUME_RUNS + VOLUME_CONTEXT
    histories = {name: normal_default.load_field_history(name) for name, _ in history_runs}
    histories = {name: history for name, history in histories.items() if history and len(history["time"]) > 5}
    logs = {name: constraint_log(LOG_ROOT / "cavity" / name) for name, _ in VOLUME_RUNS}
    edges = np.arange(0.0, 100.0 + 1e-9, NOISE_BIN)
    density_rows, pressure_rows, negative_rows, noise_rows, theta_rows = [], [], [], [], []
    summary["histories"], summary["theta"] = {}, {}
    for name, text in history_runs:
        history = histories.get(name)
        if not history:
            continue
        time = history["time"]
        entry = {"instants": {}}
        density_cells, pressure_cells, negative_cells = [text], [text], [text]
        for instant in DENSITY_INSTANTS:
            if instant > time[-1] + 0.3:
                density_cells.append("—")
                pressure_cells.append("—")
                negative_cells.append("—")
                continue
            index = int(np.argmin(np.abs(time - instant)))
            entry["instants"][f"{instant:g}"] = {
                "time": float(time[index]),
                "density_mean_minus_rest": float(history["density_mean"][index] - normal_default.REST_DENSITY),
                "pressure_mean": float(history["pressure_mean"][index]),
                "pressure_negative_fraction": float(history["pressure_negative_fraction"][index])}
            density_cells.append(f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.5f}")
            pressure_cells.append(f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.4f}")
            negative_cells.append(f"{100.0 * history['pressure_negative_fraction'][index]:.1f} %")
        inside = (time >= RATE_WINDOW[0]) & (time <= RATE_WINDOW[1] + 0.3)
        if inside.sum() >= 5 and time[-1] >= RATE_WINDOW[1] - 1.0:
            slope, intercept = np.polyfit(time[inside], history["density_mean"][inside], 1)
            mean_level = float(history["density_mean"][inside].mean() - normal_default.REST_DENSITY)
            spread = float(history["density_mean"][inside].std())
            pressure_window = history["pressure_mean"][inside] / normal_default.DYNAMIC_PRESSURE
            negative_window = history["pressure_negative_fraction"][inside]
            entry.update({"density_rate_60_100": float(slope), "density_mean_60_100": mean_level,
                          "density_std_60_100": spread, "pressure_mean_60_100": float(pressure_window.mean()),
                          "pressure_std_60_100": float(pressure_window.std()),
                          "negative_mean_60_100": float(negative_window.mean()),
                          "negative_range_60_100": [float(negative_window.min()), float(negative_window.max())]})
            density_cells += [f"{slope:+.6f}", f"{mean_level:+.5f} ± {spread:.5f}"]
            pressure_cells.append(f"{pressure_window.mean():+.4f} ± {pressure_window.std():.4f}")
            negative_cells.append(f"{100 * negative_window.mean():.1f} %（{100 * negative_window.min():.1f} 到 "
                                  f"{100 * negative_window.max():.1f}）")
        else:
            density_cells += ["—", "—"]
            pressure_cells.append("—")
            negative_cells.append("—")
        pressure = history["pressure_mean"] / normal_default.DYNAMIC_PRESSURE
        high_pass = pressure[1:-1] - 0.5 * (pressure[:-2] + pressure[2:])
        centre_time = time[1:-1]
        cells, bins = [text], []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = (centre_time >= low) & (centre_time < high)
            value = float(np.sqrt((high_pass[chosen] ** 2).mean())) if chosen.sum() >= 5 else None
            bins.append(value)
            cells.append("—" if value is None else f"{value:.4f}")
        late = centre_time >= 80.0
        entry.update({"noise_bins": bins,
                      "noise_late": float(np.sqrt((high_pass[late] ** 2).mean())) if late.sum() >= 5 else None})
        cells.append("—" if entry["noise_late"] is None else f"{entry['noise_late']:.4f}")
        summary["histories"][name] = entry
        density_rows.append(density_cells)
        pressure_rows.append(pressure_cells)
        negative_rows.append(negative_cells)
        noise_rows.append(cells)
    for name, text in VOLUME_RUNS:
        log = logs.get(name)
        if not log:
            continue
        cells = [text]
        for instant in DENSITY_INSTANTS:
            if instant > log["time"][-1] + 0.3:
                cells.append("—")
                continue
            index = int(np.argmin(np.abs(log["time"] - instant)))
            cells.append(f"{log['theta'][index]:+.2e}")
        entry = {"updates": int(log["time"].size), "update_interval": float(np.diff(log["time"]).mean()),
                 "theta_maximum": float(np.abs(log["theta"]).max()),
                 "theta_maximum_time": float(log["time"][int(np.argmax(np.abs(log["theta"])))])}
        inside = (log["time"] >= RATE_WINDOW[0]) & (log["time"] <= RATE_WINDOW[1] + 0.3)
        if inside.sum() >= 5 and log["time"][-1] >= RATE_WINDOW[1] - 1.0:
            entry.update({"theta_mean_60_100": float(log["theta"][inside].mean()),
                          "theta_std_60_100": float(log["theta"][inside].std()),
                          "theta_range_60_100": [float(log["theta"][inside].min()), float(log["theta"][inside].max())]})
            cells += [f"{entry['theta_mean_60_100']:+.2e} ± {entry['theta_std_60_100']:.1e}",
                      f"{-normal_default.REST_DENSITY * entry['theta_mean_60_100']:+.5f}"]
        else:
            cells += ["—", "—"]
        cells.append(f"{entry['updates']}（间隔 {entry['update_interval']:.3f}）")
        summary["theta"][name] = entry
        theta_rows.append(cells)
    instant_header = [f"t = {instant:g}" for instant in DENSITY_INSTANTS]
    if density_rows:
        write_text("volume_density.md",
                   "流体平均密度 − ρ0（kg/m³），各时刻最近的那个样本（样本间隔 0.21）。变化率 = t ∈ [60, 100] 的线性拟合斜率，"
                   "kg/m³ 每时间单位；最后一列是这一段的平均值 ± 标准差。\n\n"
                   + markdown_table(["运行"] + instant_header + ["变化率，t ∈ [60, 100]", "t ∈ [60, 100] 的平均 ± 标准差"],
                                    density_rows))
        write_text("volume_pressure.md",
                   "流体平均压力（ρ0U²），同样的时刻；最后一列是 t ∈ [60, 100] 的平均 ± 标准差。\n\n"
                   + markdown_table(["运行"] + instant_header + ["t ∈ [60, 100]"], pressure_rows))
        write_text("volume_negative.md",
                   "压力为负的流体粒子比例，同样的时刻；最后一列是 t ∈ [60, 100] 的平均（最小到最大）。\n\n"
                   + markdown_table(["运行"] + instant_header + ["t ∈ [60, 100]"], negative_rows))
        write_text("volume_noise.md",
                   "平均压力的逐样本涨落（ρ0U²）：一个样本减去前后两个样本平均值之后的 rms，样本间隔 0.21 个时间单位，"
                   f"按每 {NOISE_BIN:g} 个时间单位一段给出；最后一列是 t ≥ 80。\n\n"
                   + markdown_table(["运行"] + [f"t {low:g}–{high:g}" for low, high in zip(edges[:-1], edges[1:])]
                                    + ["t ≥ 80"], noise_rows))
    if theta_rows:
        write_text("volume_theta.md",
                   "θ = (V_now − V_target) / V_target，各时刻最近的那次更新。θ > 0 表示密度整体偏低；−ρ0·θ 是它对应的平均"
                   "密度偏差。\n\n"
                   + markdown_table(["运行"] + instant_header + ["t ∈ [60, 100] 的平均 ± 标准差", "对应的密度偏差 −ρ0·θ（kg/m³）",
                                                               "更新次数"], theta_rows))
    if density_rows:
        setup_style()
        styles = {"n250_reflection_full_volume": (INK_PRIMARY, "solid"),
                  "n500_reflection_full_volume": (INK_PRIMARY, (0, (4, 2))),
                  "n250_reflection_full": (INK_SECONDARY, (0, (1, 1.5))),
                  "n250_crossing_normal": (first_colour, "solid"), "n500_crossing_normal": (first_colour, (0, (4, 2)))}
        figure, axes = plt.subplots(1, 5, figsize=(20.0, 4.4))
        panels = (("density_mean", "流体平均密度 − ρ0（kg/m³）"), ("theta", "θ（×10⁻⁶）"),
                  ("pressure_mean", "流体平均压力 / (ρ0 U²)"),
                  ("noise", f"平均压力的逐样本涨落 / (ρ0 U²)，每 {NOISE_BIN:g} 个时间单位一段"),
                  ("pressure_negative_fraction", "负压粒子比例（%），t ≥ 1"))
        handles, labels = [], []
        for axis, (key, title) in zip(axes, panels):
            for name, text in history_runs:
                history = histories.get(name)
                if not history:
                    continue
                colour, style = styles[name]
                if key == "theta":
                    log = logs.get(name)
                    if log:
                        axis.plot(log["time"], 1.0e6 * log["theta"], color=colour, linestyle=style, zorder=3,
                                  linewidth=1.2)
                    continue
                if key == "noise":
                    values = summary["histories"][name]["noise_bins"]
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    known = [index for index, value in enumerate(values) if value is not None]
                    axis.plot(centres[known], [values[index] for index in known], color=colour, linestyle=style,
                              marker="o", markersize=4, zorder=3, linewidth=1.6)
                    continue
                shown = history["time"] >= (1.0 if key == "pressure_negative_fraction" else 0.0)
                values = history[key][shown]
                values = {"density_mean": values - normal_default.REST_DENSITY,
                          "pressure_mean": values / normal_default.DYNAMIC_PRESSURE,
                          "pressure_negative_fraction": 100.0 * values}[key]
                line, = axis.plot(history["time"][shown], values, color=colour, linestyle=style, zorder=3,
                                  linewidth=1.4)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            if key in ("density_mean", "pressure_mean", "theta"):
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            if key == "density_mean":
                axis.set_ylim(-0.05, 0.05)          # the unconstrained run leaves the panel (see the table)
            if key == "pressure_mean":
                axis.set_ylim(-0.5, 0.5)
            if key == "noise":
                axis.set_ylim(bottom=0.0)
            axis.set_xlim(0, 100)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔：完整方案 + 体积约束（黑），没有约束的完整方案（灰，画到出框为止），一阶（红）", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=1.8)
        figure.savefig(REPORT_ROOT / "figures" / "volume_history.png")
        plt.close(figure)

    # ---------------------------------------------------------------- time per step from the run logs
    def clock_history(name: str) -> dict:
        path = LOG_ROOT / "cavity" / name / "field_history.jsonl"
        if not path.exists():
            return {}
        by_step = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("step") is not None:
                    by_step[int(row["step"])] = row
        steps = sorted(by_step)
        return {key: np.array([by_step[step][key] for step in steps], dtype=float) for key in ("step", "time", "unix")}

    def step_rate(name: str):
        history = clock_history(name)
        if not history:
            return None
        inside = (history["time"] >= TIMING_WINDOW[0]) & (history["time"] <= TIMING_WINDOW[1])
        if inside.sum() < 3 or history["time"][-1] < TIMING_WINDOW[1] - 1.0:
            return None
        return float((history["step"][inside][-1] - history["step"][inside][0])
                     / (history["unix"][inside][-1] - history["unix"][inside][0]))

    def device_of(name: str) -> str:
        uuids = analysed.get(name, {}).get("device_uuids") or []
        return {"fb83f0dafd9e1d38dc05e7ae643e22a8": "显示卡", "ae137c668a40f5acaab90a83f7cda175": "无头卡"} \
            .get(uuids[0] if uuids else "", "?")

    rows = []
    summary["timing"] = {}
    for name, reference_name, text in TIMING_PAIRS:
        rate, reference_rate = step_rate(name), step_rate(reference_name)
        if rate is None or reference_rate is None:
            continue
        ratio = reference_rate / rate
        summary["timing"][f"{name}|{reference_name}"] = {"steps_per_second": rate,
                                                        "reference_steps_per_second": reference_rate,
                                                        "time_per_step_ratio": ratio}
        rows.append([text, f"{rate:.0f}（{device_of(name)}）", f"{reference_rate:.0f}（{device_of(reference_name)}）",
                     f"**{ratio:.3f}**"])
    if rows:
        write_text("volume_timing.md",
                   f"每秒步数取自运行日志（field_history.jsonl 的墙钟时间），t ∈ [{TIMING_WINDOW[0]:g}, "
                   f"{TIMING_WINDOW[1]:g}]，含运行器的采样。每步时间之比 = 参照的每秒步数 ÷ 本次的每秒步数。括号里是跑这次运行"
                   "的卡。\n\n"
                   + markdown_table(["对比", "本次：每秒步数", "参照：每秒步数", "每步时间之比"], rows))

    # ---------------------------------------------------------------- 3.5 hydrostatic boxes
    hydrostatic_runs = {}
    for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
        entries = ((f"2d_n250_{direction}_crossing_normal_r1", "一阶（现缺省），第 1 次", 0),
                   (f"2d_n250_{direction}_crossing_normal_r2", "一阶（现缺省），第 2 次", 0),
                   (f"2d_n250_{direction}_crossing_reflection_full_volume", "完整方案 + 体积约束", 1))
        present = tuple(entry for entry in entries if (LOG_ROOT / "hydrostatic" / entry[0] / "summary.json").exists())
        if any(form == 1 for _, _, form in present):
            hydrostatic_runs[direction] = (direction_text, present)
    if len(hydrostatic_runs) == len(HYDROSTATIC_DIRECTIONS):
        introduction = (
            "2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始，到 t = 10.02）。速度以 "
            "√(|g| H) 为单位，压力以 ρ0 |g| H 为单位。“去电平” = 减去全场平均偏差后的误差；“电平” = 压力减去 "
            "ρ0 g·(r − r_中心) 之后对全部流体粒子的平均。一阶的两次是第 9 节的运行，完整方案 + 体积约束每个方向一次。")
        hydrostatic = delta_wall_experiment.hydrostatic_comparison(
            records_by_id, runs_by_direction=hydrostatic_runs,
            forms=(("一阶（现缺省）", first_colour, "solid"), ("完整方案 + 体积约束", INK_PRIMARY, "solid")),
            introduction=introduction, table_name="volume_hydrostatic.md", figure_name="volume_hydrostatic.png",
            figure_title="2-D 静水盒：一阶（现缺省）与完整方案 + 体积约束（压力以 ρ0 g H 为单位）")
        ratio_rows, theta_rows = [], []
        summary["hydrostatic"] = {"runs": hydrostatic, "ratios": {}, "theta": {}}
        for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
            entries = hydrostatic.get(direction, {})
            first = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                     if "normal_r" in name]
            second = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                      if name.endswith("_reflection_full_volume")]
            if not first or not second:
                continue
            ratio = second[0] / (sum(first) / len(first))
            summary["hydrostatic"]["ratios"][direction] = {"first_order": first, "constrained": second[0],
                                                          "ratio_to_first_order_mean": ratio}
            ratio_rows.append([direction_text, " / ".join(f"{value:.2e}" for value in first), f"{second[0]:.2e}",
                               f"{ratio:.2f}", "通过" if ratio <= 2.0 else "**不通过**"])
            log = constraint_log(LOG_ROOT / "hydrostatic" / f"2d_n250_{direction}_crossing_reflection_full_volume")
            if log:
                late = log["time"] >= 3.0
                entry = {"theta_maximum": float(np.abs(log["theta"]).max()),
                         "theta_late_mean": float(log["theta"][late].mean()),
                         "theta_late_std": float(log["theta"][late].std()),
                         "theta_last": float(log["theta"][-1]), "updates": int(log["time"].size)}
                summary["hydrostatic"]["theta"][direction] = entry
                cells = [direction_text]
                for instant in (0.18, 1.08, 2.52, 5.04, 7.56, 9.9):
                    index = int(np.argmin(np.abs(log["time"] - instant)))
                    cells.append(f"{log['theta'][index]:+.2e}")
                cells += [f"{entry['theta_late_mean']:+.2e} ± {entry['theta_late_std']:.1e}",
                          f"{-normal_default.REST_DENSITY * entry['theta_late_mean']:+.5f}"]
                theta_rows.append(cells)
        write_text("volume_hydrostatic_ratio.md", markdown_table(
            ["方向", "一阶的残余速度 rms / √(gH)（末 1/5 段；两次）", "完整方案 + 体积约束", "÷ 一阶两次的平均", "不超过 2 倍"],
            ratio_rows))
        if theta_rows:
            write_text("volume_hydrostatic_theta.md",
                       "静水盒里约束的 θ（各时刻最近的那次更新）；最后两列是 t ≥ 3 的平均 ± 标准差和它对应的密度偏差。\n\n"
                       + markdown_table(["方向", "t = 0.18", "t = 1.08", "t = 2.52", "t = 5.04", "t = 7.56", "t = 9.9",
                                         "t ≥ 3 的平均 ± 标准差", "−ρ0·θ（kg/m³）"], theta_rows))

    out = REPORT_ROOT / "pairwise_reflection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "volume_constraint.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                                encoding="utf-8", newline="\n")
    print("[pairwise_reflection] volume-constraint tables written; runs: "
          + "; ".join(f"{name} t = {entry['t_last']:.2f}" + (" complete" if entry["complete"] else "")
                      for name, entry in summary["runs"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
