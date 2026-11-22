"""
wall_store_register_probe.py - where the wall store's near-wall pipelines spend their registers
(claude/wall_store_report.md). Compile only: no case is run.

    .venv/Scripts/python.exe -m experiment.wall_crossing.wall_store_register_probe --device 1 [--source DIRECTORY]

For every probe, copies of the shader sources (experiment/v7/shaders, or --source) go to a temporary directory,
text replacements take one block out of the WALL_CROSSING build (the results are WRONG: register counting only,
never a production build), correction_density_wall_crossing / force_wall_crossing are compiled with the release
flags into a copy of the SPIR-V directory, and pipeline_statistics.py (V7_SPV_DIR = that copy,
V7_WALL_CROSSING_SPLIT_DISPATCH=1) reads the driver's register count of the near-wall pipelines (dispatch class 2)
of a 2-D (250^2) and a 3-D (50 spacings per side) crossing case. Probes:
  fused: inverse (A_f^-1: identity, nothing written), rule (the two rule functions: constants), geometry (the pair
         geometry: constants), tables (plane / body / wall material table reads: constants), gradient_output (the
         density gradient not kept to the end), wall_correction (no wall pair in the correction matrix), wall_rate
         (no wall pair in S), all;
  force: pressure_gradient (no g_f of the pressure in the fluid segment), inverse_read (A_f^-1: constants), rule,
         geometry, tables, viscous (no wall viscous term), shifting (no wall shifting term), all.
The probe texts match the shader sources before the round-2 register items (commit 351c27c; round 2's attribution
tables register_probe.md / after_rb_register_probe.md); on a later tree use --baseline-only (with --environment
V7_WALL_CROSSING_<item>=<revert value> for one item taken back) or point --source at such a checkout's shaders.
Theoretical occupancy as in wall_store_check.theoretical_occupancy. Writes
claude/wall_store/tables/<prefix>register_probe.md and logs/wall_store/register_probe/<prefix>summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SHADERS = REPOSITORY_ROOT / "experiment" / "v7" / "shaders"
CASES = (("2-D", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml"),
         ("3-D", "cases/cavity3d_n50_crossing/case.yaml"))


def replace_call(text: str, function: str, replacement: str, occurrence: int = 0, statement: bool = False) -> str:
    """Replace the occurrence-th call of function (through its matching ')', and the ';' with statement) by
    replacement."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(function + "(", start + 1)
    depth = 0
    for index in range(start + len(function), len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            end = index + 1
            break
    if statement:
        end = text.index(";", end) + 1
    return text[:start] + replacement + text[end:]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) < 1:
        raise ValueError(f"probe text not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def remove_block(text: str, opening: str, occurrence: int) -> str:
    """Replace the occurrence-th block that starts with opening ('if (...) {') by 'if (false) {' + its body kept."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(opening, start + 1)
    return text[:start] + "if (false) {" + text[start + len(opening):]


def tables(glsl: str) -> str:
    glsl = replace_once(glsl, "    return wall_bodies[5u * body_id].xyz + cross(wall_bodies[5u * body_id + 1u].xyz, arm);",
                        "    return vec3(0.0) * arm;")
    glsl = replace_once(glsl, """    return wall_bodies[5u * body_id + 3u].xyz + cross(wall_bodies[5u * body_id + 4u].xyz, arm)
         + cross(angular_velocity, cross(angular_velocity, arm));""", "    return vec3(0.0) * arm;")
    glsl = replace_once(glsl, "    vec3 arm = position - wall_bodies[5u * body_id + 2u].xyz;",
                        "    vec3 arm = position;")
    glsl = replace_once(glsl, "    vec3 arm = position - wall_bodies[5u * body_id + 2u].xyz;",
                        "    vec3 arm = position;")
    glsl = replace_once(glsl, "    vec3 angular_velocity = wall_bodies[5u * body_id + 1u].xyz;\n", "")
    glsl = replace_once(glsl, "        vec3 plane_normal = wall_planes[2u * plane_id].xyz;",
                        "        vec3 plane_normal = vec3(float(plane_id & 1u), 1.0 - float(plane_id & 1u), 0.0);")
    glsl = replace_once(glsl, "        vec3 plane_point  = wall_planes[2u * plane_id + 1u].xyz;",
                        "        vec3 plane_point  = vec3(-0.5);")
    return glsl


RULE_CONSTANTS = {"WALL_CROSSING_RULE_CONTINUITY_VELOCITY": "vec4(1.0, 1.0, 0.0, 0.0)",
                  "WALL_CROSSING_RULE_VISCOUS_VELOCITY": "vec4(1.0, 1.0, 0.0, 0.0)",
                  "WALL_CROSSING_RULE_PRESSURE": "vec4(0.0, 1.0, 1.0, 0.0)",
                  "WALL_CROSSING_RULE_DENSITY": "vec4(0.0, 1.0, 1.0, 0.0)"}


def rule_tables(comp: str) -> str:
    for field, constant in RULE_CONSTANTS.items():
        comp = re.sub(r"wall_rules\[rule_base \+ " + field + r"\]", constant, comp)
    return comp


def geometry(comp: str) -> str:
    return replace_call(comp, "wall_crossing_pair_geometry",
                        "crossed_wall_normal = vec3(1.0, 0.0, 0.0); depth_ratio = 0.5; "
                        "mirror_displacement = vec3(0.001, 0.0, 0.0);", statement=True)


WALL_RATE_BLOCK = """                    // S += q_j ⊗ ∇W_ij, symmetric part
                    if (DIMENSION == 2u) {
                        density_rate_diagonal.xy    += rate_vector.xy * kernel_gradient_value.xy;
                        density_rate_off_diagonal.x += rate_vector.x * kernel_gradient_value.y
                                                     + rate_vector.y * kernel_gradient_value.x;
                    } else {
                        density_rate_diagonal     += rate_vector * kernel_gradient_value;
                        density_rate_off_diagonal += rate_vector.xxy * kernel_gradient_value.yzz
                                                   + rate_vector.yzz * kernel_gradient_value.xxy;
                    }
"""

FUSED_PROBES = {
    "inverse": lambda comp, glsl: (replace_once(replace_once(
        comp, "    bool fluid_inverse_valid = wall_crossing_invert_fluid_matrix(correction_matrix, fluid_correction_inverse);",
        "    fluid_correction_inverse = mat3(1.0);\n    bool fluid_inverse_valid = true;"),
        """    wall_crossing_fluid_inverse[self_particle_id * 2u]      = fluid_inverse_diagonal_and_xy;
    wall_crossing_fluid_inverse[self_particle_id * 2u + 1u] = fluid_inverse_xz_and_yz;
""", ""), glsl),
    "rule": lambda comp, glsl: (replace_call(replace_call(comp, "wall_crossing_rule_velocity_difference", "vec3(0.5)"),
                                             "wall_crossing_rule_scalar_difference", "0.25"), glsl),
    "geometry": lambda comp, glsl: (geometry(comp), glsl),
    "tables": lambda comp, glsl: (rule_tables(comp), tables(glsl)),
    "gradient_output": lambda comp, glsl: (replace_once(
        comp, "    density_gradient_kernel_sum[self_particle_id] = vec4(density_gradient, kernel_sum);",
        "    density_gradient_kernel_sum[self_particle_id] = vec4(vec3(0.0), kernel_sum);"), glsl),
    "wall_correction": lambda comp, glsl: (replace_once(comp, """                    correction_matrix += neighbor_volume
                        * outerProduct(position_difference_self_to_neighbor, kernel_gradient_value);
""", ""), glsl),
    "wall_rate": lambda comp, glsl: (replace_once(comp, WALL_RATE_BLOCK, ""), glsl),
}
FORCE_PROBES = {
    "pressure_gradient": lambda comp, glsl: (replace_once(comp, """                        fluid_pressure_gradient_sum += neighbor_volume_dynamic
                                                     * (neighbor_pressure - self_pressure)
                                                     * kernel_gradient_raw;""", ""), glsl),
    "inverse_read": lambda comp, glsl: (replace_once(replace_once(
        comp, "        vec4 fluid_inverse_diagonal_and_xy = wall_crossing_fluid_inverse[self_particle_id * 2u];",
        "        vec4 fluid_inverse_diagonal_and_xy = vec4(1.0, 1.0, 1.0, 0.0);"),
        "        vec4 fluid_inverse_xz_and_yz       = wall_crossing_fluid_inverse[self_particle_id * 2u + 1u];",
        "        vec4 fluid_inverse_xz_and_yz       = vec4(0.0, 0.0, 0.0, 1.0);"), glsl),
    "rule": lambda comp, glsl: (replace_call(replace_call(comp, "wall_crossing_rule_scalar_difference", "0.25"),
                                             "wall_crossing_rule_velocity_difference", "vec3(0.5)"), glsl),
    "geometry": lambda comp, glsl: (geometry(comp), glsl),
    "tables": lambda comp, glsl: (rule_tables(comp), tables(glsl)),
    "viscous": lambda comp, glsl: (replace_once(comp, """                        a_viscosity +=
                            morris_coefficient * viscosity_uniform * neighbor_volume
                            * dot(viscous_velocity_difference, position_difference_neighbor_to_self)
                            / (distance * distance + EPS_H_SQUARED)
                            * kernel_gradient_corrected;""", ""), glsl),
    "shifting": lambda comp, glsl: (remove_block(comp, "if (USE_PST) {", 1), glsl),
}


def build(work: pathlib.Path, source: pathlib.Path, fused_names: list, force_names: list) -> pathlib.Path:
    from experiment.v7.compile_shaders_v7 import glslc_command
    sources = work / "shaders"
    if sources.exists():
        shutil.rmtree(sources)
    shutil.copytree(source, sources, ignore=shutil.ignore_patterns("spv", "render"))
    spv = work / "spv"
    if spv.exists():
        shutil.rmtree(spv)
    shutil.copytree(SHADERS / "spv", spv, ignore=shutil.ignore_patterns("render"))
    glsl_path = sources / "wall_boundary.glsl"
    base_glsl = glsl_path.read_text(encoding="utf-8")
    for source_name, names, probes, output in (
            ("correction_density.comp", fused_names, FUSED_PROBES, "correction_density_wall_crossing"),
            ("force.comp", force_names, FORCE_PROBES, "force_wall_crossing")):
        comp = (sources / source_name).read_text(encoding="utf-8")
        glsl = base_glsl
        for name in names:
            comp, glsl = probes[name](comp, glsl)
        (sources / source_name).write_text(comp, encoding="utf-8", newline="\n")
        glsl_path.write_text(glsl, encoding="utf-8", newline="\n")
        command = glslc_command(str(sources / source_name), str(spv / f"{output}.comp.spv"), ("WALL_CROSSING",))
        command[command.index("-I") + 1] = str(sources)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"glslc failed for {output} {names}: {result.stderr[:2000]}")
        glsl_path.write_text(base_glsl, encoding="utf-8", newline="\n")
    return spv


def registers(spv: pathlib.Path, case: str, device: int, out: pathlib.Path, extra_environment: dict) -> dict:
    environment = dict(os.environ, V7_SPV_DIR=str(spv), OPENBLAS_NUM_THREADS="1",
                       VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation", V7_WALL_CROSSING_SPLIT_DISPATCH="1",
                       **extra_environment)
    subprocess.run([sys.executable, str(REPOSITORY_ROOT / "experiment/wall_crossing/pipeline_statistics.py"),
                    "--case", case, "--device", str(device), "--filter",
                    "correction_density_all,force_all", "--out", str(out)],
                   cwd=REPOSITORY_ROOT, env=environment, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    pipelines = json.loads(out.read_text(encoding="utf-8"))["pipelines"]
    result = {}
    for name, key in (("correction_density_all_split_near_wall", "fused"), ("force_all_split_near_wall", "force"),
                      ("correction_density_all_split_interior", "fused_interior"),
                      ("force_all_split_interior", "force_interior"),
                      ("correction_density_all", "fused_one_pipeline"), ("force_all", "force_one_pipeline")):
        entry = pipelines[name]
        entry = entry[0] if isinstance(entry, list) else entry
        statistics = entry.get("statistics", entry)
        result[key] = int(statistics["Register Count"])
    return result


def probe_runs() -> list:
    fused, force = list(FUSED_PROBES), list(FORCE_PROBES)
    runs = [("baseline", [], [])]
    for index in range(max(len(fused), len(force))):
        runs.append((f"probe_{index}", fused[index:index + 1], force[index:index + 1]))
    runs.append(("all", fused, force))
    return runs


def main() -> int:
    from experiment.wall_crossing.wall_store_check import markdown_table, theoretical_occupancy, write_table
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--source", default=str(SHADERS))
    parser.add_argument("--prefix", default="")
    parser.add_argument("--baseline-only", action="store_true",
                        help="only the unpatched build (the register count of a source tree)")
    parser.add_argument("--environment", action="append", default=[],
                        help="NAME=VALUE for the statistics run (e.g. a debug switch)")
    arguments = parser.parse_args()
    extra_environment = dict(item.split("=", 1) for item in arguments.environment)
    out = REPOSITORY_ROOT / "logs" / "wall_store" / "register_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    runs = probe_runs()[:1] if arguments.baseline_only else probe_runs()
    with tempfile.TemporaryDirectory(prefix="wall_store_register_probe_") as temporary:
        work = pathlib.Path(temporary)
        for name, fused_names, force_names in runs:
            spv = build(work, pathlib.Path(arguments.source), fused_names, force_names)
            summary[name] = {"fused_probes": fused_names, "force_probes": force_names}
            for dimension, case in CASES:
                summary[name][dimension] = registers(spv, case, arguments.device,
                                                     out / f"{arguments.prefix}{name}_{dimension}.json",
                                                     extra_environment)
            print(name, json.dumps(summary[name]), flush=True)
    (out / f"{arguments.prefix}summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if arguments.baseline_only:
        return 0
    rows = []
    baseline = summary["baseline"]
    for kernel, text in (("fused", "融合遍"), ("force", "受力")):
        probes = FUSED_PROBES if kernel == "fused" else FORCE_PROBES
        for probe in ["（全部保留）"] + list(probes) + ["all"]:
            run = ("baseline" if probe == "（全部保留）" else
                   "all" if probe == "all" else next(name for name, entry in summary.items()
                                                    if probe in entry[f"{kernel}_probes"] and name != "all"))
            cells = [text, probe]
            for dimension, _ in CASES:
                count = summary[run][dimension][kernel]
                change = count - baseline[dimension][kernel]
                cells.append(f"{count}（{change:+d}）· {100 * theoretical_occupancy(count):.1f} %")
            rows.append(cells)
    write_table(f"{arguments.prefix}register_probe.md", markdown_table(
        ["遍（近壁管线）", "去掉的块", "2-D：寄存器（变化）· 理论占用率", "3-D：寄存器（变化）· 理论占用率"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
