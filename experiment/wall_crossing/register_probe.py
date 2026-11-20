"""
register_probe.py - report section 27, 2.1: where the near-wall builds spend their registers.

    .venv/Scripts/python.exe -m experiment.wall_crossing.register_probe --device 1

For every probe, copies of experiment/v7/shaders go to a temporary directory, correction_density.comp and / or
force.comp get text replacements that take one block out of the WALL_CROSSING_NEAR_WALL build (the results are
WRONG: register counting only, never a production build), the two near-wall modules are compiled with the same glslc
flags into a copy of the SPIR-V directory, and experiment/wall_crossing/pipeline_statistics.py (V7_SPV_DIR = that
copy) reads the driver's register count of the near-wall pipelines of a 2-D and a 3-D crossing case. A removed
in-loop accumulator is replaced after the loop by a value that is live anyway, so the code after the loop keeps its
work and only the accumulator's own registers go. Probes (one fused block and one force block per run):
  fused: fluid_matrix (A_f), fluid_gradient (g_f of the density), delta_wall (the two wall sums of the delta term),
         pair_geometry (constants instead of wall_crossing_pair_geometry), epilogue (A_f^-1 and the wall terms after
         the loop), all (every fused block);
  force: wall_gradient (sum_wall V_j grad W_ij), pressure_gradient (g_f of the pressure), mirror_matrix (M_d),
         pair_geometry, viscous_factor (the wall neighbour's viscous factor), epilogue (the wall pressure after the
         loop), all.
Theoretical occupancy as in efficiency_round1.py. Writes claude/wall_crossing/tables/register_probe.md and
logs/wall_crossing/section27/register_probe/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.efficiency_round1 import theoretical_occupancy     # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

SHADERS = REPOSITORY_ROOT / "experiment" / "v7" / "shaders"
CASES = (("2-D", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml"),
         ("3-D", "cases/cavity3d_n50_crossing/case.yaml"))
FUSED_AFTER_LOOP = ("    // ========================================================================\n"
                    "    // correction.comp's epilogue: regularize, threshold / invert / Frobenius\n")
FORCE_AFTER_LOOP = ("#ifdef WALL_CROSSING\n    // ========================================================================\n"
                    "    // wall_boundary crossing: the wall neighbours' pressure term.")

FUSED_PROBES = {
    "fluid_matrix": [
        ("""                        if (!neighbor_is_wall) {
                            fluid_correction_matrix += neighbor_volume
                                * outerProduct(position_difference_self_to_neighbor, kernel_gradient_value);
                        }
""", ""),
        (FUSED_AFTER_LOOP, "#ifdef WALL_CROSSING\n    fluid_correction_matrix = correction_matrix;\n#endif\n" + FUSED_AFTER_LOOP)],
    "fluid_gradient": [
        ("""                    if (DIMENSION == 2u) {
                        density_gradient.xy += neighbor_volume
                                             * (neighbor_stored_density - self_stored_density)
                                             * kernel_gradient_value.xy;
                    } else {
                        density_gradient += neighbor_volume
                                          * (neighbor_stored_density - self_stored_density)
                                          * kernel_gradient_value;
                    }
""", ""),
        (FUSED_AFTER_LOOP, "    density_gradient = vec3(kernel_sum);\n" + FUSED_AFTER_LOOP)],
    "delta_wall": [
        ("""                            wall_diffusion_vector += (neighbor_volume * distance * distance
                                                      / (distance * distance + EPS_H_SQUARED))
                                                   * kernel_gradient_value;
                            if (WALL_CROSSING_DELTA_REFLECTION) {
                                wall_reflection_diffusion_vector +=
                                    (neighbor_volume * length(kernel_gradient_value) * distance
                                     / (distance * distance + EPS_H_SQUARED))
                                    * pair_mirror_displacement;
                            }
""", ""),
        (FUSED_AFTER_LOOP, "#ifdef WALL_CROSSING\n    wall_diffusion_vector = density_rate_diagonal;\n"
                           "    wall_reflection_diffusion_vector = density_rate_off_diagonal;\n#endif\n" + FUSED_AFTER_LOOP)],
    "pair_geometry": [
        ("""                            wall_crossing_pair_geometry(self_position, neighbor_position, pair_wall_normal,
                                                        pair_depth_ratio, pair_mirror_displacement);""",
         """                            pair_wall_normal = vec3(0.0, 1.0, 0.0); pair_depth_ratio = 0.5;
                            pair_mirror_displacement = vec3(0.0, 0.01, 0.0);""")],
    "epilogue": [
        ("""        mat3 fluid_correction_inverse;
        bool fluid_inverse_valid =
            wall_crossing_invert_fluid_matrix(fluid_correction_matrix, fluid_correction_inverse);""",
         """        mat3 fluid_correction_inverse = fluid_correction_matrix;
        bool fluid_inverse_valid = false;
        density_rate += dot(wall_diffusion_vector + wall_reflection_diffusion_vector, density_gradient);"""),
    ],
}
FORCE_PROBES = {
    "wall_gradient": [
        ("                        wall_kernel_gradient_sum += neighbor_volume_dynamic * kernel_gradient_raw;\n", ""),
        (FORCE_AFTER_LOOP, "#ifdef WALL_CROSSING\n    wall_kernel_gradient_sum = a_viscosity;\n#endif\n" + FORCE_AFTER_LOOP)],
    "pressure_gradient": [
        ("""                    if (WALL_CROSSING_ACTIVE) {
                        fluid_pressure_gradient_sum += neighbor_volume_dynamic
                                                     * (neighbor_pressure - self_pressure)
                                                     * kernel_gradient_raw;
                    }
""", ""),
        (FORCE_AFTER_LOOP, "#ifdef WALL_CROSSING\n    fluid_pressure_gradient_sum = particle_shift_main;\n#endif\n"
                           + FORCE_AFTER_LOOP)],
    "mirror_matrix": [
        ("""                        if (WALL_CROSSING_PRESSURE_REFLECTION) {
                            wall_mirror_matrix += neighbor_volume_dynamic
                                                * outerProduct(kernel_gradient_raw, pair_mirror_displacement);
                        }
""", ""),
        (FORCE_AFTER_LOOP, "#ifdef WALL_CROSSING\n    wall_mirror_matrix = mat3(a_viscosity, a_pressure, particle_shift_anti);\n"
                           "#endif\n" + FORCE_AFTER_LOOP)],
    "pair_geometry": [
        ("""                        wall_crossing_pair_geometry(self_position, neighbor_position, pair_wall_normal,
                                                    pair_depth_ratio, pair_mirror_displacement);""",
         """                        pair_wall_normal = vec3(0.0, 1.0, 0.0); pair_depth_ratio = 0.5;
                        pair_mirror_displacement = vec3(0.0, 0.01, 0.0);""")],
    "viscous_factor": [
        ("""                    if (neighbor_is_wall) {
                        if (WALL_CROSSING_VISCOUS_QUADRATIC && self_velocity_gradient_valid) {""",
         """                    if (false) {
                        if (WALL_CROSSING_VISCOUS_QUADRATIC && self_velocity_gradient_valid) {""")],
    "epilogue": [
        ("""        if (fluid_inverse_xz_and_yz.w != 0.0) {
            mat3 fluid_correction_inverse =""",
         """        wall_pressure_sum += fluid_pressure_gradient_sum + wall_mirror_matrix * wall_kernel_gradient_sum;
        if (false) {
            mat3 fluid_correction_inverse =""")],
}


def patched(text: str, replacements: list) -> str:
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"probe text found {text.count(old)} times: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def probe_runs() -> list:
    """(name, fused probe names, force probe names): one block of each per run, then each 'all' alone."""
    fused, force = list(FUSED_PROBES), list(FORCE_PROBES)
    runs = [("baseline", [], [])]
    for index in range(max(len(fused), len(force))):
        runs.append((f"probe_{index}", fused[index:index + 1], force[index:index + 1]))
    runs.append(("all", fused, force))
    return runs


def build(work: pathlib.Path, fused_names: list, force_names: list) -> pathlib.Path:
    from experiment.v7.compile_shaders_v7 import glslc_command
    sources = work / "shaders"
    if sources.exists():
        shutil.rmtree(sources)
    shutil.copytree(SHADERS, sources, ignore=shutil.ignore_patterns("spv", "render"))
    spv = work / "spv"
    if spv.exists():
        shutil.rmtree(spv)
    shutil.copytree(SHADERS / "spv", spv, ignore=shutil.ignore_patterns("render"))
    for source_name, names, probes, output in (
            ("correction_density.comp", fused_names, FUSED_PROBES, "correction_density_wall_crossing_near_wall"),
            ("force.comp", force_names, FORCE_PROBES, "force_wall_crossing_near_wall")):
        text = (sources / source_name).read_text(encoding="utf-8")
        for name in names:
            text = patched(text, probes[name])
        (sources / source_name).write_text(text, encoding="utf-8", newline="\n")
        command = glslc_command(str(sources / source_name), str(spv / f"{output}.comp.spv"),
                                ("WALL_CROSSING", "WALL_CROSSING_NEAR_WALL"))
        command[command.index("-I") + 1] = str(sources)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"glslc failed for {output} {names}: {result.stderr[:2000]}")
    return spv


def registers(spv: pathlib.Path, case: str, device: int, out: pathlib.Path) -> dict:
    environment = dict(os.environ, V7_SPV_DIR=str(spv), OPENBLAS_NUM_THREADS="1",
                       VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
    subprocess.run([sys.executable, str(REPOSITORY_ROOT / "experiment/wall_crossing/pipeline_statistics.py"),
                    "--case", case, "--device", str(device), "--filter",
                    "correction_density_all_split_near_wall,force_all_split_near_wall", "--out", str(out)],
                   cwd=REPOSITORY_ROOT, env=environment, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    pipelines = json.loads(out.read_text(encoding="utf-8"))["pipelines"]
    result = {}
    for name, key in (("correction_density_all_split_near_wall", "fused"), ("force_all_split_near_wall", "force")):
        entry = pipelines[name]
        entry = entry[0] if isinstance(entry, list) else entry
        statistics = entry.get("statistics", entry)
        result[key] = {"registers": int(statistics["Register Count"]),
                       "shared_memory": int(statistics.get("Shared Memory Size", 0))}
    return result


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1)
    arguments = parser.parse_args()
    out = LOG_ROOT / "section27" / "register_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with tempfile.TemporaryDirectory(prefix="register_probe_") as temporary:
        work = pathlib.Path(temporary)
        for name, fused_names, force_names in probe_runs():
            spv = build(work, fused_names, force_names)
            summary[name] = {"fused_probes": fused_names, "force_probes": force_names}
            for dimension, case in CASES:
                summary[name][dimension] = registers(spv, case, arguments.device, out / f"{name}_{dimension}.json")
            print(name, json.dumps({key: value for key, value in summary[name].items()}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    rows = []
    baseline = summary["baseline"]
    for kernel, text in (("fused", "fused"), ("force", "受力")):
        probes = FUSED_PROBES if kernel == "fused" else FORCE_PROBES
        for probe in ["(全部保留)"] + list(probes) + ["all"]:
            run = ("baseline" if probe == "(全部保留)" else
                   "all" if probe == "all" else next(name for name, entry in summary.items()
                                                    if probe in entry[f"{kernel}_probes"] and name not in ("all",)))
            cells = [text, probe]
            for dimension, _ in CASES:
                count = summary[run][dimension][kernel]["registers"]
                change = count - baseline[dimension][kernel]["registers"]
                cells.append(f"{count}（{change:+d}）· {100 * theoretical_occupancy(count):.1f} %")
            rows.append(cells)
    write("register_probe.md", markdown_table(
        ["遍", "去掉的块", "2-D：寄存器（变化）· 理论占用率", "3-D：寄存器（变化）· 理论占用率"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
