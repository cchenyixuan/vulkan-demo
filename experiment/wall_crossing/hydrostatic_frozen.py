"""
hydrostatic_frozen.py - the hydrostatic regression of the wall store (claude/wall_store_report.md): one frozen state of
the 2-D tilted-gravity box, evaluated once by the tag crossing-pool and once by this branch (frozen_evaluate.py, GPU),
and term by term on the CPU (reference.py, float64) in both forms of the wall terms:

  density rate  drift (continuity) fluid / wall, density diffusion fluid / wall, the wall part as the tag sums it after
                the loop (G . L sum ..., the mirror part without L) and pair by pair (psi_ij of the wall rule in the
                fluid pair's formula, this branch);
  acceleration  pressure fluid / wall (summed after the loop / pair by pair), viscous fluid / wall, gravity;
  shift         the total (the same formulas in both).

    .venv/Scripts/python.exe -m experiment.wall_crossing.hydrostatic_frozen --state <checkpoint.npz> \
        --tag <tag evaluation.npz> --new <new evaluation.npz> [--label g30]

Per layer of wall distance (phi / dx in [0, 1), [1, 2), [2, 3)): the rms of every term, the rms of the GPU difference
new - tag, and the rms of the CPU model's mismatch against each build's GPU (it must be at rounding level for the
decomposition to explain the difference). Writes claude/wall_store/tables/frozen_<label>.md and
logs/wall_store/frozen/<label>.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LAYERS = ((0, 1), (1, 2), (2, 3))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def main() -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing.reference import ReferenceConstants, evaluate_reference
    from experiment.wall_crossing.wall_store_check import markdown_table, write_table
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="cases/wall_crossing/hydrostatic_2d_n250_g30_crossing/case.yaml")
    parser.add_argument("--state", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--label", default="g30")
    parser.add_argument("--keep-all-walls-note", default="")
    arguments = parser.parse_args()
    case = load_case_v7(REPOSITORY_ROOT / arguments.case)
    constants = ReferenceConstants.from_case(case)
    with np.load(arguments.state) as archive:
        state = {name: archive[name] for name in archive.files if not name.startswith("_")}
        offset = float(archive["_stored_density_offset"])
    kinds = np.asarray([material.kind for material in case.materials])
    is_wall = kinds[state["material"].astype(np.int64)] == 1
    reference = evaluate_reference(state["position_voxel_id"][:, :3], state["velocity_mass"][:, :3],
                                   state["density_pressure"][:, 0].astype(np.float64) + offset,
                                   state["density_pressure"][:, 1], state["velocity_mass"][:, 3], is_wall, constants)
    tag = np.load(arguments.tag)
    new = np.load(arguments.new)
    rows = reference.fluid_index                                  # state rows of the fluid particles
    tag_rate = tag["density_rate"][rows]                          # the tag evaluated every row
    new_position_of_row = {int(row): index for index, row in enumerate(new["state_rows"])}
    new_index = np.array([new_position_of_row[int(row)] for row in rows])
    new_rate = new["density_rate"][new_index]
    tag_acceleration = tag["acceleration"][rows]
    new_acceleration = new["acceleration"][new_index]
    tag_shift, new_shift = tag["shift"][rows], new["shift"][new_index]
    spacing = 2.0 * constants.particle_radius
    layer_of = reference.wall_distance_raw / spacing
    gravity = np.asarray(constants.gravity)

    terms = {
        "drift_fluid": reference.density_rate_drift_fluid,
        "drift_wall": reference.density_rate_drift_wall,
        "diffusion_fluid": reference.density_rate_diffusion_fluid,
        "diffusion_wall_summed": reference.density_rate_diffusion_wall,
        "diffusion_wall_pairwise": reference.density_rate_diffusion_wall_pairwise,
        "pressure_fluid": reference.acceleration_pressure_fluid,
        "pressure_wall_summed": reference.acceleration_pressure - reference.acceleration_pressure_fluid,
        "pressure_wall_pairwise": reference.acceleration_pressure_pairwise - reference.acceleration_pressure_fluid,
        "viscous_fluid": reference.acceleration_viscosity - reference.acceleration_viscosity_wall,
        "viscous_wall": reference.acceleration_viscosity_wall,
    }
    model_rate_tag = (terms["drift_fluid"] + terms["drift_wall"] + terms["diffusion_fluid"]
                      + terms["diffusion_wall_summed"] + reference.density_rate_volume_constraint)
    model_rate_new = (terms["drift_fluid"] + terms["drift_wall"] + terms["diffusion_fluid"]
                      + terms["diffusion_wall_pairwise"] + reference.density_rate_volume_constraint)
    model_acceleration_tag = reference.acceleration
    model_acceleration_new = reference.acceleration_pressure_pairwise + reference.acceleration_viscosity + gravity
    result = {"label": arguments.label, "layers": {}}
    rows_table = []
    for low, high in LAYERS:
        chosen = (layer_of >= low) & (layer_of < high)
        entry = {"particles": int(chosen.sum())}
        for name, values in terms.items():
            entry[name] = rms(values[chosen])
        entry["gravity"] = float(np.linalg.norm(gravity))
        entry["gpu_rate_tag"], entry["gpu_rate_new"] = rms(tag_rate[chosen]), rms(new_rate[chosen])
        entry["gpu_rate_difference"] = rms((new_rate - tag_rate)[chosen])
        entry["model_rate_difference"] = rms((model_rate_new - model_rate_tag)[chosen])
        entry["model_mismatch_rate_tag"] = rms((model_rate_tag - tag_rate)[chosen])
        entry["model_mismatch_rate_new"] = rms((model_rate_new - new_rate)[chosen])
        entry["gpu_acceleration_difference"] = rms((new_acceleration - tag_acceleration)[chosen])
        entry["model_acceleration_difference"] = rms((model_acceleration_new - model_acceleration_tag)[chosen])
        entry["model_mismatch_acceleration_tag"] = rms((model_acceleration_tag - tag_acceleration)[chosen])
        entry["model_mismatch_acceleration_new"] = rms((model_acceleration_new - new_acceleration)[chosen])
        entry["gpu_shift_difference"] = rms((new_shift - tag_shift)[chosen])
        entry["shift"] = rms(tag_shift[chosen])
        result["layers"][f"{low}-{high}"] = entry
        rows_table.append([f"{low}–{high} Δx（{entry['particles']:,}）",
                           f"{entry['drift_fluid']:.3f}", f"{entry['drift_wall']:.3f}",
                           f"{entry['diffusion_fluid']:.3f}", f"{entry['diffusion_wall_summed']:.3f}",
                           f"{entry['diffusion_wall_pairwise']:.3f}",
                           f"{entry['gpu_rate_tag']:.3f} / {entry['gpu_rate_new']:.3f}",
                           f"{entry['gpu_rate_difference']:.3f}（模型 {entry['model_rate_difference']:.3f}）",
                           f"{entry['model_mismatch_rate_tag']:.1e} / {entry['model_mismatch_rate_new']:.1e}"])
    acceleration_rows = []
    for low, high in LAYERS:
        entry = result["layers"][f"{low}-{high}"]
        acceleration_rows.append([f"{low}–{high} Δx", f"{entry['pressure_fluid']:.4f}",
                                  f"{entry['pressure_wall_summed']:.4f}", f"{entry['pressure_wall_pairwise']:.4f}",
                                  f"{entry['viscous_fluid']:.2e}", f"{entry['viscous_wall']:.2e}",
                                  f"{entry['gravity']:.1f}",
                                  f"{entry['gpu_acceleration_difference']:.2e}（模型 {entry['model_acceleration_difference']:.2e}）",
                                  f"{entry['model_mismatch_acceleration_tag']:.1e} / {entry['model_mismatch_acceleration_new']:.1e}",
                                  f"{entry['shift']:.2e} / {entry['gpu_shift_difference']:.1e}"])
    table = (markdown_table(["离墙（流体粒子数）", "连续性 流体", "连续性 壁面", "密度扩散 流体",
                             "密度扩散 壁面（标签：循环后合成）", "密度扩散 壁面（新：逐对）",
                             "GPU dρ/dt 标签 / 新", "GPU 差 新 − 标签（CPU 模型的差）", "CPU 模型对 GPU 的偏差 标签 / 新"],
                            rows_table)
             + "\n" + markdown_table(["离墙", "压力 流体", "压力 壁面（标签：合成）", "压力 壁面（新：逐对）", "粘性 流体",
                                      "粘性 壁面", "重力", "GPU 加速度差 新 − 标签（模型）",
                                      "CPU 模型对 GPU 的偏差 标签 / 新", "位移修正 / 两版之差"], acceleration_rows))
    write_table(f"frozen_{arguments.label}.md", table)
    out = REPOSITORY_ROOT / "logs" / "wall_store" / "frozen" / f"{arguments.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
