"""
density_rounding.py - how much of one step's density increment does the stored float32 density take?

    .venv/Scripts/python.exe -m experiment.wall_crossing.density_rounding

For the final state of a box run (final.npz: the fluid; the walls are the case's, they do not move): the density
rate of every fluid particle from the numpy reference (float64; a simple case through the settings under which the
reference's density rate is simple's), then the solver's update emulated on the host,
    rho_new = float32(rho + dt x rate),
against dt x rate itself. The stored density is a float32 near 1000 kg/m^3, whose grid there is 2^-14 = 6.1e-5
kg/m^3: an increment below half a grid step leaves the density unchanged.

It is one instant (the state after the last step of the run), so the field means it prints fluctuate with the
acoustics; the share of particles below half a grid step is what it is for. The time-integrated effect of the
rounding is measured by the density budget (density_budget.py: applied change - the sum of the four terms).

Output: claude/wall_crossing/tables/density_rounding.md, claude/wall_crossing/density_budget/rounding.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.reference import ReferenceConstants, evaluate_reference        # noqa: E402
from experiment.wall_crossing.report_tables import markdown_table, run_records                # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                                         # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
RUNS = (("budget_n250_crossing", "crossing（缺省）", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml"),
        ("budget_n250_simple", "simple", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml"))


def probe(name: str, case_path: str) -> dict:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    case = load_case_v7(REPOSITORY_ROOT / case_path)
    final = np.load(LOG_ROOT / "box_run" / name / "final.npz")
    kinds = np.asarray([material.kind for material in case.materials])
    group = np.asarray(case.initial.material_group)
    is_wall_initial = kinds[group] == 1
    wall_positions = np.asarray(case.initial.positions)[is_wall_initial].astype(np.float64)
    wall_velocities = np.asarray(case.initial.velocities)[is_wall_initial].astype(np.float64)
    material_mass = np.asarray([np.float32(material.rest_density * material.volume) for material in case.materials],
                               dtype=np.float64)
    rest_density = float(next(material for material in case.materials if material.kind == 0).rest_density)
    fluid_count = final["positions"].shape[0]
    wall_count = wall_positions.shape[0]
    positions = np.concatenate([final["positions"].astype(np.float64), wall_positions])
    velocities = np.concatenate([final["velocities"].astype(np.float64), wall_velocities])
    densities = np.concatenate([final["densities"].astype(np.float64), np.full(wall_count, rest_density)])
    pressures = np.concatenate([final["pressures"].astype(np.float64), np.zeros(wall_count)])
    masses = np.concatenate([np.full(fluid_count, material_mass[group[~is_wall_initial][0]]),
                             material_mass[group[is_wall_initial]]])
    is_wall = np.concatenate([np.zeros(fluid_count, dtype=bool), np.ones(wall_count, dtype=bool)])
    constants = ReferenceConstants.from_case(case)
    rate = evaluate_reference(positions, velocities, densities, pressures, masses, is_wall, constants).density_rate
    timestep = constants.timestep
    density32 = final["densities"].astype(np.float32)
    increment = timestep * rate
    applied = (density32.astype(np.float64) + increment).astype(np.float32).astype(np.float64) \
        - density32.astype(np.float64)
    grid = np.spacing(density32).astype(np.float64)
    ratio = np.abs(increment) / grid
    return {
        "name": name, "fluid_particles": int(fluid_count), "timestep": timestep,
        "grid_step": float(grid.max()), "rate_of_half_a_grid_step": float(0.5 * grid.max() / timestep),
        "rate_root_mean_square": float(np.sqrt((rate ** 2).mean())), "rate_median_absolute": float(np.median(np.abs(rate))),
        "below_half_grid_step_fraction": float((ratio < 0.5).mean()),
        "below_one_grid_step_fraction": float((ratio < 1.0).mean()),
        "above_two_grid_steps_fraction": float((ratio > 2.0).mean()),
        "mean_rate": float(rate.mean()), "mean_applied_rate": float(applied.mean() / timestep),
        "mean_rounding_rate": float((applied - increment).mean() / timestep),
        "mean_rounding_rate_standard_error": float((applied - increment).std() / np.sqrt(fluid_count) / timestep)}


def main() -> int:
    records = {record["name"]: record for record in run_records()
               if record.get("kind") == "box_run" and record.get("name")}
    results = {name: probe(name, case_path) for name, _, case_path in RUNS}
    labels = [text for _, text, _ in RUNS]
    names = [name for name, _, _ in RUNS]

    def row(label: str, key: str, form: str, scale: float = 1.0) -> list:
        return [label] + [form.format(scale * results[name][key]) for name in names]

    rows = [
        row("float32 密度在 1000 kg/m³ 处的网格（kg/m³）", "grid_step", "{:.3e}"),
        row("对应半格的变化率 = 网格 / (2 dt)（kg/m³ 每秒）", "rate_of_half_a_grid_step", "{:.3f}"),
        row("流体粒子密度变化率的 rms（kg/m³ 每秒）", "rate_root_mean_square", "{:.3f}"),
        row("流体粒子密度变化率绝对值的中位数", "rate_median_absolute", "{:.3f}"),
        row("单步增量不到半格（这一步密度不变）的流体粒子", "below_half_grid_step_fraction", "{:.1f} %", 100.0),
        row("单步增量不到一格的流体粒子", "below_one_grid_step_fraction", "{:.1f} %", 100.0),
        row("单步增量超过两格的流体粒子", "above_two_grid_steps_fraction", "{:.2f} %", 100.0),
        ["这一时刻全场平均：变化率 / 写入 / 舍入（kg/m³ 每时间单位）"]
        + [f"{results[name]['mean_rate']:+.4f} / {results[name]['mean_applied_rate']:+.4f} / "
           f"{results[name]['mean_rounding_rate']:+.4f} ± {results[name]['mean_rounding_rate_standard_error']:.4f}"
           for name in names],
        ["状态来自的运行（提交）"] + [f"{records.get(name, {}).get('run_id', '?')}（{records.get(name, {}).get('commit', '?')}）"
                             for name in names],
    ]
    text = ("收支运行的末态（t = 5.01）上，用 numpy 参考解算出每个流体粒子的密度变化率，在主机上仿照求解器的更新 "
            "ρ ← float32(ρ + dt × 变化率)。只是一个时刻，全场平均值随声波起伏；舍入在时间上的累计效果以收支表里的"
            "“float32 存储舍入”一行为准。\n\n"
            + markdown_table(["量", *labels], rows))
    (REPORT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "tables" / "density_rounding.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    (REPORT_ROOT / "density_budget").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "density_budget" / "rounding.json").write_text(json.dumps(results, indent=1), encoding="utf-8",
                                                                  newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
