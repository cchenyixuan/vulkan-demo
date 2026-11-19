"""
grid_offset_check.py - report section 21: where the voxel boundaries sit against the particle layers of a case, and
the grid origin shift case_loader_v7 applies (case_loader_v7._place_grid_origin).

    .venv/Scripts/python.exe -m experiment.wall_crossing.grid_offset_check [CASE.yaml ...]
    .venv/Scripts/python.exe -m experiment.wall_crossing.grid_offset_check --tables

Loads each case (default: the three of section 21) and writes claude/wall_crossing/tables/grid_offset_cases.md and
claude/wall_crossing/grid_offset/cases.json: per axis the smallest distance of an initial particle (fluid and walls)
to a voxel boundary before and after, in particle spacings, the shift, the voxels added, and the origin and voxel
counts before and after.

--tables: the tables of the section's runs (logs/wall_crossing/grid_offset/): grid_offset_watch.md (test 3, the
48^3 box with the watch on the new and the old grid: event, initialization_seam_clamp_count, step-1 and total
migrations, overflow counters) and grid_offset_n250.md (2-D 250^2 on the new grid against itself and the old grid).
"""
from __future__ import annotations

import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.report_tables import REPORT_ROOT, markdown_table, write     # noqa: E402

CASES = ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml",
         "cases/cavity3d_1m_crossing/case.yaml",
         "cases/wall_crossing/hydrostatic_3d_n48_gtilt_crossing/case.yaml")


RUN_ROOT = REPOSITORY_ROOT / "logs" / "wall_crossing" / "grid_offset"
SWITCH_ROOT = REPOSITORY_ROOT / "logs" / "wall_crossing" / "particle_watch" / "switch_check"
OVERFLOW_KEYS = ("overflow_ghost_count", "overflow_inside_count", "overflow_incoming_count", "overflow_install_tail",
                 "overflow_install_inside", "overflow_departed_count", "far_migration_count", "stamp_error_count")


def run_records() -> dict:
    records = {}
    for line in (REPOSITORY_ROOT / "logs" / "wall_crossing" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        output = str(record.get("output", ""))
        if "grid_offset" in output:
            records[pathlib.Path(output).stem] = record
    return records


def watch_run_table() -> None:
    """Test 3: the 48^3 gtilt box, K = 2, --watch, 3-D default pools, on the new and the old grid."""
    import numpy as np
    records = run_records()
    rows = []
    for name, label in (("test3_new_grid", "新网格（原点挪 0.5 dx）"), ("test3_old_grid", "旧网格（同样设置，我加的对照）")):
        steps_path = RUN_ROOT / name / "watch_steps.jsonl"
        if not steps_path.exists():
            continue
        steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
        event_path = RUN_ROOT / name / "event.json"
        if event_path.exists():
            event = json.loads(event_path.read_text(encoding="utf-8"))
            status = event["raw_files"][-1]["status"]
            value = event["events"]["wall_layer"] if "wall_layer" in event["events"] else None
            stopped = (f"第 {event['event_step']} 步：{value['count']} 个流体粒子越过盒面" if value
                       else f"第 {event['event_step']} 步：" + "、".join(event["events"]))
        else:
            with np.load(RUN_ROOT / f"{name}.npz") as archive:
                status = json.loads(str(archive["meta"]))["status"]
            stopped = f"跑完 {steps[-1]['step']} 步，没有事件"
        def per_slab(values):
            return " / ".join(str(value) for value in values)
        overflow = "；".join(f"{key} {per_slab(int(slab.get(key, 0)) for slab in status)}" for key in OVERFLOW_KEYS
                            if any(int(slab.get(key, 0)) for slab in status)) or "全为 0"
        rows.append([label, stopped,
                     per_slab(int(slab.get("initialization_seam_clamp_count", 0)) for slab in status),
                     per_slab(slab["delta"]["migration_install_count"] for slab in steps[1]["slabs"]),
                     per_slab(sum(record["slabs"][index]["delta"]["migration_install_count"] for record in steps)
                              for index in range(len(steps[0]["slabs"]))),
                     overflow,
                     per_slab(max(record["slabs"][index]["own_alive_unlisted"] for record in steps)
                              for index in range(len(steps[0]["slabs"]))),
                     f"{max(record['largest_fluid_speed'] for record in steps):.3f}",
                     f"{max(record['largest_fluid_displacement_since_previous_check']['value'] for record in steps[1:]):.1e}",
                     records[name]["run_id"] if name in records else "—"])
    write("grid_offset_watch.md", markdown_table(
        ["网格", "结果", "initialization_seam_clamp_count（卡 0 / 1）", "第 1 步迁入（卡 0 / 1）", "全程迁入",
         "overflow 计数（到最后一步，卡 0 / 1）", "最多时不在任何体素表里的自有粒子", "最大速度", "最大单步位移（m）",
         "记录"], rows))


def two_dimensional_table() -> None:
    """In place of test 2 (the 2-D case moved): the new grid run twice, and against the old grid, next to the old
    grid's own K = 1 / K = 2 difference."""
    import numpy as np
    pairs = (("新网格 K = 1 对 新网格 K = 1（跑两次）", RUN_ROOT / "n250_k1_new_r1.npz", RUN_ROOT / "n250_k1_new_r2.npz"),
             ("新网格 K = 1 对 旧网格 K = 1", RUN_ROOT / "n250_k1_new_r1.npz", SWITCH_ROOT / "n250_k1_reference_r1.npz"),
             ("新网格 K = 2 对 旧网格 K = 2", RUN_ROOT / "n250_k2_new_r1.npz", SWITCH_ROOT / "n250_k2_reference_r1.npz"),
             ("旧网格 K = 1 对 旧网格 K = 2（尺度）", SWITCH_ROOT / "n250_k1_reference_r1.npz",
              SWITCH_ROOT / "n250_k2_reference_r1.npz"),
             ("新网格 K = 1 对 新网格 K = 2", RUN_ROOT / "n250_k1_new_r1.npz", RUN_ROOT / "n250_k2_new_r1.npz"))
    rows = []
    for label, first, second in pairs:
        if not (first.exists() and second.exists()):
            continue
        with np.load(first) as a, np.load(second) as b:
            meta_a, meta_b = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
            cells = []
            identical = True
            for field, columns in (("position_voxel_id", slice(0, 3)), ("velocity_mass", slice(0, 3)),
                                   ("density_pressure", slice(0, 1))):
                x, y = a[field][:, columns], b[field][:, columns]
                identical &= bool((x.view(np.uint32) == y.view(np.uint32)).all())
                difference = np.abs(x.astype(np.float64) - y.astype(np.float64))
                cells.append(f"{np.sqrt((difference ** 2).mean()):.1e} / {difference.max():.1e}")
            rows.append([label, f"{meta_a['alive']} / {meta_b['alive']}",
                         "位置、速度、密度逐位相同" if identical else "不逐位相同", *cells,
                         f"{meta_a['repo_head']} / {meta_b['repo_head']}"])
    write("grid_offset_n250.md", markdown_table(
        ["比较（2-D 250² crossing，700 步，canonical）", "存活", "结果", "位置（x, y, z）差 均方根 / 最大（m）",
         "速度差 均方根 / 最大（m/s）", "密度差 均方根 / 最大（kg/m³）", "提交"], rows))


def main() -> int:
    if sys.argv[1:] == ["--tables"]:
        watch_run_table()
        two_dimensional_table()
        return 0
    from experiment.v7.utils import case_loader_v7
    cases = sys.argv[1:] or list(CASES)
    rows, records = [], []
    for case in cases:
        loaded = case_loader_v7.load_case_v7(REPOSITORY_ROOT / case)
        placement = json.loads(json.dumps(case_loader_v7.LAST_GRID_PLACEMENT))
        placement["case"] = case
        placement["particles"] = int(loaded.initial.positions.shape[0])
        records.append(placement)
        for axis, item in placement["axes"].items():
            index = "xyz".index(axis)
            rows.append([pathlib.Path(case).parent.name, axis, f"{item['clearance_before']:.4f}",
                         f"{item['clearance_after']:.4f}", f"{item['shift']:.4f}", str(item["voxels_added"]),
                         f"{placement['origin_before'][index]:.6f} → {placement['origin_after'][index]:.6f}",
                         f"{placement['dims_before'][index]} → {placement['dims_after'][index]}",
                         f"{placement['smoothing_length'] / placement['particle_spacing']:.2f}"])
    write("grid_offset_cases.md", markdown_table(
        ["算例", "方向", "改前：离体素边界最近（dx）", "改后（dx）", "δ（dx，向负方向）", "加的体素", "原点（m）",
         "体素数", "h / dx"], rows))
    target = REPORT_ROOT / "grid_offset"
    target.mkdir(parents=True, exist_ok=True)
    (target / "cases.json").write_text(json.dumps(records, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
