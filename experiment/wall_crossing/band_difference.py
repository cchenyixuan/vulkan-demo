"""
band_difference.py - report section 23, 2.5: where does numerics.wall_crossing.split_dispatch with phase C band mode 1
(the near-wall build alone over the band) first differ from the switch off on two slabs?

    .venv/Scripts/python.exe -m experiment.wall_crossing.band_difference first-step --name NAME DIGESTS_A DIGESTS_B
        two files of canonical_dump.py --digest-steps (a sha256 per field after every step): per field the first step
        whose digests differ -> claude/wall_crossing/tables/band_difference_steps_<NAME>.md
    .venv/Scripts/python.exe -m experiment.wall_crossing.band_difference rows --name NAME --case CASE A.npz B.npz
        two canonical dumps of the same step: per field the rows that are not bit-identical, and for them the material
        (fluid / wall), the near-wall class of split_dispatch.py (global grid, float32 binning), the voxel column
        relative to the cut (column - cut column: 0 is the first column of slab 1, -1 the last of slab 0), and the
        largest absolute / relative difference -> claude/wall_crossing/tables/band_difference_rows_<NAME>.md and
        logs/wall_crossing/band_difference/<NAME>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

FIELD_LABELS = {"position_voxel_id": "位置", "velocity_mass": "速度与质量", "density_pressure": "密度与压力",
                "density_pressure_scratch": "密度与压力（暂存）", "acceleration": "加速度", "shift": "位移修正",
                "material": "材料", "correction_inverse": "修正矩阵（L）",
                "density_gradient_kernel_sum": "密度梯度与核和", "extension_fields": "扩展字段（全局编号）",
                "global_id": "全局编号"}


def first_step(arguments) -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    records = []
    for path in (arguments.first, arguments.second):
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        records.append({record["step"]: record for record in (json.loads(line) for line in lines if line.strip())})
    common = sorted(set(records[0]) & set(records[1]))
    fields = sorted(records[0][common[0]]["digests"])
    rows, result = [], {}
    for field in fields:
        differing = [step for step in common
                     if records[0][step]["digests"][field] != records[1][step]["digests"][field]]
        result[field] = differing[0] if differing else None
        rows.append([FIELD_LABELS.get(field, field), f"`{field}`",
                     str(differing[0]) if differing else f"无（到第 {common[-1]} 步）",
                     str(len(differing))])
    alive = [(step, records[0][step]["alive"], records[1][step]["alive"]) for step in common
             if records[0][step]["alive"] != records[1][step]["alive"]]
    write(f"band_difference_steps_{arguments.name}.md", markdown_table(
        ["量", "字段", "第一个不同的步", "不同的步数"], rows)
        + f"\n比较的步：{common[0]} – {common[-1]}（{len(common)} 个）；活着的粒子数不同的步：{len(alive)}。\n")
    out = LOG_ROOT / "band_difference"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"steps_{arguments.name}.json").write_text(json.dumps({"first_step": result, "steps": [common[0], common[-1]]},
                                                                 indent=1), encoding="utf-8")
    print(json.dumps(result))
    return 0


def rows(arguments) -> int:
    import numpy as np
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing.report_tables import markdown_table, write
    from experiment.wall_crossing.split_dispatch import near_wall_rows
    case = load_case_v7(REPOSITORY_ROOT / arguments.case)
    with np.load(arguments.first) as first_archive, np.load(arguments.second) as second_archive:
        first = {name: first_archive[name] for name in first_archive.files if name != "meta"}
        second = {name: second_archive[name] for name in second_archive.files if name != "meta"}
        meta = json.loads(str(first_archive["meta"]))
        second_meta = json.loads(str(second_archive["meta"]))
    if not np.array_equal(first["global_id"], second["global_id"]):
        raise SystemExit("the two dumps hold different particles")
    near, fluid = near_wall_rows(first, case)
    support = np.float32(case.physics.smoothing_length)
    column = np.floor((first["position_voxel_id"][:, 0].astype(np.float32) - np.float32(case.grid.origin_x))
                      / support).astype(np.int64)
    cuts = meta.get("cuts") or []
    cut = int(cuts[0]) if cuts else 0
    offset = column - cut
    table_rows, record = [], {"first": str(arguments.first), "second": str(arguments.second), "steps": meta["steps"],
                              "second_steps": second_meta["steps"], "cut_column": cut, "fields": {}}
    union = np.zeros(first["global_id"].size, dtype=bool)
    for field in first:
        if field == "global_id" or field not in second or first[field].dtype.kind not in "fiu":
            continue
        a, b = first[field], second[field]
        if a.ndim == 1:
            a, b = a[:, None], b[:, None]
        if a.dtype.kind == "f":
            differs = (a.view(np.uint32) != b.view(np.uint32)).any(axis=1)
        else:
            differs = (a != b).any(axis=1)
        if not differs.any():
            record["fields"][field] = {"rows": 0}
            continue
        union |= differs
        difference = np.abs(a[differs].astype(np.float64) - b[differs].astype(np.float64))
        scale = np.maximum(np.abs(a[differs].astype(np.float64)), np.abs(b[differs].astype(np.float64)))
        relative = np.where(scale > 0, difference / np.where(scale > 0, scale, 1.0), 0.0)
        components = [int(index) for index in np.nonzero((a[differs] != b[differs]).any(axis=0))[0]]
        offsets = offset[differs]
        record["fields"][field] = {
            "rows": int(differs.sum()), "fluid": int((differs & fluid).sum()), "wall": int((differs & ~fluid).sum()),
            "near_wall": int((differs & near).sum()), "not_near_wall": int((differs & ~near).sum()),
            "column_offset_range": [int(offsets.min()), int(offsets.max())],
            "column_offset_counts": {str(int(value)): int((offsets == value).sum()) for value in np.unique(offsets)},
            "components": components, "max_abs_difference": float(difference.max()),
            "max_relative_difference": float(relative.max())}
        entry = record["fields"][field]
        table_rows.append([FIELD_LABELS.get(field, field), f"{entry['rows']:,}", f"{entry['fluid']:,} / {entry['wall']:,}",
                           f"{entry['near_wall']:,} / {entry['not_near_wall']:,}",
                           f"{entry['column_offset_range'][0]} … {entry['column_offset_range'][1]}",
                           ", ".join(f"{key}: {value:,}" for key, value in entry["column_offset_counts"].items()),
                           str(components), f"{entry['max_abs_difference']:.2e}",
                           f"{entry['max_relative_difference']:.2e}"])
    record["union_rows"] = int(union.sum())
    write(f"band_difference_rows_{arguments.name}.md", markdown_table(
        ["量", "不同的行", "流体 / 壁面", "近壁 / 非近壁", "列（相对分界）范围", "各列的行数", "不同的分量",
         "最大绝对差", "最大相对差"], table_rows or [["（没有不同的行）", "", "", "", "", "", "", "", ""]])
        + f"\n第 {meta['steps']} 步；分界在全局第 {cut} 列（列 0 = 卡 1 的第一列，列 −1 = 卡 0 的最后一列）；"
          f"至少一个量不同的行：{int(union.sum()):,}。\n")
    out = LOG_ROOT / "band_difference"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arguments.name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({field: value.get("rows") for field, value in record["fields"].items()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    steps = sub.add_parser("first-step")
    steps.add_argument("--name", required=True)
    steps.add_argument("first")
    steps.add_argument("second")
    row_parser = sub.add_parser("rows")
    row_parser.add_argument("--name", required=True)
    row_parser.add_argument("--case", required=True)
    row_parser.add_argument("first")
    row_parser.add_argument("second")
    arguments = parser.parse_args()
    return first_step(arguments) if arguments.command == "first-step" else rows(arguments)


if __name__ == "__main__":
    sys.exit(main())
