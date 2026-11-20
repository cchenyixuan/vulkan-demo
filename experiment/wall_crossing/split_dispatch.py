"""
split_dispatch.py - report section 22 A: numerics.wall_crossing.split_dispatch (off by default then; on by default since report section 25). The fused pass and
force of a crossing slab as two dispatches each: the plain code for the particles of the voxels with no wall particle
in their 3^d neighbourhood, the crossing code for the others (shaders/wall_crossing_split.glsl).

    .venv/Scripts/python.exe -m experiment.wall_crossing.split_dispatch make-cases
        writes cases/<case>_split/case.yaml for the cases of SPLIT_CASES: the case with
        numerics.wall_crossing.split_dispatch: true, everything else the same (the particle files referenced)
    .venv/Scripts/python.exe -m experiment.wall_crossing.split_dispatch near-wall --device 1
        per case: a one-slab bootstrap, then the near-wall flags against the particles
        (SphSimulatorV7.readback_wall_crossing_near_wall) -> logs/wall_crossing/split_dispatch/near_wall.json and
        claude/wall_crossing/tables/split_near_wall.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

SPLIT_CASES = ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
               "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing",     # report section 23
               "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing",
               "cases/cavity3d_1m_crossing")


def split_case_path(case_directory: str) -> pathlib.Path:
    return REPOSITORY_ROOT / f"{case_directory}_split" / "case.yaml"


def make_cases() -> int:
    for case_directory in SPLIT_CASES:
        source = (REPOSITORY_ROOT / case_directory / "case.yaml").read_text(encoding="utf-8")
        # numerics.wall_crossing is a mapping of the numerics section (two-space indent); add the key under it
        match = re.search(r"^  wall_crossing:[^\n]*\n((?:    [^\n]*\n)*)", source, flags=re.M)
        if match is None:
            raise SystemExit(f"{case_directory}: no numerics.wall_crossing mapping")
        if "split_dispatch" in match.group(0):
            raise SystemExit(f"{case_directory}: already has split_dispatch")
        insert_at = match.end()
        text = (f"# VARIANT of {case_directory}/case.yaml (report section 22, 2026-11-19): "
                "numerics.wall_crossing.split_dispatch true; everything else identical.\n"
                + source[:insert_at]
                + "    split_dispatch: true      # the interior particles in a build without the mode's code\n"
                + source[insert_at:])
        target = split_case_path(case_directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        from experiment.v7.utils.case_loader_v7 import load_case_v7
        loaded = load_case_v7(target)
        assert loaded.numerics.wall_crossing.split_dispatch
        print(f"[split_dispatch] wrote {target.relative_to(REPOSITORY_ROOT)}")
    return 0


def near_wall(arguments) -> int:
    import os
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    from experiment.wall_crossing.report_tables import markdown_table, write
    rows, records = [], []
    for case_directory in SPLIT_CASES:
        case_path = split_case_path(case_directory)
        with LoggedRun(kind="split_near_wall", case=str(case_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                       gpu_uuids=[arguments.uuid] if arguments.uuid else []) as run:
            case = load_case_v7(case_path)
            context = VulkanContextV7.create(device_index=arguments.device, enable_validation=False,
                                             application_name="split_near_wall")
            simulator = SphSimulatorV7(context, case)
            simulator.bootstrap()
            statistics = simulator.readback_wall_crossing_near_wall()
            simulator.destroy()
            context.destroy()
            statistics.update({"case": str(case_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                               "run_id": run.run_id})
            run.extra["near_wall"] = statistics
        records.append(statistics)
        rows.append([pathlib.Path(case_directory).name, f"{statistics['fluid']:,}", f"{statistics['fluid_near_wall']:,}",
                     f"{100.0 * statistics['fluid_near_wall'] / statistics['fluid']:.2f} %",
                     f"{statistics['voxels_near_wall']:,} / {statistics['voxels']:,}",
                     str(statistics["wall_not_near_wall"]), statistics["run_id"]])
    out = LOG_ROOT / "split_dispatch"
    out.mkdir(parents=True, exist_ok=True)
    (out / "near_wall.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    write("split_near_wall.md", markdown_table(
        ["算例", "流体粒子", "其中在近壁体素里", "比例", "近壁体素 / 全部体素", "不在近壁体素里的壁面粒子（应为 0）", "记录"],
        rows))
    return 0


EQUIVALENCE_ROOT = LOG_ROOT / "split_dispatch" / "equivalence"
COMPARED_FIELDS = (("position_voxel_id", slice(0, 3)), ("velocity_mass", slice(0, 3)), ("density_pressure", slice(0, 2)),
                   ("acceleration", slice(0, 3)), ("shift", slice(0, 3)), ("correction_inverse", slice(0, 8)),
                   ("density_gradient_kernel_sum", slice(0, 4)))
SHORT_NAMES = {"position_voxel_id": "位置", "velocity_mass": "速度", "density_pressure": "密度压力",
               "acceleration": "加速度", "shift": "位移修正", "correction_inverse": "校正矩阵",
               "density_gradient_kernel_sum": "密度梯度与核和"}


def near_wall_rows(dump: dict, case) -> "np.ndarray":
    """Per row of a canonical dump (any number of slabs): does its voxel of the global grid (float32 binning of the
    position, as the kernels bin) lie in the 3^d neighbourhood of a voxel holding a wall particle (the marker's
    rule). A K = 2 dump stores slab-local voxel ids, so the position is used, not the stored id."""
    import numpy as np
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY
    support = np.float32(case.physics.smoothing_length)
    origin = np.array([case.grid.origin_x, case.grid.origin_y, case.grid.origin_z], dtype=np.float32)
    coordinates = np.floor((dump["position_voxel_id"][:, :3].astype(np.float32) - origin) / support).astype(np.int64)
    if int(case.physics.dimension) == 2:
        coordinates[:, 2] = 0
    kinds = np.asarray([material.kind for material in case.materials])[dump["material"].astype(np.int64)]
    walls = {tuple(row) for row in coordinates[kinds == KIND_BOUNDARY]}
    reach = (-1, 0, 1)
    z_reach = reach if int(case.physics.dimension) == 3 else (0,)
    near = {(x + dx, y + dy, z + dz) for (x, y, z) in walls for dx in reach for dy in reach for dz in z_reach}
    return np.array([tuple(row) in near for row in coordinates], dtype=bool), kinds != KIND_BOUNDARY


def compare(arguments) -> int:
    """Report section 22, check 3.5: per particle class (interior fluid / near-wall fluid), the rows bit-identical and
    the largest |difference| of every field between two dumps of the same steps."""
    import numpy as np
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing.report_tables import markdown_table, write
    case = load_case_v7(REPOSITORY_ROOT / arguments.case)
    rows, records = [], []
    for label, first_name, second_name in arguments.pair:
        first_path, second_path = EQUIVALENCE_ROOT / f"{first_name}.npz", EQUIVALENCE_ROOT / f"{second_name}.npz"
        with np.load(first_path) as first_archive, np.load(second_path) as second_archive:
            first = {name: first_archive[name] for name in first_archive.files if name != "meta"}
            second = {name: second_archive[name] for name in second_archive.files if name != "meta"}
            meta = json.loads(str(first_archive["meta"]))
        if not np.array_equal(first["global_id"], second["global_id"]):
            raise SystemExit(f"{label}: the two dumps hold different particles")
        near, fluid = near_wall_rows(first, case)
        record = {"label": label, "first": first_name, "second": second_name, "steps": meta["steps"], "classes": {}}
        for class_name, mask in (("interior", fluid & ~near), ("near_wall", fluid & near)):
            fields = {}
            for field, columns in COMPARED_FIELDS:
                a, b = first[field][mask][:, columns], second[field][mask][:, columns]
                equal = (a.view(np.uint32) == b.view(np.uint32)).all(axis=1)
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                fields[field] = {"identical_rows": int(equal.sum()), "rows": int(mask.sum()),
                                 "max_abs_difference": float(difference.max()) if difference.size else 0.0}
            record["classes"][class_name] = fields
            summary = "；".join(
                f"{SHORT_NAMES[field]} {value['identical_rows']}/{value['rows']} {value['max_abs_difference']:.1e}"
                for field, value in fields.items())
            rows.append([label, str(meta["steps"]), {"interior": "内部流体", "near_wall": "近壁流体"}[class_name],
                         str(int(mask.sum())), summary])
        records.append(record)
    (EQUIVALENCE_ROOT / f"{arguments.name}.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    write(f"split_equivalence_{arguments.name}.md", markdown_table(
        ["比较", "步数", "粒子", "个数", "各量：逐位相同的行 / 行数，最大差（位置、速度、密度压力、加速度、位移修正、校正矩阵、密度梯度与核和）"],
        rows))
    return 0


def classes(arguments) -> int:
    """Byte-identical outcome classes of the given dumps (every compared field and the ids)."""
    import hashlib
    import numpy as np
    from experiment.wall_crossing.report_tables import markdown_table, write
    groups: dict = {}
    for name in arguments.dump:
        with np.load(EQUIVALENCE_ROOT / f"{name}.npz") as archive:
            digest = hashlib.sha256(b"".join(np.ascontiguousarray(archive[field]).tobytes()
                                             for field, _ in COMPARED_FIELDS + (("global_id", None),))).hexdigest()[:12]
        groups.setdefault(digest, []).append(name)
    rows = [[digest, str(len(members)), "、".join(members)] for digest, members in groups.items()]
    write(f"split_equivalence_classes_{arguments.name}.md", markdown_table(["类（摘要）", "运行数", "运行"], rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-cases")
    near = sub.add_parser("near-wall")
    near.add_argument("--device", type=int, default=1)
    near.add_argument("--uuid", default="ae137c668a40f5acaab90a83f7cda175")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--case", required=True, help="the case the dumps ran (for the grid and the materials)")
    comparison.add_argument("--name", required=True, help="table name: split_equivalence_<name>.md")
    comparison.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "FIRST", "SECOND"), required=True,
                            help="dump names under logs/wall_crossing/split_dispatch/equivalence (without .npz)")
    grouping = sub.add_parser("classes")
    grouping.add_argument("--name", required=True)
    grouping.add_argument("dump", nargs="+")
    arguments = parser.parse_args()
    if arguments.command == "make-cases":
        return make_cases()
    if arguments.command == "compare":
        return compare(arguments)
    if arguments.command == "classes":
        return classes(arguments)
    return near_wall(arguments)


if __name__ == "__main__":
    sys.exit(main())
