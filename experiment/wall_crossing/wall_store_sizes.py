"""
wall_store_sizes.py - the sizes of the static wall store and of the pools (claude/wall_store_report.md): per crossing
case the walls loaded / kept / discarded, the wall store's buffers (records, voxel offsets, near-wall flags, tables),
the near-wall voxels, the particle pool, and with two slabs the ghost regions (replica, migrant, departed) of this
branch against the particle-pool version's (git tag crossing-pool; its partition code run from its checkout).

    .venv/Scripts/python.exe -m experiment.wall_crossing.wall_store_sizes [--reference C:/.../vulkan-demo-wall-crossing-ref]

Writes claude/wall_store/tables/sizes_store.md, sizes_pools.md and logs/wall_store/checks/sizes.json. No GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CASES = (("2-D 250²", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml"),
         ("2-D 500²", "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing/case.yaml"),
         ("2-D 1000²", "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing/case.yaml"),
         ("3-D 1M", "cases/cavity3d_1m_crossing/case.yaml"))
# the partition of a case in a checkout (this one or the reference), as JSON on stdout
PARTITION_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from experiment.v7.utils.case_loader_v7 import load_case_v7
from experiment.v7.utils.partition_v7 import compute_chain_partition
case = load_case_v7(sys.argv[2])
single = case.capacities.own_pool_size
chain = compute_chain_partition(case, [1.0, 1.0], 1.2)
rows = []
for slab in chain.slabs:
    capacity = slab.capacities
    rows.append({"own_pool": capacity.own_pool_size, "own_particles": int(slab.initial.positions.shape[0]),
                 "leading_ghost_pool": capacity.leading_ghost_pool_size,
                 "trailing_ghost_pool": capacity.trailing_ghost_pool_size,
                 "replica_region": capacity.replica_region_size, "departed_pool": capacity.departed_pool_size,
                 "total_pool": capacity.total_pool_capacity()})
print("PARTITION " + json.dumps({"single_pool": single, "single_particles": int(case.initial.positions.shape[0]),
                                  "cuts": chain.cuts, "slabs": rows}))
"""


def partition(checkout: pathlib.Path, case: str) -> dict:
    python = sys.executable
    completed = subprocess.run([python, "-c", PARTITION_SCRIPT, str(checkout), str(REPOSITORY_ROOT / case)],
                               cwd=checkout, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in completed.stdout.splitlines():
        if line.startswith("PARTITION "):
            return json.loads(line[len("PARTITION "):])
    raise RuntimeError(f"partition of {case} in {checkout} failed:\n{completed.stderr[-2000:]}")


def main() -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.wall_store_v7 import (WALL_STORE_BODY_BYTES, WALL_STORE_PLANE_BYTES,
                                                   WALL_STORE_RECORD_BYTES, WALL_STORE_RULE_ROW_BYTES,
                                                   layout_for_grid)
    from experiment.wall_crossing.wall_store_check import TABLE_ROOT, markdown_table, write_table
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default=str(REPOSITORY_ROOT.parent / "vulkan-demo-wall-crossing-ref"))
    arguments = parser.parse_args()
    reference = pathlib.Path(arguments.reference)
    result = {}
    store_rows, pool_rows = [], []
    for label, path in CASES:
        case = load_case_v7(REPOSITORY_ROOT / path)
        source = case.wall_source
        layout = layout_for_grid(source, case.grid)
        voxels = case.grid.total_voxel_count()
        chain = compute_chain_partition(case, [1.0, 1.0], 1.2)
        slab_walls = []
        for slab in chain.slabs:
            slab_layout = layout_for_grid(source, slab.grid)
            slab_walls.append({"walls": slab_layout.statistics["walls"],
                               "near_wall_voxels": slab_layout.statistics["near_wall_voxels"],
                               "voxels": slab_layout.statistics["voxels"]})
        entry = {"walls_loaded": source.loaded_count, "walls_kept": source.count,
                 "walls_discarded": source.discarded_count, "voxels": voxels,
                 "near_wall_voxels": layout.near_wall_voxel_count,
                 "record_bytes": source.count * WALL_STORE_RECORD_BYTES, "offset_bytes": 4 * (voxels + 2),
                 "flag_bytes": 4 * (voxels + 1), "planes": int(source.planes.shape[0]),
                 "bodies": int(source.bodies.shape[0]), "rules": int(source.rules.shape[0]),
                 "table_bytes": (source.planes.shape[0] * WALL_STORE_PLANE_BYTES
                                 + source.bodies.shape[0] * WALL_STORE_BODY_BYTES
                                 + source.rules.shape[0] * WALL_STORE_RULE_ROW_BYTES),
                 "largest_voxel_wall_count": layout.statistics["largest_voxel_wall_count"],
                 "slabs": slab_walls}
        entry["new_partition"] = partition(REPOSITORY_ROOT, path)
        entry["reference_partition"] = partition(reference, path)
        result[label] = entry
        store_rows.append([label, f"{entry['walls_loaded']:,}", f"{entry['walls_kept']:,}",
                           f"{entry['walls_discarded']:,}（{100 * entry['walls_discarded'] / entry['walls_loaded']:.0f} %）",
                           f"{entry['record_bytes'] / 1024:,.0f} KiB", f"{(entry['offset_bytes'] + entry['flag_bytes']) / 1024:,.0f} KiB",
                           f"{entry['planes']} / {entry['bodies']} / {entry['rules']}（{entry['table_bytes']} B）",
                           f"{entry['near_wall_voxels']:,} / {voxels:,}", f"{entry['largest_voxel_wall_count']}",
                           " / ".join(f"{slab['walls']:,}" for slab in slab_walls)])
        new, old = entry["new_partition"], entry["reference_partition"]
        new_slab, old_slab = new["slabs"][0], old["slabs"][0]

        def ghost(slab):
            return max(slab["leading_ghost_pool"], slab["trailing_ghost_pool"])
        pool_rows.append([label, f"{old['single_pool']:,} → {new['single_pool']:,}",
                          f"{old['single_particles']:,} → {new['single_particles']:,}",
                          " / ".join(f"{slab['own_pool']:,}" for slab in old["slabs"]) + " → "
                          + " / ".join(f"{slab['own_pool']:,}" for slab in new["slabs"]),
                          f"{ghost(old_slab):,} → {ghost(new_slab):,}",
                          f"{old_slab['replica_region']:,} → {new_slab['replica_region']:,}",
                          f"{ghost(old_slab) - 2 * old_slab['replica_region']:,} → "
                          f"{ghost(new_slab) - 2 * new_slab['replica_region']:,}",
                          f"{old_slab['departed_pool']:,} → {new_slab['departed_pool']:,}",
                          f"{old['cuts']} → {new['cuts']}"])
    write_table("sizes_store.md", markdown_table(
        ["算例", "壁面粒子（算例文件）", "保留（到流体盒 < h）", "丢掉（更深）", "壁面记录（32 B 一条）",
         "体素偏移 + 近壁标记", "平面 / 物体 / 壁面材料（字节）", "近壁体素 / 体素", "一个体素里最多的壁面粒子",
         "双卡每张卡上传的壁面粒子"], store_rows))
    write_table("sizes_pools.md", markdown_table(
        ["算例", "单卡粒子池（标签 → 本分支）", "单卡池里的粒子", "双卡每张卡的粒子池", "双卡每个方向的 ghost 池",
         "其中副本区（一层）", "其中迁移区", "离开区（每张卡）", "切点（体素列）"], pool_rows))
    out = REPOSITORY_ROOT / "logs" / "wall_store" / "checks" / "sizes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
