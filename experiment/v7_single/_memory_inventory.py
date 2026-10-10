"""
_memory_inventory.py — device-local memory of v7_single per case ("§3 补测" item 5), from the simulator's own
buffer table built on the CPU (experiment/v7_single/_v7_cpu_fake.fake_single_simulator: the same
_build_buffer_specs and defrag twins the GPU run allocates, no device).

Per case: bytes per pool slot (the set-0 SoA fields + the defrag twins = DEFRAG_SET0_BINDINGS), the voxel lists
(set 1) in bytes per voxel and per particle, the small fixed buffers, the device-local total and its bytes per
particle and per pool slot. The pool is the case's capacities (own_pool_size + slot 0; no ghost / departed pool
on one GPU), so bytes per particle = bytes per slot x pool / N for the SoA part.

    python experiment/v7_single/_memory_inventory.py --case cases/lid_driven_cavity_2d_gen/case.yaml \
        --case cases/lid_driven_cavity_2d_8m/case.yaml --out docs/seam_audit/v7_single/memory.json --out-md ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

SIZE_TAGS = {"lid_driven_cavity_2d_gen": "1m", "lid_driven_cavity_2d_2m": "2m", "lid_driven_cavity_2d_4m": "4m",
             "lid_driven_cavity_2d_8m": "8m", "lid_driven_cavity_2d_16m": "16m", "lid_driven_cavity_2d_32m": "32m",
             "cavity3d_8m_b4": "3d_8m"}


def inventory(case_path: str, obj_cache: str | None) -> dict:
    from _run_single_baseline_bench import install_obj_cache
    import experiment.v7_single.utils.simulator_single as simulator_single
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7_single._v7_cpu_fake import fake_single_simulator
    from experiment.v7_single.utils.switches import configured_switches
    if obj_cache:
        install_obj_cache(pathlib.Path(obj_cache))
    case = load_case_v7(case_path)
    simulator = fake_single_simulator(simulator_single, case, configured_switches())
    pool = case.capacities.total_pool_capacity()
    own_pool = case.capacities.own_pool_size
    particles = int(case.initial.positions.shape[0])
    voxels = 1 + case.grid.total_voxel_count()          # slot 0 unused, ids 1-based
    rows = []
    soa_per_slot = 0
    twin_per_slot = 0
    voxel_bytes = 0
    fixed_bytes = 0
    for spec in simulator._buffer_specs:
        twin = simulator.scratch_buffers.get(spec.name)
        twin_size = int(twin.size) if twin is not None else 0
        if spec.set_index == 0 and spec.size >= pool:
            kind = "soa"
            soa_per_slot += spec.size // pool
            twin_per_slot += twin_size // pool
            per = f"{spec.size // pool} B/slot" + (f" + twin {twin_size // pool} B/slot" if twin_size else "")
        elif spec.set_index == 1 and spec.size >= voxels:
            kind = "voxel_lists"
            voxel_bytes += spec.size
            per = f"{spec.size // voxels} B/voxel"
        else:
            kind = "fixed"
            fixed_bytes += spec.size
            per = f"{spec.size} B"
        rows.append({"name": spec.name, "set": spec.set_index, "binding": spec.binding, "bytes": int(spec.size),
                     "twin_bytes": twin_size, "kind": kind, "per": per})
    buffers_total = sum(row["bytes"] for row in rows)
    twins_total = sum(row["twin_bytes"] for row in rows)
    total = buffers_total + twins_total
    return {"case": case_path, "size": SIZE_TAGS.get(pathlib.Path(case_path).parent.name, pathlib.Path(case_path).parent.name),
            "particles": particles, "dimension": int(case.physics.dimension),
            "pool_capacity": int(pool), "own_pool_size": int(own_pool), "pool_over_particles": pool / particles,
            "voxels": int(voxels), "max_particles_per_voxel": int(case.capacities.max_particles_per_voxel),
            "max_incoming_per_voxel": int(case.capacities.max_incoming_per_voxel),
            "soa_bytes_per_slot": int(soa_per_slot), "twin_bytes_per_slot": int(twin_per_slot),
            "slot_bytes": int(soa_per_slot + twin_per_slot),
            "soa_bytes": int(sum(row["bytes"] for row in rows if row["kind"] == "soa")), "twin_bytes": int(twins_total),
            "voxel_list_bytes": int(voxel_bytes), "voxel_list_bytes_per_voxel": voxel_bytes / voxels,
            "voxel_list_bytes_per_particle": voxel_bytes / particles,
            "fixed_bytes": int(fixed_bytes),
            "device_local_bytes": int(total), "device_local_bytes_per_particle": total / particles,
            "device_local_bytes_per_slot": total / pool,
            "slot_bytes_per_particle": (soa_per_slot + twin_per_slot) * pool / particles,
            "buffer_count": len(rows), "twin_count": sum(1 for row in rows if row["twin_bytes"]),
            "switches": simulator.switches.table(), "rows": rows}


def markdown(records: list[dict]) -> str:
    lines = ["# v7_single device-local memory per case (buffer table built on the CPU, no device)", "",
             "Pool slot = the set-0 SoA fields (position_voxel_id 16, density_pressure 8, density_pressure_scratch 8, "
             "velocity_mass 16, acceleration 16, shift 16, material 4, correction_inverse 32, "
             "density_gradient_kernel_sum 16, extension_fields 16 B; wall_dummy_velocity 16 B only with adami walls) "
             "plus the defrag twins of the nine DEFRAG_SET0_BINDINGS fields (everything but density_pressure_scratch "
             "and wall_dummy_velocity). Voxel lists = set 1: inside / incoming counts (4 B each), inside index "
             "(4 x max_per_voxel), incoming index (4 x max_incoming), voxel_base_offset (4), per voxel (slot 0 "
             "unused). Fixed = global status, pool health, material parameters, counters and the stubs.", "",
             "| case | particles | pool | pool / N | B / slot (SoA + twin) | SoA B | twin B | voxels | "
             "B / voxel (lists) | lists B / particle | fixed B | device-local total | B / particle |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['size']} | {r['particles']:,} | {r['pool_capacity']:,} | {r['pool_over_particles']:.3f} | "
                     f"{r['slot_bytes']} ({r['soa_bytes_per_slot']} + {r['twin_bytes_per_slot']}) | "
                     f"{r['soa_bytes'] / 2**20:,.1f} MiB | {r['twin_bytes'] / 2**20:,.1f} MiB | {r['voxels']:,} | "
                     f"{r['voxel_list_bytes_per_voxel']:.0f} | {r['voxel_list_bytes_per_particle']:.1f} | "
                     f"{r['fixed_bytes']:,} | {r['device_local_bytes'] / 2**20:,.1f} MiB | "
                     f"{r['device_local_bytes_per_particle']:.1f} |")
    lines += ["", "Bytes per particle = device-local total / N; the SoA + twin part alone is "
              + ", ".join(f"{r['size']} {r['slot_bytes_per_particle']:.1f}" for r in records)
              + " B / particle (slot bytes x pool / N).", ""]
    for r in records:
        lines += [f"## {r['size']} ({r['case']}): {r['buffer_count']} buffers + {r['twin_count']} twins", "",
                  "| buffer | set | binding | bytes | twin bytes | per |", "|---|---|---|---|---|---|"]
        for row in r["rows"]:
            lines.append(f"| {row['name']} | {row['set']} | {row['binding']} | {row['bytes']:,} | "
                         f"{row['twin_bytes']:,} | {row['per']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    arguments = parser.parse_args()
    records = [inventory(case, arguments.obj_cache) for case in arguments.case]
    for r in records:
        print(f"[memory] {r['size']}: N={r['particles']:,} pool={r['pool_capacity']:,} slot {r['slot_bytes']} B "
              f"({r['soa_bytes_per_slot']} + {r['twin_bytes_per_slot']}), lists {r['voxel_list_bytes_per_particle']:.1f} "
              f"B/particle, total {r['device_local_bytes'] / 2**20:,.1f} MiB = {r['device_local_bytes_per_particle']:.1f} "
              f"B/particle", flush=True)
    if arguments.out:
        pathlib.Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.out).write_text(json.dumps(records, indent=1), encoding="utf-8")
    if arguments.out_md:
        pathlib.Path(arguments.out_md).write_text(markdown(records), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
