"""
e8_vram.py — E8 phase 0 item 3: per-card device memory of the 3-D cavity cases, from the v7-rc3 buffer layout.

Static model of experiment/v7/utils/simulator_v7._build_buffer_specs + _allocate_scratch_buffers for one slab:
  particle pool      set 0 SoA, 148 B per pool slot (position 16, density/pressure 8, its scratch 8, velocity 16,
                     acceleration 16, shift 16, material 4, correction inverse 32, density gradient / kernel sum
                     16, extension 16) + the defrag scratch twins, 140 B per slot (every set-0 field but the
                     density scratch)
  voxel lists        set 1, per extended voxel: inside count 4 + incoming count 4 + inside list 4 C + incoming
                     list 4 C_inc + base offset 4 + first replica pid 4 (3-D, h/dx = 4: C = max_per_voxel = 128,
                     C_inc = max_incoming = 32 -> 656 B per voxel) + the deep-wall marker 8 B per voxel (3-D,
                     V7_DEEP_WALL_SKIP=auto)
  packed replicas    4 x 2 x ghost_layers x 8 words x R (V7_PACKED_REPLICAS=1), 4 B at K = 1
  state block        global status 160 B, pool health 16, materials 48 x 3, defrag counter 4, band meta 128,
                     the stubs (wall dummy velocity 16, band list 4, skip record 16)
Pool slots of a slab (partition_v7, release factors of E6b for 3-D: f = V7_GHOST_POOL_FACTOR 0.5, migrant
factor 0.02, departed face fraction 0.64):
  own pool O         = min(ceil(1.2 x own particles / 128) x 128, ceil(1.15 x total / 128) x 128)   (--pool-safety 1.2)
  replica region R   = ceil(NY NZ (C + C_inc) f) per layer, 2 layers per side with a peer
  migrant region M   = max(64, ceil(NY NZ C_inc x 0.02)) per side with a peer
  ghost pool / side  = 2 R + M
  departed pool D    = max(64, ceil(0.64 NY NZ x peer sides))
  pool capacity P    = 1 + L + O + T + D
The lattice of utils/geometry/_demo_cavity_case_aligned.py (3-D, border 4, h/dx = 4): fluid columns F_x = n_x / 4
(4 n^2 fluid + 16 n lid + side walls = 4 (n + 8)^2 particles each), 2 wall columns per side of which the inner one
holds the 4 wall layers (4 (n + 8)^2 particles) and the outer one is empty; NY = NZ = n / 4 + 4. Equal weights:
the cuts of partition_v7.chain_cuts_from_counts on the fluid histogram (4 n^2 per fluid column).
Host memory is not modelled (the staging buffers are host RAM).

The model is checked against the E7 smoke / batch-1 / batch-3 runs (parse_run_v6 fields device_local_mb and
defrag_scratch_mb = the simulator's own "[SimV7] device-local buffers" / "defrag scratch buffers" prints, and the
telemetry VRAM peak): --validate prints both tables.

Usage:
    python docs/cluster_v6/scripts/e8_vram.py [--validate] [--md OUT.md] [--json OUT.json] [--capacity-gib 31.84]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from experiment.v7.utils.partition_v7 import MINIMUM_OWN_COLUMNS_HARD, chain_cuts_from_counts  # noqa: E402

MIB = 1024.0 * 1024.0
C_PER_VOXEL, C_INCOMING = 128, 32            # 3-D h/dx = 4 capacities of the aligned generator
GHOST_POOL_FACTOR, MIGRANT_POOL_FACTOR, DEPARTED_FACE_FRACTION = 0.5, 0.02, 0.64
GHOST_LAYERS = 2
POOL_SAFETY, YAML_POOL_FACTOR, WORKGROUP = 1.2, 1.15, 128
SET0_BYTES, SCRATCH_BYTES = 148, 140
VOXEL_BYTES = 4 + 4 + 4 * C_PER_VOXEL + 4 * C_INCOMING + 4 + 4
DEEP_WALL_BYTES = 8
PACKED_WORDS = 8
STATE_BYTES = 160 + 16 + 48 * 3 + 4 + 128 + 16 + 4 + 16
CONTEXT_MIB = 50.0                            # measured gap telemetry peak - (device-local + scratch): 40-57 MiB

# E8 cases: name -> (n, n_x); lattice border 4, h/dx 4
CASES = {
    "cavity3d_n48": (48, 48), "cavity3d_n152": (152, 152), "cavity3d_n192": (192, 192), "cavity3d_n244": (244, 244),
    "cavity3d_n312": (312, 312), "cavity3d_n392": (392, 392),
    "cavity3d_n496": (496, 496),            # e8_C (user 2026-10-22): 128M = 504^3, 124 fluid columns, K = 8 -> 15 / 16 per card
    "cavity3d_n192_k2": (192, 384), "cavity3d_n192_k4": (192, 768), "cavity3d_n192_k8": (192, 1536),
}
SHORT = {"cavity3d_n48": "56^3 (0.18M)", "cavity3d_n152": "4M (160^3)", "cavity3d_n192": "8M (200^3)", "cavity3d_n244": "16M (252^3)",
         "cavity3d_n312": "32M (320^3)", "cavity3d_n392": "64M (400^3)", "cavity3d_n496": "128M (504^3)", "cavity3d_n192_k2": "weak 2 x 8M",
         "cavity3d_n192_k4": "weak 4 x 8M", "cavity3d_n192_k8": "weak 8 x 8M"}
# E7 measurements (rc1, same layout): (n, n_x, K, own columns per slab, own particles per slab,
#                                      device_local_mb per slab, scratch_mb per slab, VRAM peak MiB per card)
MEASURED = [
    ("3d_n200_k8 K1", 200, 1600, 1, [404], [69_568_512], [12057.53], [10681.67], [22780]),
    ("3d_n416 K1", 416, 416, 1, [108], [76_225_024], [13274.5], [11703.72], [25020]),
    ("3d_n416 K2", 416, 416, 2, [54, 54], [38_112_512, 38_112_512], [7243.28, 7243.28], [6357.45, 6357.45],
     [13658, 13642]),
    ("3d_n416 K4", 416, 416, 4, [28, 26, 26, 28], [19_415_808, 18_696_704, 18_696_704, 19_415_808],
     [3886.86, 4030.57, 4030.57, 3886.86], [3361.93, 3497.87, 3497.87, 3361.93], [7306, 7570, 7570, 7289]),
    ("3d_n416 K8", 416, 416, 8, [15, 13, 13, 13, 13, 13, 13, 15],
     [10_067_456] + [9_348_352] * 6 + [10_067_456], [2208.64] + [2352.35] * 6 + [2208.64],
     [1864.15] + [2000.1] * 6 + [1864.15], [4130] + [4393] * 6 + [4113]),
    ("3d_n200_k8 K8 (calibrated)", 200, 1600, 8, [52, 49, 50, 51, 51, 49, 50, 52],
     [8_825_856, 8_479_744, 8_652_800, 8_825_856, 8_825_856, 8_479_744, 8_652_800, 8_825_856],
     [1688.24, 1694.17, 1725.29, 1756.44, 1756.44, 1694.17, 1725.29, 1688.24],
     [1476.86, 1484.19, 1511.91, 1539.65, 1539.65, 1484.19, 1511.91, 1476.86],
     [3222, 3219, 3278, 3337, 3337, 3219, 3278, 3206]),
    ("3d_n160_k8 K8 (calibrated)", 160, 1280, 8, [41, 41, 40, 40, 40, 41, 40, 41],
     [4_515_840, 4_628_736, 4_515_840, 4_515_840, 4_515_840, 4_628_736, 4_515_840, 4_515_840],
     [879.92, 945.54, 925.2, 925.2, 925.2, 945.54, 925.2, 879.92],
     [765.2, 824.99, 806.89, 806.89, 806.89, 824.99, 806.89, 765.2], [1702, 1811, 1773, 1773, 1773, 1811, 1773, 1686]),
]


def lattice(n: int, n_x: int) -> dict:
    fluid_columns = n_x // 4
    per_column = 4 * (n + 8) ** 2                 # every non-empty column (fluid + lid + side walls, or the end wall)
    columns = [0, per_column] + [per_column] * fluid_columns + [per_column, 0]
    fluid = [0, 0] + [4 * n * n] * fluid_columns + [0, 0]
    face = (n // 4 + 4) ** 2
    total = (n_x + 8) * (n + 8) ** 2
    return {"n": n, "n_x": n_x, "fluid_columns": fluid_columns, "nx": fluid_columns + 4, "face": face,
            "columns": np.array(columns, dtype=np.int64), "fluid": np.array(fluid, dtype=np.int64), "total": total,
            "fluid_total": 4 * n * n * fluid_columns,
            "wall_total": total - 4 * n * n * fluid_columns - 16 * n * fluid_columns,
            "lid_total": 16 * n * fluid_columns}


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def slab_memory(face: int, total: int, own_columns: int, own_particles: int, leading: bool, trailing: bool,
                slabs: int) -> dict:
    """Bytes and slots of one slab (the chain's own cuts give own_columns / own_particles)."""
    global_pool = round_up(total * YAML_POOL_FACTOR, WORKGROUP)
    own_pool = min(round_up(own_particles * POOL_SAFETY, WORKGROUP), global_pool)
    peers = int(leading) + int(trailing)
    replica = int(math.ceil(face * (C_PER_VOXEL + C_INCOMING) * GHOST_POOL_FACTOR)) if slabs > 1 else 0
    migrant = max(64, int(math.ceil(face * C_INCOMING * MIGRANT_POOL_FACTOR))) if slabs > 1 else 0
    ghost_per_side = GHOST_LAYERS * replica + migrant
    departed = max(64, int(math.ceil(DEPARTED_FACE_FRACTION * face * peers))) if peers else 0
    pool = 1 + peers * ghost_per_side + own_pool + departed
    extended_columns = own_columns + GHOST_LAYERS * peers
    voxels = 1 + extended_columns * face
    packed = 4 * 2 * GHOST_LAYERS * PACKED_WORDS * replica if replica else 4
    set0 = SET0_BYTES * pool
    voxel_lists = VOXEL_BYTES * voxels
    deep_wall = DEEP_WALL_BYTES * voxels
    state = STATE_BYTES
    device_local = set0 + voxel_lists + deep_wall + packed + state
    scratch = SCRATCH_BYTES * pool
    return {"own_particles": own_particles, "own_columns": own_columns, "own_pool": own_pool,
            "own_margin": own_pool - own_particles, "replica_region": replica, "migrant_region": migrant,
            "ghost_per_side": ghost_per_side, "peers": peers, "departed": departed, "pool": pool, "voxels": voxels,
            "set0_mib": set0 / MIB, "scratch_mib": scratch / MIB, "voxel_lists_mib": voxel_lists / MIB,
            "deep_wall_mib": deep_wall / MIB, "packed_mib": packed / MIB, "state_mib": state / MIB,
            "device_local_mib": device_local / MIB, "total_mib": (device_local + scratch) / MIB,
            "vram_mib": (device_local + scratch) / MIB + CONTEXT_MIB}


def chain(n: int, n_x: int, slabs: int, minimum: int = MINIMUM_OWN_COLUMNS_HARD) -> list[dict]:
    grid = lattice(n, n_x)
    cuts = chain_cuts_from_counts(grid["fluid"], [1.0] * slabs, minimum) if slabs > 1 else []
    boundaries = [0] + cuts + [grid["nx"]]
    out = []
    for index in range(slabs):
        first, last = boundaries[index], boundaries[index + 1]
        own_particles = int(grid["columns"][first:last].sum())
        fluid_columns = int(np.count_nonzero(grid["fluid"][first:last]))
        entry = slab_memory(grid["face"], grid["total"], last - first, own_particles, index > 0,
                            index < slabs - 1, slabs)
        entry.update({"slab": index, "first_column": first, "last_column": last - 1, "fluid_columns": fluid_columns,
                      "fluid_particles": int(grid["fluid"][first:last].sum())})
        out.append(entry)
    return out


def validate() -> list[str]:
    lines = ["| run | slab | own columns | own particles | device-local MiB model / measured | "
             "scratch MiB model / measured | VRAM peak MiB model / measured |", "|---|---|---|---|---|---|---|"]
    worst = 0.0
    for name, n, n_x, slabs, columns, particles, device_local, scratch, peaks in MEASURED:
        grid = lattice(n, n_x)
        for index in range(slabs):
            entry = slab_memory(grid["face"], grid["total"], columns[index], particles[index], index > 0,
                                index < slabs - 1, slabs)
            error = entry["device_local_mib"] / device_local[index] - 1
            worst = max(worst, abs(error), abs(entry["scratch_mib"] / scratch[index] - 1))
            lines.append(f"| {name} | {index} | {columns[index]} | {particles[index]:,} | "
                         f"{entry['device_local_mib']:,.1f} / {device_local[index]:,.1f} ({error * 100:+.2f} %) | "
                         f"{entry['scratch_mib']:,.1f} / {scratch[index]:,.1f} "
                         f"({(entry['scratch_mib'] / scratch[index] - 1) * 100:+.2f} %) | "
                         f"{entry['vram_mib']:,.0f} / {peaks[index]:,} "
                         f"({(entry['vram_mib'] / peaks[index] - 1) * 100:+.2f} %) |")
    lines.append(f"\nlargest relative error of the device-local / scratch sizes: {worst * 100:.2f} %")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--capacity-gib", type=float, default=31.84,
                        help="usable VRAM of an RTX 5090 (nvidia-smi memory.total)")
    parser.add_argument("--slab-counts", default="1,2,4,8")
    arguments = parser.parse_args()
    capacity_mib = arguments.capacity_gib * 1024.0
    lines: list[str] = []
    report = {"cases": {}, "constants": {
        "C": C_PER_VOXEL, "C_inc": C_INCOMING, "f": GHOST_POOL_FACTOR, "migrant_factor": MIGRANT_POOL_FACTOR,
        "departed_face_fraction": DEPARTED_FACE_FRACTION, "pool_safety": POOL_SAFETY, "context_mib": CONTEXT_MIB,
        "capacity_mib": capacity_mib}}
    if arguments.validate:
        lines += ["## Model check against the E7 runs (v7-rc1, same layout)", ""] + validate() + [""]
    lines += ["## Lattices", "",
              "| case | n | n_x | fluid | wall | lid | total | fluid columns | grid columns (nx) | NY = NZ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (n, n_x) in CASES.items():
        grid = lattice(n, n_x)
        lines.append(f"| {name} ({SHORT[name]}) | {n} | {n_x} | {grid['fluid_total']:,} | {grid['wall_total']:,} | "
                     f"{grid['lid_total']:,} | {grid['total']:,} | {grid['fluid_columns']} | {grid['nx']} | "
                     f"{n // 4 + 4} |")
    lines.append("")
    slab_counts = [int(item) for item in arguments.slab_counts.split(",")]
    lines += ["## Per-card device memory (model; MiB = 2^20 B)", "",
              "| case | K | slab | own columns (fluid) | own particles | own pool O (margin) | "
              "ghost / side = 2R + M | departed D | pool P | voxels | SoA + scratch | voxel lists | packed | "
              "device-local | VRAM est. | fits |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (n, n_x) in CASES.items():
        report["cases"][name] = {}
        for slabs in slab_counts:
            if name.endswith(("_k2", "_k4", "_k8")) and slabs != int(name[-1]):
                continue
            if name == "cavity3d_n48" and slabs not in (1, 2):
                continue
            try:
                slabs_memory = chain(n, n_x, slabs)
            except ValueError as error:
                lines.append(f"| {name} | {slabs} | - | rejected: {error} | | | | | | | | | | | |")
                report["cases"][name][slabs] = {"error": str(error)}
                continue
            report["cases"][name][slabs] = slabs_memory
            shown = {0} | ({1} if slabs > 2 else set()) | ({slabs - 1} if slabs > 1 else set())
            for entry in slabs_memory:
                if entry["slab"] not in shown and entry["own_columns"] == slabs_memory[1]["own_columns"]:
                    continue
                fits = "yes" if entry["vram_mib"] <= capacity_mib else "NO"
                lines.append(
                    f"| {name} | {slabs} | {entry['slab']} | {entry['own_columns']} ({entry['fluid_columns']}) | "
                    f"{entry['own_particles']:,} | {entry['own_pool']:,} ({entry['own_margin']:,}) | "
                    f"{entry['ghost_per_side']:,} = 2 x {entry['replica_region']:,} + {entry['migrant_region']:,} "
                    f"x {entry['peers']} | {entry['departed']:,} | {entry['pool']:,} | {entry['voxels']:,} | "
                    f"{entry['set0_mib'] + entry['scratch_mib']:,.0f} | "
                    f"{entry['voxel_lists_mib'] + entry['deep_wall_mib']:,.0f} | "
                    f"{entry['packed_mib']:,.0f} | {entry['device_local_mib']:,.0f} | {entry['vram_mib']:,.0f} | "
                    f"{fits} |")
    text = "\n".join(lines)
    print(text)
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
