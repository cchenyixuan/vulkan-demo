"""
lean_registers.py - registers, theoretical occupancy and binary size (the driver's statistics through
VK_KHR_pipeline_executable_properties, pipeline_statistics.py; compile only) of the fused pass and force pipelines of
simple and crossing, default and EXPERIMENTAL builds (claude/wall_store_report.md section 9: the lean items, crossing's
factored near-wall force, the near-wall lanes builds).

    .venv/Scripts/python.exe -m experiment.wall_crossing.lean_registers --device 1

Cases: 2-D 500^2 and 3-D 1M (simple and crossing, the K = 1 step's pipelines). Writes
logs/wall_store/round3/registers/<case>_<build>.json, summary.json and claude/wall_store/tables/round3_registers.md.
The binary size is the SASS size in bytes (16 bytes per instruction on this architecture: a static instruction count).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.lean_bits import BUILDS                        # noqa: E402
from experiment.wall_crossing.wall_store_check import theoretical_occupancy  # noqa: E402

OUT = REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "registers"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
CASES = {"2-D": ("cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025/case.yaml",
                 "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing/case.yaml"),
         "3-D": ("cases/cavity3d_1m/case.yaml", "cases/cavity3d_1m_crossing/case.yaml")}
SIMPLE_BUILDS = ("base", "zero_z", "distance_squared", "kernel_range", "zero_z_distance_squared",
                 "zero_z_kernel_range", "distance_squared_kernel_range", "all")
CROSSING_BUILDS = SIMPLE_BUILDS + ("factored", "all_factored", "lanes2", "lanes4", "lanes8")
PIPELINES = {"simple": ("correction_density_all", "force_all"),
             "crossing": ("correction_density_all_split_interior", "correction_density_all_split_near_wall",
                          "force_all_split_interior", "force_all_split_near_wall")}


def collect(device: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dimension, (simple_case, crossing_case) in CASES.items():
        for method, case, builds in (("simple", simple_case, SIMPLE_BUILDS), ("crossing", crossing_case,
                                                                              CROSSING_BUILDS)):
            for build in builds:
                if dimension == "3-D" and build.startswith("zero_z"):
                    continue
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith(("V7_FLUID_SEGMENT_", "V7_WALL_CROSSING_NEAR_WALL_LANES"))}
                environment.update(BUILDS[build])
                environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
                out = OUT / f"{dimension}_{method}_{build}.json"
                command = [sys.executable, str(REPOSITORY_ROOT / "experiment" / "wall_crossing" /
                                               "pipeline_statistics.py"),
                           "--case", case, "--device", str(device), "--filter", ",".join(PIPELINES[method]),
                           "--out", str(out)]
                result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                        cwd=REPOSITORY_ROOT)
                if result.returncode != 0:
                    print(result.stdout[-2000:], result.stderr[-2000:])
                    raise SystemExit(f"{dimension} {method} {build}: exit {result.returncode}")
                report = json.loads(out.read_text(encoding="utf-8"))
                entry = {}
                for name in PIPELINES[method]:
                    statistics = report["pipelines"][name][0]["statistics"]
                    entry[name] = {"registers": statistics["Register Count"],
                                   "binary_size": statistics["Binary Size"],
                                   "occupancy": theoretical_occupancy(statistics["Register Count"])}
                summary.setdefault(dimension, {}).setdefault(method, {})[build] = entry
                print(f"[lean_registers] {dimension} {method} {build}: "
                      + ", ".join(f"{name} {value['registers']}" for name, value in entry.items()), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def write_table(summary: dict) -> None:
    lines = ["| 维数 | 方法 | 版本 | 管线 | 寄存器 | 理论占用率 | 二进制（字节） | 静态指令数 |", "|---|---|---|---|---|---|---|---|"]
    for dimension, methods in summary.items():
        for method, builds in methods.items():
            for build, entry in builds.items():
                for name, value in entry.items():
                    lines.append(f"| {dimension} | {method} | {build} | {name} | {value['registers']} | "
                                 f"{100 * value['occupancy']:.1f} % | {value['binary_size']:,} | "
                                 f"{value['binary_size'] // 16:,} |")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "round3_registers.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--table-only", action="store_true")
    arguments = parser.parse_args()
    summary = (json.loads((OUT / "summary.json").read_text(encoding="utf-8")) if arguments.table_only
               else collect(arguments.device))
    write_table(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
