"""
_compare_recordings.py — acceptance 1 of v7_single (CPU, no GPU): the command streams of v7's single-GPU
mode and of v7_single, recorded on the CPU and compared command for command.

For every case and switch setting, both simulators are built without a device (_v7_cpu_fake.py) and
their recorders run with the vkCmd* entry points replaced by recorders: the time step, the two bootstrap
command buffers, the defrag and the restart voxelization. Every "bind pipeline" event is resolved to the
pipeline's (SPIR-V file, specialization constants) through each side's own pipeline builder, so the
comparison covers the SPIR-V bound, the specialization constants, the dispatch sizes, the barriers (stage
and access masks), the fills and the copies, in order. The only difference the inventory allows
(docs/seam_audit/v7_single.md, inventory 7) is the fill of global_status.migration_install_count in v7's
defrag; everything else must be identical, or the run reports NOT IDENTICAL and exits 1.

    .venv/Scripts/python.exe experiment/v7_single/_compare_recordings.py --case cases/.../case.yaml [--case ...] \\
        [--settings default,unfused,transfer_copy] [--out logs/v7_single/acceptance/cpu_compare.json] [--inventory]

--inventory also prints both sides' buffer tables (name, bytes, v7's sharing mode) and the pipeline counts.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.v7_single._v7_cpu_fake import (entries_to_dict, fake_single_simulator,  # noqa: E402
                                               fake_v7_simulator, recording, single_pipeline_specs,
                                               v7_pipeline_specs, v7_switch_values)

SETTINGS = {
    "default":       {"fused": 1, "copy_compute": 1, "deep_wall": "auto"},
    "unfused":       {"fused": 0, "copy_compute": 1, "deep_wall": "auto"},
    "transfer_copy": {"fused": 1, "copy_compute": 0, "deep_wall": "auto"},
}
RECORDINGS = ("step", "bootstrap_init", "bootstrap_compute", "defrag", "restart")
# v7's defrag zeroes global_status.migration_install_count (byte 48); single-GPU runs never write the field.
ALLOWED_V7_ONLY = {("fill", "buffer:global_status", 48, 4, 0)}


def normalize(events: list, specs: dict) -> list:
    """Hashable events with every bind resolved to (shader, sorted spec constants)."""
    out = []
    for event in events:
        if event[0] == "bind":
            key = event[1].split("pipeline:", 1)[1] if isinstance(event[1], str) and event[1].startswith("pipeline:") else event[1]
            if key not in specs:
                raise KeyError(f"bound pipeline {key!r} has no spec record")
            shader, constants = specs[key]
            out.append(("bind", shader, tuple(sorted((identifier, fmt, value) for identifier, (fmt, value) in constants.items()))))
        elif event[0] == "sets":
            out.append(("sets", int(event[2]), int(event[3])))
        else:
            out.append(tuple(event))
    return out


def capture_v7(simulator_v7, case, setting: dict) -> tuple[dict[str, list], dict]:
    import experiment.v7.utils.simulator_v7 as module
    with v7_switch_values(simulator_v7, fused=setting["fused"], copy_compute=setting["copy_compute"],
                          deep_wall=setting["deep_wall"]):
        specs = v7_pipeline_specs(simulator_v7, case, fused=setting["fused"], copy_compute=setting["copy_compute"],
                                  deep_wall=setting["deep_wall"])
        simulator = fake_v7_simulator(simulator_v7, case)
        streams = {}
        for name, record in (("step", simulator._record_step_single_cmd),
                             ("bootstrap_init", simulator._record_bootstrap_init_cmd),
                             ("bootstrap_compute", simulator._record_bootstrap_compute_cmd),
                             ("defrag", simulator._record_defrag_cmd)):
            events: list = []
            with recording(module, events):
                record()
            streams[name] = normalize(events, specs)
        # the restart's inline recording: uploads and readbacks stubbed, an empty state (0 rows)
        pool_capacity = case.capacities.total_pool_capacity()
        simulator._upload_initial_state = lambda payload=None: None
        simulator.readback_global_status = lambda: {"alive_particle_count": 0, "overflow_inside_count": 0}
        simulator.readback_buffer_by_name = lambda name: bytes(pool_capacity * 16)
        empty = {name: np.zeros((0,) if components == 1 else (0, components), dtype=dtype)
                 for name, (components, dtype) in simulator.RESTART_FIELD_LAYOUT.items()}
        events = []
        with recording(module, events):
            simulator.restart_init(empty)
        streams["restart"] = normalize(events, specs)
        inventory = {"buffers": [(spec.name, spec.set_index, spec.binding, spec.size,
                                  "CONCURRENT" if spec.name in simulator_v7._CONCURRENT_BUFFER_NAMES else "EXCLUSIVE")
                                 for spec in simulator._buffer_specs],
                     "scratch_bytes": sum(buffer.size for buffer in simulator.scratch_buffers.values()),
                     "pipelines": sorted(specs), "pipeline_count": len(specs),
                     "deep_wall": list(simulator.deep_wall_skip_resolution),
                     "fused": list(simulator.fused_correction_density_resolution)}
    return streams, inventory


def capture_single(simulator_single, case, setting: dict) -> tuple[dict[str, list], dict]:
    from experiment.v7_single.utils.switches import Switches
    switches = Switches(fused_correction_density=setting["fused"], density_copy_compute=setting["copy_compute"],
                        deep_wall_skip=setting["deep_wall"])
    specs = single_pipeline_specs(simulator_single, case, switches)
    simulator = fake_single_simulator(simulator_single, case, switches)
    streams = {}
    for name, record in (("step", simulator.record_step), ("bootstrap_init", simulator.record_bootstrap_init),
                         ("bootstrap_compute", simulator.record_bootstrap_compute),
                         ("defrag", simulator.record_defrag), ("restart", simulator.record_restart_init)):
        events: list = []
        with recording(simulator_single, events):
            simulator_single.vkBeginCommandBuffer("cmd", None)
            record("cmd")
            simulator_single.vkEndCommandBuffer("cmd")
        streams[name] = normalize(events, specs)
    inventory = {"buffers": [(spec.name, spec.set_index, spec.binding, spec.size, "EXCLUSIVE")
                             for spec in simulator._buffer_specs],
                 "scratch_bytes": sum(buffer.size for buffer in simulator.scratch_buffers.values()),
                 "pipelines": sorted(specs), "pipeline_count": len(specs),
                 "deep_wall": list(simulator.deep_wall_skip_resolution),
                 "fused": list(simulator.fused_correction_density_resolution)}
    return streams, inventory


def describe(event) -> str:
    if event[0] == "bind":
        constants = ", ".join(f"{identifier}={value}" for identifier, _fmt, value in event[2]
                              if identifier not in (0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                    32, 33, 40, 41, 42))
        return f"bind {event[1]} [{constants}]"
    return " ".join(str(part) for part in event)


def compare_streams(first: list, second: list, allowed_first_only=ALLOWED_V7_ONLY) -> dict:
    """Align two streams; unexplained differences make the pair not identical."""
    matcher = difflib.SequenceMatcher(a=first, b=second, autojunk=False)
    allowed = []
    differences = []
    for tag, first_lo, first_hi, second_lo, second_hi in matcher.get_opcodes():
        if tag == "equal":
            continue
        first_part, second_part = first[first_lo:first_hi], second[second_lo:second_hi]
        if tag == "delete" and all(event in allowed_first_only for event in first_part):
            allowed += [describe(event) for event in first_part]
            continue
        differences.append({"tag": tag, "v7": [describe(event) for event in first_part],
                            "v7_single": [describe(event) for event in second_part],
                            "position": [first_lo, second_lo]})
    counts = {side: {"dispatches": sum(1 for event in stream if event[0] == "dispatch"),
                     "barriers": sum(1 for event in stream if event[0] == "barrier"),
                     "binds": sum(1 for event in stream if event[0] == "bind"),
                     "fills": sum(1 for event in stream if event[0] == "fill"),
                     "copies": sum(1 for event in stream if event[0] == "copy"),
                     "events": len(stream)}
              for side, stream in (("v7", first), ("v7_single", second))}
    return {"identical": not differences, "allowed_v7_only": allowed, "differences": differences, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    arguments = parser.parse_args()
    for name in [name for name in os.environ if name.startswith("V7_")]:
        del os.environ[name]
    os.environ["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    from _run_single_baseline_bench import install_obj_cache
    install_obj_cache(_REPOSITORY_ROOT / arguments.obj_cache)
    import experiment.v7.utils.simulator_v7 as simulator_v7
    import experiment.v7_single.utils.simulator_single as simulator_single
    from experiment.v7.utils.case_loader_v7 import load_case_v7

    settings = [name.strip() for name in arguments.settings.split(",") if name.strip()]
    report = {"cases": {}, "all_identical": True, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for case_path in arguments.case:
        case = load_case_v7(case_path)
        case_report = {}
        for setting_name in settings:
            setting = SETTINGS[setting_name]
            v7_streams, v7_inventory = capture_v7(simulator_v7, case, setting)
            single_streams, single_inventory = capture_single(simulator_single, case, setting)
            results = {}
            for name in RECORDINGS:
                results[name] = compare_streams(v7_streams[name], single_streams[name])
                report["all_identical"] &= results[name]["identical"]
            case_report[setting_name] = {"recordings": results, "v7": v7_inventory, "v7_single": single_inventory}
            verdicts = " ".join(f"{name}={'identical' if results[name]['identical'] else 'DIFFERENT'}"
                                f"{'(+allowed fill)' if results[name]['allowed_v7_only'] else ''}"
                                for name in RECORDINGS)
            counts = results["step"]["counts"]["v7"]
            print(f"[compare] {case_path} [{setting_name}] deep_wall={'on' if v7_inventory['deep_wall'][0] else 'off'}: "
                  f"{verdicts}; step = {counts['dispatches']} dispatches, {counts['barriers']} barriers, "
                  f"{counts['binds']} binds; pipelines built v7 {v7_inventory['pipeline_count']} / "
                  f"v7_single {single_inventory['pipeline_count']}", flush=True)
            for name in RECORDINGS:
                for difference in results[name]["differences"]:
                    print(f"    DIFFERENT in {name} at {difference['position']}: v7 {difference['v7']} | "
                          f"v7_single {difference['v7_single']}")
            if arguments.inventory and setting_name == settings[0]:
                v7_total = sum(size for _, _, _, size, _ in v7_inventory["buffers"])
                single_total = sum(size for _, _, _, size, _ in single_inventory["buffers"])
                single_sizes = {name: size for name, _, _, size, _ in single_inventory["buffers"]}
                print(f"    buffers: v7 {len(v7_inventory['buffers'])} ({v7_total:,} B, "
                      f"{sum(1 for item in v7_inventory['buffers'] if item[4] == 'CONCURRENT')} CONCURRENT) + scratch "
                      f"{v7_inventory['scratch_bytes']:,} B; v7_single {len(single_inventory['buffers'])} "
                      f"({single_total:,} B) + scratch {single_inventory['scratch_bytes']:,} B")
                for name, set_index, binding, size, mode in v7_inventory["buffers"]:
                    other = single_sizes.get(name)
                    print(f"      set{set_index}.{binding:<2} {name:30s} v7 {size:>12,d} {mode:10s} v7_single "
                          f"{other if other is None else f'{other:,}':>12}")
                print(f"    pipelines v7: {', '.join(v7_inventory['pipelines'])}")
                print(f"    pipelines v7_single: {', '.join(single_inventory['pipelines'])}")
        report["cases"][case_path] = case_report
    print(f"[compare] {'ALL IDENTICAL' if report['all_identical'] else 'NOT IDENTICAL'} "
          f"({len(arguments.case)} cases x {len(settings)} settings x {len(RECORDINGS)} recordings; "
          f"allowed v7-only command: defrag's migration_install_count fill)")
    if arguments.out:
        out = pathlib.Path(arguments.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[compare] -> {out}")
    return 0 if report["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
