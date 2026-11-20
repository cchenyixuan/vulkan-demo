"""
volume_constraint_check.py - report section 22 B: numerics.volume_constraint.gpu_mean_volume (off by default then; on by default since report section 25), the
mean-volume measure of the volume constraint reduced on the GPU (utils/volume_constraint_v7.py,
shaders/volume_constraint_reduction.comp).

    .venv/Scripts/python.exe -m experiment.wall_crossing.volume_constraint_check make-cases
        writes <case>_gpu_mean/case.yaml for the cases of GPU_MEAN_CASES: numerics.volume_constraint.gpu_mean_volume
        true, everything else the same (relative file paths rewritten for the new directory)
    .venv/Scripts/python.exe -m experiment.wall_crossing.volume_constraint_check theta --device-map 1
        per case, a chain run (chain_runner.run_chain) of --steps steps; at every sample (a drained defrag boundary)
        theta from the GPU partial sums of every slab and theta from the host's float64 sums of the same state
        (SphSimulatorV7.volume_constraint_measure) -> logs/wall_crossing/volume_constraint/theta_<label>.json and
        claude/wall_crossing/tables/volume_constraint_theta_<label>.md
    .venv/Scripts/python.exe -m experiment.wall_crossing.volume_constraint_check logs --name LABEL RUN [RUN ...]
        the constraint logs (volume_constraint.jsonl) of box_run runs side by side, update by update ->
        claude/wall_crossing/tables/volume_constraint_logs_<label>.md
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

GPU_MEAN_CASES = ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
                  "cases/wall_crossing/hydrostatic_3d_n48_gtilt_crossing")


def gpu_mean_case_path(case_directory: str) -> pathlib.Path:
    return REPOSITORY_ROOT / f"{case_directory}_gpu_mean" / "case.yaml"


def make_cases() -> int:
    import os
    for case_directory in GPU_MEAN_CASES:
        source_directory = REPOSITORY_ROOT / case_directory
        target = gpu_mean_case_path(case_directory)
        source = (source_directory / "case.yaml").read_text(encoding="utf-8")
        match = re.search(r"^  volume_constraint:[^\n]*\n((?:    [^\n]*\n)*)", source, flags=re.M)
        if match is None or "enabled: true" not in match.group(0):
            raise SystemExit(f"{case_directory}: no numerics.volume_constraint with enabled: true")

        def relocate(path_match):
            old = path_match.group(2)
            if os.path.isabs(old):
                return path_match.group(0)
            new = os.path.relpath((source_directory / old).resolve(), target.parent.resolve()).replace("\\", "/")
            return path_match.group(1) + new + path_match.group(3)

        text = source[:match.end()] + "    gpu_mean_volume: true   # report section 22\n" + source[match.end():]
        text = re.sub(r"(material_library:\s*)([^\s#]+)(.*)", relocate, text)
        text = re.sub(r"(frame:\s*)([^\s#]+)(.*)", relocate, text)
        text = re.sub(r"(file:\s*)([^,}\s]+)(.*)", relocate, text)
        text = (f"# VARIANT of {case_directory}/case.yaml (report section 22, 2026-11-19): "
                "numerics.volume_constraint.gpu_mean_volume true; everything else identical.\n" + text)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        from experiment.v7.utils.case_loader_v7 import load_case_v7
        loaded = load_case_v7(target)
        reference = load_case_v7(source_directory / "case.yaml")
        assert loaded.numerics.volume_constraint.gpu_mean_volume
        assert loaded.initial.positions.shape == reference.initial.positions.shape
        print(f"[volume_constraint_check] wrote {target.relative_to(REPOSITORY_ROOT)}")
    return 0


def theta(arguments) -> int:
    import os
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing.chain_runner import run_chain
    from experiment.wall_crossing.report_tables import markdown_table, write
    device_map = [int(item) for item in arguments.device_map.split(",")]
    rows, records = [], []
    for case_directory in GPU_MEAN_CASES:
        case_path = gpu_mean_case_path(case_directory)
        case = load_case_v7(case_path)
        cadence = int(case.numerics.defrag_cadence)
        samples = []

        def on_sample(step, simulators, slabs):
            parts = [simulator.volume_constraint_measure() for simulator in simulators]
            deviation = sum(part["deviation_sum"] for part in parts)
            count = sum(part["fluid_count"] for part in parts)
            volume = sum(part["host_volume"] for part in parts)
            host_count = sum(part["host_fluid_count"] for part in parts)
            reference = parts[0]["reference_mean_volume"]
            reference_float32 = parts[0]["reference_float32"]
            theta_gpu = (reference_float32 / reference) * (1.0 + deviation / count) - 1.0
            theta_host = (volume / host_count) / reference - 1.0
            samples.append({"step": step, "theta_gpu": theta_gpu, "theta_host": theta_host,
                            "difference": theta_gpu - theta_host, "fluid_count": count, "host_fluid_count": host_count,
                            "reference_mean_volume": reference, "reference_float32": reference_float32})

        label = case_path.parent.name
        with LoggedRun(kind="volume_constraint_theta", case=str(case_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                       gpu_uuids=arguments.uuid) as run:
            run.extra["device_map"] = device_map
            result = run_chain(case, device_map, total_steps=arguments.steps // cadence * cadence,
                               sample_steps=cadence, on_sample=on_sample, application_name="volume_constraint_theta")
            run.extra["samples"] = samples
        records.append({"case": label, "device_map": device_map, "samples": samples, "run_id": run.run_id,
                        "cuts": result.get("cuts")})
        for sample in samples:
            rows.append([label, f"K = {len(device_map)}", str(sample["step"]), f"{sample['theta_gpu']:+.12e}",
                         f"{sample['theta_host']:+.12e}", f"{sample['difference']:+.2e}",
                         f"{sample['fluid_count']:,} / {sample['host_fluid_count']:,}", run.run_id])
    out = LOG_ROOT / "volume_constraint"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"theta_{arguments.name}.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    write(f"volume_constraint_theta_{arguments.name}.md", markdown_table(
        ["算例", "卡数", "步", "θ（GPU 部分和）", "θ（主机双精度）", "差", "流体粒子数（GPU / 主机）", "记录"], rows))
    return 0


def logs(arguments) -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    rows = []
    for run_name in arguments.run:
        path = LOG_ROOT / "box_run" / run_name / "volume_constraint.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "theta" not in record:
                rows.append([run_name, "—", "—", f"segment {record.get('segment')} from step "
                             f"{record.get('segment_start_step')}", "", "", ""])
                continue
            rows.append([run_name, str(record["step"]), f"{record['time']:.4f}", f"{record['theta']:+.15e}",
                         repr(record.get("reference_mean_volume")), str(record.get("fluid_particles")),
                         str(record.get("slabs"))])
    write(f"volume_constraint_logs_{arguments.name}.md", markdown_table(
        ["运行", "步（本段）", "t（本段）", "θ", "v_ref", "流体粒子", "卡数"], rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-cases")
    measure = sub.add_parser("theta")
    measure.add_argument("--device-map", default="1")
    measure.add_argument("--uuid", action="append", default=[])
    measure.add_argument("--steps", type=int, default=3000)
    measure.add_argument("--name", default="k1")
    log_table = sub.add_parser("logs")
    log_table.add_argument("--name", required=True)
    log_table.add_argument("run", nargs="+")
    arguments = parser.parse_args()
    if arguments.command == "make-cases":
        return make_cases()
    if arguments.command == "theta":
        return theta(arguments)
    return logs(arguments)


if __name__ == "__main__":
    sys.exit(main())
