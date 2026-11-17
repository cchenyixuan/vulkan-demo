"""
seam_audit_run.py - the repository's seam audit (experiment/seam_audit: dump_state.py + analyze.py) for one case:
a chain of K slabs against one slab, particle by particle, measured against the one-slab run-to-run noise.

    .venv/Scripts/python.exe -m experiment.wall_crossing.seam_audit_run \
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml --label n250_crossing \
        --single-device 1 --chain-devices 0,1 --horizons 300,2000 \
        --uuid ae137c668a40f5acaab90a83f7cda175 --uuid fb83f0dafd9e1d38dc05e7ae643e22a8

Four runs from the case's initial state with the solver's normal (non-canonical) bootstrap: the reference A1,
A2 (one slab, twice) and the test B1, B2 (the chain, twice), each dumped at every horizon; then, per horizon,
experiment.seam_audit.analyze matches the particles by their global id and reports, per voxel-column distance d
to the cut,
    ratio = statistic(|B1 - A1|) / statistic(|A1 - A2|)
for the acceleration, the shift and the density (a ratio near 1 = the chain differs from one slab by no more
than one slab differs from itself; the audit's "seam excess" is the rms ratio of the acceleration at d = 0),
and the invariants of every run (particle drift, id checks, stamp errors, overflows).

Output: <out>/<label>/ (dumps, logs, analysis_N<horizon>/report.json + report.md, summary.json, summary.md); one
runs.jsonl record per dump and per analysis.
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

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

PRIMARY_FIELDS = ("acceleration", "shift", "density")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--single-device", default="1")
    parser.add_argument("--chain-devices", default="0,1")
    parser.add_argument("--horizons", default="300,2000")
    parser.add_argument("--uuid", action="append", default=[])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    out = ((REPOSITORY_ROOT / arguments.out).resolve() if arguments.out else LOG_ROOT / "seam_audit") / arguments.label
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8",
                       VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
    slabs = len(arguments.chain_devices.split(","))
    horizons = [int(item) for item in arguments.horizons.split(",")]
    run_ids = []
    runs = {"A1": (1, arguments.single_device), "A2": (1, arguments.single_device),
            "B1": (slabs, arguments.chain_devices), "B2": (slabs, arguments.chain_devices)}
    for name, (slab_count, devices) in runs.items():
        if all((out / f"{name}_N{horizon}.npz").exists() for horizon in horizons):
            continue
        command = [sys.executable, "-m", "experiment.seam_audit.dump_state", "--version", "v7",
                   "--case", arguments.case, "--slabs", str(slab_count), "--device-map", devices,
                   "--horizons", arguments.horizons, "--out-dir", str(out), "--run-name", name]
        with LoggedRun(kind="seam_audit_dump", case=arguments.case, gpu_uuids=arguments.uuid,
                       command=" ".join(command[1:])) as run:
            run.extra["label"] = f"{arguments.label} {name} (K = {slab_count}, devices {devices})"
            with open(out / f"{name}.log", "w", encoding="utf-8", newline="\n") as log:
                code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            run.extra["exit_code"] = code
            run.extra["status"] = "ok" if code == 0 else f"failed: exit code {code}"
            run_ids.append(run.run_id)
        print(f"[seam_audit_run] {arguments.label} {name}: dump exit code {code}", flush=True)
        if code != 0:
            return 1

    summary = {"case": arguments.case, "label": arguments.label, "slabs": slabs, "horizons": {}}
    lines = [f"# 接缝审计：{arguments.label}（K = {slabs} 对 K = 1）", "",
             f"算例 `{arguments.case}`；参考 A1、A2 = 单卡（设备 {arguments.single_device}）两次，"
             f"测试 B1、B2 = 链（设备 {arguments.chain_devices}）两次；粒子按全局编号配对。"
             "比值 = rms(|B1 − A1|) / rms(|A1 − A2|)，d = 到切口的体素列距离。", "",
             "| 步数 N | 配对粒子 | 加速度 d = 0（接缝超出量） | 位移修正 d = 0 | 密度 d = 0 | 第二对 (B2, A2) 加速度 d = 0 | "
             "链自身噪声 \\|B1 − B2\\| / \\|A1 − A2\\| 加速度 d = 0 | 远区最坏 rms 比值（加速度 / 位移 / 密度） | "
             "不变量（漂移、编号、戳、溢出） |", "|" + "---|" * 9]
    for horizon in horizons:
        analysis_directory = out / f"analysis_N{horizon}"
        command = [sys.executable, "-m", "experiment.seam_audit.analyze", "--reference",
                   str(out / f"A1_N{horizon}.npz"), str(out / f"A2_N{horizon}.npz"), "--test",
                   str(out / f"B1_N{horizon}.npz"), str(out / f"B2_N{horizon}.npz"), "--out",
                   str(analysis_directory), "--match", "id", "--no-figures"]
        with LoggedRun(kind="seam_audit_analysis", case=arguments.case, gpu_uuids=[],
                       command=" ".join(command[1:])) as run:
            run.extra["label"] = f"{arguments.label} analysis N = {horizon}"
            with open(out / f"analysis_N{horizon}.log", "w", encoding="utf-8", newline="\n") as log:
                code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            run.extra["exit_code"] = code
            run.extra["status"] = "ok" if code == 0 else f"failed: exit code {code}"
            run_ids.append(run.run_id)
        if code != 0:
            print(f"[seam_audit_run] analysis N = {horizon} failed, see {out / f'analysis_N{horizon}.log'}")
            return 1
        report_files = sorted(analysis_directory.glob("*.json"))
        report = None
        for path in report_files:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in candidate:
                report = candidate
                break
        if report is None:
            print(f"[seam_audit_run] no report with a summary in {analysis_directory}")
            return 1
        row = report["summary"]
        summary["horizons"][str(horizon)] = row
        d0 = row["d0_rms_ratio"]
        far = row["far_bin_worst_rms_ratio"]
        second = row.get("second_pair_d0_rms_ratio") or {}
        noise = row.get("test_noise_d0_rms_ratio") or {}

        def text(value) -> str:
            return "—" if value is None else f"{value:.2f}"

        invariants = row["invariants"]
        clean = row["all_runs_valid"] and all(
            (values.get("drift") or 0) == 0 and (values.get("overflow_total") or 0) == 0
            for values in invariants.values())
        lines.append(
            f"| {horizon} | {row['triplets']:,} | {text(row['seam_excess'])} | {text(d0['shift']['all'])} | "
            f"{text(d0['density']['all'])} | {text(second.get('acceleration'))} | {text(noise.get('acceleration'))} | "
            f"{text(far['acceleration']['value'])} / {text(far['shift']['value'])} / {text(far['density']['value'])} | "
            f"{'全部为 0，四个运行有效' if clean else json.dumps(invariants)} |")
    summary["run_ids"] = run_ids
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    if run_ids:
        lines += ["", f"记录：{run_ids[0]} … {run_ids[-1]}（共 {len(run_ids)} 条）"]
    text_out = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text_out, encoding="utf-8", newline="\n")
    print(text_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
