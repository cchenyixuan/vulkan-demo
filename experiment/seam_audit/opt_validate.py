"""
opt_validate.py - correctness gate for one v6 optimisation switch set
(docs/seam_audit/v6_opt.md). Runs, in order:

  audit   seam audit K=1 vs K=2, 2-D 1M, N = 2000, seam configuration (1,2)
          plus the switches: two v6 test runs with the 400-frame crossing
          window, capture requests, two fresh v6 K=1 references (v6 K=1 = v5 K=1; avoids the v5 working tree), analysis
          (run_matrix.py two-pass procedure).
          PASS: every column-0 rms ratio (acceleration, shift, velocity,
          density, pressure, kernel_sum) of both pairs <= AUDIT_LIMIT, every
          crossing-window group <= AUDIT_LIMIT, all runs valid.
  single  single-step test from the kept 2-D 1M N = 2000 snapshot: K=1
          reference + K=1 shuffled-order noise + two (1,2)+switch restarts,
          k = 1, 2, 5, 10, 50.
          PASS (k = 1): for both (1,2) runs, column 0 and column 1 rms and
          median ratios of a, shift, rho, p, kernel_sum, L <= SINGLE_LIMIT
          (exactly-equal bins count as passing), and the reconstruction's
          measured rms / noise <= SINGLE_LIMIT.
  k4      K=4 smoke (device map 0,1,0,1), 2-D 1M, 1000 frames, chain bench
          with its seam / overflow / stamp / far_migration checks.

The (1,1)/(0,1) defect sizes these limits must exclude: column-0 acceleration
6.9e3-1.7e4 x noise at k = 1 (defect 1); audit column-0 shift 29-68 x and
kernel_sum 15-55 x at N = 2000 (defect 2).

Usage:
  .venv/Scripts/python.exe -m experiment.seam_audit.opt_validate --name lean \\
      --env V6_LEAN_TRANSPORT=1 [--env KEY=VALUE ...] [--steps audit,single,k4]
Output: logs/seam_audit/opt/validate_<name>/verdict.json (+ every sub-run's data).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

AUDIT_LIMIT = 2.0
SINGLE_LIMIT = 2.5
SEAM_L2 = {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}
CASE_1M = "cases/lid_driven_cavity_2d_gen/case.yaml"
SNAPSHOT_DIRECTORY = "logs/seam_audit/single_step/cavity2d_1m"
AUDIT_FIELDS = ("acceleration", "shift", "velocity", "density", "pressure", "kernel_sum")
SINGLE_FIELDS = ("acceleration", "shift", "density", "pressure", "kernel_sum", "correction_inverse")
PRODUCTION_2D = {"V6_WORKER_COUNT_AWARE": "1", "V6_GHOST_POOL_FACTOR": "0.25",
                 "V6_SPLIT_TRANSFER_QUEUES": "1", "V6_CASCADE_FORCE": "1",
                 "V6_BAND_VOXEL_DISPATCH": "1"}


def run(command, environment, log_path, timeout):
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                   env=environment, cwd=str(_REPO_ROOT))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = -9
    print(f"[opt_validate]   {' '.join(command[2:6])} ... exit {code} ({time.time() - started:.0f} s)",
          flush=True)
    return code


def base_environment():
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


# --------------------------------------------------------------------------- audit
def run_audit(out_dir: pathlib.Path, switches: dict, name: str, timeout: float) -> dict:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    test_name = f"v6_l2_{name}"
    matrix = {
        "description": f"optimisation gate {name}: v6 (1,2) + {switches} vs fresh v6 K=1 references",
        "cases": [{"name": "cavity2d_1m", "path": CASE_1M, "dimension": 2}],
        "horizons": [2000], "window": 400, "window_horizons": [2000],
        "depth": 2, "pool_safety": 1.2, "sync_scheme": "per-direction", "timeout_s": timeout,
        "match": "kdtree", "particles": "fluid", "far_bin_start": 8,
        "reference": {"name": "v6_K1", "version": "v6", "slabs": 1, "device_map": "1", "trials": 2},
        "tests": [{"name": test_name, "version": "v6", "slabs": 2, "device_map": "0,1",
                   "env": dict(SEAM_L2, **switches), "trials": 2}],
    }
    matrix_path = audit_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=1), encoding="utf-8")
    python = sys.executable
    base = [python, "-m", "experiment.seam_audit.run_matrix", "--matrix", str(matrix_path),
            "--out", str(audit_dir)]
    environment = base_environment()
    steps = [("tests", ["--roles", "test", "--no-analysis"]),
             ("requests", ["--build-capture-requests"]),
             ("references", ["--roles", "reference", "--no-analysis"]),
             ("analysis", ["--analyze-only"])]
    codes = {}
    for label, extra in steps:
        codes[label] = run(base + extra, environment, audit_dir / f"run_{label}.log", 4 * timeout)
    summary_path = audit_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rows = summary.get("rows", []) if isinstance(summary, dict) else summary
    rows = [row for row in rows if isinstance(row, dict) and row.get("test", "").startswith(test_name)]
    result = {"codes": codes, "pairs": [], "limit": AUDIT_LIMIT}
    worst = 0.0
    for row in rows:
        report_path = pathlib.Path(row["out_dir"]) / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        column0 = report["verdict"].get("column0_rms_ratio", {})
        ratios = {field: column0.get(field, {}).get("all") for field in AUDIT_FIELDS}
        pair = {"test_run": row["test_run"], "reference_run": row["reference_run"],
                "column0_rms_ratio": ratios,
                "second_pair_d0_rms_ratio": row.get("second_pair_d0_rms_ratio"),
                "far_bin_worst_rms_ratio": {field: value["value"] for field, value in
                                            report["verdict"]["far_bin_worst_rms_ratio"].items()},
                "window": row.get("window"), "all_runs_valid": row.get("all_runs_valid")}
        values = [value for value in ratios.values() if value is not None]
        values += [value for value in (row.get("second_pair_d0_rms_ratio") or {}).values()
                   if value is not None]
        window = row.get("window") or {}
        window_groups = {}
        for pair_label, pair_data in (window.items() if isinstance(window, dict) else ()):
            if not isinstance(pair_data, dict):
                continue
            for group in ("flagged", "departed", "arrived", "control"):
                entry = pair_data.get(group)
                if not isinstance(entry, dict) or not entry.get("n"):
                    continue
                for key in ("acceleration_rms", "shift_rms", "density_rms", "kernel_sum_rms"):
                    if entry.get(key) is not None:
                        values.append(entry[key])
                        window_groups[f"{pair_label} | {group} | {key}"] = entry[key]
        pair["window_groups"] = window_groups
        pair["worst"] = max(values) if values else None
        worst = max(worst, pair["worst"] or 0.0)
        result["pairs"].append(pair)
    result["worst"] = worst
    result["pass"] = (bool(rows) and all(code == 0 for code in codes.values())
                      and all(pair["all_runs_valid"] for pair in result["pairs"])
                      and worst <= AUDIT_LIMIT)
    return result


# --------------------------------------------------------------------------- single step
def run_single(out_dir: pathlib.Path, switches: dict, timeout: float) -> dict:
    single_root = out_dir / "single"
    directory = single_root / "cavity2d_1m"
    directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    runs = [("k1_a", 1, "1", {}, 0), ("k1_shuffled", 1, "1", {}, 1),
            ("keep1_layers2_t1", 2, "0,1", dict(SEAM_L2, **switches), 0),
            ("keep1_layers2_t2", 2, "0,1", dict(SEAM_L2, **switches), 0)]
    codes = {}
    for run_name, slabs, device_map, environment_extra, seed in runs:
        environment = base_environment()
        environment.update(environment_extra)
        command = [python, "-m", "experiment.seam_audit.single_step", "restart",
                   "--case", CASE_1M, "--out-dir", str(directory),
                   "--snapshot-dir", SNAPSHOT_DIRECTORY, "--run-name", run_name,
                   "--slabs", str(slabs), "--device-map", device_map, "--snapshot-step", "2000",
                   "--shuffle-seed", str(seed), "--steps", "1,2,5,10,50",
                   "--step1-window", "12", "--window", "12"]
        codes[run_name] = run(command, environment, directory / f"{run_name}.log", timeout)
    analyze = [python, "-m", "experiment.seam_audit.single_step", "analyze", "--out", str(single_root),
               "--cases", "cavity2d_1m", "--snapshots", "2000", "--steps", "1,2,5,10,50",
               "--snapshot-root", "logs/seam_audit/single_step"]
    codes["analyze"] = run(analyze, base_environment(), single_root / "analyze.log", timeout)
    analysis_path = single_root / "analysis_cavity2d_1m.json"
    result = {"codes": codes, "limit": SINGLE_LIMIT, "runs": {}}
    worst = 0.0
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        step = analysis["snapshots"]["2000"]["steps"]["1"]
        reconstruction = step.get("stale_ghost_reconstruction", {})
        noise = reconstruction.get("noise_acceleration_rms")
        for run_name in ("keep1_layers2_t1", "keep1_layers2_t2"):
            data = step["runs"].get(run_name, {})
            entry = {}
            for field in SINGLE_FIELDS:
                bins = data.get(field, {}).get("clean", [])
                for index in (0, 1):
                    if index < len(bins):
                        for key in ("ratio", "median_ratio"):
                            value = bins[index][key]
                            entry[f"{field}_c{index}_{key}"] = value
                            if value == value and value != float("inf"):
                                worst = max(worst, value)
                            elif value == float("inf"):
                                worst = max(worst, float("inf"))
            measured = (reconstruction.get("runs", {}).get(run_name, {})
                        .get("acceleration", {}).get("measured_rms"))
            if measured is not None and noise:
                entry["reconstruction_measured_over_noise"] = measured / noise
                worst = max(worst, measured / noise)
            result["runs"][run_name] = entry
        result["horizons"] = {
            key: {run_name: {"acceleration_c0_rms": value["runs"].get(run_name, {})
                             .get("acceleration", {}).get("clean", [{}])[0].get("ratio"),
                             "acceleration_c0_median": value["runs"].get(run_name, {})
                             .get("acceleration", {}).get("clean", [{}])[0].get("median_ratio")}
                  for run_name in ("keep1_layers2_t1",)}
            for key, value in analysis["snapshots"]["2000"]["steps"].items() if not value.get("missing")}
    result["worst"] = worst
    result["pass"] = (analysis_path.exists() and all(code == 0 for code in codes.values())
                      and worst <= SINGLE_LIMIT)
    return result


# --------------------------------------------------------------------------- K = 4 smoke
def run_k4(out_dir: pathlib.Path, switches: dict, timeout: float) -> dict:
    directory = out_dir / "k4"
    directory.mkdir(parents=True, exist_ok=True)
    environment = base_environment()
    environment.update(PRODUCTION_2D)
    environment.update(SEAM_L2)
    environment.update(switches)
    command = [sys.executable, "experiment/v6/_run_v6_chain_bench.py", "--case", CASE_1M,
               "--weights", "1,1,1,1", "--device-map", "0,1,0,1", "--max-steps", "1000",
               "--warmup", "200"]
    attempts = []
    for attempt in range(2):          # K=4 has stalled once in 16 runs on this rig (WDDM)
        code = run(command, environment, directory / f"chain_k4_attempt{attempt + 1}.log", timeout)
        attempts.append(code)
        if code == 0:
            break
    log_text = (directory / f"chain_k4_attempt{len(attempts)}.log").read_text(encoding="utf-8",
                                                                              errors="replace")
    tail = [line for line in log_text.splitlines() if "[chain_v6]" in line][-12:]
    return {"attempts": attempts, "pass": attempts[-1] == 0, "log_tail": tail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE switch (repeatable)")
    parser.add_argument("--steps", default="audit,single,k4")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    switches = dict(item.split("=", 1) for item in arguments.env)
    out_dir = pathlib.Path(arguments.out or f"logs/seam_audit/opt/validate_{arguments.name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8")) if verdict_path.exists() else {}
    verdict.update({"name": arguments.name, "switches": switches})
    for step in [item for item in arguments.steps.split(",") if item]:
        print(f"[opt_validate] {arguments.name}: {step}", flush=True)
        if step == "audit":
            verdict["audit"] = run_audit(out_dir, switches, arguments.name, arguments.timeout)
        elif step == "single":
            verdict["single"] = run_single(out_dir, switches, arguments.timeout)
        elif step == "k4":
            verdict["k4"] = run_k4(out_dir, switches, arguments.timeout)
        verdict_path.write_text(json.dumps(verdict, indent=1, default=str), encoding="utf-8")
        print(f"[opt_validate]   -> {step} pass={verdict[step]['pass']} "
              f"worst={verdict[step].get('worst')}", flush=True)
    verdict["pass"] = all(verdict[step]["pass"] for step in ("audit", "single", "k4") if step in verdict)
    verdict_path.write_text(json.dumps(verdict, indent=1, default=str), encoding="utf-8")
    print(f"[opt_validate] {arguments.name}: PASS={verdict['pass']}", flush=True)
    return 0 if verdict["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
