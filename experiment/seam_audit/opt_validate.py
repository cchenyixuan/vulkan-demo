"""
opt_validate.py - correctness gate for one v6 optimisation switch set
(docs/seam_audit/v6_opt.md). Runs, in order:

  audit   seam audit K=1 vs K=2, 2-D 1M, N = 2000, seam configuration (1,2)
          plus the switches: two v6 test runs with the 400-frame crossing
          window, capture requests, two fresh v6 K=1 references (v6 K=1 = v5 K=1; avoids the v5 working tree), analysis
          (run_matrix.py two-pass procedure).
          PASS: both test trials and both references produced valid dumps
          (every invariant 0, far_migration_count included), the analysis row is
          complete (no partial inputs), the column-0 rms ratios of the first pair
          (acceleration, shift, velocity, density, pressure, kernel_sum) and of
          the second pair (acceleration, shift, density: analyze.PRIMARY_FIELDS)
          and every crossing-window group of both pairs are <= AUDIT_LIMIT; a
          missing, NaN or infinite statistic fails the step.
          Density (E14): the test runs AND the K=1 references run with
          V6_DELTA_DENSITY=1 by default and the dumps keep rho in float64
          (dump_state). With rho ~ 1000 in float32 the K1 - K1 noise
          (~3e-6 kg/m^3) is far below the 6.1e-5 float32 spacing, so the
          plain-density ratios were quotients of quantised values. An explicit
          V6_DELTA_DENSITY in --env sets both sides (--reference-env can still
          override the references); V6_DELTA_DENSITY=0 is the old audit.
  single  single-step test from the kept 2-D 1M N = 2000 snapshot: K=1
          reference + K=1 shuffled-order noise + two (1,2)+switch restarts,
          k = 1, 2, 5, 10, 50.
          The (1,2) restarts run with the production transport (count-aware
          worker, split transfer queues) like the audit and K=4 steps.
          PASS (k = 1): for both (1,2) runs, column 0 and column 1 rms and
          median ratios of a, shift, rho, p, kernel_sum, L <= SINGLE_LIMIT
          (exactly-equal bins count as passing; NaN fails), and the
          reconstruction's measured rms / noise <= SINGLE_LIMIT. Note: from this
          snapshot the first seam crossing is at k = 5, so this step does not
          exercise the migrant path; experiment/seam_audit/ab_restart.py does.
  k4      K=4 smoke (device map 0,1,0,1), 2-D 1M, 1000 frames, chain bench
          with its seam / overflow / stamp / far_migration checks. A second
          attempt is made only when the first stalled (no final line); any
          '*** VALIDATION FAILED ***' fails the step.

--reference-env KEY=VALUE applies to the K=1 references of the audit and the
single-step test (switches that change the arithmetic, e.g. V6_DELTA_DENSITY,
must run on both sides). A rerun into an existing output directory needs
--fresh (the previous directory is renamed, never reused).

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
# E14 (2026-10-05): the audit runs with delta density on both sides unless --env sets V6_DELTA_DENSITY
AUDIT_DELTA_DENSITY_DEFAULT = "1"
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
    """Caller's environment without V6_* plus the pre-E6b defaults
    (partition_v6.LEGACY_DEFAULTS): every gate step runs (1,2) production + ONLY
    the switch set under test (--env overrides)."""
    from experiment.v6.utils.partition_v6 import LEGACY_DEFAULTS
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
    environment.update(LEGACY_DEFAULTS)
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


# --------------------------------------------------------------------------- audit
def _number(value):
    """Ratio values from the analysis JSON: None / 'inf' / NaN -> inf (a failure)."""
    if value is None:
        return float("inf")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if number == number else float("inf")


def run_audit(out_dir: pathlib.Path, switches: dict, name: str, timeout: float,
              reference_switches: dict | None = None) -> dict:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    test_name = f"v6_l2_{name}"
    # delta density on both sides (see the module docstring); the references follow the test side
    delta_density = switches.get("V6_DELTA_DENSITY", AUDIT_DELTA_DENSITY_DEFAULT)
    switches = dict(switches, V6_DELTA_DENSITY=delta_density)
    reference_switches = dict({"V6_DELTA_DENSITY": delta_density}, **(reference_switches or {}))
    matrix = {
        "description": f"optimisation gate {name}: v6 (1,2) + {switches} vs fresh v6 K=1 references",
        "cases": [{"name": "cavity2d_1m", "path": CASE_1M, "dimension": 2}],
        "horizons": [2000], "window": 400, "window_horizons": [2000],
        "depth": 2, "pool_safety": 1.2, "sync_scheme": "per-direction", "timeout_s": timeout,
        "match": "kdtree", "particles": "fluid", "far_bin_start": 8,
        "reference": {"name": "v6_K1", "version": "v6", "slabs": 1, "device_map": "1", "trials": 2,
                      "env": dict(reference_switches or {})},
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
    problems = []
    # every run produced a valid dump (invariants incl. far_migration_count)
    dump_directory = audit_dir / "dumps" / "cavity2d_1m"
    expected = [f"{test_name}_K2_t1", f"{test_name}_K2_t2"]
    expected += [path.stem.replace("_N2000", "") for path in sorted(dump_directory.glob("reference_*_N2000.json"))]
    if len([item for item in expected if item.startswith("reference_")]) < 2:
        problems.append("fewer than two reference dumps")
    for run_name in expected:
        sidecar = dump_directory / f"{run_name}_N2000.json"
        if not sidecar.exists():
            problems.append(f"{run_name}: no N=2000 dump")
            continue
        invariants = json.loads(sidecar.read_text(encoding="utf-8")).get("invariants", {})
        if not invariants.get("valid") or invariants.get("far_migration_count", 0) != 0:
            problems.append(f"{run_name}: invariants {invariants}")
    if not rows:
        problems.append("no analysis row")
    for row in rows:
        if row.get("status") != "ok" or row.get("missing"):
            problems.append(f"row {row.get('test_run')}: status {row.get('status')} missing {row.get('missing')}")
        if row.get("window_error") or not row.get("window"):
            problems.append(f"row {row.get('test_run')}: crossing window missing ({row.get('window_error')})")
        if not row.get("second_pair_d0_rms_ratio"):
            problems.append(f"row {row.get('test_run')}: second pair missing")
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
        values = [_number(value) for value in ratios.values()]
        values += [_number(value) for value in (row.get("second_pair_d0_rms_ratio") or {}).values()]
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
                    values.append(_number(entry.get(key)))
                    window_groups[f"{pair_label} | {group} | {key}"] = entry.get(key)
        pair["window_groups"] = window_groups
        pair["worst"] = max(values) if values else None
        worst = max(worst, pair["worst"] or 0.0)
        result["pairs"].append(pair)
    result["worst"] = worst
    result["problems"] = problems
    result["pass"] = (bool(rows) and not problems and all(code == 0 for code in codes.values())
                      and all(pair["all_runs_valid"] for pair in result["pairs"])
                      and worst <= AUDIT_LIMIT)
    return result


# --------------------------------------------------------------------------- single step
def run_single(out_dir: pathlib.Path, switches: dict, timeout: float,
               reference_switches: dict | None = None) -> dict:
    single_root = out_dir / "single"
    directory = single_root / "cavity2d_1m"
    directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    reference = dict(reference_switches or {})
    production = dict(PRODUCTION_2D, **SEAM_L2)
    runs = [("k1_a", 1, "1", reference, 0), ("k1_shuffled", 1, "1", reference, 1),
            ("keep1_layers2_t1", 2, "0,1", dict(production, **switches), 0),
            ("keep1_layers2_t2", 2, "0,1", dict(production, **switches), 0)]
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
                            else:          # NaN or inf: a failure
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
    validation_failed = False
    for attempt in range(2):          # K=4 has stalled once in 16 runs on this rig (WDDM)
        log_path = directory / f"chain_k4_attempt{attempt + 1}.log"
        code = run(command, environment, log_path, timeout)
        attempts.append(code)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "*** VALIDATION FAILED ***" in text:
            validation_failed = True
            break                     # a real failure is never retried away
        stalled = "[chain_v6] final:" not in text
        if code == 0 or not stalled:
            break
    log_text = (directory / f"chain_k4_attempt{len(attempts)}.log").read_text(encoding="utf-8",
                                                                              errors="replace")
    tail = [line for line in log_text.splitlines() if "[chain_v6]" in line][-12:]
    return {"attempts": attempts, "validation_failed": validation_failed,
            "pass": attempts[-1] == 0 and not validation_failed, "log_tail": tail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE switch (repeatable)")
    parser.add_argument("--steps", default="audit,single,k4")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--reference-env", action="append", default=[],
                        help="KEY=VALUE for the K=1 references too (switches that change the arithmetic)")
    parser.add_argument("--fresh", action="store_true",
                        help="rename an existing output directory instead of refusing to run")
    arguments = parser.parse_args()
    switches = dict(item.split("=", 1) for item in arguments.env)
    reference_switches = dict(item.split("=", 1) for item in arguments.reference_env)
    out_dir = pathlib.Path(arguments.out or f"logs/seam_audit/opt/validate_{arguments.name}").resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not arguments.fresh:
            print(f"[opt_validate] {out_dir} exists; rerun with --fresh (it is renamed, never reused)", flush=True)
            return 2
        previous = out_dir.with_name(out_dir.name + time.strftime(".previous-%Y%m%d-%H%M%S"))
        out_dir.rename(previous)
        print(f"[opt_validate] previous results moved to {previous}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "verdict.json"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=_REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "experiment/v6"], capture_output=True,
                                text=True, cwd=_REPO_ROOT).stdout.strip())
    verdict = {"name": arguments.name, "switches": switches, "reference_switches": reference_switches,
               "head": head, "experiment_v6_dirty": dirty}
    for step in [item for item in arguments.steps.split(",") if item]:
        print(f"[opt_validate] {arguments.name}: {step}", flush=True)
        if step == "audit":
            verdict["audit"] = run_audit(out_dir, switches, arguments.name, arguments.timeout,
                                         reference_switches)
        elif step == "single":
            verdict["single"] = run_single(out_dir, switches, arguments.timeout, reference_switches)
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
