"""
e31_campaign.py — E31 local runs (docs/perf_model/E31.md), 2 x RTX 5090, production defaults, depth 2.

Modes (run in this order; every run id is skipped when already done, so a stopped campaign resumes):
  eta      (a) eta with the E31 cuts, trace off: 3-D narrow, 3-D 8M, 2-D 1M; 3 trials each; a trial = the
           K = 1 pair (GPU 0 and GPU 1 at the same time) and K = 2 (GPU 0,1), their order alternating
           (trial 1 and 3: K = 1 first; trial 2: K = 2 first)
  trace    (a) + (b) one K = 2 run with --step-trace --step-trace-detail full: 3-D narrow, 3-D 8M, 2-D 1M,
           2-D 10k (the phase-B -> C wait with the E31 cuts, and the per-kernel phase C)
  phasec   (b) the K = 1 reference of the same, full ticks, GPU 1 (no desktop): 2-D 10k, 2-D 1M, 3-D 8M
  nowait   (c) K = 2, V6_PHASE_A_NO_WAIT = 0 / 1 alternating, 3 pairs each (trace off): 2-D 10k, 2-D 1M;
           then one traced run (phases detail) per case and setting
Steps 3000 with warmup 1000 (steady = last 2000), as E29; 3-D 8M 2000 with warmup 1000 (machine time).
Before every timed run nvidia-smi must show no python compute process on any GPU and no compute process on a
GPU without a display (step_trace_campaign.gpu_state); the check is logged with the run.

    .venv/Scripts/python.exe -m experiment.v6.analysis.e31_campaign --out logs/e31/campaign [--modes eta,trace]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

from experiment.v6.analysis.step_trace_campaign import (
    BENCH,
    REPOSITORY,
    environment,
    finish,
    wait_for_idle_gpus,
)

CASES = {   # name: (case yaml, dimension, steps, warmup)
    "2d_10k": ("cases/cavity_2d_10k/case.yaml", 2, 3000, 1000),
    "2d_1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, 3000, 1000),
    "3d_narrow": ("cases/cavity3d_narrow/case.yaml", 3, 3000, 1000),
    "3d_8m": ("cases/cavity3d_8m/case.yaml", 3, 2000, 1000),
}
ETA_CASES = ("3d_narrow", "3d_8m", "2d_1m")
TRACE_CASES = ("3d_narrow", "3d_8m", "2d_1m", "2d_10k")
PHASEC_K1_CASES = ("2d_10k", "2d_1m", "3d_8m")
NOWAIT_CASES = ("2d_10k", "2d_1m")
MODES = ("eta", "trace", "phasec", "nowait")


def command(name: str, weights: str, device_map: str, trace_dir=None, detail: str = "phases") -> list:
    case_path, _, steps, warmup = CASES[name]
    arguments = [sys.executable, BENCH, "--case", case_path, "--weights", weights, "--device-map", device_map,
                 "--max-steps", str(steps), "--warmup", str(warmup)]
    if trace_dir is not None:
        arguments += ["--step-trace", str(trace_dir), "--step-trace-detail", detail]
    return arguments


def launch(arguments, log_path: pathlib.Path, extra_env: dict):
    env = environment()
    env.update(extra_env)
    handle = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(arguments, cwd=REPOSITORY, env=env, stdout=handle, stderr=subprocess.STDOUT), handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default=",".join(MODES))
    arguments = parser.parse_args()
    out = (REPOSITORY / arguments.out).resolve()
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    log_file = open(out / "campaign.log", "a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    done = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("rc") == 0:
                done.add(record["run_id"])

    def run_one(run_id: str, name: str, kind: str, weights: str, device_map: str, extra: dict,
                trace_detail=None, **fields) -> None:
        if run_id in done:
            return
        state = wait_for_idle_gpus(log)
        log(f"start {run_id}: gpu check {json.dumps(state)}")
        stem = run_id.replace("/", "__")
        trace = runs / stem if trace_detail else None
        process, handle = launch(command(name, weights, device_map, trace, trace_detail or "phases"),
                                 runs / f"{stem}.log", fields.get("env", {}))
        record = {"run_id": run_id, "case": name, "dimension": CASES[name][1], "kind": kind, "gpu_check": state,
                  "trace_dir": f"runs/{stem}" if trace else None, "log": f"runs/{stem}.log", **extra}
        finish(process, handle, runs / f"{stem}.log", record, results)

    def run_pair(run_id: str, name: str, extra: dict, trace_detail=None) -> None:
        """K = 1 on GPU 0 and GPU 1 at the same time."""
        ids = [f"{run_id}/g{gpu}" for gpu in (0, 1)]
        if all(item in done for item in ids):
            return
        state = wait_for_idle_gpus(log)
        log(f"start {run_id} (GPU 0 and GPU 1 at once): gpu check {json.dumps(state)}")
        pending = []
        for gpu, item in zip((0, 1), ids):
            stem = item.replace("/", "__")
            trace = runs / stem if trace_detail else None
            process, handle = launch(command(name, "1", str(gpu), trace, trace_detail or "phases"),
                                     runs / f"{stem}.log", {})
            pending.append((process, handle, stem, gpu, item, trace))
        for process, handle, stem, gpu, item, trace in pending:
            finish(process, handle, runs / f"{stem}.log",
                   {"run_id": item, "case": name, "dimension": CASES[name][1], "kind": "k1", "gpu": gpu,
                    "gpu_check": state, "trace_dir": f"runs/{stem}" if trace else None, "log": f"runs/{stem}.log",
                    **extra}, results)

    modes = arguments.modes.split(",")
    if "eta" in modes:
        for name in ETA_CASES:
            for trial in (1, 2, 3):
                order = ("k1", "k2") if trial % 2 == 1 else ("k2", "k1")
                for kind in order:
                    extra = {"mode": "eta", "trial": trial, "trace": False}
                    if kind == "k1":
                        run_pair(f"eta/{name}/t{trial}/k1", name, extra)
                    else:
                        run_one(f"eta/{name}/t{trial}/k2", name, "k2", "1,1", "0,1", extra)
    if "trace" in modes:
        for name in TRACE_CASES:
            run_one(f"trace/{name}/k2", name, "k2", "1,1", "0,1", {"mode": "trace", "trial": 1, "trace": True},
                    trace_detail="full")
    if "phasec" in modes:
        for name in PHASEC_K1_CASES:
            run_one(f"phasec/{name}/k1", name, "k1", "1", "1", {"mode": "phasec", "trial": 1, "trace": True, "gpu": 1},
                    trace_detail="full")
    if "nowait" in modes:
        for name in NOWAIT_CASES:
            for trial in (1, 2, 3):
                order = ("0", "1") if trial % 2 == 1 else ("1", "0")
                for setting in order:
                    run_one(f"nowait/{name}/t{trial}/w{setting}", name, "k2", "1,1", "0,1",
                            {"mode": "nowait", "trial": trial, "trace": False, "phase_a_no_wait": int(setting)},
                            env={"V6_PHASE_A_NO_WAIT": setting})
            for setting in ("0", "1"):
                run_one(f"nowait/{name}/trace/w{setting}", name, "k2", "1,1", "0,1",
                        {"mode": "nowait", "trial": 0, "trace": True, "phase_a_no_wait": int(setting)},
                        trace_detail="phases", env={"V6_PHASE_A_NO_WAIT": setting})
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
