"""
e32_campaign.py — E32 part 2 local runs (docs/perf_model/E32.md), 2 x RTX 5090, code defaults (release set,
phase A no-wait), depth 2.

Modes (in this order; every run id is skipped when already done, so a stopped campaign resumes):
  calibrate  per case one pilot calibration (--weights auto --calibrate-only): OUT/weights/<case>.json. As
             E7 will do it: calibrate once per (case, K, node), every trial reuses the file, the pilot is never
             inside a timing window.
  eta        (a) 2-D 1M, 3-D 8M, 3-D narrow at K = 2, trace off, 3 trials; a trial = the K = 1 pair (GPU 0 and
             GPU 1 at the same time), K = 2 equal weights and K = 2 with the calibrated weights (--weights-file),
             the order rotating (trial 1: K=1, equal, auto; trial 2: auto, equal, K=1; trial 3: K=1, auto,
             equal)
  trace      (a) one --step-trace run (phases detail) per case and weighting: the per-sim B -> C waits
  k3         (b) 2-D 4M, K = 3, device map 0,0,1 (two sims share GPU 0): calibration, then equal and calibrated
             weights alternating, 2 each (algorithm check, not performance data)
  tracecost  diagnosis of (a): 3-D 8M, K = 1 on GPU 0 and GPU 1 at the same time, without and with the step trace
             (phase ticks, as the pilot reads them), 2 trials alternating (trial 1: off, on; trial 2: on, off);
             1500 steps, warmup 500
  matched    diagnosis of (a): 3-D 8M calibrated again (weights/3d_8m_k2_matched.json) and right after it K = 2
             equal and calibrated weights alternating, 3 each (equal, auto | auto, equal | equal, auto), trace off;
             GPU 0 drives the desktop, whose state (awake / asleep, logged with every idle check) changes the
             card's speed, so the pilot and the timed runs here share one state
Steps 3000 with warmup 1000 (steady = last 2000), 3-D 8M 2000 with warmup 1000, as E31. Before every timed run
nvidia-smi must show no python compute process on any GPU and no compute process on a GPU without a display
(step_trace_campaign.gpu_state); the check is logged with the run.

    .venv/Scripts/python.exe -m experiment.v7.analysis.e32_campaign --out logs/e32/campaign [--modes calibrate,eta]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from experiment.v7.analysis.step_trace_campaign import (
    BENCH,
    REPOSITORY,
    environment,
    finish,
    wait_for_idle_gpus,
)

CASES = {   # name: (case yaml, dimension, steps, warmup)
    "2d_1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, 3000, 1000),
    "3d_narrow": ("cases/cavity3d_narrow/case.yaml", 3, 3000, 1000),
    "3d_8m": ("cases/cavity3d_8m/case.yaml", 3, 2000, 1000),
    "2d_4m": ("cases/lid_driven_cavity_2d_4m/case.yaml", 2, 3000, 1000),
}
ETA_CASES = ("2d_1m", "3d_narrow", "3d_8m")
K3_CASE, K3_MAP = "2d_4m", "0,0,1"
TRIAL_ORDERS = {1: ("k1", "equal", "auto"), 2: ("auto", "equal", "k1"), 3: ("k1", "auto", "equal")}
MODES = ("calibrate", "eta", "trace", "k3", "tracecost", "matched")
TRACECOST_CASE, TRACECOST_STEPS, TRACECOST_WARMUP = "3d_8m", 1500, 500


def command(name: str, weighting: list, trace_dir=None, extra=()) -> list:
    case_path, _, steps, warmup = CASES[name]
    arguments = [sys.executable, BENCH, "--case", case_path, *weighting, "--max-steps", str(steps),
                 "--warmup", str(warmup), *extra]
    if trace_dir is not None:
        arguments += ["--step-trace", str(trace_dir)]
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default=",".join(MODES))
    arguments = parser.parse_args()
    out = (REPOSITORY / arguments.out).resolve()
    runs, weights = out / "runs", out / "weights"
    runs.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
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

    def launch(arguments_list, log_path):
        handle = open(log_path, "w", encoding="utf-8")
        return subprocess.Popen(arguments_list, cwd=REPOSITORY, env=environment(), stdout=handle,
                                stderr=subprocess.STDOUT), handle

    def run_one(run_id: str, name: str, weighting: list, record_fields: dict, trace=False, extra=()) -> None:
        if run_id in done:
            return
        state = wait_for_idle_gpus(log)
        log(f"start {run_id}: gpu check {json.dumps(state)}")
        stem = run_id.replace("/", "__")
        trace_dir = runs / stem if trace else None
        process, handle = launch(command(name, weighting, trace_dir, extra), runs / f"{stem}.log")
        record = {"run_id": run_id, "case": name, "dimension": CASES[name][1], "gpu_check": state,
                  "trace_dir": f"runs/{stem}" if trace else None, "log": f"runs/{stem}.log", **record_fields}
        finish(process, handle, runs / f"{stem}.log", record, results)

    def run_pair(run_id: str, name: str, record_fields: dict, trace=False, steps=None) -> None:
        """K = 1 on GPU 0 and GPU 1 at the same time."""
        ids = [f"{run_id}/g{gpu}" for gpu in (0, 1)]
        if all(item in done for item in ids):
            return
        state = wait_for_idle_gpus(log)
        log(f"start {run_id} (GPU 0 and GPU 1 at once): gpu check {json.dumps(state)}")
        pending = []
        for gpu, item in zip((0, 1), ids):
            stem = item.replace("/", "__")
            trace_dir = runs / stem if trace else None
            arguments_list = command(name, ["--weights", "1", "--device-map", str(gpu)], trace_dir)
            if steps is not None:
                arguments_list[arguments_list.index("--max-steps") + 1] = str(steps[0])
                arguments_list[arguments_list.index("--warmup") + 1] = str(steps[1])
            process, handle = launch(arguments_list, runs / f"{stem}.log")
            pending.append((process, handle, stem, gpu, item, trace_dir))
        for process, handle, stem, gpu, item, trace_dir in pending:
            finish(process, handle, runs / f"{stem}.log",
                   {"run_id": item, "case": name, "dimension": CASES[name][1], "kind": "k1", "gpu": gpu,
                    "gpu_check": state, "trace_dir": f"runs/{stem}" if trace_dir else None,
                    "log": f"runs/{stem}.log", **record_fields}, results)

    def weights_file(name: str, slab_count: int) -> str:
        return str((weights / f"{name}_k{slab_count}.json").relative_to(REPOSITORY)).replace("\\", "/")

    def calibrate(name: str, device_map: str) -> None:
        slab_count = len(device_map.split(","))
        run_id = f"calibrate/{name}/k{slab_count}"
        run_one(run_id, name, ["--weights", "auto", "--device-map", device_map, "--weights-file",
                               weights_file(name, slab_count), "--calibrate-only"],
                {"mode": "calibrate", "kind": f"k{slab_count}", "arm": "calibrate"})

    modes = arguments.modes.split(",")
    if "calibrate" in modes:
        for name in ETA_CASES:
            calibrate(name, "0,1")
    if "eta" in modes:
        for name in ETA_CASES:
            for trial in (1, 2, 3):
                for kind in TRIAL_ORDERS[trial]:
                    fields = {"mode": "eta", "trial": trial, "trace": False}
                    if kind == "k1":
                        run_pair(f"eta/{name}/t{trial}/k1", name, {**fields, "arm": "k1"})
                    elif kind == "equal":
                        run_one(f"eta/{name}/t{trial}/equal", name, ["--weights", "1,1", "--device-map", "0,1"],
                                {**fields, "kind": "k2", "arm": "equal"})
                    else:
                        run_one(f"eta/{name}/t{trial}/auto", name, ["--weights-file", weights_file(name, 2),
                                                                    "--device-map", "0,1"],
                                {**fields, "kind": "k2", "arm": "auto", "weights_file": weights_file(name, 2)})
    if "trace" in modes:
        for name in ETA_CASES:
            for arm in ("equal", "auto"):
                weighting = (["--weights", "1,1", "--device-map", "0,1"] if arm == "equal"
                             else ["--weights-file", weights_file(name, 2), "--device-map", "0,1"])
                run_one(f"trace/{name}/{arm}", name, weighting,
                        {"mode": "trace", "trial": 1, "trace": True, "kind": "k2", "arm": arm}, trace=True)
    if "k3" in modes:
        calibrate(K3_CASE, K3_MAP)
        for trial in (1, 2):
            for arm in ("equal", "auto"):
                weighting = (["--weights", "1,1,1", "--device-map", K3_MAP] if arm == "equal"
                             else ["--weights-file", weights_file(K3_CASE, 3), "--device-map", K3_MAP])
                run_one(f"k3/{K3_CASE}/t{trial}/{arm}", K3_CASE, weighting,
                        {"mode": "k3", "trial": trial, "trace": False, "kind": "k3", "arm": arm})
    if "tracecost" in modes:
        for trial, order in ((1, (False, True)), (2, (True, False))):
            for traced in order:
                label = "on" if traced else "off"
                run_pair(f"tracecost/{TRACECOST_CASE}/t{trial}/{label}", TRACECOST_CASE,
                         {"mode": "tracecost", "trial": trial, "trace": traced, "arm": label}, trace=traced,
                         steps=(TRACECOST_STEPS, TRACECOST_WARMUP))
    if "matched" in modes:
        matched_file = str((weights / "3d_8m_k2_matched.json").relative_to(REPOSITORY)).replace("\\", "/")
        run_one("matched/3d_8m/calibrate", "3d_8m", ["--weights", "auto", "--device-map", "0,1", "--weights-file",
                                                      matched_file, "--calibrate-only"],
                {"mode": "matched", "kind": "k2", "arm": "calibrate"})
        for trial, order in ((1, ("equal", "auto")), (2, ("auto", "equal")), (3, ("equal", "auto"))):
            for arm in order:
                weighting = (["--weights", "1,1", "--device-map", "0,1"] if arm == "equal"
                             else ["--weights-file", matched_file, "--device-map", "0,1"])
                run_one(f"matched/3d_8m/t{trial}/{arm}", "3d_8m", weighting,
                        {"mode": "matched", "trial": trial, "trace": False, "kind": "k2", "arm": arm})
    log("campaign finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
