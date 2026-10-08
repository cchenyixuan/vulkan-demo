"""
step_trace_campaign.py — E29 local runs with the step trace (docs/perf_model/E29_local.md).

Modes:
  overhead  2-D 1M and 2-D 16M, K = 2, the trace off / on in 3 alternating pairs
            (trial 1 off -> on, trial 2 on -> off, trial 3 off -> on).
  scan      every case, in this order: K = 1 references on GPU 0 and GPU 1 at the same
            time (two processes) and K = 2 on GPUs 0,1, both without the trace (their
            fps give eta); then the same two with --step-trace (time decomposition;
            the K = 1 traces give c_B = T_B / N).
Production defaults (no V7_* set), depth 2, 3000 steps, warmup 1000 (steady =
the last 2000). Before every timed run nvidia-smi must show no python compute
process on any GPU (another session's run) and no compute process at all on a
GPU without a display (the desktop's GPU always lists explorer, browsers, ...,
some as "[Insufficient Permissions]"); otherwise the driver waits (5 s settle,
then 15 s slices) and logs it. Utilization, power and SM clock of both GPUs are
logged with every check (GPU 0 drives the desktop, so it never reads 0 %); they
are not sampled during the runs. One line per run in OUT/results.jsonl, the bench
log and the trace directory under OUT/runs/. A run that fails is re-run on resume
(both GPUs of a K = 1 pair); the analysis keeps the last successful record per run
id. Records may carry "trial" (default 1) for repeated off sets; this driver runs
one.

    .venv/Scripts/python.exe -m experiment.v7.analysis.step_trace_campaign --mode overhead --out logs/e29_step_trace/overhead
    .venv/Scripts/python.exe -m experiment.v7.analysis.step_trace_campaign --mode scan --out logs/e29_step_trace/scan
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]
BENCH = "experiment/v7/_run_v7_chain_bench.py"
# name: (case yaml, dimension)
CASES = {
    "2d_10k": ("logs/two_hop_experiment/cases/cavity_2d_10k/case.yaml", 2),
    "2d_62k": ("cases/cavity_2d_62k/case.yaml", 2),
    "2d_250k": ("cases/cavity_2d_250k/case.yaml", 2),
    "2d_1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 2),
    "2d_4m": ("cases/lid_driven_cavity_2d_4m/case.yaml", 2),
    "2d_16m": ("cases/lid_driven_cavity_2d_16m/case.yaml", 2),
    "3d_1m": ("cases/cavity3d_1m/case.yaml", 3),
    "3d_narrow": ("cases/cavity3d_narrow/case.yaml", 3),
    "3d_8m": ("cases/cavity3d_8m/case.yaml", 3),
}
OVERHEAD_CASES = ("2d_1m", "2d_16m")
STEPS, WARMUP = 3000, 1000


def environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("V5_", "V7_"))}
    env.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8"})
    return env


def gpu_state() -> dict:
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, errors="replace").stdout
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm,uuid,"
                           "display_active", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, errors="replace").stdout
    rows = [[part.strip() for part in line.split(",")] for line in gpus.splitlines() if line.strip()]
    headless = {row[5] for row in rows if len(row) > 6 and row[6].lower() != "enabled"}
    app_rows = [[part.strip() for part in line.split(",")] for line in apps.splitlines() if line.strip()]
    python = [", ".join(row[:2]) for row in app_rows if "python" in ",".join(row[:2]).lower()]
    on_headless = [", ".join(row[:2]) for row in app_rows if len(row) > 2 and row[2] in headless]
    utilization = [float(row[1]) for row in rows if len(row) > 1 and row[1].replace(".", "").isdigit()]
    # A python compute process anywhere (another session's solver run) or any compute process on a GPU
    # without a display blocks. GPU 0 drives the desktop: its utilization never drops to 0 and it always
    # lists desktop apps, so those alone do not block (logged with every check instead).
    return {"python_compute": python, "headless_compute": sorted(set(on_headless) - set(python)),
            "gpus": [row[:5] for row in rows], "utilization": utilization, "busy": bool(python or on_headless)}


def wait_for_idle_gpus(log) -> dict:
    """nvidia-smi's utilization lags by about a second, so a run that just ended
    still shows up: settle 5 s first, then retry every 15 s while busy."""
    time.sleep(5)
    waited = 5
    while True:
        state = gpu_state()
        if not state["busy"]:
            state["waited_s"] = waited
            return state
        log(f"GPU busy, waiting 15 s: python compute {state['python_compute']} on a headless GPU "
            f"{state['headless_compute']} gpus {state['gpus']}")
        time.sleep(15)
        waited += 15


def parse_log(text: str) -> dict:
    steady = re.search(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    final = re.search(r"\[chain_v7\] final: total=([\d,]+) \(expected ([\d,]+)\) drift=(-?\d+) "
                      r"stamp_errors gpu=(\d+) host=(\d+) overflow_total=(\d+) far_migration_total=(\d+)", text)
    return {"steady_fps": float(steady.group(3)) if steady else None,       # printed with 0.1 fps
            "steady_steps": int(steady.group(1)) if steady else None,
            "steady_s": float(steady.group(2)) if steady else None,         # 0.01 s
            "drift": int(final.group(3)) if final else None,
            "stamp_errors": (int(final.group(4)) + int(final.group(5))) if final else None,
            "overflow_total": int(final.group(6)) if final else None,
            "far_migration_total": int(final.group(7)) if final else None,
            "validation_failed": "VALIDATION FAILED" in text}


def command(case_path: str, weights: str, device_map: str, trace_dir) -> list:
    arguments = [sys.executable, BENCH, "--case", case_path, "--weights", weights, "--device-map", device_map,
                 "--max-steps", str(STEPS), "--warmup", str(WARMUP)]
    if trace_dir is not None:
        arguments += ["--step-trace", str(trace_dir)]
    return arguments


def launch(arguments, log_path: pathlib.Path):
    handle = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(arguments, cwd=REPOSITORY, env=environment(), stdout=handle,
                            stderr=subprocess.STDOUT), handle


def finish(process, handle, log_path: pathlib.Path, record: dict, results: pathlib.Path) -> dict:
    record["rc"] = process.wait()
    handle.close()
    record.update(parse_log(log_path.read_text(encoding="utf-8", errors="replace")))
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overhead", "scan"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=None, help="comma list (default: all of the mode)")
    arguments = parser.parse_args()
    out = (REPOSITORY / arguments.out).resolve() if not pathlib.Path(arguments.out).is_absolute() \
        else pathlib.Path(arguments.out)
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
    if arguments.mode == "overhead":
        names = arguments.cases.split(",") if arguments.cases else list(OVERHEAD_CASES)
        for name in names:
            case_path, dimension = CASES[name]
            for trial in (1, 2, 3):
                for arm in (("off", "on") if trial % 2 == 1 else ("on", "off")):
                    run_id = f"{name}/{arm}/t{trial}"
                    if run_id in done:
                        continue
                    state = wait_for_idle_gpus(log)
                    log(f"start {run_id}: gpu check {json.dumps(state)}")
                    stem = f"{name}__{arm}__t{trial}"
                    trace = runs / stem if arm == "on" else None
                    process, handle = launch(command(case_path, "1,1", "0,1", trace), runs / f"{stem}.log")
                    finish(process, handle, runs / f"{stem}.log",
                           {"run_id": run_id, "case": name, "case_path": case_path, "dimension": dimension,
                            "kind": "k2", "arm": arm, "trial": trial, "gpu_check": state,
                            "trace_dir": str(trace.relative_to(out)) if trace else None,
                            "log": f"runs/{stem}.log"}, results)
    else:
        names = arguments.cases.split(",") if arguments.cases else list(CASES)
        for name in names:
            case_path, dimension = CASES[name]
            # trace off first (fps for eta), then trace on (time decomposition, c_B)
            for trace in (False, True):
                tag = "on" if trace else "off"
                if f"{name}/k1/{tag}" not in done:
                    state = wait_for_idle_gpus(log)
                    log(f"start {name}/k1/{tag} (GPU 0 and GPU 1 at once): gpu check {json.dumps(state)}")
                    pending = []
                    for gpu in (0, 1):
                        stem = f"{name}__k1__{tag}__g{gpu}"
                        process, handle = launch(command(case_path, "1", str(gpu), runs / stem if trace else None),
                                                 runs / f"{stem}.log")
                        pending.append((process, handle, stem, gpu))
                    records = []
                    for process, handle, stem, gpu in pending:
                        records.append(finish(process, handle, runs / f"{stem}.log",
                                              {"run_id": f"{name}/k1/{tag}/g{gpu}", "case": name,
                                               "case_path": case_path, "dimension": dimension, "kind": "k1",
                                               "trace": trace, "gpu": gpu, "gpu_check": state,
                                               "trace_dir": f"runs/{stem}" if trace else None,
                                               "log": f"runs/{stem}.log"}, results))
                    if all(record["rc"] == 0 for record in records):
                        with open(results, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"run_id": f"{name}/k1/{tag}", "rc": 0, "marker": True}) + "\n")
                if f"{name}/k2/{tag}" not in done:
                    state = wait_for_idle_gpus(log)
                    log(f"start {name}/k2/{tag}: gpu check {json.dumps(state)}")
                    stem = f"{name}__k2__{tag}"
                    process, handle = launch(command(case_path, "1,1", "0,1", runs / stem if trace else None),
                                             runs / f"{stem}.log")
                    finish(process, handle, runs / f"{stem}.log",
                           {"run_id": f"{name}/k2/{tag}", "case": name, "case_path": case_path,
                            "dimension": dimension, "kind": "k2", "trace": trace, "gpu_check": state,
                            "trace_dir": f"runs/{stem}" if trace else None, "log": f"runs/{stem}.log"}, results)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
