"""
_run_single_baseline_campaign.py — single-GPU baseline campaign for the paper.

Compares the V0 reference solver (utils/sph) with the multi-GPU V5 solver in
single-GPU mode (experiment/v5) on ONE headless RTX 5090 (pinned by UUID), for
the 2-D lid-driven cavity at 1M..32M particles, at 1 and 2 frames in flight,
over 3 interleaved trials. Every run is its own subprocess of
_run_single_baseline_bench.py (fresh driver state, crash isolation).

Outputs (logs/single_baseline_<date>/):
  results.jsonl    one record per run, appended immediately (resumable:
                   finished (size, solver, in_flight, trial) runs are skipped)
  telemetry.csv    nvidia-smi 1 Hz samples (SM/mem clock, power, temp, util,
                   VRAM) for the whole campaign — join on epoch_start/end
  runs/*.log       full stdout+stderr of each bench subprocess
  campaign.log     this driver's progress (tee'd by the caller)

Usage:
    .venv/Scripts/python.exe _run_single_baseline_campaign.py [--out DIR]
        [--trials 3] [--sizes 1m,2m,...] [--configs v0:1,v0:2,v5:1,v5:2]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parent
_PYTHON = sys.executable
_ENV = {**os.environ, "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation"}

# Headless RTX 5090: nvidia-smi index 1, PCI 03:00.0 (index 0 carries the desktop).
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
EXPECT_GPU = "RTX 5090"

WARMUP_STEPS = 1000
# (size tag, case yaml, measured steps). Measured windows are >= ~20 s at the
# expected single-5090 fps and are multiples of the 1000-step defrag cadence,
# so every window contains exactly one defrag per 1000 steps.
SIZES = [
    ("1m",      "cases/lid_driven_cavity_2d_gen/case.yaml", 10000),
    ("1m_orig", "cases/lid_driven_cavity_2d/case.yaml",     10000),
    ("2m",      "cases/lid_driven_cavity_2d_2m/case.yaml",  6000),
    ("4m",      "cases/lid_driven_cavity_2d_4m/case.yaml",  4000),
    ("6m",      "cases/lid_driven_cavity_2d_6m/case.yaml",  3000),
    ("8m",      "cases/lid_driven_cavity_2d_8m/case.yaml",  2000),
    ("10m",     "cases/lid_driven_cavity_2d_10m/case.yaml", 2000),
    ("14m",     "cases/lid_driven_cavity_2d_14m/case.yaml", 2000),
    ("16m",     "cases/lid_driven_cavity_2d_16m/case.yaml", 2000),
    ("32m",     "cases/lid_driven_cavity_2d_32m/case.yaml", 2000),
]
CONFIGS = [("v0", 1), ("v0", 2), ("v5", 1), ("v5", 2)]
TRIALS = 3
RUN_TIMEOUT_SECONDS = 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="single-GPU baseline campaign")
    parser.add_argument("--out", type=str,
                        default=f"logs/single_baseline_{datetime.date.today():%Y%m%d}")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--sizes", type=str, default=None,
                        help="comma-separated subset of size tags")
    parser.add_argument("--configs", type=str, default=None,
                        help="comma-separated solver:in_flight pairs")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--obj-cache", type=str, default="logs/_obj_npy_cache")
    parser.add_argument("--no-telemetry", action="store_true")
    return parser.parse_args()


def load_done(results_path: pathlib.Path) -> set[tuple]:
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add((record["size"], record["solver"],
                          record["in_flight"], record["trial"]))
    return done


def start_telemetry(csv_path: pathlib.Path):
    handle = open(csv_path, "a")
    process = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,"
         "temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "1"],
        stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def resolve_device_indices(solvers: list[str]) -> dict[str, int]:
    """One probe subprocess per solver at campaign start; the bench runs then
    get a plain --device (the raw enumeration order is not stable across
    reboots, so the campaign pins the card by UUID exactly once)."""
    indices = {}
    for solver in solvers:
        completed = subprocess.run(
            [_PYTHON, str(_REPO / "_run_single_baseline_bench.py"),
             "--resolve-only", "--solver", solver, "--gpu-uuid", HEADLESS_5090_UUID],
            cwd=_REPO, env=_ENV, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit("device resolution failed:\n" + completed.stdout
                             + completed.stderr)
        indices[solver] = int(completed.stdout.strip().splitlines()[-1])
    return indices


def run_one(size: str, case: str, solver: str, in_flight: int, trial: int,
            warmup: int, measure: int, obj_cache: str,
            run_log: pathlib.Path, device_index: int) -> dict:
    command = [
        _PYTHON, str(_REPO / "_run_single_baseline_bench.py"),
        "--solver", solver, "--in-flight", str(in_flight),
        "--case", case, "--device", str(device_index),
        "--expect-gpu", EXPECT_GPU,
        "--warmup", str(warmup), "--measure", str(measure),
        "--obj-cache", obj_cache, "--tag", size, "--trial", str(trial),
    ]
    t0 = time.time()
    try:
        completed = subprocess.run(command, cwd=_REPO, env=_ENV,
                                   capture_output=True, text=True,
                                   timeout=RUN_TIMEOUT_SECONDS)
        text = completed.stdout + completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as timeout_error:
        text = ((timeout_error.stdout or "") + (timeout_error.stderr or "")
                + f"\n[campaign] TIMEOUT after {RUN_TIMEOUT_SECONDS}s\n")
        returncode = -1
    run_log.write_text(text, encoding="utf-8", errors="replace")

    record = None
    for line in text.splitlines():
        if line.startswith("RESULT "):
            record = json.loads(line[len("RESULT "):])
    if record is None:
        record = {"status": "failed", "solver": solver, "in_flight": in_flight,
                  "case": case, "tail": text[-600:]}
    record.update({"size": size, "trial": trial, "returncode": returncode,
                   "gpu_uuid": HEADLESS_5090_UUID,
                   "wall_seconds": round(time.time() - t0, 1),
                   "run_log": str(run_log.relative_to(_REPO))})
    return record


def main() -> int:
    args = parse_args()
    out_dir = (_REPO / args.out)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    (_REPO / args.obj_cache).mkdir(parents=True, exist_ok=True)

    sizes = SIZES
    if args.sizes:
        wanted = args.sizes.split(",")
        sizes = [entry for entry in SIZES if entry[0] in wanted]
    configs = CONFIGS
    if args.configs:
        configs = [(pair.split(":")[0], int(pair.split(":")[1]))
                   for pair in args.configs.split(",")]

    done = load_done(results_path)
    device_indices = resolve_device_indices(sorted({solver for solver, _ in configs}))
    print(f"[campaign] out={out_dir}  trials={args.trials}  sizes="
          f"{[s[0] for s in sizes]}  configs={configs}  already_done={len(done)}",
          flush=True)
    print(f"[campaign] headless 5090 uuid={HEADLESS_5090_UUID} -> device index per "
          f"solver: {device_indices}", flush=True)

    telemetry = None
    if not args.no_telemetry:
        telemetry = start_telemetry(out_dir / "telemetry.csv")

    campaign_t0 = time.time()
    try:
        for trial in range(1, args.trials + 1):
            # Rotate the config order per trial so no solver always runs
            # first (cold) or last (warmest) within a size.
            rotation = (trial - 1) % len(configs)
            ordered_configs = configs[rotation:] + configs[:rotation]
            for size, case, measure in sizes:
                for solver, in_flight in ordered_configs:
                    key = (size, solver, in_flight, trial)
                    if key in done:
                        continue
                    run_log = out_dir / "runs" / f"{size}_{solver}_d{in_flight}_t{trial}.log"
                    print(f"[campaign] t={time.time() - campaign_t0:7.0f}s  "
                          f"trial {trial}  {size:8s} {solver} in_flight={in_flight} ...",
                          end="", flush=True)
                    # Short pause so the previous process has fully exited
                    # (VRAM released) before the next one allocates.
                    time.sleep(1.0)
                    record = run_one(size, case, solver, in_flight, trial,
                                     args.warmup, measure, args.obj_cache, run_log,
                                     device_indices[solver])
                    with open(results_path, "a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if record["status"] == "ok":
                        print(f"  fps={record['fps']:8.2f}  drift={record['drift']:+d}  "
                              f"({record['wall_seconds']}s)", flush=True)
                        done.add(key)
                    else:
                        print(f"  {record['status'].upper()} rc={record['returncode']} "
                              f"see {record['run_log']}", flush=True)
    finally:
        if telemetry is not None:
            process, handle = telemetry
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            handle.close()

    print(f"[campaign] DONE in {(time.time() - campaign_t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
