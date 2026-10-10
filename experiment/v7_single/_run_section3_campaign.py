"""
_run_section3_campaign.py — "§3 补测" of docs/seam_audit/v7_single.md: the GPU runs and CPU statistics behind
Table 2 (item 2) and the per-size GPU step / candidate table (item 3).

run        GPU runs of _run_kernel_breakdown.py --solver v7_single --interleave 0: warmup 1000, then frames
           1001-3000 all timed (one submit + fence per step, BOTTOM_OF_PIPE timestamps at the production
           recording's tick labels, the throughput campaign's window), defrag frames excluded from the per-kernel
           means, the end state dumped (positions, fluid mask, voxel ids); 3 runs on 2-D 8M and cavity3d_8m_b4,
           1 run on the other five sizes of measurement A (run 1 of every size first, then runs 2 and 3 of the
           two 8M cases). Before every run: process-based idle check. nvidia-smi 1 Hz telemetry (SM clock, power,
           temperature, PCIe link) for the whole campaign. Refuses to run from a dirty tracked tree (--allow-dirty)
           and records the commit in every record. Stops on the first run with drift or overflow != 0. Then `stats`.
stats      CPU statistics (_neighbor_stats.py): the loaded initial state binned on the loader's grid and every
           dumped end state — candidates per particle, neighbours per particle (|r_ij| < h), occupancy per voxel.
summarize  _summarize_section3.py -> summary.json, table2.md, sizes.md, summary.csv.

    python experiment/v7_single/_run_section3_campaign.py run --out logs/v7_single/section3
    python experiment/v7_single/_run_section3_campaign.py stats --out logs/v7_single/section3
    python experiment/v7_single/_run_section3_campaign.py summarize --out logs/v7_single/section3
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

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
EXPECT_GPU = "RTX 5090"
TELEMETRY_INDEX = 1
# (size tag, case, independent runs); measurement A's seven cases
SIZES = [("8m", "cases/lid_driven_cavity_2d_8m/case.yaml", 3),
         ("3d_8m", "cases/cavity3d_8m_b4/case.yaml", 3),
         ("1m", "cases/lid_driven_cavity_2d_gen/case.yaml", 1),
         ("2m", "cases/lid_driven_cavity_2d_2m/case.yaml", 1),
         ("4m", "cases/lid_driven_cavity_2d_4m/case.yaml", 1),
         ("16m", "cases/lid_driven_cavity_2d_16m/case.yaml", 1),
         ("32m", "cases/lid_driven_cavity_2d_32m/case.yaml", 1)]


def log_line(out: pathlib.Path, message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(out / "campaign.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def git_state() -> dict:
    def run(*arguments):
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    untracked = [line for line in run("status", "--porcelain", "--untracked-files=normal").splitlines()
                 if line.startswith("??")]
    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD"),
            "describe": run("describe", "--tags", "--always"), "tracked_dirty": tracked,
            "untracked": untracked}


def wait_until_idle(out: pathlib.Path) -> None:
    """Block until no other solver process exists (two consecutive checks); the campaign's own launcher pair
    (venv python.exe + interpreter, both naming this script) is ignored."""
    from _run_single_baseline_campaign import solver_processes
    quiet = 0
    waited = 0
    while quiet < 2:
        busy = solver_processes(os.getpid())
        if busy is None:
            log_line(out, "idle check skipped (process query failed)")
            return
        busy = [entry for entry in busy if "_run_section3_campaign" not in entry]
        if not busy:
            quiet += 1
        else:
            quiet = 0
            waited += 1
            if waited % 30 == 1:
                log_line(out, f"solver processes still running, waiting: {busy}")
        time.sleep(1.0)
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        reading = next((line for line in completed.stdout.splitlines()
                        if line.strip().startswith(f"{TELEMETRY_INDEX},")), "")
        log_line(out, f"idle check: no other solver process; nvidia-smi index {TELEMETRY_INDEX}: {reading.strip()}")
    except Exception:
        pass


def start_telemetry(out: pathlib.Path):
    handle = open(out / "telemetry.csv", "a")
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,"
         "memory.used,pcie.link.gen.current,pcie.link.width.current", "--format=csv", "-l", "1"],
        stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def load_done(path: pathlib.Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("drift") == 0 and record.get("overflow") == 0:
                    done.add((record["size"], record["run"]))
    return done


def plan() -> list[tuple[str, str, int]]:
    order = []
    for run in range(1, max(runs for _, _, runs in SIZES) + 1):
        for size, case, runs in SIZES:
            if runs >= run:
                order.append((size, case, run))
    return order


def command_environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_campaign(arguments, out: pathlib.Path) -> int:
    state = git_state()
    if state["tracked_dirty"] and not arguments.allow_dirty:
        log_line(out, f"tracked tree is dirty, refusing to measure: {state['tracked_dirty'][:5]}")
        return 2
    log_line(out, f"commit {state['commit']} ({state['describe']}), tracked dirty {len(state['tracked_dirty'])}, "
                  f"untracked {len(state['untracked'])}")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = load_done(results_path)
    env = command_environment()
    sampler, handle = start_telemetry(out)
    stop = False
    try:
        for size, case, run in plan():
            if (size, run) in done:
                continue
            if stop:
                break
            wait_until_idle(out)
            dump = out / "runs" / f"{size}_run{run}_state.npz"
            command = [PYTHON, str(_REPOSITORY_ROOT / "_run_kernel_breakdown.py"), "--solver", "v7_single",
                       "--gpu-uuid", HEADLESS_5090_UUID, "--expect-gpu", EXPECT_GPU, "--case", case,
                       "--warmup", str(arguments.warmup), "--measure", str(arguments.measure), "--interleave", "0",
                       "--obj-cache", arguments.obj_cache, "--tag", size, "--run", str(run),
                       "--dump-state", str(dump)]
            log_line(out, f"run {size} #{run}: {case}")
            started = time.time()
            completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=env, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
            text = completed.stdout + "\n" + completed.stderr
            (out / "runs" / f"{size}_run{run}.log").write_text(text, encoding="utf-8")
            record = None
            for line in text.splitlines():
                if line.startswith("RESULT "):
                    record = json.loads(line[len("RESULT "):])
            if record is None:
                log_line(out, f"  FAILED rc {completed.returncode} (see runs/{size}_run{run}.log)")
                stop = True
                break
            record.update({"stage": "table2" if size in ("8m", "3d_8m") else "sizes", "size": size,
                           "commit": state["commit"], "wall_s": round(time.time() - started, 1)})
            with open(results_path, "a", encoding="utf-8") as results:
                results.write(json.dumps(record) + "\n")
            candidates = record["candidates"]
            log_line(out, f"  gpu frame {record['mean_us']['gpu_frame']:.0f} us (amortized "
                          f"{record['gpu_step_amortized_us']:.0f}), timed wall {record['timed_us_per_step']:.0f} us/step, "
                          f"candidates/particle {candidates['candidates_per_particle']:.1f}, drift {record['drift']} "
                          f"overflow {record['overflow']} ({record['wall_s']} s)")
            if record["drift"] != 0 or record["overflow"] != 0:
                log_line(out, "  STOP: drift or overflow != 0 (the spec says ask the user)")
                stop = True
                break
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
        handle.close()
    if stop:
        return 3
    return run_stats(arguments, out)


def run_stats(arguments, out: pathlib.Path) -> int:
    (out / "stats").mkdir(parents=True, exist_ok=True)
    env = command_environment()
    for size, case, runs in SIZES:
        targets = [("initial", ["--case", case], out / "stats" / f"{size}_initial.json")]
        for run in range(1, runs + 1):
            dump = out / "runs" / f"{size}_run{run}_state.npz"
            if dump.exists():
                targets.append((f"run{run} end", ["--state", str(dump)], out / "stats" / f"{size}_run{run}_end.json"))
        for label, source, target in targets:
            if target.exists() and not arguments.redo_stats:
                continue
            log_line(out, f"stats {size} {label} ...")
            started = time.time()
            completed = subprocess.run([PYTHON, str(_REPOSITORY_ROOT / "experiment/v7_single/_neighbor_stats.py"),
                                        *source, "--out", str(target), "--obj-cache", arguments.obj_cache],
                                       cwd=_REPOSITORY_ROOT, env=env, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
            tail = [line for line in (completed.stdout + completed.stderr).splitlines() if line.strip()][-1:]
            log_line(out, f"  rc {completed.returncode} ({time.time() - started:.0f} s): {tail[0] if tail else ''}")
            if completed.returncode != 0:
                (out / "stats" / f"{target.stem}.log").write_text(completed.stdout + "\n" + completed.stderr,
                                                                   encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "stats", "summarize"))
    parser.add_argument("--out", default="logs/v7_single/section3")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=2000)
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--redo-stats", action="store_true")
    arguments = parser.parse_args()
    out = _REPOSITORY_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    if arguments.command == "run":
        return run_campaign(arguments, out)
    if arguments.command == "stats":
        return run_stats(arguments, out)
    completed = subprocess.run([PYTHON, str(_REPOSITORY_ROOT / "experiment/v7_single/_summarize_section3.py"),
                                "--dir", str(out)], cwd=_REPOSITORY_ROOT, env=command_environment())
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
