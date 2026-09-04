"""
_run_v5_soak.py — long-duration (hours-scale) dual-GPU soak runner.

Purpose: run the production submit-ahead pipeline for a WALL-CLOCK duration
(e.g. 12 h) and record the full performance / health time series:

  * interval fps (per defrag interval, i.e. every ``defrag_cadence`` frames,
    measured at the pipeline-drained defrag boundary — the only safe hook)
  * conservation (alive total + drift vs the case's initial count)
  * migration series + pool-health peaks per sim
  * ghost-worker memcpy duration trend (host-side transport health)
  * host process working-set (leak detector)
  * GPU telemetry via a background ``nvidia-smi`` sampler (utilization,
    power, temperature, SM clock, memory) into a separate CSV

Design constraints honored here:
  * ``run_pipelined`` restarts frame numbering per call, so a soak must be
    ONE call — wall-clock stop is implemented by raising ``_SoakDone`` from
    the ``on_defrag`` callback (fires with the pipeline fully drained).
  * ``GhostMigrationWorker.timestamps`` grows per frame; the callback clears
    it every interval (after folding the interval's copy-time stats).
  * ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` keeps
    Windows awake for the process lifetime without touching power settings
    (display is allowed to sleep; compute continues).

Outputs under --out-dir (default logs/soak_<case>_<scheme>_<stamp>/):
    meta.json        run configuration + expected particle total
    intervals.jsonl  one row per defrag interval (the fps/health curve)
    telemetry.csv    nvidia-smi samples every --telemetry-interval seconds
    summary.json     end-of-run totals (also written on crash)

Usage (12 h, 8M, symmetric 2x5090):
    .venv/Scripts/python.exe experiment/v5/_run_v5_soak.py \\
        --case cases/lid_driven_cavity_2d_8m/case.yaml \\
        --weights 1.0,1.0 --sync-scheme per-direction --hours 12
"""

from __future__ import annotations

import argparse
import ctypes
import sys
if sys.platform == "win32":
    import ctypes.wintypes
import datetime
import json
import pathlib
import subprocess
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _SoakDone(Exception):
    """Raised from the defrag callback when the wall-clock budget is spent."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 dual-GPU long-duration soak runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d_8m/case.yaml")
    p.add_argument("--weights", default="1.0,1.0",
                   help="K comma-separated slab weights (chain)")
    p.add_argument("--device-map", default=None,
                   help="K comma-separated device indices; default "
                        "round-robin over 0,1")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--sync-scheme", default="per-direction",
                   choices=["aggregated", "per-direction"],
                   help="frame sync scheme (see sync_scheme_v5.py)")
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--pool-safety", type=float, default=None)
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--out-dir", default=None,
                   help="output directory; default logs/soak_<case>_<scheme>_<stamp>")
    p.add_argument("--telemetry-interval", type=int, default=30,
                   help="seconds between nvidia-smi samples (0 = disable)")
    p.add_argument("--print-every", type=int, default=50,
                   help="echo every Nth interval row to stdout")
    return p.parse_args()


def _keep_system_awake() -> None:
    """Prevent Windows sleep for this process's lifetime (auto-reverts on
    exit). ES_CONTINUOUS | ES_SYSTEM_REQUIRED — display may still sleep.
    No-op on Linux (servers do not sleep under us)."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print("[soak] SetThreadExecutionState: system sleep inhibited")
    except Exception as e:  # unexpected failure — not fatal
        print(f"[soak] WARNING: could not inhibit system sleep: {e!r}")


def _working_set_mb() -> float:
    """Current process resident memory in MB (leak detector).
    Windows: psapi working set. Linux: /proc/self/status VmRSS."""
    if sys.platform != "win32":
        try:
            with open("/proc/self/status", encoding="ascii") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0   # kB -> MB
        except Exception:
            pass
        return float("nan")
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        # GetCurrentProcess() pseudo-handle is 64-bit -1; ctypes' default
        # 32-bit restype truncates it to an invalid handle — use the
        # constant directly.
        handle = ctypes.c_void_p(-1)
        get_memory_info = getattr(ctypes.windll.kernel32,
                                  "K32GetProcessMemoryInfo", None)
        if get_memory_info is None:
            get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        if not get_memory_info(handle, ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return float("nan")


def _start_telemetry(out_dir: pathlib.Path, interval_s: int):
    """Background nvidia-smi sampler -> telemetry.csv. Returns Popen or None."""
    if interval_s <= 0:
        return None
    telemetry_path = out_dir / "telemetry.csv"
    try:
        handle = open(telemetry_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=timestamp,index,utilization.gpu,power.draw,"
             "temperature.gpu,clocks.sm,memory.used",
             "--format=csv", f"-l", str(interval_s)],
            stdout=handle, stderr=subprocess.DEVNULL)
        print(f"[soak] telemetry: nvidia-smi every {interval_s}s -> {telemetry_path}")
        return process
    except Exception as e:
        print(f"[soak] WARNING: telemetry disabled ({e!r})")
        return None


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_chain_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    _keep_system_awake()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    case_tag = pathlib.Path(args.case).parent.name.replace("lid_driven_cavity_2d", "cavity") or "case"
    out_dir = (pathlib.Path(args.out_dir) if args.out_dir else
               pathlib.Path("logs") / f"soak_{case_tag}_{args.sync_scheme}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = [float(w) for w in args.weights.split(",")]
    slab_count = len(weights)
    if args.device_map is not None:
        device_map = [int(d) for d in args.device_map.split(",")]
        if len(device_map) != slab_count:
            raise SystemExit(f"--device-map needs {slab_count} entries")
    else:
        device_map = [index % 2 for index in range(slab_count)]
    global_case = load_case_v5(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    chain = compute_chain_partition(
        global_case, weights, pool_safety=args.pool_safety)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    budget_s = args.hours * 3600.0
    # Hard upper bound on steps (defense in depth if the timer path breaks):
    # generous 250 fps x budget => at most ~2x the intended wall time.
    max_steps = int(budget_s * 250)

    meta = {
        "case": args.case, "weights": weights, "device_map": device_map,
        "depth": args.depth,
        "sync_scheme": args.sync_scheme, "hours": args.hours,
        "defrag_cadence": defrag_cadence, "pool_safety": args.pool_safety,
        "expected_total": expected_total, "max_steps_bound": max_steps,
        "started_iso": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[soak] out_dir={out_dir}")
    print(f"[soak] {json.dumps(meta)}")

    telemetry = _start_telemetry(out_dir, args.telemetry_interval)
    intervals_file = open(out_dir / "intervals.jsonl", "w", encoding="utf-8")

    contexts, sims = [], []
    for index in range(slab_count):
        contexts.append(VulkanContextV5.create(
            device_index=device_map[index],
            application_name=f"soak_v5_s{index}"))
        sims.append(SphSimulatorV5(contexts[-1], chain.slabs[index],
                                   sync_scheme=args.sync_scheme))

    soak_state = {
        "t_start": None, "t_last": None, "frame_last": 0,
        "interval_index": 0, "min_fps": float("inf"), "max_fps": 0.0,
        "fps_sum": 0.0, "last_row": None,
    }
    summary: dict = {"outcome": "unknown"}

    def worker_interval_stats(orch) -> dict:
        stats = {}
        for worker in orch.workers:
            copies_us = []
            for frame_stamps in worker.timestamps.values():
                if "wait_ns" in frame_stamps and "copy_ns" in frame_stamps:
                    copies_us.append(
                        (frame_stamps["copy_ns"] - frame_stamps["wait_ns"]) / 1000.0)
            if copies_us:
                copies_us.sort()
                stats[worker.label] = {
                    "copy_us_p50": copies_us[len(copies_us) // 2],
                    "copy_us_max": copies_us[-1],
                }
            worker.timestamps.clear()
        return stats

    try:
        with ChainOrchestratorV5(sims, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            def on_defrag(frame_n: int, report: list) -> None:
                now = time.perf_counter()
                elapsed = now - soak_state["t_start"]
                interval_frames = frame_n - soak_state["frame_last"]
                interval_s = now - soak_state["t_last"]
                interval_fps = interval_frames / interval_s if interval_s > 0 else 0.0
                soak_state["t_last"] = now
                soak_state["frame_last"] = frame_n
                soak_state["interval_index"] += 1
                soak_state["min_fps"] = min(soak_state["min_fps"], interval_fps)
                soak_state["max_fps"] = max(soak_state["max_fps"], interval_fps)
                soak_state["fps_sum"] += interval_fps

                alive_total = sum(r["alive"] for r in report)
                row = {
                    "iso": datetime.datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s": round(elapsed, 3),
                    "frame_n": frame_n,
                    "interval_fps": round(interval_fps, 2),
                    "alive_total": alive_total,
                    "drift": alive_total - expected_total,
                    "alive": [r["alive"] for r in report],
                    "interval_migration": [r["interval_migration"]
                                           for r in report],
                    "used_fraction": [round(r["used_fraction"], 4)
                                      for r in report],
                    "overflow": [r["overflow_install_tail"] for r in report],
                    "overflow_incoming": [r.get("overflow_incoming", 0)
                                          for r in report],
                    "overflow_inside": [r.get("overflow_inside", 0)
                                        for r in report],
                    "overflow_ghost": [r.get("overflow_ghost", 0)
                                       for r in report],
                    "overflow_install_inside": [
                        r.get("overflow_install_inside", 0) for r in report],
                    "ghost_send": [[r.get("ghost_send_leading", 0),
                                    r.get("ghost_send_trailing", 0)]
                                   for r in report],
                    "ghost_recv": [[r.get("ghost_recv_leading", 0),
                                    r.get("ghost_recv_trailing", 0)]
                                   for r in report],
                    "workers": worker_interval_stats(orch),
                    "working_set_mb": round(_working_set_mb(), 1),
                }
                soak_state["last_row"] = row
                intervals_file.write(json.dumps(row) + "\n")
                intervals_file.flush()
                if soak_state["interval_index"] % args.print_every == 1:
                    print(f"[soak] {row['iso']} frame={frame_n:>9,} "
                          f"fps={interval_fps:7.1f} drift={row['drift']} "
                          f"ws={row['working_set_mb']}MB", flush=True)
                if elapsed >= budget_s:
                    raise _SoakDone()

            soak_state["t_start"] = time.perf_counter()
            soak_state["t_last"] = soak_state["t_start"]
            try:
                orch.run_pipelined(max_steps, depth=args.depth, warmup=0,
                                   on_defrag=on_defrag)
                summary["outcome"] = "hit_max_steps_bound"
            except _SoakDone:
                summary["outcome"] = "completed"

            # Final drain state is clean (callback fires pipeline-drained).
            for sim in sims:
                sim.submit_defrag_and_wait()
            total = sum(sim.readback_global_status()["alive_particle_count"]
                        for sim in sims)
            elapsed_total = time.perf_counter() - soak_state["t_start"]
            intervals_done = soak_state["interval_index"]
            summary.update({
                "frames_total": soak_state["frame_last"],
                "elapsed_s": round(elapsed_total, 1),
                "elapsed_h": round(elapsed_total / 3600.0, 3),
                "mean_interval_fps": round(
                    soak_state["fps_sum"] / intervals_done, 2) if intervals_done else None,
                "min_interval_fps": round(soak_state["min_fps"], 2),
                "max_interval_fps": round(soak_state["max_fps"], 2),
                "final_alive": total,
                "final_drift": total - expected_total,
                "pool_health": [sim.readback_pool_health() for sim in sims],
                "finished_iso": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[soak] DONE: {summary['frames_total']:,} frames in "
                  f"{summary['elapsed_h']:.2f}h  mean_fps={summary['mean_interval_fps']}"
                  f"  drift={summary['final_drift']}")
    except BaseException as e:
        summary["outcome"] = f"crashed: {type(e).__name__}: {e}"
        summary["crash_last_row"] = soak_state["last_row"]
        print(f"[soak] CRASHED: {e!r}", flush=True)
        raise
    finally:
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        intervals_file.close()
        if telemetry is not None:
            telemetry.terminate()
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
