"""
_run_v5_soak_supervisor.py — segment-auto-restart wrapper for ultra-long soaks.

Context (2026-07-20): K=8-on-2-GPUs (4 VkDevices per physical GPU) wedges
every 1.9-6.9 h on Windows — autopsy-proven driver/WDDM-level fault (a
submitted batch with satisfied waits never executes; round 3 escalated to
VkErrorDeviceLost), striking EITHER GPU, with the host stack proven
innocent. Production configs (1-2 sims/GPU) are unaffected. To still
deliver an N-hour accumulated limit soak on the emulation config, this
supervisor runs _run_v5_soak.py in segments: each wedge is logged as a
data point (MTTF distribution) and a fresh segment starts, until the
TOTAL wall-clock budget is spent.

Per segment: seg_<N>/ under --out-dir with the full soak output set.
Supervisor-level: supervisor.jsonl (one row per segment: outcome,
frames, hours, drift) + supervisor_summary.json.

An inactivity watchdog (intervals.jsonl mtime) kills segments whose
teardown hangs in vkDeviceWaitIdle on a wedged device (round-2 mode).

Usage:
    .venv/Scripts/python.exe experiment/v5/_run_v5_soak_supervisor.py \\
        --hours-total 60 --out-dir logs/soak_8m_k8_60h_supervised \\
        -- --case cases/lid_driven_cavity_2d_8m/case.yaml \\
           --weights 1,1,1,1,1,1,1,1 --device-map 0,1,0,1,0,1,0,1 \\
           --pool-safety 1.2 --sync-scheme per-direction
(everything after ``--`` is forwarded to _run_v5_soak.py verbatim)
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOAK = pathlib.Path(__file__).parent / "_run_v5_soak.py"

INACTIVITY_KILL_S = 600          # no intervals.jsonl update -> kill segment
POST_WEDGE_COOLDOWN_S = 30       # let the driver settle between segments


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hours-total", type=float, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-segments", type=int, default=64)
    p.add_argument("soak_args", nargs=argparse.REMAINDER,
                   help="arguments after -- go to _run_v5_soak.py")
    return p.parse_args()


def run_segment(segment_dir: pathlib.Path, hours: float,
                soak_args: list) -> dict:
    segment_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(segment_dir / "run.log", "w", encoding="utf-8")
    command = [sys.executable, str(_SOAK), "--hours", f"{hours:.4f}",
               "--out-dir", str(segment_dir)] + soak_args
    process = subprocess.Popen(command, stdout=log_handle,
                               stderr=subprocess.STDOUT, cwd=str(_REPO_ROOT))
    intervals_path = segment_dir / "intervals.jsonl"
    killed_for_inactivity = False
    while True:
        try:
            process.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            pass
        if intervals_path.exists():
            age = time.time() - intervals_path.stat().st_mtime
            if age > INACTIVITY_KILL_S:
                killed_for_inactivity = True
                print(f"[supervisor] segment inactive {age:.0f}s "
                      f"(teardown hang?) — killing", flush=True)
                process.kill()
                process.wait(timeout=60)
                break
    log_handle.close()

    result = {"returncode": process.returncode,
              "killed_for_inactivity": killed_for_inactivity}
    summary_path = segment_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            result["outcome"] = summary.get("outcome")
            result["frames"] = summary.get("frames_total")
            result["drift"] = summary.get("final_drift")
            crash_row = summary.get("crash_last_row") or {}
            if result.get("frames") is None:
                result["frames"] = crash_row.get("frame_n")
            if result.get("drift") is None:
                result["drift"] = crash_row.get("drift")
        except Exception as error:  # noqa: BLE001
            result["outcome"] = f"summary unreadable: {error!r}"
    else:
        result["outcome"] = "no summary.json (hard kill or crash-at-boot)"
    return result


def main() -> int:
    args = parse_args()
    soak_args = [a for a in args.soak_args if a != "--"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = open(out_dir / "supervisor.jsonl", "a", encoding="utf-8")

    budget_s = args.hours_total * 3600.0
    spent_s = 0.0
    segments = []
    for segment_index in range(args.max_segments):
        remaining_h = (budget_s - spent_s) / 3600.0
        if remaining_h <= 0.02:
            break
        segment_dir = out_dir / f"seg_{segment_index:02d}"
        print(f"[supervisor] segment {segment_index}: {remaining_h:.2f} h "
              f"budget remaining -> {segment_dir}", flush=True)
        t0 = time.perf_counter()
        result = run_segment(segment_dir, remaining_h, soak_args)
        elapsed_s = time.perf_counter() - t0
        spent_s += elapsed_s
        result.update({
            "segment": segment_index,
            "segment_hours": round(elapsed_s / 3600.0, 3),
            "cumulative_hours": round(spent_s / 3600.0, 3),
            "iso": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        segments.append(result)
        journal.write(json.dumps(result) + "\n")
        journal.flush()
        print(f"[supervisor] segment {segment_index} ended after "
              f"{result['segment_hours']} h: {str(result['outcome'])[:100]} "
              f"(frames={result.get('frames')}, drift={result.get('drift')})",
              flush=True)
        if result.get("outcome") == "completed":
            break
        time.sleep(POST_WEDGE_COOLDOWN_S)

    summary = {
        "hours_total_budget": args.hours_total,
        "hours_accumulated": round(spent_s / 3600.0, 3),
        "segment_count": len(segments),
        "wedge_count": sum(1 for s in segments
                           if s.get("outcome") != "completed"),
        "segment_hours": [s["segment_hours"] for s in segments],
        "all_drift_zero": all((s.get("drift") or 0) == 0 for s in segments),
        "finished_iso": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "supervisor_summary.json").write_text(
        json.dumps(summary, indent=2))
    journal.close()
    print(f"[supervisor] DONE: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
