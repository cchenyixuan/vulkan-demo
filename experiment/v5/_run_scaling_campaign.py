"""
_run_scaling_campaign.py — full scaling-curve campaign on the 5090 + 7900 XTX rig.

For each problem size: measure single-GPU steady baselines (5090 + AMD), then sweep
the K_split weight (dual, depth-2) to find the optimum + best fps + strong-scaling
efficiency. Each result is appended to logs/scaling_results.jsonl immediately so a
crash leaves partial data. Runs SERIALLY (both GPUs per dual run — no parallelism).

Validation off (VK_LOADER_LAYERS_DISABLE), warmup 5000 throughout (matches the
existing 1M/4M/6M data points). Depth-2 (6M showed depth-3 gives no gain at scale).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PYTHON = str(_REPO / ".venv/Scripts/python.exe")
_ENV = {**os.environ, "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation"}
_RESULTS = _REPO / "logs/scaling_2x5090.jsonl"

# 2x RTX 5090 (same-vendor, symmetric, host-staged — no P2P on consumer cards).
# Extension to 32M/64M (64GB VRAM unlocks these); symmetric optimum is w=1.0.
SIZES = ["32m", "64m"]
WEIGHTS = [1.0]
SINGLE_MAX_STEPS, SINGLE_WARMUP, SINGLE_WINDOW = 8000, 5000, 3000
DUAL_MAX_STEPS, DUAL_WARMUP, DUAL_DEPTH = 11000, 5000, 2


def case_path(size: str) -> str:
    if size == "1m":                       # the 1M case has no _1m suffix
        return "cases/lid_driven_cavity_2d/case.yaml"
    return f"cases/lid_driven_cavity_2d_{size}/case.yaml"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=_REPO, env=_ENV, capture_output=True, text=True)
    return result.stdout + result.stderr


def _log(record: dict) -> None:
    record["t"] = round(time.time(), 1)
    with open(_RESULTS, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def single_fps(device: int, size: str):
    out = _run([_PYTHON, "experiment/v5/_run_v5_single_bench.py", "--device", str(device),
                "--case", case_path(size), "--max-steps", str(SINGLE_MAX_STEPS),
                "--warmup", str(SINGLE_WARMUP), "--bench-window", str(SINGLE_WINDOW)])
    # steady fps = 1e6 / last-window frame_total_us p50
    matches = re.findall(r"frame_total_us\s+mean=\s*[\d.]+us\s+p50=\s*([\d.]+)us", out)
    alive_ok = "WARN: alive drift" not in out
    if not matches:
        return None, out[-600:]
    return 1e6 / float(matches[-1]), ("ok" if alive_ok else "ALIVE_DRIFT")


def dual(size: str, weight: float):
    out = _run([_PYTHON, "experiment/v5/_run_v5_dual_pipeline.py", "--device-a", "0", "--device-b", "1",
                "--weights", f"{weight},1.0", "--pool-safety", "1.2", "--case", case_path(size),
                "--depth", str(DUAL_DEPTH), "--max-steps", str(DUAL_MAX_STEPS), "--warmup", str(DUAL_WARMUP)])
    steady = re.search(r"STEADY.*=\s*([\d.]+)\s*fps", out)
    drift = re.search(r"drift=(-?\d+)", out)
    overflow = "OVERFLOWED" in out
    return (float(steady.group(1)) if steady else None,
            int(drift.group(1)) if drift else None,
            overflow)


def main() -> int:
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _log({"event": "campaign_start", "sizes": SIZES, "weights": WEIGHTS})
    for size in SIZES:
        t0 = time.time()
        nv_fps, nv_status = single_fps(0, size)
        print(f"[{size}] single 5090 = {nv_fps} ({nv_status})", flush=True)
        amd_fps, amd_status = single_fps(1, size)
        print(f"[{size}] single AMD  = {amd_fps} ({amd_status})", flush=True)
        ceiling = (nv_fps + amd_fps) if (nv_fps and amd_fps) else None
        _log({"size": size, "kind": "single", "nv_fps": nv_fps, "amd_fps": amd_fps,
              "ceiling": ceiling, "nv_status": nv_status, "amd_status": amd_status})

        sweep = {}
        for weight in WEIGHTS:
            fps, drift, overflow = dual(size, weight)
            sweep[weight] = fps
            print(f"[{size}] dual w={weight} -> {fps} fps drift={drift} overflow={overflow}", flush=True)
            _log({"size": size, "kind": "dual", "weight": weight, "fps": fps,
                  "drift": drift, "overflow": overflow})

        valid = {w: f for w, f in sweep.items() if f}
        if valid and ceiling:
            best_w = max(valid, key=valid.get)
            best_fps = valid[best_w]
            eta = best_fps / ceiling
            _log({"size": size, "kind": "summary", "best_weight": best_w, "best_fps": best_fps,
                  "eta_strong": round(eta, 4), "nv_fps": nv_fps, "amd_fps": amd_fps,
                  "ceiling": ceiling, "minutes": round((time.time() - t0) / 60, 1)})
            print(f"[{size}] *** BEST w={best_w} fps={best_fps:.1f} eta={eta*100:.1f}% "
                  f"({round((time.time()-t0)/60,1)} min) ***", flush=True)
        else:
            _log({"size": size, "kind": "summary", "error": "no valid runs", "sweep": sweep})
    _log({"event": "campaign_done"})
    print("CAMPAIGN DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
