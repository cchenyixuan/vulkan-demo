"""
_run_single_ceiling.py — find the single-GPU (one 5090) memory ceiling.

Extends the single-GPU baseline past 64M on the HEADLESS card (device 1, full
32GB) by generating progressively larger cavity cases and single-benching each
until VK_ERROR_OUT_OF_DEVICE_MEMORY. Logs fps/OOM to logs/single_ceiling.jsonl.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PYTHON = str(_REPO / ".venv/Scripts/python.exe")
_ENV = {**os.environ, "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation"}
_RESULTS = _REPO / "logs/single_ceiling.jsonl"

DEVICE = 1                          # headless 5090 (full 32GB; dev0 carries the desktop)
# (tag, half) — pool = 1.15*total; single OOMs near ~93M total on 32GB.
SIZES = [("80m", 4471), ("90m", 4743), ("96m", 4899), ("104m", 5099)]


def run(cmd):
    return subprocess.run(cmd, cwd=_REPO, env=_ENV, capture_output=True, text=True)


def log(rec):
    with open(_RESULTS, "a") as h:
        h.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main() -> int:
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for tag, half in SIZES:
        case_dir = _REPO / f"cases/lid_driven_cavity_2d_{tag}"
        if not (case_dir / "domain.obj").exists():
            print(f"[{tag}] generating case (half={half})...", flush=True)
            run([_PYTHON, "utils/geometry/_demo_cavity_case.py", "--half", str(half),
                 "--out", f"cases/lid_driven_cavity_2d_{tag}", "--no-preview"])
        print(f"[{tag}] single-bench on device {DEVICE}...", flush=True)
        out = run([_PYTHON, "experiment/v5/_run_v5_single_bench.py", "--device", str(DEVICE),
                   "--case", f"cases/lid_driven_cavity_2d_{tag}/case.yaml",
                   "--max-steps", "4000", "--warmup", "2000", "--bench-window", "2000"])
        text = out.stdout + out.stderr
        p50 = re.findall(r"frame_total_us\s+mean=\s*[\d.]+us\s+p50=\s*([\d.]+)us", text)
        oom = bool(re.search(r"OutOfDeviceMemory|OUT_OF_DEVICE|VkError.*Memory", text))
        if p50:
            fps = 1e6 / float(p50[-1])
            log({"size": tag, "fps": round(fps, 2), "status": "ok"})
        elif oom or out.returncode != 0:
            log({"size": tag, "fps": None, "status": "OOM", "tail": text[-400:]})
            print(f"[{tag}] OOM — single-GPU ceiling reached.", flush=True)
            break
        else:
            log({"size": tag, "fps": None, "status": "unknown", "tail": text[-400:]})
            break
    print("SINGLE CEILING PROBE DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
