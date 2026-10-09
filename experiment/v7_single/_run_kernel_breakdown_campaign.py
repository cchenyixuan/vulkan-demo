"""
_run_kernel_breakdown_campaign.py — measurement B of docs/seam_audit/v7_single.md: per-kernel GPU time of
v7_single on 2-D 8M and cavity3d_8m_b4, fused (default) and unfused (V7_FUSED_CORRECTION_DENSITY=0), 3 runs
each, interleaved (run 1 of every configuration, then run 2, ...), each run its own subprocess of
_run_kernel_breakdown.py (warmup 1000 + 2000 timed steps, then 2000 production steps for the timed-vs-production
difference). nvidia-smi 1 Hz telemetry for the whole campaign. Writes <out>/results.jsonl (resumable) and, with
``summarize``, <out>/summary_<setting>.md through _summarize_kernel_breakdown.py (one summary per switch setting:
the stage lists differ).

    .venv/Scripts/python.exe experiment/v7_single/_run_kernel_breakdown_campaign.py run --out logs/v7_single/kernel_breakdown
    .venv/Scripts/python.exe experiment/v7_single/_run_kernel_breakdown_campaign.py summarize --out logs/v7_single/kernel_breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
CASES = [("cavity2d_8m", "cases/lid_driven_cavity_2d_8m/case.yaml"), ("cavity3d_8m_b4", "cases/cavity3d_8m_b4/case.yaml")]
SETTINGS = {"fused": {}, "unfused": {"V7_FUSED_CORRECTION_DENSITY": "0"}}
RUNS = 3


def load_done(path: pathlib.Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("drift") == 0 and record.get("overflow") == 0:
                    done.add((record["tag"], record["setting"], record["run"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--out", default="logs/v7_single/kernel_breakdown")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    arguments = parser.parse_args()
    out = _REPOSITORY_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    if arguments.command == "summarize":
        for setting in SETTINGS:
            records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            selected = [record for record in records if record["setting"] == setting]
            if not selected:
                continue
            subset = out / f"results_{setting}.jsonl"
            subset.write_text("".join(json.dumps(record) + "\n" for record in selected), encoding="utf-8")
            subprocess.run([PYTHON, str(_REPOSITORY_ROOT / "_summarize_kernel_breakdown.py"), "--results", str(subset),
                            "--out-md", str(out / f"summary_{setting}.md")], cwd=_REPOSITORY_ROOT, check=False)
        return 0
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    done = load_done(results_path)
    telemetry = open(out / "telemetry.csv", "a")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,"
                                "temperature.gpu,utilization.gpu,memory.used", "--format=csv", "-l", "1"],
                               stdout=telemetry, stderr=subprocess.STDOUT)
    try:
        for run in range(1, arguments.runs + 1):
            for setting, extra in SETTINGS.items():
                for tag, case in CASES:
                    if (tag, setting, run) in done:
                        continue
                    command = [PYTHON, str(_REPOSITORY_ROOT / "_run_kernel_breakdown.py"), "--solver", "v7_single",
                               "--gpu-uuid", HEADLESS_5090_UUID, "--case", case, "--warmup", str(arguments.warmup),
                               "--measure", str(arguments.measure), "--obj-cache", arguments.obj_cache,
                               "--tag", tag, "--run", str(run)]
                    print(f"[breakdown] run {run} {setting} {tag} ...", end="", flush=True)
                    started = time.time()
                    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env={**env, **extra}, capture_output=True,
                                               text=True, encoding="utf-8", errors="replace")
                    text = completed.stdout + "\n" + completed.stderr
                    (out / f"{tag}_{setting}_run{run}.log").write_text(text, encoding="utf-8")
                    record = None
                    for line in text.splitlines():
                        if line.startswith("RESULT "):
                            record = json.loads(line[len("RESULT "):])
                    if record is None:
                        print(f" FAILED rc {completed.returncode}", flush=True)
                        continue
                    record["setting"] = setting
                    record["wall_s"] = round(time.time() - started, 1)
                    with open(results_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    frame = record["mean_us"]["gpu_frame"]
                    production = record["production"]["us_per_step"]
                    print(f" gpu frame {frame:.0f} us, production {production:.0f} us/step, drift {record['drift']} "
                          f"overflow {record['overflow']} ({record['wall_s']} s)", flush=True)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
        telemetry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
