"""
delta_density_perf.py — interleaved single-card throughput of V6_DELTA_DENSITY off / on (opt item 六).

The single-buffer form of k1_chain_vs_single.py (bootstrap + one defrag, one combined step command
buffer, fence per step) on the headless 5090 (device 1), 2-D 1M and 4M, 3 trials, variants alternating.
Appends to logs/seam_audit/opt/delta_density/perf.jsonl and prints trial-wise ratios.

    .venv/Scripts/python.exe experiment/seam_audit/delta_density_perf.py
"""
import json
import os
import pathlib
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "logs" / "seam_audit" / "opt" / "delta_density" / "perf.jsonl"
CASES = [("1m", "cases/lid_driven_cavity_2d_gen/case.yaml", 1000, 6000),
         ("4m", "cases/lid_driven_cavity_2d_4m/case.yaml", 500, 3000)]
records = []
for trial in (1, 2, 3):
    for name, case, warmup, steps in CASES:
        for variant in ("baseline", "delta"):
            environment = {k: v for k, v in os.environ.items() if not k.startswith("V6_")}
            environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
            if variant == "delta":
                environment["V6_DELTA_DENSITY"] = "1"
            completed = subprocess.run(
                [sys.executable, "-m", "experiment.seam_audit.k1_chain_vs_single", "--worker", "--form", "single",
                 "--case", case, "--warmup", str(warmup), "--steps", str(steps), "--device", "1"],
                env=environment, capture_output=True, text=True, cwd=REPO, timeout=1800)
            line = [l for l in completed.stdout.splitlines() if l.startswith("[k1_vs_single] RESULT ")][-1]
            result = json.loads(line[len("[k1_vs_single] RESULT "):])
            record = {"trial": trial, "case": name, "variant": variant, "fps": result["fps"], "drift": result["drift"]}
            records.append(record)
            print(record, flush=True)
            with OUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
for name, *_ in CASES:
    base = {r["trial"]: r["fps"] for r in records if r["case"] == name and r["variant"] == "baseline"}
    delta = {r["trial"]: r["fps"] for r in records if r["case"] == name and r["variant"] == "delta"}
    ratios = [delta[t] / base[t] for t in base]
    print(name, "baseline", round(statistics.mean(base.values()), 1), "delta", round(statistics.mean(delta.values()), 1),
          "ratio %.2f +- %.2f %%" % (100 * statistics.mean(ratios), 100 * statistics.stdev(ratios)))
