"""
cluster_campaign_a.py — Campaign A on the 3090 cluster: weak + strong scaling
K=1..5 at depth 1 (the drift-clean regime; see project-cluster-drift-investigation).

Runs ON THE HEAD NODE (nohup'd, survives connection drops); every GPU point is
its own time-capped srun on partition gpu2. Provenance per point: full log,
case.yaml sha256, the physical GPUs the job actually received, git head.

    nohup ~/swq/venv/bin/python remote/cluster_campaign_a.py > logs/campaign_a_driver.log 2>&1 &

Matrix:
  weak   : constant 2M/slab — single on cavity_weak_k1_2m, chains k2..k5 (k5 generated here)
  strong8 : lid_driven_cavity_2d_8m,  K=1..5
  strong16: lid_driven_cavity_2d_16m, K=1..5
Protocol: warmup 5000 + 20000 measured, per-direction, DEPTH 1, pool_safety 1.2,
equal weights (symmetric 3090s). Any drift != 0 marks the point FAILED (kept, flagged).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
_OUT = _REPO / "logs/campaign_a"
_SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
PYTHON = sys.executable

WARMUP = 5000
MAX_STEPS = 25000

# (case_dir, generator_args) — tracked yamls regenerate geometry with
# --objs-only (NEVER touch the yaml: 2026-09-04 lesson); k5 is new -> full gen.
CASE_GEN = [
    ("cases/cavity_weak_k1_2m",  "--half 707 --half-x 707  --objs-only --no-preview"),
    ("cases/cavity_weak_k2_4m",  "--half 707 --half-x 1415 --objs-only --no-preview"),
    ("cases/cavity_weak_k3_6m",  "--half 707 --half-x 2122 --objs-only --no-preview"),
    ("cases/cavity_weak_k4_8m",  "--half 707 --half-x 2830 --objs-only --no-preview"),
    ("cases/cavity_weak_k5_10m", "--half 707 --half-x 3537 --no-preview"),  # NEW: full gen
    ("cases/lid_driven_cavity_2d_8m",  None),   # derive --half from yaml dx
    ("cases/lid_driven_cavity_2d_16m", None),
]

WEAK_CASES = {k: f"cases/cavity_weak_k{k}_{2*k}m/case.yaml" for k in (1, 2, 3, 4, 5)}
STRONG_CASES = {"strong8": "cases/lid_driven_cavity_2d_8m/case.yaml",
                "strong16": "cases/lid_driven_cavity_2d_16m/case.yaml"}

_STEADY_RE = re.compile(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps")
_DRIFT_RE = re.compile(r"drift=(-?\d+)")
_SINGLE_FINAL_RE = re.compile(r"\[bench_v5_single\] (\d+) steps in ([\d.]+)s = ([\d.]+) fps")
_ALIVE_WARN = "WARN: alive drift"


def srun(inner: str, *, gres: str | None, cpus: int, minutes: int,
         job: str) -> tuple[int, str]:
    cmd = [f"{_SLURM_BIN}/srun", "-p", "gpu2", "-c", str(cpus),
           "-t", str(minutes), "-J", job]
    if gres:
        cmd.insert(3, f"--gres={gres}")
    cmd += ["bash", "-c", inner]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=minutes * 60 + 300)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def half_from_yaml(case_dir: pathlib.Path) -> int:
    for line in (case_dir / "case.yaml").read_text().splitlines():
        if line.strip().startswith("h:"):
            smoothing_length = float(line.split(":")[1].split("#")[0])
            return round(0.5 / (smoothing_length / 5.0))
    raise RuntimeError(f"no h: in {case_dir}")


def ensure_cases() -> None:
    for rel, args in CASE_GEN:
        case_dir = _REPO / rel
        if (case_dir / "domain.obj").exists():
            print(f"[gen] {rel}: geometry present", flush=True)
            continue
        if args is None:
            half = half_from_yaml(case_dir)
            args = f"--half {half} --objs-only --no-preview"
        inner = (f"cd {_REPO} && {PYTHON} utils/geometry/_demo_cavity_case.py "
                 f"{args} --out {rel}")
        print(f"[gen] {rel}: {args}", flush=True)
        code, output = srun(inner, gres=None, cpus=4, minutes=90, job="cA_gen")
        # Success judged by the generator's own completion line — the srun ran
        # on node02 and the head node's NFS attribute cache can lag several
        # seconds behind, so an immediate exists() check false-negatives
        # (this exact trap killed the first driver launch).
        success = "wrote frame.obj domain.obj" in output
        if not success:
            for _ in range(12):
                if (case_dir / "domain.obj").exists():
                    success = True
                    break
                time.sleep(5)
        if not success:
            print(f"[gen] FAILED {rel} (srun rc={code}):\n{output[-1500:]}",
                  flush=True)
            raise SystemExit(1)


def run_point(name: str, case_rel: str, k: int) -> None:
    log_path = _OUT / f"{name}.log"
    case_path = _REPO / case_rel
    minutes = 80 if "16m" in case_rel and k == 1 else 45
    # env.sh provides LD_LIBRARY_PATH for the bundled Vulkan loader — without
    # it python-vulkan dies on import (launch-2 lesson: all 15 points in 3s).
    env_prefix = "source ~/swq/env.sh && "
    if k == 1:
        inner = (env_prefix + f"cd {_REPO} && nvidia-smi --query-gpu=index,name,pci.bus_id "
                 f"--format=csv,noheader; {PYTHON} experiment/v5/_run_v5_single_bench.py "
                 f"--case {case_rel} --device 0 --max-steps {MAX_STEPS} "
                 f"--warmup {WARMUP} --bench-window {MAX_STEPS - WARMUP}")
        gres, cpus = "gpu:1", 8
    else:
        weights = ",".join(["1"] * k)
        device_map = ",".join(str(i) for i in range(k))
        inner = (env_prefix + f"cd {_REPO} && nvidia-smi --query-gpu=index,name,pci.bus_id "
                 f"--format=csv,noheader; {PYTHON} experiment/v5/_run_v5_chain_bench.py "
                 f"--case {case_rel} --weights {weights} --device-map {device_map} "
                 f"--sync-scheme per-direction --depth 1 "
                 f"--max-steps {MAX_STEPS} --warmup {WARMUP}")
        gres, cpus = f"gpu:{k}", min(8 * k, 48)

    t_start = time.time()
    code, output = srun(inner, gres=gres, cpus=cpus, minutes=minutes, job=f"cA_{name}")
    log_path.write_text(output, encoding="utf-8")

    row = {"point": name, "case": case_rel, "k": k, "depth": 1,
           "returncode": code, "wall_s": round(time.time() - t_start, 1),
           "case_sha256": sha256_of(case_path / "case.yaml"
                                    if case_path.is_dir() else case_path),
           "warmup": WARMUP, "max_steps": MAX_STEPS,
           "gpus": [l.strip() for l in output.splitlines()[:k]
                    if "3090" in l or "NVIDIA" in l],
           "t": round(time.time(), 1)}
    steady = _STEADY_RE.search(output) or _SINGLE_FINAL_RE.search(output)
    if steady:
        steps, seconds, fps = steady.groups()
        row.update(steady_fps=float(fps), steady_steps=int(steps),
                   steady_s=float(seconds))
    drift = _DRIFT_RE.search(output)
    if drift:
        row["drift"] = int(drift.group(1))
    row["failed"] = (code != 0 or row.get("drift") not in (0, None)
                     or _ALIVE_WARN in output)
    with open(_OUT / "summary.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"[point] {name}: fps={row.get('steady_fps')} drift={row.get('drift')} "
          f"failed={row['failed']} ({row['wall_s']}s)", flush=True)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    try:
        git_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=_REPO, capture_output=True,
                                  text=True).stdout.strip() or "n/a"
    except OSError:
        git_head = "n/a (no git on head node; deployed from tarball)"
    print(f"[campaign A] start git={git_head} python={PYTHON}", flush=True)

    ensure_cases()

    points = []
    for k in (1, 2, 3, 4, 5):
        points.append((f"weak_k{k}", WEAK_CASES[k], k))
    for label, case_rel in STRONG_CASES.items():
        for k in (1, 2, 3, 4, 5):
            points.append((f"{label}_k{k}", case_rel, k))

    for name, case_rel, k in points:
        run_point(name, case_rel, k)

    print("[campaign A] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
