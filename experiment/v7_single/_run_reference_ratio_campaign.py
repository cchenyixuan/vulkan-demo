"""
_run_reference_ratio_campaign.py — §5.1 figure: the chain code at K = 1 (experiment/v7, ChainOrchestratorV7, one slab)
against the single-device solver experiment/v7_single, same card, same cases as fig:baseline (§3.5).

Per case, in this order:
  1. bit-exactness: experiment/seam_audit/canonical_dump.py for both solvers (--canonical-lists, 100 steps, the voxel
     lists sorted after the initial voxelisation) and --compare: the per-particle state must be BIT-IDENTICAL, or the
     campaign stops and reports (docs/validation's protocol, the v7_single acceptance).
  2. timing: _run_single_baseline_campaign.py --configs v7_single:2,v7_chain:2 --trials 3 --warmup 1000 --idle-check:
     two steps in flight, the two solvers alternately (the config order rotates every trial: S C / C S / S C), the
     measured window of fig:baseline (SIZES), nvidia-smi telemetry, a process-based idle check before every run.

    .venv/Scripts/python.exe experiment/v7_single/_run_reference_ratio_campaign.py --out logs/v7_single/reference_ratio

Resumable: a case whose bit-exact verdict is recorded is not dumped again; finished timing runs are skipped by the
inner campaign. Refuses a dirty tracked tree. Summary / CSV / figure: _summarize_reference_ratio.py.
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

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
CHECK_STEPS = 100
# (size tag of _run_single_baseline_campaign.SIZES, dimension); fig:baseline's 2-D sizes (the 4M point is the
# generator-built case) and the 3-D cases that are built
CASES = [("1m", 2), ("2m", 2), ("4m_gen", 2), ("6m", 2), ("8m", 2), ("10m", 2), ("14m", 2), ("16m", 2), ("32m", 2),
         ("3d_1m", 3), ("3d_8m", 3)]


def log_line(out: pathlib.Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (out / "campaign.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def git_state() -> dict:
    def run(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD"), "tracked_dirty": tracked}


def environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(command: list[str], log_path: pathlib.Path) -> tuple[int, str]:
    started = time.time()
    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=environment(), capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    text = completed.stdout + "\n" + completed.stderr + f"\n[reference_ratio] wall {time.time() - started:.1f} s rc {completed.returncode}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return completed.returncode, text


def wait_until_idle(out: pathlib.Path) -> None:
    from _run_single_baseline_campaign import solver_processes
    quiet = 0
    waited = 0
    while quiet < 2:
        busy = solver_processes(os.getpid())
        if busy is None:
            log_line(out, "idle check skipped (process query failed)")
            return
        busy = [entry for entry in busy if "_run_reference_ratio_campaign" not in entry]
        if not busy:
            quiet += 1
        else:
            quiet = 0
            waited += 1
            if waited % 30 == 1:
                log_line(out, f"solver processes still running, waiting: {busy}")
        time.sleep(1.0)


def parse_verdict(text: str) -> str:
    """The verdict line of canonical_dump --compare (the acceptance's parser)."""
    from experiment.v7_single._run_acceptance import parse_compare
    try:
        return str(parse_compare(text).get("verdict", "unknown"))
    except Exception:
        match = re.search(r"\b(BIT-IDENTICAL)\b", text)
        return match.group(1) if match else "unknown"


def bit_exact_check(out: pathlib.Path, tag: str, case: str, device: int) -> dict:
    """canonical_dump S and C (100 steps, canonical lists), --compare; the dumps share two file names (overwritten
    case after case: a 32M dump is ~5 GB)."""
    dumps = out / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    record = {"case": case, "steps": CHECK_STEPS, "dumps": {}, "invariants": {}}
    for build, build_arguments in (("S", ["--solver", "v7_single"]), ("C", ["--solver", "v7"])):
        dump = dumps / f"{build}.npz"
        if dump.exists():
            dump.unlink()
        command = [PYTHON, "experiment/seam_audit/canonical_dump.py", *build_arguments, "--case", case,
                   "--device-map", str(device), "--steps", str(CHECK_STEPS), "--canonical-lists", "--out", str(dump)]
        rc, text = run(command, logs / f"{tag}_check_{build}.log")
        if rc != 0 or not dump.exists():
            record[build] = {"rc": rc, "tail": text[-800:]}
            record["verdict"] = f"dump {build} failed"
            return record
        meta = {}
        try:
            import numpy as np
            with np.load(dump, allow_pickle=True) as data:
                meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        except Exception:
            meta = {}
        record["invariants"][build] = {key: meta.get(key) for key in ("alive", "expected", "overflow", "fps") if key in meta}
        record["dumps"][build] = str(dump)
    command = [PYTHON, "-m", "experiment.seam_audit.canonical_dump", "--compare", record["dumps"]["S"], record["dumps"]["C"]]
    rc, text = run(command, logs / f"{tag}_check_S_vs_C.log")
    record["compare_rc"] = rc
    record["verdict"] = parse_verdict(text)
    record["compare_tail"] = text[-1200:]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="logs/v7_single/reference_ratio")
    parser.add_argument("--only", default="", help="comma-separated size tags")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--skip-check", action="store_true",
                        help="timing only: no bit-exact dumps (S = C bit-identical was established by the v7_single acceptance "
                             "battery and by the first case of this campaign)")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    out = (_REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = git_state()
    log_line(out, f"campaign start: commit {state['short']}, tracked dirty: {state['tracked_dirty'] or 'none'}")
    if state["tracked_dirty"] and not arguments.allow_dirty:
        log_line(out, "refusing to measure from a dirty tracked tree")
        return 2
    only = {name.strip() for name in arguments.only.split(",") if name.strip()}
    from experiment.v7_single._run_single_bench import resolve_device_index
    device = resolve_device_index(HEADLESS_5090_UUID)
    sizes = {tag: (case, steps) for tag, case, steps in __import__("_run_single_baseline_campaign").SIZES}
    checks_path = out / "bit_exact.jsonl"
    verdicts = {}
    if checks_path.exists():
        for line in checks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                verdicts[record["tag"]] = record
    for tag, dimension in CASES:
        if only and tag not in only:
            continue
        case, steps = sizes[tag]
        # ---- 1. bit-exact ------------------------------------------------------------------------------------------
        if arguments.skip_check:
            log_line(out, f"{tag}: bit-exact check skipped (--skip-check: S = C was established by the v7_single acceptance)")
        elif verdicts.get(tag, {}).get("verdict") == "BIT-IDENTICAL":
            log_line(out, f"{tag}: bit-exact verdict recorded earlier (BIT-IDENTICAL), skipped")
        else:
            wait_until_idle(out)
            log_line(out, f"{tag}: bit-exact check (S and C, {CHECK_STEPS} steps, canonical lists) ...")
            record = bit_exact_check(out, tag, case, device)
            record.update({"tag": tag, "dimension": dimension, "commit": state["short"], "when": time.strftime("%Y-%m-%d %H:%M:%S")})
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            verdicts[tag] = record
            log_line(out, f"{tag}: S vs C after {CHECK_STEPS} steps: {record['verdict']}")
            if record["verdict"] != "BIT-IDENTICAL":
                log_line(out, f"{tag}: NOT bit-identical; stopping the campaign (see {out / 'logs'})")
                return 1
        # ---- 2. timing -------------------------------------------------------------------------------------------
        wait_until_idle(out)
        log_line(out, f"{tag}: timing S and C, {arguments.trials} trials, warmup {arguments.warmup}, measure {steps} ...")
        command = [PYTHON, str(_REPOSITORY_ROOT / "_run_single_baseline_campaign.py"), "--out", str(out.relative_to(_REPOSITORY_ROOT)),
                   "--sizes", tag, "--configs", "v7_single:2,v7_chain:2", "--trials", str(arguments.trials),
                   "--warmup", str(arguments.warmup), "--idle-check"]
        rc, text = run(command, out / "logs" / f"{tag}_timing.log")
        bad = [line for line in text.splitlines() if "status" in line and ("failed" in line or "bad_state" in line)]
        log_line(out, f"{tag}: timing rc {rc}" + (f", problems: {bad[:3]}" if bad else ""))
        if rc != 0:
            log_line(out, f"{tag}: timing campaign failed; stopping")
            return 1
    log_line(out, "campaign done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
