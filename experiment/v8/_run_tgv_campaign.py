"""
_run_tgv_campaign.py — the Taylor–Green campaign of experiment/v8 on one GPU: generate the cases (cases/tgv_2d/<case>),
run every configuration of RUNS as its own process (experiment/v8/_run_tgv.py), one after the other, from a clean
tracked tree, with a process-based idle check before each run.

    .venv/Scripts/python.exe experiment/v8/_run_tgv_campaign.py --out logs/v8/tgv [--only re100_n200,...] [--dry-run]

Finished runs (result.json with status ok) are skipped on a rerun. Every run's result is appended to
<out>/results.jsonl; the campaign log is <out>/campaign.log. The campaign stops at the first run whose drift or
overflow is not 0.
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
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.v8._make_tgv_case import parser as case_parser, build as build_case   # noqa: E402

PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
CASE_ROOT = "cases/tgv_2d"

# (run name, case name, generator arguments, run arguments). The case is generated once; runs sharing a case
# (the repeats) reuse it.
RUNS: list[tuple[str, str, list[str], list[str]]] = []


def _add(run_name: str, case_name: str, generator: list[str], run: list[str] | None = None) -> None:
    # set v1 (2026-10-11, the first campaign): every case at the cavity eps^2 = 0.0025 h^2, stated explicitly since the
    # generator default became 1e-4 h^2
    RUNS.append((run_name, case_name, ["--epsilon-squared-factor", "0.0025"] + generator if "--epsilon-squared-factor" not in generator else generator, run or []))


# A. Re = 100 resolution series (the convergence study): L/dx = 50 ... 800 (2.5 k ... 640 k particles)
for side in (50, 100, 200, 400, 800):
    _add(f"re100_n{side}", f"re100_n{side}", ["--side", str(side), "--reynolds", "100"])
# B. Re = 1000 series
for side in (100, 200, 400, 800):
    _add(f"re1000_n{side}", f"re1000_n{side}", ["--side", str(side), "--reynolds", "1000"])
# C. ablations at Re = 100, 200^2
_add("re100_n200_nopst", "re100_n200_nopst", ["--side", "200", "--reynolds", "100", "--no-pst"])
_add("re100_n200_nodelta", "re100_n200_nodelta", ["--side", "200", "--reynolds", "100", "--no-delta"])
_add("re100_n200_nokcg", "re100_n200_nokcg", ["--side", "200", "--reynolds", "100", "--no-kcg"])
_add("re100_n200_vanilla", "re100_n200_vanilla", ["--side", "200", "--reynolds", "100", "--no-pst", "--no-delta", "--no-kcg"])
# D. Mach number at Re = 100, 200^2 (Ma 0.1 is the series' value)
for mach, label in ((0.05, "ma0p05"), (0.02, "ma0p02"), (0.01, "ma0p01")):
    _add(f"re100_n200_{label}", f"re100_n200_{label}", ["--side", "200", "--reynolds", "100", "--mach", str(mach)])
# E. initial density rho0 everywhere (the series starts from rho(p(x, 0)))
_add("re100_n200_uniform", "re100_n200_uniform", ["--side", "200", "--reynolds", "100", "--uniform-density"])
# F. kernel support at Re = 100 (h/dx 5 is the series' value): 4 and 3.5
_add("re100_n200_hdx4", "re100_n200_hdx4", ["--side", "200", "--reynolds", "100", "--hdx", "4"])
_add("re100_n210_hdx3p5", "re100_n210_hdx3p5", ["--side", "210", "--reynolds", "100", "--hdx", "3.5"])
# G. run-to-run repeats of the 200^2 point (atomic list order is the one source of noise)
# H. diagnostics of the viscosity deficit at Re = 100, 200^2: eps^2 -> 0 and the old 0.01 h^2, and dt / 10 at Ma 0.1
_add("re100_n200_eps0", "re100_n200_eps0", ["--side", "200", "--reynolds", "100", "--epsilon-squared-factor", "1e-8"])
_add("re100_n200_eps0p01", "re100_n200_eps0p01", ["--side", "200", "--reynolds", "100", "--epsilon-squared-factor", "0.01"])
_add("re100_n200_cfl0p015", "re100_n200_cfl0p015", ["--side", "200", "--reynolds", "100", "--cfl", "0.015"])
# I. eps^2 sweep at Re = 100, 200^2 (the default 0.0025 h^2 is re100_n200, 1e-8 is re100_n200_eps0, 0.01 is _eps0p01)
for factor, label in ((1e-5, "eps1e-5"), (1e-4, "eps1e-4"), (4e-4, "eps4e-4"), (1e-3, "eps1e-3")):
    _add(f"re100_n200_{label}", f"re100_n200_{label}", ["--side", "200", "--reynolds", "100", "--epsilon-squared-factor", str(factor)])
_add("re100_n200_r2", "re100_n200", ["--side", "200", "--reynolds", "100"], ["--tag", "repeat2"])
_add("re100_n200_r3", "re100_n200", ["--side", "200", "--reynolds", "100"], ["--tag", "repeat3"])


# ---- set v2 (2026-10-11, after the eps^2 sweep): the v8 default eps^2 = 1e-4 h^2, resolution up to 3200^2 ----------
RUNS_V2: list[tuple[str, str, list[str], list[str]]] = []
for side in (50, 100, 200, 400, 800, 1600):
    RUNS_V2.append((f"re100_n{side}", f"re100_n{side}", ["--side", str(side), "--reynolds", "100"], []))
# 3200^2 (10.24 M particles) at Re 100 runs 655 k viscous-limited steps to t = 2; to t = 1 (snapshots 0, 0.5, 1)
RUNS_V2.append(("re100_n3200", "re100_n3200", ["--side", "3200", "--reynolds", "100", "--t-end", "1"],
                ["--t-end", "1", "--snapshot-times", "0,0.5,1"]))
for side in (100, 200, 400, 800, 1600, 3200):
    RUNS_V2.append((f"re1000_n{side}", f"re1000_n{side}", ["--side", str(side), "--reynolds", "1000"], []))
# ---- set v3 (2026-10-11): v2 + numerics.delta_density (rho - rho0 stored): v2's 1600^2 came out worse than its 800^2
# (nu_eff/nu 0.9952 vs 0.9961, velocity error 0.44 vs 0.37 %) because under the viscous dt limit (dt ~ dx^2) the
# per-step density increment (~2e-8 at 1600^2) is below the float32 ULP of rho = 1 (1.2e-7). v2 keeps 50^2-1600^2
# as the without-delta reference; its 3200^2 run was stopped.
RUNS_V3: list[tuple[str, str, list[str], list[str]]] = []
for side in (50, 100, 200, 400, 800, 1600):
    RUNS_V3.append((f"re100_n{side}", f"re100_n{side}", ["--side", str(side), "--reynolds", "100", "--delta-density"], []))
RUNS_V3.append(("re100_n3200", "re100_n3200", ["--side", "3200", "--reynolds", "100", "--delta-density", "--t-end", "1"],
                ["--t-end", "1", "--snapshot-times", "0,0.5,1"]))
for side in (100, 200, 400, 800, 1600, 3200):
    RUNS_V3.append((f"re1000_n{side}", f"re1000_n{side}", ["--side", str(side), "--reynolds", "1000", "--delta-density"], []))
SETS = {"v1": (RUNS, "cases/tgv_2d", "logs/v8/tgv"), "v2": (RUNS_V2, "cases/tgv_2d_eps1e-4", "logs/v8/tgv_eps1e-4"),
        "v3": (RUNS_V3, "cases/tgv_2d_eps1e-4_delta", "logs/v8/tgv_eps1e-4_delta")}


def log_line(out: pathlib.Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (out / "campaign.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def git_state() -> dict:
    def run(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD"), "tracked_dirty": tracked}


def wait_until_idle(out: pathlib.Path) -> None:
    """Block until no other solver process exists (two consecutive checks); this campaign's own launcher pair is
    ignored (_run_single_baseline_campaign.solver_processes: the process-based check, nvidia-smi is unusable)."""
    from _run_single_baseline_campaign import solver_processes
    quiet = 0
    waited = 0
    while quiet < 2:
        busy = solver_processes(os.getpid())
        if busy is None:
            log_line(out, "idle check skipped (process query failed)")
            return
        busy = [entry for entry in busy if "_run_tgv_campaign" not in entry]
        if not busy:
            quiet += 1
        else:
            quiet = 0
            waited += 1
            if waited % 30 == 1:
                log_line(out, f"solver processes still running, waiting: {busy}")
        time.sleep(1.0)


def generate_case(case_name: str, generator: list[str], case_root: str = CASE_ROOT) -> pathlib.Path:
    case_dir = _REPOSITORY_ROOT / case_root / case_name
    arguments = case_parser().parse_args(generator + ["--out", f"{case_root}/{case_name}"])
    build_case(arguments, " ".join(generator))
    return case_dir


def finished(run_dir: pathlib.Path) -> bool:
    result = run_dir / "result.json"
    if not result.exists():
        return False
    try:
        return json.loads(result.read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=sorted(SETS), default="v1", help="run set: v1 (eps^2 0.0025 h^2) or v2 (1e-4 h^2, up to 3200^2)")
    parser.add_argument("--out", default=None, help="default: the set's log directory")
    parser.add_argument("--gpu-uuid", default=HEADLESS_5090_UUID)
    parser.add_argument("--only", default="", help="comma-separated run names")
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--sample-dt", type=float, default=0.02)
    parser.add_argument("--in-flight", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="generate the cases and list the runs only")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    runs, case_root, default_out = SETS[arguments.set]
    out = (_REPOSITORY_ROOT / (arguments.out or default_out)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = {name.strip() for name in arguments.only.split(",") if name.strip()}
    selected = [entry for entry in runs if not only or entry[0] in only]
    state = git_state()
    log_line(out, f"campaign start: set {arguments.set} ({case_root}), commit {state['short']} ({len(selected)} runs), tracked dirty: {state['tracked_dirty'] or 'none'}")
    if state["tracked_dirty"] and not arguments.allow_dirty:
        log_line(out, "refusing to measure from a dirty tracked tree (commit first, or --allow-dirty)")
        return 2
    generated: set[str] = set()
    for _run_name, case_name, generator, _run in selected:
        if case_name not in generated:
            generate_case(case_name, generator, case_root)
            generated.add(case_name)
    if arguments.dry_run:
        for run_name, case_name, generator, run in selected:
            print(f"  {run_name:28s} case {case_name:24s} {' '.join(generator)} {' '.join(run)}")
        return 0
    environment = dict(os.environ)
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    environment["PYTHONIOENCODING"] = "utf-8"
    for run_name, case_name, _generator, run in selected:
        run_dir = out / run_name
        if finished(run_dir):
            log_line(out, f"{run_name}: finished earlier, skipped")
            continue
        wait_until_idle(out)
        command = [PYTHON, "experiment/v8/_run_tgv.py", "--case", f"{case_root}/{case_name}/case.yaml",
                   "--gpu-uuid", arguments.gpu_uuid, "--t-end", str(arguments.t_end), "--sample-dt", str(arguments.sample_dt),
                   "--in-flight", str(arguments.in_flight), "--out", str(run_dir.relative_to(_REPOSITORY_ROOT)), *run]
        log_line(out, f"{run_name}: start ({' '.join(command[2:])})")
        started = time.time()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        result = {}
        if (run_dir / "result.json").exists():
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        record = {"run": run_name, "case": case_name, "rc": completed.returncode, "wall_s": round(elapsed, 1),
                  "status": result.get("status"), "drift": result.get("drift"), "overflow": result.get("overflow"),
                  "steps": result.get("steps"), "steps_per_second": result.get("steps_per_second"),
                  "commit": state["short"]}
        with (out / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        log_line(out, f"{run_name}: rc {completed.returncode} status {record['status']} drift {record['drift']} "
                      f"overflow {record['overflow']} steps {record['steps']} ({record['steps_per_second']} steps/s, "
                      f"{elapsed:.0f} s)")
        if completed.returncode != 0 or record["status"] != "ok":
            log_line(out, f"{run_name}: stopping the campaign (see {run_dir / 'run.log'})")
            return 1
    log_line(out, "campaign done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
