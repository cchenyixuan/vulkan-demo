"""
_run_two_hop_campaign.py — the three-hop vs two-hop measurement matrix.

For every case:
  1. --trials fps trials per transport, INTERLEAVED (three, two, three,
     two, ...), each in a fresh process: pipelined depth-2, no GPU timers.
  2. one instrumented depth-1 run per transport (per-frame GPU timestamps).

Every run is one invocation of _run_two_hop_bench.py; the only argument
that differs between the two arms is --transport. Validation layers are
disabled for all of them. One JSON line per run is appended to
<output>/results.jsonl as soon as the run ends, so a crash keeps what was
measured.

Usage:
    .venv/Scripts/python.exe experiment/two_hop/_run_two_hop_campaign.py \\
        --sizes 1m,2m,4m,8m,16m --trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER = "experiment/two_hop/_run_two_hop_bench.py"
_TRANSPORT_ORDER = ("three_hop", "two_hop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="two-hop measurement campaign")
    parser.add_argument("--sizes", default="1m,2m,4m,8m,16m",
                        help="comma-separated cavity sizes (empty = none)")
    parser.add_argument("--case", action="append", default=[],
                        metavar="LABEL=PATH", help="additional case (repeatable)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--steps", action="append", default=[],
                        metavar="LABEL=WARMUP:MAX_STEPS",
                        help="per-case step plan overriding --warmup / "
                             "--max-steps (repeatable)")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--loop-trace", action="store_true",
                        help="pass --loop-trace to every fps trial")
    parser.add_argument("--switch-interval-ms", type=float, default=None)
    parser.add_argument("--minimum-own-columns", type=int, default=None)
    parser.add_argument("--weights", default="1.0,1.0")
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--skip-instrumented", action="store_true")
    parser.add_argument("--environment", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="extra environment for BOTH arms (sensitivity "
                             "groups only; repeatable)")
    parser.add_argument("--group", default="campaign",
                        help="tag written into every result line")
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="keep the runs already present in "
                             "<output>/runs and measure only the missing ones")
    parser.add_argument("--run-timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def case_path(size: str) -> str:
    if size == "1m":                       # the 1M case has no _1m suffix
        return "cases/lid_driven_cavity_2d/case.yaml"
    return f"cases/lid_driven_cavity_2d_{size}/case.yaml"


def run_one(args, environment: dict, output_directory: pathlib.Path,
            case_label: str, case: str, transport: str, trial: int,
            instrumented: bool) -> dict:
    kind = "instrumented" if instrumented else f"trial{trial}"
    stem = f"{case_label}_{transport}_{kind}"
    run_result_path = output_directory / "runs" / f"{stem}.json"
    run_result_path.parent.mkdir(parents=True, exist_ok=True)
    if run_result_path.exists():
        if args.resume:
            print(f"[campaign] {case_label:>6} {transport:<9} {kind:<12} "
                  f"already measured, skipped", flush=True)
            return json.loads(run_result_path.read_text().splitlines()[-1])
        run_result_path.unlink()
    warmup, max_steps = args.step_plan.get(
        case_label, (args.warmup, args.max_steps))
    command = [
        sys.executable, _RUNNER,
        "--transport", transport,
        "--case", case,
        "--weights", args.weights,
        "--sync-scheme", "per-direction",
        "--depth", str(args.depth),
        "--pool-safety", str(args.pool_safety),
        "--warmup", str(warmup),
        "--max-steps", str(max_steps),
        "--first-trial-index", str(trial),
        "--result-json", str(run_result_path),
    ]
    # Per-frame CSVs: every instrumented run, and the first fps trial (the
    # other trials keep their per-segment statistics in results.jsonl).
    if instrumented or trial == 0:
        command += ["--bench-csv",
                    str(output_directory / "frames" / f"{stem}.csv")]
    if instrumented:
        command.append("--instrumented")
    elif args.loop_trace:
        command.append("--loop-trace")
    if args.switch_interval_ms is not None:
        command += ["--switch-interval-ms", str(args.switch_interval_ms)]
    if args.minimum_own_columns is not None:
        command += ["--minimum-own-columns", str(args.minimum_own_columns)]
    started = time.time()
    log_path = output_directory / "logs" / f"{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command, cwd=_REPO_ROOT, env=environment, capture_output=True,
            text=True, timeout=args.run_timeout_seconds)
        output = completed.stdout + "\n--- stderr ---\n" + completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = f"TIMEOUT after {args.run_timeout_seconds}s\n{error.stdout}\n{error.stderr}"
        exit_code = -1
    log_path.write_text(output, encoding="utf-8", errors="replace")

    if run_result_path.exists():
        result = json.loads(run_result_path.read_text().splitlines()[-1])
    else:
        result = {"case": case, "transport": transport, "trial": trial,
                  "instrumented": instrumented, "valid": False,
                  "error": output[-1500:]}
    result.update({
        "case_label": case_label,
        "group": args.group,
        "exit_code": exit_code,
        "wall_seconds": round(time.time() - started, 1),
        "log": str(log_path.relative_to(output_directory)),
    })
    with open(output_directory / "results.jsonl", "a") as handle:
        handle.write(json.dumps(result) + "\n")
    print(f"[campaign] {case_label:>6} {transport:<9} {kind:<12} "
          f"fps={result.get('steady_fps')} drift={result.get('drift')} "
          f"stamps={result.get('stamp_errors_gpu')}/"
          f"{result.get('stamp_errors_host')} "
          f"overwrite={result.get('overwrite_errors_host')} "
          f"seam={result.get('seam_ok')} valid={result.get('valid')} "
          f"({result['wall_seconds']}s)", flush=True)
    return result


def main() -> int:
    args = parse_args()
    cases = [(size, case_path(size))
             for size in args.sizes.split(",") if size]
    for entry in args.case:
        label, _, path = entry.partition("=")
        cases.append((label, path))
    for _label, path in cases:
        if not (_REPO_ROOT / path).exists():
            sys.exit(f"case not found: {path}")
    args.step_plan = {}
    for entry in args.steps:
        label, _, plan = entry.partition("=")
        warmup_text, _, max_steps_text = plan.partition(":")
        args.step_plan[label] = (int(warmup_text), int(max_steps_text))

    output_directory = pathlib.Path(
        args.output_directory
        or f"logs/two_hop_experiment/{args.group}_{time.strftime('%Y%m%d_%H%M%S')}")
    if not output_directory.is_absolute():
        output_directory = _REPO_ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    environment = {**os.environ,
                   "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation"}
    for entry in args.environment:
        name, _, value = entry.partition("=")
        environment[name] = value
    (output_directory / "campaign.json").write_text(json.dumps({
        "arguments": {name: value for name, value in vars(args).items()
                      if name != "step_plan"},
        "step_plan": {label: list(plan)
                      for label, plan in args.step_plan.items()},
        "cases": cases,
        "extra_environment": args.environment,
        "started_unix_time": round(time.time(), 1),
    }, indent=2))
    print(f"[campaign] output -> {output_directory}")

    all_valid = True
    for case_label, case in cases:
        for trial in range(args.trials):
            for transport in _TRANSPORT_ORDER:
                result = run_one(args, environment, output_directory,
                                 case_label, case, transport, trial,
                                 instrumented=False)
                all_valid &= bool(result.get("valid"))
        if not args.skip_instrumented:
            for transport in _TRANSPORT_ORDER:
                result = run_one(args, environment, output_directory,
                                 case_label, case, transport, 0,
                                 instrumented=True)
                all_valid &= bool(result.get("valid"))
    print(f"[campaign] done; all runs valid: {all_valid}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
