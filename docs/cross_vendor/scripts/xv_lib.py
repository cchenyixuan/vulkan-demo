"""xv_lib.py — the S6.6 cross-vendor campaign (RTX 5090 + RX 7900 XTX, one Windows machine) with the E7 cluster
protocol, run locally: every run goes through docs/cluster_v6/scripts/run_chain_v6.py --solver v7 (the E7 harness:
defrag events, build stages, start barrier) around experiment/v7/_run_v7_chain_bench.py, its log is classified by
parse_run_v6.py (pass / threshold_only / hard_failure; invariants, seam checks, defrag events, steady fps with and
without the defrag events) and the row is appended to results.jsonl of the experiment directory.

Protocol (the user's spec, 2026-10-20): 1500 steps, two in flight (--depth 2), T = the last 1000 steps, defrag events
taken out (steady_fps_nodefrag); 3 trials per point; the K = 1 references run on both cards at the same time behind
one start barrier; every run must pass the built-in checks (particle count conserved, every counter 0), a failure is
reported, never re-run and dropped. Calibration = the sec:slab rule as weight_calibration.py implements it: a pilot
with equal weights, another one with the new cuts when they moved, the last pilot's weights (--calibrate-rounds 2).

Devices: VulkanContextV7's discrete-first order on this machine is [0] RTX 5090, [1] RX 7900 XTX, [2] Ryzen iGPU;
--device-map 0,1 puts slab 0 (left) on the 5090 and slab 1 (right) on the 7900 XTX; weights are given left to right
(omega_5090, omega_XTX).

Every function returns the parsed row (dict) of the run; a hard failure raises RunFailed after the row was recorded
(the caller stops, reports, never retries).
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]
PYTHON = pathlib.Path(r"C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe")
HARNESS = REPOSITORY / "docs" / "cluster_v6" / "scripts" / "run_chain_v6.py"
PARSER = REPOSITORY / "docs" / "cluster_v6" / "scripts" / "parse_run_v6.py"
TREE_HASH = REPOSITORY / "docs" / "cluster_v6" / "scripts" / "tree_hash.py"
CASES = REPOSITORY / "cases" / "aligned"
DEVICE_MAP = "0,1"                       # 5090, 7900 XTX
DEVICE_NAMES = {0: "RTX 5090", 1: "RX 7900 XTX"}
COMMON = ["--sync-scheme", "per-direction", "--depth", "2", "--pool-safety", "1.2"]
STEPS, WARMUP = 1500, 500
SWITCH_INTERVAL_MS = "0.2"
SOLVER_TAG = "v7-rc3"
WRAPPER_DEFAULTS: list = []              # run_chain_v6 arguments every launch gets (xv_campaign sets --obj-cache DIR)


class RunFailed(RuntimeError):
    pass


def case_yaml(case: str) -> pathlib.Path:
    return CASES / case / "case.yaml"


def environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("V5_", "V6_", "V7_"))}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    return env


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, capture_output=True, text=True, check=True).stdout.strip()


def provenance() -> dict:
    """Solver commit / tag / experiment-v7 tree, harness scripts, machine."""
    head = git("rev-parse", "HEAD")
    tree = subprocess.run([str(PYTHON), str(TREE_HASH), str(REPOSITORY / "experiment" / "v7")], capture_output=True, text=True).stdout.strip()
    return {"head": head, "head_short": head[:7], "tag": git("describe", "--tags", "--exact-match", "HEAD") if _tag_at_head() else None,
            "solver_commit": git("rev-parse", SOLVER_TAG + "^{commit}"), "experiment_v7_tree": tree,
            "experiment_v7_tree_at_tag": git("rev-parse", f"{SOLVER_TAG}:experiment/v7"),
            "harness_scripts_from": git("rev-parse", "e7-cluster"), "hostname": platform.node(), "python": sys.version.split()[0]}


def _tag_at_head() -> bool:
    return subprocess.run(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=REPOSITORY, capture_output=True).returncode == 0


class Experiment:
    """One experiment directory: logs/, results.jsonl, weights/, traces/, telemetry.csv, index.tsv."""

    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        for name in ("logs", "weights", "traces", "barriers"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self.results = directory / "results.jsonl"
        self.telemetry = directory / "telemetry.csv"
        self.index = directory / "index.tsv"
        self.telemetry_process = None
        self.stop_file = directory / "telemetry.stop"

    # -- telemetry ---------------------------------------------------------------------------------------------
    def start_telemetry(self, interval: float = 1.0, path: pathlib.Path = None) -> None:
        if self.stop_file.exists():
            self.stop_file.unlink()
        if path is not None:
            self.telemetry = path
        self.telemetry_process = subprocess.Popen([str(PYTHON), str(REPOSITORY / "docs" / "cross_vendor" / "scripts" / "xv_telemetry.py"),
                                                   str(self.telemetry), "--stop-file", str(self.stop_file), "--interval", str(interval)],
                                                  stdout=(self.directory / "telemetry.log").open("w"), stderr=subprocess.STDOUT)
        time.sleep(3)

    def stop_telemetry(self) -> None:
        if self.telemetry_process is None:
            return
        self.stop_file.write_text("stop")
        try:
            self.telemetry_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.telemetry_process.terminate()
        self.telemetry_process = None

    # -- one run -----------------------------------------------------------------------------------------------
    def launch(self, label: str, bench_arguments: list, wrapper_arguments: list = ()) -> tuple:
        """Start one harness process (not waited): (process, log path, start epoch)."""
        log = self.directory / "logs" / f"{label}.log"
        command = [str(PYTHON), "-u", str(HARNESS), "--solver", "v7", "--switch-interval-ms", SWITCH_INTERVAL_MS,
                   *WRAPPER_DEFAULTS, *wrapper_arguments, "--", *bench_arguments]
        with (self.directory / "logs" / f"{label}.cmd").open("w", encoding="utf-8") as handle:
            handle.write(subprocess.list2cmdline(command) + "\n")
        start = time.time()
        process = subprocess.Popen(command, cwd=REPOSITORY, env=environment(), stdout=log.open("w", encoding="utf-8"),
                                   stderr=subprocess.STDOUT)
        return process, log, start

    def finish(self, label: str, process, log: pathlib.Path, start: float, timeout: float, optional: bool = False) -> dict:
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = 124
        end = time.time()
        parse = subprocess.run([str(PYTHON), str(PARSER), "--solver", "v7", "--log", str(log), "--label", label, "--rc", str(return_code),
                                "--start", f"{start:.3f}", "--end", f"{end:.3f}", "--telemetry", str(self.telemetry),
                                "--results", str(self.results), "--node", platform.node(), "--expect-root", str(REPOSITORY),
                                "--solver-tag", SOLVER_TAG, "--solver-tree", git("rev-parse", f"{SOLVER_TAG}:experiment/v7")],
                               cwd=REPOSITORY, capture_output=True, text=True, env=environment())
        result_line = next((line for line in parse.stdout.splitlines() if line.startswith("RESULT")), parse.stdout.strip()[-400:])
        print(result_line, flush=True)
        if parse.stderr.strip():
            print(parse.stderr.strip()[-800:], file=sys.stderr, flush=True)
        row = json.loads(self.results.read_text(encoding="utf-8").splitlines()[-1])
        with self.index.open("a", encoding="utf-8") as handle:
            handle.write(f"{label}\t{row.get('status')}\t{return_code}\t{row.get('steady_fps')}\t{row.get('steady_fps_nodefrag')}\t"
                         f"{end - start:.1f}\t{log.name}\n")
        if row.get("status") == "hard_failure" and not optional:
            raise RunFailed(f"{label}: {result_line}")
        return row

    def run(self, label: str, bench_arguments: list, timeout: float, wrapper_arguments: list = (), optional: bool = False) -> dict:
        process, log, start = self.launch(label, bench_arguments, wrapper_arguments)
        return self.finish(label, process, log, start, timeout, optional)

    # -- the protocol's pieces ---------------------------------------------------------------------------------
    def calibrate(self, case: str, label: str, timeout: float = 1800.0, device_map: str = DEVICE_MAP) -> tuple:
        """The pilot(s) of the sec:slab rule (e7_lib.sh e7_calibrate: a --calibrate-only process prints no final bench
        line, so it is not classified by parse_run_v6; it must exit 0 and write the weights file).
        Returns (weights file, {"rc", "seconds", "calibrate_lines"}, the weights record)."""
        weights_file = self.directory / "weights" / f"{label}.json"
        if weights_file.exists():
            weights_file.unlink()
        process, log, start = self.launch(label, ["--case", str(case_yaml(case)), "--weights", "auto", "--calibrate-only",
                                                  "--weights-file", str(weights_file), "--device-map", device_map, *COMMON])
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = 124
        seconds = time.time() - start
        lines = [line.rstrip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("[calibrate]")]
        status = "calibrated" if return_code == 0 and weights_file.exists() else "hard_failure"
        with self.index.open("a", encoding="utf-8") as handle:
            handle.write(f"{label}\t{status}\t{return_code}\t-\t-\t{seconds:.1f}\t{log.name}\n")
        for line in lines[-3:]:
            print(line[:200], flush=True)
        if status != "calibrated":
            raise RunFailed(f"{label}: rc={return_code}, weights file {'present' if weights_file.exists() else 'missing'}")
        record = json.loads(weights_file.read_text(encoding="utf-8"))
        return weights_file, {"rc": return_code, "seconds": seconds, "calibrate_lines": lines, "status": status}, record

    def chain(self, case: str, label: str, weights_file: pathlib.Path = None, weights: str = None, timeout: float = 1800.0,
              trace: bool = False, device_map: str = DEVICE_MAP, steps: int = STEPS, warmup: int = WARMUP) -> dict:
        """One K-run: --weights-file (calibrated) or explicit --weights; traced runs keep the seam check, timing runs skip it."""
        arguments = ["--case", str(case_yaml(case))]
        arguments += ["--weights-file", str(weights_file)] if weights_file else ["--weights", weights]
        arguments += ["--device-map", device_map, *COMMON, "--max-steps", str(steps), "--warmup", str(warmup)]
        if trace:
            trace_directory = self.directory / "traces" / label
            if trace_directory.exists():
                shutil.rmtree(trace_directory)
            arguments += ["--step-trace", str(trace_directory)]
        else:
            arguments += ["--no-seam-check"]
        return self.run(label, arguments, timeout)

    def references(self, case: str, label: str, members=(0, 1), timeout: float = 1800.0, steps: int = STEPS, warmup: int = WARMUP) -> dict:
        """K = 1 on every member card at once, one start barrier; {member: row}."""
        barrier = self.directory / "barriers" / label
        if barrier.exists():
            shutil.rmtree(barrier)
        barrier.mkdir(parents=True)
        launched = {}
        for member in members:
            member_label = f"{label}_g{member}"
            launched[member] = (member_label, *self.launch(member_label, ["--case", str(case_yaml(case)), "--weights", "1", "--device-map",
                                                                         str(member), *COMMON, "--max-steps", str(steps), "--warmup",
                                                                         str(warmup), "--no-seam-check"],
                                                           ["--barrier", str(barrier), "--barrier-parties", str(len(members)),
                                                            "--barrier-timeout", str(timeout / 2)]))
        rows, failures = {}, []
        for member, (member_label, process, log, start) in launched.items():
            try:
                rows[member] = self.finish(member_label, process, log, start, timeout)
            except RunFailed as error:
                failures.append(str(error))
        if failures:
            raise RunFailed("; ".join(failures))
        return rows
