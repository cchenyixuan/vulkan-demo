"""micropolar_cost.py - cost of the thesis configuration (docs/micropolar_design.md, Stage 3).

Throughput of one card on the 2-D lid-driven cavity at about 1M and 4M particles for three configurations:

  paper              the Newtonian case as paper 2 measured it (Monaghan-Gingold viscous term,
                     epsilon_squared_factor 0.0025, the case's own wall option)
  thesis             Newtonian with the thesis configuration: numerics.viscous_form laplacian,
                     epsilon_squared_factor 1e-6, wall_boundary adami
  thesis_micropolar  the thesis configuration with physics.micropolar on (materials/micropolar.yaml,
                     micropolar_water_re1000)

and one control, paper_adami: the paper configuration with wall_boundary adami only, which separates the wall
option from the viscous form in the step from the first row to the second.

Protocol = the reference working point of paper 2's Section 5.1: a K = 1 chain (_run_v7_chain_bench.py, depth 2,
per-direction sync scheme, pool safety 1.2, defrag at the case cadence, validation layers off, no GPU timers,
no seam check), warm-up 1000 steps, then 6000 (1M) / 3000 (4M) measured steps, three trials interleaved
(trial-major, then size, then configuration), each run its own process. Throughput = particles (fluid and wall)
times the steady steps per second of the bench's post-warm-up window.

    run        the throughput table                               (--device, --trials, --out)
    trace      one more run per size and configuration with the per-kernel GPU timestamps of
               --step-trace-detail full (a separate run: the timers cost throughput)
    pair       the K = 2 point: one case on --device-map A,B with calibrated weights, three trials, each trial
               preceded by a K = 1 run of the same case on both cards at the same time (the reference of the
               efficiency: the sum of the two simultaneous K = 1 throughputs)
    summarise  the tables from the logs under --out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

CASES = {
    "paper": {"1m": "cases/lid_driven_cavity_2d_gen/case.yaml", "4m": "cases/lid_driven_cavity_2d_4m/case.yaml"},
    "paper_adami": {"1m": "cases/cost_cavity_1m_paper_adami/case.yaml",
                    "4m": "cases/cost_cavity_4m_paper_adami/case.yaml"},
    "thesis": {"1m": "cases/cost_cavity_1m_thesis/case.yaml", "4m": "cases/cost_cavity_4m_thesis/case.yaml"},
    "thesis_micropolar": {"1m": "cases/cost_cavity_1m_thesis_micropolar/case.yaml",
                          "4m": "cases/cost_cavity_4m_thesis_micropolar/case.yaml"},
}
# size: (warm-up steps, measured steps)
PROTOCOL = {"1m": (1000, 6000), "4m": (1000, 3000)}
COMMON = ["--sync-scheme", "per-direction", "--depth", "2", "--pool-safety", "1.2", "--no-seam-check"]
BENCH = "experiment/v7/_run_v7_chain_bench.py"

STEADY = re.compile(r"\[chain_v7\] STEADY \(post-warmup (\d+)\): (\d+) steps in ([0-9.]+)s = ([0-9.]+) fps")
FINAL = re.compile(r"\[chain_v7\] final: total=([0-9,]+) \(expected ([0-9,]+)\) drift=(-?\d+) .*"
                   r"overflow_total=(\d+) far_migration_total=(\d+)")
DEVICE_LOCAL = re.compile(r"\[SimV7\] device-local buffers: (\d+), ([0-9.]+) MB")
DEFRAG_SCRATCH = re.compile(r"\[SimV7\] defrag scratch buffers: (\d+), ([0-9.]+) MB")
INITIALIZED = re.compile(r"\[SimV7\] init complete on (.+?) \(own=(\d+)")
WEIGHTS = re.compile(r"\[chain_v7\] K=(\d+) weights=(\[[^\]]*\])")


def _environment() -> dict:
    environment = dict(os.environ)
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    return environment


def _bench(log_path: pathlib.Path, options: list[str], wait: bool = True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", BENCH, *options], cwd=_REPO_ROOT, env=_environment(),
                               stdout=log, stderr=subprocess.STDOUT)
    if not wait:
        return process, log
    code = process.wait()
    log.close()
    return code


def parse_log(log_path: pathlib.Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    record = {"log": log_path.name, "failed": "VALIDATION FAILED" in text or "Traceback" in text}
    match = STEADY.search(text)
    if match:
        record.update(warmup=int(match.group(1)), steady_steps=int(match.group(2)),
                      steady_seconds=float(match.group(3)), steps_per_second=float(match.group(4)))
    match = FINAL.search(text)
    if match:
        record.update(particles=int(match.group(1).replace(",", "")), drift=int(match.group(3)),
                      overflow=int(match.group(4)), far_migration=int(match.group(5)))
    # one line per slab
    record["device_local_megabytes"] = [float(item[1]) for item in DEVICE_LOCAL.findall(text)]
    record["device_local_buffers"] = [int(item[0]) for item in DEVICE_LOCAL.findall(text)]
    record["defrag_scratch_megabytes"] = [float(item[1]) for item in DEFRAG_SCRATCH.findall(text)]
    record["defrag_scratch_buffers"] = [int(item[0]) for item in DEFRAG_SCRATCH.findall(text)]
    record["devices"] = [item[0] for item in INITIALIZED.findall(text)]
    record["slots"] = [int(item[1]) for item in INITIALIZED.findall(text)]
    match = WEIGHTS.search(text)
    if match:
        record.update(slabs=int(match.group(1)), weights=json.loads(match.group(2)))
    if "steps_per_second" in record and "particles" in record:
        record["particle_steps_per_second"] = record["particles"] * record["steps_per_second"]
    return record


def command_run(arguments) -> int:
    out = _REPO_ROOT / arguments.out
    for trial in range(arguments.first_trial, arguments.trials + 1):
        for size in arguments.sizes:
            warmup, measured = PROTOCOL[size]
            for configuration in arguments.configurations:
                log_path = out / f"{size}_{configuration}_t{trial}.log"
                if log_path.exists() and "steps_per_second" in parse_log(log_path):
                    print(f"[cost] {log_path.name}: done already", flush=True)
                    continue
                code = _bench(log_path, ["--case", CASES[configuration][size], "--weights", "1", "--device-map",
                                         str(arguments.device), *COMMON, "--warmup", str(warmup), "--max-steps",
                                         str(warmup + measured)])
                record = parse_log(log_path)
                print(f"[cost] {log_path.name}: exit {code}, {record.get('steps_per_second')} steps/s", flush=True)
    return 0


def command_trace(arguments) -> int:
    out = _REPO_ROOT / arguments.out
    for size in arguments.sizes:
        warmup = PROTOCOL[size][0]
        for configuration in arguments.configurations:
            trace_dir = out / f"trace_{size}_{configuration}"
            if (trace_dir / "steps_device.csv").exists():
                print(f"[cost] {trace_dir.name}: done already", flush=True)
                continue
            code = _bench(out / f"trace_{size}_{configuration}.log",
                          ["--case", CASES[configuration][size], "--weights", "1", "--device-map",
                           str(arguments.device), *COMMON, "--warmup", str(warmup), "--max-steps",
                           str(warmup + arguments.steps), "--step-trace", str(trace_dir), "--step-trace-detail",
                           "full"])
            print(f"[cost] {trace_dir.name}: exit {code}", flush=True)
    return 0


def command_pair(arguments) -> int:
    out = _REPO_ROOT / arguments.out
    warmup, measured = PROTOCOL[arguments.size]
    case = CASES[arguments.configuration][arguments.size]
    devices = arguments.device_map.split(",")
    steps = ["--warmup", str(warmup), "--max-steps", str(warmup + measured)]
    label = f"pair_{arguments.size}_{arguments.configuration}"
    weights_file = out / f"{label}_weights.json"
    if not weights_file.exists():
        code = _bench(out / f"{label}_calibrate.log",
                      ["--case", case, "--weights", "auto", "--weights-file", str(weights_file), "--calibrate-only",
                       "--device-map", arguments.device_map, *COMMON])
        print(f"[cost] calibration: exit {code}", flush=True)
        if code or not weights_file.exists():
            return 1
    for trial in range(1, arguments.trials + 1):
        # the K = 1 reference of every participating card, at the same time
        logs = [out / f"{label}_k1_device{device}_t{trial}.log" for device in devices]
        if not all(path.exists() and "steps_per_second" in parse_log(path) for path in logs):
            running = [_bench(path, ["--case", case, "--weights", "1", "--device-map", device, *COMMON, *steps],
                              wait=False) for path, device in zip(logs, devices)]
            for process, log in running:
                process.wait()
                log.close()
        log_path = out / f"{label}_k2_t{trial}.log"
        if not (log_path.exists() and "steps_per_second" in parse_log(log_path)):
            _bench(log_path, ["--case", case, "--weights-file", str(weights_file), "--device-map",
                              arguments.device_map, *COMMON, *steps])
        print(f"[cost] {label} trial {trial}: K = 1 "
              + ", ".join(f"{parse_log(path).get('steps_per_second')}" for path in logs)
              + f"; K = 2 {parse_log(log_path).get('steps_per_second')} steps/s", flush=True)
    return 0


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def kernel_split(trace_dir: pathlib.Path, warmup: int, defrag_cadence: int = 1000) -> dict:
    """Mean interval, in microseconds, between successive GPU timestamps of a step (steps after the warm-up, steps
    with a defrag left out), labelled by the tick that ends the interval. Ticks: host-clock nanoseconds."""
    sums: dict[str, float] = {}
    order: list[str] = []
    rows = 0
    with open(trace_dir / "steps_device.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        ticks = [name for name in reader.fieldnames if name.endswith("_start") or name.endswith("_end")]
        for row in reader:
            step = int(row["step"])
            if step < warmup or row.get("complete") != "1" or step % defrag_cadence == 0:
                continue
            present = [(name, int(row[name])) for name in ticks if row[name] != ""]
            for (_previous_name, previous), (name, value) in zip(present, present[1:]):
                if name not in sums:
                    sums[name] = 0.0
                    order.append(name)
                sums[name] += value - previous
            rows += 1
    intervals = {name: sums[name] / rows / 1000.0 for name in order if rows}
    return {"steps": rows, "intervals_microseconds": intervals,
            "sum_microseconds": float(sum(intervals.values()))}


def command_summarise(arguments) -> int:
    out = _REPO_ROOT / arguments.out
    summary = {"protocol": {size: {"warmup": PROTOCOL[size][0], "measured": PROTOCOL[size][1]} for size in PROTOCOL},
               "cases": CASES, "throughput": {}, "kernels": {}, "pair": {}}
    print(f"{'size':<5}{'configuration':<20}{'particles':>11}{'steps/s':>20}{'M particle-steps/s':>22}"
          f"{'vs paper':>10}{'device-local MB':>17}{'scratch MB':>12}{'B/particle':>12}{'B/slot':>9}{'trials':>8}")
    for size in PROTOCOL:
        reference: dict[int, float] = {}
        for configuration in CASES:
            records = [parse_log(path) for path in sorted(out.glob(f"{size}_{configuration}_t*.log"))]
            records = [record for record in records if "particle_steps_per_second" in record]
            if not records:
                continue
            rates = [record["steps_per_second"] for record in records]
            trial_numbers = [int(record["log"].rsplit("_t", 1)[1].split(".")[0]) for record in records]
            mean, deviation = _mean_std(rates)
            particles = records[0]["particles"]
            megabytes = records[0]["device_local_megabytes"][0] + records[0]["defrag_scratch_megabytes"][0]
            item = {"particles": particles, "slots": records[0]["slots"][0], "device": records[0]["devices"][0],
                    "steps_per_second": rates, "steps_per_second_mean": mean, "steps_per_second_std": deviation,
                    "particle_steps_per_second_mean": particles * mean,
                    "particle_steps_per_second_std": particles * deviation,
                    "device_local_megabytes": records[0]["device_local_megabytes"][0],
                    "device_local_buffers": records[0]["device_local_buffers"][0],
                    "defrag_scratch_megabytes": records[0]["defrag_scratch_megabytes"][0],
                    "defrag_scratch_buffers": records[0]["defrag_scratch_buffers"][0],
                    "bytes_per_particle": megabytes * 1024 * 1024 / particles,
                    "bytes_per_slot": megabytes * 1024 * 1024 / records[0]["slots"][0],
                    "drift": [record.get("drift") for record in records],
                    "failed": any(record["failed"] for record in records)}
            if configuration == "paper":
                reference = dict(zip(trial_numbers, rates))
            # trial by trial against the paper row of the same trial (the trials are interleaved)
            ratios = [rate / reference[trial] for trial, rate in zip(trial_numbers, rates) if trial in reference]
            item["trials"] = trial_numbers
            item["ratio_to_paper_by_trial"] = ratios
            item["ratio_to_paper"] = statistics.mean(ratios) if ratios else None
            item["ratio_to_paper_std"] = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
            summary["throughput"][f"{size}_{configuration}"] = item
            ratio_text = f"{item['ratio_to_paper']:.4f}" if ratios else "-"
            print(f"{size:<5}{configuration:<20}{particles:>11,}{mean:>12.1f} +- {deviation:<5.1f}"
                  f"{particles * mean / 1e6:>14.1f} +- {particles * deviation / 1e6:<5.1f}{ratio_text:>10}"
                  f"{item['device_local_megabytes']:>17.1f}{item['defrag_scratch_megabytes']:>12.1f}"
                  f"{item['bytes_per_particle']:>12.1f}{item['bytes_per_slot']:>9.1f}{len(records):>8}"
                  + ("  FAILED" if item["failed"] or any(item["drift"]) else ""))
    for size in PROTOCOL:
        for configuration in CASES:
            trace_dir = out / f"trace_{size}_{configuration}"
            if (trace_dir / "steps_device.csv").exists():
                summary["kernels"][f"{size}_{configuration}"] = kernel_split(trace_dir, PROTOCOL[size][0])
    if summary["kernels"]:
        names: list[str] = []
        for item in summary["kernels"].values():
            names += [name for name in item["intervals_microseconds"] if name not in names]
        print("\nmean interval ending at each GPU timestamp, microseconds per step (traced runs)")
        print(f"{'tick':<38}" + "".join(f"{key:>24}" for key in summary["kernels"]))
        for name in names:
            print(f"{name:<38}" + "".join(f"{item['intervals_microseconds'].get(name, float('nan')):>24.1f}"
                                          for item in summary["kernels"].values()))
        print(f"{'sum':<38}" + "".join(f"{item['sum_microseconds']:>24.1f}" for item in summary["kernels"].values()))
    for weights_file in sorted(out.glob("pair_*_weights.json")):
        label = weights_file.name[:-len("_weights.json")]
        chain = [parse_log(path) for path in sorted(out.glob(f"{label}_k2_t*.log"))]
        chain = [record for record in chain if "steps_per_second" in record]
        if not chain:
            continue
        trials = []
        for record in chain:
            trial = record["log"].rsplit("_t", 1)[1].split(".")[0]
            references = [parse_log(path) for path in sorted(out.glob(f"{label}_k1_device*_t{trial}.log"))]
            trials.append({"trial": int(trial), "chain_steps_per_second": record["steps_per_second"],
                           "reference_steps_per_second": {item["devices"][0]: item["steps_per_second"]
                                                          for item in references},
                           "efficiency": record["steps_per_second"]
                           / sum(item["steps_per_second"] for item in references),
                           "drift": record.get("drift"), "weights": record.get("weights")})
        rate_mean, rate_std = _mean_std([trial["chain_steps_per_second"] for trial in trials])
        efficiency_mean, efficiency_std = _mean_std([trial["efficiency"] for trial in trials])
        megabytes = [a + b for a, b in zip(chain[0]["device_local_megabytes"], chain[0]["defrag_scratch_megabytes"])]
        summary["pair"][label] = {"particles": chain[0]["particles"], "devices": chain[0]["devices"],
                                  "weights": chain[0].get("weights"), "trials": trials,
                                  "steps_per_second_mean": rate_mean, "steps_per_second_std": rate_std,
                                  "efficiency_mean": efficiency_mean, "efficiency_std": efficiency_std,
                                  "megabytes_per_slab": megabytes}
        print(f"\n{label}: K = 2 on {chain[0]['devices']}, weights {chain[0].get('weights')}: "
              f"{rate_mean:.1f} +- {rate_std:.1f} steps/s = {chain[0]['particles'] * rate_mean / 1e6:.1f} M "
              f"particle-steps/s; efficiency against the two simultaneous K = 1 runs {efficiency_mean:.4f} +- "
              f"{efficiency_std:.4f}")
        for trial in trials:
            print(f"  trial {trial['trial']}: K = 2 {trial['chain_steps_per_second']:.1f}; K = 1 "
                  f"{trial['reference_steps_per_second']}; efficiency {trial['efficiency']:.4f}; drift "
                  f"{trial['drift']}")
    if arguments.json:
        path = _REPO_ROOT / arguments.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="logs/micropolar/cost")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("run", command_run), ("trace", command_trace)):
        command = commands.add_parser(name)
        command.add_argument("--device", type=int, default=0)
        command.add_argument("--sizes", nargs="+", default=list(PROTOCOL), choices=list(PROTOCOL))
        command.add_argument("--configurations", nargs="+", default=list(CASES), choices=list(CASES))
        if name == "run":
            command.add_argument("--trials", type=int, default=3, help="number of the last trial")
            command.add_argument("--first-trial", type=int, default=1,
                                 help="number of the first trial (a later session adds trials to the same directory)")
        else:
            command.add_argument("--steps", type=int, default=2000, help="traced steps after the warm-up")
        command.set_defaults(handler=handler)
    pair = commands.add_parser("pair")
    pair.add_argument("--device-map", default="0,1")
    pair.add_argument("--size", default="4m", choices=list(PROTOCOL))
    pair.add_argument("--configuration", default="thesis_micropolar", choices=list(CASES))
    pair.add_argument("--trials", type=int, default=3)
    pair.set_defaults(handler=command_pair)
    summarise = commands.add_parser("summarise")
    summarise.add_argument("--json", default=None)
    summarise.set_defaults(handler=command_summarise)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
