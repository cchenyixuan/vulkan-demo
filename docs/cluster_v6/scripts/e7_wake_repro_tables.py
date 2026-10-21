"""
e7_wake_repro_tables.py — tables of the E7 wake reproduction (e7_wake_repro.py) for the report.

Per run directory (one pass of the program: DIR/cases.json, and DIR/waits.csv or waits.csv.gz when present):
  1. per case (cases.json): signal, waiters, method, pattern, load, blocked waits, late (> 5 ms after the signal),
     latency p50 / p99 / max over the blocked waits, the late waits' return after their own start (p50), how many
     of them fell in 10.00-10.25 ms, late first starters / later starters, driver calls per second, the load
     thread's work units per busy second against the case without waiters;
  2. per waiter of the multi-waiter cases: blocked, late, latency, driver calls per waiting second, the waiter's
     Python share while waiting (GIL) and its thread CPU per wall second;
  3. from waits.csv(.gz): per case the on-time wake p50 / p90 / p99, the late waits' return after their own start
     (p01 / p50 / p99 and the most common 0.25 ms bins), after the signal, how long before the signal they began,
     and per iteration how many waiters were late.

  4. with --histogram OUT.json (first RUN_DIR only): per case the blocked waits' return after their own start in
     0.25 ms bins from 0 to 12 ms (the figure of the batch-3 report; counts outside the range listed apart).

Usage: python docs/cluster_v6/scripts/e7_wake_repro_tables.py RUN_DIR [RUN_DIR ...] [--histogram OUT.json [CASE ...]]
       (default cases of the histogram: HISTOGRAM_CASES)
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

LATE_NANOSECONDS = 5_000_000
HISTOGRAM_BIN_MS = 0.25
HISTOGRAM_BINS = 48                 # 0-12 ms
HISTOGRAM_CASES = ("gpu_w2_infinite_load", "gpu_w2_relay_condition_load", "gpu_w2_timeout0.2_load",
                   "gpu_w1_infinite_load")


def number(value, digits=0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def case_tables(directory: pathlib.Path) -> None:
    data = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    environment = data["environment"]
    print(f"## {directory.name}: {environment['device']} driver {environment.get('driver_info')} "
          f"({environment.get('driver_name')}), python {environment['python'].split()[0]}, switch interval "
          f"{environment['switch_interval_s'] * 1e3:.1f} ms, host clock {environment['host_clock']}, "
          f"time domain {environment['time_domain']}, affinity {len(environment.get('affinity') or [])} CPUs")
    print("| case | signal | waiters | method | pattern | load | blocked | late | late % | p50 us | p99 us | max us | "
          "late from start p50 us | in 10.00-10.25 ms | first / later starters late | driver calls/s | "
          "load vs baseline |")
    print("|" + "---|" * 17)
    for case in data["cases"]:
        if not case["waiters"]:
            print(f"| {case['case']} | - | 0 | - | - | {case['load']} | - | - | - | - | - | - | - | - | - | - | "
                  f"{number(case['load_units_per_busy_s'])} units/busy s |")
            continue
        first, later = case["late_first_starter"], case["late_later_starters"]
        relative = case.get("load_relative_to_baseline")
        print(f"| {case['case']} | {case['signal']} | {case['waiters']} | {case['method']} | {case['pattern']} | "
              f"{case['load']} | {case['blocked']} | {case['late']} | {number(100 * case['late_fraction'], 2)} | "
              f"{number(case['latency_us_p50'])} | {number(case['latency_us_p99'])} | {number(case['latency_us_max'])} | "
              f"{number(case['late_from_start_us_p50'])} | {case['late_in_10_00_to_10_25_ms']} | "
              f"{first[0]}/{first[1]} , {later[0]}/{later[1]} | {number(case['driver_calls_per_s_all_waiters'])} | "
              f"{number(100 * relative, 1) + ' %' if relative else '-'} |")
    print()
    print("per waiter (cases with more than one waiter):")
    for case in data["cases"]:
        if case["waiters"] < 2:
            continue
        parts = []
        for waiter in case["per_waiter"]:
            parts.append(f"w{waiter['waiter']} blocked {waiter['blocked']} late {waiter['late']} p50 "
                         f"{number(waiter['latency_us_p50'])} p99 {number(waiter['latency_us_p99'])} max "
                         f"{number(waiter['latency_us_max'])} calls/waiting-s {number(waiter['driver_calls_per_wait_s'])} "
                         f"python {number(100 * waiter['python_share_while_waiting'], 2)} % cpu/wall "
                         f"{number(waiter['thread_cpu_per_wall_s'], 3)}")
        print(f"  {case['case']}: " + "; ".join(parts))
    print()


def open_waits(directory: pathlib.Path):
    if (directory / "waits.csv.gz").exists():
        return gzip.open(directory / "waits.csv.gz", "rt", encoding="utf-8")
    if (directory / "waits.csv").exists():
        return open(directory / "waits.csv", encoding="utf-8", newline="")
    return None


def wait_details(directory: pathlib.Path) -> None:
    handle = open_waits(directory)
    if handle is None:
        print(f"(no waits.csv in {directory})\n")
        return
    cases = defaultdict(list)
    with handle:
        for row in csv.DictReader(handle):
            cases[row["case"]].append(row)
    print(f"wait details ({directory.name}):")
    for case, rows in cases.items():
        blocked = [row for row in rows if int(row["t_start"]) < int(row["t_signal"])]
        late = [row for row in blocked if int(row["t_wake"]) - int(row["t_signal"]) > LATE_NANOSECONDS]
        from_start = [(int(row["t_wake"]) - int(row["t_start"])) / 1e6 for row in late]
        from_signal = [(int(row["t_wake"]) - int(row["t_signal"])) / 1e6 for row in late]
        began_before = [(int(row["t_signal"]) - int(row["t_start"])) / 1e6 for row in late]
        on_time = [(int(row["t_wake"]) - int(row["t_signal"])) / 1e3 for row in blocked
                   if int(row["t_wake"]) - int(row["t_signal"]) <= LATE_NANOSECONDS]
        per_iteration = defaultdict(lambda: [0, 0])
        for row in blocked:
            entry = per_iteration[row["iteration"]]
            entry[0] += 1
            entry[1] += int(row["t_wake"]) - int(row["t_signal"]) > LATE_NANOSECONDS
        pattern = Counter(f"{late_count}/{count}" for count, late_count in per_iteration.values())
        bins = Counter(f"{int(value * 4) / 4:.2f}" for value in from_start)
        top = ", ".join(f"{key} ms: {count}" for key, count in sorted(bins.items(), key=lambda item: -item[1])[:4])
        text = (f"  {case}: blocked {len(blocked)}, late {len(late)}; on-time wake p50/p90/p99 "
                f"{percentile(on_time, 50):.0f}/{percentile(on_time, 90):.0f}/{percentile(on_time, 99):.0f} us")
        if late:
            text += (f"; late return after its start p01/p50/p99 {percentile(from_start, 1):.3f}/"
                     f"{percentile(from_start, 50):.3f}/{percentile(from_start, 99):.3f} ms (top bins {top}); after the "
                     f"signal p50 {percentile(from_signal, 50):.2f} ms; began before the signal p50 "
                     f"{percentile(began_before, 50):.2f} ms")
        if any(not key.startswith("0/") for key in pattern):
            text += f"; late per iteration {dict(sorted(pattern.items()))}"
        print(text)
    print()


def histogram(directory: pathlib.Path, cases, path: pathlib.Path) -> None:
    handle = open_waits(directory)
    if handle is None:
        print(f"(no waits.csv in {directory}: no histogram)")
        return
    counts = {case: [0] * HISTOGRAM_BINS for case in cases}
    outside = {case: 0 for case in cases}
    with handle:
        for row in csv.DictReader(handle):
            if row["case"] not in counts or int(row["t_start"]) >= int(row["t_signal"]):
                continue
            index = int((int(row["t_wake"]) - int(row["t_start"])) / 1e6 / HISTOGRAM_BIN_MS)
            if 0 <= index < HISTOGRAM_BINS:
                counts[row["case"]][index] += 1
            else:
                outside[row["case"]] += 1
    path.write_text(json.dumps({"run": directory.name, "bins_ms": HISTOGRAM_BIN_MS, "counts": counts,
                                "outside": {case: value for case, value in outside.items() if value}}),
                    encoding="utf-8")
    print(f"wrote {path} ({', '.join(f'{case} {sum(values)}' for case, values in counts.items())})")


def main() -> int:
    arguments = sys.argv[1:]
    histogram_path, histogram_cases = None, HISTOGRAM_CASES
    if "--histogram" in arguments:
        position = arguments.index("--histogram")
        histogram_path = pathlib.Path(arguments[position + 1])
        histogram_cases = tuple(arguments[position + 2:]) or HISTOGRAM_CASES
        arguments = arguments[:position]
    directories = list(map(pathlib.Path, arguments))
    for directory in directories:
        case_tables(directory)
        wait_details(directory)
    if histogram_path is not None and directories:
        histogram(directories[0], histogram_cases, histogram_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
