"""
e7_late_wake.py — E7 batch report: late wake-ups of the transport workers, from the step traces
(traces/<point>_{E,C}/steps_link.csv and steps_device.csv, run_chain_v6.py --step-trace).

The worker of link a -> b waits twice per step (transport_v7.py _run, both vkWaitSemaphores with an INFINITE
timeout): the source wait on a's readback_done(n) of a -> b, then the dest guard on b's readback_done(n) of b -> a,
which is the value the worker of link b -> a waits on as its own source wait. Only waits that blocked count: the
source wait when the worker dequeued before the readback ended, the dest guard when it began (worker_source_wait)
before b's readback ended. A wake-up is late when it comes more than LATE_NANOSECONDS after the signal (the GPU's
readback_end on the fitted host clock).

Per trace directory, the steady part (the last two thirds of the steps):
  1. per link: wake-up latency of the blocked waits (p50 / p90 / p99, µs), the count of late ones and their median;
     the step period of sim 0 (a_start to a_start);
  2. late dest guards: wait duration (worker_dest_guard - worker_source_wait, host clock only, no clock fit), its
     quantiles and a 250 µs histogram, and how long before the signal the wait began;
  3. shared waits: per blocked dest guard, was the reverse link's source wait asleep over the same signal (two threads
     on one semaphore value) or alone, which began first, and the late share of each class; the reverse source waits'
     own late count;
  4. cost: per receiving sim the gap between its phase B end and phase C start on steps with and without a late
     inbound dest guard, the step period on those steps and the fps if every step were clean.

Usage:
    python docs/cluster_v6/scripts/e7_late_wake.py TRACE_DIR [TRACE_DIR ...] > late_wake.txt
"""

from __future__ import annotations

import csv
import pathlib
import statistics
import sys

LATE_NANOSECONDS = 5_000_000
HISTOGRAM_BIN_MICROSECONDS = 250


def quantile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else float("nan")


def mean_or_nan(values: list) -> float:
    return statistics.mean(values) if values else float("nan")


def load_trace(directory: pathlib.Path):
    with open(directory / "steps_link.csv", encoding="utf-8") as handle:
        link_rows = [row for row in csv.DictReader(handle) if row["complete"] == "1"]
    with open(directory / "steps_device.csv", encoding="utf-8") as handle:
        device_rows = [row for row in csv.DictReader(handle) if row["complete"] == "1"]
    steps = sorted({int(row["step"]) for row in link_rows})
    steady_steps = set(steps[len(steps) // 3:])
    by_step_and_link = {(int(row["step"]), row["sender"], row["receiver"]): row for row in link_rows}
    return link_rows, device_rows, steady_steps, by_step_and_link


def blocked_dest_guards(link_rows, steady_steps, by_step_and_link):
    """Every dest guard that blocked: (row, reverse row, signal, guard start, wake)."""
    guards = []
    for row in link_rows:
        step = int(row["step"])
        if step not in steady_steps:
            continue
        reverse = by_step_and_link.get((step, row["receiver"], row["sender"]))
        if reverse is None or not reverse["readback_end"] or not row["worker_source_wait"] or not row["worker_dest_guard"]:
            continue
        signal = int(reverse["readback_end"])
        guard_start = int(row["worker_source_wait"])
        if guard_start >= signal:
            continue
        guards.append((row, reverse, signal, guard_start, int(row["worker_dest_guard"])))
    return guards


def sim_zero_periods(device_rows, steady_steps) -> dict:
    starts = {int(row["step"]): int(row["a_start"]) for row in device_rows if row["sim"] == "0" and row["a_start"]}
    return {step: (starts[step + 1] - starts[step]) / 1e3 for step in sorted(steady_steps)
            if step in starts and step + 1 in starts}


def report_latency(link_rows, device_rows, steady_steps, by_step_and_link) -> None:
    print("  1. wake-up latency of blocked waits (µs after the signal)")
    total_late = 0
    for link in sorted({row["link"] for row in link_rows}):
        blocked = {"source": [], "dest": []}
        counted = 0
        for row in link_rows:
            step = int(row["step"])
            if row["link"] != link or step not in steady_steps:
                continue
            counted += 1
            dequeue, source_wait, dest_guard = (int(row[name]) for name in
                                                ("worker_dequeue", "worker_source_wait", "worker_dest_guard"))
            readback_end = int(row["readback_end"])
            if readback_end > dequeue:
                blocked["source"].append((source_wait - readback_end) / 1e3)
            reverse = by_step_and_link.get((step, row["receiver"], row["sender"]))
            if reverse is not None and reverse["readback_end"] and int(reverse["readback_end"]) > source_wait:
                blocked["dest"].append((dest_guard - int(reverse["readback_end"])) / 1e3)
        for wait_name, values in blocked.items():
            late = [value for value in values if value > LATE_NANOSECONDS / 1e3]
            total_late += len(late)
            print(f"     {link} {wait_name:6s} blocked {len(values):5d}/{counted}: p50 {quantile(values, 0.5):7.0f} "
                  f"p90 {quantile(values, 0.9):7.0f} p99 {quantile(values, 0.99):7.0f} | late {len(late):5d} "
                  f"(median {statistics.median(late) if late else float('nan'):.0f})")
    periods = list(sim_zero_periods(device_rows, steady_steps).values())
    print(f"     late wake-ups in total {total_late}; step period p50 {quantile(periods, 0.5):.0f} µs, "
          f"p90 {quantile(periods, 0.9):.0f}, mean {mean_or_nan(periods):.0f} -> {1e6 / mean_or_nan(periods):.1f} fps")


def report_durations(guards) -> None:
    late_durations, head_starts, on_time_durations = [], [], []
    for _, _, signal, guard_start, wake in guards:
        duration = (wake - guard_start) / 1e3
        if wake - signal > LATE_NANOSECONDS:
            late_durations.append(duration)
            head_starts.append((signal - guard_start) / 1e3)
        else:
            on_time_durations.append(duration)
    print(f"  2. dest guard wait durations (host clock): blocked {len(guards)}, late {len(late_durations)}")
    if late_durations:
        print("     late: p01 %.0f p10 %.0f p50 %.0f p90 %.0f p99 %.0f µs; began before the signal by p50 %.0f p90 %.0f µs"
              % tuple([quantile(late_durations, fraction) for fraction in (0.01, 0.1, 0.5, 0.9, 0.99)]
                      + [quantile(head_starts, 0.5), quantile(head_starts, 0.9)]))
        histogram = {}
        for duration in late_durations:
            histogram_bin = int(duration // HISTOGRAM_BIN_MICROSECONDS) * HISTOGRAM_BIN_MICROSECONDS
            histogram[histogram_bin] = histogram.get(histogram_bin, 0) + 1
        print("     late, by 250 µs bin: " + ", ".join(f"{histogram_bin / 1000:.2f} ms: {count}"
                                                      for histogram_bin, count in sorted(histogram.items())))
    if on_time_durations:
        print("     on time: p50 %.0f p90 %.0f p99 %.0f µs" % tuple(quantile(on_time_durations, fraction)
                                                                for fraction in (0.5, 0.9, 0.99)))


def report_shared(guards) -> None:
    classes = {}
    reverse_asleep = reverse_late = 0
    for _, reverse, signal, guard_start, wake in guards:
        reverse_start = int(reverse["worker_dequeue"])
        reverse_wake = int(reverse["worker_source_wait"])
        shared = reverse_start < signal
        if shared:
            reverse_asleep += 1
            reverse_late += (reverse_wake - signal) > LATE_NANOSECONDS
            order = "guard first" if guard_start < reverse_start else "reverse first"
        else:
            order = "-"
        counts = classes.setdefault(("shared" if shared else "alone", order), [0, 0])
        counts[0] += 1
        counts[1] += (wake - signal) > LATE_NANOSECONDS
    print("  3. shared waits (reverse link's source wait asleep over the same signal)")
    for (sharing, order), (count, late) in sorted(classes.items()):
        print(f"     {sharing:6s} {order:13s}: blocked dest guards {count:6d}, late {late:6d} "
              f"({100.0 * late / max(count, 1):5.1f} %)")
    print(f"     reverse source waits asleep over the same signal {reverse_asleep}, late {reverse_late}")


def report_cost(device_rows, steady_steps, guards) -> None:
    late_receivers = set()
    for row, _, signal, _, wake in guards:
        if wake - signal > LATE_NANOSECONDS:
            late_receivers.add((int(row["step"]), row["receiver"]))
    late_steps = {step for step, _ in late_receivers}
    gaps = {}
    for row in device_rows:
        step = int(row["step"])
        if step not in steady_steps or not row["c_start"] or not row["b_end"]:
            continue
        gap = (int(row["c_start"]) - int(row["b_end"])) / 1e3
        kind = "late" if (step, row["sim"]) in late_receivers else "clean"
        gaps.setdefault(row["sim"], {"late": [], "clean": []})[kind].append(gap)
    print(f"  4. cost: steps {len(steady_steps)}, with a late inbound dest guard {len(late_steps)}")
    for sim in sorted(gaps, key=int):
        late_gaps, clean_gaps = gaps[sim]["late"], gaps[sim]["clean"]
        print(f"     sim {sim}: phase B end -> phase C start, mean µs: clean {mean_or_nan(clean_gaps):7.0f} "
              f"(n {len(clean_gaps)}) | late {mean_or_nan(late_gaps):7.0f} (n {len(late_gaps)})")
    periods = sim_zero_periods(device_rows, steady_steps)
    late_periods = [period for step, period in periods.items() if step in late_steps]
    clean_periods = [period for step, period in periods.items() if step not in late_steps]
    mean_all = mean_or_nan(list(periods.values()))
    mean_clean = mean_or_nan(clean_periods)
    print(f"     step period mean µs: all {mean_all:.0f}, clean steps {mean_clean:.0f}, late steps "
          f"{mean_or_nan(late_periods):.0f}; fps if every step were clean {1e6 / mean_clean:.1f} vs "
          f"{1e6 / mean_all:.1f} ({(mean_clean / mean_all - 1) * 100:+.1f} % period)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for argument in sys.argv[1:]:
        directory = pathlib.Path(argument)
        link_rows, device_rows, steady_steps, by_step_and_link = load_trace(directory)
        print(f"== {directory.parent.parent.name}/{directory.name}: {len(steady_steps)} steady steps")
        guards = blocked_dest_guards(link_rows, steady_steps, by_step_and_link)
        report_latency(link_rows, device_rows, steady_steps, by_step_and_link)
        report_durations(guards)
        report_shared(guards)
        report_cost(device_rows, steady_steps, guards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
