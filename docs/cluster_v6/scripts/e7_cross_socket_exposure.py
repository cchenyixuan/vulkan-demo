"""
e7_cross_socket_exposure.py — E7 item 8 follow-up: how much of the cross-socket link's extra memcpy time (s3 <-> s4, K = 8) reaches the receiver's
phase C start, per step (counterfactual on the step trace).

For receiver r of the cross link: its two inbound uploads run in order on one transfer queue (leading direction
first). Counterfactual: the cross link's memcpy takes the trace's socket-0 median instead of its own time (delta =
own memcpy - socket-0 median, floored at 0), so its worker signals delta earlier. Its upload then starts at
max(actual start - delta, the end of the upload ahead of it on the queue) and the upload behind it (if any) at
max(its unconstrained start, the new end of the cross upload), with the unconstrained start = that link's worker
signal + the trace's median signal -> upload start latency of first-in-queue uploads. The phase C gate is
max(b_end, both upload ends); the delay owed to the cross link = actual gate - counterfactual gate.

Usage:
    python docs/cluster_v6/scripts/e7_cross_socket_exposure.py TRACE_DIR [TRACE_DIR ...] (K = 8 traces; the
    socket of a GPU: GPUs 4-7 on socket 1, both batch-1 nodes)
"""
import csv
import json
import pathlib
import statistics
import sys

SOCKET_1_FROM = 4


def analyse(directory: pathlib.Path) -> None:
    meta = json.loads((directory / "run_meta.json").read_text(encoding="utf-8"))
    warmup = int(meta.get("warmup", 0))
    device_map = [int(value) for value in (meta.get("device_map") or [])]
    rows = []
    with open(directory / "steps_link.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["complete"] == "1" and int(row["step"]) >= warmup and row["worker_stamp"] and row["worker_copy"]:
                rows.append(row)
    def socket(sim: int) -> int:
        return 1 if device_map[sim] >= SOCKET_1_FROM else 0

    memcpy_socket_0 = [int(row["worker_copy"]) - int(row["worker_stamp"]) for row in rows
                       if socket(int(row["sender"])) == 0 and socket(int(row["receiver"])) == 0]
    median_socket_0 = statistics.median(memcpy_socket_0)
    # signal -> upload start latency of uploads that are first on their queue (leading direction)
    latency = statistics.median(int(row["upload_start"]) - int(row["worker_dest_signal"]) for row in rows
                                if row["receiver_direction"] == "leading" and row["worker_dest_signal"])
    by_step = {}
    for row in rows:
        by_step.setdefault((int(row["step"]), int(row["receiver"])), {})[row["receiver_direction"]] = row
    periods = []
    device_rows = {}
    with open(directory / "steps_device.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["complete"] == "1" and int(row["step"]) >= warmup and row["a_start"]:
                device_rows.setdefault(int(row["sim"]), {})[int(row["step"])] = int(row["a_start"])
    for sim, starts in device_rows.items():
        steps = sorted(starts)
        periods.extend(starts[second] - starts[first] for first, second in zip(steps, steps[1:])
                       if second == first + 1)
    period_us = statistics.median(periods) / 1e3
    print(f"== {directory.parent.parent.name}/{directory.name}: socket-0 memcpy median {median_socket_0 / 1e3:.0f} us, "
          f"signal->upload latency {latency / 1e3:.1f} us, step period p50 {period_us:.0f} us")
    for receiver in range(len(device_map)):
        directions = {}
        for direction, sender in (("leading", receiver - 1), ("trailing", receiver + 1)):
            if 0 <= sender < len(device_map) and socket(sender) != socket(receiver):
                directions[direction] = sender
        if not directions:
            continue
        cross_direction = next(iter(directions))
        delays, deltas = [], []
        for (step, sim), links in by_step.items():
            if sim != receiver or cross_direction not in links:
                continue
            cross = links[cross_direction]
            delta = max(0, (int(cross["worker_copy"]) - int(cross["worker_stamp"])) - median_socket_0)
            deltas.append(delta)
            b_end = int(cross["receiver_b_end"])
            ends = {direction: int(row["upload_end"]) for direction, row in links.items()}
            gate = max([b_end] + list(ends.values()))
            new_ends = dict(ends)
            order = [direction for direction in ("leading", "trailing") if direction in links]
            previous_end = None
            for direction in order:
                row = links[direction]
                start = int(row["upload_start"])
                duration = int(row["upload_end"]) - start
                if direction == cross_direction:
                    new_start = start - delta
                else:
                    new_start = int(row["worker_dest_signal"]) + latency if row["worker_dest_signal"] else start
                    new_start = min(start, new_start)
                if previous_end is not None:
                    new_start = max(new_start, previous_end)
                new_ends[direction] = new_start + duration
                previous_end = new_ends[direction]
            new_gate = max([b_end] + list(new_ends.values()))
            delays.append(max(0, gate - new_gate))
        if not delays:
            continue
        mean_delay = statistics.mean(delays) / 1e3
        print(f"   receiver s{receiver} (cross link from s{directions[cross_direction]}, {cross_direction}, "
              f"{'first' if cross_direction == 'leading' else 'second'} on its queue): extra memcpy p50 "
              f"{statistics.median(deltas) / 1e3:.0f} us; phase C start later because of it: mean {mean_delay:.1f} us, "
              f"p50 {statistics.median(delays) / 1e3:.1f}, p90 {sorted(delays)[int(0.9 * len(delays))] / 1e3:.1f} us "
              f"= {100 * mean_delay / period_us:.2f} % of the step period")


def main() -> int:
    for path in sys.argv[1:]:
        analyse(pathlib.Path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
