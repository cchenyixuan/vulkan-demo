"""
e7_memcpy_hop.py — E7 batch report: the transport worker's host memcpy per link, by socket class (user item 8 of the
batch-1 review: is a link between socket-1 GPUs slower, and does that reach eta?).

The memcpy hop of link a -> b is worker_stamp -> worker_copy in steps_link.csv (after the waits, the upload guard and
the frame-stamp check; experiment/v7/utils/transport_v7.py, the 'memcpy' hop of step_trace_model.HOPS_HEAD). Both
stamps come from the worker thread's host clock, so no GPU clock map is involved; host_copy_bytes is what it copied
(count-aware). The worker of a link runs on its RECEIVER's NUMA cpulist (V7_WORKER_AFFINITY is indexed by the dest
device). Socket of a GPU: --socket-1-from (default 4: GPUs 4-7 on socket 1, both batch-1 nodes); the device map of
the run maps sims to GPUs (run_meta.json). Classes: socket 0 internal, socket 1 internal, cross-socket.

Per trace directory (steady steps: step >= warmup from run_meta.json, complete rows): per link p50 / p95 memcpy (us),
bytes p50 and GB/s at the p50, then per class the median of the link p50s and the GB/s range; and, for the
cross-socket links, the extra time over the socket-0 median against T_B (phase B of the receiver, p50) and the link's
share of steps where it set the worst r_chain.

Usage:
    python docs/cluster_v6/scripts/e7_memcpy_hop.py TRACE_DIR [TRACE_DIR ...] [--socket-1-from 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics


def quantile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else float("nan")


def link_class(sender_gpu: int, receiver_gpu: int, socket_1_from: int) -> str:
    sender_socket = 1 if sender_gpu >= socket_1_from else 0
    receiver_socket = 1 if receiver_gpu >= socket_1_from else 0
    if sender_socket != receiver_socket:
        return "cross-socket"
    return f"socket {sender_socket} internal"


def analyse(directory: pathlib.Path, socket_1_from: int) -> None:
    meta = json.loads((directory / "run_meta.json").read_text(encoding="utf-8"))
    warmup = int(meta.get("warmup", 0))
    device_map = [int(value) for value in meta.get("device_map") or meta.get("result", {}).get("device_map") or []]
    with open(directory / "steps_link.csv", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["complete"] == "1" and int(row["step"]) >= warmup]
    with open(directory / "steps_device.csv", encoding="utf-8") as handle:
        device_rows = [row for row in csv.DictReader(handle) if row["complete"] == "1" and int(row["step"]) >= warmup]
    phase_b = {}
    for row in device_rows:
        if row["b_start"] and row["b_end"]:
            phase_b.setdefault(int(row["sim"]), []).append((int(row["b_end"]) - int(row["b_start"])) / 1e3)
    per_link = {}
    for row in rows:
        if not row["worker_stamp"] or not row["worker_copy"]:
            continue
        elapsed_ns = int(row["worker_copy"]) - int(row["worker_stamp"])
        entry = per_link.setdefault(row["link"], {"sender": int(row["sender"]), "receiver": int(row["receiver"]),
                                                   "us": [], "bytes": []})
        entry["us"].append(elapsed_ns / 1e3)
        entry["bytes"].append(int(row["host_copy_bytes"] or 0))
    print(f"== {directory.parent.parent.name}/{directory.name}: steady steps from {warmup}, device map {device_map}")
    classes = {}
    for link, entry in sorted(per_link.items(), key=lambda item: (item[1]["sender"], item[1]["receiver"])):
        sender_gpu = device_map[entry["sender"]] if device_map else entry["sender"]
        receiver_gpu = device_map[entry["receiver"]] if device_map else entry["receiver"]
        kind = link_class(sender_gpu, receiver_gpu, socket_1_from)
        p50, p95 = quantile(entry["us"], 0.5), quantile(entry["us"], 0.95)
        bytes_p50 = quantile(entry["bytes"], 0.5)
        rate = bytes_p50 / (p50 * 1e3) if p50 > 0 else float("nan")
        receiver_phase_b = statistics.median(phase_b.get(entry["receiver"], [float("nan")]))
        classes.setdefault(kind, []).append((link, p50, rate, bytes_p50, receiver_phase_b))
        print(f"   {link:9s} GPU {sender_gpu}->{receiver_gpu} {kind:18s} memcpy p50 {p50:7.0f} p95 {p95:7.0f} us, "
              f"{bytes_p50 / 1e6:6.3f} MB, {rate:5.1f} GB/s at p50; receiver T_B p50 {receiver_phase_b:7.0f} us")
    socket_0 = [item[1] for item in classes.get("socket 0 internal", [])]
    socket_0_median = statistics.median(socket_0) if socket_0 else float("nan")
    for kind in ("socket 0 internal", "socket 1 internal", "cross-socket"):
        items = classes.get(kind, [])
        if not items:
            continue
        median_p50 = statistics.median(item[1] for item in items)
        rates = [item[2] for item in items]
        line = (f"   {kind:18s}: {len(items)} links, median of link p50 {median_p50:6.0f} us "
                f"({median_p50 / socket_0_median:4.2f} x socket 0), GB/s {min(rates):5.1f}-{max(rates):5.1f}")
        if kind == "cross-socket":
            extra = [item[1] - socket_0_median for item in items]
            shares = [(item[1] - socket_0_median) / item[4] for item in items if item[4] == item[4] and item[4] > 0]
            line += (f"; extra over socket 0 {min(extra):+.0f}..{max(extra):+.0f} us = "
                     f"{100 * min(shares):.2f}..{100 * max(shares):.2f} % of the receiver's T_B")
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--socket-1-from", type=int, default=4)
    args = parser.parse_args()
    for directory in args.directories:
        analyse(pathlib.Path(directory), args.socket_1_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
