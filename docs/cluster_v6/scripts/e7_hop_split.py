"""
e7_hop_split.py — E7 batch 2, fourth supplement item 4 (user 2026-10-15): the transport chain of existing step traces
split into its three hops, readback, host copy and upload, p50 / p95 each. No new runs.

Per trace directory (experiment/v7/analysis/step_trace_model.load_run + link_metrics: steady steps = step >= warmup,
complete rows, every GPU time on the host clock), over every link and steady step (us):
  t_chain   upload_end - send_end minus the two waits on the receiver (its readback, its previous upload)
  readback  send_end -> readback_end                         (CHAIN_GROUPS: queue start + DMA + barrier, sender GPU)
  host      readback_end -> worker_copy minus the two waits  (worker: sees the readback, stamp check, memcpy)
  upload    worker_copy -> upload_end                         (host signal, queue start, DMA, receiver GPU)
and the cores inside them: readback DMA (readback_start -> readback_copy_end), memcpy (worker_stamp -> worker_copy),
upload DMA (upload_start -> upload_end); the host hop without its memcpy (host_rest = readback_to_worker:
readback_end -> worker_source_wait, + stamp_check: worker_upload_guard -> worker_stamp). The three hops add up to t_chain on every link-step; their p50s do not add
up to t_chain's p50. Per link the same quantiles (--per-link), and the bytes (host copy, DMA) per link.

Usage:
    python docs/cluster_v6/scripts/e7_hop_split.py NAME=TRACE_DIR [NAME=TRACE_DIR ...] [--json OUT] [--per-link]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from experiment.v7.analysis import step_trace_model  # noqa: E402

QUANTITIES = (("t_chain", "t_chain"), ("readback", "chain_readback"), ("host", "chain_host"),
              ("upload", "chain_upload"), ("readback_dma", "readback_dma"), ("memcpy", "memcpy"),
              ("host_rest", "host_rest"), ("readback_to_worker", "readback_to_worker"),
              ("stamp_check", "stamp_check"), ("upload_dma", "upload_dma"))


def quantiles(values) -> list:
    return [round(value, 1) for value in step_trace_model.quantiles(values, (50, 95))]


def split(trace: pathlib.Path) -> dict:
    run = step_trace_model.load_run(trace)
    metrics = step_trace_model.link_metrics(run)
    for values in metrics.values():         # the host hop without the memcpy = readback_to_worker + stamp_check
        values["host_rest"] = values["chain_host"] - values["memcpy"]
    pooled = {name: np.concatenate([values[key] for values in metrics.values()]) for name, key in QUANTITIES}
    out = {"trace": str(trace), "links": len(metrics), "link_steps": int(len(pooled["t_chain"])),
           "steps_per_link": sorted({int(len(values["step"])) for values in metrics.values()}),
           "pooled": {name: quantiles(values) for name, values in pooled.items()},
           "link_p50_range": {}, "per_link": {}}
    for name, key in QUANTITIES:
        medians = [float(np.nanmedian(values[key])) for values in metrics.values()]
        out["link_p50_range"][name] = [round(min(medians), 1), round(max(medians), 1)]
    for link, values in sorted(metrics.items(), key=lambda item: (item[1]["sender"], item[1]["receiver"])):
        out["per_link"][link] = {"sender": values["sender"], "receiver": values["receiver"],
                                 "host_copy_bytes": int(np.median(values["host_copy_bytes"])),
                                 "dma_bytes": int(np.median(values["dma_bytes"])),
                                 **{name: quantiles(values[key]) for name, key in QUANTITIES}}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="three-hop split of the transport chain (p50 / p95, us)")
    parser.add_argument("traces", nargs="+", metavar="NAME=TRACE_DIR")
    parser.add_argument("--json", default=None)
    parser.add_argument("--per-link", action="store_true")
    arguments = parser.parse_args()
    results = {}
    for item in arguments.traces:
        name, _, path = item.partition("=")
        results[name] = split(pathlib.Path(path))
    header = ("| trace | links x steps | " + " | ".join(quantity for quantity, _ in QUANTITIES) + " |\n"
              "|---|---|" + "---|" * len(QUANTITIES))
    print("pooled over every link and steady step, p50 / p95 (us)\n")
    print(header)
    for name, result in results.items():
        cells = " | ".join(f"{result['pooled'][quantity][0]:,.0f} / {result['pooled'][quantity][1]:,.0f}"
                           for quantity, _ in QUANTITIES)
        print(f"| {name} | {result['links']} x {'/'.join(map(str, result['steps_per_link']))} | {cells} |")
    print("\nrange of the per-link p50 (us)\n")
    print(header)
    for name, result in results.items():
        cells = " | ".join(f"{low:,.0f}-{high:,.0f}" for low, high in
                           (result["link_p50_range"][quantity] for quantity, _ in QUANTITIES))
        print(f"| {name} | {result['links']} | {cells} |")
    if arguments.per_link:
        for name, result in results.items():
            print(f"\n{name}: per link, p50 / p95 (us), bytes p50\n")
            print("| link | host copy B | DMA B | " + " | ".join(quantity for quantity, _ in QUANTITIES) + " |")
            print("|---|---|---|" + "---|" * len(QUANTITIES))
            for link, values in result["per_link"].items():
                cells = " | ".join(f"{values[quantity][0]:,.0f} / {values[quantity][1]:,.0f}" for quantity, _ in QUANTITIES)
                print(f"| {link} | {values['host_copy_bytes']:,} | {values['dma_bytes']:,} | {cells} |")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
