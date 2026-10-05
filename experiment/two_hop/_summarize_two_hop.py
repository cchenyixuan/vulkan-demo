"""
_summarize_two_hop.py — summary tables for the two-hop campaign.

Reads one or more campaign directories (results.jsonl written by
_run_two_hop_campaign.py) and prints Markdown tables:

  1. fps per (case, transport): median and range over the trials, plus the
     two-hop / three-hop ratio of the medians and whether the two ranges
     overlap.
  2. mechanism per (case, transport), from the instrumented depth-1 run and
     the worker timestamps: b_to_c_gap, upload -> phase C slack, DMA
     durations, worker segments.
  3. correctness per (case, transport): drift, stamp errors, seam check.

Usage:
    .venv/Scripts/python.exe experiment/two_hop/_summarize_two_hop.py \\
        logs/two_hop_experiment/campaign_20260928 --output summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

_TRANSPORTS = ("three_hop", "two_hop")
# A frame counts as "exposed" when phase C started this long after phase B
# ended; the hidden-chain floor of b_to_c_gap is ~6 µs on this rig.
_EXPOSED_GAP_US = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", default=None,
                        help="also write the tables to this Markdown file")
    return parser.parse_args()


def load_results(directories: list) -> tuple[list, dict]:
    """(results, {id(result): campaign directory})."""
    results, directory_of = [], {}
    for directory in directories:
        path = pathlib.Path(directory) / "results.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                result = json.loads(line)
                results.append(result)
                directory_of[id(result)] = pathlib.Path(directory)
    return results, directory_of


def frame_gap_statistics(directory, result: dict):
    """(mean b_to_c_gap, percent of exposed frames) over the per-frame CSV
    of an instrumented run; (None, None) when there is none."""
    if directory is None or not result.get("instrumented"):
        return None, None
    path = (directory / "frames"
            / f"{result['case_label']}_{result['transport']}_instrumented.csv")
    if not path.exists():
        return None, None
    gaps = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            for column in ("s0_b_to_c_gap_us", "s1_b_to_c_gap_us"):
                if row.get(column):
                    gaps.append(float(row[column]))
    if not gaps:
        return None, None
    exposed = sum(1 for gap in gaps if gap > _EXPOSED_GAP_US)
    return statistics.fmean(gaps), exposed / len(gaps) * 100.0


def median_of_workers(result: dict, segment: str):
    """Median over the two workers of the per-worker p50."""
    values = [segments[segment]["p50"]
              for segments in result.get("worker_segments_us", {}).values()
              if segment in segments]
    return statistics.median(values) if values else None


def median_of_sims(result: dict, metric: str):
    values = [sim[metric]["p50"] for sim in result.get("gpu", [])
              if metric in sim]
    return statistics.median(values) if values else None


def format_value(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    results, directory_of = load_results(args.directories)
    case_labels = list(dict.fromkeys(result["case_label"] for result in results))
    lines: list = []

    # ---- Table 1: fps -------------------------------------------------------
    lines += [
        "### fps (pipelined depth 2, no GPU timers)",
        "",
        "| case | particles | transport | trials | fps median | fps min–max | "
        "two/three (medians) | ranges overlap |",
        "|---|---:|---|---:|---:|---|---:|---|",
    ]
    for case_label in case_labels:
        per_transport = {}
        for transport in _TRANSPORTS:
            per_transport[transport] = [
                result for result in results
                if result["case_label"] == case_label
                and result["transport"] == transport
                and not result["instrumented"]
                and result.get("steady_fps")]
        medians = {
            transport: statistics.median(
                [result["steady_fps"] for result in trials])
            for transport, trials in per_transport.items() if trials}
        ranges = {
            transport: (min(result["steady_fps"] for result in trials),
                        max(result["steady_fps"] for result in trials))
            for transport, trials in per_transport.items() if trials}
        for transport in _TRANSPORTS:
            trials = per_transport[transport]
            if not trials:
                continue
            ratio = overlap = ""
            if transport == "two_hop" and len(medians) == 2:
                ratio = f"{medians['two_hop'] / medians['three_hop'] * 100:.2f}%"
                overlap = ("yes" if (ranges["two_hop"][0] <= ranges["three_hop"][1]
                                     and ranges["three_hop"][0] <= ranges["two_hop"][1])
                           else "no")
            lines.append(
                f"| {case_label} | {trials[0]['particles']:,} | {transport} | "
                f"{len(trials)} | {medians[transport]:.1f} | "
                f"{ranges[transport][0]:.1f}–{ranges[transport][1]:.1f} | "
                f"{ratio} | {overlap} |")

    # ---- Table 2: mechanism -------------------------------------------------
    lines += [
        "",
        "### mechanism (µs)",
        "",
        "GPU columns: instrumented depth-1 run, median over the two GPUs of "
        "the per-GPU p50; `gap mean` and `exposed frames` are over all "
        "post-warmup frames of both GPUs (exposed = b_to_c_gap > "
        f"{_EXPOSED_GAP_US:.0f} µs). Worker columns: pipelined depth-2 "
        "trials, median over trials and the two workers.",
        "",
        "| case | transport | staging KB/dir | phase B | readback sched gap | "
        "readback DMA | worker copy | worker signal | upload DMA | "
        "b_to_c_gap p50 | b_to_c_gap mean | exposed frames | "
        "upload→C slack p50 | worker upload wait | consumed signal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_label in case_labels:
        for transport in _TRANSPORTS:
            selected = [result for result in results
                        if result["case_label"] == case_label
                        and result["transport"] == transport]
            instrumented = [r for r in selected if r["instrumented"] and "gpu" in r]
            pipelined = [r for r in selected
                         if not r["instrumented"] and "worker_segments_us" in r]
            if not selected:
                continue
            gpu = instrumented[-1] if instrumented else {}

            def worker_median(segment: str):
                values = [median_of_workers(result, segment)
                          for result in pipelined]
                values = [value for value in values if value is not None]
                return statistics.median(values) if values else None

            staging = None
            for result in selected:
                sizes = [size for per_sim in result.get(
                    "staging_bytes_per_direction", [])
                    for size in per_sim.values()]
                if sizes:
                    staging = statistics.median(sizes) / 1024
                    break
            gap_mean, exposed_percent = frame_gap_statistics(
                directory_of.get(id(gpu)), gpu)
            lines.append(
                f"| {case_label} | {transport} | {format_value(staging, 0)} | "
                f"{format_value(median_of_sims(gpu, 'phase_b_us'))} | "
                f"{format_value(median_of_sims(gpu, 'readback_sched_gap_us'))} | "
                f"{format_value(median_of_sims(gpu, 'readback_dma_us'))} | "
                f"{format_value(worker_median('copy'))} | "
                f"{format_value(worker_median('signal'))} | "
                f"{format_value(median_of_sims(gpu, 'upload_dma_us'))} | "
                f"{format_value(median_of_sims(gpu, 'b_to_c_gap_us'))} | "
                f"{format_value(gap_mean)} | "
                f"{format_value(exposed_percent, 2)}% | "
                f"{format_value(median_of_sims(gpu, 'upload_to_c_gap_us'))} | "
                f"{format_value(worker_median('upload_wait'))} | "
                f"{format_value(worker_median('consumed_signal'))} |")

    # ---- Table 3: correctness -----------------------------------------------
    lines += [
        "",
        "### correctness (all runs of the group)",
        "",
        "| case | transport | runs | drift ≠ 0 | GPU stamp errors | "
        "host stamp errors | overwrite-during-upload | install drops | "
        "seam check failed | invalid runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_label in case_labels:
        for transport in _TRANSPORTS:
            selected = [result for result in results
                        if result["case_label"] == case_label
                        and result["transport"] == transport]
            if not selected:
                continue
            lines.append(
                f"| {case_label} | {transport} | {len(selected)} | "
                f"{sum(1 for r in selected if r.get('drift') not in (0,))} | "
                f"{sum(r.get('stamp_errors_gpu') or 0 for r in selected)} | "
                f"{sum(r.get('stamp_errors_host') or 0 for r in selected)} | "
                f"{sum(r.get('overwrite_errors_host') or 0 for r in selected)} | "
                f"{sum(r.get('install_tail_drops') or 0 for r in selected)} | "
                f"{sum(1 for r in selected if r.get('seam_ok') is False)} | "
                f"{sum(1 for r in selected if not r.get('valid'))} |")

    text = "\n".join(lines) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    if args.output:
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
