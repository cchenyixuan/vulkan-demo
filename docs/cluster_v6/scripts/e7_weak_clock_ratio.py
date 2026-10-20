"""e7_weak_clock_ratio.py (user 2026-10-20 night): for the weak-scaling points of the paper's weak figure (2M / 8M / 32M
per device, K = 2..8, the (job, point) the figure chose: scaling_figures.csv) the SM clock ratio of every trial,

    ratio = clock of the chain run / clock of the same trial's K = 1 reference runs,

where a run's clock is its loop-window SM clock median per GPU (parse_run_v6 telemetry_loop sm_clock_median; the chain
run: the mean over its K GPUs; the reference set: the mean over its K runs, one GPU each), as e7_batch2_tables computes
it (clock_ratio per trial), then the mean over the trials (min-max alongside). 8M K = 5 is job 1718639's recalibrated
arm (point 2d_n2840_k5_K5 of that job).

Also, for the 32M series, the per-card particle count of the chain (the calibrated cuts: the chain log's
"[partition_v7] chain N=K: ... particles ..." line of the first trial, every trial of a point shares the weights file)
against the reference block (cavity2d_n5640, 32,058,244 particles incl. walls, one card): N / K and the slowest /
fastest card, as percentages.

Usage:
    python docs/cluster_v6/scripts/e7_weak_clock_ratio.py --tables data/e7/batch2/tables_b2.json
        --figure-csv data/e7/batch2/figures/scaling_figures.csv --logs ../vulkan-demo/logs/n56 [--md OUT] [--json OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

PARTITION = re.compile(r"\[partition_v7\] chain N=(\d+): .* particles (\d{1,3}(?:,\d{3})*(?:, \d{1,3}(?:,\d{3})*)*)")
REFERENCE_PARTICLES = {"cavity2d_n1440": 2_137_444, "cavity2d_n2840": 8_191_044, "cavity2d_n5640": 32_058_244}
LOADS = (("cavity2d_n1440", "2M per device (2.1M)"), ("cavity2d_n2840", "8M per device (8.1M)"),
         ("cavity2d_n5640", "32M per device (32.0M)"))


def mirror_dir(logs: pathlib.Path, job: str) -> pathlib.Path:
    hits = sorted(logs.glob(f"*/*_{job}"))
    if not hits:
        raise SystemExit(f"no mirror of {job} under {logs}")
    return hits[-1]


def partition_particles(log: pathlib.Path) -> list:
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PARTITION.search(line)
        if match:
            return [int(value.replace(",", "")) for value in match.group(2).split(", ")]
    raise SystemExit(f"no partition line in {log}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True)
    parser.add_argument("--figure-csv", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    chosen = [row for row in csv.DictReader(open(arguments.figure_csv, encoding="utf-8", newline=""))
              if row["figure"] == "weak_scaling"]
    by_key = {(entry["job"], entry["point"]): entry for entry in tables["points"] if entry["arm"] == "C"}
    logs = pathlib.Path(arguments.logs)
    results = {}
    for row in chosen:
        entry = by_key[(row["job"], row["point"])]
        reference_name = row["reference"]
        ratios, clocks_k, clocks_1 = [], [], []
        for trial in entry["trials"]:
            reference = trial["references"].get(reference_name)
            if not reference or reference.get("clock_ratio") is None:
                continue
            ratios.append(reference["clock_ratio"])
            clocks_k.append(trial["sm_clock"])
            clocks_1.append(reference["sm_clock_1"])
        reference_case = reference_name.split(":", 1)[1]
        record = {"load": reference_case, "K": int(row["K"]), "point": row["point"], "job": row["job"], "host": row["host"],
                  "trials": len(ratios), "ratio_mean": statistics.fmean(ratios), "ratio_min": min(ratios),
                  "ratio_max": max(ratios), "ratios": ratios, "clock_k_mean": statistics.fmean(clocks_k),
                  "clock_1_mean": statistics.fmean(clocks_1)}
        if reference_case == "cavity2d_n5640":
            log = mirror_dir(logs, row["job"]) / f"{row['point']}_t1_C.log"
            cards = partition_particles(log)
            block = REFERENCE_PARTICLES[reference_case]
            record["cards"] = cards
            record["per_card_mean"] = sum(cards) / len(cards)
            record["diff_mean_pct"] = (record["per_card_mean"] / block - 1.0) * 100.0
            record["diff_min_pct"] = (min(cards) / block - 1.0) * 100.0
            record["diff_max_pct"] = (max(cards) / block - 1.0) * 100.0
            record["partition_log"] = str(log)
        results[f"{reference_case}:K{row['K']}"] = record
    lines = ["| per device | K = 2 | K = 3 | K = 4 | K = 5 | K = 6 | K = 7 | K = 8 |", "|---|---|---|---|---|---|---|---|"]
    for reference_case, label in LOADS:
        cells = []
        for slabs in range(2, 9):
            record = results.get(f"{reference_case}:K{slabs}")
            cells.append(f"{record['ratio_mean']:.3f} ({record['ratio_min']:.3f}–{record['ratio_max']:.3f}, n={record['trials']})"
                         if record else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| per device | K | point | job (node) | trials | chain SM MHz (mean of trials) | reference SM MHz | ratio per trial |",
              "|---|---|---|---|---|---|---|---|"]
    for reference_case, label in LOADS:
        for slabs in range(2, 9):
            record = results.get(f"{reference_case}:K{slabs}")
            if record:
                lines.append(f"| {label} | {slabs} | {record['point']} | {record['job']} ({record['host']}) | {record['trials']} | "
                             f"{record['clock_k_mean']:.0f} | {record['clock_1_mean']:.0f} | "
                             + ", ".join(f"{value:.4f}" for value in record["ratios"]) + " |")
    lines += ["", "32M series: chain per-card particles (calibrated cuts, incl. walls) against the reference block cavity2d_n5640 "
              f"({REFERENCE_PARTICLES['cavity2d_n5640']:,} on one card)", "",
              "| K | point | job | N / K | N / K − block | smallest card | largest card | cards |", "|---|---|---|---|---|---|---|---|"]
    for slabs in range(2, 9):
        record = results.get(f"cavity2d_n5640:K{slabs}")
        if record:
            lines.append(f"| {slabs} | {record['point']} | {record['job']} | {record['per_card_mean']:,.0f} | {record['diff_mean_pct']:+.2f} % | "
                         f"{min(record['cards']):,} ({record['diff_min_pct']:+.2f} %) | {max(record['cards']):,} ({record['diff_max_pct']:+.2f} %) | "
                         + ", ".join(f"{value:,}" for value in record["cards"]) + " |")
    text = "\n".join(lines)
    print(text)
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({
            "definition": "ratio = chain run loop-window SM clock (mean over its K GPUs of each GPU's median) / mean over the "
                          "same trial's K reference runs of their loop-window SM clock median; mean over trials",
            "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
