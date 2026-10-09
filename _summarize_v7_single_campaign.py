"""
_summarize_v7_single_campaign.py — tables of measurement A of docs/seam_audit/v7_single.md from a
_run_single_baseline_campaign.py directory (results.jsonl + telemetry.csv).

Builds: S = v7_single, V = v7 (single mode), Vx = v7x (V7_CONCURRENT_BUFFERS=0), C = v7_chain (K = 1,
depth 2). For every size: fps and million particle-steps per second (mean ± std over trials), the
trial-wise ratios V/S, Vx/S (at each depth) and C/S (C at depth 2 against S at depth 2), the depth-1
loss of every build (fps at depth 1 / fps at depth 2 - 1, trial-wise), and the covariates (SM clock,
power, max temperature inside each window). Writes <dir>/summary.md, summary.csv and summary.json; the
JSON also carries the numbers the paper quotes (throughput range of S, depth-1 loss at 1M and 32M,
C/S per size).

    .venv/Scripts/python.exe _summarize_v7_single_campaign.py --dir logs/v7_single/throughput
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

_REPOSITORY = pathlib.Path(__file__).resolve().parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from _run_single_baseline_campaign import SIZES  # noqa: E402
from _summarize_single_baseline import covariates, load_results, load_telemetry, mean_std  # noqa: E402

BUILDS = {"v7_single": "S", "v7": "V", "v7x": "Vx", "v7_chain": "C"}
CONFIGS = [("v7_single", 1), ("v7_single", 2), ("v7", 1), ("v7", 2), ("v7x", 1), ("v7x", 2), ("v7_chain", 2)]
RATIOS = [("V/S d1", ("v7", 1), ("v7_single", 1)), ("V/S d2", ("v7", 2), ("v7_single", 2)),
          ("Vx/S d1", ("v7x", 1), ("v7_single", 1)), ("Vx/S d2", ("v7x", 2), ("v7_single", 2)),
          ("C/S", ("v7_chain", 2), ("v7_single", 2)), ("Vx/V d1", ("v7x", 1), ("v7", 1)),
          ("Vx/V d2", ("v7x", 2), ("v7", 2))]
DEPTH_LOSS = [("v7_single", "S"), ("v7", "V"), ("v7x", "Vx")]


def aggregate(ok: list[dict], samples) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for record in ok:
        groups.setdefault((record["size"], record["solver"], record["in_flight"]), []).append(record)
    rows = {}
    for key, records in groups.items():
        fps_values = [record["fps"] for record in records]
        mean, std = mean_std(fps_values)
        cov = [covariates(samples, record["epoch_start"], record["epoch_end"]) for record in records]

        def cov_mean(field):
            values = [item[field] for item in cov if item[field] is not None]
            return statistics.fmean(values) if values else None
        rows[key] = {"size": key[0], "solver": key[1], "in_flight": key[2], "particles": records[0]["particles"],
                     "n": len(records), "fps_mean": mean, "fps_std": std, "fps_min": min(fps_values),
                     "fps_max": max(fps_values),
                     "mparticle_steps_per_s": records[0]["particles"] * mean / 1e6,
                     "mparticle_steps_per_s_std": records[0]["particles"] * std / 1e6,
                     "sm_mhz": cov_mean("sm_mhz"), "power_w": cov_mean("power_w"), "temp_c": cov_mean("temp_c"),
                     "drift_max_abs": max(abs(record["drift"]) for record in records),
                     "overflow": any(record.get("overflow_all") for record in records),
                     "by_trial": {record["trial"]: record["fps"] for record in records}}
    return rows


def trial_ratio(rows, size, numerator, denominator) -> tuple[float, float, int]:
    first = rows.get((size,) + numerator)
    second = rows.get((size,) + denominator)
    if not first or not second:
        return float("nan"), float("nan"), 0
    common = sorted(set(first["by_trial"]) & set(second["by_trial"]))
    ratios = [first["by_trial"][trial] / second["by_trial"][trial] for trial in common]
    mean, std = mean_std(ratios)
    return mean, std, len(ratios)


def cell(ratio) -> str:
    return "n/a" if ratio[2] == 0 else f"{100 * ratio[0]:.2f} % ± {100 * ratio[1]:.2f} (n={ratio[2]})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs/v7_single/throughput")
    parser.add_argument("--gpu-index", type=int, default=1)
    arguments = parser.parse_args()
    out_dir = _REPOSITORY / arguments.dir
    ok, failed = load_results(out_dir / "results.jsonl")
    samples = load_telemetry(out_dir / "telemetry.csv", arguments.gpu_index)
    rows = aggregate(ok, samples)
    sizes = [tag for tag, _, _ in SIZES if any(key[0] == tag for key in rows)]

    lines = ["# v7_single measurement A: throughput and host overhead on one RTX 5090", "",
             "Builds: S = v7_single, V = v7 single-GPU mode, Vx = V with V7_CONCURRENT_BUFFERS=0, C = v7 chain K = 1 "
             "(depth 2). fps = measured steps / wall time (empty queue at both ends; C: the orchestrator's steady "
             "window); warmup 1000; defrag every 1000 steps inside the window; mean ± std over interleaved trials.", "",
             "## Throughput (fps)", "",
             "| size | particles | " + " | ".join(f"{BUILDS[solver]} d{depth}" for solver, depth in CONFIGS) + " |",
             "|" + "---|" * (2 + len(CONFIGS))]
    for size in sizes:
        particles = next(rows[key]["particles"] for key in rows if key[0] == size)
        cells = []
        for solver, depth in CONFIGS:
            row = rows.get((size, solver, depth))
            cells.append(f"{row['fps_mean']:.2f} ± {row['fps_std']:.2f} (n={row['n']})" if row else "n/a")
        lines.append(f"| {size} | {particles:,} | " + " | ".join(cells) + " |")
    lines += ["", "## Particle throughput (million particle-steps per second, mean ± std)", "",
              "| size | " + " | ".join(f"{BUILDS[solver]} d{depth}" for solver, depth in CONFIGS) + " |",
              "|" + "---|" * (1 + len(CONFIGS))]
    for size in sizes:
        cells = []
        for solver, depth in CONFIGS:
            row = rows.get((size, solver, depth))
            cells.append(f"{row['mparticle_steps_per_s']:.1f} ± {row['mparticle_steps_per_s_std']:.1f}" if row else "n/a")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines += ["", "## Ratios (trial-wise, mean ± std; a run against the run of the same trial)", "",
              "| size | " + " | ".join(name for name, _, _ in RATIOS) + " |", "|" + "---|" * (1 + len(RATIOS))]
    ratio_table = {}
    for size in sizes:
        cells = []
        for name, numerator, denominator in RATIOS:
            ratio = trial_ratio(rows, size, numerator, denominator)
            ratio_table[(size, name)] = ratio
            cells.append(cell(ratio))
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines += ["", "## Depth-1 loss (fps at 1 in flight / fps at 2 in flight - 1, trial-wise)", "",
              "| size | " + " | ".join(label for _, label in DEPTH_LOSS) + " |", "|" + "---|" * (1 + len(DEPTH_LOSS))]
    depth_table = {}
    for size in sizes:
        cells = []
        for solver, label in DEPTH_LOSS:
            ratio = trial_ratio(rows, size, (solver, 1), (solver, 2))
            depth_table[(size, label)] = ratio
            cells.append("n/a" if ratio[2] == 0 else f"{100 * (ratio[0] - 1):+.2f} % ± {100 * ratio[1]:.2f}")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines += ["", "## Covariates (nvidia-smi 1 Hz inside each window, mean over trials) and invariants", "",
              "| size | build | SM MHz | W | max °C | max |drift| | overflow |", "|---|---|---|---|---|---|---|"]
    for size in sizes:
        for solver, depth in CONFIGS:
            row = rows.get((size, solver, depth))
            if row:
                lines.append(f"| {size} | {BUILDS[solver]} d{depth} | {row['sm_mhz'] and f'{row['sm_mhz']:.0f}'} | "
                             f"{row['power_w'] and f'{row['power_w']:.0f}'} | {row['temp_c'] and f'{row['temp_c']:.0f}'} | "
                             f"{row['drift_max_abs']} | {'YES' if row['overflow'] else 'none'} |")
    if failed:
        lines += ["", f"## Failed runs: {len(failed)}", ""]
        for record in failed:
            lines.append(f"- {record.get('size')} {record.get('solver')} d{record.get('in_flight')} trial {record.get('trial')}: "
                         f"{record.get('status')} (rc={record.get('returncode')}, {record.get('run_log')})")
    # paper numbers
    s_rows = [rows[key] for key in rows if key[1] == "v7_single"]
    paper = {
        "S_fps_range": {f"{size}": {depth: rows[(size, 'v7_single', depth)]["fps_mean"]
                                     for depth in (1, 2) if (size, "v7_single", depth) in rows} for size in sizes},
        "S_mparticle_steps_per_s_range": [min(row["mparticle_steps_per_s"] for row in s_rows),
                                          max(row["mparticle_steps_per_s"] for row in s_rows)] if s_rows else None,
        "depth1_loss_S": {size: {"mean": depth_table[(size, "S")][0] - 1, "std": depth_table[(size, "S")][1]}
                          for size in sizes if depth_table[(size, "S")][2]},
        "C_over_S": {size: {"mean": ratio_table[(size, "C/S")][0], "std": ratio_table[(size, "C/S")][1]}
                     for size in sizes if ratio_table[(size, "C/S")][2]},
        "V_over_S_d2": {size: {"mean": ratio_table[(size, "V/S d2")][0], "std": ratio_table[(size, "V/S d2")][1]}
                        for size in sizes if ratio_table[(size, "V/S d2")][2]},
        "Vx_over_S_d2": {size: {"mean": ratio_table[(size, "Vx/S d2")][0], "std": ratio_table[(size, "Vx/S d2")][1]}
                         for size in sizes if ratio_table[(size, "Vx/S d2")][2]},
    }
    lines += ["", "## Paper numbers", "", "```json", json.dumps(paper, indent=1), "```"]
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    fields = ["size", "solver", "in_flight", "particles", "n", "fps_mean", "fps_std", "fps_min", "fps_max",
              "mparticle_steps_per_s", "mparticle_steps_per_s_std", "sm_mhz", "power_w", "temp_c", "drift_max_abs"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows, key=lambda item: ([tag for tag, _, _ in SIZES].index(item[0]), item[1], item[2])):
            writer.writerow({field: rows[key][field] for field in fields})
    (out_dir / "summary.json").write_text(json.dumps({
        "rows": {"/".join(map(str, key)): {field: value for field, value in row.items() if field != "by_trial"}
                 | {"by_trial": {str(trial): fps for trial, fps in row["by_trial"].items()}} for key, row in rows.items()},
        "ratios": {f"{size}|{name}": {"mean": ratio[0], "std": ratio[1], "n": ratio[2]}
                   for (size, name), ratio in ratio_table.items()},
        "depth1_loss": {f"{size}|{label}": {"mean": ratio[0] - 1, "std": ratio[1], "n": ratio[2]}
                        for (size, label), ratio in depth_table.items()},
        "paper": paper, "ok_runs": len(ok), "failed_runs": len(failed)}, indent=1), encoding="utf-8")
    print(f"[summary] ok runs={len(ok)} failed={len(failed)} telemetry samples={len(samples)} -> {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
