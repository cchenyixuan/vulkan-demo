"""
a100_summary.py — E30 A100 section: tables from one mirrored a100_smoke job.

  * scaling at equal weights: per-card K=1 fps (all cards at once), K=2 on
    cards 0,1 and K=4 on 0-3, eta = fps_K / (K * mean K=1 of the cards the run
    uses) and eta_min = fps_K / (K * min of them) — ONE trial each, illustrative
    only (the standard protocol needs >= 3 trials);
  * the SM clock / power / temperature of every card inside each run's window
    (nvidia-smi samples with utilization >= 90 %), as the covariate;
  * the pilot weight calibration (weights file): per round the weights, cuts,
    fluid per slab, busy per device, omega; end-slab vs middle-slab busy;
    weighted vs equal-weight fps.

fps comes from the STEADY line of each run log ("N steps in S s"), not the
one-decimal fps the line also prints.

Usage:
    python docs/cluster_v6/scripts/a100_summary.py --job-dir logs/n32h/.../06_e30_a100_1550154 \
        [--weights-file .../weights_2d16m_k4.json]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import re
import statistics

STEADY_PATTERN = re.compile(r"STEADY \(post-warmup (\d+)\): (\d+) steps in ([0-9.]+)s")
RUN_PATTERN = re.compile(r"^=== RUN (\S+) \((\d\d:\d\d:\d\d)\)")


def steady_fps(log_path: pathlib.Path):
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        match = STEADY_PATTERN.search(line)
        if match:
            return int(match.group(2)) / float(match.group(3))
    return None


def run_windows(out_path: pathlib.Path, rows: dict, day: str) -> dict:
    """label -> (start, end) datetimes from the '=== RUN label (HH:MM:SS)' lines + the row's wall seconds."""
    windows = {}
    for line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RUN_PATTERN.match(line)
        if match and match.group(1) in rows:
            start = datetime.datetime.strptime(f"{day} {match.group(2)}", "%Y/%m/%d %H:%M:%S")
            wall = rows[match.group(1)].get("wall_seconds") or 0.0
            windows[match.group(1)] = (start, start + datetime.timedelta(seconds=float(wall)))
    return windows


def telemetry_covariates(telemetry_path: pathlib.Path, window, cards) -> dict:
    samples = {card: [] for card in cards}
    with open(telemetry_path, encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for fields in reader:
            if len(fields) < 9:
                continue
            try:
                stamp = datetime.datetime.strptime(fields[0].strip()[:19], "%Y/%m/%d %H:%M:%S")
                card = int(fields[1])
                utilization = float(fields[5].strip().rstrip("%").strip())
                power = float(fields[6].strip().rstrip("W").strip())
                temperature = float(fields[7])
                clock = float(fields[8].strip().rstrip("MHz").strip())
            except ValueError:
                continue
            if card in samples and window[0] <= stamp <= window[1] and utilization >= 90.0:
                samples[card].append((clock, power, temperature))
    result = {}
    for card, values in samples.items():
        if values:
            result[card] = {"sm_clock_mhz": statistics.mean(value[0] for value in values),
                            "power_w": statistics.mean(value[1] for value in values),
                            "temperature_c": statistics.mean(value[2] for value in values),
                            "samples": len(values)}
    return result


def covariate_text(covariates: dict) -> str:
    return "; ".join(f"{card}: {value['sm_clock_mhz']:.0f} MHz, {value['power_w']:.0f} W, {value['temperature_c']:.0f} °C"
                     for card, value in sorted(covariates.items())) or "-"


def scaling_tables(job_dir: pathlib.Path, rows: dict, windows: dict) -> str:
    output = []
    for case_label, case_title in (("2d16m", "2-D 16M（2000 步，warmup 500）"), ("3d8m", "3-D 8M（1000 步，warmup 500）")):
        single = {}
        for card in range(4):
            label = f"k1_{case_label}_g{card}"
            if label in rows:
                single[card] = steady_fps(job_dir / f"{label}.log")
        output.append(f"**{case_title}**\n")
        output.append("| run | cards | steady fps | η | η_min | SM clock / power / temperature during the run |")
        output.append("|---|---|---|---|---|---|")
        for card, fps in sorted(single.items()):
            label = f"k1_{case_label}_g{card}"
            covariates = telemetry_covariates(job_dir / "telemetry.csv", windows[label], [card]) if label in windows else {}
            output.append(f"| K = 1 | {card} | {fps:.3f} | – | – | {covariate_text(covariates)} |")
        for slab_count, cards in ((2, [0, 1]), (4, [0, 1, 2, 3])):
            label = f"k{slab_count}_{case_label}"
            if label not in rows:
                continue
            fps = steady_fps(job_dir / f"{label}.log")
            references = [single[card] for card in cards if single.get(card)]
            if fps is None or len(references) != len(cards):
                output.append(f"| K = {slab_count} | {cards} | {fps} | – | – | missing reference |")
                continue
            eta = fps / (slab_count * statistics.mean(references))
            eta_min = fps / (slab_count * min(references))
            covariates = telemetry_covariates(job_dir / "telemetry.csv", windows[label], cards) if label in windows else {}
            output.append(f"| K = {slab_count} | {','.join(map(str, cards))} | {fps:.3f} | {100 * eta:.1f} % | "
                          f"{100 * eta_min:.1f} % | {covariate_text(covariates)} |")
        spread = [value for value in single.values() if value]
        if spread:
            output.append(f"\nK = 1 spread over the 4 cards: {min(spread):.3f}–{max(spread):.3f} fps "
                          f"({100 * (max(spread) / min(spread) - 1):.1f} %).\n")
    return "\n".join(output)


def calibration_text(weights_path: pathlib.Path, job_dir: pathlib.Path) -> str:
    record = json.loads(weights_path.read_text(encoding="utf-8"))
    lines = [f"weights file: device map {record['device_map']}, grid_nx {record['grid_nx']}, "
             f"settings {record['settings']}, {record['seconds']:.1f} s, {len(record['rounds'])} round(s)\n",
             "| round | weights | cuts | own columns | fluid per slab | busy per device µs | omega | next cuts |",
             "|---|---|---|---|---|---|---|---|"]
    for entry in record["rounds"]:
        lines.append(f"| {entry['round']} | {[round(value, 4) for value in entry['weights']]} | {entry['cuts']} | "
                     f"{entry['own_columns']} | {entry['fluid']} | {[round(value, 1) for value in entry['busy_us']]} | "
                     f"{[round(value, 4) for value in entry['omega']]} | {entry['next_cuts']} |")
    first = record["rounds"][0]
    busy = first["busy_us"]
    if len(busy) >= 3:
        end_mean = statistics.mean([busy[0], busy[-1]])
        middle_mean = statistics.mean(busy[1:-1])
        lines.append(f"\nround 1 (equal weights): end-slab busy {busy[0]:.1f} / {busy[-1]:.1f} µs, middle "
                     f"{', '.join(f'{value:.1f}' for value in busy[1:-1])} µs; end / middle = "
                     f"{end_mean / middle_mean:.4f}")
        per_fluid = [value / fluid for value, fluid in zip(busy, first["fluid"])]
        lines.append("round 1 busy per fluid particle (ns): " + ", ".join(f"{1000 * value:.3f}" for value in per_fluid))
    equal = steady_fps(job_dir / "k4_2d16m.log")
    weighted = steady_fps(job_dir / "k4_2d16m_weighted.log")
    if equal and weighted:
        lines.append(f"\nK = 4 2-D 16M: equal weights {equal:.3f} fps, calibrated weights {weighted:.3f} fps "
                     f"({100 * (weighted / equal - 1):+.2f} %, one trial each)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--out", default=None, help="the job's SLURM .out (default: the only *.out in --job-dir)")
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--day", default="2026/10/06")
    arguments = parser.parse_args()
    job_dir = pathlib.Path(arguments.job_dir)
    rows = {}
    with open(job_dir / "results.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["label"]] = row
    out_path = pathlib.Path(arguments.out) if arguments.out else next(job_dir.glob("*.out"))
    windows = run_windows(out_path, rows, arguments.day)
    print(scaling_tables(job_dir, rows, windows))
    if arguments.weights_file:
        print()
        print(calibration_text(pathlib.Path(arguments.weights_file), job_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
