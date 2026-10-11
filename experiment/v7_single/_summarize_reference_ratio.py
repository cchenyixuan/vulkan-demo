"""
_summarize_reference_ratio.py — §5.1: the CSV, the figure and the three-line summary of the reference-ratio campaign
(_run_reference_ratio_campaign.py: v7 chain K = 1 against v7_single, two in flight, three alternating trials).

    .venv/Scripts/python.exe experiment/v7_single/_summarize_reference_ratio.py --runs logs/v7_single/reference_ratio \\
        --csv manuscripts/bench/reference_ratio.csv --figure manuscripts/fig/reference_ratio

CSV: one row per trial (solver, dim, N, trial, steps, mean_step_ms, throughput_Mps), solver = chain | single.
Ratio per case: chain / single of the MEAN throughput over the trials, in %; the error bar is the min and max of the
trial-wise ratios (trial i of chain over trial i of single: the trials alternate, so pairs are neighbours in time).
Figure: ratio against N on a log axis, one series per dimension, a dashed line at 100 %, the style of
single_gpu_throughput (single-column width, fonts >= 7 pt). Summary: the worst ratio and its N, and the worst ratio
from 4M up.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

SOLVER_NAMES = {"v7_single": "single", "v7_chain": "chain"}
SIZE_ORDER = ["1m", "2m", "4m_gen", "6m", "8m", "10m", "14m", "16m", "32m", "3d_1m", "3d_8m"]
DIMENSION = {tag: (3 if tag.startswith("3d") else 2) for tag in SIZE_ORDER}


def load_records(runs: pathlib.Path) -> list[dict]:
    path = runs / "results.jsonl"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok" and record.get("in_flight") == 2 and record.get("solver") in SOLVER_NAMES:
                records.append(record)
    return records


def rows_of(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        particles = int(record["particles"])
        fps = float(record["fps"])
        rows.append({"solver": SOLVER_NAMES[record["solver"]], "dim": DIMENSION.get(record["size"], 2), "N": particles,
                     "size": record["size"], "trial": int(record["trial"]), "steps": int(record["measure"]),
                     "mean_step_ms": round(1000.0 / fps, 4), "throughput_Mps": round(fps * particles / 1e6, 3),
                     "fps": fps, "drift": record.get("drift"), "overflow": record.get("overflow_all") or record.get("overflow")})
    return rows


def ratios_of(rows: list[dict]) -> dict[str, dict]:
    by_case: dict[str, dict] = {}
    for row in rows:
        entry = by_case.setdefault(row["size"], {"dim": row["dim"], "N": row["N"], "single": {}, "chain": {}})
        entry[row["solver"]][row["trial"]] = row["throughput_Mps"]
    result = {}
    for size, entry in by_case.items():
        single, chain = entry["single"], entry["chain"]
        trials = sorted(set(single) & set(chain))
        if not trials:
            continue
        mean_single = statistics.fmean(single[t] for t in trials)
        mean_chain = statistics.fmean(chain[t] for t in trials)
        trial_ratios = [100.0 * chain[t] / single[t] for t in trials]
        result[size] = {"dim": entry["dim"], "N": entry["N"], "trials": trials, "ratio_mean": 100.0 * mean_chain / mean_single,
                        "ratio_min": min(trial_ratios), "ratio_max": max(trial_ratios), "trial_ratios": trial_ratios,
                        "single_Mps": mean_single, "chain_Mps": mean_chain,
                        "single_std": statistics.pstdev(single[t] for t in trials) if len(trials) > 1 else 0.0,
                        "chain_std": statistics.pstdev(chain[t] for t in trials) if len(trials) > 1 else 0.0}
    return result


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {tag: index for index, tag in enumerate(SIZE_ORDER)}
    rows = sorted(rows, key=lambda r: (r["dim"], r["N"], order.get(r["size"], 99), r["trial"], r["solver"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["solver", "dim", "N", "trial", "steps", "mean_step_ms", "throughput_Mps"])
        for row in rows:
            writer.writerow([row["solver"], row["dim"], row["N"], row["trial"], row["steps"], row["mean_step_ms"], row["throughput_Mps"]])


def size_label(n: int) -> str:
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def figure(ratios: dict[str, dict], out: pathlib.Path, width: float, height: float, font_size: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from _plot_single_baseline_paper import COLOR_V0, setup_style, tidy
    setup_style(font_size)
    fig, ax = plt.subplots(figsize=(width, height))
    styles = {2: dict(color=COLOR_V0, marker="o", linestyle="-", label="2-D cavity"),
              3: dict(color="#c0392b", marker="s", linestyle=(0, (4, 2)), label="3-D cavity")}
    ticks = {}
    for dimension, style in styles.items():
        entries = sorted((e for e in ratios.values() if e["dim"] == dimension), key=lambda e: e["N"])
        if not entries:
            continue
        x = [e["N"] for e in entries]
        y = [e["ratio_mean"] for e in entries]
        lower = [e["ratio_mean"] - e["ratio_min"] for e in entries]
        upper = [e["ratio_max"] - e["ratio_mean"] for e in entries]
        ax.errorbar(x, y, yerr=[lower, upper], linewidth=1.0, markersize=4, markeredgewidth=0.9, capsize=2.5,
                    elinewidth=0.7, markerfacecolor="white" if dimension == 3 else style["color"], **style)
        for e in entries:
            ticks.setdefault(e["N"], size_label(e["N"]))
    ax.axhline(100.0, color="black", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    labeled = sorted(n for n in ticks if size_label(n) in ("1M", "2M", "4M", "8M", "16M", "32M"))
    ax.set_xticks(labeled)
    ax.set_xticklabels([size_label(n) for n in labeled])
    ax.set_xticks([n for n in ticks if n not in labeled], minor=True)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    values = [v for e in ratios.values() for v in (e["ratio_min"], e["ratio_max"])]
    ax.set_ylim(min(values) - 0.4, max(max(values), 100.0) + 0.4)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("particles (fluid + wall)")
    ax.set_ylabel("chain $K=1$ / single-device throughput (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right", handlelength=2.6)
    tidy(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_lines(ratios: dict[str, dict]) -> list[str]:
    entries = sorted(ratios.values(), key=lambda e: (e["dim"], e["N"]))
    if not entries:
        return ["no complete case"]
    worst = min(entries, key=lambda e: e["ratio_mean"])
    large = [e for e in entries if e["N"] >= 4_000_000]
    worst_large = min(large, key=lambda e: e["ratio_mean"]) if large else None
    smallest = min(entries, key=lambda e: e["N"])
    where = "the smallest case" if worst is smallest else f"N = {worst['N']:,}"
    lines = [f"worst ratio chain/single: {worst['ratio_mean']:.2f} % (min {worst['ratio_min']:.2f}, max {worst['ratio_max']:.2f}) "
             f"at N = {worst['N']:,} ({worst['dim']}-D, {len(worst['trials'])} trials)",
             f"it occurs at {where}; ratio by N: "
             + ", ".join(f"{size_label(e['N'])}{' (3-D)' if e['dim'] == 3 else ''} {e['ratio_mean']:.2f}" for e in entries)]
    if worst_large:
        lines.append(f"worst ratio from 4M up: {worst_large['ratio_mean']:.2f} % (min {worst_large['ratio_min']:.2f}, max "
                     f"{worst_large['ratio_max']:.2f}) at N = {worst_large['N']:,} ({worst_large['dim']}-D)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="logs/v7_single/reference_ratio")
    parser.add_argument("--csv", default="manuscripts/bench/reference_ratio.csv")
    parser.add_argument("--figure", default="manuscripts/fig/reference_ratio")
    parser.add_argument("--summary", default="docs/seam_audit/v7_single/reference_ratio_summary.md")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--height", type=float, default=2.4)
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--dims", default="2,3", help="dimensions to include (the 2-D series alone: 2)")
    arguments = parser.parse_args()
    dims = {int(d) for d in arguments.dims.split(",") if d.strip()}
    records = [r for r in load_records(_REPOSITORY_ROOT / arguments.runs) if DIMENSION.get(r["size"], 2) in dims]
    rows = rows_of(records)
    ratios = ratios_of(rows)
    write_csv(rows, _REPOSITORY_ROOT / arguments.csv)
    figure(ratios, _REPOSITORY_ROOT / arguments.figure, arguments.width, arguments.height, arguments.font_size)
    lines = summary_lines(ratios)
    table = ["| case | dim | N | single (M particle-steps/s) | chain | chain/single % (mean) | min | max | trials |",
             "|---|---|---|---|---|---|---|---|---|"]
    for size, e in sorted(ratios.items(), key=lambda item: (item[1]["dim"], item[1]["N"])):
        table.append(f"| {size} | {e['dim']} | {e['N']:,} | {e['single_Mps']:.1f} ± {e['single_std']:.1f} | {e['chain_Mps']:.1f} ± {e['chain_std']:.1f} | "
                     f"{e['ratio_mean']:.2f} | {e['ratio_min']:.2f} | {e['ratio_max']:.2f} | {len(e['trials'])} |")
    summary_path = _REPOSITORY_ROOT / arguments.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("# Reference ratio (chain K = 1 / single-device), generated by _summarize_reference_ratio.py\n\n"
                            + "\n".join(table) + "\n\n" + "\n".join(f"- {line}" for line in lines) + "\n", encoding="utf-8")
    (_REPOSITORY_ROOT / arguments.summary).with_suffix(".json").write_text(json.dumps(ratios, indent=1), encoding="utf-8")
    print("\n".join(table))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
