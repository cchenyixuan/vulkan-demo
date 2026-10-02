"""
single_step_tables.py - markdown tables for docs/seam_audit/v6_single_step.md
from single_step.py analysis files (analysis_<case>.json and, for the
long-horizon pass, analysis_<case>_long.json).

Usage:
  .venv/Scripts/python.exe -m experiment.seam_audit.single_step_tables \\
      --root logs/seam_audit/single_step --out <markdown file>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

CASES = (("cavity2d_1m", "2-D 1M"), ("cavity2d_4m", "2-D 4M"), ("cavity3d_1m", "3-D 1M"))
SNAPSHOTS = ("300", "2000")
STEP1_RUNS = (("k1_b", "K=1 重启 b(同序)"), ("original", "原 run 续走(自检)"),
              ("keep0_layers1_t1", "(0,1) = v5"), ("keep1_layers1_t1", "(1,1) t1"),
              ("keep1_layers1_t2", "(1,1) t2"), ("keep1_layers2_t1", "(1,2) t1"),
              ("keep1_layers2_t2", "(1,2) t2"))
STEP1_QUANTITIES = (("acceleration", "a"), ("shift", "δr"), ("density", "ρⁿ⁺¹"),
                    ("pressure", "pⁿ⁺¹"), ("kernel_sum", "Σ V W"), ("correction_inverse", "L"))


def load(root: pathlib.Path, name: str):
    path = root / f"analysis_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ratio_cell(entry: dict) -> str:
    if entry["n"] == 0:
        return "–"
    if entry["rms"] == 0 and entry["noise_rms"] == 0:
        return "≡"
    value = entry["ratio"]
    if math.isinf(value):
        return "∞(噪声 ≡ 0)"
    return f"{value:.3g}"


def median_cell(entry: dict) -> str:
    if entry["n"] == 0:
        return "–"
    if entry["median"] == 0 and entry["noise_median"] == 0:
        return "≡"
    value = entry["median_ratio"]
    if math.isinf(value):
        return "∞"
    return f"{value:.3g}"


def step1_table(analysis: dict, snapshot: str) -> list:
    step = analysis["snapshots"][snapshot]["steps"]["1"]
    lines = [f"| 运行(N = {snapshot},k = 1,column 0 / column 1 / 最远列) | "
             + " | ".join(label for _, label in STEP1_QUANTITIES) + " |",
             "|---|" + "---|" * len(STEP1_QUANTITIES)]
    for run, label in STEP1_RUNS:
        data = step["runs"].get(run)
        if not data:
            continue
        cells = []
        for quantity, _ in STEP1_QUANTITIES:
            if quantity not in data:
                cells.append("–")
                continue
            bins = data[quantity]["clean"]
            cells.append(" / ".join(ratio_cell(bins[index]) for index in (0, 1, len(bins) - 1)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def reconstruction_lines(analysis: dict, snapshot: str, title: str) -> list:
    step = analysis["snapshots"][snapshot]["steps"]["1"]
    reconstruction = step.get("stale_ghost_reconstruction")
    if not reconstruction:
        return []
    statistics = reconstruction["statistics"]
    run = reconstruction["runs"].get("keep1_layers1_t1", {})
    run_two = reconstruction["runs"].get("keep1_layers2_t1", {})
    clean = step["runs"]["keep1_layers1_t1"]["acceleration"]["clean"][0]
    return [f"| {title} | {snapshot} | {statistics['self_particles']:,} | "
            f"{statistics['ghost_pairs_per_particle']:.1f} | "
            f"{statistics.get('ghost_density_changed_fraction', float('nan')):.3f} | "
            f"{statistics['ghost_pressure_change_rms']:.3g} / {statistics['ghost_pressure_rms']:.3g} | "
            f"{reconstruction['reference_acceleration_rms']:.3g} | "
            f"{run['acceleration']['measured_rms']:.3g} | {reconstruction['predicted_acceleration_rms']:.3g} | "
            f"{run['acceleration']['residual_rms']:.2g} | {reconstruction['noise_acceleration_rms']:.2g} | "
            f"{run['acceleration']['correlation']:.9f} | "
            f"{clean['relative_rms']:.2e} / {clean['relative_median']:.1e} | "
            f"{run_two['acceleration']['measured_rms']:.2g} | "
            f"{reconstruction['predicted_shift_rms']:.1e} / {reconstruction['noise_shift_rms']:.1e} |"]


def horizon_rows(analysis: dict, snapshot: str, label: str) -> list:
    rows = []
    steps = analysis["snapshots"][snapshot]["steps"]
    for step_key in sorted(steps, key=int):
        step = steps[step_key]
        if step.get("missing"):
            continue
        runs = step["runs"]
        cells = [label, step_key, str(step["crossing_particles"])]
        for run in ("keep1_layers1_t1", "keep1_layers2_t1", "keep0_layers1_t1"):
            if run not in runs:
                cells += ["–", "–"]
                continue
            bins = runs[run]["acceleration"]["clean"]
            cells.append(f"{ratio_cell(bins[0])} / {ratio_cell(bins[1])}")
            cells.append(f"{median_cell(bins[0])} / {median_cell(bins[1])}")
        noise = runs["keep1_layers1_t1"]["acceleration"]["clean"][0] if "keep1_layers1_t1" in runs else None
        if noise:
            cells.append(f"{noise['noise_rms']:.2g} / {noise['noise_median']:.2g}")
            cells.append(f"{noise['rms']:.2g} / {noise['median']:.2g}")
            density = runs["keep1_layers1_t1"]["density"]["clean"][0]
            cells.append(f"{density['differing_fraction']:.3f} / {density['noise_differing_fraction']:.3f}")
            pressure = runs["keep1_layers1_t1"]["pressure"]["clean"][0]
            cells.append(f"{pressure['relative_rms']:.1e}" if pressure["rms"] > 0 else "0")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def crossing_rows(analysis: dict, snapshot: str, label: str) -> list:
    rows = []
    steps = analysis["snapshots"][snapshot]["steps"]
    for step_key in sorted(steps, key=int):
        step = steps[step_key]
        if step.get("missing") or not step.get("crossing_neighbourhood"):
            continue
        cells = [label, snapshot, step_key, str(step["crossing_particles"]),
                 str(step["crossing_neighbourhood"])]
        for run in ("keep0_layers1_t1", "keep1_layers1_t1", "keep1_layers2_t1"):
            data = step["runs"].get(run)
            if not data:
                cells.append("–")
                continue
            parts = []
            for quantity in ("acceleration", "kernel_sum", "shift"):
                bins = data[quantity]["crossing_neighbourhood"]
                parts.append(ratio_cell(bins[0]))
            cells.append(" / ".join(parts))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def profile_lines(analysis: dict, snapshot: str, title: str, columns=range(-4, 4)) -> list:
    """Acceleration rms ratio per signed column (t = column - cut) at k = 1."""
    step = analysis["snapshots"][snapshot]["steps"]["1"]
    profile = step.get("acceleration_profile", {})
    lines = []
    for run, label in (("keep1_layers1_t1", "(1,1) t1"), ("keep1_layers1_t2", "(1,1) t2"),
                       ("keep1_layers2_t1", "(1,2) t1"), ("original", "自检"),
                       ("k1_b", "K=1 重启 b")):
        rows = {row["column"]: row for row in profile.get(run, [])}
        cells = []
        for column in columns:
            row = rows.get(column)
            cells.append("–" if row is None else (f"{row['ratio']:.3g}"
                                                  if math.isfinite(row["ratio"]) else "∞"))
        lines.append(f"| {title} N={snapshot} | {label} | " + " | ".join(cells) + " |")
    return lines


def long_rows(analysis: dict, snapshot: str, label: str) -> list:
    """Long-horizon pass: column-0 acceleration (rms / median ratio) per
    configuration, (0,1) vs (1,1) shift and kernel_sum, absolute noise."""
    rows = []
    steps = analysis["snapshots"][snapshot]["steps"]
    for step_key in sorted(steps, key=int):
        step = steps[step_key]
        if step.get("missing"):
            continue
        runs = step["runs"]
        cells = [label, step_key, str(step["crossing_particles"])]
        for run in ("keep1_layers1_t1", "keep1_layers2_t1", "keep0_layers1_t1"):
            entry = runs[run]["acceleration"]["clean"][0]
            cells.append(f"{ratio_cell(entry)} / {median_cell(entry)}")
        for quantity in ("shift", "kernel_sum"):
            cells.append(" / ".join(ratio_cell(runs[run][quantity]["clean"][0])
                                    for run in ("keep0_layers1_t1", "keep1_layers1_t1")))
        noise = runs["keep1_layers1_t1"]["acceleration"]["clean"][0]
        cells.append(f"{noise['noise_rms']:.2g} / {noise['noise_median']:.2g}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="logs/seam_audit/single_step")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.root)
    lines = ["<!-- step1 -->"]
    for case, title in CASES:
        analysis = load(root, case)
        if not analysis:
            continue
        for snapshot in SNAPSHOTS:
            if snapshot in analysis["snapshots"]:
                lines += ["", f"**{title},N = {snapshot}**", ""] + step1_table(analysis, snapshot)
    lines += ["", "<!-- reconstruction -->", "",
              "| 算例 | N | column 0 粒子 | 每粒子 ghost 邻居 | ghost ρ 本步变化的比例 | ghost ΔP rms / P rms (Pa) | rms a (m/s²) | (1,1) 实测 rms Δa | 预测 rms Δa | 残差 rms | 噪声 rms | 相关系数 | (1,1) Δa/a rms / 中位 | (1,2) 实测 rms Δa | 预测 rms Δδr / 噪声 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case, title in CASES:
        analysis = load(root, case)
        if analysis:
            for snapshot in SNAPSHOTS:
                if snapshot in analysis["snapshots"]:
                    lines += reconstruction_lines(analysis, snapshot, title)
    lines += ["", "<!-- profile -->", "",
              "| 算例 | 运行 | " + " | ".join(f"t = {column}" for column in range(-4, 4)) + " |",
              "|---|---|" + "---|" * 8]
    for case, title in CASES:
        analysis = load(root, case)
        if analysis:
            for snapshot in SNAPSHOTS:
                if snapshot in analysis["snapshots"]:
                    lines += profile_lines(analysis, snapshot, title)
    lines += ["", "<!-- horizons -->", "",
              "| 算例 / N | k | 越界粒子 | (1,1) rms 比 c0 / c1 | (1,1) 中位比 c0 / c1 | (1,2) rms 比 | (1,2) 中位比 | (0,1) rms 比 | (0,1) 中位比 | 噪声 |Δa| rms / 中位 | (1,1) |Δa| rms / 中位 | ρ 有差的比例 (1,1) / 噪声 | (1,1) Δp rms / p rms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case, title in CASES:
        analysis = load(root, case)
        if not analysis:
            continue
        for snapshot in SNAPSHOTS:
            if snapshot in analysis["snapshots"]:
                lines += horizon_rows(analysis, snapshot, f"{title} N={snapshot}")
    lines += ["", "<!-- long -->", "",
              "| 算例 / N | k | 越界粒子 | (1,1) a:rms / 中位比 | (1,2) a | (0,1) a | δr rms 比 (0,1) / (1,1) | ΣVW rms 比 (0,1) / (1,1) | 噪声 |Δa| rms / 中位 (m/s²) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for case, title in CASES:
        analysis = load(root, case + "_long")
        if not analysis:
            continue
        for snapshot in SNAPSHOTS:
            if snapshot in analysis["snapshots"]:
                lines += long_rows(analysis, snapshot, f"{title} N={snapshot}")
    lines += ["", "<!-- crossings -->", "",
              "| 算例 | N | k | 越界粒子 | 邻域粒子 | (0,1) a / ΣVW / δr | (1,1) | (1,2) |",
              "|---|---|---|---|---|---|---|---|"]
    for case, title in CASES:
        for suffix in ("", "_long"):
            analysis = load(root, case + suffix)
            if analysis:
                for snapshot in SNAPSHOTS:
                    if snapshot in analysis["snapshots"]:
                        lines += crossing_rows(analysis, snapshot, title + ("(长)" if suffix else ""))
    pathlib.Path(arguments.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
