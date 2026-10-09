"""
_summarize_kernel_breakdown.py — per-kernel cost table (Tab kernelcost) from
_run_kernel_breakdown.py results: 2-D and 3-D columns, density kernel and
scratch->primary copy listed separately, mean and p50 over independent runs,
acceptance checks, Markdown + LaTeX rows.

Usage:
    .venv/Scripts/python.exe _summarize_kernel_breakdown.py
        --results docs/single_gpu_baseline_20260924/kernel_breakdown_v2/results.jsonl
        --reference-2d-us 13920
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

STAGES = ["predict", "update_voxel", "correction", "density_kernel",
          "density_copy", "force"]
STAGE_LABEL = {
    "predict": "predict", "update_voxel": "update\\_voxel", "correction": "correction",
    "density_kernel": "density (kernel)", "density_copy": "density (scratch$\\to$primary copy)",
    "force": "force",
    # v7_single (measurement B of docs/seam_audit/v7_single.md)
    "correction_density": "correction + density (fused)", "deep_wall_marker": "deep-wall marker",
    "wall_extrapolate": "wall extrapolation (adami)",
}
DISPATCH = {
    "predict": "per particle", "update_voxel": "per voxel", "correction": "per particle",
    "density_kernel": "per particle", "density_copy": "buffer copy", "force": "per particle",
    "correction_density": "per particle", "deep_wall_marker": "per voxel (2 passes)",
    "wall_extrapolate": "per particle",
}


def load(path: pathlib.Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def stages_of(records: list[dict]) -> list[str]:
    """The stage list of the records (their own, v7_single's tick labels; the fixed STAGES for v0 / v5)."""
    return list(records[0].get("stages") or STAGES)


def summarize_dimension(records: list[dict]) -> dict:
    """Mean over runs of each run's mean / p50; std over runs of the frame total."""
    runs = len(records)
    particles = records[0]["alive"]
    out = {"runs": runs, "particles": particles, "drift_ok": all(r["drift"] == 0 for r in records),
           "cpu_fps": statistics.fmean(r["cpu_fps"] for r in records), "stages": {},
           "stage_names": stages_of(records), "switches": records[0].get("switches")}
    if all(r.get("candidates") for r in records):
        out["candidates_per_particle"] = statistics.fmean(r["candidates"]["candidates_per_particle"] for r in records)
        out["candidate_pairs"] = statistics.fmean(r["candidates"]["candidate_pairs"] for r in records)
    if all(r.get("production") for r in records):
        out["production_us_per_step"] = statistics.fmean(r["production"]["us_per_step"] for r in records)
        out["timed_us_per_step"] = statistics.fmean(r["timed_us_per_step"] for r in records)
        # the copy of the density stage includes the scratch->primary copy's own cost (v7_single: 'density_copy')
    for stage in stages_of(records) + ["gpu_frame"]:
        means = [r["mean_us"][stage] for r in records]
        p50s = [r["p50_us"][stage] for r in records]
        out["stages"][stage] = {
            "mean_us": statistics.fmean(means),
            "mean_std_us": statistics.stdev(means) if runs > 1 else 0.0,
            "p50_us": statistics.fmean(p50s),
            "p50_std_us": statistics.stdev(p50s) if runs > 1 else 0.0,
        }
    frame = out["stages"]["gpu_frame"]
    out["frame_std_pct"] = 100.0 * frame["mean_std_us"] / frame["mean_us"]
    return out


def ns_per_particle(us: float, particles: int) -> float:
    return us * 1000.0 / particles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--reference-2d-us", type=float, default=None,
                        help="earlier 2-D step total to reproduce within 1%")
    parser.add_argument("--out-md", type=str, default=None)
    args = parser.parse_args()

    records = load(pathlib.Path(args.results))
    by_dim = {}
    for record in records:
        by_dim.setdefault(record["dimension"], []).append(record)
    summary = {dim: summarize_dimension(recs) for dim, recs in sorted(by_dim.items())}
    dims = sorted(summary)

    lines = []
    header_particles = ", ".join(
        f"{dim}-D: {summary[dim]['particles']:,} particles ({summary[dim]['runs']} runs)" for dim in dims)
    solvers = sorted({record["solver"] for record in records})
    solver_label = {"v0": "V0 reference solver", "v5": "V5 single-GPU mode"}
    solver_text = " / ".join(solver_label.get(s, s) for s in solvers)
    gpu = records[0].get("gpu", "GPU")
    lines.append(f"# Per-kernel cost, {solver_text}, one {gpu} ({header_particles})\n")
    lines.append("Sync loop (1 frame in flight), warmup 1000, 2000 timed steps per run, defrag "
                 "frames excluded, BOTTOM_OF_PIPE timestamps. Values: mean over runs of the "
                 "per-run mean (p50 in parentheses); ns/particle uses the alive count incl. walls.\n")

    head = "| kernel | dispatch |" + "".join(
        f" {dim}-D µs/step | {dim}-D ns/particle | {dim}-D % |" for dim in dims)
    lines.append(head)
    lines.append("|" + "---|" * (2 + 3 * len(dims)))
    stage_names = []
    for dim in dims:
        for stage in summary[dim]["stage_names"]:
            if stage not in stage_names:
                stage_names.append(stage)
    for stage in stage_names + ["gpu_frame"]:
        label = "step total" if stage == "gpu_frame" else stage
        dispatch = "" if stage == "gpu_frame" else DISPATCH.get(stage, "")
        cells = []
        for dim in dims:
            s = summary[dim]["stages"].get(stage)
            if s is None:
                cells.append(" – | – | – |")
                continue
            total = summary[dim]["stages"]["gpu_frame"]["mean_us"]
            cells.append(f" {s['mean_us']:.1f} ({s['p50_us']:.1f}) | "
                         f"{ns_per_particle(s['mean_us'], summary[dim]['particles']):.2f} | "
                         f"{100 * s['mean_us'] / total:.1f}% |")
        lines.append(f"| {label} | {dispatch} |" + "".join(cells))

    lines.append("\n## Acceptance\n")
    for dim in dims:
        s = summary[dim]
        frame = s["stages"]["gpu_frame"]
        lines.append(f"- {dim}-D: drift = 0 on all runs: **{'yes' if s['drift_ok'] else 'NO'}**; "
                     f"step total {frame['mean_us']:.1f} ± {frame['mean_std_us']:.1f} µs over "
                     f"{s['runs']} runs (std {s['frame_std_pct']:.2f}%, limit 1%): "
                     f"**{'ok' if s['frame_std_pct'] <= 1.0 else 'FAIL'}**; CPU-side {s['cpu_fps']:.2f} fps.")
        if dim == 2 and args.reference_2d_us:
            delta = 100.0 * (frame["mean_us"] - args.reference_2d_us) / args.reference_2d_us
            lines.append(f"- 2-D step total vs reference {args.reference_2d_us:.0f} µs: {delta:+.2f}% "
                         f"(limit ±1%): **{'ok' if abs(delta) <= 1.0 else 'FAIL'}**")
        if s.get("production_us_per_step"):
            lines.append(f"- {dim}-D: timed GPU frame {frame['mean_us']:.1f} µs; the timed frames' wall time "
                         f"{s['timed_us_per_step']:.1f} µs/step (submit + fence, fresh fence per step: "
                         f"{s['timed_us_per_step'] - frame['mean_us']:+.1f} µs = the exposed host round trip); the "
                         f"production frames' wall time {s['production_us_per_step']:.1f} µs/step (interleaved, one "
                         f"persistent fence, no timestamps): {s['production_us_per_step'] - frame['mean_us']:+.1f} µs "
                         f"({100 * (s['production_us_per_step'] / frame['mean_us'] - 1):+.2f} %) over the GPU frame, "
                         f"{s['production_us_per_step'] - s['timed_us_per_step']:+.1f} µs against the timed frames.")
        if s.get("switches"):
            lines.append(f"- {dim}-D switches: {s['switches']}")
        if s.get("candidate_pairs"):
            lines.append(f"- {dim}-D candidates (particles listed in the 3^d voxels around each particle's voxel, "
                         f"from the end-of-run lists): {s['candidate_pairs']:,.0f} pairs per step = "
                         f"{s['candidates_per_particle']:.1f} per particle; ps per candidate of the neighbour "
                         f"kernels: " + ", ".join(
                             f"{stage} {1e6 * s['stages'][stage]['mean_us'] / s['candidate_pairs']:.2f}"
                             for stage in s["stage_names"]
                             if stage in ("correction", "density_kernel", "correction_density", "force",
                                          "wall_extrapolate") and stage in s["stages"]))

    lines.append("\n## LaTeX rows\n")
    lines.append("```latex")
    lines.append("% columns: kernel & dispatch & " + " & ".join(
        f"{dim}-D $\\mu$s/step & {dim}-D ns/particle & {dim}-D \\%" for dim in dims) + " \\\\")
    lines.append("% " + "; ".join(f"{dim}-D: {summary[dim]['particles']:,} particles" for dim in dims))
    for stage in stage_names + ["gpu_frame"]:
        label = "\\textbf{step total}" if stage == "gpu_frame" else STAGE_LABEL.get(stage, stage)
        dispatch = "" if stage == "gpu_frame" else DISPATCH.get(stage, "")
        cells = []
        for dim in dims:
            s = summary[dim]["stages"].get(stage)
            if s is None:
                cells.append("-- & -- & --")
                continue
            total = summary[dim]["stages"]["gpu_frame"]["mean_us"]
            cells.append(f"{s['mean_us']:.0f} & {ns_per_particle(s['mean_us'], summary[dim]['particles']):.2f} & "
                         f"{100 * s['mean_us'] / total:.1f}")
        if stage == "gpu_frame":
            lines.append("\\midrule")
        lines.append(f"{label} & {dispatch} & " + " & ".join(cells) + " \\\\")
    lines.append("```")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out_md:
        pathlib.Path(args.out_md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
