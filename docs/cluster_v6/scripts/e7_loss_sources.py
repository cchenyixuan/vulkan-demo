"""
e7_loss_sources.py — E7 batch 2, the data of the paper's section 6.5 (user 2026-10-18), from existing step traces and
the tables JSON of e7_batch2_tables.py; no new runs.

  1. phases   per sim, the medians over the steady steps (step >= warmup, the next step present) of T_A = a_end -
              a_start, T_B = b_end - b_start, T_C = c_end - c_start, the gaps A->B, B->C, C->next A and the period
              a_start(n + 1) - a_start(n) (step_trace_model.cycle_components); mean over the sims and min-max. The K = 1
              reference: the same medians of a K = 1 trace of the reference case (/ K for a strong point, as they are
              for a weak point) when one exists; for every point also the reference period from the untraced K = 1
              reference runs of the point: 1e6 / mean fps (with and without the defrag events), / K for a strong point.
  2. period   per point, every sim's steady periods pooled: p50, mean, p95 (+ p99, max), and the mean without the defrag
              event (the trace run's in-window defrag wall time, from its log, taken off the step that holds it); the
              per-sim p50 / p95 range; the steps longer than 2 x the pooled p50 (other than the defrag step).
  3. bytes    per link and step of a trace (all steps, warm-up included): the worker's recorded host copy bytes B against
              B = 32 (n0 + n1) + 44 n_mig + 16 Ny Nz + 16. The trace records B only, not n0, n1, n_mig (V7_POOL_PEAKS=0):
              per link-step, R = B - 16 Ny Nz - 16 must be >= 0 and 4 | R; 8 a + 11 b = R / 4 then has the solutions
              b = b0 + 8 j, a = a0 - 11 j (b0 = 3 R / 4 mod 8); a0, b0 are listed. Counts from the trace's voxel
              snapshot (inside_particle_count after the first drained defrag boundary >= warmup, own_column_particles):
              a_snapshot = the sender's two own columns at the seam; (R - 32 a_snapshot) / 44 at the steps around it;
              the run log's pool_health peak_departed_count per sim (most migrants one sim sent in one frame, both
              directions, the whole run) against the largest b0 of the sim's two outgoing links.
  4. figure   loss_sources.pdf: every K = 8 strong cavity point, eta (no defrag, mean, min-max bar) against r_own p50
              (log axis), horizontal bar r_own p50 -> p95; dashed line at r_own = 1; band eta = 1 +- 0.02; style and
              point selection of e7_scaling_figures.py (2d_n11320, reference = K = 2 pairs, with its other marker).

Usage:
    python docs/cluster_v6/scripts/e7_loss_sources.py --tables tables_b2.json --logs LOGS_N56 --out DIR
        [--exclude-job e7_b2T_1695731 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v7.analysis import step_trace_model  # noqa: E402
import e7_scaling_figures as style  # noqa: E402

# point name, job, kind, the trace directory (job dir, trace name), its run log, the reference K = 1 trace (or None);
# eta_job: the job of the point's timed trials and K = 1 references when another job ran its trace
JOB_DIRS = {"e7_b2A_1694327": "2026-10-12/02_e7_b2A_1694327", "e7_b2B_1694328": "2026-10-12/03_e7_b2B_1694328",
            "e7_b2S_1695411": "2026-10-13/01_e7_b2S_1695411", "e7_b2V_1703604": "2026-10-15/01_e7_b2V_1703604",
            "e7_b2Y_1704741": "2026-10-16/01_e7_b2Y_1704741"}
K1_TRACE = ("e7_b2A_1694327", "trace_2d_n8000_K1", "trace_2d_n8000_K1.log")
POINTS = (
    {"label": "2d_n4000 K8", "point": "2d_n4000_K8", "job": "e7_b2S_1695411", "kind": "strong", "trace": "2d_n4000_K8_C",
     "log": "2d_n4000_K8_trace_C.log", "k1_trace": None},
    {"label": "2d_n5640 K8", "point": "2d_n5640_K8", "job": "e7_b2V_1703604", "kind": "strong", "trace": "2d_n5640_K8_C",
     "log": "2d_n5640_K8_trace_C.log", "k1_trace": None},
    {"label": "2d_n1440_k8 K8", "point": "2d_n1440_k8_K8", "job": "e7_b2Y_1704741", "kind": "weak",
     "trace": "2d_n1440_k8_K8_C", "log": "2d_n1440_k8_K8_trace_C.log", "k1_trace": None,
     "eta_job": "e7_b2S_1695411"},        # b2Y ran only the trace of this point; its 5 timed trials are in b2S
    {"label": "2d_n8000 K8", "point": "x_2d_n8000_K8", "job": "e7_b2B_1694328", "kind": "strong",
     "trace": "x_2d_n8000_K8_C", "log": "x_2d_n8000_K8_trace_C.log", "k1_trace": K1_TRACE},
)
PHASES = ("a", "a_to_b", "b", "b_to_c", "c", "c_to_next_a", "period")
DEFRAG_LINE = re.compile(r"defrag_event call=\d+ f(\d+) .*?wall_ms=([\d.]+)")
NODEFRAG_LINE = re.compile(r"defrag_events .*steady_fps=([\d.]+) steady_fps_nodefrag=([\d.]+)")
GRID_LINE = re.compile(r"\[case_loader_v7\] grid: dim=\((\d+), (\d+), (\d+)\)")
SEAM_LINE = re.compile(r"replica regions 2 x ([\d,]+) \+ migrants ([\d,]+)")
POOL_HEALTH_LINE = re.compile(r"sim(\d+) \(dev\d+\) pool_health: .*peak_departed_count=(\d+)")


def stats(values) -> dict:
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def defrag_events(log: pathlib.Path, warmup: int, steps: int) -> list:
    """(frame, wall_ms) of every defrag event inside the steady window; the event at frame f sits in the period of
    step f - 1 (a_start(f - 1) -> a_start(f))."""
    events = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DEFRAG_LINE.search(line)
        if match and warmup < int(match.group(1)) < steps:
            events.append((int(match.group(1)), float(match.group(2))))
    return events


def trace_fps(log: pathlib.Path) -> list:
    """[steady fps, steady fps without the defrag events] of a trace run (its log's defrag_events line)."""
    match = NODEFRAG_LINE.search(log.read_text(encoding="utf-8", errors="replace"))
    return [float(match.group(1)), float(match.group(2))]


def periods(run: dict) -> dict:
    """sim -> (steps, periods in us) over the steady steps whose next step is present."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    out = {}
    for sim in sorted(set(device["sim"].astype(int))):
        rows = step_trace_model.select(device, (device["sim"] == sim) & (device["complete"] == 1))
        order = np.argsort(rows["step"])
        steps, a_start = rows["step"][order].astype(int), rows["a_start"][order]
        consecutive = steps[1:] == steps[:-1] + 1
        keep = consecutive & (steps[:-1] >= warmup)
        out[sim] = (steps[:-1][keep], (np.diff(a_start) / 1e3)[keep])
    return out


def by_step(run: dict) -> dict:
    """step -> [period of every sim] (us), steady steps."""
    out: dict = {}
    for steps, values in periods(run).values():
        for step, value in zip(steps, values):
            out.setdefault(int(step), []).append(float(value))
    return out


def period_summary(run: dict, events: list, divide: int = 1) -> dict:
    per_sim = periods(run)
    pooled_steps = np.concatenate([steps for steps, _ in per_sim.values()])
    pooled = np.concatenate([values for _, values in per_sim.values()]) / divide
    defrag_steps = {frame - 1 for frame, _ in events}
    defrag_ms = sum(wall for _, wall in events)
    sims = len(per_sim)
    p50 = float(np.percentile(pooled, 50))
    long_mask = (pooled > 2.0 * p50) & ~np.isin(pooled_steps, list(defrag_steps))
    long_steps = sorted({int(step) for step in pooled_steps[long_mask]})
    return {"sims": sims, "steps_per_sim": sorted({int(len(steps)) for steps, _ in per_sim.values()}),
            "p50": p50, "mean": float(np.mean(pooled)),
            "mean_without_defrag": float((pooled.sum() - sims * 1e3 * defrag_ms / divide) / len(pooled)),
            "p95": float(np.percentile(pooled, 95)), "p99": float(np.percentile(pooled, 99)),
            "max": float(np.max(pooled)), "defrag_events": events, "defrag_steps": sorted(defrag_steps),
            "per_sim_p50": [float(np.percentile(values, 50)) / divide for _, values in per_sim.values()],
            "per_sim_p95": [float(np.percentile(values, 95)) / divide for _, values in per_sim.values()],
            "per_sim_mean": [float(np.mean(values)) / divide for _, values in per_sim.values()],
            "long_steps_over_2x_p50": long_steps,
            "long_step_sims": {str(step): int(np.sum(long_mask & (pooled_steps == step))) for step in long_steps},
            "long_step_periods_us": {str(step): [round(float(value), 1) for value in pooled[long_mask & (pooled_steps == step)]]
                                     for step in long_steps}}


def phase_summary(run: dict, divide: int = 1) -> dict:
    components = step_trace_model.cycle_components(run)
    out = {"per_sim": components}
    for part in PHASES + ("period_mean",):
        out[part] = stats([entry[part] / divide for entry in components])
    return out


def reference_period(tables: dict, point: dict, kind: str) -> dict:
    job = point.get("eta_job", point["job"])
    found = [p for p in tables["points"] if p["job"] == job and p["point"] == point["point"] and p["arm"] == "C"
             and p["trials"]]
    assert len(found) == 1, (point, len(found))
    entry = found[0]
    (reference_name, reference), = entry["references"].items()
    runs = [r for r in tables["runs"] if r["job"] == job and r["point"] == point["point"]
            and r["role"] == "reference" and r["status"] == "pass"]
    slabs = entry["K"] if kind == "strong" else 1
    out = {"job": job, "reference": reference_name, "runs": len(runs), "divide": slabs, "trials": len(entry["trials"]),
           "K_fps": entry["fps"], "K_fps_nodefrag": entry["fps_nodefrag"], "eta_nodefrag": reference.get("eta_nodefrag")}
    for key in ("fps", "fps_nodefrag"):
        values = [r[key] for r in runs if r.get(key)]
        mean = statistics.mean(values)
        out["k1_" + key] = {"mean": mean, "min": min(values), "max": max(values), "n": len(values)}
        out["k1_period_" + key] = 1e6 / mean / slabs          # us, 1e6 / (slabs x mean fps_1)
        out["k1_period_" + key + "_runs"] = [1e6 / max(values) / slabs, 1e6 / min(values) / slabs]
    return out


def link_bytes(trace_dir: pathlib.Path, log: pathlib.Path, run: dict) -> dict:
    text = log.read_text(encoding="utf-8", errors="replace")
    nx, ny, nz = (int(value) for value in GRID_LINE.search(text).groups())
    replica_region, migrant_region = (int(value.replace(",", "")) for value in SEAM_LINE.search(text).groups())
    fixed = 16 * ny * nz + 16
    layout_bytes = 4 * 16 * replica_region + 44 * migrant_region + fixed
    snapshot = run["meta"]["slabs"]
    columns = [sim["own_column_particles"] for sim in snapshot["per_sim"]]
    rows = list(csv.DictReader(open(trace_dir / "steps_link.csv", newline="")))
    # the run's largest number of migrants one sim sent away in one frame (ghost_send's atomicMax, never reset)
    peak_departed = {int(sim): int(value) for sim, value in POOL_HEALTH_LINE.findall(text)}
    out = {"grid": [nx, ny, nz], "fixed_bytes": fixed, "replica_region": replica_region, "migrant_region": migrant_region,
           "staging_bytes_from_layout": layout_bytes, "snapshot_frame": snapshot["frame"],
           "peak_departed_per_sim": [peak_departed[sim] for sim in sorted(peak_departed)], "links": {}}
    by_link: dict = {}
    for row in rows:
        by_link.setdefault(row["link"], []).append(row)
    for name, link_rows in sorted(by_link.items(), key=lambda item: (int(item[1][0]["sender"]), int(item[1][0]["receiver"]))):
        sender, direction = int(link_rows[0]["sender"]), link_rows[0]["sender_direction"]
        seam_columns = columns[sender][-2:] if direction == "trailing" else columns[sender][:2]
        a_snapshot = int(sum(seam_columns))
        steps, values, residual_ok, a0, b0 = [], [], 0, [], []
        for row in link_rows:
            if row["host_copy_bytes"] == "":
                continue
            value = int(row["host_copy_bytes"])
            rest = value - fixed
            steps.append(int(row["step"]))
            values.append(value)
            if rest >= 0 and rest % 4 == 0:
                residual_ok += 1
                quarter = rest // 4
                b = (3 * quarter) % 8
                b0.append(b)
                a0.append((quarter - 11 * b) // 8)
        near = {}
        for step, value in zip(steps, values):
            if abs(step - snapshot["frame"]) <= 1:
                migrants = (value - fixed - 32 * a_snapshot) / 44
                near[str(step)] = {"bytes": value, "(B - fixed - 32 a_snapshot) / 44": round(migrants, 3)}
        out["links"][name] = {
            "sender": sender, "receiver": int(link_rows[0]["receiver"]), "sender_direction": direction,
            "rows": len(link_rows), "rows_with_bytes": len(values), "steps": [min(steps), max(steps)],
            "dma_bytes": sorted({int(row["dma_bytes"]) for row in link_rows}),
            "bytes_min": min(values), "bytes_max": max(values), "bytes_distinct": len(set(values)),
            "residual_nonnegative_multiple_of_4": residual_ok,
            "a0_range": [min(a0), max(a0)], "b0_counts": {str(b): b0.count(b) for b in sorted(set(b0))},
            "a_snapshot": a_snapshot, "seam_columns": seam_columns, "near_snapshot": near}
    return out


def figure(tables: dict, exclude: set, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
    chosen = style.select(tables["points"], exclude, "1500/500", "eta_nodefrag")
    by_key = {(p["job"], p["point"], p["arm"]): p for p in tables["points"]}
    rows = []
    for (case, slabs), row in chosen.items():
        if slabs != 8 or re.search(r"_k\d+$", case) or row["kind"] not in ("strong", "pairs"):
            continue
        trace = by_key[(row["job"], row["point"], "C")]["trace"]
        rows.append(dict(row, r_own_p50=trace["r_own_p50"], r_own_p95=trace["r_own_p95"]))
    rows.sort(key=lambda r: r["per_gpu"])
    main = [row for row in rows if row["kind"] == "strong"]
    other = [row for row in rows if row["kind"] == "pairs"]
    panel, height_panel, left, right, bottom, top = 2.65, 1.95, 0.42, 0.12, 0.40, 0.12
    fig = plt.figure(figsize=(left + panel + right, bottom + height_panel + top))
    axis = style.axes_inches(fig, left, bottom, panel, height_panel)
    style.band(axis)
    style.frame(axis)
    axis.axvline(1.0, color="#555555", lw=0.6, ls=(0, (3.0, 2.0)), zorder=0.6)

    def draw(selection, label, **marker):
        xs = [row["r_own_p50"] for row in selection]
        ys = [row["eta"] for row in selection]
        xerr = [[0.0] * len(selection), [row["r_own_p95"] - row["r_own_p50"] for row in selection]]
        yerr = [[row["eta"] - row["eta_min"] for row in selection], [row["eta_max"] - row["eta"] for row in selection]]
        axis.errorbar(xs, ys, xerr=xerr, yerr=yerr, capsize=1.6, elinewidth=0.6, capthick=0.6, ls="none", label=label,
                      **marker)
    draw(main, r"$K = 8$", color=style.COLOR_MAIN, marker="o", ms=3.4, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN)
    if other:                                           # its particles per GPU in the legend (a label would overlap)
        draw(other, rf"$K = 8$, {other[0]['per_gpu'] / 1e6:.1f}M per GPU, pair reference", color=style.COLOR_RED,
             marker="s", ms=3.4, mfc="none", mec=style.COLOR_RED, mew=0.8)
    for row in main:                                    # particles per GPU next to each point
        right_of = row["r_own_p50"] < 1.0
        axis.annotate(f"{row['per_gpu'] / 1e6:.2f}M" if row["per_gpu"] < 1e6 else f"{row['per_gpu'] / 1e6:.1f}M",
                      xy=(row["r_own_p50"], row["eta"]), xytext=(-4.0, -1.0) if right_of else (4.0, 3.0),
                      textcoords="offset points", ha="right" if right_of else "left",
                      va="top" if right_of else "bottom", fontsize=style.FONT_SIZE - 2.0, color="#444444")
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis.xaxis.set_minor_formatter(NullFormatter())
    xs = [row["r_own_p50"] for row in rows] + [row["r_own_p95"] for row in rows]
    axis.set_xlim(min(xs) / 1.8, max(xs) * 1.6)
    axis.set_ylim(0.0, 1.06)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_xlabel(r"$r_\mathrm{own}$ (p50; bar to p95)", labelpad=1.5)
    axis.set_ylabel(r"$\eta$", labelpad=1.5)
    axis.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    style.save(fig, out / "loss_sources.pdf")
    plt.close(fig)
    return rows


def ms(value: float) -> str:
    return f"{value / 1e3:.3f}"


def spread(stat: dict) -> str:
    return f"{ms(stat['mean'])}（{ms(stat['min'])}–{ms(stat['max'])}）"


def job_id(job: str) -> str:
    return job.rsplit("_", 1)[-1]


def markdown(result: dict) -> str:
    points = result["points"]
    lines = ["# E7 第 2 批：6.5 节用的数据（现有 trace，不新跑）", "",
             "工具 `docs/cluster_v6/scripts/e7_loss_sources.py`；全部 trace 在 wqd10nbj04g2（8 × 575 W）。稳态步 = step ≥ 500 且"
             "下一步存在（每个 sim 999 步：500…1498）；周期 = a_start(n + 1) − a_start(n)；T_A = a_end − a_start，T_B、T_C 同理"
             "（GPU 时间换到主机时钟）。", "",
             "## 1. 相位中位数（ms）：每个 sim 取稳态步的中位数，再对 8 个 sim 取均值（最小–最大）", "",
             "| 点 | trace（作业） | T_A | T_B | T_C | 周期 | A→B | B→C | C→下一步 A |", "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in points.items():
        phases = entry["phases"]
        lines.append(f"| {name} | {pathlib.Path(entry['trace']).name}（{job_id(entry['job'])}） | {spread(phases['a'])} | "
                     f"{spread(phases['b'])} | {spread(phases['c'])} | {spread(phases['period'])} | {spread(phases['a_to_b'])} | "
                     f"{spread(phases['b_to_c'])} | {spread(phases['c_to_next_a'])} |")
    lines += ["", "K = 1 参考（强扩展 ÷ K，弱扩展不除）：", "",
              "| 点 | K = 1 参考算例 | K = 1 trace 的 T_A / T_B / T_C / 周期（中位数） | K = 1 参考运行的平均周期 = 1e6 / 均值 fps："
              "含 defrag；无 defrag（括号：单次运行的范围） | 参考运行数（作业） |", "|---|---|---|---|---|"]
    for name, entry in points.items():
        reference = entry["reference"]
        divide = reference["divide"]
        case = reference["reference"].split(":")[1].replace("cavity", "")
        trace = "无 K = 1 trace"
        if "k1_trace" in entry:
            phases = entry["k1_trace"]["phases"]
            trace = (f"{ms(phases['a']['mean'])} / {ms(phases['b']['mean'])} / {ms(phases['c']['mean'])} / "
                     f"{ms(phases['period']['mean'])}（{pathlib.Path(entry['k1_trace']['trace']).name}，1694327，GPU 0）")
        low, high = reference["k1_period_fps_runs"]
        low_nd, high_nd = reference["k1_period_fps_nodefrag_runs"]
        lines.append(f"| {name} | {case}{'，÷ ' + str(divide) if divide > 1 else '，不除'} | {trace} | "
                     f"{ms(reference['k1_period_fps'])}（{ms(low)}–{ms(high)}）；{ms(reference['k1_period_fps_nodefrag'])}"
                     f"（{ms(low_nd)}–{ms(high_nd)}） | {reference['runs']}（{job_id(reference['job'])}） |")
    lines += ["", "## 2. 每步周期的分布（ms；8 个 sim × 999 步合并）", "",
              "均值（无 defrag）= (Σ 周期 − trace 运行窗口内 defrag 事件的墙钟) / 步数，与表格的“fps 无 defrag”同口径；"
              "defrag 事件在 f1000，落在第 999 步的周期里。", "",
              "| 点 | p50 | 均值 | 均值（无 defrag） | p95 | p99 | 各 sim 的 p50 | 各 sim 的 p95 | trace 运行 fps（含 / 无 defrag） "
              "| 计时运行 fps 均值（含 / 无 defrag；次数） |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, entry in points.items():
        period, fps, reference = entry["period"], entry["trace_fps_with_nodefrag"], entry["reference"]
        lines.append(f"| {name} | {ms(period['p50'])} | {ms(period['mean'])} | {ms(period['mean_without_defrag'])} | "
                     f"{ms(period['p95'])} | {ms(period['p99'])} | {ms(min(period['per_sim_p50']))}–{ms(max(period['per_sim_p50']))} | "
                     f"{ms(min(period['per_sim_p95']))}–{ms(max(period['per_sim_p95']))} | {fps[0]:.2f} / {fps[1]:.2f} | "
                     f"{reference['K_fps']['mean']:.2f} / {reference['K_fps_nodefrag']['mean']:.2f}；{reference['trials']} |")
    for name, entry in points.items():
        if "k1_trace" in entry:
            period, fps = entry["k1_trace"]["period"], entry["k1_trace"]["trace_fps_with_nodefrag"]
            lines.append(f"| {name} 的 K = 1 trace ÷ {entry['k1_trace']['divide']} | {ms(period['p50'])} | {ms(period['mean'])} | "
                         f"{ms(period['mean_without_defrag'])} | {ms(period['p95'])} | {ms(period['p99'])} | – | – | "
                         f"{fps[0]:.3f} / {fps[1]:.3f}（K = 1） | – |")
    steps = sorted({int(step) for entry in points.values() for step in entry["period"]["long_steps_over_2x_p50"]})
    lines += ["", f"长于 2 × p50 的步（任一点的任一 sim；不含 defrag 那一步）：{', '.join(map(str, steps))}。这些步在四个点上的周期"
              "（8 个 sim 的均值，括号为最小–最大，ms）：", "",
              "| 步 | " + " | ".join(points) + " |", "|---|" + "---|" * len(points)]
    for step in steps:
        cells = []
        for entry in points.values():
            values = entry["period_by_step"].get(str(step))
            cells.append(f"{ms(statistics.mean(values))}（{ms(min(values))}–{ms(max(values))}）" if values else "–")
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    lines.append("| 对照：p50 | " + " | ".join(ms(entry["period"]["p50"]) for entry in points.values()) + " |")
    data = result["bytes"]
    nx, ny, nz = data["grid"]
    lines += ["", "## 3. 2d_n8000 K = 8 每条链路每步的主机拷贝字节数（1694328，x_2d_n8000_K8_C）", "",
              f"网格 {nx} × {ny} × {nz}，16 Ny Nz + 16 = {data['fixed_bytes']:,} B；副本区 {data['replica_region']:,} 槽 / 层，"
              f"迁移区 {data['migrant_region']:,} 槽；按式 B 用这些容量算出的 staging 总字节 {data['staging_bytes_from_layout']:,} B，"
              "等于 trace 记录的 DMA 字节。trace 只记 B，不记 n0、n1、n_mig；每个链路步 R = B − 16 Ny Nz − 16，"
              "8a + 11b = R / 4 的解为 b = b0 + 8j、a = a0 − 11j（b0 = 3R/4 mod 8）。a_snapshot = trace 在 f1000 读回的发送端"
              "两个接缝列的粒子数。", "",
              "| 链路 | 方向 | 步（全部，含暖机） | B 最小–最大 | 不同值 | R ≥ 0 且 4 整除 R | a0 范围 | b0 = 0 / 1 / 2 的步数 | "
              "a_snapshot（两列） | 第 999 / 1000 / 1001 步的 (B − 25,712 − 32 a_snapshot) / 44 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, link in data["links"].items():
        counts = link["b0_counts"]
        near = " / ".join(f"{value['(B - fixed - 32 a_snapshot) / 44']:g}" for value in link["near_snapshot"].values())
        columns = " + ".join(f"{value:,}" for value in link["seam_columns"])
        lines.append(f"| {name} | {link['sender_direction']} | {link['steps'][0]}–{link['steps'][1]}（{link['rows_with_bytes']}） | "
                     f"{link['bytes_min']:,}–{link['bytes_max']:,} | {link['bytes_distinct']} | "
                     f"{link['residual_nonnegative_multiple_of_4']} / {link['rows_with_bytes']} | "
                     f"{link['a0_range'][0]:,}–{link['a0_range'][1]:,} | {counts.get('0', 0)} / {counts.get('1', 0)} / "
                     f"{counts.get('2', 0)} | {link['a_snapshot']:,}（{columns}） | {near} |")
    sims = len(data["peak_departed_per_sim"])
    b0_max = [max(int(b) for link in data["links"].values() if link["sender"] == sim for b in link["b0_counts"])
              for sim in range(sims)]
    lines += ["", "运行日志的 pool_health：每个 sim 一帧送出的迁移粒子最多（peak_departed_count，ghost_send 的 atomicMax，"
              f"整个运行不清零，两个方向合计）sim 0–{sims - 1} = {' / '.join(map(str, data['peak_departed_per_sim']))}；"
              f"每个 sim 两条出链路的 b0 最大值 = {' / '.join(map(str, b0_max))}。"]
    lines += ["", "## 4. loss_sources.pdf 的点（K = 8 强扩展，η 无 defrag）", "",
              "| 算例 | 每卡粒子数 | 作业 | 次数 | η（最小–最大） | r_own p50 / p95 | 参考 |", "|---|---|---|---|---|---|---|"]
    for row in result["figure"]:
        lines.append(f"| {row['point']} | {row['per_gpu'] / 1e6:.2f}M | {job_id(row['job'])} | {row['trials']} | "
                     f"{row['eta']:.3f}（{row['eta_min']:.3f}–{row['eta_max']:.3f}） | "
                     f"{row['r_own_p50']:.3f} / {row['r_own_p95']:.3f} | {row['reference']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 section 6.5 data from existing traces")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--logs", required=True, help="logs/n56 (the job directories with traces/)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-job", action="append", default=[])
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style.setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    logs = pathlib.Path(arguments.logs)
    out = pathlib.Path(arguments.out)
    result = {"points": {}, "bytes": None, "figure": None}
    k1_job, k1_name, k1_log = K1_TRACE
    k1_run = step_trace_model.load_run(logs / JOB_DIRS[k1_job] / "traces" / k1_name)
    k1_events = defrag_events(logs / JOB_DIRS[k1_job] / k1_log, k1_run["meta"]["warmup"], k1_run["meta"]["max_steps"])
    for point in POINTS:
        job_dir = logs / JOB_DIRS[point["job"]]
        run = step_trace_model.load_run(job_dir / "traces" / point["trace"])
        meta = run["meta"]
        events = defrag_events(job_dir / point["log"], meta["warmup"], meta["max_steps"])
        entry = {"label": point["label"], "job": point["job"], "trace": str(job_dir / "traces" / point["trace"]),
                 "kind": point["kind"], "K": meta["K"], "case": meta["case"], "warmup": meta["warmup"],
                 "max_steps": meta["max_steps"], "trace_steady_fps": meta["result"]["steady_fps"],
                 "trace_fps_with_nodefrag": trace_fps(job_dir / point["log"]),
                 "phases": phase_summary(run), "period": period_summary(run, events),
                 "reference": reference_period(tables, point, point["kind"])}
        if point["k1_trace"]:
            divide = meta["K"] if point["kind"] == "strong" else 1
            entry["k1_trace"] = {"trace": str(logs / JOB_DIRS[k1_job] / "traces" / k1_name), "divide": divide,
                                 "case": k1_run["meta"]["case"], "device_map": k1_run["meta"]["device_map"],
                                 "trace_steady_fps": k1_run["meta"]["result"]["steady_fps"],
                                 "trace_fps_with_nodefrag": trace_fps(logs / JOB_DIRS[k1_job] / k1_log),
                                 "phases": phase_summary(k1_run, divide),
                                 "period": period_summary(k1_run, k1_events, divide)}
        entry["period_by_step"] = {str(step): values for step, values in by_step(run).items()}
        result["points"][point["label"]] = entry
        print(f"[phases] {point['label']}: done", flush=True)
    n8000 = POINTS[3]
    job_dir = logs / JOB_DIRS[n8000["job"]]
    run = step_trace_model.load_run(job_dir / "traces" / n8000["trace"])
    result["bytes"] = link_bytes(job_dir / "traces" / n8000["trace"], job_dir / n8000["log"], run)
    result["figure"] = figure(tables, set(arguments.exclude_job), out / "figures")
    (out / "loss_sources.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    with open(out / "figures" / "loss_sources.csv", "w", encoding="utf-8", newline="") as handle:
        columns = ("case", "K", "point", "job", "host", "reference", "particles", "per_gpu", "trials", "eta", "eta_min",
                   "eta_max", "r_own_p50", "r_own_p95")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["figure"])
    (out / "loss_sources.md").write_text(markdown(result), encoding="utf-8")
    print(f"[json] wrote {out / 'loss_sources.json'} and loss_sources.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
