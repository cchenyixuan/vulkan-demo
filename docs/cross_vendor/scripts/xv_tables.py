"""xv_tables.py — the tables and the figure of the S6.6 cross-vendor campaign from the campaign directories
(xv_campaign.py e12 / e3: results.jsonl, weights/, traces/, logs/).

E1 (references): per size and trial the step time of each card without the defrag events
(tau = 1000 / steady_fps_nodefrag, ms), the throughput ratio tau_XTX / tau_5090, means and ranges.
E2 (calibrated K = 2): the two pilot rounds' weights and cuts, the final omega_5090 / omega_XTX, the fluid particles
per card at the final cut (fluid per column from round 1's equal split: 3 wall columns on each side), the particles
per slab from the bench's partition line; per trial T = 1000 / fps_nodefrag of the K = 2 run, eta_t = tau_bar_t /
(2 T_t) with tau_bar_t the harmonic mean of the trial's two reference step times; mean and min-max over the trials
(also with the defrag events); the traced run's per-card phase medians (A, B, C, step) and the two links' t_own,
r_own and parts (xv_e0.trace_summary: E7 definitions); the SM / GFX clock medians of the runs (telemetry_loop).
E3 (weight sweep): per ratio eta (mean, min-max; references = the sweep's own reference pair of the same trial) and
each card's step median (the traced run's period p50, mean over the trials); the figure weight_sweep.pdf marks the
E2 calibrated ratio, the E1 throughput ratio and the measured optimum.

Usage:
    python docs/cross_vendor/scripts/xv_tables.py --e12 docs/cross_vendor/data/e12 [--e3 docs/cross_vendor/data/e3]
        --out docs/cross_vendor/data/tables
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
HERE = pathlib.Path(__file__).resolve()
REPOSITORY = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY / "docs" / "cluster_v6" / "scripts"))
import xv_e0  # noqa: E402

CARD = {0: "5090", 1: "XTX"}


def load_rows(directory: pathlib.Path) -> dict:
    rows = {}
    for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["label"]] = row                       # the last row of a label wins
    return rows


def step_ms(row: dict, nodefrag: bool = True) -> float:
    fps = row.get("steady_fps_nodefrag") if nodefrag else row.get("steady_fps")
    if fps is None:
        fps = row.get("steady_fps")
    return 1000.0 / fps


def harmonic(a: float, b: float) -> float:
    return 2.0 / (1.0 / a + 1.0 / b)


def mean_range(values: list) -> dict:
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "n": len(values)} if values else None


def clocks(row: dict) -> dict:
    loop = row.get("telemetry_loop") or {}
    return {CARD[int(index)]: entry.get("sm_clock_median") for index, entry in loop.items() if index in ("0", "1")}


def partition_particles(log_path: pathlib.Path):
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\[partition_v7\] chain N=2: columns \[0,(\d+)\), \[\d+,(\d+)\) of (\d+); particles ([\d,]+), ([\d,]+)", line)
        if match:
            return {"cut": int(match.group(1)), "columns": int(match.group(3)),
                    "particles": [int(match.group(4).replace(",", "")), int(match.group(5).replace(",", ""))]}
    return None


# ---------------------------------------------------------------------------------------------------------------
def e12_tables(directory: pathlib.Path, sizes: list, trials: int) -> dict:
    rows = load_rows(directory)
    out = {"sizes": {}}
    for size in sizes:
        case, name = f"cavity2d_n{size}", f"2d_n{size}"
        entry = {"case": case, "particles_total": None, "skipped": (directory / f"skip_{case}.json").exists()}
        if entry["skipped"]:
            entry["skip"] = json.loads((directory / f"skip_{case}.json").read_text(encoding="utf-8"))
            out["sizes"][str(size)] = entry
            continue
        # E1
        references = {}
        for trial in range(1, trials + 1):
            pair = {}
            for member in (0, 1):
                row = rows.get(f"ref_{name}_K1_t{trial}_g{member}")
                if row is None or row.get("status") not in ("pass", "threshold_only"):
                    pair = None
                    break
                pair[CARD[member]] = {"status": row["status"], "fps": row["steady_fps"], "fps_nodefrag": row.get("steady_fps_nodefrag"),
                                      "tau_ms": step_ms(row), "tau_ms_defrag": step_ms(row, nodefrag=False), "clock": clocks(row).get(CARD[member]),
                                      "barrier": (row.get("barrier") or {}).get("wait"), "alive": [row.get("alive_start"), row.get("alive_end")]}
            if pair:
                pair["ratio"] = pair["XTX"]["tau_ms"] / pair["5090"]["tau_ms"]
                pair["tau_bar"] = harmonic(pair["5090"]["tau_ms"], pair["XTX"]["tau_ms"])
                references[trial] = pair
                entry["particles_total"] = pair["5090"]["alive"][0]
        entry["e1"] = {"trials": references,
                       "tau_5090": mean_range([p["5090"]["tau_ms"] for p in references.values()]),
                       "tau_XTX": mean_range([p["XTX"]["tau_ms"] for p in references.values()]),
                       "ratio": mean_range([p["ratio"] for p in references.values()])}
        # E2 calibration
        weights_path = directory / "weights" / f"cal_{name}_K2.json"
        if weights_path.exists():
            record = json.loads(weights_path.read_text(encoding="utf-8"))
            first = record["rounds"][0]
            fluid_per_column = first["fluid"][0] / (first["cuts"][0] - 3)
            cut = record["cuts"][0]
            fluid_left = round((cut - 3) * fluid_per_column)
            total_fluid = sum(first["fluid"])
            entry["e2_calibration"] = {
                "rounds": [{"round": r["round"], "weights": r["weights"], "cuts": r["cuts"], "omega": r["omega"], "next_cuts": r["next_cuts"],
                            "changed": r["changed"], "busy_us": r["busy_us"], "fluid": r["fluid"], "all": r["all"]} for r in record["rounds"]],
                "weights": record["weights"], "ratio": record["weights"][0] / record["weights"][1], "cut": cut,
                "fluid": [fluid_left, total_fluid - fluid_left], "fluid_share_5090": fluid_left / total_fluid,
                "partition": partition_particles(directory / "logs" / f"{name}_K2_trace.log")}
        # E2 timing runs
        runs = {}
        for trial in range(1, trials + 1):
            row = rows.get(f"{name}_K2_t{trial}")
            if row is None or row.get("status") not in ("pass", "threshold_only"):
                continue
            item = {"status": row["status"], "fps": row["steady_fps"], "fps_nodefrag": row.get("steady_fps_nodefrag"), "T_ms": step_ms(row),
                    "T_ms_defrag": step_ms(row, nodefrag=False), "clocks": clocks(row), "alive": [row.get("alive_start"), row.get("alive_end")],
                    "defrag_events": (row.get("defrag_events_summary") or {}).get("n")}
            pair = references.get(trial)
            if pair:
                item["eta"] = pair["tau_bar"] / (2.0 * item["T_ms"])
                item["eta_defrag"] = harmonic(pair["5090"]["tau_ms_defrag"], pair["XTX"]["tau_ms_defrag"]) / (2.0 * item["T_ms_defrag"])
            trial_trace = directory / "traces" / f"{name}_K2_t{trial}"
            if trial_trace.exists():                   # extra trials whose K = 2 run was traced (xv_campaign more --trace)
                try:
                    item["trace"] = xv_e0.trace_summary(trial_trace)
                    item["telemetry_loop"] = row.get("telemetry_loop")
                except Exception as error:  # noqa: BLE001
                    item["trace_error"] = repr(error)
            runs[trial] = item
        entry["e2"] = {"trials": runs, "T_ms": mean_range([r["T_ms"] for r in runs.values()]),
                       "fps_nodefrag": mean_range([r["fps_nodefrag"] for r in runs.values() if r["fps_nodefrag"]]),
                       "eta": mean_range([r["eta"] for r in runs.values() if "eta" in r]),
                       "eta_defrag": mean_range([r["eta_defrag"] for r in runs.values() if "eta_defrag" in r])}
        # E2 trace
        trace_row = rows.get(f"{name}_K2_trace")
        trace_directory = directory / "traces" / f"{name}_K2_trace"
        if trace_row and trace_directory.exists():
            try:
                summary = xv_e0.trace_summary(trace_directory)
                summary["fps"] = trace_row["steady_fps"]
                summary["fps_nodefrag"] = trace_row.get("steady_fps_nodefrag")
                summary["status"] = trace_row["status"]
                summary["seam"] = trace_row.get("seam_checks")
                summary["clocks"] = clocks(trace_row)
                entry["e2_trace"] = summary
            except Exception as error:  # noqa: BLE001
                entry["e2_trace_error"] = repr(error)
        out["sizes"][str(size)] = entry
    return out


# ---------------------------------------------------------------------------------------------------------------
def e3_tables(directory: pathlib.Path, case: str, ratios: list, trials: int) -> dict:
    rows = load_rows(directory)
    name = case.replace("cavity", "")
    references = {}
    for trial in range(1, trials + 1):
        pair = {}
        for member in (0, 1):
            row = rows.get(f"ref_{name}_K1_t{trial}_g{member}")
            if row is None or row.get("status") not in ("pass", "threshold_only"):
                pair = None
                break
            pair[CARD[member]] = {"fps_nodefrag": row.get("steady_fps_nodefrag"), "tau_ms": step_ms(row), "clock": clocks(row).get(CARD[member])}
        if pair:
            pair["tau_bar"] = harmonic(pair["5090"]["tau_ms"], pair["XTX"]["tau_ms"])
            pair["ratio"] = pair["XTX"]["tau_ms"] / pair["5090"]["tau_ms"]
            references[trial] = pair
    points = {}
    for ratio in ratios:
        per_trial = {}
        for trial in range(1, trials + 1):
            label = f"{name}_K2_r{ratio:.1f}_t{trial}"
            row = rows.get(label)
            if row is None or row.get("status") not in ("pass", "threshold_only"):
                continue
            item = {"fps_nodefrag": row.get("steady_fps_nodefrag"), "T_ms": step_ms(row), "clocks": clocks(row)}
            pair = references.get(trial)
            if pair:
                item["eta"] = pair["tau_bar"] / (2.0 * item["T_ms"])
            trace_directory = directory / "traces" / label
            if trace_directory.exists():
                try:
                    summary = xv_e0.trace_summary(trace_directory)
                    item["period_ms"] = {CARD[int(card["sim"])]: card["period"] / 1e3 for card in summary["cards"]}
                    item["B_ms"] = {CARD[int(card["sim"])]: card["b"] / 1e3 for card in summary["cards"]}
                    item["r_own"] = {link: values["r_own_p50"] for link, values in summary["links"].items()}
                    item["partition"] = partition_particles(directory / "logs" / f"{label}.log")
                except Exception as error:  # noqa: BLE001
                    item["trace_error"] = repr(error)
            per_trial[trial] = item
        points[f"{ratio:.1f}"] = {
            "trials": per_trial, "eta": mean_range([t["eta"] for t in per_trial.values() if "eta" in t]),
            "T_ms": mean_range([t["T_ms"] for t in per_trial.values()]),
            "period_5090": mean_range([t["period_ms"]["5090"] for t in per_trial.values() if "period_ms" in t]),
            "period_XTX": mean_range([t["period_ms"]["XTX"] for t in per_trial.values() if "period_ms" in t]),
            "partition": next((t["partition"] for t in per_trial.values() if t.get("partition")), None)}
    complete = {ratio: point for ratio, point in points.items() if point["eta"]}
    optimum = max(complete.items(), key=lambda item: item[1]["eta"]["mean"]) if complete else None
    return {"case": case, "references": references, "reference_ratio": mean_range([p["ratio"] for p in references.values()]),
            "points": points, "optimum": {"ratio": float(optimum[0]), "eta": optimum[1]["eta"]} if optimum else None}


# ---------------------------------------------------------------------------------------------------------------
def weight_sweep_figure(e3: dict, calibrated_ratio: float, e1_ratio: float, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import e7_scaling_figures as style
    """Single-column figure in the loss_sources.pdf style (e7_scaling_figures: 8 pt serif, same panel geometry), width
    0.6 linewidth = 0.6 x 384/72 in: eta (filled circles, min-max error bars) against omega_5090 / omega_XTX; two vertical
    lines, the pilot calibration ("pilot") and the one-device throughput ratio tau_XTX / tau_5090 ("one device"); no title."""
    style.setup_style()
    width = 0.6 * style.TEXT_WIDTH                     # 3.200 in
    left, bottom, panel_height, top, right = 0.46, 0.40, 1.85, 0.08, 0.10
    figure = plt.figure(figsize=(width, bottom + panel_height + top))
    axis = style.axes_inches(figure, left, bottom, width - left - right, panel_height)
    ratios, means, lows, highs = [], [], [], []
    for ratio, point in sorted(e3["points"].items(), key=lambda item: float(item[0])):
        if not point["eta"]:
            continue
        ratios.append(float(ratio)); means.append(point["eta"]["mean"]); lows.append(point["eta"]["min"]); highs.append(point["eta"]["max"])
    import numpy as np
    means_array = np.array(means)
    axis.errorbar(ratios, means, yerr=[means_array - np.array(lows), np.array(highs) - means_array], fmt="o", color=style.COLOR_MAIN,
                  ms=3.0, lw=0.8, capsize=1.5, mew=0.8, zorder=3)
    axis.axvline(calibrated_ratio, color=style.COLOR_RED, lw=0.8, ls=(0, (4, 2)), label="pilot", zorder=2)
    axis.axvline(e1_ratio, color="#555555", lw=0.8, ls=(0, (1, 1.5)), label=r"$\dfrac{\tau_{\mathrm{XTX}}}{\tau_{5090}}$", zorder=2)
    axis.set_xlabel(r"$\omega_\mathrm{5090}/\omega_\mathrm{XTX}$")
    axis.set_ylabel(r"$\eta$")
    axis.set_xlim(1.3, 2.7)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_ylim(0.75, 1.0)
    axis.yaxis.set_major_locator(MultipleLocator(0.05))
    style.frame(axis)
    axis.legend(loc="upper left", frameon=False, handlelength=2.2, labelspacing=0.7)
    style.save(figure, out)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------------------------
def fmt(value, digits=3, scale=1.0):
    return "—" if value is None else f"{value * scale:.{digits}f}"


def markdown(e12: dict, e3: dict, sizes: list) -> str:
    lines = []
    lines += ["### E1：单卡参考（两卡同时各一个 K = 1，同一屏障；步长不计 defrag，ms；比值 = τ_XTX / τ_5090）", "",
              "| 尺寸 | 粒子数 | τ_5090 均值（最小–最大） | τ_XTX 均值（最小–最大） | τ_XTX/τ_5090 均值（最小–最大） | 每次试验 τ_5090 / τ_XTX | 5090 SM / XTX GFX 时钟中位（MHz，各试验） |",
              "|---|---|---|---|---|---|---|"]
    for size in sizes:
        entry = e12["sizes"].get(str(size))
        if not entry or entry.get("skipped"):
            lines.append(f"| {size}² | — | 跳过（放不下，见 1.1 节） | | | | |")
            continue
        e1 = entry["e1"]
        if not e1["trials"]:
            lines.append(f"| {size}² | — | 没有数据 | | | | |")
            continue
        per_trial = "；".join(f"t{t}: {p['5090']['tau_ms']:.3f} / {p['XTX']['tau_ms']:.3f}" for t, p in sorted(e1["trials"].items()))
        clock = "；".join(f"{p['5090']['clock'] or '—'} / {p['XTX']['clock'] or '—'}" for t, p in sorted(e1["trials"].items()))
        lines.append(f"| {size}² | {entry['particles_total']:,} | {fmt(e1['tau_5090']['mean'])}（{fmt(e1['tau_5090']['min'])}–{fmt(e1['tau_5090']['max'])}） | "
                     f"{fmt(e1['tau_XTX']['mean'])}（{fmt(e1['tau_XTX']['min'])}–{fmt(e1['tau_XTX']['max'])}） | "
                     f"{fmt(e1['ratio']['mean'])}（{fmt(e1['ratio']['min'])}–{fmt(e1['ratio']['max'])}） | {per_trial} | {clock} |")
    lines += ["", "### E2：标定（sec:slab 规则；权重 = (ω_5090, ω_XTX)）", "",
              "| 尺寸 | 第 1 轮（等权重）切口 → busy 5090 / XTX µs → ω | 第 2 轮 权重、切口 → busy → ω | 最终 ω_5090/ω_XTX | 最终切口 / 列数 | 流体粒子 5090 / XTX（份额） | 粒子（含壁面）5090 / XTX |",
              "|---|---|---|---|---|---|---|"]
    for size in sizes:
        entry = e12["sizes"].get(str(size))
        if not entry or entry.get("skipped") or "e2_calibration" not in entry:
            continue
        cal = entry["e2_calibration"]
        rounds = cal["rounds"]
        r1 = rounds[0]
        r1_text = f"{r1['cuts'][0]} → {r1['busy_us'][0]:.0f} / {r1['busy_us'][1]:.0f} → {r1['omega'][0]:.4f} / {r1['omega'][1]:.4f}（切口 → {r1['next_cuts'][0]}{'，变' if r1['changed'] else '，不变'}）"
        if len(rounds) > 1:
            r2 = rounds[1]
            r2_text = f"{r2['weights'][0]:.4f} / {r2['weights'][1]:.4f}，{r2['cuts'][0]} → {r2['busy_us'][0]:.0f} / {r2['busy_us'][1]:.0f} → {r2['omega'][0]:.4f} / {r2['omega'][1]:.4f}（→ {r2['next_cuts'][0]}{'，变' if r2['changed'] else '，不变'}）"
        else:
            r2_text = "不需要（切口没变）"
        partition = cal.get("partition") or {}
        particles = f"{partition['particles'][0]:,} / {partition['particles'][1]:,}" if partition else "—"
        columns = partition.get("columns", "—") if partition else "—"
        lines.append(f"| {size}² | {r1_text} | {r2_text} | **{cal['ratio']:.3f}** | {cal['cut']} / {columns} | "
                     f"{cal['fluid'][0]:,} / {cal['fluid'][1]:,}（{cal['fluid_share_5090'] * 100:.1f} %） | {particles} |")
    lines += ["", "### E2：K = 2 的 η（η_t = τ̄_t / (2 T_t)，τ̄_t = 同一试验两张卡 K = 1 步长的调和平均；不计 defrag；括号 = 含 defrag）", "",
              "| 尺寸 | T 均值 ms（最小–最大） | fps 无 defrag 均值 | η 均值（最小–最大） | 每次试验 η | η 含 defrag 均值 | K = 2 运行 5090 / XTX 时钟中位 |",
              "|---|---|---|---|---|---|---|"]
    for size in sizes:
        entry = e12["sizes"].get(str(size))
        if not entry or entry.get("skipped") or not entry["e2"]["trials"]:
            continue
        e2 = entry["e2"]
        per_trial = "；".join(f"t{t}: {r['eta']:.4f}" for t, r in sorted(e2["trials"].items()) if "eta" in r)
        clock = "；".join(f"{r['clocks'].get('5090') or '—'} / {r['clocks'].get('XTX') or '—'}" for t, r in sorted(e2["trials"].items()))
        lines.append(f"| {size}² | {fmt(e2['T_ms']['mean'])}（{fmt(e2['T_ms']['min'])}–{fmt(e2['T_ms']['max'])}） | {fmt(e2['fps_nodefrag']['mean'], 1)} | "
                     f"**{fmt(e2['eta']['mean'])}**（{fmt(e2['eta']['min'])}–{fmt(e2['eta']['max'])}） | {per_trial} | "
                     f"{fmt(e2['eta_defrag']['mean']) if e2['eta_defrag'] else '—'} | {clock} |")
    lines += ["", "### E2：trace 运行的分相（第 500–1499 步各卡中位数，ms）", "",
              "| 尺寸 | trace 运行 fps（无 defrag） | 卡 | A | A→B | B | B→C | C | C→下一 A | 步长中位 | 步长均值 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for size in sizes:
        entry = e12["sizes"].get(str(size))
        if not entry or entry.get("skipped") or "e2_trace" not in entry:
            continue
        trace = entry["e2_trace"]
        for card in trace["cards"]:
            label = CARD[int(card["sim"])]
            lines.append(f"| {size}² | {trace['fps']:.1f}（{fmt(trace['fps_nodefrag'], 1)}） | {label} | {card['a'] / 1e3:.3f} | {card['a_to_b'] / 1e3:.3f} | "
                         f"{card['b'] / 1e3:.3f} | {card['b_to_c'] / 1e3:.3f} | {card['c'] / 1e3:.3f} | {card['c_to_next_a'] / 1e3:.3f} | "
                         f"{card['period'] / 1e3:.3f} | {card['period_mean'] / 1e3:.3f} |")
    lines += ["", "### E2：两条链路（trace 运行，稳态步中位数，ms；t_own = t_chain 减传输队列空闲；r_own = t_own / 接收卡 T_B，按步再取中位数）", "",
              "| 尺寸 | 链路 | t_chain | t_own | readback | 主机步 | upload | r_own | 步数 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for size in sizes:
        entry = e12["sizes"].get(str(size))
        if not entry or entry.get("skipped") or "e2_trace" not in entry:
            continue
        for link, values in entry["e2_trace"]["links"].items():
            direction = "5090 → XTX" if link == "s0_to_s1" else "XTX → 5090"
            lines.append(f"| {size}² | {direction} | {values['t_chain_us_p50'] / 1e3:.3f} | {values['t_own_us_p50'] / 1e3:.3f} | "
                         f"{values['readback_us_p50'] / 1e3:.3f} | {values['host_us_p50'] / 1e3:.3f} | {values['upload_us_p50'] / 1e3:.3f} | "
                         f"{values['r_own_p50']:.3f} | {values['steps']} |")
    if e3:
        lines += ["", f"### E3：权重扫描（{e3['case']}，固定权重 (r, 1)，不标定，每点 3 次试验，带 trace；η 用本扫描自己的同试验参考对）", "",
                  "| ω_5090/ω_XTX | η 均值（最小–最大） | T 均值 ms | 5090 步长中位 ms（均值，最小–最大） | XTX 步长中位 ms | 切口 / 粒子 5090 / XTX |",
                  "|---|---|---|---|---|---|"]
        for ratio, point in sorted(e3["points"].items(), key=lambda item: float(item[0])):
            if not point["eta"]:
                lines.append(f"| {ratio} | 没有数据 | | | | |")
                continue
            partition = point.get("partition") or {}
            part_text = f"{partition['cut']} / {partition['particles'][0]:,} / {partition['particles'][1]:,}" if partition else "—"
            p5090, pxtx = point["period_5090"], point["period_XTX"]
            lines.append(f"| {ratio} | **{fmt(point['eta']['mean'])}**（{fmt(point['eta']['min'])}–{fmt(point['eta']['max'])}） | {fmt(point['T_ms']['mean'])} | "
                         f"{fmt(p5090 and p5090['mean'])}（{fmt(p5090 and p5090['min'])}–{fmt(p5090 and p5090['max'])}） | "
                         f"{fmt(pxtx and pxtx['mean'])}（{fmt(pxtx and pxtx['min'])}–{fmt(pxtx and pxtx['max'])}） | {part_text} |")
        refs = e3["references"]
        lines += ["", "E3 的参考对（τ ms，5090 / XTX，比值）：" + "；".join(
            f"t{t}: {p['5090']['tau_ms']:.3f} / {p['XTX']['tau_ms']:.3f}（{p['ratio']:.3f}）" for t, p in sorted(refs.items()))]
        if e3["optimum"]:
            lines.append(f"实测最优：ω_5090/ω_XTX = {e3['optimum']['ratio']:.1f}，η = {e3['optimum']['eta']['mean']:.4f}"
                         f"（{e3['optimum']['eta']['min']:.4f}–{e3['optimum']['eta']['max']:.4f}）。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="S6.6 cross-vendor tables")
    parser.add_argument("--e12", required=True)
    parser.add_argument("--e3", default=None)
    parser.add_argument("--e3-case", default="cavity2d_n4000")
    parser.add_argument("--e3-ratios", default="1.4:2.6:0.1")
    parser.add_argument("--sizes", default="520,1000,1440,2000,2840,4000,5640,8000")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    sizes = [int(size) for size in arguments.sizes.split(",")]
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    e12 = e12_tables(pathlib.Path(arguments.e12), sizes, arguments.trials)
    e3 = None
    if arguments.e3 and (pathlib.Path(arguments.e3) / "results.jsonl").exists():
        from xv_campaign import ratios_of
        e3 = e3_tables(pathlib.Path(arguments.e3), arguments.e3_case, ratios_of(arguments.e3_ratios), arguments.trials)
        size = re.search(r"n(\d+)", arguments.e3_case).group(1)
        entry = e12["sizes"].get(size, {})
        calibrated = entry.get("e2_calibration", {}).get("ratio")
        e1_ratio = (entry.get("e1", {}).get("ratio") or {}).get("mean")
        if calibrated and e1_ratio and e3["optimum"]:
            weight_sweep_figure(e3, calibrated, e1_ratio, out / "figures" / "weight_sweep.pdf")
            e3["marks"] = {"calibrated_ratio": calibrated, "e1_ratio": e1_ratio}
    (out / "tables.json").write_text(json.dumps({"e12": e12, "e3": e3}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    text = markdown(e12, e3, sizes)
    (out / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
