"""
e8_phases_table.py — E8 phase 2, deliverable 3: the paper's tab:phases from the step traces, one row block per trace
point, with the ideal-chain row computed from the K = 1 trace of the same cube (or, where K = 1 does not fit one card,
from the K = 2 pair trace of the cube: e8_C 128M).

Rules (e7_phases_table.py's): steady steps = the trace's steps >= warmup whose next step is present, minus the defrag
step (the defrag event at frame f sits in the period of step f - 1; the events come from the traced run's results row,
defrag_events); T_A = a_end - a_start, T_B = b_end - b_start, T_C = c_end - c_start, period = a_start(n+1) - a_start(n),
GPU ticks on the host clock; per card the median over its kept steps, then the mean over the cards (min-max alongside);
the step mean two ways: "window" = (sum of the steady periods - the in-window defrag wall time) / steady steps (the
fps-without-defrag convention), "skip" = the mean of the kept steps. The ideal-chain row: the K = 1 trace of the point's
cube, its phases and period divided by K for a strong point (the reference runs the whole case), not divided for a
weak point (the per-card cube), its K = 1 fps. A cube without a K = 1 trace but with a K = 2 pair trace
(trace_3d_n<n>_K2, cards 0-1, equal weights = a reference pair): the ideal-chain row = the pair trace's phases and
period divided by K / 2 (the pair runs the whole case on two cards; a quarter at K = 8), ideal_kind k2_pair; only for
K > 2 (at K = 2 the pair is the point itself). Ratio = ideal step mean / chain step mean (window - defrag means).

Usage:
    python docs/cluster_v6/scripts/e8_phases_table.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/phases_table.md, phases_table.json, phases_table.csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402

PHASES = ("a", "b", "c", "period")


def per_sim(run: dict, events: list) -> list:
    warmup, steps_total = run["meta"]["warmup"], run["meta"].get("max_steps") or 0
    defrag_steps = {frame - 1 for frame, _ in events}
    out = []
    for sim in common.sims_of(run):
        rows = common.sim_rows(run, sim)
        keep = common.steady_mask(rows, warmup)
        steps = rows["step"].astype(int)
        keep &= ~np.isin(steps, list(defrag_steps))
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        parts = {"a": rows["a_end"] - rows["a_start"], "b": rows["b_end"] - rows["b_start"], "c": rows["c_end"] - rows["c_start"],
                 "period": next_a - rows["a_start"]}
        entry = {"sim": sim, "steps": int(keep.sum())}
        for key, value in parts.items():
            entry[key + "_p50_ms"] = float(np.median(value[keep]) / common.NS_PER_MS) if keep.any() else None
        entry["period_mean_skip_ms"] = float(np.mean(parts["period"][keep]) / common.NS_PER_MS) if keep.any() else None
        all_steady = common.steady_mask(rows, warmup)
        entry["period_sum_ms"] = float(np.sum(parts["period"][all_steady]) / common.NS_PER_MS) if all_steady.any() else None
        entry["steady_steps"] = int(all_steady.sum())
        out.append(entry)
    return out


def aggregate(sims: list, events: list, divide: int = 1) -> dict:
    out = {"sims": len(sims), "divide": divide}
    for key in PHASES:
        values = [entry[key + "_p50_ms"] / divide for entry in sims if entry.get(key + "_p50_ms") is not None]
        out[key] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)} if values else None
    skip = [entry["period_mean_skip_ms"] / divide for entry in sims if entry.get("period_mean_skip_ms") is not None]
    out["step_mean_skip"] = statistics.fmean(skip) if skip else None
    defrag_ms = sum(wall for _, wall in events)
    window = [(entry["period_sum_ms"] - defrag_ms) / entry["steady_steps"] / divide for entry in sims
              if entry.get("period_sum_ms") and entry.get("steady_steps")]
    out["step_mean_window"] = statistics.fmean(window) if window else None
    out["defrag_events"] = events
    return out


def analyze_trace(trace_dir, row: dict, divide: int = 1) -> dict:
    run = common.load_trace(trace_dir)
    meta = run["meta"]
    events = common.defrag_events_in_window(row, meta["warmup"], meta.get("max_steps") or 10 ** 9)
    sims = per_sim(run, events)
    fps = common.fps_pair(row)
    return {"trace": str(trace_dir), "case": meta.get("case"), "K": meta.get("K"), "warmup": meta["warmup"],
            "max_steps": meta.get("max_steps"), "per_sim": sims, "aggregate": aggregate(sims, events, divide),
            "fps": fps[0], "fps_nodefrag": fps[1], "trace_steady_fps": (meta.get("result") or {}).get("steady_fps")}


def ms(stat) -> str:
    if not stat:
        return "-"
    return f"{stat['mean']:.3f} ({stat['min']:.3f}–{stat['max']:.3f})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    jobs = [common.load_job(directory) for directory in arguments.job]
    # the K = 1 traces by case (trace_3d_n<n>_K1 in any job) and the K = 2 pair traces (trace_3d_n<n>_K2, e8_C)
    k1, k2 = {}, {}
    for job in jobs:
        for name, path in job["traces"].items():
            if name.startswith("trace_") and name.endswith("_K1"):
                case = f"cavity{name[len('trace_'):-len('_K1')]}"
                k1[case] = (path, job["rows"].get(name))
            elif name.startswith("trace_") and name.endswith("_K2"):
                case = f"cavity{name[len('trace_'):-len('_K2')]}"
                k2[case] = (path, job["rows"].get(name))
    blocks, results, rows_csv = [], [], []
    for job in jobs:
        for name, point in sorted(common.points_of(job).items(), key=lambda item: (item[1]["slabs"], item[1]["case"])):
            if not point.get("trace_dir"):
                blocks.append(f"**{name}** ({job['name']}): no trace")
                continue
            trace_row = (point["traces"].get("C") or point["traces"].get("E") or {}).get("row")
            case, slabs = point["case"], point["slabs"]
            weak = point["case"] not in common.CUBES
            ideal_case = "cavity3d_n192" if weak else case
            try:
                chain = analyze_trace(point["trace_dir"], trace_row)
            except Exception as error:                                   # noqa: BLE001
                blocks.append(f"**{name}**: chain trace failed: {error!r}")
                continue
            ideal, ideal_kind, ideal_divide, ideal_source = None, None, None, None
            if ideal_case in k1:
                ideal_kind, ideal_divide, ideal_source = "k1", (1 if weak else slabs), k1[ideal_case]
            elif ideal_case in k2 and not weak and slabs > 2:            # e8_C: a quarter of the pair's phases at K = 8
                ideal_kind, ideal_divide, ideal_source = "k2_pair", slabs // 2, k2[ideal_case]
            if ideal_source is not None:
                try:
                    ideal = analyze_trace(ideal_source[0], ideal_source[1], divide=ideal_divide)
                except Exception as error:                               # noqa: BLE001
                    ideal = {"error": repr(error)}
            ratio = None
            if ideal and ideal.get("aggregate") and ideal["aggregate"].get("step_mean_window") and chain["aggregate"].get("step_mean_window"):
                ratio = ideal["aggregate"]["step_mean_window"] / chain["aggregate"]["step_mean_window"]
            results.append({"point": name, "job": job["name"], "case": case, "K": slabs, "kind": "weak" if weak else "strong",
                            "ideal_case": ideal_case, "ideal_kind": ideal_kind, "chain": chain, "ideal": ideal, "ratio_ideal_over_chain": ratio})
            agg = chain["aggregate"]
            lines = [f"**{name}** — {common.CASE_LABELS.get(case, case)} K = {slabs} ({job['name']}; chain trace fps {common.fmt(chain['fps_nodefrag'], 2)} "
                     f"without defrag, {len(agg['defrag_events'])} defrag event(s) in the window)", "",
                     "| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |",
                     "|---|---|---|---|---|---|---|",
                     f"| chain K = {slabs} (per card median, mean over cards (min–max)) | {ms(agg['a'])} | {ms(agg['b'])} | {ms(agg['c'])} | "
                     f"{ms(agg['period'])} | {common.fmt(agg['step_mean_window'])} | {common.fmt(agg['step_mean_skip'])} |"]
            if ideal and ideal.get("aggregate"):
                iagg = ideal["aggregate"]
                if ideal_kind == "k2_pair":
                    label = f"ideal chain = K = 2 pair trace of {common.short_case(ideal_case)} ÷ {ideal_divide}"
                else:
                    label = f"ideal chain = K = 1 trace of {common.short_case(ideal_case)}" + (f" ÷ {slabs}" if not weak else " (per-card cube, not divided)")
                lines.append(f"| {label} | {ms(iagg['a'])} | {ms(iagg['b'])} | {ms(iagg['c'])} | {ms(iagg['period'])} | "
                             f"{common.fmt(iagg['step_mean_window'])} | {common.fmt(iagg['step_mean_skip'])} |")
                lines.append(f"| ratio ideal / chain (window − defrag means) | | | | | **{common.fmt(ratio, 4)}** | |")
            elif ideal and ideal.get("error"):
                lines.append(f"| ideal chain | failed: {ideal['error']} | | | | | |")
            else:
                lines.append("| ideal chain | no K = 1 (or K = 2 pair) trace of the cube | | | | | |")
            lines += ["", "per card (sim: device): T_A / T_B / T_C / period medians (ms): "
                      + "; ".join(f"{s['sim']}: {common.fmt(s['a_p50_ms'])} / {common.fmt(s['b_p50_ms'])} / {common.fmt(s['c_p50_ms'])} / {common.fmt(s['period_p50_ms'])}"
                                  for s in chain["per_sim"])]
            blocks.append("\n".join(lines))
            for side, data in (("chain", chain), ("ideal", ideal)):
                if data and data.get("aggregate"):
                    a = data["aggregate"]
                    rows_csv.append([name, job["name"], case, slabs, side, ideal_kind if side == "ideal" else "", a["divide"], (a["a"] or {}).get("mean"),
                                     (a["b"] or {}).get("mean"), (a["c"] or {}).get("mean"), (a["period"] or {}).get("mean"), a["step_mean_window"],
                                     a["step_mean_skip"], data.get("fps_nodefrag"), ratio if side == "ideal" else None])
    text = "\n\n".join(blocks)
    print(text[:4000])
    common.write_text(out / "phases_table.md", "## tab:phases (E8 phase 2)\n\n" + text)
    common.write_json(out / "phases_table.json", results)
    with (out / "phases_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "job", "case", "K", "side", "ideal_kind", "divide", "T_A_ms", "T_B_ms", "T_C_ms", "period_ms", "step_mean_window_ms",
                         "step_mean_skip_ms", "fps_nodefrag", "ratio_ideal_over_chain"])
        writer.writerows(rows_csv)
    print(f"[e8_phases_table] {len(results)} points -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
