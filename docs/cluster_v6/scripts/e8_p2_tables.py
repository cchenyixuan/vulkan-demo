"""
e8_p2_tables.py — E8 phase 2, deliverable 1: the per-point tables of the two jobs (e8_A, e8_B) in the E7 format.

Per point (3d_n<n>_K<K>, arm C):
  - every trial: steps / s of the K run (with the defrag events, and without them = the harness's steady_fps_nodefrag),
    the K = 1 reference of every participating card (tau_k, both conventions), eta = fps_K / (K x mean_k tau_k)
    (strong; weak: fps_K / mean_k tau_k of the per-card case; pairs = a strong point whose references are K = 2
    pairs of the case (e8_C 128M, labels _RP_..._p<i>): eta = fps_K / ((K / 2) x mean_i fps_pair_i), i.e. the
    harmonic mean of the pairs' step times over K / 2 chain steps) without the defrag events (the main value) and
    with them, eta_min against the slowest card (pair); then mean / min / max over the trials and the std;
  - the traced run: per card (sim) the step period median and busy (A + B + C) median from the trace, the fps of the
    traced run;
  - the calibration: both pilot rounds' omega, busy per device and cuts (weights/<point>.json), the final weights,
    and per card own / fluid / wall particles and shares (weights/<point>.partition.json);
  - clocks: SM clock median per card of every K run and of every reference run (telemetry_loop), power and
    temperature maxima;
  - ns per particle-step of every run (step time x K / particles).
Also the run index (every row of results.jsonl with its role, point, trial, status, fps, nodefrag fps, ns per
particle) and the failed points (failed_points.txt).

Usage:
    python docs/cluster_v6/scripts/e8_p2_tables.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/tables.md, tables.json, index.csv, trials.csv.
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


def trace_periods(trace_dir) -> dict:
    """Per sim: median period, busy, T_B over the steady steps of a trace; fps of the traced run."""
    run = common.load_trace(trace_dir)
    warmup = run["meta"]["warmup"]
    out = {"sims": [], "steady_fps": (run["meta"].get("result") or {}).get("steady_fps"),
           "device_map": run["meta"].get("device_map")}
    for sim in common.sims_of(run):
        rows = common.sim_rows(run, sim)
        keep = common.steady_mask(rows, warmup)
        period = np.diff(rows["a_start"]) / common.NS_PER_MS
        keep_period = keep[:-1]
        busy = ((rows["a_end"] - rows["a_start"]) + (rows["b_end"] - rows["b_start"]) + (rows["c_end"] - rows["c_start"])) / common.NS_PER_MS
        out["sims"].append({"sim": sim, "device": (run["meta"].get("device_map") or [None] * 8)[sim] if sim < len(run["meta"].get("device_map") or []) else None,
                            "period_p50_ms": float(np.median(period[keep_period])) if keep_period.any() else None,
                            "period_mean_ms": float(np.mean(period[keep_period])) if keep_period.any() else None,
                            "busy_p50_ms": float(np.median(busy[keep])) if keep.any() else None,
                            "T_B_p50_ms": float(np.median((rows["b_end"] - rows["b_start"])[keep] / common.NS_PER_MS)) if keep.any() else None,
                            "steps": int(keep.sum())})
    return out


def analyze_point(job: dict, name: str, point: dict) -> dict:
    case, slabs = point["case"], point["slabs"]
    particles = common.particles_of(case) if case in common.CASE_LABELS else None
    weak = any(group["case"] != case for trial in point["trials"].values() for group in trial["references"].values()
               if group.get("members"))
    trials = []
    for number in sorted(point["trials"]):
        trial = point["trials"][number]
        k_run = trial.get("C") or trial.get("E")
        row = k_run["row"] if k_run else None
        fps, fps_nodefrag = common.fps_pair(row)
        entry = {"trial": number, "label": k_run["label"] if k_run else None, "status": (row or {}).get("status"),
                 "fps": fps, "fps_nodefrag": fps_nodefrag,
                 "ns_per_particle": (1e9 / fps * slabs / particles) if fps and particles else None,
                 "ns_per_particle_nodefrag": (1e9 / fps_nodefrag * slabs / particles) if fps_nodefrag and particles else None,
                 "k_clocks": {}, "references": []}
        if row:
            for card in row.get("device_map") or []:
                entry["k_clocks"][str(card)] = common.card_clock(row, card)
        for group_name, group in trial["references"].items():
            if group.get("skipped"):
                entry["references"].append({"group": group_name, "skipped": True})
                continue
            pairs = bool(group.get("pairs"))          # K = 2 pairs of the case (labels _p<i>): e8_C
            members = []
            for member in group.get("members", []):
                m_row = member["row"]
                m_fps, m_nodefrag = common.fps_pair(m_row)
                cards = common.cards_of(m_row, member["label"])
                card = cards[0] if cards else None
                per_process = common.particles_of(group["case"]) if group["case"] in common.CASE_LABELS else None
                per_card_particles = (per_process / (2 if pairs else 1)) if per_process else None     # a pair: per card
                members.append({"label": member["label"], "name": common.member_name(member["label"]), "card": card, "cards": cards,
                                "status": (m_row or {}).get("status"), "fps": m_fps, "fps_nodefrag": m_nodefrag,
                                "ns_per_particle": (1e9 / m_fps / per_card_particles) if m_fps and per_card_particles else None,
                                "clock": common.card_clock(m_row, card) if card is not None else {},
                                "clocks": {str(c): common.card_clock(m_row, c) for c in cards}})
            passing = [m for m in members if m["status"] in ("pass", "threshold_only")]
            if pairs:
                factor, kind = slabs / 2, "pairs"
            elif group["case"] != case:
                factor, kind = 1, "weak"
            else:
                factor, kind = slabs, "strong"
            reference = {"group": group_name, "case": group["case"], "kind": kind, "members": members,
                         "complete": len(passing) == len(members) and bool(members)}
            values = [m["fps"] for m in passing if m["fps"]]
            values_nd = [m["fps_nodefrag"] for m in passing if m["fps_nodefrag"]]
            if reference["complete"] and values and fps:
                reference["eta_with"] = fps / (factor * statistics.fmean(values))
                reference["eta_min_with"] = fps / (factor * min(values))
            if reference["complete"] and values_nd and len(values_nd) == len(members) and fps_nodefrag:
                reference["eta"] = fps_nodefrag / (factor * statistics.fmean(values_nd))
                reference["eta_min"] = fps_nodefrag / (factor * min(values_nd))
                reference["tau_mean_nodefrag"] = statistics.fmean(values_nd)
            entry["references"].append(reference)
        trials.append(entry)
    stats = {}
    for key in ("eta", "eta_min", "eta_with", "eta_min_with"):
        values = [reference.get(key) for trial in trials for reference in trial["references"] if reference.get(key) is not None]
        stats[key] = common.describe(values)
    stats["fps"] = common.describe([trial["fps"] for trial in trials])
    stats["fps_nodefrag"] = common.describe([trial["fps_nodefrag"] for trial in trials])
    # per card (per pair) tau over the trials (nodefrag), keyed by the member name g<i> / p<i>
    tau = {}
    for trial in trials:
        for reference in trial["references"]:
            for member in reference.get("members", []):
                if member["card"] is not None and member["fps_nodefrag"]:
                    tau.setdefault(member["name"], []).append(member["fps_nodefrag"])
    tau_stats = {name: common.describe(values) for name, values in sorted(tau.items(), key=lambda item: common.member_sort_key(item[0]))}
    reference_kinds = {reference["kind"] for trial in trials for reference in trial["references"] if reference.get("kind")}
    reference_kind = "pairs" if "pairs" in reference_kinds else ("weak" if weak else "k1")
    trace = None
    if point.get("trace_dir"):
        try:
            trace = trace_periods(point["trace_dir"])
        except Exception as error:                                      # noqa: BLE001
            trace = {"error": repr(error)}
    trace_row = (point["traces"].get("C") or point["traces"].get("E") or {}).get("row")
    trace_fps = common.fps_pair(trace_row)
    calibration = point.get("calibration") or {}
    rounds = []
    for number, entry in enumerate(calibration.get("rounds") or [], 1):
        busy = entry.get("busy_us") or [(item.get("busy_us") if isinstance(item, dict) else item) for item in (entry.get("statistics", {}).get("per_device") or [])]
        rounds.append({"round": number, "cuts": entry.get("cuts"), "fluid": entry.get("fluid"), "all": entry.get("all"),
                       "busy_us": busy, "omega": entry.get("omega"), "next_cuts": entry.get("next_cuts"),
                       "pilot_fps": entry.get("pilot_steady_fps")})
    partition = point.get("partition") or {}
    clocks = {}
    for trial in trials:
        for card, value in trial["k_clocks"].items():
            if value.get("sm_clock_median") is not None:
                clocks.setdefault(card, {"k": [], "reference": []})["k"].append(value["sm_clock_median"])
        for reference in trial["references"]:
            for member in reference.get("members", []):
                for card, clock in (member.get("clocks") or {}).items():        # both cards of a pair
                    if clock.get("sm_clock_median") is not None:
                        clocks.setdefault(str(card), {"k": [], "reference": []})["reference"].append(clock["sm_clock_median"])
    clock_rows = {card: {"k_sm_clock": statistics.fmean(v["k"]) if v["k"] else None,
                         "reference_sm_clock": statistics.fmean(v["reference"]) if v["reference"] else None}
                  for card, v in sorted(clocks.items(), key=lambda item: int(item[0]))}
    return {"point": name, "job": job["name"], "host": point.get("host"), "line": point.get("line"), "case": case,
            "K": slabs, "kind": "weak" if weak else "strong", "reference_kind": reference_kind, "particles": particles,
            "per_card": particles / slabs if particles else None, "trials": trials, "trial_count": len(trials),
            "stats": stats, "tau_per_card": tau_stats, "trace": trace, "trace_label": (point["traces"].get("C") or {}).get("label"),
            "trace_fps": trace_fps[0], "trace_fps_nodefrag": trace_fps[1], "calibration_rounds": rounds,
            "weights": calibration.get("weights"), "cuts": calibration.get("cuts"),
            "partition": {"slabs": partition.get("slabs"), "cuts": partition.get("cuts"), "grid_columns": partition.get("grid_columns"),
                          "total_particles": partition.get("total_particles"), "fluid_total": partition.get("fluid_total")} if partition else None,
            "clocks": clock_rows, "collisions": point.get("collisions"), "failed": name in job["failed_points"]}


def index_rows(job: dict) -> list:
    rows = []
    for entry in job["index"]:
        row = job["rows"].get(entry["label"])
        fps, nodefrag = common.fps_pair(row)
        case = entry.get("case")
        slabs = entry.get("slabs") if isinstance(entry.get("slabs"), int) else None
        particles = common.particles_of(case) if case in common.CASE_LABELS else None
        rows.append({"job": job["name"], "label": entry["label"], "role": entry["role"], "point": entry.get("point"),
                     "case": case, "K": slabs, "trial": entry.get("trial"), "status": (row or {}).get("status"),
                     "fps": fps, "fps_nodefrag": nodefrag, "steps": (row or {}).get("total_steps"),
                     "ns_per_particle": (1e9 / fps * slabs / particles) if fps and particles and slabs else None,
                     "drift": (row or {}).get("drift"), "overflow_total": (row or {}).get("overflow_total"),
                     "wall_seconds": (row or {}).get("wall_seconds"), "build_seconds": (row or {}).get("build_seconds"),
                     "host": entry.get("host"), "solver": entry.get("solver")})
    return rows


def member_cell(member: dict) -> str:
    """g<i>: fps for a K = 1 member, p<i> (cards a,b): fps for a K = 2 pair."""
    name = member.get("name") or f"g{member.get('card')}"
    cards = member.get("cards") or []
    where = f" (cards {','.join(str(c) for c in cards)})" if len(cards) > 1 else ""
    return f"{name}{where}: {common.fmt(member.get('fps_nodefrag'), 2)}"


def markdown(points: list, index: list, failed: dict) -> str:
    lines = ["## E8 phase 2 — points (arm C; eta without the defrag events, mean (min–max) over the trials; reference K = 1 of the case, "
             "or K = 2 pairs of the case where K = 1 does not fit one card: eta = fps_K / ((K / 2) x mean pair fps), or K = 1 of the per-card cube (weak))", "",
             "| point | job | case | K | reference | per card | trials | fps K (nodefrag) | fps K (with) | eta | eta_min | eta (with defrag) | "
             "tau_k per card / pair (nodefrag fps) | ns / particle-step | trace fps | failed |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    reference_text = {"k1": "K = 1 of the case", "pairs": "K = 2 pairs of the case", "weak": "K = 1 of the per-card cube"}
    for point in sorted(points, key=lambda p: (p["kind"], p["K"], p["particles"] or 0)):
        s = point["stats"]
        tau = "; ".join(f"{name}: {common.rng(stats, 2)}" for name, stats in point["tau_per_card"].items())
        ns = common.describe([t["ns_per_particle_nodefrag"] for t in point["trials"]])
        lines.append(f"| {point['point']} | {point['job']} | {common.CASE_LABELS.get(point['case'], point['case'])} | {point['K']} | "
                     f"{reference_text.get(point.get('reference_kind'), point.get('reference_kind') or '-')} | "
                     f"{(point['per_card'] or 0) / 1e6:.2f}M | {point['trial_count']} | {common.rng(s['fps_nodefrag'], 2)} | {common.rng(s['fps'], 2)} | "
                     f"**{common.rng(s['eta'])}** | {common.rng(s['eta_min'])} | {common.rng(s['eta_with'])} | {tau} | {common.rng(ns, 2)} | "
                     f"{common.fmt(point['trace_fps_nodefrag'], 2)} | {'yes' if point['failed'] else ''} |")
    lines += ["", "### Per trial", "", "| point | trial | label | status | fps (with) | fps (nodefrag) | reference group | members (card: fps nodefrag) | "
              "eta | eta_min | eta (with) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for point in points:
        for trial in point["trials"]:
            for reference in trial["references"]:
                members = ", ".join(member_cell(m) for m in reference.get("members", []))
                lines.append(f"| {point['point']} | {trial['trial']} | {trial['label']} | {trial['status']} | {common.fmt(trial['fps'], 2)} | "
                             f"{common.fmt(trial['fps_nodefrag'], 2)} | {reference['group']} ({reference.get('kind', '')}) | {members} | "
                             f"{common.fmt(reference.get('eta'))} | {common.fmt(reference.get('eta_min'))} | {common.fmt(reference.get('eta_with'))} |")
    lines += ["", "### Traced runs: per card step period / busy / T_B (ms, medians over the steady steps)", "",
              "| point | trace | fps (nodefrag) | sim (device): period / busy / T_B |", "|---|---|---|---|"]
    for point in points:
        trace = point.get("trace") or {}
        cells = "; ".join(f"{s['sim']} ({s['device']}): {common.fmt(s['period_p50_ms'])} / {common.fmt(s['busy_p50_ms'])} / {common.fmt(s['T_B_p50_ms'])}"
                          for s in trace.get("sims", [])) if trace.get("sims") else trace.get("error", "-")
        lines.append(f"| {point['point']} | {point.get('trace_label') or '-'} | {common.fmt(point['trace_fps_nodefrag'], 2)} | {cells} |")
    lines += ["", "### Calibration and partition", "",
              "| point | round | cuts | omega | busy per device (us) | pilot fps | next cuts |", "|---|---|---|---|---|---|---|"]
    for point in points:
        for entry in point["calibration_rounds"]:
            lines.append(f"| {point['point']} | {entry['round']} | {entry['cuts']} | {[round(v, 4) for v in (entry['omega'] or [])]} | "
                         f"{[round(v, 1) for v in (entry['busy_us'] or [])] if entry['busy_us'] else '-'} | {common.fmt(entry['pilot_fps'], 2)} | {entry['next_cuts']} |")
        if point["partition"] and point["partition"].get("slabs"):
            lines.append(f"| {point['point']} | final | {point['partition']['cuts']} | {[round(v, 4) for v in (point['weights'] or [])]} | "
                         + "per slab own / fluid / wall: " + "; ".join(f"s{s['slab']} (dev {s['device']}): {s['own_particles']:,} / {s['fluid_particles']:,} / {s['wall_particles']:,} "
                                                                         f"({s['share_of_total'] * 100:.2f} % / {s['share_of_fluid'] * 100:.2f} %)" for s in point["partition"]["slabs"])
                         + " | | |")
    lines += ["", "### SM clocks (MHz, median per card; K runs vs references, mean over the trials)", "",
              "| point | card: K run / reference |", "|---|---|"]
    for point in points:
        lines.append(f"| {point['point']} | " + "; ".join(f"g{card}: {common.fmt(v['k_sm_clock'], 0)} / {common.fmt(v['reference_sm_clock'], 0)}"
                                                           for card, v in point["clocks"].items()) + " |")
    lines += ["", "### Failed points", ""] + ([f"- {job}: {', '.join(names)}" for job, names in failed.items() if names] or ["none"])
    lines += ["", f"### Run index ({len(index)} rows; index.csv)", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    points, index, failed = [], [], {}
    for directory in arguments.job:
        job = common.load_job(directory)
        failed[job["name"]] = job["failed_points"]
        for name, point in common.points_of(job).items():
            points.append(analyze_point(job, name, point))
        index += index_rows(job)
    text = markdown(points, index, failed)
    print(text[:4000])
    common.write_text(out / "tables.md", text)
    common.write_json(out / "tables.json", {"points": points, "index": index, "failed_points": failed})
    with (out / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0].keys()) if index else ["label"])
        writer.writeheader()
        writer.writerows(index)
    with (out / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "job", "case", "K", "kind", "reference_kind", "trial", "fps", "fps_nodefrag", "reference", "eta", "eta_min", "eta_with",
                         "members_fps_nodefrag"])
        for point in points:
            for trial in point["trials"]:
                for reference in trial["references"]:
                    writer.writerow([point["point"], point["job"], point["case"], point["K"], point["kind"], reference.get("kind"), trial["trial"], trial["fps"],
                                     trial["fps_nodefrag"], reference["group"], reference.get("eta"), reference.get("eta_min"),
                                     reference.get("eta_with"), ";".join(f"{m.get('name') or m['card']}:{m['fps_nodefrag']}" for m in reference.get("members", []))])
    print(f"[e8_p2_tables] {len(points)} points, {len(index)} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
