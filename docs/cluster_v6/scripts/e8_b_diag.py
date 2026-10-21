"""
e8_b_diag.py — E8 line B diagnosis (user 2026-10-22: why eta(32M K8) = eta(64M K8) = 0.85, and where the size dependence
of the K = 1 per-particle cost comes from). Post-processing only, on a pulled e8_B job directory (e30 layout, the raw
per-step loop traces under raw/<label>/loop_trace.csv, the step traces under traces/, the calibration records under
weights/). Writes claude/e8_b_diag.md (--md) and a JSON (--json) with every number of the tables.

Conventions: the timed window is steps 500-1499 (1000 periods of the loop trace, period i = t_loop(i) - t_loop(i - 1));
the defrag at frame 1000 makes every run bimodal (the steps after it are ~5 % slower, section 2.6), so a median over the
whole window sits on the boundary between the two halves and can flip by that 5 % with one step: the tables give the
mean without the defrag wall time (the harness's fps convention), the medians of the two halves (500-999, 1000-1499)
and the whole-window median. ns per particle-step = step / N_total (the case's particles, walls included), as in the
phase-1 report (8M K = 1: 50.25 ms -> 6.28 ns).

Sections (the user's numbering):
  1. eta decomposition per K = 8 point and trial (T, tau_k, harmonic means, eta; eta' with the cube of the per-card size).
  2. K = 1 cost against size: references by card; the K = 1 traces' A / B / C; phase-1 standalone 8M vs the line-B 8M
     references (clock, power, temperature, host submit, the same window); per-kernel 8M vs 64M; neighbour statistics
     (not available: proxies); per-step series in 100-step bins normalised to steps 100-199.
  3. cuts and imbalance of 16M / 32M / 64M K8 (both pilot rounds, the final partition).
  4. per-card phases of the 16M / 32M / 64M K8 traces (A, B, C, busy, waits, period; the slowest card; ns per own
     particle against K = 1) and the per-kernel cost per own particle of the K8 cards against K = 1.
  5. per-step distribution of every timed 32M / 64M K8 run.
  6. weak K8 (8M per card, w = 48) against 64M K8 (8M per card, w = 12.25).

Usage:
    python docs/cluster_v6/scripts/e8_b_diag.py --job DIR [--phase1 docs/cluster_v6/data/e8/phase1] --md claude/e8_b_diag.md --json OUT.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402
import e8_kernel_table as kernel_table  # noqa: E402

WINDOW = (500, 1500)                    # the timed window: steps 500 .. 1499
HALVES = ((500, 1000), (1000, 1500))
CUBES = ("cavity3d_n152", "cavity3d_n192", "cavity3d_n244", "cavity3d_n312", "cavity3d_n392", "cavity3d_n416", "cavity3d_n496")
SHORT = {"cavity3d_n152": "4M", "cavity3d_n192": "8M", "cavity3d_n244": "16M", "cavity3d_n312": "32M", "cavity3d_n392": "64M",
         "cavity3d_n416": "76M", "cavity3d_n496": "128M", "cavity3d_n192_k8": "weak 8 x 8M"}
# eta': the cube whose whole size is the K = 8 point's per-card size
PER_CARD_CUBE = {"cavity3d_n392": "cavity3d_n192", "cavity3d_n312": "cavity3d_n152", "cavity3d_n244": "2M（没有算例）",
                 "cavity3d_n192": "1M（没有算例）", "cavity3d_n152": "0.5M（没有算例）", "cavity3d_n496": "cavity3d_n244",
                 "cavity3d_n416": "9.5M（没有算例）"}
K8_POINTS = ("3d_n244_K8", "3d_n312_K8", "3d_n392_K8", "3d_n416_K8", "3d_n496_K8")


# ---------------------------------------------------------------- loop traces (per-step wall periods)
def read_loop_trace(raw_dir) -> dict:
    """{frame: {t (ns, int), submit_us, wait_us, after_defrag}} of raw/<label>/loop_trace.csv (first segment / chain)."""
    path = pathlib.Path(raw_dir) / "loop_trace.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in out:
                continue
            out[frame] = {"t": int(row["t_loop_ns"]), "submit_us": float(row["submit_us"] or "nan"),
                          "wait_us": float(row["wait_us"] or "nan"), "after_defrag": int(row["after_defrag"] or 0)}
    return out


def periods_ms(trace: dict, lo: int, hi: int) -> np.ndarray:
    values = []
    for frame in range(lo, hi):
        values.append((trace[frame]["t"] - trace[frame - 1]["t"]) / 1e6 if frame in trace and frame - 1 in trace else np.nan)
    return np.asarray(values, dtype=np.float64)


def defrag_wall(row: dict, lo: int, hi: int) -> float:
    return sum(float(event.get("wall_ms") or 0.0) for event in (row or {}).get("defrag_events") or []
               if event.get("frame") is not None and lo <= int(event["frame"]) < hi)


def window_stats(values: np.ndarray, defrag_ms: float) -> dict:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {"n": 0}
    median = float(np.median(valid))
    return {"n": int(len(valid)), "median": median, "mean": float(np.mean(valid)),
            "mean_nodefrag": float((np.sum(valid) - defrag_ms) / len(valid)), "defrag_ms": defrag_ms,
            "p95": float(np.percentile(valid, 95)), "p99": float(np.percentile(valid, 99)),
            "above_2x_median": int(np.sum(valid > 2.0 * median)), "max": float(np.max(valid))}


def period_stats(trace: dict, row: dict) -> dict:
    stats = window_stats(periods_ms(trace, *WINDOW), defrag_wall(row, *WINDOW))
    if stats["n"]:
        for index, (lo, hi) in enumerate(HALVES, 1):
            half = window_stats(periods_ms(trace, lo, hi), defrag_wall(row, lo, hi))
            stats[f"median_h{index}"] = half.get("median")
            stats[f"mean_nodefrag_h{index}"] = half.get("mean_nodefrag")
        stats["submit_us_median"] = float(np.nanmedian([trace[f]["submit_us"] for f in range(*WINDOW) if f in trace]))
        stats["wait_us_median"] = float(np.nanmedian([trace[f]["wait_us"] for f in range(*WINDOW) if f in trace]))
    return stats


def harmonic(values: list) -> float:
    values = [float(v) for v in values if v]
    return len(values) / sum(1.0 / v for v in values) if values else float("nan")


# ---------------------------------------------------------------- points
def point_runs(job: dict) -> dict:
    """point -> {trial: {"C": label, "refs": {card: label}}} from the index (role calibrated / reference)."""
    points = {}
    for entry in job["index"]:
        point = entry.get("point")
        if entry["role"] == "calibrated":
            points.setdefault(point, {}).setdefault(int(entry["trial"]), {"C": None, "refs": {}})["C"] = entry["label"]
        elif entry["role"] == "reference":
            card = common.card_of(job["rows"].get(entry["label"]), entry["label"])
            points.setdefault(point, {}).setdefault(int(entry["trial"]), {"C": None, "refs": {}})["refs"][card] = entry["label"]
    return points


def run_periods(job: dict, label: str) -> dict:
    raw = job["raw"].get(label)
    row = job["rows"].get(label)
    stats = period_stats(read_loop_trace(raw), row) if raw else {"n": 0}
    stats["label"] = label
    stats["fps"], stats["fps_nodefrag"] = common.fps_pair(row)
    stats["status"] = (row or {}).get("status")
    return stats


KEYS = ("median", "mean_nodefrag", "median_h1", "median_h2")


def section1(job: dict, points: dict) -> dict:
    out = {}
    order = sorted(points, key=lambda p: common.particles_of(common.case_of_point(p)[0]) if common.case_of_point(p)[0] in common.CASE_LABELS else 0)
    for point in order:
        case, slabs = common.case_of_point(point)
        if slabs != 8 or case not in CUBES:
            continue
        entries = []
        for trial, runs in sorted(points[point].items()):
            if not runs["C"]:
                continue
            chain = run_periods(job, runs["C"])
            refs = {card: run_periods(job, label) for card, label in sorted(runs["refs"].items())}
            complete = len([r for r in refs.values() if r.get("n")]) == 8 and chain.get("n")
            entry = {"trial": trial, "chain": chain, "refs": refs, "complete": bool(complete)}
            if complete:
                for key in KEYS:
                    taus = [refs[card][key] for card in sorted(refs)]
                    entry["tau_harm_" + key] = harmonic(taus)
                    entry["eta_" + key] = entry["tau_harm_" + key] / (slabs * chain[key])
                entry["tau_min"], entry["tau_max"] = min(r["mean_nodefrag"] for r in refs.values()), max(r["mean_nodefrag"] for r in refs.values())
                entry["eta_min"] = entry["tau_min"] / (slabs * chain["mean_nodefrag"])
                fps_refs = [refs[card]["fps_nodefrag"] for card in refs]
                entry["eta_harness"] = chain["fps_nodefrag"] / (slabs * statistics.fmean(fps_refs)) if chain.get("fps_nodefrag") and all(fps_refs) else None
            entries.append(entry)
        out[point] = {"case": case, "K": slabs, "trials": entries}
    cube_tau = {}
    for point, data in out.items():
        complete = [t for t in data["trials"] if t["complete"]]
        if complete:
            cube_tau[data["case"]] = {"point": point, **{key: statistics.fmean(t["tau_harm_" + key] for t in complete) for key in KEYS},
                                      "per_trial_mean_nodefrag": [t["tau_harm_mean_nodefrag"] for t in complete]}
    for point, data in out.items():
        cube = PER_CARD_CUBE.get(data["case"])
        data["per_card_cube"] = cube
        data["tau_per_card_cube"] = cube_tau.get(cube)
        if cube in cube_tau:
            for entry in data["trials"]:
                if entry["complete"]:
                    for key in KEYS:
                        entry["eta_prime_" + key] = cube_tau[cube][key] / entry["chain"][key]
    return {"points": out, "cube_tau": cube_tau}


# ---------------------------------------------------------------- section 2
def section2_references(job: dict, points: dict) -> dict:
    out = {}
    for point, trials in points.items():
        case, slabs = common.case_of_point(point)
        if slabs != 8 or case not in CUBES:
            continue
        particles = common.particles_of(case)
        per_trial = []
        for trial, runs in sorted(trials.items()):
            cards = {}
            for card, label in sorted(runs["refs"].items()):
                stats = run_periods(job, label)
                if stats.get("n"):
                    cards[card] = {"ns": stats["mean_nodefrag"] * 1e6 / particles, "ns_median": stats["median"] * 1e6 / particles,
                                   "ns_h1": stats["median_h1"] * 1e6 / particles, "ns_h2": stats["median_h2"] * 1e6 / particles,
                                   "fps_nodefrag": stats["fps_nodefrag"], "submit_us": stats["submit_us_median"], "wait_us": stats["wait_us_median"],
                                   **common.card_clock(job["rows"].get(label) or {}, card)}
            if cards:
                values = [c["ns"] for c in cards.values()]
                per_trial.append({"trial": trial, "cards": cards, "spread": (max(values) - min(values)) / min(values),
                                  "min_card": min(cards, key=lambda c: cards[c]["ns"]), "max_card": max(cards, key=lambda c: cards[c]["ns"])})
        if per_trial:
            by_card = {}
            for entry in per_trial:
                for card, value in entry["cards"].items():
                    by_card.setdefault(card, []).append(value["ns"])
            out[case] = {"point": point, "particles": particles, "per_trial": per_trial,
                         "card_mean": {card: statistics.fmean(v) for card, v in sorted(by_card.items())},
                         "spread_mean": statistics.fmean(e["spread"] for e in per_trial), "spread_max": max(e["spread"] for e in per_trial),
                         "ns_mean": statistics.fmean(v for values in by_card.values() for v in values)}
    return out


def trace_phases(trace_dir) -> dict:
    """Per sim over the steady steps (>= warmup, next step present) without the defrag step: means of A, B, C, busy,
    period (ms); per-step medians of the waits a_end->b_start, b_end->c_start, c_end->a_start(n+1) and of period - busy;
    the means of busy over the two halves."""
    run = common.load_trace(trace_dir)
    meta = run["meta"]
    warmup = meta["warmup"]
    row = meta.get("result") or {}
    sims = []
    for sim in common.sims_of(run):
        rows = common.sim_rows(run, sim)
        keep = common.steady_mask(rows, warmup)
        if not keep.any():
            continue
        steps = rows["step"].astype(int)
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        parts = {"A": rows["a_end"] - rows["a_start"], "B": rows["b_end"] - rows["b_start"], "C": rows["c_end"] - rows["c_start"],
                 "wait_AB": rows["b_start"] - rows["a_end"], "wait_BC": rows["c_start"] - rows["b_end"],
                 "wait_CA": next_a - rows["c_end"], "period": next_a - rows["a_start"]}
        parts["busy"] = parts["A"] + parts["B"] + parts["C"]
        parts["gap"] = parts["period"] - parts["busy"]
        keep &= np.isfinite(parts["period"])
        # the defrag step: the largest gap of the window when a defrag event falls inside it (frame 1000)
        kept = keep.copy()
        if keep.sum() > 10:
            gap = np.where(keep, parts["gap"], -np.inf)
            if np.max(gap) > 10 * np.median(parts["gap"][keep]) + 1e6:
                kept[int(np.argmax(gap))] = False
        entry = {"sim": sim, "device": (meta.get("device_map") or [None] * 8)[sim] if sim < len(meta.get("device_map") or []) else None,
                 "steps": int(kept.sum()), "defrag_step_removed": int(keep.sum() - kept.sum())}
        for key in ("A", "B", "C", "busy", "period"):
            entry[key + "_ms"] = float(np.mean(parts[key][kept]) / common.NS_PER_MS)
            entry[key + "_median_ms"] = float(np.median(parts[key][kept]) / common.NS_PER_MS)
        for key in ("wait_AB", "wait_BC", "wait_CA", "gap"):
            entry[key + "_ms"] = float(np.median(parts[key][kept]) / common.NS_PER_MS)
            entry[key + "_mean_ms"] = float(np.mean(parts[key][kept]) / common.NS_PER_MS)
        for index, (lo, hi) in enumerate(HALVES, 1):
            mask = kept & (steps >= lo) & (steps < hi)
            entry[f"busy_h{index}_ms"] = float(np.mean(parts["busy"][mask]) / common.NS_PER_MS) if mask.any() else None
            entry[f"period_h{index}_ms"] = float(np.mean(parts["period"][mask]) / common.NS_PER_MS) if mask.any() else None
        sims.append(entry)
    return {"trace": str(trace_dir), "name": pathlib.Path(trace_dir).name, "case": meta.get("case"), "K": meta.get("K"), "warmup": warmup,
            "sims": sims, "device_map": meta.get("device_map"), "steady_fps": row.get("steady_fps")}


def section2_traces(job: dict) -> dict:
    out = {}
    for name, path in job["traces"].items():
        if name.startswith("trace_") and name.endswith("_K1"):
            case = f"cavity{name[len('trace_'):-len('_K1')]}"
            if case not in common.CASE_LABELS:
                continue
            try:
                phases = trace_phases(path)
            except Exception as error:                                      # noqa: BLE001
                out[case] = {"error": repr(error)}
                continue
            particles = common.particles_of(case)
            sim = phases["sims"][0] if phases["sims"] else {}
            out[case] = {"trace": name, "particles": particles, "fluid": common.lattice_of(case)["fluid_total"], "sim": sim,
                         **{key + "_ns": sim[key + "_ms"] * 1e6 / particles for key in ("A", "B", "C", "busy", "period") if sim.get(key + "_ms") is not None},
                         "fps_nodefrag": common.fps_pair(job["rows"].get(name))[1]}
    return out


def section2_phase1(phase1_dir, refs: dict, series: dict) -> dict:
    out = {"phase1": None, "line_b": None, "line_b_same_window": None}
    if phase1_dir:
        path = pathlib.Path(phase1_dir) / "results.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            row = next((r for r in rows if r.get("label", "").lower().startswith("8m_k1")), None)
            if row:
                card = common.card_of(row, row["label"])
                fps = common.fps_pair(row)
                out["phase1"] = {"label": row["label"], "node": row.get("node"), "card": card, "fps": fps[0], "fps_nodefrag": fps[1],
                                 "total_steps": row.get("total_steps"), "window": "steps 100-299 (300 steps, warmup 100)",
                                 "ns_per_particle": 1e9 / fps[1] / 8_000_000 if fps[1] else None, **common.card_clock(row, card)}
    if "cavity3d_n192" in refs:
        cards = {}
        for entry in refs["cavity3d_n192"]["per_trial"]:
            for card, value in entry["cards"].items():
                cards.setdefault(card, []).append(value)
        out["line_b"] = {card: {key: (statistics.fmean(v[key] for v in values if v.get(key) is not None) if any(v.get(key) is not None for v in values) else None)
                                for key in ("ns", "ns_h1", "ns_h2", "sm_clock_median", "sm_clock_min", "power_median", "temperature_max", "submit_us", "wait_us")}
                        for card, values in sorted(cards.items())}
    series_8m = series.get("cavity3d_n192")
    if series_8m:
        bins = {b["bin"]: b for b in series_8m["bins"]}
        same = [bins[k]["mean_kept"] for k in ("100-199", "200-299") if k in bins and bins[k].get("mean_kept")]
        if same:
            out["line_b_same_window"] = {"label": series_8m["label"], "mean_ms": statistics.fmean(same), "ns_per_particle": statistics.fmean(same) * 1e6 / 8_000_000}
    return out


def section2_kernels(job: dict, cases=("cavity3d_n192", "cavity3d_n392")) -> dict:
    out = {}
    for case in cases:
        name = f"trace_{common.short_case(case)}_K1"
        if name in job["traces"]:
            try:
                out[case] = kernel_table.analyze(job["traces"][name], case, job["raw"].get(name), None)
            except Exception as error:                                      # noqa: BLE001
                out[case] = {"error": repr(error)}
    return out


def section2_series(job: dict, points: dict, bins: int = 100) -> dict:
    """Mean period per 100-step bin (steps above 2 x the run median removed = the defrag), normalised to 100-199: one K = 1
    reference per cube (trial 1, card 0) and the trial-1 chain run of 32M / 64M K8."""
    out = {}
    targets = [(f"{SHORT[c]} K1", (points.get(f"{common.short_case(c)}_K8") or {}).get(1, {}).get("refs", {}).get(0))
               for c in ("cavity3d_n192", "cavity3d_n392", "cavity3d_n152", "cavity3d_n244", "cavity3d_n312")]
    targets += [(f"{SHORT[c]} K8 链路", (points.get(f"{common.short_case(c)}_K8") or {}).get(1, {}).get("C"))
                for c in ("cavity3d_n312", "cavity3d_n392")]
    for title, label in targets:
        raw = job["raw"].get(label) if label else None
        if not raw:
            continue
        trace = read_loop_trace(raw)
        total = max(trace) + 1 if trace else 0
        values = periods_ms(trace, 1, total)
        run_median = float(np.nanmedian(values))
        rows = []
        for start in range(0, total, bins):
            chunk = values[max(start - 1, 0): start + bins - 1]
            chunk = chunk[np.isfinite(chunk)]
            kept = chunk[chunk <= 2.0 * run_median]
            rows.append({"bin": f"{start}-{start + bins - 1}", "mean_all": float(np.mean(chunk)) if len(chunk) else None,
                         "mean_kept": float(np.mean(kept)) if len(kept) else None, "removed": int(len(chunk) - len(kept))})
        base = next((r["mean_kept"] for r in rows if r["bin"].startswith("100-")), None)
        for r in rows:
            r["normalised"] = r["mean_kept"] / base if base and r["mean_kept"] else None
        key = {"8M K1": "cavity3d_n192", "64M K1": "cavity3d_n392", "4M K1": "cavity3d_n152", "16M K1": "cavity3d_n244", "32M K1": "cavity3d_n312"}.get(title, title)
        out[key] = {"title": title, "label": label, "bins": rows, "run_median": run_median}
    return out


# ---------------------------------------------------------------- section 3
def section3(job: dict) -> dict:
    out = {}
    for point in K8_POINTS:
        weights = job["weights"].get(point)
        partition = job["partitions"].get(point)
        if not weights:
            continue
        case, slabs = common.case_of_point(point)
        lattice = common.lattice_of(case)
        rounds = []
        for entry in weights.get("rounds") or []:
            own = entry.get("own_columns") or []
            fluid_columns = [c - (2 if i in (0, len(own) - 1) else 0) for i, c in enumerate(own)]
            rounds.append({"round": entry["round"], "cuts": entry["cuts"], "own_columns": own, "fluid_columns": fluid_columns, "all": entry.get("all"),
                           "fluid": entry.get("fluid"), "busy_us": entry.get("busy_us"), "omega": entry.get("omega"), "next_cuts": entry.get("next_cuts"),
                           "changed": entry.get("changed"), "pilot_fps": entry.get("pilot_steady_fps"), "weights": entry.get("weights")})
        final = None
        if partition:
            final = {"cuts": partition.get("cuts"),
                     "slabs": [{k: s.get(k) for k in ("slab", "device", "own_columns", "fluid_columns", "own_particles", "fluid_particles", "wall_particles", "share_of_total", "weight")}
                               for s in partition.get("slabs") or []]}
        out[point] = {"case": case, "fluid_columns_total": lattice["fluid_columns"], "per_card": lattice["fluid_columns"] / slabs,
                      "rounds": rounds, "final": final, "final_weights": weights.get("weights"), "final_cuts": weights.get("cuts")}
    return out


# ---------------------------------------------------------------- section 4
def section4(job: dict, k1_refs: dict, k1_traces: dict) -> dict:
    out = {}
    for point in K8_POINTS + ("3d_n192_k8_K8",):
        trace_dir = job["traces"].get(f"{point}_C")
        if not trace_dir:
            continue
        try:
            phases = trace_phases(trace_dir)
        except Exception as error:                                          # noqa: BLE001
            out[point] = {"error": repr(error)}
            continue
        case, slabs = common.case_of_point(point)
        partition = job["partitions"].get(point) or {}
        by_slab = {s["slab"]: s for s in partition.get("slabs") or []}
        for sim in phases["sims"]:
            slab = by_slab.get(sim["sim"], {})
            sim["own_columns"], sim["fluid_columns"], sim["own_particles"] = slab.get("own_columns"), slab.get("fluid_columns"), slab.get("own_particles")
            sim["seams"] = 1 if sim["sim"] in (0, len(phases["sims"]) - 1) else 2
            sim["ns_per_own"] = sim["busy_ms"] * 1e6 / sim["own_particles"] if sim.get("own_particles") else None
        if phases["sims"]:
            slowest = max(phases["sims"], key=lambda s: s["busy_ms"])
            period = statistics.fmean(s["period_ms"] for s in phases["sims"])
            busy_mean = statistics.fmean(s["busy_ms"] for s in phases["sims"])
            phases["slowest"] = {k: slowest.get(k) for k in ("sim", "device", "busy_ms", "own_columns", "fluid_columns", "seams", "own_particles", "ns_per_own")}
            phases["period_mean_of_cards_ms"] = period
            phases["busy_mean_of_cards_ms"] = busy_mean
            phases["period_minus_slowest_busy_ms"] = period - slowest["busy_ms"]
            phases["balance"] = busy_mean / period
            own_total = sum(s["own_particles"] for s in phases["sims"] if s.get("own_particles"))
            phases["ns_per_own_mean"] = busy_mean * 1e6 / (own_total / len(phases["sims"])) if own_total else None
            phases["k1_ns_same_size"] = (k1_refs.get(case) or {}).get("ns_mean")
            per_card_cube = PER_CARD_CUBE.get(case)
            phases["k1_ns_per_card_size"] = (k1_refs.get(per_card_cube) or {}).get("ns_mean") if per_card_cube in k1_refs else None
            if case == "cavity3d_n192_k8":
                phases["k1_ns_per_card_size"] = (k1_refs.get("cavity3d_n192") or {}).get("ns_mean")
        phases["fps_nodefrag"] = common.fps_pair(job["rows"].get(f"{point}_trace_C"))[1]
        out[point] = phases
    return out


def section4_kernels(job: dict, k1_kernels: dict) -> dict:
    """Per-kernel cost per own particle of the K8 cards (32M, 64M) against the K = 1 trace of the same case."""
    out = {}
    for point in ("3d_n312_K8", "3d_n392_K8"):
        case, slabs = common.case_of_point(point)
        trace_dir = job["traces"].get(f"{point}_C")
        if not trace_dir:
            continue
        try:
            k8 = kernel_table.analyze(trace_dir, case, job["raw"].get(f"{point}_trace_C"), job["partitions"].get(point))
        except Exception as error:                                          # noqa: BLE001
            out[point] = {"error": repr(error)}
            continue
        k1 = k1_kernels.get(case)
        if not k1 or k1.get("error"):
            try:
                k1 = kernel_table.analyze(job["traces"][f"trace_{common.short_case(case)}_K1"], case, None, None)
            except Exception as error:                                      # noqa: BLE001
                k1 = {"error": repr(error)}
        names = list(k8["kernel_order"]) + [n for n in (k1.get("kernel_order") or []) if n not in k8["kernel_order"]]
        rows = []
        for name in names:
            per_card = [entry for sim in k8["sims"] for entry in sim["kernels"].values() if entry["kernel"] == name and entry.get("ns_per_particle") is not None]
            k1_entry = (k1.get("mean_over_sims") or {}).get(name) if not k1.get("error") else None
            rows.append({"kernel": name, "phase": (per_card[0]["phase"] if per_card else k1_entry["phase"]),
                         "k1_ns": k1_entry["ns_per_particle_mean"] if k1_entry else None,
                         "k8_ns_min": min(e["ns_per_particle"] for e in per_card) if per_card else None,
                         "k8_ns_mean": statistics.fmean(e["ns_per_particle"] for e in per_card) if per_card else None,
                         "k8_ns_max": max(e["ns_per_particle"] for e in per_card) if per_card else None,
                         "k8_us_per_card": [round(e["us_p50"], 1) for e in per_card]})
        # combined rows: the K = 1 interior kernel against the K8 interior + boundary variants, the G1 bookkeeping, the sums
        combos = [("correction_density（interior + boundary）", ["correction_density_interior", "correction_density_boundary", "correction_boundary", "density_boundary"]),
                  ("force（deep interior + boundary）", ["force_deep_interior", "force"]),
                  ("A 段合计（predict + voxel + ghost）", [n for n in names if (k8["mean_over_sims"].get(n) or {}).get("phase") == "A"]),
                  ("B 段合计", [n for n in names if (k8["mean_over_sims"].get(n) or {}).get("phase") == "B"]),
                  ("C 段合计", [n for n in names if (k8["mean_over_sims"].get(n) or {}).get("phase") == "C"]),
                  ("G1 簿记（ghost / expand / install / append / compact / marker / extrapolate）",
                   ["ghost_leading", "ghost_trailing", "expand", "install_leading", "install_trailing", "append_departed", "band_compact", "deep_wall_marker", "wall_extrapolate"]),
                  ("全部核之和", names)]
        for title, members in combos:
            per_card = []
            for sim in k8["sims"]:
                values = [e["ns_per_particle"] for e in sim["kernels"].values() if e["kernel"] in members and e.get("ns_per_particle") is not None]
                if values:
                    per_card.append(sum(values))
            k1_values = [(k1.get("mean_over_sims") or {}).get(n, {}).get("ns_per_particle_mean") for n in members] if not k1.get("error") else []
            k1_values = [v for v in k1_values if v is not None]
            rows.append({"kernel": title, "phase": "Σ", "combo": True, "k1_ns": sum(k1_values) if k1_values else None,
                         "k8_ns_min": min(per_card) if per_card else None, "k8_ns_mean": statistics.fmean(per_card) if per_card else None,
                         "k8_ns_max": max(per_card) if per_card else None, "k8_us_per_card": []})
        out[point] = {"case": case, "rows": rows, "g1_share_mean": k8.get("g1_share_mean"), "g1_share_range": k8.get("g1_share_range"),
                      "period_mean_ms": k8.get("period_mean_of_sims_ms"), "k1_period_ms": k1.get("period_mean_of_sims_ms") if not k1.get("error") else None,
                      "sims_own": [s.get("own_particles") for s in k8["sims"]]}
    return out


def section5(job: dict, points: dict) -> dict:
    out = {}
    for point in K8_POINTS:
        entries = []
        for trial, runs in sorted((points.get(point) or {}).items()):
            if runs["C"]:
                stats = run_periods(job, runs["C"])
                stats["trial"] = trial
                entries.append(stats)
        if f"{point}_trace_C" in job["raw"]:
            stats = run_periods(job, f"{point}_trace_C")
            stats["trial"] = "trace"
            entries.append(stats)
        if entries:
            out[point] = entries
    return out


# ---------------------------------------------------------------- markdown
def f(value, digits=3):
    return common.fmt(value, digits)


def pct(value, digits=1):
    return "-" if value is None else f"{value * 100:.{digits}f} %"


def facts(s1, s2r, s2p, s2k, s2s, s3, s4, s4k) -> list:
    """The numbers the sections give, side by side (no interpretation)."""
    L = ["## 0. 各节数据的并列（只列数字）", ""]
    rows = []
    for point, data in s1["points"].items():
        complete = [t for t in data["trials"] if t["complete"]]
        if not complete:
            continue
        eta = statistics.fmean(t["eta_mean_nodefrag"] for t in complete)
        eta_p = [t["eta_prime_mean_nodefrag"] for t in complete if t.get("eta_prime_mean_nodefrag")]
        chain = s4.get(point) or {}
        same = (s2r.get(data["case"]) or {}).get("ns_mean")
        per_card = chain.get("k1_ns_per_card_size")
        own = chain.get("ns_per_own_mean")
        balance = chain.get("balance")
        rows.append(f"| {SHORT.get(data['case'], data['case'])} K8 | {eta:.3f} | {f(statistics.fmean(eta_p)) if eta_p else '缺'} | {f(same)} | {f(per_card) if per_card else '缺'} | "
                    f"{f(own)} | {f(same / own) if same and own else '-'} | {f(per_card / own) if per_card and own else '缺'} | {f(balance)} | "
                    f"{f(balance * same / own) if balance and same and own else '-'} | {f(chain.get('period_minus_slowest_busy_ms'))} |")
    L += ["| 点 | η（nd，试验平均） | η' | K = 1 ns/粒子（同总规模，参考 8 卡平均） | K = 1 ns/粒子（每卡尺寸立方体） | 链路 ns/own 粒子（trace，各卡 busy 平均） | 同规模 K1 / 链路 own | 每卡尺寸 K1 / 链路 own | 平衡度 busy 平均 / 周期 | 平衡度 × (同规模 K1 / own) | 周期 − 最慢卡 busy (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    L += ["", "η 的两个因子：η ≈ (busy 平均 / 周期) × (K = 1 同规模成本 / 链路每 own 粒子成本)——最后一列对比第 2 列；周期 − 最慢卡 busy ≈ 0 说明链路周期就是最慢卡的 busy。", ""]
    k1 = [(SHORT[c], s2r[c]["ns_mean"]) for c in CUBES if c in s2r]
    if k1:
        L.append("K = 1 每粒子成本随规模（参考 8 卡平均，nd）：" + "，".join(f"{n} {v:.3f}" for n, v in k1) + " ns；"
                 + (f"逐核（64M / 8M，每 N_total）：correction_density_interior {f(s2k['cavity3d_n392']['mean_over_sims']['correction_density_interior']['ns_per_particle_mean'] / s2k['cavity3d_n192']['mean_over_sims']['correction_density_interior']['ns_per_particle_mean'])}，"
                    f"force_deep_interior {f(s2k['cavity3d_n392']['mean_over_sims']['force_deep_interior']['ns_per_particle_mean'] / s2k['cavity3d_n192']['mean_over_sims']['force_deep_interior']['ns_per_particle_mean'])}。"
                    if s2k.get("cavity3d_n392") and s2k.get("cavity3d_n192") and not s2k["cavity3d_n392"].get("error") and not s2k["cavity3d_n192"].get("error") else ""))
    p1, same = s2p.get("phase1"), s2p.get("line_b_same_window")
    if p1 and same:
        g0 = (s2p.get("line_b") or {}).get(0) or {}
        L.append(f"8M K = 1 同一窗口（第 100–299 步）：阶段 1 单独 {p1['ns_per_particle']:.3f} ns（SM {f(p1['sm_clock_median'], 0)} MHz，{f(p1['temperature_max'], 0)} °C）对 B 线并发 g0 {same['ns_per_particle']:.3f} ns"
                 f"（SM {f(g0.get('sm_clock_median'), 0)} MHz，{f(g0.get('temperature_max'), 0)} °C）；B 线 g0 在第 500–1499 步 {f(g0.get('ns'))} ns（h1 {f(g0.get('ns_h1'))} / h2 {f(g0.get('ns_h2'))}）。")
    series = [(s2s[k]["title"], s2s[k]["bins"]) for k in ("cavity3d_n192", "cavity3d_n392", "32M K8 链路", "64M K8 链路") if k in s2s]
    if series:
        parts = []
        for title, bins_ in series:
            by = {b["bin"]: b for b in bins_}
            if all(k in by for k in ("900-999", "1100-1199", "1400-1499")) and by["900-999"].get("normalised"):
                parts.append(f"{title} 900–999 {by['900-999']['normalised']:.3f} → 1100–1199 {by['1100-1199']['normalised']:.3f} → 1400–1499 {by['1400-1499']['normalised']:.3f}")
        L.append("逐步序列（归一到 100–199）：" + "；".join(parts) + "。第 1000 帧 defrag 之后的一跳对所有运行相同。")
    for point, data in s3.items():
        if data.get("final"):
            slabs = data["final"]["slabs"]
            hi, lo = max(s["fluid_columns"] for s in slabs), min(s["fluid_columns"] for s in slabs)
            L.append(f"{SHORT.get(data['case'], point)} K8 最终切口：流体列 {[s['fluid_columns'] for s in slabs]}（{hi} / {lo} 列），own 粒子最大 / 最小 {max(s['own_particles'] for s in slabs) / min(s['own_particles'] for s in slabs):.3f}，"
                     f"{len(data['rounds'])} 轮 pilot" + ("，最终切口未被 pilot" if data["rounds"] and data["rounds"][-1]["next_cuts"] != data["rounds"][-1]["cuts"] else "") + "。")
    for point, data in s4k.items():
        if data.get("error"):
            continue
        combo = {r["kernel"]: r for r in data["rows"] if r.get("combo")}
        cd, fo, tot = combo.get("correction_density（interior + boundary）"), combo.get("force（deep interior + boundary）"), combo.get("全部核之和")
        if cd and fo and tot:
            L.append(f"{SHORT.get(data['case'], point)} K8 逐核（ns/own 粒子，8 卡平均，对 K = 1 同规模）：correction_density {f(cd['k8_ns_mean'])} / {f(cd['k1_ns'])} = {f(cd['k8_ns_mean'] / cd['k1_ns']) if cd['k1_ns'] else '-'}，"
                     f"force {f(fo['k8_ns_mean'])} / {f(fo['k1_ns'])} = {f(fo['k8_ns_mean'] / fo['k1_ns']) if fo['k1_ns'] else '-'}，全部核 {f(tot['k8_ns_mean'])} / {f(tot['k1_ns'])} = {f(tot['k8_ns_mean'] / tot['k1_ns']) if tot['k1_ns'] else '-'}；"
                     f"G1 冗余份额 {pct(data['g1_share_mean'])}。")
    L.append("")
    return L


def markdown(job: dict, s1, s2r, s2t, s2p, s2k, s2s, s3, s4, s4k, s5) -> str:
    prov = job.get("provenance") or {}
    host = prov.get("host") or next((r.get("node") for r in job["rows"].values() if r.get("node")), "-")
    L = [f"# E8 B 线诊断（作业 {job['name']}，节点 {host}）", "",
         f"数据：`results.jsonl` / `index.tsv`（{len(job['rows'])} 次运行）；每次运行的逐步墙钟 `raw/<label>/loop_trace.csv`（`t_loop_ns` 的差 = 步长，第 500–1499 步 = 1000 个步长，第 1000 帧有一次 defrag）；"
         f"step trace `traces/`（{len(job['traces'])} 个，每核起止）；标定记录 `weights/`。每粒子成本一律按算例总粒子数（含壁面与盖板）：ns = 步长 / N_total，与阶段 1 报告一致（8M K = 1：50.25 ms → 6.28 ns）。"
         "K = 1 参考与链路在同一作业、同一节点、全部 575 W。", "",
         "**口径说明（重要）**：第 1000 帧的 defrag 之后每步慢 5 % 左右（2.6 节，五个尺寸、K = 1 与链路都如此），所以 500–1499 的分布是双峰的，"
         "整窗中位数正好落在两峰之间，差一步就能跳 5 %（K = 1 trace 的\"周期中位数 − busy 中位数\"假象就是这么来的）。各表因此同时给：去 defrag 墙钟的均值（harness 的 fps 口径）、"
         "前半 500–999 与后半 1000–1499 各自的中位数、整窗中位数。η 的几种口径在第 1 节并列。", "", "缺的数据在各节末尾标出。", ""]
    L += facts(s1, s2r, s2p, s2k, s2s, s3, s4, s4k)
    # ---- 1
    L += ["## 1. 每个 K = 8 点的 η 分解", "",
          "T = 链路步长（ms）；τ_k = 卡 k 的 K = 1 参考步长（同一窗口、同一口径）；τ̄ = τ_k 的调和平均 = 1 / mean(1/τ_k)；η = τ̄ / (8 T)。"
          "下标：nd = 去 defrag 的均值，h1 / h2 = 前后半窗中位数，med = 整窗中位数；η_harness = harness 的 fps 口径（= nd）；η_min = min τ_k / (8 T_nd)；"
          "η' = τ'(每卡尺寸立方体的 8 个并发参考的调和平均，该点各试验平均，nd 口径) / T_nd。", "",
          "| 点 | 试验 | T_nd | T_h1 | T_h2 | T_med | τ_0 … τ_7（nd，ms） | τ̄_nd | τ̄_h1 | τ̄_h2 | τ̄_med | **η_nd** | η_h1 | η_h2 | η_med | η_harness | η_min | 每卡立方体 | τ'_nd | **η'_nd** | η'_h1 | η'_h2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for point, data in s1["points"].items():
        cube = data.get("tau_per_card_cube")
        for t in data["trials"]:
            c = t["chain"]
            if not t["complete"]:
                L.append(f"| {SHORT.get(data['case'], data['case'])} K8 | {t['trial']} | 缺（{c.get('label')} 或其参考没有 raw） | | | | | | | | | | | | | | | | | | | |")
                continue
            taus = " / ".join(f"{t['refs'][k]['mean_nodefrag']:.2f}" for k in sorted(t["refs"]))
            L.append(f"| {SHORT.get(data['case'], data['case'])} K8 | {t['trial']} | {f(c['mean_nodefrag'], 2)} | {f(c['median_h1'], 2)} | {f(c['median_h2'], 2)} | {f(c['median'], 2)} | {taus} | "
                     f"{f(t['tau_harm_mean_nodefrag'], 2)} | {f(t['tau_harm_median_h1'], 2)} | {f(t['tau_harm_median_h2'], 2)} | {f(t['tau_harm_median'], 2)} | "
                     f"**{f(t['eta_mean_nodefrag'])}** | {f(t['eta_median_h1'])} | {f(t['eta_median_h2'])} | {f(t['eta_median'])} | {f(t.get('eta_harness'))} | {f(t['eta_min'])} | "
                     f"{SHORT.get(data['per_card_cube'], data['per_card_cube'])} | {f(cube['mean_nodefrag'], 2) if cube else '缺'} | "
                     f"**{f(t.get('eta_prime_mean_nodefrag')) if cube else '缺'}** | {f(t.get('eta_prime_median_h1')) if cube else '缺'} | {f(t.get('eta_prime_median_h2')) if cube else '缺'} |")
    L += ["", "τ' 的来源：各立方体自己的 8 个并发 K = 1 参考（该点全部试验；每次试验 8 卡的调和平均，再对试验平均）：", "",
          "| 立方体 | 点 | 各试验 τ̄_nd（ms） | τ̄_nd 平均 | τ̄_h1 | τ̄_h2 | τ̄_med | ns/粒子·步（nd） |", "|---|---|---|---|---|---|---|---|"]
    for case, tau in s1["cube_tau"].items():
        L.append(f"| {SHORT.get(case, case)} | {tau['point']} | {', '.join(f'{v:.2f}' for v in tau['per_trial_mean_nodefrag'])} | {tau['mean_nodefrag']:.2f} | {tau['median_h1']:.2f} | {tau['median_h2']:.2f} | {tau['median']:.2f} | "
                 f"{tau['mean_nodefrag'] * 1e6 / common.particles_of(case):.3f} |")
    L += ["", "缺：η' 的 2M / 1M / 0.5M 立方体（16M / 8M / 4M K8 的每卡尺寸）没有算例；128M K8（对 16M）等 e8_C 跑完后补。", ""]
    # ---- 2
    L += ["## 2. K = 1 成本随规模", "", "### 2.1 五个立方体的 8 个并发参考：ns/粒子·步（按卡；去 defrag 均值 / N_total）", "",
          "| 立方体 | 试验 | 卡 0 | 卡 1 | 卡 2 | 卡 3 | 卡 4 | 卡 5 | 卡 6 | 卡 7 | 卡间最大相对差 (max−min)/min | 最慢 / 最快卡 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in CUBES:
        data = s2r.get(case)
        if not data:
            continue
        for entry in data["per_trial"]:
            cells = " | ".join(f"{entry['cards'][k]['ns']:.3f}" if k in entry["cards"] else "-" for k in range(8))
            L.append(f"| {SHORT[case]} | {entry['trial']} | {cells} | {pct(entry['spread'], 2)} | g{entry['max_card']} / g{entry['min_card']} |")
        cells = " | ".join(f"{data['card_mean'][k]:.3f}" if k in data["card_mean"] else "-" for k in range(8))
        L.append(f"| {SHORT[case]} | **均值** | {cells} | 均 {pct(data['spread_mean'], 2)}，最大 {pct(data['spread_max'], 2)} | 8 卡平均 **{data['ns_mean']:.3f}** |")
    L += ["", "按前后半窗（中位数口径，8 卡平均）：", "", "| 立方体 | ns h1（500–999） | ns h2（1000–1499） | h2 / h1 |", "|---|---|---|---|"]
    for case in CUBES:
        data = s2r.get(case)
        if not data:
            continue
        h1 = statistics.fmean(c["ns_h1"] for e in data["per_trial"] for c in e["cards"].values())
        h2 = statistics.fmean(c["ns_h2"] for e in data["per_trial"] for c in e["cards"].values())
        L.append(f"| {SHORT[case]} | {h1:.3f} | {h2:.3f} | {h2 / h1:.4f} |")
    L += ["", "### 2.2 五个立方体的 K = 1 trace：三段各自的 ns/粒子·步（稳态步 ≥ 500、去 defrag 步的均值）", "",
          "| 立方体 | trace | 步数 | A (ms) | B (ms) | C (ms) | busy (ms) | 周期 (ms) | A ns | B ns | C ns | busy ns | 周期 ns | 每步 周期 − busy 中位 (ms) | A→B 等待 | B→C 等待 | C→A 等待 | busy h1 / h2 (ms) | fps (去 defrag) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in CUBES:
        t = s2t.get(case)
        if not t:
            continue
        if t.get("error"):
            L.append(f"| {SHORT[case]} | 失败: {t['error']} | | | | | | | | | | | | | | | | | |")
            continue
        s = t["sim"]
        L.append(f"| {SHORT[case]} | {t['trace']} | {s['steps']} | {f(s['A_ms'])} | {f(s['B_ms'])} | {f(s['C_ms'])} | {f(s['busy_ms'])} | {f(s['period_ms'])} | "
                 f"{f(t['A_ns'])} | {f(t['B_ns'])} | {f(t['C_ns'])} | {f(t['busy_ns'])} | {f(t['period_ns'])} | {f(s['gap_ms'])} | {f(s['wait_AB_ms'])} | {f(s['wait_BC_ms'])} | {f(s['wait_CA_ms'])} | "
                 f"{f(s['busy_h1_ms'])} / {f(s['busy_h2_ms'])} | {f(t['fps_nodefrag'], 2)} |")
    L += ["", "（K = 1 没有 ghost / 链路：A = predict + voxel，B = 深内部核（correction_density_interior、force_deep_interior），C = 边界带核（K = 1 几乎为空）。"
          "每步的周期 − busy 只有 0.01–0.06 ms：GPU 饱和，主机侧没有空隙。）", ""]
    # 2.3
    L += ["### 2.3 阶段 1 单独跑的 8M K = 1 对 B 线并发的 8M 参考", ""]
    p1, lb, same = s2p.get("phase1"), s2p.get("line_b"), s2p.get("line_b_same_window")
    L += ["| 运行 | 节点 | 卡 | 窗口 | ns/粒子·步 | SM 时钟中位 (MHz) | SM 时钟最小 | 功耗中位 (W) | 最高温度 (°C) | 主机提交 submit_us 中位 | 主机等待 wait_us 中位 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    if p1:
        L.append(f"| 阶段 1 {p1['label']}（单独，2 卡作业） | {p1.get('node') or '-'} | g{p1['card']} | 第 100–299 步 | **{f(p1['ns_per_particle'])}** | {f(p1['sm_clock_median'], 0)} | {f(p1['sm_clock_min'], 0)} | "
                 f"{f(p1['power_median'], 1)} | {f(p1['temperature_max'], 0)} | 缺（阶段 1 没开 loop trace） | 缺 |")
    else:
        L.append("| 阶段 1 8M K1 | 缺（phase-1 results.jsonl 里没找到） | | | | | | | | | |")
    if same:
        L.append(f"| B 线 8M 参考 {same['label']}（8 卡并发） | {host} | g0 | **同一窗口 第 100–299 步** | **{f(same['ns_per_particle'])}** | （同下行） | | | | | |")
    if lb:
        for card, v in lb.items():
            L.append(f"| B 线 8M 参考 g{card}（5 次试验均值，8 卡并发） | {host} | g{card} | 第 500–1499 步 | {f(v['ns'])}（h1 {f(v['ns_h1'])} / h2 {f(v['ns_h2'])}） | {f(v['sm_clock_median'], 0)} | {f(v['sm_clock_min'], 0)} | "
                     f"{f(v['power_median'], 1)} | {f(v['temperature_max'], 0)} | {f(v['submit_us'], 1)} | {f(v['wait_us'], 1)} |")
    L += ["", "判断用的数据：同一窗口（第 100–299 步）下 B 线并发的 g0 与阶段 1 单独的一张卡每粒子成本一致与否，直接说明\"6 %\"是不是窗口位置（流动发展 + 第 1000 帧 defrag 之后的 5 %）造成的；"
          "SM 时钟中位数之差给时钟能解释的上限；wait_us ≫ submit_us（主机线程几乎全程在等 GPU）和 2.2 的\"周期 − busy ≈ 0\"排除主机争用。", ""]
    # 2.4
    L += ["### 2.4 逐核计时：8M K1 对 64M K1（K = 1 trace；每核的 ns/粒子·步 = 核时长中位数 / N_total；括号内按流体粒子数 N_fluid）", "",
          "| 核 | 段 | 8M µs/步 | 8M ns/粒子 (ns/流体粒子) | 64M µs/步 | 64M ns/粒子 (ns/流体粒子) | 64M / 8M（/粒子） | 64M / 8M（/流体粒子） |", "|---|---|---|---|---|---|---|---|"]
    k8, k64 = s2k.get("cavity3d_n192") or {}, s2k.get("cavity3d_n392") or {}
    if k8.get("error") or k64.get("error") or not k8 or not k64:
        L.append(f"| 缺 | {k8.get('error') or k64.get('error') or 'trace 不在'} | | | | | | |")
    else:
        n8, n64 = 8_000_000, 64_000_000
        fl8, fl64 = common.lattice_of("cavity3d_n192")["fluid_total"], common.lattice_of("cavity3d_n392")["fluid_total"]
        order = list(k8["kernel_order"]) + [k for k in k64["kernel_order"] if k not in k8["kernel_order"]]
        total8 = total64 = 0.0
        for name in order:
            a, b = k8["mean_over_sims"].get(name), k64["mean_over_sims"].get(name)
            us_a, us_b = (a["us_p50_mean"] if a else 0.0), (b["us_p50_mean"] if b else 0.0)
            total8 += us_a
            total64 += us_b
            L.append(f"| {name} | {(a or b)['phase']} | {f(us_a, 1)} | {f(us_a * 1e3 / n8)} ({f(us_a * 1e3 / fl8)}) | {f(us_b, 1)} | {f(us_b * 1e3 / n64)} ({f(us_b * 1e3 / fl64)}) | "
                     f"{f((us_b / n64) / (us_a / n8)) if us_a and us_b else '-'} | {f((us_b / fl64) / (us_a / fl8)) if us_a and us_b else '-'} |")
        L.append(f"| **核之和** | | {total8:,.1f} | {total8 * 1e3 / n8:.3f} ({total8 * 1e3 / fl8:.3f}) | {total64:,.1f} | {total64 * 1e3 / n64:.3f} ({total64 * 1e3 / fl64:.3f}) | "
                 f"{(total64 / n64) / (total8 / n8):.3f} | {(total64 / fl64) / (total8 / fl8):.3f} |")
        p8 = (s2t.get("cavity3d_n192") or {}).get("sim", {}).get("period_ms")
        p64 = (s2t.get("cavity3d_n392") or {}).get("sim", {}).get("period_ms")
        if p8 and p64:
            L.append(f"| **trace 周期（去 defrag 步的均值，2.2 节）** | | {p8 * 1e3:,.1f} | {p8 * 1e6 / n8:.3f} ({p8 * 1e6 / fl8:.3f}) | {p64 * 1e3:,.1f} | "
                     f"{p64 * 1e6 / n64:.3f} ({p64 * 1e6 / fl64:.3f}) | {(p64 / n64) / (p8 / n8):.3f} | {(p64 / fl64) / (p8 / fl8):.3f} |")
        L.append("")
        L.append(f"粒子构成：8M 流体 {fl8:,}（{fl8 / n8 * 100:.1f} %）、壁面 + 盖板 {n8 - fl8:,}；64M 流体 {fl64:,}（{fl64 / n64 * 100:.1f} %）、壁面 + 盖板 {n64 - fl64:,}。"
                 "壁面 / 盖板粒子不做 predict 积分与 PST，但在 correction_density / force 里作为邻居被读；按流体粒子归一时两个尺寸的差更大，说明差别不在壁面份额。"
                 "（中位数口径：每核的中位数同样受第 1000 帧后的 5 % 影响，两个尺寸受影响方式相同。）")
    L += ["", "### 2.5 每粒子候选数 / h 内邻居数", "",
          "缺：v7 求解器没有逐粒子的候选数或邻居数统计（没有这样的开关、计数器或 trace 字段；只有与此无关的 deep-wall candidates 壁面标记）。可用的替代：两个尺寸用同一个 h/dx = 4 的格点，"
          "每个 voxel 恰好 4³ = 64 个粒子、每个内部粒子的候选数（27 个 voxel）由构造相同；只有壁面附近不同——距壁 1 h（4 dx）以内的流体份额 8M 1 − (184/192)³ = 12.0 %，64M 1 − (384/392)³ = 6.0 %；"
          "壁面 + 盖板占总数 8M 11.5 %、64M 5.9 %（2.4 节已按两种归一给出）。", ""]
    # 2.6
    L += ["### 2.6 每步步长的时间序列：每 100 步的均值（ms），归一到第 100–199 步", ""]
    keys = [k for k in ("cavity3d_n192", "cavity3d_n392", "cavity3d_n152", "cavity3d_n244", "cavity3d_n312", "32M K8 链路", "64M K8 链路") if k in s2s]
    if keys:
        L.append("| 步段 | " + " | ".join(f"{s2s[k]['title']} | 归一" for k in keys) + " |")
        L.append("|---|" + "---|---|" * len(keys))
        bins = s2s[keys[0]]["bins"]
        for index in range(len(bins)):
            cells = []
            for k in keys:
                b = s2s[k]["bins"][index] if index < len(s2s[k]["bins"]) else {}
                cells.append(f"{f(b.get('mean_kept'), 2)}{' (去 ' + str(b['removed']) + ' 步)' if b.get('removed') else ''} | {f(b.get('normalised'), 4)}")
            L.append(f"| {bins[index]['bin']} | " + " | ".join(cells) + " |")
        L += ["", "（去掉的步 = 步长 > 2 × 该运行整体中位数的步 = 第 1000 帧的 defrag 步；运行：" + "; ".join(f"{s2s[k]['title']} = {s2s[k]['label']}" for k in keys) + "）", ""]
    else:
        L += ["缺：raw loop trace 不在。", ""]
    # ---- 3
    L += ["## 3. 切口与不平衡（16M / 32M / 64M K8：两轮 pilot 与最终分区）", ""]
    for point, data in s3.items():
        L += [f"### {SHORT.get(data['case'], data['case'])} K8（流体 {data['fluid_columns_total']} 列，每卡 {data['per_card']:.2f}）", "",
              "| 轮 | 切口 | own 列（端 slab 含 2 壁面列） | 流体列 | own 粒子（含壁） | pilot busy (ms) | ω | 下一轮切口 | 切口变了 | pilot fps |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in data["rounds"]:
            L.append(f"| {r['round']} | {r['cuts']} | {r['own_columns']} | {r['fluid_columns']} | {r['all']} | {[round(v / 1e3, 2) for v in r['busy_us']] if r['busy_us'] else '-'} | "
                     f"{[round(v, 4) for v in r['omega']] if r['omega'] else '-'} | {r['next_cuts']} | {'是' if r['changed'] else '否'} | {f(r['pilot_fps'], 2)} |")
        if data["final"]:
            slabs = data["final"]["slabs"]
            L.append(f"| 最终（计时运行） | {data['final']['cuts']} | {[s['own_columns'] for s in slabs]} | {[s['fluid_columns'] for s in slabs]} | {[s['own_particles'] for s in slabs]} | （无 pilot） | "
                     f"{[round(w, 4) for w in data['final_weights']]} | | | |")
            hi, lo = max(x["fluid_columns"] for x in slabs), min(x["fluid_columns"] for x in slabs)
            extra = [s["slab"] for s in slabs if s["fluid_columns"] == hi]
            fewer = [s["slab"] for s in slabs if s["fluid_columns"] == lo]
            ends = {0, len(slabs) - 1}
            L += ["", f"多的列（{hi} 列）在 slab {extra}（端 {sorted(set(extra) & ends)}，内部 {sorted(set(extra) - ends)}）；少的（{lo} 列）在 slab {fewer}（端 {sorted(set(fewer) & ends)}，内部 {sorted(set(fewer) - ends)}）。"
                  f"own 粒子最大 / 最小 = {max(s['own_particles'] for s in slabs) / min(s['own_particles'] for s in slabs):.3f}。"
                  + ("只做了一轮 pilot：第一轮的 ω 没有移动切口。" if len(data["rounds"]) == 1 else
                     ("最终切口 = 第 2 轮的 next_cuts，与第 2 轮 pilot 实测的切口不同（最终分区没有被 pilot 过）。" if data["rounds"][-1]["next_cuts"] != data["rounds"][-1]["cuts"] else "")), ""]
        else:
            L.append("")
    if not s3:
        L += ["缺：weights/ 里没有标定记录。", ""]
    # ---- 4
    L += ["## 4. 链路各卡的相位（trace；A / B / C / busy / 周期 = 稳态步去 defrag 步的均值，等待 = 每步的中位数，ms）", ""]
    for point, data in s4.items():
        if point == "3d_n192_k8_K8":
            continue
        if data.get("error"):
            L += [f"### {point}: 失败 {data['error']}", ""]
            continue
        L += [f"### {SHORT.get(data['case'], point)} K8（{data['name']}；fps 去 defrag {f(data.get('fps_nodefrag'), 2)}）", "",
              "| sim (卡) | own 列 / 流体列 | seam 数 | own 粒子 | A | B | C | busy = A+B+C | ns/own 粒子 | A→B 等待 | B→C 等待 | C→A 等待 | 周期 − busy（每步中位） | 周期 | busy h1 / h2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in data["sims"]:
            L.append(f"| {s['sim']} ({s['device']}) | {s['own_columns']} / {s['fluid_columns']} | {s['seams']} | {f(s['own_particles'], 0)} | {f(s['A_ms'])} | {f(s['B_ms'])} | {f(s['C_ms'])} | "
                     f"**{f(s['busy_ms'])}** | {f(s['ns_per_own'])} | {f(s['wait_AB_ms'])} | {f(s['wait_BC_ms'])} | {f(s['wait_CA_ms'])} | {f(s['gap_ms'])} | {f(s['period_ms'])} | "
                     f"{f(s['busy_h1_ms'], 2)} / {f(s['busy_h2_ms'], 2)} |")
        slow = data["slowest"]
        L += ["", f"最慢的卡：sim {slow['sim']}（卡 {slow['device']}），busy {slow['busy_ms']:.3f} ms，own 列 {slow['own_columns']}（流体 {slow['fluid_columns']}），seam {slow['seams']} 个，"
              f"{f(slow['ns_per_own'])} ns/own 粒子；链路周期（各卡平均）{data['period_mean_of_cards_ms']:.3f} ms，**周期 − 最慢卡 busy = {data['period_minus_slowest_busy_ms']:.3f} ms**"
              f"（{data['period_minus_slowest_busy_ms'] / data['period_mean_of_cards_ms'] * 100:.1f} %）；各卡 busy 平均 {data['busy_mean_of_cards_ms']:.3f} ms，"
              f"平衡度 busy 平均 / 周期 = **{data['balance']:.3f}**；每 own 粒子 busy（各卡平均）**{f(data['ns_per_own_mean'])} ns**，"
              f"对比 K = 1：同总规模 {f(data['k1_ns_same_size'])} ns（比值 {f(data['k1_ns_same_size'] / data['ns_per_own_mean']) if data['ns_per_own_mean'] and data['k1_ns_same_size'] else '-'}），"
              f"每卡尺寸的立方体 {f(data['k1_ns_per_card_size']) if data['k1_ns_per_card_size'] else '缺'} ns"
              f"（比值 {f(data['k1_ns_per_card_size'] / data['ns_per_own_mean']) if data['ns_per_own_mean'] and data['k1_ns_per_card_size'] else '缺'}）。", ""]
    if not any(p != "3d_n192_k8_K8" for p in s4):
        L += ["缺：traces/ 里没有 K8 的 trace。", ""]
    L += ["### 4b. 链路各卡的逐核成本（ns / own 粒子，核时长中位数 / own 粒子数；8 卡的 min / 平均 / max）对 K = 1 trace（ns / N_total）", ""]
    for point, data in s4k.items():
        if data.get("error"):
            L += [f"{point}: 失败 {data['error']}", ""]
            continue
        L += [f"**{SHORT.get(data['case'], point)} K8**（own 粒子 {data['sims_own']}；G1 冗余份额 = 边界带核里为对方 ghost 算的部分 / 步长：平均 {pct(data['g1_share_mean'])}，"
              f"范围 {pct(data['g1_share_range'][0]) if data.get('g1_share_range') else '-'}–{pct(data['g1_share_range'][1]) if data.get('g1_share_range') else '-'}）", "",
              "| 核 | 段 | K = 1 ns/粒子 | K8 min | K8 平均 | K8 max | K8 平均 / K1 | K8 各卡 µs/步 |", "|---|---|---|---|---|---|---|---|"]
        for r in data["rows"]:
            ratio = f(r["k8_ns_mean"] / r["k1_ns"]) if r.get("k1_ns") and r["k1_ns"] >= 0.1 and r.get("k8_ns_mean") else "-"
            name = f"**{r['kernel']}**" if r.get("combo") else r["kernel"]
            L.append(f"| {name} | {r['phase']} | {f(r['k1_ns'])} | {f(r['k8_ns_min'])} | {f(r['k8_ns_mean'])} | {f(r['k8_ns_max'])} | {ratio} | {r['k8_us_per_card'] or ''} |")
        L += ["", "（K = 1 的边界带核几乎为空（0.2 µs），单核比值无意义，看合并行：K = 1 的 interior 核对 K8 的 interior + boundary；K8 的 ns 按各卡 own 粒子数归一，"
              "ghost 粒子上的计算（G1 冗余）算在 own 头上。）", ""]
    # ---- 5
    L += ["## 5. 链路的逐步分布（每次计时运行，第 500–1499 步的步长，ms）", "",
          "| 点 | 运行 | 步数 | 中位数 | 中位 h1 | 中位 h2 | 均值 | 均值(去 defrag) | p95 | p99 | 最大 | > 2 × 中位数的步数 | defrag 墙钟 (ms) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for point, entries in s5.items():
        for e in entries:
            if not e.get("n"):
                L.append(f"| {point} | {e['label']} | 缺 raw | | | | | | | | | | |")
                continue
            L.append(f"| {SHORT.get(common.case_of_point(point)[0], point)} K8 | {e['trial']} ({e['label']}) | {e['n']} | {f(e['median'])} | {f(e['median_h1'])} | {f(e['median_h2'])} | {f(e['mean'])} | {f(e['mean_nodefrag'])} | "
                     f"{f(e['p95'])} | {f(e['p99'])} | {f(e['max'])} | {e['above_2x_median']} | {f(e['defrag_ms'], 1)} |")
    L.append("")
    # ---- 6
    L += ["## 6. 对照：弱 K8（每卡 8M，w = 48）对 64M K8（每卡 8M，w = 12.25）", ""]
    weak, big = s4.get("3d_n192_k8_K8"), s4.get("3d_n392_K8")
    p64 = s1["points"].get("3d_n392_K8", {})
    if weak and not weak.get("error") and big and not big.get("error"):
        eta64 = [t.get("eta_mean_nodefrag") for t in p64.get("trials", []) if t.get("eta_mean_nodefrag")]
        eta_prime = [t.get("eta_prime_mean_nodefrag") for t in p64.get("trials", []) if t.get("eta_prime_mean_nodefrag")]
        L += ["| 点 | w（流体列 / 卡） | 参考 | η（nd） | η'（nd） | 各卡 busy（ms） | busy 平均 | 最慢卡 busy | 周期 | 周期 − 最慢 busy | 平衡度 | ns/own 粒子（平均） | K = 1 8M ns |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        L.append(f"| 64M K8 | 12.25 | 64M K1 ×8（η）/ 8M K1 ×8（η'） | {f(statistics.fmean(eta64)) if eta64 else '-'} | {f(statistics.fmean(eta_prime)) if eta_prime else '缺'} | "
                 f"{', '.join(f'{s["busy_ms"]:.2f}' for s in big['sims'])} | {f(big['busy_mean_of_cards_ms'])} | {f(big['slowest']['busy_ms'])} | "
                 f"{f(big['period_mean_of_cards_ms'])} | {f(big['period_minus_slowest_busy_ms'])} | {f(big['balance'])} | {f(big['ns_per_own_mean'])} | {f(big['k1_ns_per_card_size'])} |")
        L.append(f"| 弱 8 × 8M K8 | 48 | 8M K1 ×8（η = fps_K / mean fps_ref，与 η' 同口径） | {f(weak.get('eta_weak'))}（各试验 {', '.join(f'{v:.3f}' for v in weak.get('eta_weak_per_trial', []))}） | （同左） | "
                 f"{', '.join(f'{s["busy_ms"]:.2f}' for s in weak['sims'])} | {f(weak['busy_mean_of_cards_ms'])} | {f(weak['slowest']['busy_ms'])} | "
                 f"{f(weak['period_mean_of_cards_ms'])} | {f(weak['period_minus_slowest_busy_ms'])} | {f(weak['balance'])} | {f(weak['ns_per_own_mean'])} | {f(weak['k1_ns_per_card_size'])} |")
        L.append("")
    else:
        L += ["缺：弱 8 × 8M K8 的 trace / 计时运行还没进快照（B 作业的最后一个点），作业结束后重跑本脚本补上。", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase1", default=None, help="the phase-1 data directory (results.jsonl of the standalone 8M K = 1)")
    parser.add_argument("--md", required=True)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    job = common.load_job(arguments.job)
    points = point_runs(job)
    s1 = section1(job, points)
    s2r = section2_references(job, points)
    s2t = section2_traces(job)
    s2s = section2_series(job, points)
    s2p = section2_phase1(arguments.phase1, s2r, s2s)
    s2k = section2_kernels(job)
    s3 = section3(job)
    s4 = section4(job, s2r, s2t)
    weak = points.get("3d_n192_k8_K8") or {}
    etas = []
    for trial, runs in weak.items():
        chain = common.fps_pair(job["rows"].get(runs["C"]))[1] if runs["C"] else None
        refs = [common.fps_pair(job["rows"].get(label))[1] for label in runs["refs"].values()]
        refs = [v for v in refs if v]
        if chain and len(refs) == 8:
            etas.append(chain / statistics.fmean(refs))
    if "3d_n192_k8_K8" in s4 and not s4["3d_n192_k8_K8"].get("error"):
        s4["3d_n192_k8_K8"]["eta_weak"] = statistics.fmean(etas) if etas else None
        s4["3d_n192_k8_K8"]["eta_weak_per_trial"] = etas
    s4k = section4_kernels(job, s2k)
    s5 = section5(job, points)
    text = markdown(job, s1, s2r, s2t, s2p, s2k, s2s, s3, s4, s4k, s5)
    common.write_text(arguments.md, text)
    if arguments.json:
        common.write_json(arguments.json, {"job": job["name"], "section1": s1, "section2_references": s2r, "section2_traces": s2t, "section2_phase1": s2p,
                                           "section2_kernels": s2k, "section2_series": s2s, "section3": s3, "section4": s4, "section4_kernels": s4k, "section5": s5})
    print(f"[e8_b_diag] wrote {arguments.md}" + (f" and {arguments.json}" if arguments.json else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
