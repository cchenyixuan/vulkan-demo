"""
step_trace_model.py — E29 analysis of the step traces (experiment/v6/utils/phase_trace_v6.StepTracer).

Per run (one --step-trace directory, steady steps = step >= warmup):
  checks   causal order of every link's chain on the common host clock, coverage
           (steps with complete records), readback / upload DMA medians;
  per link and step
           t_tr = upload_end - send_end, tiled into hops (HOPS below);
           t_chain = t_tr minus the two waits on the receiver (its readback_done(n)
           and its previous upload): the transport chain itself, without the
           phase coupling (the worker of a link into a lagging GPU waits for that
           GPU's own readback);
           T_B (receiver), phase offset D' = receiver b_start - send_end,
           exposure = max(0, upload_end - receiver b_end), B->C wait =
           receiver c_start - receiver b_end;
           r = t_tr / T_B (the task's definition, includes the phase coupling),
           r_chain = t_chain / T_B, r_receiver = (upload_end - receiver b_start) / T_B
           (> 1 = exposed); each per step on the worse link;
  per sim  A-start gating = a_start(n) - c_end(n - 1).
Per campaign (step_trace_campaign.py --mode scan output):
  eta = fps_K2 / (2 mean(fps_K1 GPU 0, GPU 1)) from the runs WITHOUT the trace (also
  with it, to show the instrument's effect); c_B from the traced K = 1 runs
  (T_B = c_B N, one line through the origin per dimension); from the traced K = 2
  runs: t_tr p50 = tau0 + B_host / beta over both links of every run (as specified,
  includes the phase coupling) and the same line for t_chain (used for r_pred),
  plus one line per hop of the chain (readback vs DMA bytes, host vs host bytes,
  upload vs DMA bytes); r_pred = (tau0 + B_host / beta) / (c_B n_B) and the closed
  form tau0 / (c_B n_B) + 64 / (c_B beta (w - b)); tables + 4 figures.

    .venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_model \\
        --scan logs/e29_step_trace/scan --overhead logs/e29_step_trace/overhead --out docs/perf_model/e29
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

# The chain of one link on the host clock (each must not precede the previous one).
CAUSAL_CHAIN = ("send_end", "readback_start", "readback_end", "worker_source_wait", "worker_copy",
                "upload_start", "upload_end", "receiver_c_start")
# Which clock each chain point is on: the sender's GPU, the host, the receiver's GPU.
CHAIN_DOMAIN = {"send_end": "sender", "readback_start": "sender", "readback_end": "sender",
                "worker_source_wait": "host", "worker_copy": "host", "upload_start": "receiver",
                "upload_end": "receiver", "receiver_c_start": "receiver"}
# t_tr = upload_end - send_end, tiled without overlap (name, start column, end column).
HOPS = (("readback_start_delay", "send_end", "readback_start"),
        ("readback_dma", "readback_start", "readback_copy_end"),
        ("readback_barrier", "readback_copy_end", "readback_end"),
        ("readback_to_worker", "readback_end", "worker_source_wait"),
        ("wait_receiver_readback", "worker_source_wait", "worker_dest_guard"),
        ("wait_receiver_previous_upload", "worker_dest_guard", "worker_upload_guard"),
        ("stamp_check", "worker_upload_guard", "worker_stamp"),
        ("memcpy", "worker_stamp", "worker_copy"),
        ("signal", "worker_copy", "worker_signal"),
        ("signal_to_upload", "worker_signal", "upload_start"),
        ("upload_dma", "upload_start", "upload_end"))
# The task's five groups: readback, readback end -> worker sees, host segment (two waits on the
# receiver, stamp check, memcpy), copy end -> upload start (signal + queue start), upload.
HOP_GROUPS = (("readback", ("readback_start_delay", "readback_dma", "readback_barrier")),
              ("readback_to_worker", ("readback_to_worker",)),
              ("host", ("wait_receiver_readback", "wait_receiver_previous_upload", "stamp_check", "memcpy")),
              ("copy_to_upload", ("signal", "signal_to_upload")),
              ("upload", ("upload_dma",)))
# The chain without the two waits on the receiver, in three hops for the per-hop fits.
RECEIVER_WAITS = ("wait_receiver_readback", "wait_receiver_previous_upload")
CHAIN_GROUPS = (("readback", ("readback_start_delay", "readback_dma", "readback_barrier")),
                ("host", ("readback_to_worker", "stamp_check", "memcpy", "signal")),
                ("upload", ("signal_to_upload", "upload_dma")))
# The host step (worker), E24's six segments.
HOST_SEGMENTS = (("wait_sender_readback", "worker_dequeue", "worker_source_wait"),
                 ("wait_receiver_readback", "worker_source_wait", "worker_dest_guard"),
                 ("wait_receiver_previous_upload", "worker_dest_guard", "worker_upload_guard"),
                 ("stamp_check", "worker_upload_guard", "worker_stamp"),
                 ("memcpy", "worker_stamp", "worker_copy"),
                 ("signal", "worker_copy", "worker_signal"))
# E24 depth-1 anatomy p50 (us) of build 21ce20d with bands 2/2/3 (v6_opt.md "host 步拆分"):
# (readback s0->s1, readback s1->s0, upload s0->s1, upload s1->s0).
E24_DMA_P50 = {"2d_1m": (19.2, 19.0, 30.5, 28.0), "2d_16m": (70.1, 70.9, 102.1, 99.7),
               "3d_8m": (908.5, 911.4, 1002.5, 992.8)}
TEXT_COLUMNS = ("link", "sender_direction", "receiver_direction")
RATIOS = ("r", "r_chain", "r_receiver")


def read_table(path: pathlib.Path) -> dict:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    table = {}
    for index, name in enumerate(header):
        if name in TEXT_COLUMNS:
            table[name] = np.array([row[index] for row in rows])
        else:
            table[name] = np.array([float(row[index]) if row[index] != "" else np.nan for row in rows])
    return table


def load_run(run_dir) -> dict:
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "meta": meta, "device": read_table(run_dir / "steps_device.csv"),
            "link": read_table(run_dir / "steps_link.csv")}


def quantiles(values, points=(50, 95)) -> list:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return [float(np.percentile(values, p)) if len(values) else float("nan") for p in points]


def select(table: dict, mask) -> dict:
    return {name: column[mask] for name, column in table.items()}


def link_metrics(run: dict) -> dict:
    """Per link: steady-step arrays (us) of t_tr, its hops, t_chain, T_B, D', exposure, B->C wait,
    the three ratios, bytes."""
    warmup = run["meta"]["warmup"]
    link = run["link"]
    out = {}
    for name in sorted(set(link["link"])):
        rows = select(link, (link["link"] == name) & (link["step"] >= warmup) & (link["complete"] == 1))
        order = np.argsort(rows["step"])
        rows = {key: value[order] for key, value in rows.items()}
        metrics = {"step": rows["step"], "t_tr": (rows["upload_end"] - rows["send_end"]) / 1e3,
                   "T_B": (rows["receiver_b_end"] - rows["receiver_b_start"]) / 1e3,
                   "phase_offset": (rows["receiver_b_start"] - rows["send_end"]) / 1e3,
                   "exposure": np.maximum(0.0, rows["upload_end"] - rows["receiver_b_end"]) / 1e3,
                   "b_to_c_wait": (rows["receiver_c_start"] - rows["receiver_b_end"]) / 1e3,
                   "upload_end_after_b_start": (rows["upload_end"] - rows["receiver_b_start"]) / 1e3,
                   "host_copy_bytes": rows["host_copy_bytes"], "dma_bytes": rows["dma_bytes"],
                   "sender": int(rows["sender"][0]), "receiver": int(rows["receiver"][0])}
        for hop, start, end in HOPS:
            metrics[hop] = (rows[end] - rows[start]) / 1e3
        for group, members in HOP_GROUPS:
            metrics["group_" + group] = sum(metrics[member] for member in members)
        for group, members in CHAIN_GROUPS:
            metrics["chain_" + group] = sum(metrics[member] for member in members)
        for segment, start, end in HOST_SEGMENTS:
            metrics["host_" + segment] = (rows[end] - rows[start]) / 1e3
        metrics["t_chain"] = metrics["t_tr"] - sum(metrics[wait] for wait in RECEIVER_WAITS)
        metrics["r"] = metrics["t_tr"] / metrics["T_B"]
        metrics["r_chain"] = metrics["t_chain"] / metrics["T_B"]
        metrics["r_receiver"] = metrics["upload_end_after_b_start"] / metrics["T_B"]
        out[name] = metrics
    return out


def worst_link(metrics: dict, key: str) -> np.ndarray:
    """Per steady step, the larger value of ``key`` over the links (steps present on every link)."""
    common = None
    for values in metrics.values():
        steps = set(values["step"].astype(int))
        common = steps if common is None else common & steps
    if not common:
        return np.array([])
    steps = np.array(sorted(common))
    stacked = []
    for values in metrics.values():
        index = {int(step): position for position, step in enumerate(values["step"])}
        stacked.append(values[key][[index[step] for step in steps]])
    return np.max(np.vstack(stacked), axis=0)


def device_gating(run: dict) -> dict:
    """Per sim: a_start(n) - c_end(n - 1), steady steps, us."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    out = {}
    for sim in sorted(set(device["sim"].astype(int))):
        rows = select(device, device["sim"] == sim)
        order = np.argsort(rows["step"])
        steps, a_start, c_end = rows["step"][order], rows["a_start"][order], rows["c_end"][order]
        consecutive = steps[1:] == steps[:-1] + 1
        out[sim] = (a_start[1:] - c_end[:-1])[consecutive & (steps[1:] >= warmup)] / 1e3
    return out


CYCLE_PARTS = ("a", "a_to_b", "b", "b_to_c", "c", "c_to_next_a")


def cycle_components(run: dict) -> list:
    """Per sim, medians over steady steps (us): phase A, A->B gap, phase B, B->C wait, phase C and
    c_end(n) -> a_start(n + 1), and the step period a_start(n) -> a_start(n + 1). Medians do not add up to
    the median period exactly."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    out = []
    for sim in sorted(set(device["sim"].astype(int))):
        rows = select(device, (device["sim"] == sim) & (device["complete"] == 1))
        order = np.argsort(rows["step"])
        rows = {key: value[order] for key, value in rows.items()}
        steps = rows["step"]
        consecutive = np.concatenate([steps[1:] == steps[:-1] + 1, [False]])
        steady = (steps >= warmup) & consecutive
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        parts = {"a": rows["a_end"] - rows["a_start"], "a_to_b": rows["b_start"] - rows["a_end"],
                 "b": rows["b_end"] - rows["b_start"], "b_to_c": rows["c_start"] - rows["b_end"],
                 "c": rows["c_end"] - rows["c_start"], "c_to_next_a": next_a - rows["c_end"],
                 "period": next_a - rows["a_start"]}
        out.append({"sim": sim, **{key: float(np.median(value[steady]) / 1e3) for key, value in parts.items()}})
    return out


def causal_check(run: dict) -> dict:
    """Fraction of steady (step, link) chains whose consecutive points go backwards by more than the
    tolerance: 0 for points on one GPU (compute and transfer pools share the device clock and its
    map), the map's largest residual for a GPU <-> host pair."""
    meta = run["meta"]
    fits = meta["clock"]["fits"]
    link = run["link"]
    rows = select(link, (link["step"] >= meta["warmup"]) & (link["complete"] == 1))
    result = {"chains": int(len(rows["step"])), "pairs": []}
    any_violation = np.zeros(len(rows["step"]), dtype=bool)
    for earlier, later in zip(CAUSAL_CHAIN, CAUSAL_CHAIN[1:]):
        gap = rows[later] - rows[earlier]
        domains = (CHAIN_DOMAIN[earlier], CHAIN_DOMAIN[later])
        tolerance = np.zeros(len(gap))
        if domains[0] != domains[1]:
            for position, (sender, receiver) in enumerate(zip(rows["sender"].astype(int),
                                                              rows["receiver"].astype(int))):
                gpu = sender if "sender" in domains else receiver
                tolerance[position] = fits[gpu]["residual_max_ns"]
        violation = gap < -tolerance
        any_violation |= violation
        result["pairs"].append({"pair": f"{earlier} <= {later}", "domains": "/".join(domains),
                                "negative_fraction": float(np.mean(gap < 0)) if len(gap) else 0.0,
                                "violation_fraction": float(np.mean(violation)) if len(gap) else 0.0,
                                "min_gap_us": float(np.min(gap) / 1e3) if len(gap) else float("nan"),
                                "p50_gap_us": float(np.median(gap) / 1e3) if len(gap) else float("nan")})
    result["chains_with_violation_fraction"] = float(np.mean(any_violation)) if len(any_violation) else 0.0
    result["clock_fit_residual_max_ns"] = [fit["residual_max_ns"] for fit in fits]
    result["clock_fit_residual_rms_ns"] = [fit["residual_rms_ns"] for fit in fits]
    result["driver_max_deviation_min_ns"] = [fit.get("driver_max_deviation_min_ns") for fit in fits]
    return result


def coverage(run: dict) -> dict:
    meta = run["meta"]
    device, link = run["device"], run["link"]
    steps = sorted(set(device["step"].astype(int)))
    sims = len(set(device["sim"].astype(int)))
    links = len(set(link["link"])) if len(link["step"]) else 0
    complete_device, complete_link = {}, {}
    for step, complete in zip(device["step"].astype(int), device["complete"].astype(int)):
        complete_device[step] = complete_device.get(step, 0) + complete
    for step, complete in zip(link["step"].astype(int), link["complete"].astype(int)):
        complete_link[step] = complete_link.get(step, 0) + complete
    good = [step for step in steps if complete_device.get(step, 0) == sims and complete_link.get(step, 0) == links]
    expected = meta["max_steps"]
    steady = [step for step in good if step >= meta["warmup"]]
    return {"expected_steps": expected, "recorded_steps": len(steps), "complete_steps": len(good),
            "coverage": len(good) / expected if expected else float("nan"),
            "steady_coverage": len(steady) / (expected - meta["warmup"]) if expected > meta["warmup"] else float("nan")}


LINK_KEYS = (("t_tr", "t_chain", "T_B", "phase_offset", "exposure", "b_to_c_wait", "upload_end_after_b_start")
             + RATIOS + tuple(hop for hop, _, _ in HOPS) + tuple("group_" + group for group, _ in HOP_GROUPS)
             + tuple("chain_" + group for group, _ in CHAIN_GROUPS)
             + tuple("host_" + segment for segment, _, _ in HOST_SEGMENTS))


def run_summary(run: dict) -> dict:
    metrics = link_metrics(run)
    meta = run["meta"]
    slabs = (meta.get("slabs") or {}).get("per_sim", [])
    summary = {"case": meta["case"], "K": meta["K"], "steady_fps": meta["result"].get("steady_fps"),
               "coverage": coverage(run), "links": {}, "sims": []}
    for index, slab in enumerate(slabs):
        summary["sims"].append({"sim": index, "own_columns": slab["own_columns"], "n_B": slab["n_B"],
                                "own_particles": slab["own_particles"], "band_widths": slab["band_widths"]})
    gating = device_gating(run)
    device = run["device"]
    for sim in sorted(set(device["sim"].astype(int))):
        rows = select(device, (device["sim"] == sim) & (device["step"] >= meta["warmup"]) & (device["complete"] == 1))
        entry = next((item for item in summary["sims"] if item["sim"] == sim), None)
        if entry is None:
            entry = {"sim": sim}
            summary["sims"].append(entry)
        entry["T_B_p50_us"], entry["T_B_p95_us"] = quantiles((rows["b_end"] - rows["b_start"]) / 1e3)
        entry["phase_a_p50_us"] = quantiles((rows["a_end"] - rows["a_start"]) / 1e3)[0]
        entry["phase_c_p50_us"] = quantiles((rows["c_end"] - rows["c_start"]) / 1e3)[0]
        entry["gating_p50_us"], entry["gating_p95_us"] = quantiles(gating.get(sim, []))
    for name, values in metrics.items():
        entry = {"sender": values["sender"], "receiver": values["receiver"], "steps": int(len(values["step"])),
                 "host_copy_bytes_p50": float(np.median(values["host_copy_bytes"])),
                 "dma_bytes": float(np.median(values["dma_bytes"])),
                 "exposed_fraction": float(np.mean(values["exposure"] > 0))}
        for key in LINK_KEYS:
            entry[key + "_p50"], entry[key + "_p95"] = quantiles(values[key])
        summary["links"][name] = entry
    for key in RATIOS:
        worst = worst_link(metrics, key)
        summary[key + "_worst_p50"], summary[key + "_worst_p95"] = quantiles(worst)
    exposure = worst_link(metrics, "exposure")
    summary["exposed_steps_fraction"] = float(np.mean(exposure > 0)) if len(exposure) else float("nan")
    summary["exposure_mean_us"] = float(np.mean(exposure)) if len(exposure) else float("nan")
    if len(metrics) > 1:
        summary["checks"] = {"causal": causal_check(run)}
    return summary


# ------------------------------------------------------------------ campaign

def read_results(campaign: pathlib.Path) -> list:
    records = []
    for line in (campaign / "results.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not record.get("marker"):
            records.append(record)
    return records


def fit_line(x, y, through_origin=False) -> dict:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if through_origin:
        slope, intercept = float(np.sum(x * y) / np.sum(x * x)), 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    residual = y - (intercept + slope * x)
    return {"slope": slope, "intercept": intercept, "n": int(len(x)),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "relative_residual_rms": float(np.sqrt(np.mean((residual / y) ** 2))),
            "relative_residual_max": float(np.max(np.abs(residual / y)))}


def analyze_scan(scan: pathlib.Path) -> dict:
    records = [r for r in read_results(scan) if r.get("rc") == 0]
    cases = []
    for name in dict.fromkeys(record["case"] for record in records):
        def pick(kind, trace):
            return [r for r in records if r["case"] == name and r["kind"] == kind and bool(r.get("trace")) == trace]
        k1_off, k2_off, k1_on, k2_on = pick("k1", False), pick("k2", False), pick("k1", True), pick("k2", True)
        if len(k1_off) != 2 or len(k2_off) != 1 or len(k1_on) != 2 or len(k2_on) != 1:
            print(f"[step_trace_model] {name}: incomplete ({len(k1_off)}/{len(k2_off)}/{len(k1_on)}/{len(k2_on)})",
                  file=sys.stderr)
            continue
        k1_runs = [load_run(scan / r["trace_dir"]) for r in k1_on]
        k2_run = load_run(scan / k2_on[0]["trace_dir"])
        cycles = {"k1": [cycle_components(run)[0] for run in k1_runs], "k2": cycle_components(k2_run)}
        k1_summaries = [run_summary(run) for run in k1_runs]
        fps_k1_off = [r["steady_fps"] for r in k1_off]
        fps_k1_on = [r["steady_fps"] for r in k1_on]
        k1_c_b = [{"gpu": r["gpu"], "N": s["sims"][0]["n_B"], "T_B_p50_us": s["sims"][0]["T_B_p50_us"],
                   "c_B_ns": 1e3 * s["sims"][0]["T_B_p50_us"] / s["sims"][0]["n_B"]}
                  for r, s in zip(k1_on, k1_summaries)]
        cases.append({"case": name, "dimension": k2_on[0]["dimension"],
                      "fps_k1": fps_k1_off, "fps_k2": k2_off[0]["steady_fps"],
                      "eta": k2_off[0]["steady_fps"] / (2.0 * float(np.mean(fps_k1_off))),
                      "fps_k1_traced": fps_k1_on, "fps_k2_traced": k2_on[0]["steady_fps"],
                      "eta_traced": k2_on[0]["steady_fps"] / (2.0 * float(np.mean(fps_k1_on))),
                      "k1": k1_c_b, "k2": run_summary(k2_run), "k2_run": k2_run, "cycles": cycles})
    return {"cases": cases}


def constants(scan_result: dict) -> dict:
    out = {"c_B": {}, "transport": {}, "transport_spec": {}, "hops": {}}
    for dimension in (2, 3):
        points = [(item["N"], item["T_B_p50_us"]) for case in scan_result["cases"] if case["dimension"] == dimension
                  for item in case["k1"]]
        if points:
            fit = fit_line([p[0] for p in points], [p[1] for p in points], through_origin=True)
            out["c_B"][dimension] = {"c_B_ns": 1e3 * fit["slope"], **fit}
    t_tr, t_chain, host_bytes, dma_bytes = [], [], [], []
    hop_values = {group: [] for group, _ in CHAIN_GROUPS}
    for case in scan_result["cases"]:
        for link in case["k2"]["links"].values():
            t_tr.append(link["t_tr_p50"])
            t_chain.append(link["t_chain_p50"])
            host_bytes.append(link["host_copy_bytes_p50"])
            dma_bytes.append(link["dma_bytes"])
            for group, _ in CHAIN_GROUPS:
                hop_values[group].append(link["chain_" + group + "_p50"])
    if t_chain:
        for key, values in (("transport", t_chain), ("transport_spec", t_tr)):
            fit = fit_line(host_bytes, values)
            out[key] = {"tau0_us": fit["intercept"],
                        "beta_GBps": 1e-3 / fit["slope"] if fit["slope"] > 0 else float("nan"), **fit}
        for group, x in (("readback", dma_bytes), ("host", host_bytes), ("upload", dma_bytes)):
            fit = fit_line(x, hop_values[group])
            out["hops"][group] = {"intercept_us": fit["intercept"],
                                  "bandwidth_GBps": 1e-3 / fit["slope"] if fit["slope"] > 0 else float("nan"), **fit}
    return out


def predictions(scan_result: dict, model: dict) -> None:
    tau0, slope = model["transport"]["tau0_us"], model["transport"]["slope"]   # us, us per byte (= 1 / beta)
    for case in scan_result["cases"]:
        c_b = model["c_B"][case["dimension"]]["slope"]                       # us per particle
        per_link = []
        for name, link in case["k2"]["links"].items():
            receiver = case["k2"]["sims"][link["receiver"]]
            n_b, w, b = receiver["n_B"], receiver["own_columns"], receiver["band_widths"][2]
            t_b_pred = c_b * n_b
            case_c_b = float(np.mean([item["c_B_ns"] for item in case["k1"]])) / 1e3   # this case's K = 1 T_B / N
            per_link.append({"link": name, "n_B": n_b, "w": w, "b": b,
                             "r_pred": (tau0 + slope * link["host_copy_bytes_p50"]) / t_b_pred,
                             "r_pred_case_c_B": (tau0 + slope * link["host_copy_bytes_p50"]) / (case_c_b * n_b),
                             "r_closed_form": tau0 / t_b_pred + 64.0 * slope / (c_b * (w - b)),
                             "T_B_pred_us": t_b_pred, "T_B_measured_us": receiver["T_B_p50_us"],
                             "T_B_deviation": receiver["T_B_p50_us"] / t_b_pred - 1.0,
                             "k1_c_B_ns": float(np.mean([item["c_B_ns"] for item in case["k1"]]))})
        worst = max(per_link, key=lambda item: item["r_pred"])
        case["prediction"] = {"links": per_link, "r_pred": worst["r_pred"], "r_closed_form": worst["r_closed_form"],
                              "r_pred_case_c_B": max(item["r_pred_case_c_B"] for item in per_link)}


def overhead(campaign: pathlib.Path) -> list:
    records = [r for r in read_results(campaign) if r.get("rc") == 0]
    rows = []
    for name in dict.fromkeys(r["case"] for r in records):
        pairs = []
        for trial in sorted({r["trial"] for r in records if r["case"] == name}):
            on = [r for r in records if r["case"] == name and r["trial"] == trial and r["arm"] == "on"]
            off = [r for r in records if r["case"] == name and r["trial"] == trial and r["arm"] == "off"]
            if on and off:
                pairs.append({"trial": trial, "on": on[0]["steady_fps"], "off": off[0]["steady_fps"],
                              "ratio": on[0]["steady_fps"] / off[0]["steady_fps"] - 1.0,
                              "order": "off->on" if trial % 2 == 1 else "on->off"})
        ratios = np.array([pair["ratio"] for pair in pairs])
        rows.append({"case": name, "pairs": pairs, "mean": float(ratios.mean()) if len(ratios) else float("nan"),
                     "std": float(ratios.std(ddof=1)) if len(ratios) > 1 else float("nan")})
    return rows


# ------------------------------------------------------------------ figures

def figures(scan_result: dict, model: dict, out: pathlib.Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    markers = {2: "o", 3: "s"}
    colors = {2: "tab:blue", 3: "tab:orange"}

    # 1. t_tr / t_chain vs B_host, fit and hops
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    chain, spec = model["transport"], model["transport_spec"]
    all_bytes = []
    for case in scan_result["cases"]:
        for link in case["k2"]["links"].values():
            x = link["host_copy_bytes_p50"] / 1024
            all_bytes.append(link["host_copy_bytes_p50"])
            axes[0].plot(x, link["t_tr_p50"], markers[case["dimension"]], mfc="none", color="0.55", ms=6)
            axes[0].errorbar(x, link["t_chain_p50"], yerr=[[0], [link["t_chain_p95"] - link["t_chain_p50"]]],
                             fmt=markers[case["dimension"]], color=colors[case["dimension"]], ms=5, capsize=2)
            axes[0].annotate(case["case"], (x, link["t_chain_p50"]), fontsize=6, alpha=0.7, xytext=(3, -8),
                             textcoords="offset points")
            for group, color, x_key in (("readback", "tab:green", "dma_bytes"),
                                        ("host", "tab:red", "host_copy_bytes_p50"),
                                        ("upload", "tab:purple", "dma_bytes")):
                axes[1].plot(link[x_key] / 1024, link["chain_" + group + "_p50"], markers[case["dimension"]],
                             color=color, ms=4)
    grid = np.logspace(np.log10(min(all_bytes)), np.log10(max(all_bytes)), 50)
    axes[0].plot(grid / 1024, chain["tau0_us"] + chain["slope"] * grid, "k--", lw=1,
                 label=f"t_chain = {chain['tau0_us']:.0f} us + B_host / {chain['beta_GBps']:.2f} GB/s")
    axes[0].plot(grid / 1024, spec["tau0_us"] + spec["slope"] * grid, ":", color="0.4", lw=1,
                 label=f"t_tr (open, incl. waits on the receiver): {spec['tau0_us']:.0f} us + B_host / "
                       f"{spec['beta_GBps']:.2f} GB/s")
    for group, color, label in (("readback", "tab:green", "readback (start + DMA + barrier) vs DMA bytes"),
                                ("host", "tab:red", "host (wake + stamp + memcpy + signal) vs host bytes"),
                                ("upload", "tab:purple", "upload (queue start + DMA) vs DMA bytes")):
        fit = model["hops"][group]
        axes[1].plot(grid / 1024, fit["intercept"] + fit["slope"] * grid, "--", color=color, lw=1,
                     label=f"{label}: {fit['intercept']:.0f} us + bytes / {fit['bandwidth_GBps']:.1f} GB/s")
    for axis, location in zip(axes, ("upper left", "lower right")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("bytes per link per step (KiB)")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7, loc=location)
    axes[0].set_ylabel("p50 (us); bar to the t_chain p95")
    axes[1].set_ylabel("hop p50 (us)")
    axes[0].set_title("transport chain vs host bytes (o 2-D, s 3-D)")
    axes[1].set_title("per hop of the chain")
    figure.tight_layout()
    paths.append(out / "e29_ttr_vs_bytes.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 2. T_B vs n_B with c_B lines
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for case in scan_result["cases"]:
        dimension = case["dimension"]
        for item in case["k1"]:
            axis.plot(item["N"], item["T_B_p50_us"], markers[dimension], mfc="none", color=colors[dimension], ms=6)
        for sim in case["k2"]["sims"]:
            axis.plot(sim["n_B"], sim["T_B_p50_us"], markers[dimension], color=colors[dimension], ms=5)
    for dimension, fit in model["c_B"].items():
        values = [item["N"] for case in scan_result["cases"] if case["dimension"] == dimension for item in case["k1"]]
        span = np.logspace(np.log10(min(values) / 2), np.log10(max(values) * 1.3), 20)
        axis.plot(span, fit["slope"] * span, "--", color=colors[dimension], lw=1,
                  label=f"{dimension}-D c_B = {fit['c_B_ns']:.2f} ns per particle (K = 1)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("particles in the phase B force sweep (K = 1: N; K = 2: n_B)")
    axis.set_ylabel("T_B p50 (us)")
    axis.set_title("phase B: K = 1 (open) and K = 2 (filled)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    paths.append(out / "e29_tb_vs_nb.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 3. r_pred vs measured
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    values = []
    for case in scan_result["cases"]:
        predicted = case["prediction"]["r_pred"]
        chain_measured, spec_measured = case["k2"]["r_chain_worst_p50"], case["k2"]["r_worst_p50"]
        values += [predicted, chain_measured, spec_measured]
        axis.plot(predicted, chain_measured, markers[case["dimension"]], color=colors[case["dimension"]], ms=6)
        axis.plot(predicted, spec_measured, markers[case["dimension"]], mfc="none", color="0.5", ms=6)
        axis.plot(case["prediction"]["r_pred_case_c_B"], chain_measured, "x", color=colors[case["dimension"]], ms=6)
        values.append(case["prediction"]["r_pred_case_c_B"])
        axis.plot([predicted, predicted], [chain_measured, spec_measured], "-", color="0.75", lw=0.7)
        axis.annotate(case["case"], (predicted, chain_measured), fontsize=7, xytext=(3, 3), textcoords="offset points")
    span = [min(values) / 1.5, max(values) * 1.5]
    axis.plot(span, span, "k--", lw=1, label="y = r_pred")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("r_pred = (tau0 + B_host / beta) / (c_B n_B)")
    axis.set_ylabel("measured p50, worse link per step:\nfilled r_chain = t_chain / T_B, open r = t_tr / T_B;\n"
                    "x = r_chain against r_pred with the case's own K = 1 c_B")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    paths.append(out / "e29_rpred_vs_r.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 4. eta vs r, with the phase offset
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for case in scan_result["cases"]:
        dimension, k2 = case["dimension"], case["k2"]
        for key, style, label in (("r_chain", dict(mfc=colors[dimension]), "r_chain"),
                                  ("r", dict(mfc="none"), "r")):
            axes[0].plot(k2[key + "_worst_p50"], case["eta"], markers[dimension], color=colors[dimension], ms=6, **style)
            axes[0].plot([k2[key + "_worst_p50"], k2[key + "_worst_p95"]], [case["eta"]] * 2, "-",
                         color=colors[dimension], lw=0.7, alpha=0.6)
        axes[0].annotate(case["case"], (k2["r_chain_worst_p50"], case["eta"]), fontsize=7, xytext=(-6, -11),
                         textcoords="offset points")
    axes[0].axvline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("p50 per step, worse link (line to p95): filled r_chain, open r = t_tr / T_B")
    axes[0].set_ylabel("eta = fps_K2 / (2 mean fps_K1), trace off")
    axes[0].grid(alpha=0.3, which="both")
    labels, data = [], []
    for case in scan_result["cases"]:
        for name, values in link_metrics(case["k2_run"]).items():
            labels.append(f"{case['case']}\n{name.replace('_to_', '>')}")
            data.append(values["phase_offset"] / values["T_B"])
    axes[1].boxplot(data, whis=(5, 95), showfliers=False)
    axes[1].set_xticks(range(1, len(labels) + 1), labels, rotation=90, fontsize=6)
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_ylabel("D' / T_B (receiver B start - sender send end)")
    axes[1].grid(alpha=0.3, axis="y")
    axes[0].set_title("efficiency vs r")
    axes[1].set_title("phase offset per link (box 25-75 %, whiskers 5-95 %)")
    figure.tight_layout()
    paths.append(out / "e29_eta_vs_r.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)
    return paths


# ------------------------------------------------------------------ report

def fmt(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:,.{digits}f}"


def tables(scan_result: dict, model: dict, overhead_rows: list) -> str:
    lines = ["## Runs (K = 2 trace on; eta from the runs without the trace)\n",
             "| case | w (s0 / s1) | n_B (s0 / s1) | B_host KiB | t_tr p50 / p95 us | t_chain p50 / p95 us | T_B us (s0 / s1) "
             "| r p50 / p95 | r_chain p50 / p95 | r_pred | closed form | r_pred, case c_B | r_receiver p95 | exposed steps "
             "| eta | D' p50 us |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        k2 = case["k2"]
        sims, links = k2["sims"], list(k2["links"].values())
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} | {} | {} | {} | {} | {} | {} |".format(
            case["case"], " / ".join(str(s["own_columns"]) for s in sims), " / ".join(f"{s['n_B']:,}" for s in sims),
            " / ".join(fmt(l["host_copy_bytes_p50"] / 1024) for l in links),
            " / ".join(f"{fmt(l['t_tr_p50'])}/{fmt(l['t_tr_p95'])}" for l in links),
            " / ".join(f"{fmt(l['t_chain_p50'])}/{fmt(l['t_chain_p95'])}" for l in links),
            " / ".join(fmt(s["T_B_p50_us"]) for s in sims),
            fmt(k2["r_worst_p50"], 3), fmt(k2["r_worst_p95"], 3),
            fmt(k2["r_chain_worst_p50"], 3), fmt(k2["r_chain_worst_p95"], 3),
            fmt(case["prediction"]["r_pred"], 3), fmt(case["prediction"]["r_closed_form"], 3),
            fmt(case["prediction"]["r_pred_case_c_B"], 3), fmt(k2["r_receiver_worst_p95"], 3), fmt(100 * k2["exposed_steps_fraction"], 1) + " %",
            fmt(100 * case["eta"], 1) + " %", " / ".join(fmt(l["phase_offset_p50"]) for l in links)))
    lines += ["\n## eta with and without the trace\n",
              "| case | fps K = 1 GPU 0 / 1 (off) | fps K = 2 (off) | eta (off) | fps K = 1 (on) | fps K = 2 (on) | eta (on) |",
              "|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        lines.append(f"| {case['case']} | {' / '.join(fmt(v) for v in case['fps_k1'])} | {fmt(case['fps_k2'])} | "
                     f"{fmt(100 * case['eta'], 1)} % | {' / '.join(fmt(v) for v in case['fps_k1_traced'])} | "
                     f"{fmt(case['fps_k2_traced'])} | {fmt(100 * case['eta_traced'], 1)} % |")
    lines.append("\n## Constants\n")
    for dimension, fit in model["c_B"].items():
        lines.append(f"- c_B {dimension}-D = {fit['c_B_ns']:.3f} ns per particle (K = 1, {fit['n']} runs, through the "
                     f"origin; relative residual rms {100 * fit['relative_residual_rms']:.1f} %, max "
                     f"{100 * fit['relative_residual_max']:.1f} %)")
    for key, label in (("transport", "t_chain"), ("transport_spec", "t_tr (as specified)")):
        fit = model[key]
        lines.append(f"- {label}: tau0 = {fit['tau0_us']:.1f} us, beta = {fit['beta_GBps']:.2f} GB/s ({fit['n']} links; "
                     f"residual rms {fit['residual_rms']:.1f} us, relative rms {100 * fit['relative_residual_rms']:.1f} %, "
                     f"max {100 * fit['relative_residual_max']:.1f} %)")
    for group, fit in model["hops"].items():
        lines.append(f"- chain hop {group}: {fit['intercept']:.1f} us + bytes / {fit['bandwidth_GBps']:.2f} GB/s "
                     f"(residual rms {fit['residual_rms']:.1f} us, relative max {100 * fit['relative_residual_max']:.1f} %)")
    lines += ["\n## T_B vs c_B n_B (K = 2)\n", "| case | link | n_B | c_B n_B us | T_B us | deviation | K = 1 c_B of the case ns |",
              "|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for item in case["prediction"]["links"]:
            lines.append(f"| {case['case']} | {item['link']} | {item['n_B']:,} | {fmt(item['T_B_pred_us'])} | "
                         f"{fmt(item['T_B_measured_us'])} | {fmt(100 * item['T_B_deviation'], 1)} % | "
                         f"{fmt(item['k1_c_B_ns'], 3)} |")
    lines += ["\n## Host step (worker) p50 / p95 us\n",
              "| case | link | " + " | ".join(name for name, _, _ in HOST_SEGMENTS) + " | host bytes KiB | memcpy GB/s |",
              "|---|---|" + "---|" * (len(HOST_SEGMENTS) + 2)]
    for case in scan_result["cases"]:
        for name, link in case["k2"]["links"].items():
            memcpy = link["host_memcpy_p50"]
            bandwidth = link["host_copy_bytes_p50"] / (memcpy * 1e3) if memcpy > 0 else float("nan")
            lines.append(f"| {case['case']} | {name} | " + " | ".join(
                f"{fmt(link['host_' + segment + '_p50'])} / {fmt(link['host_' + segment + '_p95'])}"
                for segment, _, _ in HOST_SEGMENTS) + f" | {fmt(link['host_copy_bytes_p50'] / 1024)} | {fmt(bandwidth, 2)} |")
    lines += ["\n## t_tr hops p50 us\n", "| case | link | " + " | ".join(hop for hop, _, _ in HOPS) + " | t_tr | t_chain |",
              "|---|---|" + "---|" * (len(HOPS) + 2)]
    for case in scan_result["cases"]:
        for name, link in case["k2"]["links"].items():
            lines.append(f"| {case['case']} | {name} | " + " | ".join(fmt(link[hop + '_p50']) for hop, _, _ in HOPS)
                         + f" | {fmt(link['t_tr_p50'])} | {fmt(link['t_chain_p50'])} |")
    lines += ["\n## Step composition, medians us (traced runs; K = 1 = mean of GPU 0 and 1)\n",
              "| case | run | " + " | ".join(CYCLE_PARTS) + " | period | eta (off) |",
              "|---|---|" + "---|" * (len(CYCLE_PARTS) + 2)]
    for case in scan_result["cases"]:
        k1 = {key: float(np.mean([item[key] for item in case["cycles"]["k1"]])) for key in CYCLE_PARTS + ("period",)}
        lines.append(f"| {case['case']} | K = 1 | " + " | ".join(fmt(k1[key]) for key in CYCLE_PARTS)
                     + f" | {fmt(k1['period'])} | |")
        lines.append(f"| {case['case']} | K = 1 / 2 | " + " | ".join(fmt(k1[key] / 2) for key in CYCLE_PARTS)
                     + f" | {fmt(k1['period'] / 2)} | |")
        for item in case["cycles"]["k2"]:
            lines.append(f"| {case['case']} | K = 2 s{item['sim']} | " + " | ".join(fmt(item[key]) for key in CYCLE_PARTS)
                         + f" | {fmt(item['period'])} | {fmt(100 * case['eta'], 1)} % |")
    lines += ["\n## Per sim (K = 2 trace on)\n",
              "| case | sim | T_B p50 / p95 us | phase A p50 us | phase C p50 us | gating p50 / p95 us |", "|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for sim in case["k2"]["sims"]:
            lines.append(f"| {case['case']} | {sim['sim']} | {fmt(sim['T_B_p50_us'])} / {fmt(sim['T_B_p95_us'])} | "
                         f"{fmt(sim['phase_a_p50_us'])} | {fmt(sim['phase_c_p50_us'])} | "
                         f"{fmt(sim['gating_p50_us'])} / {fmt(sim['gating_p95_us'])} |")
    lines += ["\n## Checks per K = 2 run\n",
              "| case | coverage (steady) | chains with a violation | min gap readback_end -> worker us | "
              "min gap copy -> upload_start us | clock map residual rms / max ns | driver maxDeviation min ns |",
              "|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        check = case["k2"]["checks"]["causal"]
        pairs = {pair["pair"]: pair for pair in check["pairs"]}
        lines.append(f"| {case['case']} | {fmt(100 * case['k2']['coverage']['steady_coverage'], 2)} % | "
                     f"{fmt(100 * check['chains_with_violation_fraction'], 3)} % | "
                     f"{fmt(pairs['readback_end <= worker_source_wait']['min_gap_us'])} | "
                     f"{fmt(pairs['worker_copy <= upload_start']['min_gap_us'])} | "
                     + "; ".join(f"{fmt(a, 0)} / {fmt(b, 0)}" for a, b in zip(check["clock_fit_residual_rms_ns"],
                                                                         check["clock_fit_residual_max_ns"])) + " | "
                     + " / ".join(fmt(v, 0) for v in check["driver_max_deviation_min_ns"]) + " |")
    lines += ["\n## DMA medians vs E24 (depth-1 anatomy, 21ce20d bands 2/2/3) us\n",
              "| case | link | readback here | readback E24 | upload here | upload E24 |", "|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        reference = E24_DMA_P50.get(case["case"])
        if reference is None:
            continue
        for position, (name, link) in enumerate(sorted(case["k2"]["links"].items())):
            lines.append(f"| {case['case']} | {name} | {fmt(link['readback_dma_p50'])} | {fmt(reference[position])} | "
                         f"{fmt(link['upload_dma_p50'])} | {fmt(reference[2 + position])} |")
    if overhead_rows:
        lines += ["\n## Overhead (trace on vs off, K = 2)\n", "| case | trial | order | off fps | on fps | on / off - 1 |",
                  "|---|---|---|---|---|---|"]
        for row in overhead_rows:
            for pair in row["pairs"]:
                lines.append(f"| {row['case']} | {pair['trial']} | {pair['order']} | {fmt(pair['off'])} | "
                             f"{fmt(pair['on'])} | {fmt(100 * pair['ratio'], 2)} % |")
            lines.append(f"| {row['case']} | mean ± std | | | | {fmt(100 * row['mean'], 2)} ± {fmt(100 * row['std'], 2)} % |")
    return "\n".join(lines) + "\n"


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items() if key != "k2_run"}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", default=None)
    parser.add_argument("--overhead", default=None)
    parser.add_argument("--run", default=None, help="one trace directory: print its summary and checks")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    if arguments.run:
        print(json.dumps(jsonable(run_summary(load_run(arguments.run))), indent=1))
        return 0
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    overhead_rows = overhead(pathlib.Path(arguments.overhead)) if arguments.overhead else []
    scan_result = analyze_scan(pathlib.Path(arguments.scan)) if arguments.scan else {"cases": []}
    model = constants(scan_result) if scan_result["cases"] else {}
    figure_paths = []
    if scan_result["cases"]:
        predictions(scan_result, model)
        figure_paths = figures(scan_result, model, out)
    report = tables(scan_result, model, overhead_rows) if scan_result["cases"] else ""
    (out / "e29_tables.md").write_text(report, encoding="utf-8")
    summary = {"model": model, "overhead": overhead_rows,
               "cases": [{key: value for key, value in case.items() if key != "k2_run"} for case in scan_result["cases"]],
               "figures": [str(path) for path in figure_paths]}
    (out / "e29_summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
