"""
step_trace_model.py — E29 analysis of the step traces (experiment/v6/utils/phase_trace_v6.StepTracer).

Per run (one --step-trace directory, steady steps = step >= warmup):
  clock    traces of the first scan (clock fit version 1: every calibration pair, device ticks on a 256 ns
           float grid) are re-mapped on load with the current fit (clock_map_v6.fit_clock: only pairs within
           1 us of the smallest driver maxDeviation); their 256 ns tick grid stays;
  checks   causal order of every link's chain on the common host clock (tolerance of a GPU <-> host pair =
           that GPU map's largest residual + half its tick grid; per link: min gap - tolerance = margin),
           coverage (steps whose recorded ticks and worker points are all present), readback / upload DMA;
  per link and step
           t_tr = upload_end - send_end in consecutive hops (HOPS below: they telescope and do not overlap);
           t_chain = t_tr minus the two waits on the receiver (its readback_done(n) and its previous
           upload): the transport chain itself, without the phase coupling (the worker of a link into a
           lagging GPU waits for that GPU's own readback);
           T_B (receiver), phase offset D' = receiver b_start - send_end (and |D'| / T_B; the leading GPU can
           change from step to step, so the signed median can sit near 0 while the GPUs are well apart),
           exposure = max(0, upload_end - receiver b_end), B->C wait = receiver c_start - receiver b_end;
           r = t_tr / T_B (the task's definition, includes the phase coupling), r_chain = t_chain / T_B,
           r_receiver = (upload_end - receiver b_start) / T_B (> 1 = exposed); each per step on the worse link;
  per sim  C->A gap = a_start(n) - c_end(n - 1): idle time on the GPU (the host has queued A(n) long before).
Per campaign (step_trace_campaign.py --mode scan output; the last successful record of every run id):
  eta = fps_K2 / (2 mean(fps_K1 GPU 0, GPU 1)) and eta_min = fps_K2 / (2 min(...)) from the runs WITHOUT
  the trace, per trial (records may carry "trial"; the E29 scan has one, so eta is single-trial) and from
  the traced runs as a second sample; fps at full precision (run_meta, or steps / seconds when the log's
  0.1 fps is coarser); c_B from the traced K = 1 runs (T_B = c_B N, one line through the origin per
  dimension); from the traced K = 2 runs: t_tr p50 = tau0 + B_host / beta over both links of every run (as
  specified, includes the phase coupling) and the same line for t_chain, plus one line per chain hop
  (readback vs DMA bytes, host vs host bytes, upload vs DMA bytes); r_pred(t_chain) = (tau0 + B_host / beta)
  / (c_B n_B) with the t_chain line, r_pred(t_tr) with the t_tr line, the closed form tau0 / (c_B n_B) +
  64 / (c_B beta (w - b)); T_B against c_B n_B and against a two-term check (phase B's correction and density
  sweeps run outside band 2, only its force sweep outside band 3 = n_B); tables + 4 figures.
Phase-1 checks (--checks NAME=DIR ...): coverage, causal order with per-link margins, DMA vs E24.

    .venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_model \\
        --scan logs/e29_step_trace/scan --overhead logs/e29_step_trace/overhead --out docs/perf_model/e29 \\
        --checks 2d_1m=logs/e29_step_trace/overhead/runs/2d_1m__on__t1 \\
                 2d_16m=logs/e29_step_trace/checks_final/2d_16m_k2 3d_8m=logs/e29_step_trace/checks_final/3d_8m_k2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys

import numpy as np

from experiment.v6.utils.clock_map_v6 import clock_to_host, fit_clock, host_to_clock

# The chain of one link on the host clock (each must not precede the previous one). Traces of the fixed
# instrument add worker_dest_signal (stamped right before the worker_done host signal).
CAUSAL_CHAIN = ("send_end", "readback_start", "readback_end", "worker_source_wait", "worker_copy",
                "upload_start", "upload_end", "receiver_c_start")
CAUSAL_CHAIN_V2 = ("send_end", "readback_start", "readback_end", "worker_source_wait", "worker_copy",
                   "worker_dest_signal", "upload_start", "upload_end", "receiver_c_start")
# Which clock each chain point is on: the sender's GPU, the host, the receiver's GPU.
CHAIN_DOMAIN = {"send_end": "sender", "readback_start": "sender", "readback_end": "sender",
                "worker_source_wait": "host", "worker_copy": "host", "worker_dest_signal": "host",
                "upload_start": "receiver", "upload_end": "receiver", "receiver_c_start": "receiver"}
# GPU-derived columns of steps_link.csv and whose GPU they are on.
GPU_LINK_COLUMNS = {"sender": ("send_end", "readback_start", "readback_copy_end", "readback_end"),
                    "receiver": ("upload_start", "upload_end", "receiver_b_start", "receiver_b_end",
                                 "receiver_c_start")}
DEVICE_HOST_COLUMNS = ("step", "sim", "device", "parity", "host_read", "missing_ticks", "complete")
# t_tr = upload_end - send_end in consecutive hops (name, start column, end column). They telescope (the sum
# is t_tr on every step) and do not overlap: every start causally precedes its end. The first scan's
# instrument stamped worker_signal only after both host signals had returned and the thread had the GIL
# back, when the upload had sometimes already started (0.1-2.4 % of steps): there copy -> upload start is
# one hop. The fixed instrument splits it at worker_dest_signal.
HOPS_HEAD = (("readback_start_delay", "send_end", "readback_start"),
             ("readback_dma", "readback_start", "readback_copy_end"),
             ("readback_barrier", "readback_copy_end", "readback_end"),
             ("readback_to_worker", "readback_end", "worker_source_wait"),
             ("wait_receiver_readback", "worker_source_wait", "worker_dest_guard"),
             ("wait_receiver_previous_upload", "worker_dest_guard", "worker_upload_guard"),
             ("stamp_check", "worker_upload_guard", "worker_stamp"),
             ("memcpy", "worker_stamp", "worker_copy"))
HOPS_V1 = HOPS_HEAD + (("copy_to_upload", "worker_copy", "upload_start"),
                       ("upload_dma", "upload_start", "upload_end"))
HOPS_V2 = HOPS_HEAD + (("signal", "worker_copy", "worker_dest_signal"),
                       ("signal_to_upload", "worker_dest_signal", "upload_start"),
                       ("upload_dma", "upload_start", "upload_end"))
ALL_HOPS = tuple(hop for hop, _, _ in HOPS_HEAD) + ("copy_to_upload", "signal", "signal_to_upload", "upload_dma")
# The task's five groups (start, end): readback, readback end -> worker sees, host segment (two waits on the
# receiver, stamp check, memcpy), copy end -> upload start (host signal + queue start), upload.
HOP_GROUPS = (("readback", "send_end", "readback_end"),
              ("readback_to_worker", "readback_end", "worker_source_wait"),
              ("host", "worker_source_wait", "worker_copy"),
              ("copy_to_upload", "worker_copy", "upload_start"),
              ("upload", "upload_start", "upload_end"))
# The chain without the two waits on the receiver, in three hops for the per-hop fits (start, end, minus).
RECEIVER_WAITS = ("wait_receiver_readback", "wait_receiver_previous_upload")
CHAIN_GROUPS = (("readback", "send_end", "readback_end", ()),
                ("host", "readback_end", "worker_copy", RECEIVER_WAITS),
                ("upload", "worker_copy", "upload_end", ()))
# The host step (worker), E24's six segments ("signal" = both host signals returned).
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
STEADY_LINE = re.compile(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps")


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


def remap_version1(run: dict) -> None:
    """A clock-fit-version-1 trace: invert its stored map (back to device ns on the 256 ns grid), refit
    every GPU's map from calibration.csv with fit_clock and map the GPU columns again. Also marks a device
    row incomplete when a tick its sim records is missing (version 1 only checked the derived phase ends)."""
    meta, device, link = run["meta"], run["device"], run["link"]
    calibration = read_table(run["dir"] / "calibration.csv")
    transforms, fits = [], []
    for sim, old in enumerate(meta["clock"]["fits"]):
        mask = calibration["sim"] == sim
        device_absolute = calibration["device_ns"][mask]            # float64 epoch ns, 256 ns grid
        origin = float(device_absolute[0])
        fit = fit_clock(device_absolute - origin, calibration["host_ns"][mask],   # exact differences
                        calibration["max_deviation_ns"][mask])
        fit.update({"device_tick_quantum_ns": 256.0, "remapped_from_fit_version": 1,
                    "fit_version_1": {key: old.get(key) for key in ("samples_kept", "residual_rms_ns", "residual_max_ns")}})
        transforms.append((old, fit, old["device_center_ns"] - origin))
        fits.append(fit)

    def remap(values, gpus):
        out = values.copy()
        for gpu, (old, fit, center_offset) in enumerate(transforms):
            mask = (gpus == gpu) & np.isfinite(values)
            if mask.any():
                out[mask] = np.round(clock_to_host(fit, host_to_clock(old, values[mask], centered=True) + center_offset))
        return out
    sims = device["sim"].astype(int)
    for column in list(device):
        if column not in DEVICE_HOST_COLUMNS:
            device[column] = remap(device[column], sims)
    for role, columns in GPU_LINK_COLUMNS.items():
        gpus = link[role].astype(int)
        for column in columns:
            link[column] = remap(link[column], gpus)
    missing = np.zeros(len(sims))
    for sim, labels in enumerate(meta["labels"]["compute"]):
        mask = sims == sim
        for label in labels:
            if label in device:
                missing[mask] += ~np.isfinite(device[label][mask])
    device["missing_ticks"] = missing
    device["complete"] = device["complete"] * (missing == 0)
    meta["clock"]["fits"] = fits
    meta["clock"]["fit_version"] = 2


def load_run(run_dir) -> dict:
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    run = {"dir": run_dir, "meta": meta, "device": read_table(run_dir / "steps_device.csv"),
           "link": read_table(run_dir / "steps_link.csv")}
    run["dest_signal"] = bool("worker_dest_signal" in run["link"]
                              and np.isfinite(run["link"]["worker_dest_signal"]).any())
    if meta["clock"].get("fit_version", 1) < 2:
        remap_version1(run)
    return run


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
    hops = HOPS_V2 if run["dest_signal"] else HOPS_V1
    out = {}
    for name in sorted(set(link["link"])):
        rows = select(link, (link["link"] == name) & (link["step"] >= warmup) & (link["complete"] == 1))
        order = np.argsort(rows["step"])
        rows = {key: value[order] for key, value in rows.items()}

        def span(start, end):
            return (rows[end] - rows[start]) / 1e3
        metrics = {"step": rows["step"], "t_tr": span("send_end", "upload_end"),
                   "T_B": span("receiver_b_start", "receiver_b_end"),
                   "phase_offset": span("send_end", "receiver_b_start"),
                   "exposure": np.maximum(0.0, rows["upload_end"] - rows["receiver_b_end"]) / 1e3,
                   "b_to_c_wait": span("receiver_b_end", "receiver_c_start"),
                   "upload_end_after_b_start": span("receiver_b_start", "upload_end"),
                   "signal_return_after_upload_start": span("upload_start", "worker_signal"),
                   "host_copy_bytes": rows["host_copy_bytes"], "dma_bytes": rows["dma_bytes"],
                   "sender": int(rows["sender"][0]), "receiver": int(rows["receiver"][0])}
        for hop, start, end in hops:
            metrics[hop] = span(start, end)
        metrics["copy_to_upload"] = span("worker_copy", "upload_start")
        for group, start, end in HOP_GROUPS:
            metrics["group_" + group] = span(start, end)
        for group, start, end, minus in CHAIN_GROUPS:
            metrics["chain_" + group] = span(start, end) - sum(metrics[wait] for wait in minus)
        for segment, start, end in HOST_SEGMENTS:
            metrics["host_" + segment] = span(start, end)
        metrics["t_chain"] = metrics["t_tr"] - sum(metrics[wait] for wait in RECEIVER_WAITS)
        metrics["r"] = metrics["t_tr"] / metrics["T_B"]
        metrics["r_chain"] = metrics["t_chain"] / metrics["T_B"]
        metrics["r_receiver"] = metrics["upload_end_after_b_start"] / metrics["T_B"]
        metrics["phase_offset_ratio"] = metrics["phase_offset"] / metrics["T_B"]
        metrics["phase_offset_abs_ratio"] = np.abs(metrics["phase_offset_ratio"])
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


def c_to_a_gap(run: dict) -> dict:
    """Per sim: a_start(n) - c_end(n - 1), steady steps, us."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    out = {}
    for sim in sorted(set(device["sim"].astype(int))):
        rows = select(device, (device["sim"] == sim) & (device["complete"] == 1))
        order = np.argsort(rows["step"])
        steps, a_start, c_end = rows["step"][order], rows["a_start"][order], rows["c_end"][order]
        consecutive = steps[1:] == steps[:-1] + 1
        out[sim] = (a_start[1:] - c_end[:-1])[consecutive & (steps[1:] >= warmup)] / 1e3
    return out


CYCLE_PARTS = ("a", "a_to_b", "b", "b_to_c", "c", "c_to_next_a")


def cycle_components(run: dict) -> list:
    """Per sim, medians over steady steps (us): phase A, A->B gap, phase B, B->C wait, phase C and
    c_end(n) -> a_start(n + 1), the step period a_start(n) -> a_start(n + 1) and its mean. Medians do not add
    up to the median period exactly."""
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
        entry = {"sim": sim, **{key: float(np.median(value[steady]) / 1e3) for key, value in parts.items()}}
        entry["period_mean"] = float(np.mean(parts["period"][steady]) / 1e3)
        if "b_density_deep_interior_end" in rows:     # phase B split: correction + density | force
            entry["b_correction_density"] = float(np.nanmedian(
                (rows["b_density_deep_interior_end"] - rows["b_start"])[steady]) / 1e3)
            entry["b_force"] = float(np.nanmedian((rows["b_end"] - rows["b_density_deep_interior_end"])[steady]) / 1e3)
        out.append(entry)
    return out


def causal_check(run: dict) -> dict:
    """Fraction of steady (step, link) chains whose consecutive points go backwards by more than the
    tolerance: 0 for points on one GPU (compute and transfer pools share the device clock and its map;
    the map is monotonic), for a GPU <-> host pair that GPU map's largest residual plus half its tick grid.
    Per link and cross-domain pair: min gap, tolerance and margin = min gap - tolerance (> 0: the order is
    resolved beyond the clock's uncertainty). Also the rate of worker_signal (both host signals returned)
    after upload_start: allowed, the upload starts once the dest signal lands."""
    meta = run["meta"]
    fits = meta["clock"]["fits"]
    tolerance_by_gpu = [fit["residual_max_ns"] + 0.5 * fit.get("device_tick_quantum_ns", 0.0) for fit in fits]
    chain = CAUSAL_CHAIN_V2 if run["dest_signal"] else CAUSAL_CHAIN
    link = run["link"]
    rows = select(link, (link["step"] >= meta["warmup"]) & (link["complete"] == 1))
    names = sorted(set(rows["link"]))
    result = {"chains": int(len(rows["step"])), "chain": list(chain), "pairs": [], "links": {name: {} for name in names}}
    any_violation = np.zeros(len(rows["step"]), dtype=bool)
    for earlier, later in zip(chain, chain[1:]):
        gap = rows[later] - rows[earlier]
        domains = (CHAIN_DOMAIN[earlier], CHAIN_DOMAIN[later])
        tolerance = np.zeros(len(gap))
        if domains[0] != domains[1]:
            role = "sender" if "sender" in domains else "receiver"
            tolerance = np.array([tolerance_by_gpu[gpu] for gpu in rows[role].astype(int)])
        violation = gap < -tolerance
        any_violation |= violation
        pair = f"{earlier} <= {later}"
        result["pairs"].append({"pair": pair, "domains": "/".join(domains),
                                "negative_fraction": float(np.mean(gap < 0)) if len(gap) else 0.0,
                                "violation_fraction": float(np.mean(violation)) if len(gap) else 0.0,
                                "min_gap_us": float(np.min(gap) / 1e3) if len(gap) else float("nan"),
                                "p50_gap_us": float(np.median(gap) / 1e3) if len(gap) else float("nan")})
        for name in names:
            mask = rows["link"] == name
            if not mask.any():
                continue
            minimum, allowed = float(np.min(gap[mask])), float(np.max(tolerance[mask]))
            result["links"][name][pair] = {"min_gap_us": minimum / 1e3, "tolerance_us": allowed / 1e3,
                                           "margin_us": (minimum - allowed) / 1e3 if domains[0] != domains[1] else None}
    overlap = rows["worker_signal"] - rows["upload_start"]
    result["signal_return_after_upload_start_fraction"] = float(np.mean(overlap > 0)) if len(overlap) else 0.0
    result["chains_with_violation_fraction"] = float(np.mean(any_violation)) if len(any_violation) else 0.0
    result["clock_fit_residual_max_ns"] = [fit["residual_max_ns"] for fit in fits]
    result["clock_fit_residual_rms_ns"] = [fit["residual_rms_ns"] for fit in fits]
    result["clock_fit_samples_kept"] = [fit["samples_kept"] for fit in fits]
    result["clock_fit_samples"] = [fit["samples"] for fit in fits]
    result["clock_fit_version_1_residual_max_ns"] = [fit.get("fit_version_1", {}).get("residual_max_ns") for fit in fits]
    result["clock_fit_version_1_residual_rms_ns"] = [fit.get("fit_version_1", {}).get("residual_rms_ns") for fit in fits]
    result["device_tick_quantum_ns"] = [fit.get("device_tick_quantum_ns", 0.0) for fit in fits]
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
    missing = device.get("missing_ticks")
    return {"expected_steps": expected, "recorded_steps": len(steps), "complete_steps": len(good),
            "coverage": len(good) / expected if expected else float("nan"),
            "steady_coverage": len(steady) / (expected - meta["warmup"]) if expected > meta["warmup"] else float("nan"),
            "device_rows_missing_ticks": int(np.sum(missing > 0)) if missing is not None else None}


LINK_KEYS = (("t_tr", "t_chain", "T_B", "phase_offset", "phase_offset_ratio", "phase_offset_abs_ratio", "exposure",
              "b_to_c_wait", "upload_end_after_b_start", "signal_return_after_upload_start")
             + RATIOS + ALL_HOPS + tuple("group_" + group for group, _, _ in HOP_GROUPS)
             + tuple("chain_" + group for group, _, _, _ in CHAIN_GROUPS)
             + tuple("host_" + segment for segment, _, _ in HOST_SEGMENTS))


def run_summary(run: dict) -> dict:
    metrics = link_metrics(run)
    meta = run["meta"]
    slabs = (meta.get("slabs") or {}).get("per_sim", [])
    summary = {"case": meta["case"], "K": meta["K"], "steady_fps": meta["result"].get("steady_fps"),
               "dest_signal": run["dest_signal"], "coverage": coverage(run), "links": {}, "sims": []}
    for index, slab in enumerate(slabs):
        summary["sims"].append({"sim": index, "own_columns": slab["own_columns"], "n_B": slab["n_B"],
                                "n_correction_interior": slab.get("n_correction_interior"),
                                "n_density_deep_interior": slab.get("n_density_deep_interior"),
                                "own_particles": slab["own_particles"], "band_widths": slab["band_widths"]})
    gaps = c_to_a_gap(run)
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
        entry["c_to_a_gap_p50_us"], entry["c_to_a_gap_p95_us"] = quantiles(gaps.get(sim, []))
    for name, values in metrics.items():
        entry = {"sender": values["sender"], "receiver": values["receiver"], "steps": int(len(values["step"])),
                 "host_copy_bytes_p50": float(np.median(values["host_copy_bytes"])),
                 "dma_bytes": float(np.median(values["dma_bytes"])),
                 "exposed_fraction": float(np.mean(values["exposure"] > 0)),
                 "exposure_mean_us": float(np.mean(values["exposure"]))}
        entry["phase_offset_ratio_p5"], entry["phase_offset_ratio_p95"] = quantiles(values["phase_offset_ratio"], (5, 95))
        for key in LINK_KEYS:
            if key in values:
                entry[key + "_p50"], entry[key + "_p95"] = quantiles(values[key])
        summary["links"][name] = entry
    for key in RATIOS:
        worst = worst_link(metrics, key)
        summary[key + "_worst_p50"], summary[key + "_worst_p95"] = quantiles(worst)
    exposure = worst_link(metrics, "exposure")
    summary["exposed_steps_fraction"] = float(np.mean(exposure > 0)) if len(exposure) else float("nan")
    summary["exposure_mean_us"] = float(np.mean(exposure)) if len(exposure) else float("nan")
    summary["exposure_mean_when_exposed_us"] = (float(np.mean(exposure[exposure > 0])) if np.any(exposure > 0)
                                                else float("nan"))
    if len(metrics) > 1:
        summary["checks"] = {"causal": causal_check(run)}
    return summary


# ------------------------------------------------------------------ campaign

def read_results(campaign: pathlib.Path) -> list:
    """Records of the campaign; per run id only the LAST successful one (a resumed campaign re-runs both
    GPUs of a K = 1 pair when one failed, which appends a second record for the other)."""
    latest = {}
    for line in (campaign / "results.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not record.get("marker") and record.get("rc") == 0:
            latest[record["run_id"]] = record
    return list(latest.values())


def precise_fps(record: dict, campaign: pathlib.Path) -> float:
    """Steady fps at full precision: the trace's run_meta when traced, else the more precise of the bench
    log's printed fps (0.1 fps) and its steps / seconds (0.01 s)."""
    if record.get("trace_dir"):
        meta_path = campaign / record["trace_dir"] / "run_meta.json"
        if meta_path.exists():
            value = json.loads(meta_path.read_text(encoding="utf-8"))["result"].get("steady_fps")
            if value:
                return float(value)
    log_path = campaign / record["log"] if record.get("log") else None
    if log_path is not None and log_path.exists():
        match = STEADY_LINE.search(log_path.read_text(encoding="utf-8", errors="replace"))
        if match:
            steps, seconds, printed = int(match.group(1)), float(match.group(2)), float(match.group(3))
            if seconds > 0 and 0.005 / seconds < 0.05 / printed:
                return steps / seconds
            return printed
    return float(record["steady_fps"])


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


def efficiency(k1_fps: list, k2_fps: float) -> dict:
    mean, low = float(np.mean(k1_fps)), float(np.min(k1_fps))
    return {"fps_k1": k1_fps, "fps_k2": k2_fps, "eta": k2_fps / (2.0 * mean), "eta_min": k2_fps / (2.0 * low),
            "k1_spread": (max(k1_fps) - low) / mean}


def analyze_scan(scan: pathlib.Path) -> dict:
    records = read_results(scan)
    cases = []
    for name in dict.fromkeys(record["case"] for record in records):
        def pick(kind, trace, trial):
            return sorted((r for r in records if r["case"] == name and r["kind"] == kind
                           and bool(r.get("trace")) == trace and r.get("trial", 1) == trial),
                          key=lambda r: r.get("gpu", 0))
        trials = sorted({r.get("trial", 1) for r in records if r["case"] == name})
        off, on = [], []
        for trial in trials:
            for trace, sink in ((False, off), (True, on)):
                k1, k2 = pick("k1", trace, trial), pick("k2", trace, trial)
                if len(k1) == 2 and len(k2) == 1:
                    sink.append({"trial": trial, "k1": k1, "k2": k2[0],
                                 **efficiency([precise_fps(r, scan) for r in k1], precise_fps(k2[0], scan))})
        if not off or not on:
            print(f"[step_trace_model] {name}: incomplete (trials with a full off set {len(off)}, on set {len(on)})",
                  file=sys.stderr)
            continue
        traced = on[-1]
        k1_runs = [load_run(scan / r["trace_dir"]) for r in traced["k1"]]
        k2_run = load_run(scan / traced["k2"]["trace_dir"])
        cycles = {"k1": [cycle_components(run)[0] for run in k1_runs], "k2": cycle_components(k2_run)}
        k1_summaries = [run_summary(run) for run in k1_runs]
        k1_items = []
        for record, summary, cycle in zip(traced["k1"], k1_summaries, cycles["k1"]):
            item = {"gpu": record["gpu"], "N": summary["sims"][0]["n_B"], "T_B_p50_us": summary["sims"][0]["T_B_p50_us"],
                    "c_B_ns": 1e3 * summary["sims"][0]["T_B_p50_us"] / summary["sims"][0]["n_B"]}
            for key in ("b_correction_density", "b_force"):
                if key in cycle:
                    item[key + "_p50_us"] = cycle[key]
            k1_items.append(item)
        etas = np.array([item["eta"] for item in off])
        cases.append({"case": name, "dimension": traced["k2"]["dimension"], "trials": len(off),
                      "eta_trials": [{key: item[key] for key in ("trial", "fps_k1", "fps_k2", "eta", "eta_min", "k1_spread")}
                                     for item in off],
                      "fps_k1": off[0]["fps_k1"], "fps_k2": off[0]["fps_k2"], "k1_spread": off[0]["k1_spread"],
                      "eta": float(etas.mean()), "eta_std": float(etas.std(ddof=1)) if len(etas) > 1 else float("nan"),
                      "eta_min": float(np.mean([item["eta_min"] for item in off])),
                      "fps_k1_traced": traced["fps_k1"], "fps_k2_traced": traced["fps_k2"],
                      "k1_spread_traced": traced["k1_spread"],
                      "eta_traced": traced["eta"], "eta_min_traced": traced["eta_min"],
                      "k2_on_off": traced["fps_k2"] / off[-1]["fps_k2"] - 1.0,
                      "k1": k1_items, "k2": run_summary(k2_run), "k2_run": k2_run, "cycles": cycles})
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
    hop_values = {group: [] for group, _, _, _ in CHAIN_GROUPS}
    for case in scan_result["cases"]:
        for link in case["k2"]["links"].values():
            t_tr.append(link["t_tr_p50"])
            t_chain.append(link["t_chain_p50"])
            host_bytes.append(link["host_copy_bytes_p50"])
            dma_bytes.append(link["dma_bytes"])
            for group, _, _, _ in CHAIN_GROUPS:
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
    tau0_spec, slope_spec = model["transport_spec"]["tau0_us"], model["transport_spec"]["slope"]
    for case in scan_result["cases"]:
        c_b = model["c_B"][case["dimension"]]["slope"]                       # us per particle
        case_c_b = float(np.mean([item["c_B_ns"] for item in case["k1"]])) / 1e3   # this case's K = 1 T_B / N
        per_link = []
        for name, link in case["k2"]["links"].items():
            receiver = case["k2"]["sims"][link["receiver"]]
            n_b, w, b = receiver["n_B"], receiver["own_columns"], receiver["band_widths"][2]
            t_b_pred = c_b * n_b
            # two-term check with this case's K = 1 split: correction + density sweeps outside band 2
            # (n_correction_interior, n_density_deep_interior), force outside band 3 (n_B)
            two_term = float(np.mean([
                item["b_correction_density_p50_us"] / item["N"]
                * 0.5 * (receiver["n_correction_interior"] + receiver["n_density_deep_interior"])
                + item["b_force_p50_us"] / item["N"] * n_b
                for item in case["k1"]])) if all("b_force_p50_us" in item for item in case["k1"]) else float("nan")
            per_link.append({"link": name, "n_B": n_b, "w": w, "b": b,
                             "r_pred": (tau0 + slope * link["host_copy_bytes_p50"]) / t_b_pred,
                             "r_pred_spec": (tau0_spec + slope_spec * link["host_copy_bytes_p50"]) / t_b_pred,
                             "r_pred_case_c_B": (tau0 + slope * link["host_copy_bytes_p50"]) / (case_c_b * n_b),
                             "r_closed_form": tau0 / t_b_pred + 64.0 * slope / (c_b * (w - b)),
                             "r_chain_p50": link["r_chain_p50"], "r_p50": link["r_p50"],
                             "T_B_pred_us": t_b_pred, "T_B_measured_us": receiver["T_B_p50_us"],
                             "T_B_deviation": receiver["T_B_p50_us"] / t_b_pred - 1.0,
                             "T_B_two_term_us": two_term,
                             "T_B_two_term_deviation": receiver["T_B_p50_us"] / two_term - 1.0,
                             "k1_c_B_ns": 1e3 * case_c_b})
        worst = max(per_link, key=lambda item: item["r_pred"])
        case["prediction"] = {"links": per_link, "r_pred": worst["r_pred"], "r_closed_form": worst["r_closed_form"],
                              "r_pred_spec": max(item["r_pred_spec"] for item in per_link),
                              "r_pred_case_c_B": max(item["r_pred_case_c_B"] for item in per_link)}


def overhead(campaign: pathlib.Path) -> list:
    records = read_results(campaign)
    rows = []
    for name in dict.fromkeys(r["case"] for r in records):
        pairs = []
        for trial in sorted({r["trial"] for r in records if r["case"] == name}):
            on = [r for r in records if r["case"] == name and r["trial"] == trial and r["arm"] == "on"]
            off = [r for r in records if r["case"] == name and r["trial"] == trial and r["arm"] == "off"]
            if on and off:
                fps_on, fps_off = precise_fps(on[0], campaign), precise_fps(off[0], campaign)
                pairs.append({"trial": trial, "on": fps_on, "off": fps_off, "ratio": fps_on / fps_off - 1.0,
                              "order": "off->on" if trial % 2 == 1 else "on->off"})
        ratios = np.array([pair["ratio"] for pair in pairs])
        rows.append({"case": name, "pairs": pairs, "mean": float(ratios.mean()) if len(ratios) else float("nan"),
                     "std": float(ratios.std(ddof=1)) if len(ratios) > 1 else float("nan")})
    return rows


def phase1_checks(named_runs: list) -> dict:
    """Phase-1 checks a-c of each named K = 2 trace: coverage, causal order (per link margins), DMA medians
    and bandwidths against E24."""
    out = {}
    for name, path in named_runs:
        run = load_run(path)
        summary = run_summary(run)
        entry = {"dir": str(path), "dest_signal": run["dest_signal"], "coverage": summary["coverage"],
                 "causal": summary["checks"]["causal"], "steady_fps": summary["steady_fps"], "dma": {}}
        reference = E24_DMA_P50.get(name.split("@")[0])          # NAME@label: same E24 reference
        for position, (link_name, link) in enumerate(sorted(summary["links"].items())):
            row = {"dma_bytes": link["dma_bytes"], "readback_us": link["readback_dma_p50"], "upload_us": link["upload_dma_p50"],
                   "readback_GBps": link["dma_bytes"] / (1e3 * link["readback_dma_p50"]),
                   "upload_GBps": link["dma_bytes"] / (1e3 * link["upload_dma_p50"])}
            if reference is not None:
                row.update({"readback_e24_us": reference[position], "upload_e24_us": reference[2 + position],
                            "readback_e24_GBps": link["dma_bytes"] / (1e3 * reference[position]),
                            "upload_e24_GBps": link["dma_bytes"] / (1e3 * reference[2 + position]),
                            "readback_vs_e24": link["readback_dma_p50"] / reference[position] - 1.0,
                            "upload_vs_e24": link["upload_dma_p50"] / reference[2 + position] - 1.0})
            entry["dma"][link_name] = row
        out[name] = entry
    return out


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
    for group, color, label in (("readback", "tab:green", "readback (start, DMA, barrier) / DMA bytes"),
                                ("host", "tab:red", "host (wake, stamp, memcpy) / host bytes"),
                                ("upload", "tab:purple", "upload (signal, start, DMA) / DMA bytes")):
        fit = model["hops"][group]
        axes[1].plot(grid / 1024, fit["intercept"] + fit["slope"] * grid, "--", color=color, lw=1,
                     label=f"{label}: {fit['intercept']:.0f} us + bytes / {fit['bandwidth_GBps']:.1f} GB/s")
    for axis, location in zip(axes, ("upper left", "upper left")):
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
    axis.set_xlabel("r_pred(t_chain) = (tau0 + B_host / beta) / (c_B n_B)")
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
        for key, style in (("r_chain", dict(mfc=colors[dimension])), ("r", dict(mfc="none"))):
            axes[0].plot(k2[key + "_worst_p50"], case["eta"], markers[dimension], color=colors[dimension], ms=6, **style)
            axes[0].plot([k2[key + "_worst_p50"], k2[key + "_worst_p95"]], [case["eta"]] * 2, "-",
                         color=colors[dimension], lw=0.7, alpha=0.6)
        axes[0].plot([k2["r_chain_worst_p50"]] * 2, [case["eta_min"], case["eta_traced"]], ":",
                     color=colors[dimension], lw=0.8)
        axes[0].annotate(case["case"], (k2["r_chain_worst_p50"], case["eta"]), fontsize=7, xytext=(-6, -11),
                         textcoords="offset points")
    axes[0].axvline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("p50 per step, worse link (line to p95): filled r_chain, open r = t_tr / T_B")
    axes[0].set_ylabel("eta = fps_K2 / (2 mean fps_K1), trace off, one trial\n"
                       "(dotted: from eta_min to eta of the traced runs)")
    axes[0].grid(alpha=0.3, which="both")
    labels, data = [], []
    for case in scan_result["cases"]:
        for name, values in link_metrics(case["k2_run"]).items():
            labels.append(f"{case['case']}\n{name.replace('_to_', '>')}")
            data.append(values["phase_offset_ratio"])
    axes[1].boxplot(data, whis=(5, 95), showfliers=False)
    for position, case_values in enumerate(data, start=1):
        axes[1].plot(position, np.median(np.abs(case_values)), "v", color="tab:red", ms=4)
    axes[1].set_xticks(range(1, len(labels) + 1), labels, rotation=90, fontsize=6)
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_ylabel("D' / T_B (receiver B start - sender send end);\nred: median |D'| / T_B")
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


def checks_table(checks: dict) -> str:
    lines = ["## Phase-1 checks (a: causal order per link, b: coverage, c: DMA vs E24)\n",
             "| case | coverage (steady) | missing-tick rows | chains with a violation | link | pair | min gap us | "
             "tolerance us | margin us |", "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in checks.items():
        causal = entry["causal"]
        for link_name, pairs in causal["links"].items():
            for pair, values in pairs.items():
                if values["margin_us"] is None:
                    continue
                lines.append(f"| {name} | {fmt(100 * entry['coverage']['steady_coverage'], 2)} % | "
                             f"{entry['coverage']['device_rows_missing_ticks']} | "
                             f"{fmt(100 * causal['chains_with_violation_fraction'], 3)} % | {link_name} | {pair} | "
                             f"{fmt(values['min_gap_us'], 2)} | {fmt(values['tolerance_us'], 2)} | {fmt(values['margin_us'], 2)} |")
    lines += ["\n| case | GPU | clock map residual rms / max ns | version-1 map rms / max ns | pairs kept / all | "
              "driver maxDeviation min ns | worker_signal after upload_start |", "|---|---|---|---|---|---|---|"]
    for name, entry in checks.items():
        causal = entry["causal"]
        for gpu in range(len(causal["clock_fit_residual_rms_ns"])):
            lines.append(f"| {name} | {gpu} | {fmt(causal['clock_fit_residual_rms_ns'][gpu], 0)} / "
                         f"{fmt(causal['clock_fit_residual_max_ns'][gpu], 0)} | "
                         f"{fmt(causal['clock_fit_version_1_residual_rms_ns'][gpu], 0)} / "
                         f"{fmt(causal['clock_fit_version_1_residual_max_ns'][gpu], 0)} | "
                         f"{causal['clock_fit_samples_kept'][gpu]} / {causal['clock_fit_samples'][gpu]} | "
                         f"{fmt(causal['driver_max_deviation_min_ns'][gpu], 0)} | "
                         f"{fmt(100 * causal['signal_return_after_upload_start_fraction'], 2)} % |")
    lines += ["\n| case | link | DMA bytes | readback us (GB/s) | E24 | vs E24 | upload us (GB/s) | E24 | vs E24 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in checks.items():
        for link_name, row in entry["dma"].items():
            lines.append(f"| {name} | {link_name} | {row['dma_bytes']:,.0f} | {fmt(row['readback_us'])} "
                         f"({fmt(row['readback_GBps'], 2)}) | {fmt(row.get('readback_e24_us'))} "
                         f"({fmt(row.get('readback_e24_GBps'), 2)}) | {fmt(100 * row['readback_vs_e24'], 1) if 'readback_vs_e24' in row else '—'} % | "
                         f"{fmt(row['upload_us'])} ({fmt(row['upload_GBps'], 2)}) | {fmt(row.get('upload_e24_us'))} "
                         f"({fmt(row.get('upload_e24_GBps'), 2)}) | {fmt(100 * row['upload_vs_e24'], 1) if 'upload_vs_e24' in row else '—'} % |")
    return "\n".join(lines) + "\n"


def tables(scan_result: dict, model: dict, overhead_rows: list) -> str:
    lines = ["## Runs (K = 2 trace on; eta from the runs without the trace, ONE trial)\n",
             "| case | w (s0 / s1) | n_B (s0 / s1) | B_host KiB | t_tr p50 / p95 us | t_chain p50 / p95 us | T_B us (s0 / s1) "
             "| r p50 / p95 | r_pred(t_tr) | r_chain p50 / p95 | r_pred(t_chain) | closed form | r_pred(t_chain), case c_B "
             "| r_receiver p95 | exposed steps | eta (off) | median abs(D') / T_B |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        k2 = case["k2"]
        sims, links = k2["sims"], list(k2["links"].values())
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} / {} | {} | {} / {} | {} | {} | {} | {} | {} | {} | {} |".format(
            case["case"], " / ".join(str(s["own_columns"]) for s in sims), " / ".join(f"{s['n_B']:,}" for s in sims),
            " / ".join(fmt(l["host_copy_bytes_p50"] / 1024) for l in links),
            " / ".join(f"{fmt(l['t_tr_p50'])}/{fmt(l['t_tr_p95'])}" for l in links),
            " / ".join(f"{fmt(l['t_chain_p50'])}/{fmt(l['t_chain_p95'])}" for l in links),
            " / ".join(fmt(s["T_B_p50_us"]) for s in sims),
            fmt(k2["r_worst_p50"], 3), fmt(k2["r_worst_p95"], 3), fmt(case["prediction"]["r_pred_spec"], 3),
            fmt(k2["r_chain_worst_p50"], 3), fmt(k2["r_chain_worst_p95"], 3),
            fmt(case["prediction"]["r_pred"], 3), fmt(case["prediction"]["r_closed_form"], 3),
            fmt(case["prediction"]["r_pred_case_c_B"], 3), fmt(k2["r_receiver_worst_p95"], 3),
            fmt(100 * k2["exposed_steps_fraction"], 1) + " %", fmt(100 * case["eta"], 1) + " %",
            " / ".join(fmt(l["phase_offset_abs_ratio_p50"], 2) for l in links)))
    lines += ["\n## eta, one trial per case (off = without the trace; on = with it, a second sample)\n",
              "| case | fps K = 1 GPU 0 / 1 (off) | K = 1 spread | fps K = 2 (off) | eta (off) | eta_min (off) | "
              "fps K = 1 GPU 0 / 1 (on) | fps K = 2 (on) | eta (on) | eta_min (on) | K = 2 on / off - 1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        lines.append(f"| {case['case']} | {' / '.join(fmt(v, 2) for v in case['fps_k1'])} | "
                     f"{fmt(100 * case['k1_spread'], 1)} % | {fmt(case['fps_k2'], 2)} | {fmt(100 * case['eta'], 1)} % | "
                     f"{fmt(100 * case['eta_min'], 1)} % | {' / '.join(fmt(v, 2) for v in case['fps_k1_traced'])} | "
                     f"{fmt(case['fps_k2_traced'], 2)} | {fmt(100 * case['eta_traced'], 1)} % | "
                     f"{fmt(100 * case['eta_min_traced'], 1)} % | {fmt(100 * case['k2_on_off'], 2)} % |")
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
    lines += ["\n## r_pred per link (worse link per case in the runs table)\n",
              "| case | link | r_chain p50 | r_pred(t_chain) | r_pred / r_chain - 1 | closed form | r p50 | r_pred(t_tr) |",
              "|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for item in case["prediction"]["links"]:
            lines.append(f"| {case['case']} | {item['link']} | {fmt(item['r_chain_p50'], 3)} | {fmt(item['r_pred'], 3)} | "
                         f"{fmt(100 * (item['r_pred'] / item['r_chain_p50'] - 1), 1)} % | {fmt(item['r_closed_form'], 3)} | "
                         f"{fmt(item['r_p50'], 3)} | {fmt(item['r_pred_spec'], 3)} |")
    lines += ["\n## T_B vs c_B n_B (K = 2)\n",
              "| case | link | n_B | c_B n_B us | T_B us | deviation | two-term us | deviation | K = 1 c_B of the case ns |",
              "|---|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for item in case["prediction"]["links"]:
            lines.append(f"| {case['case']} | {item['link']} | {item['n_B']:,} | {fmt(item['T_B_pred_us'])} | "
                         f"{fmt(item['T_B_measured_us'])} | {fmt(100 * item['T_B_deviation'], 1)} % | "
                         f"{fmt(item['T_B_two_term_us'])} | {fmt(100 * item['T_B_two_term_deviation'], 1)} % | "
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
    lines += ["\n## t_tr hops p50 us (copy_to_upload = signal + signal_to_upload where the trace has worker_dest_signal)\n",
              "| case | link | " + " | ".join(ALL_HOPS) + " | t_tr | t_chain |",
              "|---|---|" + "---|" * (len(ALL_HOPS) + 2)]
    for case in scan_result["cases"]:
        for name, link in case["k2"]["links"].items():
            lines.append(f"| {case['case']} | {name} | " + " | ".join(fmt(link.get(hop + '_p50')) for hop in ALL_HOPS)
                         + f" | {fmt(link['t_tr_p50'])} | {fmt(link['t_chain_p50'])} |")
    lines += ["\n## Step composition, medians us (traced runs; K = 1 = mean of GPU 0 and 1)\n",
              "| case | run | " + " | ".join(CYCLE_PARTS) + " | period p50 | period mean | eta (off) |",
              "|---|---|" + "---|" * (len(CYCLE_PARTS) + 3)]
    for case in scan_result["cases"]:
        k1 = {key: float(np.mean([item[key] for item in case["cycles"]["k1"]]))
              for key in CYCLE_PARTS + ("period", "period_mean")}
        lines.append(f"| {case['case']} | K = 1 | " + " | ".join(fmt(k1[key]) for key in CYCLE_PARTS)
                     + f" | {fmt(k1['period'])} | {fmt(k1['period_mean'])} | |")
        lines.append(f"| {case['case']} | K = 1 / 2 | " + " | ".join(fmt(k1[key] / 2) for key in CYCLE_PARTS)
                     + f" | {fmt(k1['period'] / 2)} | {fmt(k1['period_mean'] / 2)} | |")
        for item in case["cycles"]["k2"]:
            lines.append(f"| {case['case']} | K = 2 s{item['sim']} | " + " | ".join(fmt(item[key]) for key in CYCLE_PARTS)
                         + f" | {fmt(item['period'])} | {fmt(item['period_mean'])} | {fmt(100 * case['eta'], 1)} % |")
    lines += ["\n## Per sim (K = 2 trace on)\n",
              "| case | sim | T_B p50 / p95 us | phase A p50 us | phase C p50 us | C->A gap p50 / p95 us |", "|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for sim in case["k2"]["sims"]:
            lines.append(f"| {case['case']} | {sim['sim']} | {fmt(sim['T_B_p50_us'])} / {fmt(sim['T_B_p95_us'])} | "
                         f"{fmt(sim['phase_a_p50_us'])} | {fmt(sim['phase_c_p50_us'])} | "
                         f"{fmt(sim['c_to_a_gap_p50_us'])} / {fmt(sim['c_to_a_gap_p95_us'])} |")
    lines += ["\n## Phase offset and exposure per link (K = 2 trace on)\n",
              "| case | link | D' p50 us | D' / T_B p5 / p50 / p95 | median abs(D') / T_B | exposed steps | "
              "exposure mean us (all steps) | B->C wait p50 / p95 us |", "|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        for name, link in case["k2"]["links"].items():
            lines.append(f"| {case['case']} | {name} | {fmt(link['phase_offset_p50'])} | "
                         f"{fmt(link['phase_offset_ratio_p5'], 2)} / {fmt(link['phase_offset_ratio_p50'], 2)} / "
                         f"{fmt(link['phase_offset_ratio_p95'], 2)} | {fmt(link['phase_offset_abs_ratio_p50'], 2)} | "
                         f"{fmt(100 * link['exposed_fraction'], 1)} % | {fmt(link['exposure_mean_us'])} | "
                         f"{fmt(link['b_to_c_wait_p50'])} / {fmt(link['b_to_c_wait_p95'])} |")
    lines += ["\n## Checks per K = 2 run (scan)\n",
              "| case | coverage (steady) | chains with a violation | min margin readback_end -> worker us (s0>s1 / s1>s0) | "
              "min margin copy -> upload_start us | clock map residual rms / max ns | version-1 map rms / max ns | "
              "worker_signal after upload_start |",
              "|---|---|---|---|---|---|---|---|"]
    for case in scan_result["cases"]:
        check = case["k2"]["checks"]["causal"]
        margins = {}
        for name, pairs in check["links"].items():
            for pair, values in pairs.items():
                margins.setdefault(pair, []).append(values["margin_us"])
        upload_pair = next(pair for pair in margins if pair.endswith("<= upload_start"))
        lines.append(f"| {case['case']} | {fmt(100 * case['k2']['coverage']['steady_coverage'], 2)} % | "
                     f"{fmt(100 * check['chains_with_violation_fraction'], 3)} % | "
                     f"{' / '.join(fmt(v, 2) for v in margins['readback_end <= worker_source_wait'])} | "
                     f"{' / '.join(fmt(v, 2) for v in margins[upload_pair])} | "
                     + "; ".join(f"{fmt(a, 0)} / {fmt(b, 0)}" for a, b in zip(check["clock_fit_residual_rms_ns"],
                                                                         check["clock_fit_residual_max_ns"])) + " | "
                     + "; ".join(f"{fmt(a, 0)} / {fmt(b, 0)}" for a, b in zip(check["clock_fit_version_1_residual_rms_ns"],
                                                                         check["clock_fit_version_1_residual_max_ns"])) + " | "
                     + f"{fmt(100 * check['signal_return_after_upload_start_fraction'], 2)} % |")
    if overhead_rows:
        lines += ["\n## Overhead (trace on vs off, K = 2)\n", "| case | trial | order | off fps | on fps | on / off - 1 |",
                  "|---|---|---|---|---|---|"]
        for row in overhead_rows:
            for pair in row["pairs"]:
                lines.append(f"| {row['case']} | {pair['trial']} | {pair['order']} | {fmt(pair['off'], 2)} | "
                             f"{fmt(pair['on'], 2)} | {fmt(100 * pair['ratio'], 2)} % |")
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
    parser.add_argument("--checks", nargs="*", default=[], metavar="NAME=DIR",
                        help="phase-1 check traces (K = 2); NAME (before an optional @label) selects the E24 "
                             "reference (2d_1m, 2d_16m, 3d_8m)")
    parser.add_argument("--run", default=None, help="one trace directory: print its summary and checks")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    if arguments.run:
        print(json.dumps(jsonable(run_summary(load_run(arguments.run))), indent=1))
        return 0
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = phase1_checks([(item.split("=", 1)[0], pathlib.Path(item.split("=", 1)[1])) for item in arguments.checks])
    overhead_rows = overhead(pathlib.Path(arguments.overhead)) if arguments.overhead else []
    scan_result = analyze_scan(pathlib.Path(arguments.scan)) if arguments.scan else {"cases": []}
    model = constants(scan_result) if scan_result["cases"] else {}
    figure_paths = []
    if scan_result["cases"]:
        predictions(scan_result, model)
        figure_paths = figures(scan_result, model, out)
    report = (checks_table(checks) if checks else "") + (tables(scan_result, model, overhead_rows) if scan_result["cases"] else "")
    (out / "e29_tables.md").write_text(report, encoding="utf-8")
    summary = {"model": model, "overhead": overhead_rows, "phase1_checks": checks,
               "cases": [{key: value for key, value in case.items() if key != "k2_run"} for case in scan_result["cases"]],
               "figures": [str(path) for path in figure_paths]}
    (out / "e29_summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
