"""
e8_loss_sources.py — E8 phase 2, deliverable 4: the fig:loss quantities of every traced chain point (every K; the
8-card points are the figure's), per the E7 definitions (e7_loss_sources.py / e7_full_summary.own_chain):
  t_chain   per link and step, upload_end - send_end minus the receiver's two waits (step_trace_model.link_metrics);
  t_own     t_chain minus the transfer queues' idle waits (e7_full_summary.transfer_queue_idle): the chain's own work;
  r_own     per step the worst link's t_own / T_B (T_B = the receiver's phase B of that step), median over the steady
            steps (and p95);
  r_chain   the same with t_chain;
  T_B       per card the median phase B, mean over the cards;
  t_own and its three parts (readback / host / upload of the chain groups) = the slowest link per step (by t_own),
            medians over the steps; the host part is the chain's host hop (readback_end -> worker_copy minus the
            receiver waits), its memcpy inside it.

Usage:
    python docs/cluster_v6/scripts/e8_loss_sources.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/loss_sources.md, loss_sources.json, loss_sources.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402
import e7_full_summary as summary  # noqa: E402
from experiment.v7.analysis import step_trace_model  # noqa: E402


def analyze(trace_dir) -> dict:
    run = common.load_trace(trace_dir)
    metrics = step_trace_model.link_metrics(run)
    idle = summary.transfer_queue_idle(run)
    links = {}
    for name, values in metrics.items():
        pairs = [idle.get((name, int(step)), (math.nan, math.nan)) for step in values["step"]]
        readback_idle = np.array([pair[0] for pair in pairs])
        upload_idle = np.array([pair[1] for pair in pairs])
        values["t_own"] = values["t_chain"] - readback_idle - upload_idle
        values["r_own"] = values["t_own"] / values["T_B"]
        values["own_readback"] = values["chain_readback"] - readback_idle
        values["own_host"] = values["chain_host"]
        values["own_upload"] = values["chain_upload"] - upload_idle
        links[name] = {"sender": values["sender"], "receiver": values["receiver"], "steps": int(len(values["step"])),
                       **{key + "_p50": float(np.nanmedian(values[key])) for key in ("t_chain", "t_own", "T_B", "r_chain", "r_own",
                                                                                      "own_readback", "own_host", "memcpy", "own_upload",
                                                                                      "host_copy_bytes")}}
    # worst link per step by t_own: its parts
    steps = sorted(set.intersection(*[set(values["step"].astype(int).tolist()) for values in metrics.values()])) if metrics else []
    by_step = {}
    for name, values in metrics.items():
        for index, step in enumerate(values["step"].astype(int)):
            by_step.setdefault(int(step), []).append((values["t_own"][index], values["t_chain"][index], values["T_B"][index],
                                                      values["own_readback"][index], values["own_host"][index], values["memcpy"][index],
                                                      values["own_upload"][index]))
    worst = []
    for step in steps:
        candidates = by_step.get(step, [])
        if candidates:
            worst.append(max(candidates, key=lambda item: item[0] if np.isfinite(item[0]) else -1))
    worst = np.array(worst) if worst else np.zeros((0, 7))
    r_own = step_trace_model.worst_link(metrics, "r_own") if metrics else np.array([])
    r_chain = step_trace_model.worst_link(metrics, "r_chain") if metrics else np.array([])
    t_b = [float(np.nanmedian(values["T_B"])) for values in metrics.values()]
    sims_t_b = {}
    for sim in common.sims_of(run):
        rows = common.sim_rows(run, sim)
        keep = common.steady_mask(rows, run["meta"]["warmup"])
        sims_t_b[sim] = float(np.median((rows["b_end"] - rows["b_start"])[keep] / common.NS_PER_MS)) if keep.any() else None
    quantile = step_trace_model.quantiles
    return {"trace": str(trace_dir), "case": run["meta"].get("case"), "K": run["meta"].get("K"), "links": links,
            "steps": len(steps), "r_own_p50": quantile(r_own)[0] if len(r_own) else None, "r_own_p95": quantile(r_own)[1] if len(r_own) else None,
            "r_chain_p50": quantile(r_chain)[0] if len(r_chain) else None, "r_chain_p95": quantile(r_chain)[1] if len(r_chain) else None,
            "T_B_mean_of_cards_ms": statistics.fmean(v for v in sims_t_b.values() if v is not None) if any(v is not None for v in sims_t_b.values()) else None,
            "T_B_per_card_ms": sims_t_b, "T_B_receiver_median_links_us": statistics.median(t_b) if t_b else None,
            "worst_link_per_step_medians_us": {key: float(np.nanmedian(worst[:, index])) if len(worst) else None
                                               for index, key in enumerate(("t_own", "t_chain", "T_B", "readback", "host", "memcpy", "upload"))}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = []
    for directory in arguments.job:
        job = common.load_job(directory)
        for name, point in sorted(common.points_of(job).items(), key=lambda item: (item[1]["slabs"], item[1]["case"])):
            if not point.get("trace_dir") or point["slabs"] < 2:
                continue
            try:
                result = analyze(point["trace_dir"])
            except Exception as error:                                      # noqa: BLE001
                result = {"error": repr(error)}
            result.update({"point": name, "job": job["name"], "case": point["case"], "K": point["slabs"],
                           "per_card": common.particles_of(point["case"]) / point["slabs"] if point["case"] in common.CASE_LABELS else None})
            results.append(result)
    lines = ["## fig:loss quantities (E8 phase 2; us unless noted; worst link per step = largest t_own)", "",
             "| point | K | per card | T_B mean of cards (ms) | r_own p50 / p95 | r_chain p50 / p95 | t_own (worst link, median) | "
             "its readback / host (memcpy) / upload | t_chain (worst link) | links | steps |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_csv = []
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['point']} | {r['K']} | | failed: {r['error']} | | | | | | | |")
            continue
        w = r["worst_link_per_step_medians_us"]
        lines.append(f"| {r['point']} | {r['K']} | {(r['per_card'] or 0) / 1e6:.2f}M | {common.fmt(r['T_B_mean_of_cards_ms'])} | "
                     f"{common.fmt(r['r_own_p50'])} / {common.fmt(r['r_own_p95'])} | {common.fmt(r['r_chain_p50'])} / {common.fmt(r['r_chain_p95'])} | "
                     f"{common.fmt(w['t_own'], 0)} | {common.fmt(w['readback'], 0)} / {common.fmt(w['host'], 0)} ({common.fmt(w['memcpy'], 0)}) / {common.fmt(w['upload'], 0)} | "
                     f"{common.fmt(w['t_chain'], 0)} | {len(r['links'])} | {r['steps']} |")
        rows_csv.append([r["point"], r["job"], r["case"], r["K"], r["per_card"], r["T_B_mean_of_cards_ms"], r["r_own_p50"], r["r_own_p95"],
                         r["r_chain_p50"], r["r_chain_p95"], w["t_own"], w["readback"], w["host"], w["memcpy"], w["upload"], w["t_chain"]])
    lines += ["", "### Per link (p50, us)", "", "| point | link | sender → receiver | t_chain | t_own | T_B | r_chain | r_own | own readback | host | memcpy | own upload | host copy B |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for name, link in (r.get("links") or {}).items():
            lines.append(f"| {r['point']} | {name} | {link['sender']} → {link['receiver']} | {common.fmt(link['t_chain_p50'], 0)} | {common.fmt(link['t_own_p50'], 0)} | "
                         f"{common.fmt(link['T_B_p50'], 0)} | {common.fmt(link['r_chain_p50'])} | {common.fmt(link['r_own_p50'])} | {common.fmt(link['own_readback_p50'], 0)} | "
                         f"{common.fmt(link['own_host_p50'], 0)} | {common.fmt(link['memcpy_p50'], 0)} | {common.fmt(link['own_upload_p50'], 0)} | {common.fmt(link['host_copy_bytes_p50'], 0)} |")
    text = "\n".join(lines)
    print(text[:3000])
    common.write_text(out / "loss_sources.md", text)
    common.write_json(out / "loss_sources.json", results)
    with (out / "loss_sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "job", "case", "K", "per_card", "T_B_mean_ms", "r_own_p50", "r_own_p95", "r_chain_p50", "r_chain_p95",
                         "t_own_us", "readback_us", "host_us", "memcpy_us", "upload_us", "t_chain_us"])
        writer.writerows(rows_csv)
    print(f"[e8_loss_sources] {len(results)} traced points -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
