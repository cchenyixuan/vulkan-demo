"""
e8_bytes_check.py — E8 phase 2, deliverable 6: eq:bytes against the measured host copy, step by step.

    B_link = 32 (n0 + n1) + 44 n_mig + 16 NY NZ + 16

with n0 / n1 = the inner / outer replica counts and n_mig the migrants of the packet (the sender's header counters,
recorded per frame by the transport worker with V7_POOL_PEAKS=1 and written by run_chain_v6 --pool-demand to
raw/<label>/pool_demand_<link>.csv: columns index, inner, outer, migrant; index = the frame as the worker counts them
from its first), NY NZ = the voxel face of the case (e8_vram.lattice: (n / 4 + 4)^2), against host_copy_bytes of the
same link and step in the trace's steps_link.csv. Every traced chain point with a raw directory is checked (the five
cubes' K = 8 traces are the deliverable); per link: steps compared, exact matches, max |deviation| (bytes and relative),
mean deviation, the first mismatching steps.

Usage:
    python docs/cluster_v6/scripts/e8_bytes_check.py --job DIR [--job DIR ...] --out-dir OUT [--max-steps-listed 5]
Writes OUT/bytes_check.md, bytes_check.json.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402


def predicted(n0, n1, n_mig, face: int) -> np.ndarray:
    return 32.0 * (n0 + n1) + 44.0 * n_mig + 16.0 * face + 16.0


def check(trace_dir, raw_dir, case: str, listed: int) -> dict:
    run = common.load_trace(trace_dir)
    demand = common.read_pool_demand(raw_dir)
    face = common.lattice_of(case)["face"]
    link_table = run["link"]
    out = {"trace": str(trace_dir), "raw": str(raw_dir), "case": case, "K": run["meta"].get("K"), "face": face, "links": {}}
    for name in sorted(set(link_table["link"])):
        rows = {key: value[link_table["link"] == name] for key, value in link_table.items()}
        order = np.argsort(rows["step"])
        steps = rows["step"][order].astype(int)
        measured = rows["host_copy_bytes"][order].astype(np.float64)
        counts = demand.get(name)
        if counts is None:
            out["links"][name] = {"error": "no pool_demand file"}
            continue
        index = counts["index"].astype(int)
        inner, outer = counts.get("inner"), counts.get("outer")
        migrant = counts.get("migrant", np.zeros_like(index, dtype=np.float64))
        if inner is None or outer is None:
            out["links"][name] = {"error": f"regions {list(counts)}"}
            continue
        by_index = {int(i): k for k, i in enumerate(index)}
        pred, meas, used = [], [], []
        for k, step in enumerate(steps):
            j = by_index.get(int(step))
            if j is None or not np.isfinite(measured[k]):
                continue
            pred.append(predicted(inner[j], outer[j], migrant[j], face))
            meas.append(measured[k])
            used.append(int(step))
        pred, meas = np.array(pred), np.array(meas)
        if not len(pred):
            out["links"][name] = {"error": "no overlapping steps between the trace and the pool demand"}
            continue
        deviation = meas - pred
        mismatches = [int(step) for step, d in zip(used, deviation) if d != 0]
        out["links"][name] = {"steps_compared": int(len(pred)), "exact": int(np.sum(deviation == 0)),
                              "max_abs_deviation_bytes": float(np.max(np.abs(deviation))),
                              "max_relative_deviation": float(np.max(np.abs(deviation) / meas)) if meas.size else None,
                              "mean_deviation_bytes": float(np.mean(deviation)), "measured_p50": float(np.median(meas)),
                              "predicted_p50": float(np.median(pred)), "n0_p50": float(np.median(inner)), "n1_p50": float(np.median(outer)),
                              "n_mig_max": float(np.max(migrant)), "n_mig_mean": float(np.mean(migrant)),
                              "first_mismatches": mismatches[:listed], "mismatch_count": len(mismatches)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps-listed", type=int, default=5)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = []
    for directory in arguments.job:
        job = common.load_job(directory)
        for name, point in sorted(common.points_of(job).items(), key=lambda item: (item[1]["slabs"], item[1]["case"])):
            if not point.get("trace_dir") or not point.get("raw_dir") or point["slabs"] < 2:
                continue
            try:
                result = check(point["trace_dir"], point["raw_dir"], point["case"], arguments.max_steps_listed)
            except Exception as error:                                      # noqa: BLE001
                result = {"error": repr(error)}
            result.update({"point": name, "job": job["name"]})
            results.append(result)
    lines = ["## eq:bytes check (E8 phase 2): B = 32 (n0 + n1) + 44 n_mig + 16 NY NZ + 16 vs the measured host copy, per link and step", "",
             "| point | link | steps | exact | max |dev| (B) | max rel. dev | mean dev (B) | measured p50 (B) | n0 / n1 p50 | n_mig mean / max | first mismatching steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['point']} | failed: {r['error']} | | | | | | | | | |")
            continue
        for name, link in r["links"].items():
            if link.get("error"):
                lines.append(f"| {r['point']} | {name} | {link['error']} | | | | | | | | |")
                continue
            worst = max(worst, link["max_relative_deviation"] or 0.0)
            lines.append(f"| {r['point']} | {name} | {link['steps_compared']} | {link['exact']} | {link['max_abs_deviation_bytes']:,.0f} | "
                         f"{link['max_relative_deviation']:.2e} | {link['mean_deviation_bytes']:,.1f} | {link['measured_p50']:,.0f} | "
                         f"{link['n0_p50']:,.0f} / {link['n1_p50']:,.0f} | {link['n_mig_mean']:.1f} / {link['n_mig_max']:.0f} | {link['first_mismatches'] or '-'} |")
    lines += ["", f"largest relative deviation over every link and step: {worst:.2e}"]
    text = "\n".join(lines)
    print(text[:3000])
    common.write_text(out / "bytes_check.md", text)
    common.write_json(out / "bytes_check.json", {"results": results, "largest_relative_deviation": worst})
    print(f"[e8_bytes_check] {len(results)} traces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
