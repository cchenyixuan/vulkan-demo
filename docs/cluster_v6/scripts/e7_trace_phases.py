"""e7_trace_phases.py: per-card phases of step-traced chain runs (revision 10, user 2026-10-20: the two traced K = 5
runs of 2d_n2840_k5 with the new and the old weights).

For every trace directory, per sim (= card) the medians over the steady steps (step >= warmup whose next step is present;
the defrag step is one of them and is not dropped) of T_A, T_B, T_C, busy = T_A + T_B + T_C (per step, then the median),
the waits A -> B, B -> C and C -> next A, and the step period a_start(n + 1) - a_start(n) (ms; GPU timestamps on the host
clock, experiment/v7/analysis/step_trace_model), with each sim's own particles and own columns (run_meta.json). The
spread of busy across the cards (max / min - 1) and the slowest card are printed per trace.

Usage:
    python docs/cluster_v6/scripts/e7_trace_phases.py NAME=TRACE_DIR ... [--md OUT] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from experiment.v7.analysis import step_trace_model  # noqa: E402

COLUMNS = ("a", "a_to_b", "b", "b_to_c", "c", "c_to_next_a", "busy", "period")
HEADERS = ("T_A", "A→B", "T_B", "B→C", "T_C", "C→A'", "busy", "period")


def phases(trace_dir: pathlib.Path) -> dict:
    run = step_trace_model.load_run(trace_dir)
    meta = run["meta"]
    warmup = meta["warmup"]
    device = run["device"]
    slabs = (meta.get("slabs") or {}).get("per_sim", [])
    device_map = meta.get("device_map") or meta.get("devices") or []
    sims = []
    for sim in sorted(int(value) for value in set(device["sim"].astype(int))):
        rows = step_trace_model.select(device, (device["sim"] == sim) & (device["complete"] == 1))
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
        parts["busy"] = parts["a"] + parts["b"] + parts["c"]
        entry = {"sim": sim, "device": (device_map[sim] if sim < len(device_map) else None), "steps": int(steady.sum()),
                 "first_step": int(steps[steady][0]) if steady.any() else None,
                 "last_step": int(steps[steady][-1]) if steady.any() else None,
                 **{key: float(np.median(value[steady]) / 1e3) for key, value in parts.items()},
                 "period_mean": float(np.mean(parts["period"][steady]) / 1e3),
                 "busy_p95": float(np.percentile(parts["busy"][steady], 95) / 1e3)}
        if sim < len(slabs):
            entry["own_particles"] = slabs[sim].get("own_particles")
            entry["own_columns"] = slabs[sim].get("own_columns")
        sims.append(entry)
    busy = [entry["busy"] for entry in sims]
    slowest = max(sims, key=lambda entry: entry["busy"])
    return {"trace": str(trace_dir), "case": meta.get("case"), "K": meta.get("K"), "warmup": warmup,
            "steady_fps": (meta.get("result") or {}).get("steady_fps"), "sims": sims,
            "busy_spread": max(busy) / min(busy) - 1.0, "slowest_sim": slowest["sim"],
            "period_mean_over_sims": float(np.mean([entry["period_mean"] for entry in sims]))}


def markdown(name: str, result: dict) -> str:
    lines = [f"**{name}** — {result['case']} K = {result['K']}, steady steps {result['sims'][0]['first_step']}–"
             f"{result['sims'][0]['last_step']} ({result['sims'][0]['steps']}), fps {result['steady_fps']}; "
             f"busy max / min − 1 = {result['busy_spread'] * 100:.2f} % (slowest sim {result['slowest_sim']}); "
             f"period mean over sims {result['period_mean_over_sims']:.3f} ms", "",
             "| sim | device | own particles | columns | " + " | ".join(HEADERS) + " | period mean | busy p95 |",
             "|---|---|---|---|" + "---|" * len(HEADERS) + "---|---|"]
    for entry in result["sims"]:
        particles = entry.get("own_particles")
        lines.append(f"| {entry['sim']} | {entry.get('device', '')} | {particles:,} | {entry.get('own_columns', '')} | "
                     + " | ".join(f"{entry[key]:.3f}" for key in COLUMNS)
                     + f" | {entry['period_mean']:.3f} | {entry['busy_p95']:.3f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", metavar="NAME=TRACE_DIR")
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    results, blocks = {}, []
    for item in arguments.traces:
        name, _, path = item.partition("=")
        results[name] = phases(pathlib.Path(path))
        blocks.append(markdown(name, results[name]))
    text = "\n\n".join(blocks)
    print(text)
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
