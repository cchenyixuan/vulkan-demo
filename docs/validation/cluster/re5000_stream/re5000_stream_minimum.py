"""re5000_stream_minimum.py (2026-11-06): psi_min and the primary vortex centre of the Re = 5000, 2500^2 cavity for
tab:cavity5000, by the method of tab:cavity / tab:cavity3200 (paper Tables 4 / 5).

Nothing new is computed here: the paper's Re 1000 / 3200 rows came from experiment/validation/adami_k8_report.py, i.e.
    load_run -> restrict_to_steps(common sample steps) -> analyse_window(..., with_stream_function=True)
        -> cavity_analysis.analyse_run -> cavity_analysis.stream_function_minimum(run, analysis["window"])
and the wall-frame keys psi_min_wall / x_wall / y_wall (summary_row). This script calls exactly those functions on
    5090 + XTX  n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000
    2x5090      n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1
restricted to the 14 sample steps of re5000_mixed/report.md section 4 (33,567,000 ... 34,438,000 every 67,000,
t 100.701 - 103.314). stream_function_minimum takes the particle snapshots inside that window (one per 333,000 steps:
t 100.899 / 101.898 / 102.897, the same steps in both runs), MLS-interpolates u onto a 513 x 513 grid spanning the wall
rows, averages, integrates psi upward from the bottom wall row (psi = 0 there, trapezoid), and refines the interior
minimum (|x|, |y| < 0.3) by a quadratic surface over 5 x 5 points. The 14 samples hold only line data (28 + 28 points,
dense centre lines), so the per-field spread below is over the 3 snapshots, each through the same function with a
window holding only that snapshot.

Also checks that analyse_window's extrema (dense centre lines averaged over the same 14 samples) reproduce section 4.

    .venv/Scripts/python.exe docs/validation/cluster/re5000_stream/re5000_stream_minimum.py
Outputs next to this script: summary.json, report.md (no existing file is touched).
"""
from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.validation import cavity_reference  # noqa: E402
from experiment.validation.adami_k8_report import analyse_window, restrict_to_steps  # noqa: E402
from experiment.validation.cavity_analysis import GRID_POINTS, load_run, stream_function_minimum  # noqa: E402

RUNS = _ROOT / "logs" / "validation" / "cavity_re5000_v7"
CHAINS = {"2x5090": "n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1",
          "5090 + XTX": "n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000"}
FIRST_STEP, LAST_STEP, SAMPLE_STEPS = 33_567_000, 34_438_000, 67_000
STEPS = set(range(FIRST_STEP, LAST_STEP + 1, SAMPLE_STEPS))
MIXED_SUMMARY = _ROOT / "docs" / "validation" / "cluster" / "re5000_mixed" / "summary.json"
OUT = pathlib.Path(__file__).resolve().parent
# adami_k8_report defaults (--steady-tol, --window); they only feed the sliding steady test, not the numbers below
STEADY_TOLERANCE, STEADY_WINDOW = 1.0e-3, 10.0


def analyse_chain(label: str) -> dict:
    run = load_run(CHAINS[label], RUNS)
    missing = sorted(STEPS - {row["step"] for row in run["rows"]})
    if missing:
        raise SystemExit(f"{label}: sample steps missing {missing}")
    restricted = restrict_to_steps(run, STEPS)
    analysis = analyse_window(restricted, STEADY_TOLERANCE, STEADY_WINDOW, True, reynolds=5000)
    dt = float(run["meta"]["dt"])
    snapshot_steps = sorted(int(path.stem[1:]) for path in (run["dir"] / "snapshots").glob("t*.npz")
                            if analysis["window"][0] - 1e-9 <= int(path.stem[1:]) * dt <= analysis["window"][1] + 1e-9)
    per_snapshot = []
    for step in snapshot_steps:
        single = stream_function_minimum(restricted, (step * dt, step * dt))
        assert single["snapshots"] == 1, single
        per_snapshot.append({"step": step, "t": step * dt, **single})
        print(f"[{label}] snapshot {step}: psi {single['psi_min_wall']:.6f} at ({single['x_wall']:.5f}, {single['y_wall']:.5f})", flush=True)
    extrema = {name: {"value": float(item["value"]), "position_wall": float(item["position_wall"])}
               for name, item in analysis["extrema"].items()}
    print(f"[{label}] window {analysis['window']}, {analysis['samples_in_window']} samples, stream {analysis['stream']}", flush=True)
    return {"label": label, "run": CHAINS[label], "window": list(analysis["window"]), "samples": analysis["samples_in_window"],
            "sample_steps": [row["step"] for row in restricted["rows"]], "spacing": run["spacing"], "dt": dt,
            "support_radius": float(run["meta"]["support_radius"]), "stream": analysis["stream"],
            "snapshot_steps": snapshot_steps, "per_snapshot": per_snapshot, "extrema": extrema}


def main() -> int:
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = dict(zip(CHAINS, pool.map(analyse_chain, CHAINS)))
    boost, mixed = results["2x5090"], results["5090 + XTX"]
    assert boost["snapshot_steps"] == mixed["snapshot_steps"], (boost["snapshot_steps"], mixed["snapshot_steps"])
    assert boost["sample_steps"] == mixed["sample_steps"]
    reference = cavity_reference.marchi2021(5000)["extrema"]
    marchi = {"psi": reference["psi_min"][0], "x": reference["x_at_psi_min"][0], "y": reference["y_at_psi_min"][0]}
    keys = {"psi": "psi_min_wall", "x": "x_wall", "y": "y_wall"}

    def values(result: dict) -> dict:
        return {short: float(result["stream"][key]) for short, key in keys.items()}

    def spread(result: dict) -> dict:
        return {short: (min(item[key] for item in result["per_snapshot"]), max(item[key] for item in result["per_snapshot"]))
                for short, key in keys.items()}

    difference = {short: values(mixed)[short] - values(boost)[short] for short in keys}
    snapshot_difference = {short: [m[key] - b[key] for m, b in zip(mixed["per_snapshot"], boost["per_snapshot"])]
                           for short, key in keys.items()}

    # section 4 check: analyse_window's extrema against re5000_mixed/summary.json (same 14 samples, dense lines)
    section4 = json.loads(MIXED_SUMMARY.read_text(encoding="utf-8"))["marchi"]["window_common"]
    extrema_check = {}
    for label, source in (("5090 + XTX", "mixed"), ("2x5090", "boost")):
        for name in ("u_min", "v_max", "v_min"):
            mine = results[label]["extrema"][name]
            theirs = section4[source]
            extrema_check[f"{label} {name}"] = {"value_here": mine["value"], "value_section4": theirs[name],
                                                "position_here": mine["position_wall"], "position_section4": theirs[f"{name}_position"],
                                                "max_abs_difference": max(abs(mine["value"] - theirs[name]),
                                                                          abs(mine["position_wall"] - theirs[f"{name}_position"]))}

    summary = {"method": "adami_k8_report.analyse_window -> cavity_analysis.stream_function_minimum (wall frame)",
               "grid_points": GRID_POINTS, "reference": "Marchi et al. 2021, Table 27 (Re 5000; Re 1000 Table 19, Re 3200 Table 23)",
               "marchi": marchi, "chains": results, "difference_mixed_minus_2x5090": difference,
               "snapshot_difference_mixed_minus_2x5090": snapshot_difference, "extrema_check_section4": extrema_check,
               "section4_window": section4["mixed"]["samples"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    def cell(value: float) -> str:
        return f"{value:.4f}".replace("-", "−")

    def range_cell(pair: tuple[float, float]) -> str:
        return f"{cell(pair[0])} … {cell(pair[1])}"

    lines = ["# Re 5000, 2500²: ψ_min and the primary vortex centre (tab:cavity5000)", "",
             f"Window t {boost['window'][0]:.3f}–{boost['window'][1]:.3f}: {boost['samples']} sample steps "
             f"({FIRST_STEP:,}–{LAST_STEP:,}), particle snapshots inside it at steps {', '.join(f'{s:,}' for s in boost['snapshot_steps'])} "
             f"(t {', '.join(f'{s * boost['dt']:.3f}' for s in boost['snapshot_steps'])}), the same in both runs. "
             f"Method of paper Tables 4 / 5 (adami_k8_report → cavity_analysis.stream_function_minimum): time-averaged MLS grid of u, "
             f"{GRID_POINTS} × {GRID_POINTS} over the wall rows, ψ integrated upward from the bottom wall (ψ = 0), interior minimum refined by a "
             "5 × 5 quadratic surface; wall frame.", "",
             "| | ψ_min | x | y |", "|---|---|---|---|",
             f"| Marchi 2021, Table 27 | {cell(marchi['psi'])} | {cell(marchi['x'])} | {cell(marchi['y'])} |"]
    for label in ("2x5090", "5090 + XTX"):
        value = values(results[label])
        lines.append(f"| {label} ({len(results[label]['per_snapshot'])}-snapshot mean) | {cell(value['psi'])} | {cell(value['x'])} | {cell(value['y'])} |")
    lines.append(f"| difference (5090 + XTX − 2x5090) | {difference['psi']:+.1e} | {difference['x']:+.1e} | {difference['y']:+.1e} |")
    lines += ["", "Per snapshot (each through the same function, one snapshot in the window), min … max:", "",
              "| | ψ_min | x | y |", "|---|---|---|---|"]
    for label in ("2x5090", "5090 + XTX"):
        ranges = spread(results[label])
        lines.append(f"| {label} | {range_cell(ranges['psi'])} | {range_cell(ranges['x'])} | {range_cell(ranges['y'])} |")
    lines.append("| difference at the same snapshot | " + " | ".join(
        f"{min(snapshot_difference[s]):+.1e} … {max(snapshot_difference[s]):+.1e}" for s in ("psi", "x", "y")) + " |")
    lines += ["", "Snapshots one by one (ψ_min, x, y):", ""]
    for label in ("2x5090", "5090 + XTX"):
        for item in results[label]["per_snapshot"]:
            lines.append(f"- {label}, t {item['t']:.3f}: {item['psi_min_wall']:.6f}, {item['x_wall']:.5f}, {item['y_wall']:.5f}")
    lines += ["", "Section 4 check (analyse_window's dense-line extrema on the same 14 samples vs re5000_mixed/summary.json):", ""]
    lines += [f"- {key}: {item['value_here']:.6f} @ {item['position_here']:.5f} vs {item['value_section4']:.6f} @ "
              f"{item['position_section4']:.5f} (max |Δ| {item['max_abs_difference']:.1e})" for key, item in extrema_check.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.buffer.write((OUT / "report.md").read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
