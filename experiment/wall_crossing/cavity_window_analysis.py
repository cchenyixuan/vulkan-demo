"""
cavity_window_analysis.py - one cavity run (cavity_run.py's layout) analysed over a fixed time window against Marchi
et al. 2021, as a small JSON (claude/wall_store_report.md section 9, A2: run where the snapshots are, the JSON comes
back). The analysis is experiment/validation/cavity_analysis.analyse_run (the one of the particle-pool report's
tables), on the run's samples cut at --t1 so its averaging window is (t0, t1].

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity_window_analysis --root logs/round3/cavity \
        --name r3_n500_store --t0 80 --t1 100 --out logs/round3/r3_n500_store_analysis.json

Output: the window and its sample count; the online / sliding steady times; psi_min in the wall frame (unit square =
the innermost wall rows, L = 1 + 2 dx) and in the mid frame (unit square = the crossing box, L = 1 + dx: psi_mid =
psi_wall (1 + 2 dx) / (1 + dx)), the vortex centre; the three centre-line extrema (wall frame) with their positions;
the L2 / Linf of the centre lines against Marchi in the wall frame (MLS samples) and in the mid and fluid frames (the
dense lines); every value also relative to Marchi's; the kinetic energy and, from field_history.jsonl, the mean
density and mean pressure over the window.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    from experiment.validation import cavity_reference
    from experiment.validation.cavity_analysis import analyse_run, load_run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t1", type=float, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    root = (REPOSITORY_ROOT / arguments.root).resolve()
    run = load_run(arguments.name, root=root)
    if run is None:
        sys.exit(f"no run {arguments.name} under {root}")
    last_time = float(run["times"].max())
    keep = run["times"] <= arguments.t1 + 1e-9
    run["times"] = run["times"][keep]
    run["kinetic_energy"] = run["kinetic_energy"][keep]
    run["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    run["rows"] = [row for row, kept in zip(run["rows"], keep) if kept]
    analysis = analyse_run(run, average_span=arguments.t1 - arguments.t0, tolerance=1.0e-3, window=10.0,
                           with_stream_function=True)
    reference = cavity_reference.marchi2021()["extrema"]
    spacing = float(run["spacing"])
    stream = analysis["stream"] or {}
    psi_wall = stream.get("psi_min_wall")
    psi_mid = None if psi_wall is None else psi_wall * (1.0 + 2.0 * spacing) / (1.0 + spacing)
    psi_reference = float(reference["psi_min"][0])

    def deviation(value, reference_value):
        return None if value is None else (abs(value) - abs(reference_value)) / abs(reference_value)

    result = {"name": arguments.name, "run_dir": str(run["dir"]), "spacing": spacing, "last_time_of_run": last_time,
              "window": analysis["window"], "samples_in_window": analysis["samples_in_window"],
              "t_steady_online": analysis["t_steady_online"], "t_steady_sliding": analysis["t_steady_sliding"],
              "kinetic_energy_window": {key: float(np.asarray(value).ravel()[0])
                                        for key, value in analysis["kinetic_energy_window"].items()
                                        if np.asarray(value).size == 1},
              "psi_min": {"wall": psi_wall, "mid": psi_mid, "reference": psi_reference,
                          "deviation_wall": deviation(psi_wall, psi_reference),
                          "deviation_mid": deviation(psi_mid, psi_reference)},
              "stream": {key: value for key, value in stream.items() if not isinstance(value, np.ndarray)},
              "extrema": {}, "errors": {}}
    for name, entry in analysis["extrema"].items():
        reference_value = float(reference[name][0]) if name in reference else None
        result["extrema"][name] = {"value": entry["value"], "position_wall": entry["position_wall"],
                                   "reference": reference_value,
                                   "deviation": deviation(entry["value"], reference_value)
                                   if reference_value is not None else None}
    for (frame, method, line), entry in analysis["errors"].items():
        result["errors"][f"{frame}_{method}_{line}"] = {"L2": entry["L2"], "Linf": entry["Linf"],
                                                        "L2_relative": entry["L2_relative"]}
    history_path = run["dir"] / "field_history.jsonl"
    if history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        chosen = [row for row in rows if arguments.t0 - 1e-9 <= row.get("time", -1.0) <= arguments.t1 + 1e-9]
        for key in ("density_mean", "pressure_mean"):
            values = [row[key] for row in chosen if key in row]
            if values:
                result[f"{key}_window"] = {"mean": float(np.mean(values)), "minimum": float(np.min(values)),
                                           "maximum": float(np.max(values)), "count": len(values)}
    pathlib.Path(arguments.out).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"psi_min": result["psi_min"], "window": result["window"]}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
