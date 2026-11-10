"""re3200_decide.py (user 2026-10-20 night): the finished Re = 3200 K = 1 run against Marchi 2021 (Re 3200), the decision
rule (relative L2 of u and v, wall frame, linear MLS, time average over the averaging window, both <= 1 % -> "cluster":
continue the run at K = 8 on the cluster from the checkpoint just before the steady window; else "refine": the 2000^2 case
warm-started from this run's final snapshot, locally), and the facts the report needs (window, extrema with positions
against Ghia 1982 Re 3200, psi_min, invariants, wall clock, the checkpoints on disk).

    python docs/validation/cluster/re3200_decide.py --run-dir <run dir> [--reynolds 3200] [--threshold 1.0] [--json OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def ghia_extrema(reynolds: int) -> dict:
    rows = []
    with open(_REPO_ROOT / "docs" / "validation" / "data" / "ghia1982_extrema.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    return {row["extremum"]: (float(row["tabulated_value"]), float(row["tabulated_position"]))
            for row in rows if int(row["reynolds"]) == reynolds}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reynolds", type=int, default=3200)
    parser.add_argument("--threshold", type=float, default=1.0, help="per cent, relative L2 of u and v (wall frame, MLS)")
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--no-stream-function", action="store_true")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    from experiment.validation import cavity_analysis as analysis_module
    run_dir = pathlib.Path(arguments.run_dir).resolve()
    run = analysis_module.load_run(run_dir.name, run_dir.parent)
    if run is None:
        sys.exit(f"no samples in {run_dir}")
    analysis = analysis_module.analyse_run(run, arguments.average_span, arguments.steady_tol, arguments.window,
                                           not arguments.no_stream_function, reynolds=arguments.reynolds)
    errors = {f"{frame}_{method}_{line}": {k: v for k, v in item.items() if k != "pointwise" and k != "values"}
              for (frame, method, line), item in analysis["errors"].items()}
    l2_u = errors["wall_mls_u"]["L2_relative"] * 100.0
    l2_v = errors["wall_mls_v"]["L2_relative"] * 100.0
    decision = "cluster" if max(l2_u, l2_v) <= arguments.threshold else "refine"
    steady_time = run["steady"].get("steady_time")
    checkpoints = []
    manifest = run_dir / "checkpoints" / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if (run_dir / "checkpoints" / entry["file"]).exists():
                    checkpoints.append(entry)
    chosen = None
    if steady_time is not None:
        before = [entry for entry in checkpoints if entry["time"] <= steady_time + 1e-9]
        chosen = before[-1] if before else None
    ghia = ghia_extrema(arguments.reynolds)
    extrema = {}
    for name, item in analysis["extrema"].items():
        reference = ghia.get(name)
        extrema[name] = {"value": item["value"], "position": item["position_fluid"], "sem": item["sem"],
                         "ghia_value": reference[0] if reference else None, "ghia_position": reference[1] if reference else None}
    stream = analysis.get("stream") or {}
    invariants = {"segments": [(segment.get("segment"), segment.get("status"), segment.get("totals")) for segment in run["segments"]],
                  "result": run["result"].get("status"), "wall_s": sum(float(segment.get("wall_s") or 0.0) for segment in run["segments"])}
    out = {"run": run_dir.name, "reynolds": arguments.reynolds, "threshold_pct": arguments.threshold,
           "window": analysis["window"], "samples_in_window": analysis["samples_in_window"], "t_last": analysis["t_last"],
           "t_steady_online": steady_time, "t_steady_sliding": analysis["t_steady_sliding"],
           "L2_relative_pct": {"u": l2_u, "v": l2_v}, "errors": errors, "extrema": extrema, "stream": stream,
           "decision": decision, "checkpoints": checkpoints, "chosen_checkpoint": chosen, "invariants": invariants,
           "frame_reynolds": analysis.get("frame_reynolds")}
    print(json.dumps(out, indent=1, default=float))
    print(f"\nDECISION {decision}: rel L2 u {l2_u:.3f} % v {l2_v:.3f} % (threshold {arguments.threshold} %), window {analysis['window']}, "
          f"steady {steady_time}, chosen checkpoint {chosen}")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
