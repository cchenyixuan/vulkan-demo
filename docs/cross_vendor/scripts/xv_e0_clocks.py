"""xv_e0_clocks.py — E0's clock / temperature record: the 1500-step protocol run on 1000^2 lasts under 2 s, too short
for a 1 Hz sampler, so one longer K = 2 cross-vendor run (the E0 weights, --no-seam-check, STEPS steps) is sampled at
0.25 s and the per-card ranges inside its loop window are written to data/e0/clocks_summary.json (telemetry in
data/e0/telemetry_clocks.csv). Not a measurement of the campaign; its RESULT row is kept in results.jsonl under the
label clocks_<case>_K2.

Usage:
    python docs/cross_vendor/scripts/xv_e0_clocks.py [--case cavity2d_n1000] [--steps 15000] [--out docs/cross_vendor/data/e0]
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xv_lib  # noqa: E402
from xv_e0 import telemetry_ranges  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="cavity2d_n1000")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--out", default=str(xv_lib.REPOSITORY / "docs" / "cross_vendor" / "data" / "e0"))
    parser.add_argument("--suffix", default="", help="appended to the run label and the telemetry / summary file names")
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    experiment = xv_lib.Experiment(out)
    weights_file = out / "weights" / f"cal_{arguments.case}_K2.json"
    experiment.start_telemetry(interval=0.25, path=out / f"telemetry_clocks{arguments.suffix}.csv")
    label = f"clocks_{arguments.case}_K2{arguments.suffix}"
    summary = {"label": label, "started": datetime.datetime.now().isoformat(timespec="seconds"), "steps": arguments.steps}
    try:
        row = experiment.chain(arguments.case, label, weights_file=weights_file, timeout=1800, trace=False, steps=arguments.steps, warmup=500)
        summary["result"] = {key: row.get(key) for key in ("status", "steady_fps", "steady_fps_nodefrag", "drift", "overflow_total",
                                                            "alive_start", "alive_end", "defrag_events_summary", "telemetry_loop")}
        stages = row.get("stages") or {}
        loop_start, loop_end = (stages.get("loop_start") or {}).get("epoch"), (stages.get("loop_end") or {}).get("epoch")
        time.sleep(1)
        experiment.stop_telemetry()
        summary["loop_seconds"] = (loop_end - loop_start) if loop_start and loop_end else None
        if loop_start and loop_end:
            summary["ranges"] = telemetry_ranges(experiment.telemetry, loop_start, loop_end)
    except xv_lib.RunFailed as error:
        summary["failed"] = str(error)
        experiment.stop_telemetry()
    summary["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
    (out / f"clocks_summary{arguments.suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 1 if "failed" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
