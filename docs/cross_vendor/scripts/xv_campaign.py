"""xv_campaign.py — E1 + E2 and E3 of the S6.6 cross-vendor campaign (RTX 5090 + RX 7900 XTX), the cluster protocol
through xv_lib (run_chain_v6.py --solver v7 + parse_run_v6.py, 1500 / 500 steps, depth 2, 3 trials, simultaneous
K = 1 references behind one barrier, built-in checks, never rerun-and-drop).

Sub-commands:

  fit  --case CASE          the "does it fit" pre-check of a size (the user's item 1 for 8000^2): one K = 1 process per
                            card at the same time (the real concurrency of a reference pair), 300 / 100 steps; a
                            process that dies (out of device memory or anything else) is reported in fit_<case>.json
                            and the size is skipped by `e12` (state file skip_<case>.json). Not a measurement.
  e12  --sizes 520,1000,... E1 and E2 interleaved per size, the way the cluster's e7_point rotated references and K
                            runs: calibration (weight_calibration.py: equal-weight pilot, a second pilot with the new
                            cuts when they moved, the second pilot's weights), then for trial t = 1..3 the K = 1
                            reference pair (E1) and the calibrated K = 2 run (E2) — references first on odd trials,
                            K = 2 first on even trials — and finally one traced K = 2 run (phases, links, seam check).
                            eta of trial t = harmonic mean of the trial's two K = 1 step times / (2 T_t).
  e3   --case CASE --ratios 1.4:2.6:0.1
                            the weight sweep: for trial t = 1..3 one reference pair, then every ratio r as the fixed
                            weights "r,1" (omega_5090 / omega_XTX = r, no pilot), traced (per-card step medians), ascending
                            on odd trials and descending on even trials.

Labels (results.jsonl / index.tsv / logs / traces): ref_<short>_K1_t<t>_g<0|1>, <short>_K2_t<t>, <short>_K2_trace,
cal_<short>_K2, fit_<short>_K1_g<0|1>, e3: ref_<short>_K1_t<t>_g<i>, <short>_K2_r<ratio>_t<t> (traced).
A label whose pass row already exists in results.jsonl is skipped (resume after an interruption); a hard failure
stops the sub-command (RunFailed), nothing is retried.

Usage (checkout root, vulkan-demo venv, PYTHONIOENCODING=utf-8):
    python docs/cross_vendor/scripts/xv_campaign.py fit --case cavity2d_n8000 --out docs/cross_vendor/data/e12
    python docs/cross_vendor/scripts/xv_campaign.py e12 --sizes 520,1000,1440,2000,2840,4000,5640,8000 --out docs/cross_vendor/data/e12
    python docs/cross_vendor/scripts/xv_campaign.py e3 --case cavity2d_n4000 --ratios 1.4:2.6:0.1 --out docs/cross_vendor/data/e3
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xv_lib  # noqa: E402

OBJ_CACHE = xv_lib.REPOSITORY / "logs" / "xv_objcache"           # logs/ is git-ignored


def short(case: str) -> str:
    return case.replace("cavity", "")


def particles_of(case: str) -> int:
    match = re.search(r"n(\d+)", case)
    n = int(match.group(1))
    return n * n


def timeout_for(case: str, kind: str) -> float:
    """Generous: 10 min + 60 s per million particles (a 64M K = 1 on the XTX runs ~3 min)."""
    return 600.0 + 60.0 * particles_of(case) / 1e6 * (2.0 if kind == "reference" else 1.0)


def done_labels(experiment: xv_lib.Experiment) -> set:
    labels = set()
    if experiment.results.exists():
        for line in experiment.results.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") in ("pass", "threshold_only"):
                labels.add(row["label"])
    return labels


def log(message: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def fit(arguments) -> int:
    out = pathlib.Path(arguments.out)
    experiment = xv_lib.Experiment(out)
    experiment.start_telemetry()
    case = arguments.case
    label = f"fit_{short(case)}_K1"
    record = {"case": case, "started": datetime.datetime.now().isoformat(timespec="seconds"), "steps": 300, "warmup": 100}
    try:
        rows = experiment.references(case, label, members=(0, 1), timeout=timeout_for(case, "reference"), steps=300, warmup=100)
        record["cards"] = {str(member): {key: row.get(key) for key in ("status", "steady_fps", "reasons", "telemetry_peaks", "rc")}
                           for member, row in rows.items()}
        record["fits"] = True
    except xv_lib.RunFailed as error:
        record["fits"] = False
        record["failed"] = str(error)
        # the per-card verdicts and the error text of the dead process
        for member in (0, 1):
            log_path = out / "logs" / f"{label}_g{member}.log"
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                errors = [line.strip() for line in text.splitlines() if re.search(r"Error|OUT_OF|MemoryError|Traceback", line)]
                record.setdefault("errors", {})[str(member)] = errors[-6:]
        (out / f"skip_{case}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    finally:
        time.sleep(1)
        experiment.stop_telemetry()
        record["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
        (out / f"fit_{case}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"fit {case}: {'fits' if record['fits'] else 'DOES NOT FIT / FAILED'}")
    print(json.dumps(record, indent=1, ensure_ascii=False, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------------
def e12(arguments) -> int:
    out = pathlib.Path(arguments.out)
    experiment = xv_lib.Experiment(out)
    sizes = [int(size) for size in arguments.sizes.split(",")]
    trials = arguments.trials
    experiment.start_telemetry()
    state_path = out / "campaign_e12.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"sizes": {}}
    try:
        for size in sizes:
            case = f"cavity2d_n{size}"
            name = short(case)
            if (out / f"skip_{case}.json").exists():
                log(f"{case}: skipped (skip file from the fit pre-check)")
                state["sizes"][case] = {"skipped": True}
                continue
            entry = state["sizes"].setdefault(case, {})
            done = done_labels(experiment)
            weights_file = out / "weights" / f"cal_{name}_K2.json"
            if not weights_file.exists():
                log(f"{case}: calibration")
                weights_file, cal_row, record = experiment.calibrate(case, f"cal_{name}_K2", timeout=timeout_for(case, "calibration"))
                entry["calibration"] = {"weights": record["weights"], "cuts": record["cuts"], "seconds": cal_row["seconds"],
                                        "rounds": [{"round": r["round"], "weights": r["weights"], "cuts": r["cuts"], "omega": r["omega"],
                                                    "next_cuts": r["next_cuts"], "changed": r["changed"], "busy_us": r["busy_us"],
                                                    "fluid": r["fluid"], "all": r["all"]} for r in record["rounds"]]}
            for trial in range(1, trials + 1):
                reference_label = f"ref_{name}_K1_t{trial}"
                chain_label = f"{name}_K2_t{trial}"
                order = ("reference", "chain") if trial % 2 == 1 else ("chain", "reference")
                for item in order:
                    if item == "reference":
                        if {f"{reference_label}_g0", f"{reference_label}_g1"} <= done:
                            continue
                        log(f"{case}: trial {trial} references (both cards)")
                        experiment.references(case, reference_label, members=(0, 1), timeout=timeout_for(case, "reference"))
                    else:
                        if chain_label in done:
                            continue
                        log(f"{case}: trial {trial} K = 2")
                        experiment.chain(case, chain_label, weights_file=weights_file, timeout=timeout_for(case, "chain"))
            trace_label = f"{name}_K2_trace"
            if trace_label not in done:
                log(f"{case}: traced K = 2")
                experiment.chain(case, trace_label, weights_file=weights_file, timeout=timeout_for(case, "chain"), trace=True)
            entry["done"] = datetime.datetime.now().isoformat(timespec="seconds")
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    except xv_lib.RunFailed as error:
        log(f"STOPPED (hard failure, nothing retried): {error}")
        state["stopped"] = str(error)
        state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        experiment.stop_telemetry()
        return 1
    time.sleep(1)
    experiment.stop_telemetry()
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    log("e12 done")
    return 0


# ---------------------------------------------------------------------------------------------------------------
def ratios_of(text: str) -> list:
    start, stop, step = (float(part) for part in text.split(":"))
    count = int(round((stop - start) / step)) + 1
    return [round(start + index * step, 2) for index in range(count)]


def e3(arguments) -> int:
    out = pathlib.Path(arguments.out)
    experiment = xv_lib.Experiment(out)
    case, name = arguments.case, short(arguments.case)
    ratios = ratios_of(arguments.ratios)
    experiment.start_telemetry()
    state_path = out / "campaign_e3.json"
    state = {"case": case, "ratios": ratios, "trials": arguments.trials, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        for trial in range(1, arguments.trials + 1):
            done = done_labels(experiment)
            reference_label = f"ref_{name}_K1_t{trial}"
            if not {f"{reference_label}_g0", f"{reference_label}_g1"} <= done:
                log(f"e3 trial {trial}: references (both cards)")
                experiment.references(case, reference_label, members=(0, 1), timeout=timeout_for(case, "reference"))
            for ratio in (ratios if trial % 2 == 1 else ratios[::-1]):
                label = f"{name}_K2_r{ratio:.1f}_t{trial}"
                if label in done:
                    continue
                log(f"e3 trial {trial}: ratio {ratio:.1f}")
                experiment.chain(case, label, weights=f"{ratio:.1f},1", timeout=timeout_for(case, "chain"), trace=True)
    except xv_lib.RunFailed as error:
        log(f"STOPPED (hard failure, nothing retried): {error}")
        state["stopped"] = str(error)
        state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        experiment.stop_telemetry()
        return 1
    time.sleep(1)
    experiment.stop_telemetry()
    state["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    log("e3 done")
    return 0


def more(arguments) -> int:
    """Extra trials of one size in the e12 directory (the user's 1440^2 item, 2026-10-20 evening): the E2 flow per trial
    (reference pair and the calibrated K = 2 run, references first on odd trials), the K = 2 run traced when --trace is
    given (so that the phases of the trial can be reported); telemetry at 0.25 s."""
    out = pathlib.Path(arguments.out)
    experiment = xv_lib.Experiment(out)
    case, name = arguments.case, short(arguments.case)
    weights_file = out / "weights" / f"cal_{name}_K2.json"
    if not weights_file.exists():
        log(f"{case}: no weights file {weights_file}")
        return 2
    trials = [int(trial) for trial in arguments.trials.split(",")]
    experiment.start_telemetry(interval=0.25, path=out / f"telemetry_more_{name}.csv")
    try:
        for trial in trials:
            done = done_labels(experiment)
            reference_label = f"ref_{name}_K1_t{trial}"
            chain_label = f"{name}_K2_t{trial}"
            order = ("reference", "chain") if trial % 2 == 1 else ("chain", "reference")
            for item in order:
                if item == "reference":
                    if {f"{reference_label}_g0", f"{reference_label}_g1"} <= done:
                        continue
                    log(f"{case}: trial {trial} references (both cards)")
                    experiment.references(case, reference_label, members=(0, 1), timeout=timeout_for(case, "reference"))
                else:
                    if chain_label in done:
                        continue
                    log(f"{case}: trial {trial} K = 2{' (traced)' if arguments.trace else ''}")
                    experiment.chain(case, chain_label, weights_file=weights_file, timeout=timeout_for(case, "chain"), trace=arguments.trace)
    except xv_lib.RunFailed as error:
        log(f"STOPPED (hard failure, nothing retried): {error}")
        experiment.stop_telemetry()
        return 1
    time.sleep(1)
    experiment.stop_telemetry()
    log("more done")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="S6.6 cross-vendor campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    p_more = sub.add_parser("more")
    p_more.add_argument("--case", required=True)
    p_more.add_argument("--trials", required=True, help="comma-separated trial numbers, e.g. 4,5")
    p_more.add_argument("--trace", action="store_true", help="step-trace the K = 2 run of every trial (seam check on)")
    p_more.add_argument("--out", required=True)
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--case", required=True)
    p_fit.add_argument("--out", required=True)
    p_e12 = sub.add_parser("e12")
    p_e12.add_argument("--sizes", default="520,1000,1440,2000,2840,4000,5640,8000")
    p_e12.add_argument("--trials", type=int, default=3)
    p_e12.add_argument("--out", required=True)
    p_e3 = sub.add_parser("e3")
    p_e3.add_argument("--case", required=True)
    p_e3.add_argument("--ratios", default="1.4:2.6:0.1")
    p_e3.add_argument("--trials", type=int, default=3)
    p_e3.add_argument("--out", required=True)
    arguments = parser.parse_args()
    OBJ_CACHE.mkdir(parents=True, exist_ok=True)
    xv_lib.WRAPPER_DEFAULTS[:] = ["--obj-cache", str(OBJ_CACHE)]
    return {"fit": fit, "e12": e12, "e3": e3, "more": more}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
