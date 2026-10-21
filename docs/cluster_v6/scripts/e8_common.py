"""
e8_common.py — shared helpers of the E8 phase-2 post-processing tools (e8_p2_tables, e8_phases_table, e8_loss_sources,
e8_hops, e8_bytes_check, e8_kernel_table, e8_correctness_tables, e8_single_ratio, e8_migration_stats, e8_p2_summary).

A pulled job directory has the e30 layout (results.jsonl, index.tsv, <label>.log, traces/<name>/, raw/<label>/,
weights/<point>.json + .partition.json, compare/, dumps/). Labels of the E8 jobs (e7_lib.sh, e7_plan.py revision 11):
points 3d_n<n>_K<K>, their K runs <point>_t<t>_C, references <point>_t<t>_R1_3d_n<n>_g<i>, traced run <point>_trace_C
(trace dir traces/<point>_C, raw dir raw/<point>_trace_C), K = 1 traces trace_3d_n<n>_K1 (trace dir traces/<label>),
dumps dump_3d_n<n>_K1a / K1b / K8, comparisons compare_3d_n<n>_K8, the v7_single pairs single_3d_n<n>_p<i>_{single,chain}.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
SCRIPTS = HERE.parent
REPO = HERE.parents[3]
for entry in (str(SCRIPTS), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import e7_full_summary as summary  # noqa: E402
import e8_vram  # noqa: E402
from experiment.v7.analysis import step_trace_model  # noqa: E402

CASE_LABELS = {"cavity3d_n152": "4M (160^3)", "cavity3d_n192": "8M (200^3)", "cavity3d_n244": "16M (252^3)",
               "cavity3d_n312": "32M (320^3)", "cavity3d_n392": "64M (400^3)", "cavity3d_n192_k2": "weak 2 x 8M",
               "cavity3d_n192_k4": "weak 4 x 8M", "cavity3d_n192_k8": "weak 8 x 8M", "cavity3d_n48": "56^3"}
CUBES = ("cavity3d_n152", "cavity3d_n192", "cavity3d_n244", "cavity3d_n312", "cavity3d_n392")
ROLE_K_RUN = ("equal", "calibrated")
POINT_PATTERN = re.compile(r"^(?:[a-z]+_)?3d_n(\d+)(?:_k(\d+))?_K(\d+)$")
NS_PER_MS = 1e6          # the traces' ticks are host-clock nanoseconds (step_trace_model: / 1e3 = us)
NS_PER_US = 1e3


def short_case(case: str) -> str:
    return case.replace("cavity", "")


def case_of_point(point: str) -> tuple:
    """(case, K) of a point label 3d_n192_K8 / 3d_n192_k8_K8 (a prefix such as old_ is tolerated)."""
    match = POINT_PATTERN.match(point)
    if not match:
        return None, None
    n, weak, slabs = match.groups()
    case = f"cavity3d_n{n}" + (f"_k{weak}" if weak else "")
    return case, int(slabs)


def particles_of(case: str) -> int:
    n, n_x = e8_vram.CASES[case]
    return e8_vram.lattice(n, n_x)["total"]


def lattice_of(case: str) -> dict:
    n, n_x = e8_vram.CASES[case]
    return e8_vram.lattice(n, n_x)


def load_job(directory) -> dict:
    """e7_full_summary.load_job + the E8 extras: partition records, failed points, the directory's trace / raw dirs."""
    directory = pathlib.Path(directory)
    job = summary.load_job(directory)
    job["partitions"] = {}
    weights_dir = directory / "weights"
    if weights_dir.exists():
        for path in sorted(weights_dir.glob("*.partition.json")):
            try:
                job["partitions"][path.name[:-len(".partition.json")]] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                job["partitions"][path.name[:-len(".partition.json")]] = None
    # the harness appends the point to failed_points.txt (one per line) and failed_points.tsv (point, job, ...)
    failed = []
    for name in ("failed_points.txt", "failed_points.tsv"):
        path = directory / name
        if path.exists():
            failed += [line.split("\t")[0].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    job["failed_points"] = sorted(set(failed))
    job["traces"] = {path.name: path for path in sorted((directory / "traces").glob("*")) if (path / "run_meta.json").exists()} \
        if (directory / "traces").exists() else {}
    job["raw"] = {path.name: path for path in sorted((directory / "raw").glob("*")) if path.is_dir()} \
        if (directory / "raw").exists() else {}
    return job


def points_of(job: dict) -> dict:
    """e7_full_summary.collect_points with the case / K filled from the point label when the index lacks them."""
    points = summary.collect_points(job)
    for name, point in points.items():
        case, slabs = case_of_point(name)
        point["case"] = point.get("case") or case
        point["slabs"] = int(point["slabs"]) if point.get("slabs") not in (None, "", "-") else slabs
        point["trace_dir"] = job["traces"].get(f"{name}_C") or job["traces"].get(f"{name}_E")
        point["raw_dir"] = job["raw"].get(f"{name}_trace_C") or job["raw"].get(f"{name}_trace_E")
        point["partition"] = job["partitions"].get(name)
    return points


def card_of(row: dict, label: str = ""):
    """The physical device of a K = 1 run (its device map), else the g<i> suffix of the label."""
    if row and row.get("device_map"):
        return int(row["device_map"][0])
    match = re.search(r"_g(\d+)$", label or "")
    return int(match.group(1)) if match else None


def fps_pair(row: dict) -> tuple:
    """(fps with the defrag events, fps without them) of a run row; None where unknown."""
    if not row:
        return None, None
    return summary.precise_fps(row), summary.nodefrag_fps(row)


def card_clock(row: dict, card) -> dict:
    loop = (row or {}).get("telemetry_loop") or {}
    entry = loop.get(str(card)) or {}
    return {"sm_clock_median": entry.get("sm_clock_median"), "sm_clock_min": entry.get("sm_clock_min"),
            "power_median": entry.get("power_median"), "temperature_max": entry.get("temperature_max")}


def describe(values: list) -> dict:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "values": values}


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def rng(stats: dict, digits: int = 3) -> str:
    if not stats or not stats.get("n"):
        return "-"
    if stats["n"] == 1:
        return fmt(stats["mean"], digits)
    return f"{fmt(stats['mean'], digits)} ({fmt(stats['min'], digits)}–{fmt(stats['max'], digits)})"


def load_trace(trace_dir):
    return step_trace_model.load_run(pathlib.Path(trace_dir))


def steady_mask(rows: dict, warmup: int) -> np.ndarray:
    """Steady steps of one sim's sorted rows: step >= warmup and the next step present (for periods)."""
    steps = rows["step"].astype(int)
    consecutive = np.concatenate([steps[1:] == steps[:-1] + 1, [False]])
    return (steps >= warmup) & consecutive


def sim_rows(run: dict, sim: int) -> dict:
    device = run["device"]
    rows = step_trace_model.select(device, (device["sim"] == sim) & (device["complete"] == 1))
    order = np.argsort(rows["step"])
    return {key: value[order] for key, value in rows.items()}


def sims_of(run: dict) -> list:
    return sorted(int(value) for value in set(run["device"]["sim"].astype(int)))


def defrag_events_in_window(row: dict, warmup: int, steps: int) -> list:
    """(frame, wall_ms) of the run row's defrag events inside the steady window (frames in (warmup, steps))."""
    out = []
    for event in (row or {}).get("defrag_events") or []:
        frame = event.get("frame", event.get("f"))
        wall = event.get("wall_ms")
        if frame is None or wall is None:
            continue
        if warmup < int(frame) < steps:
            out.append((int(frame), float(wall)))
    return out


def read_pool_demand(raw_dir) -> dict:
    """{link: {"index": array, "inner": array, "outer": array, "migrant": array}} of raw/<label>/pool_demand_<link>.csv
    (every column the worker recorded), plus "_meta" (capacities)."""
    raw_dir = pathlib.Path(raw_dir)
    out = {}
    for path in sorted(raw_dir.glob("pool_demand_*.csv")):
        link = path.stem[len("pool_demand_"):]
        text = path.read_text(encoding="utf-8").splitlines()
        if not text:
            continue
        header = text[0].split(",")
        columns = {name: [] for name in header}
        for line in text[1:]:
            if not line.strip():
                continue
            for name, value in zip(header, line.split(",")):
                columns[name].append(float(value) if value != "" else np.nan)
        out[link] = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    meta = raw_dir / "pool_demand_meta.json"
    out["_meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return out


def write_text(path, text: str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def write_json(path, document) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, default=jsonable), encoding="utf-8")


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)
