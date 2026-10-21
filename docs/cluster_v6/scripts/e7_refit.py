"""
e7_refit.py — E7 batch report, item 10: e7_plan.py's time model refitted to a batch's measurements; the batch's
items against the plan and the refit; the node-hour budget of every batch under both.

Inputs per job directory (the e7_lib.sh layout as mirrored from ~/run/logs/e7_b<batch><line>_<id>, see
e7_full_summary.py) and the job's stdout (its SLURM .out: the one *.out file in the directory, or --stdout in the
order of the directories):
  - per run (results.jsonl): the wrapper's stage stamps (run_chain_v6.py: case_loaded, partition_done,
    bootstrap_start, bootstrap_end, loop_start, loop_end, exit; s since the bench started, and epochs) and the
    steady-window fps;
  - per calibration (cal_<point>.log): the stamps of every pilot chain;
  - the timeline: the clock time (1 s) of every item (PRECHECK, POINT, EXTRA), run (RUN) and step (STEP) header.
A group is one run, or the simultaneous members of one reference set or pre-check (<label>_g<i> / _p<i>); it lasts
from its header to the next header; its critical member is the one whose bootstrap ended last (the start barrier
holds the others).

Refit (a field without measurements keeps the plan's value):
  cost_2d / cost_3d           K = 1 reference runs: sum of steady step times / sum of particles (time-weighted);
  efficiency, efficiency_3d   timing arms (E, C) and K = 2 pair references with more than LIGHT_PER_CARD particles
                              per card: mean of cost x per card / step time, per K;
  floor                       the lighter ones more than 5 % slower than cost x per card / efficiency: mean step
                              time, per K;
  load_per_million            least squares of partition_done (read-in + partition) on millions (the intercept,
                              the bench's imports, ends up in run_overhead);
  cold_ / warm_per_simulator  bootstrap_start - partition_done of the critical member (and of every calibration
                              pilot) per simulator, summed over the cold groups (the plan's rule: pilots, the first
                              trial's reference sets, pre-checks, traced runs, extras) and over the warm ones;
  bootstrap_*                 least squares of bootstrap_end - bootstrap_start on millions (groups and pilots);
  post_per_*                  least squares (no intercept) of the last exit - loop_end on millions and slabs,
                              groups without the seam check;
  seam_2d_*, trace_write      least squares of exit - loop_end on millions, 1 and "traced" over the 2-D seam-checked
                              groups (traced runs and soaks; both kinds and two sizes needed);
  seam_3d_*                   least squares (no intercept) on millions and slabs x millions over the 3-D ones, the
                              trace write taken off the traced ones (without the data for it: the plan's terms
                              scaled to the measured sum);
  run_overhead                mean over the groups of the duration minus the refitted model without it (process
                              start, imports, teardown, parse, log sync, warmup excess: what the stamps leave);
  calibration_rounds / _fixed the median pilot count; the mean of the duration minus the refitted pilots;
  job_overhead                mean over the jobs of the elapsed time (sacct, else the E30 wall) minus the items.
Report: the fields (plan, refit, data), the step times, every item of the batch (plan, refit as planned, refit as
run, measured) and the budget of every batch (node-h per line) under the plan and the refit.

Usage:
    python docs/cluster_v6/scripts/e7_refit.py JOB_DIR [JOB_DIR ...] [--stdout FILE ...] [--out-model MODEL.json]
        [--out REPORT.md] [--json OUT.json]
    python docs/cluster_v6/scripts/e7_plan.py --model-json MODEL.json      (the refitted plan and job scripts)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e7_plan  # noqa: E402
from e7_full_summary import load_job, precise_fps  # noqa: E402

CLOCK = re.compile(r"\((\d{1,2}):(\d{2}):(\d{2})\)")
MEMBER = re.compile(r"(.+)_[gp]\d+$")
STAGE = re.compile(r"\[e30\] stage (\w+): t=([\d.]+)s")
STAMPS = ("partition_done", "bootstrap_start", "bootstrap_end", "loop_start", "loop_end", "exit")
COLD_ROLES = {"precheck", "selftest", "anatomy", "soak", "fulltrace", "trace_equal", "trace_calibrated"}
SEAM_ROLES = {"soak", "trace_equal", "trace_calibrated"}
TRACED_ROLES = {"trace_equal", "trace_calibrated", "fulltrace"}
ARM_ROLES = {"equal", "calibrated"}
FLOOR_MARGIN = 1.05     # a lighter point's step time counts as the floor only this far above its compute time


def short(case: str) -> str:
    return re.sub(r"^cavity([23]d)_", r"\1_", case)


def dimension(case: str) -> int:
    return 3 if case.startswith("cavity3d") else 2


def particles(case: str):
    return e7_plan.PARTICLES.get(case)


# ----------------------------------------------------------------------------- timeline

def clock_of(text: str):
    match = CLOCK.search(text)
    return None if match is None else int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def read_timeline(path: pathlib.Path) -> dict:
    """The job's header events in order, {t, kind, name}, t in s with midnight crossings unwrapped: start (the E30
    banner, 12- or 24-hour clock), step, ready (end of the E7 prelude), precheck / point / extra (the items), run,
    point_done, end (the E30 done line: start + its wall)."""
    events, point_starts = [], {}
    offset, last, wall = 0, None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if line.startswith("=== E30 ") and " done: " in line:
            match = re.search(r"wall (\d+)s", line)
            wall = int(match.group(1)) if match else None
            continue
        if line.startswith("=== E30 ") and " job " in line:
            match = re.search(r" (\d{1,2}):(\d{2}):(\d{2})(?: ([AP]M))? ", line)
            if match is None or any(event["kind"] == "start" for event in events):
                continue
            hours = int(match.group(1))
            if match.group(4):
                hours = hours % 12 + (12 if match.group(4) == "PM" else 0)
            kind, name, clock = "start", "", hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))
        elif line.startswith(("=== RUN ", "=== STEP ")) and len(fields) > 2:
            kind, name, clock = fields[1].lower(), fields[2], clock_of(line)
        elif line.startswith("=== E7 job ") and " ready at " in line:
            kind, name = "ready", ""
            clock = clock_of("(" + line.split(" ready at ", 1)[1][:8] + ")")
        elif line.startswith("##### POINT ") and " done in " in line:
            match = re.search(r" done in (\d+) s", line)
            if match and fields[2] in point_starts:
                events.append({"t": point_starts[fields[2]] + int(match.group(1)), "kind": "point_done",
                               "name": fields[2]})
            continue
        elif line.startswith("##### POINT ") and len(fields) > 2:
            kind, name, clock = "point", fields[2], clock_of(line)
        elif line.startswith("##### PRECHECK ") and len(fields) > 4:
            kind, name, clock = "precheck", f"pre_{short(fields[2])}_{fields[3]}_{fields[4]}", clock_of(line)
        elif line.startswith("##### EXTRA ") and len(fields) > 3:
            kind, name, clock = "extra", fields[3], clock_of(line)
        else:
            continue
        if clock is None:
            continue
        if last is not None and clock + offset < last - 6 * 3600:
            offset += 86400
        t = clock + offset
        last = t
        if kind == "point":
            point_starts[name] = t
        events.append({"t": t, "kind": kind, "name": name})
    start = next((event["t"] for event in events if event["kind"] == "start"), None)
    if start is not None and wall is not None:
        events.append({"t": start + wall, "kind": "end", "name": ""})
    return {"events": events, "start": start, "wall": wall}


def timeline_groups(events: list) -> list:
    """Run and step groups (consecutive headers of one group's members merge), each lasting until the next header."""
    groups = []
    for index, event in enumerate(events):
        if event["kind"] not in ("run", "step"):
            continue
        match = MEMBER.match(event["name"]) if event["kind"] == "run" else None
        key = match.group(1) if match else event["name"]
        if groups and groups[-1]["key"] == key and groups[-1]["last_index"] == index - 1:
            groups[-1]["labels"].append(event["name"])
            groups[-1]["last_index"] = index
            continue
        groups.append({"key": key, "kind": event["kind"], "start": event["t"], "labels": [event["name"]],
                       "last_index": index})
    for group in groups:
        following = events[group["last_index"] + 1:]
        group["duration"] = following[0]["t"] - group["start"] if following else None
    return groups


def timeline_items(events: list) -> list:
    items = [event for event in events if event["kind"] in ("precheck", "point", "extra")]
    end = next((event["t"] for event in events if event["kind"] == "end"), events[-1]["t"] if events else None)
    out = []
    for index, event in enumerate(items):
        following = items[index + 1]["t"] if index + 1 < len(items) else end
        out.append({"kind": event["kind"], "label": event["name"], "start": event["t"],
                    "duration": None if following is None else following - event["t"]})
    return out


# ----------------------------------------------------------------------------- measurements

def stamp(row: dict, name: str, key: str = "seconds"):
    return ((row or {}).get("stages") or {}).get(name, {}).get(key)


def stamped(row: dict) -> bool:
    return all(stamp(row, name) is not None and stamp(row, name, "epoch") is not None for name in STAMPS)


def calibration_pilots(path: pathlib.Path) -> list:
    """Stage stamps (s since the bench started) of every pilot chain of a calibration log, one dict per pilot."""
    pilots = []
    if not path.exists():
        return pilots
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STAGE.search(line)
        if match is None:
            continue
        name, seconds = match.group(1), float(match.group(2))
        if name == "partition_done":
            pilots.append({"partition_done": seconds})
        elif pilots and name in STAMPS:
            pilots[-1][name] = seconds
    return [pilot for pilot in pilots if all(name in pilot for name in STAMPS[:5])]


def measure_group(group: dict, job: dict, index: dict):
    entries = [index.get(label) for label in group["labels"]]
    entry = entries[0] or {}
    role = entry.get("role")
    if group["kind"] == "step":
        if role != "calibration":
            return None             # prelude steps (bring-up, capabilities, cache stage-in): job overhead
        pilots = calibration_pilots(job["directory"] / f"{group['key']}.log")
        return {"job": job["name"], "group": group["key"], "role": role, "case": entry["case"],
                "slabs": entry["slabs"], "point": entry.get("point"), "start": group["start"],
                "duration": group["duration"], "pilots": pilots, "pilot_count": len(pilots), "cold": True,
                "seam": False, "traced": False}
    rows = [job["rows"].get(label) for label in group["labels"]]
    case = entry.get("case")
    trial = entry.get("trial") or 0
    record = {"job": job["name"], "group": group["key"], "role": role, "case": case, "point": entry.get("point"),
              "trial": trial, "start": group["start"], "duration": group["duration"], "members": len(rows),
              "member_runs": [{"label": label, "status": (row or {}).get("status"), "fps": precise_fps(row)}
                              for label, row in zip(group["labels"], rows)],
              "cold": role in COLD_ROLES or (role == "reference" and trial == 1),
              "seam": role in SEAM_ROLES, "traced": role in TRACED_ROLES}
    complete = [row for row in rows if stamped(row)]
    first = next((row for row in rows if row), {})
    record["slabs"] = first.get("slab_count") or entry.get("slabs")
    record["steps"] = first.get("total_steps")
    if not complete:
        return record
    critical = max(complete, key=lambda row: stamp(row, "bootstrap_end", "epoch"))
    last_exit = max(stamp(row, "exit", "epoch") for row in complete)
    bench_start = min(stamp(row, "exit", "epoch") - stamp(row, "exit") for row in complete)
    record.update({
        "slabs": critical.get("slab_count") or record["slabs"], "steps": critical.get("total_steps"),
        "read_in": stamp(critical, "partition_done"),
        "construction": stamp(critical, "bootstrap_start") - stamp(critical, "partition_done"),
        "bootstrap": stamp(critical, "bootstrap_end") - stamp(critical, "bootstrap_start"),
        "loop": stamp(critical, "loop_end") - stamp(critical, "loop_start"),
        "post": last_exit - stamp(critical, "loop_end", "epoch"),
        "in_bench": last_exit - bench_start})
    return record


def elapsed_seconds(job: dict):
    for record in job["sacct"]:
        if "." in record.get("JobID", ""):
            continue
        match = re.match(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", record.get("Elapsed", ""))
        if match:
            days, hours, minutes, seconds = (int(value) if value else 0 for value in match.groups())
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
    return None


def measure_job(directory: pathlib.Path, stdout: pathlib.Path) -> dict:
    job = load_job(directory)
    index = {entry["label"]: entry for entry in job["index"]}
    timeline = read_timeline(stdout)
    groups = [record for record in (measure_group(group, job, index) for group in timeline_groups(timeline["events"]))
              if record is not None]
    items = timeline_items(timeline["events"])
    for item in items:
        end = item["start"] + (item["duration"] or 0)
        item["groups"] = [group for group in groups if item["start"] <= group["start"] < end
                          or (item["duration"] is None and group["start"] >= item["start"])]
    name = job["provenance"].get("job") or job["name"]
    elapsed = elapsed_seconds(job)
    return {"name": name, "directory": str(directory), "stdout": str(stdout), "groups": groups, "items": items,
            "elapsed": elapsed if elapsed is not None else timeline["wall"],
            "elapsed_source": "sacct" if elapsed is not None else "E30 wall",
            "ready": next((event["t"] - timeline["start"] for event in timeline["events"]
                           if event["kind"] == "ready" and timeline["start"] is not None), None)}


# ----------------------------------------------------------------------------- fits

def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def fit_line(points: list):
    """y = a + b x by least squares over (x, y), a and b kept >= 0; None without two distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    count = len(points)
    mean_x, mean_y = sum(x for x, _ in points) / count, sum(y for _, y in points) / count
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points))
    intercept = mean_y - slope * mean_x
    if slope < 0:
        return max(mean_y, 0.0), 0.0
    if intercept < 0:
        return 0.0, sum(x * y for x, y in points) / sum(x * x for x, _ in points)
    return intercept, slope


def least_squares(rows: list):
    """Coefficients c minimising sum (y - c . x)^2 over rows (y, [x...]) (normal equations, Gauss-Jordan); None
    when singular."""
    size = len(rows[0][1])
    matrix = [[sum(x[i] * x[j] for _, x in rows) for j in range(size)] + [sum(y * x[i] for y, x in rows)]
              for i in range(size)]
    scale = max(abs(matrix[i][i]) for i in range(size)) or 1.0
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) <= 1e-10 * scale:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for row in range(size):
            if row != column:
                factor = matrix[row][column] / matrix[column][column]
                matrix[row] = [left - factor * right for left, right in zip(matrix[row], matrix[column])]
    return [matrix[index][size] / matrix[index][index] for index in range(size)]


def fit_two(points: list):
    """y = c1 x1 + c2 x2 by least squares over (x1, x2, y), no intercept, both kept >= 0; None if singular."""
    a11 = sum(x1 * x1 for x1, _, _ in points)
    a12 = sum(x1 * x2 for x1, x2, _ in points)
    a22 = sum(x2 * x2 for _, x2, _ in points)
    b1 = sum(x1 * y for x1, _, y in points)
    b2 = sum(x2 * y for _, x2, y in points)
    determinant = a11 * a22 - a12 * a12
    if not points or abs(determinant) <= 1e-9 * max(a11 * a22, 1e-30):
        return None
    first, second = (b1 * a22 - b2 * a12) / determinant, (a11 * b2 - a12 * b1) / determinant
    if first < 0:
        return 0.0, b2 / a22
    if second < 0:
        return b1 / a11, 0.0
    return first, second


def seam_seconds(model: e7_plan.Model, case: str, slabs: int) -> float:
    """The model's post-run time with the seam check (setup's seam term)."""
    millions = particles(case) / 1e6
    if dimension(case) == 3:
        return (model.seam_3d_per_million + model.seam_3d_per_million_per_slab * slabs) * millions
    return model.seam_2d_per_million * millions + model.seam_2d_fixed


def predict_group(model: e7_plan.Model, group: dict):
    case, slabs = group.get("case"), group.get("slabs")
    if case not in e7_plan.PARTICLES or not slabs:
        return None
    if group["role"] == "calibration":
        pilot = model.setup(case, slabs, cold=True) + model.pilot_steps * model.step_time(case, slabs)
        return (group.get("pilot_count") or model.calibration_rounds) * pilot + model.calibration_fixed
    if not group.get("steps"):
        return None
    seconds = model.run(case, slabs, group["steps"], cold=group["cold"], seam_check=group["seam"])
    return seconds + (model.trace_write if group["traced"] else 0.0)


def refit(jobs: list, plan: e7_plan.Model) -> tuple:
    """(refitted model, {field: description of its data})."""
    groups = [group for job in jobs for group in job["groups"]]
    runs = [group for group in groups if group["role"] != "calibration" and group.get("case") in e7_plan.PARTICLES]
    measured = [group for group in runs if "construction" in group]
    pilots = [(group, pilot) for group in groups if group["role"] == "calibration"
              and group.get("case") in e7_plan.PARTICLES for pilot in group["pilots"]]
    fields, notes = {}, {}

    def passing(group):
        return [member["fps"] for member in group["member_runs"] if member["status"] == "pass" and member["fps"]]

    # loop: K = 1 cost per particle-step, efficiency per K, light-load floor per K
    costs = {}
    for dim in (2, 3):
        samples = [(particles(group["case"]), 1.0 / fps) for group in runs
                   if group["role"] == "reference" and group["slabs"] == 1 and dimension(group["case"]) == dim
                   for fps in passing(group)]
        if samples:
            costs[dim] = sum(step for _, step in samples) / sum(count for count, _ in samples)
            fields[f"cost_{dim}d"] = costs[dim]
            notes[f"cost_{dim}d"] = (f"{len(samples)} K = 1 reference runs ("
                                     + ", ".join(sorted({short(group['case']) for group in runs if group['role'] ==
                                                         'reference' and group['slabs'] == 1
                                                         and dimension(group['case']) == dim})) + ")")

    def cost_of(dim):
        return costs.get(dim, plan.cost_3d if dim == 3 else plan.cost_2d)

    timing = [(dimension(group["case"]), group["slabs"], particles(group["case"]) / group["slabs"], 1.0 / fps,
               short(group["case"])) for group in runs
              if group["role"] in ARM_ROLES or (group["role"] == "reference" and group["slabs"] == 2)
              for fps in passing(group)]
    efficiencies = {2: dict(plan.efficiency), 3: dict(plan.efficiency_3d or plan.efficiency)}
    for dim in (2, 3):
        for slabs in sorted({sample[1] for sample in timing if sample[0] == dim}):
            heavy = [sample for sample in timing if sample[:2] == (dim, slabs) and sample[2] > e7_plan.LIGHT_PER_CARD]
            if heavy:
                efficiencies[dim][slabs] = mean(cost_of(dim) * per_card / step for _, _, per_card, step, _ in heavy)
                key = "efficiency" if dim == 2 else "efficiency_3d"
                notes.setdefault(key, [])
                notes[key].append(f"K={slabs}: {len(heavy)} runs ({', '.join(sorted({s[4] for s in heavy}))})")
    fields["efficiency"], fields["efficiency_3d"] = efficiencies[2], efficiencies[3]
    floors = dict(plan.floor)
    for slabs in sorted({sample[1] for sample in timing}):
        light = [sample for sample in timing if sample[1] == slabs and sample[2] <= e7_plan.LIGHT_PER_CARD
                 and sample[3] > FLOOR_MARGIN * cost_of(sample[0]) * sample[2] / efficiencies[sample[0]][slabs]]
        if light:
            floors[slabs] = mean(step for _, _, _, step, _ in light)
            notes.setdefault("floor", []).append(f"K={slabs}: {len(light)} runs "
                                                 f"({', '.join(sorted({s[4] for s in light}))})")
    fields["floor"] = floors
    for key in ("efficiency", "efficiency_3d", "floor"):
        if key in notes:
            notes[key] = "; ".join(notes[key])

    # construction, bootstrap, post-run
    load = fit_line([(particles(group["case"]) / 1e6, group["read_in"]) for group in measured])
    load_intercept = 0.0
    if load:
        load_intercept, fields["load_per_million"] = load
        notes["load_per_million"] = f"{len(measured)} groups (intercept {load_intercept:.2f} s -> run_overhead)"
    for warm in (False, True):
        samples = [(group["slabs"], group["construction"]) for group in measured if group["cold"] != warm]
        if not warm:
            samples += [(group["slabs"], pilot["bootstrap_start"] - pilot["partition_done"]) for group, pilot in pilots]
        if samples:
            key = "warm_per_simulator" if warm else "cold_per_simulator"
            fields[key] = sum(seconds for _, seconds in samples) / sum(slabs for slabs, _ in samples)
            notes[key] = f"{len(samples)} groups" + ("" if warm else f" (calibration pilots {len(pilots)})")
    bootstrap = fit_line([(particles(group["case"]) / 1e6, group["bootstrap"]) for group in measured]
                         + [(particles(group["case"]) / 1e6, pilot["bootstrap_end"] - pilot["bootstrap_start"])
                            for group, pilot in pilots])
    if bootstrap:
        fields["bootstrap_fixed"], fields["bootstrap_per_million"] = bootstrap
        notes["bootstrap_fixed"] = notes["bootstrap_per_million"] = f"{len(measured) + len(pilots)} groups and pilots"
    plain = [group for group in measured if not group["seam"]]
    post = fit_two([(particles(group["case"]) / 1e6, group["slabs"], group["post"]) for group in plain])
    if post:
        fields["post_per_million"], fields["post_per_slab"] = post
        notes["post_per_million"] = notes["post_per_slab"] = f"{len(plain)} groups without the seam check"
    # with the seam check: the 2-D terms from the untraced seam-checked groups (soaks), then the step trace's write
    # from the 2-D traced runs (else the 3-D ones), then the 3-D terms from the 3-D groups, its write taken off
    seam = [group for group in measured if group["seam"]]

    def scale_seam(dim, chosen, trace_write):
        """The plan's terms of one dimension scaled to the chosen groups' sum (the data cannot separate them)."""
        names = (("seam_2d_per_million", "seam_2d_fixed") if dim == 2
                 else ("seam_3d_per_million", "seam_3d_per_million_per_slab"))
        planned = sum(seam_seconds(plan, group["case"], group["slabs"]) for group in chosen)
        measured_sum = sum(group["post"] - (trace_write if group["traced"] else 0.0) for group in chosen)
        scale = max(measured_sum, 0.0) / planned if planned else 1.0
        for name in names:
            fields[name] = scale * getattr(plan, name)
            notes[name] = f"{len(chosen)} seam-checked groups, {scale:.2f} x the plan's terms"

    seam_2d = [group for group in seam if dimension(group["case"]) == 2]
    trace_write = plan.trace_write
    solution = None
    if len({group["case"] for group in seam_2d}) >= 2 and {group["traced"] for group in seam_2d} == {True, False}:
        solution = least_squares([(group["post"], [particles(group["case"]) / 1e6, 1.0, float(group["traced"])])
                                  for group in seam_2d])
    if solution and min(solution) >= 0:
        fields["seam_2d_per_million"], fields["seam_2d_fixed"], trace_write = solution
        fields["trace_write"] = trace_write
        traced_count = sum(group["traced"] for group in seam_2d)
        for name in ("seam_2d_per_million", "seam_2d_fixed", "trace_write"):
            notes[name] = (f"least squares over the 2-D seam-checked groups: {traced_count} traced, "
                           f"{len(seam_2d) - traced_count} untraced")
    elif seam_2d:
        scale_seam(2, seam_2d, trace_write)
    seam_3d = [group for group in seam if dimension(group["case"]) == 3]
    if seam_3d:
        values = [(particles(group["case"]) / 1e6, group["slabs"],
                   group["post"] - (trace_write if group["traced"] else 0.0)) for group in seam_3d]
        fitted = fit_two([(millions, slabs * millions, value) for millions, slabs, value in values])
        if fitted:
            fields["seam_3d_per_million"], fields["seam_3d_per_million_per_slab"] = fitted
            notes["seam_3d_per_million"] = notes["seam_3d_per_million_per_slab"] = (
                f"least squares over {len(seam_3d)} 3-D seam-checked groups (trace write {trace_write:.1f} s off)")
        else:
            scale_seam(3, seam_3d, trace_write)

    # what the stamps leave: run_overhead; then the calibrations; then the jobs
    model = dataclasses.replace(plan, **fields, run_overhead=0.0)
    residuals = [group["duration"] - predict_group(model, group) for group in measured
                 if group["duration"] is not None and predict_group(model, group) is not None]
    if residuals:
        fields["run_overhead"] = mean(residuals)
        outside = mean(group["duration"] - group["in_bench"] for group in measured if group["duration"] is not None)
        notes["run_overhead"] = (f"{len(residuals)} groups, std {statistics.pstdev(residuals):.1f} s; outside the "
                                 f"bench {outside:.1f} s, read-in intercept {load_intercept:.1f} s")
    calibrations = [group for group in groups if group["role"] == "calibration" and group["duration"] is not None
                    and group["case"] in e7_plan.PARTICLES]
    if calibrations:
        counts = [group["pilot_count"] for group in calibrations if group["pilot_count"]]
        if counts:
            fields["calibration_rounds"] = int(round(statistics.median(counts)))
        model = dataclasses.replace(plan, **fields, calibration_fixed=0.0)
        fields["calibration_fixed"] = mean(group["duration"] - predict_group(model, group) for group in calibrations)
        notes["calibration_rounds"] = notes["calibration_fixed"] = (
            f"{len(calibrations)} calibrations, pilots " + ", ".join(str(count) for count in counts))
    overheads = []
    for job in jobs:
        durations = [item["duration"] for item in job["items"] if item["duration"] is not None]
        if job["elapsed"] is not None:
            overheads.append(job["elapsed"] - sum(durations))
    if overheads:
        fields["job_overhead"] = mean(overheads)
        notes["job_overhead"] = f"{len(overheads)} jobs: " + ", ".join(f"{value:.0f} s" for value in overheads)
    return dataclasses.replace(plan, **fields), notes


# ----------------------------------------------------------------------------- plan items and budgets

def item_label(item: dict) -> str:
    if item["kind"] in ("point", "cross"):
        point = item["point"]
        return ("x_" if item["kind"] == "cross" else "") + f"{short(point.case)}_K{point.slabs}"
    if item["kind"] == "precheck":
        return f"pre_{short(item['case'])}_{item['reference_kind']}_x{item['parties']}"
    return item["label"]


def campaign(model: e7_plan.Model) -> tuple:
    """({label: (batch, line, seconds)}, {(batch, line): job seconds}) of the whole campaign under MODEL."""
    jobs = e7_plan.build_jobs(e7_plan.campaign_points(), model)
    labels, totals = {}, {}
    for (batch, line), items in jobs.items():
        for item in items:
            labels[item_label(item)] = (batch, line, item["seconds"])
        totals[(batch, line)] = model.job_overhead + sum(item["seconds"] for item in items)
    return labels, totals


# ----------------------------------------------------------------------------- report

def number(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def field_text(name: str, value) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        scale = 1e3 if name == "floor" else 1.0
        return ", ".join(f"{key}: {item * scale:.{1 if name == 'floor' else 3}f}" for key, item in sorted(value.items()))
    if name.startswith("cost_"):
        return f"{value * 1e9:.3f}"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


UNITS = {"cost_2d": "ns per particle-step", "cost_3d": "ns per particle-step", "floor": "ms per step",
         "floor_k1": "s per step"}


def report(jobs: list, plan: e7_plan.Model, model: e7_plan.Model, notes: dict) -> tuple:
    plan_labels, plan_totals = campaign(plan)
    refit_labels, refit_totals = campaign(model)
    lines = ["## Time model refit (batch report item 10)", ""]
    for job in jobs:
        lines.append(f"- job {job['name']}: elapsed {number((job['elapsed'] or 0) / 3600, 2)} h "
                     f"({job['elapsed_source']}), prelude {number(job['ready'], 0)} s, {len(job['items'])} items, "
                     f"{len(job['groups'])} run / calibration groups ({job['stdout']})")
    lines += ["", "### Model fields: plan and refit", "", "| field | plan | refit | data |", "|---|---|---|---|"]
    for field in dataclasses.fields(e7_plan.Model):
        before, after = getattr(plan, field.name), getattr(model, field.name)
        unit = f" ({UNITS[field.name]})" if field.name in UNITS else ""
        lines.append(f"| {field.name}{unit} | {field_text(field.name, before)} | {field_text(field.name, after)} | "
                     f"{notes.get(field.name, 'not measured: the plan value')} |")

    lines += ["", "### Step times (ms): measured mean over the passing runs, the plan's and the refit's model", "",
              "| case | K | runs | kind | per card (M) | measured | plan | refit |", "|---|---|---|---|---|---|---|---|"]
    table: dict = {}
    for job in jobs:
        for group in job["groups"]:
            if group["role"] == "calibration" or group.get("case") not in e7_plan.PARTICLES:
                continue
            kind = {"reference": "K = 1 reference" if group["slabs"] == 1 else "pair reference",
                    "equal": "timing arm", "calibrated": "timing arm"}.get(group["role"])
            if kind is None:
                continue
            steps = [1.0 / member["fps"] for member in group["member_runs"] if member["status"] == "pass" and member["fps"]]
            table.setdefault((group["case"], group["slabs"], kind), []).extend(steps)
    for (case, slabs, kind), steps in sorted(table.items()):
        lines.append(f"| {case} | {slabs} | {len(steps)} | {kind} | {particles(case) / slabs / 1e6:.2f} | "
                     f"{number(mean(steps) * 1e3 if steps else None, 2)} | {plan.step_time(case, slabs) * 1e3:.2f} | "
                     f"{model.step_time(case, slabs) * 1e3:.2f} |")

    lines += ["", "### Items (min): the plan's estimate, the refit's (as planned, and on what the item ran), the "
              "timeline", "", "| job | item | plan | refit | refit as run | measured | measured / plan |",
              "|---|---|---|---|---|---|---|"]
    item_rows = []
    for job in jobs:
        for item in job["items"]:
            planned = plan_labels.get(item["label"], (None, None, None))[2]
            refitted = refit_labels.get(item["label"], (None, None, None))[2]
            as_run = [predict_group(model, group) for group in item["groups"]]
            as_run = sum(as_run) if as_run and None not in as_run else None
            item_rows.append({"job": job["name"], "label": item["label"], "plan": planned, "refit": refitted,
                              "refit_as_run": as_run, "measured": item["duration"]})
            ratio = item["duration"] / planned if planned and item["duration"] is not None else None
            lines.append(f"| {job['name']} | {item['label']} | {number(planned and planned / 60)} | "
                         f"{number(refitted and refitted / 60)} | {number(as_run and as_run / 60)} | "
                         f"{number(item['duration'] and item['duration'] / 60)} | {number(ratio, 2)} |")
        match = re.fullmatch(r"b(\d)([AB])", job["name"])
        key = (int(match.group(1)), match.group(2)) if match else None
        lines.append(f"| {job['name']} | **job (overhead {number(plan.job_overhead / 60)} / "
                     f"{number(model.job_overhead / 60)} min)** | {number(plan_totals.get(key, 0) / 60 or None)} | "
                     f"{number(refit_totals.get(key, 0) / 60 or None)} | - | "
                     f"{number(job['elapsed'] and job['elapsed'] / 60)} | "
                     f"{number(job['elapsed'] / plan_totals[key] if key in plan_totals and job['elapsed'] else None, 2)} |")

    lines += ["", "### Budgets (node-h; a batch's wall = its longer line)", "",
              "| batch | plan A | plan B | plan total | refit A | refit B | refit total | refit wall (h) |",
              "|---|---|---|---|---|---|---|---|"]
    budgets = []
    for batch in sorted({batch for batch, _ in plan_totals}):
        before = {line: plan_totals.get((batch, line), 0.0) / 3600 for line in ("A", "B")}
        after = {line: refit_totals.get((batch, line), 0.0) / 3600 for line in ("A", "B")}
        budgets.append({"batch": batch, "plan": before, "refit": after})
        lines.append(f"| {batch} | {before['A']:.2f} | {before['B']:.2f} | {sum(before.values()):.2f} | "
                     f"{after['A']:.2f} | {after['B']:.2f} | {sum(after.values()):.2f} | {max(after.values()):.2f} |")
    later = [entry for entry in budgets if entry["batch"] >= 2]
    lines += ["", f"Batches 2-4: plan {sum(sum(entry['plan'].values()) for entry in later):.1f} node-h, refit "
              f"{sum(sum(entry['refit'].values()) for entry in later):.1f} node-h (queue time not included)."]
    data = {"fields": {name: getattr(model, name) for name in (field.name for field in dataclasses.fields(e7_plan.Model))},
            "notes": notes, "items": item_rows, "budgets": budgets,
            "groups": [{key: value for key, value in group.items() if key != "pilots"} | {"pilots": len(group.get("pilots", []))}
                       for job in jobs for group in job["groups"]]}
    return "\n".join(lines) + "\n", data


def main() -> int:
    parser = argparse.ArgumentParser(description="E7: refit e7_plan.py's time model to a batch's measurements")
    parser.add_argument("directories", nargs="+", type=pathlib.Path)
    parser.add_argument("--stdout", action="append", type=pathlib.Path, default=None,
                        help="the job's SLURM .out, once per directory in order (default: the *.out in the directory)")
    parser.add_argument("--out-model", default=None, help="write the refitted model fields (e7_plan.py --model-json)")
    parser.add_argument("--out", default=None, help="write the report (markdown); default stdout")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    if arguments.stdout and len(arguments.stdout) != len(arguments.directories):
        parser.error("give --stdout once per directory")
    jobs = []
    for position, directory in enumerate(arguments.directories):
        if arguments.stdout:
            stdout = arguments.stdout[position]
        else:
            candidates = sorted(directory.glob("*.out"))
            if len(candidates) != 1:
                parser.error(f"{directory}: {len(candidates)} *.out files, give --stdout")
            stdout = candidates[0]
        jobs.append(measure_job(directory, stdout))
    plan = e7_plan.Model()
    model, notes = refit(jobs, plan)
    text, data = report(jobs, plan, model, notes)
    if arguments.out:
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    if arguments.out_model:
        pathlib.Path(arguments.out_model).write_text(json.dumps(dataclasses.asdict(model), indent=1), encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
