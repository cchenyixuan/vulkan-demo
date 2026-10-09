"""
e7_plan.py — E7 full campaign on N56 (hp_5090, whole nodes): the job list, its node-hour budget and the
generated job scripts.

Protocol (user, 2026-10-09 and 2026-10-10):
  - cases/aligned/ only, simple walls (no *_adami case); 51 points (case, K >= 2) in the families F1-F8;
  - windows: 2-D 3000 steps (warmup 1000), 3-D 1500 steps (warmup 500), as probe34-37;
  - every point: one weight calibration on its node right before its trials (--weights auto --calibrate-only
    --weights-file, explicit --device-map), then TRIALS trials, each one reference set (K = 1 of the
    reference case on every participating card at the same time, all processes held at a start barrier
    after their bootstrap), the equal-weight arm and the calibrated arm (--weights-file + --device-map), the
    order rotated over the trials (R E C / E C R / C R E, continued for trials 4 and 5); 5 trials when the
    point has at most 4M particles per card (the nominal 4M class included: actual <= 4.2M, LIGHT_PER_CARD),
    else 3; timing runs without the seam check; then two step-traced runs (E29), one per arm, with the seam
    check (eta and r of the eta-r figure must come from one arm);
  - references: strong = K = 1 of the case, eta = fps_K / (K fps_1); weak (F3, F4) = K = 1 of the per-card
    case, eta = fps_K / fps_1; n11320 (128M) = 4 simultaneous K = 2 pairs of the case, eta = fps_K /
    ((K / 2) fps_pair), reported separately; a K = 1 reference that the pre-check finds too large for one card
    switches to K = 2 pairs the same way (at K = 2 there is then no reference: the pair is the point itself);
  - line A (K <= 4, GPUs 0-3 of a whole node) and line B (K = 8), the two lines in parallel (hp5090: 16 GPUs =
    two nodes);
  - batches (user, 2026-10-10): one job per line and batch, a batch starts after the previous one is reported
    and approved. Batch 0 (preparation, no paper data) runs at the start of batch 1's jobs: on line B a
    3-minute self-test of every E7 output path on the 1M case (anatomy durations, defrag log, host loop lines,
    pool series), then on both lines the pre-checks of the large K = 1 references (2-D n8000 64.4M, n9000
    81.4M; 3-D n160_k8 36.4M, n200_k8 69.6M, n416 76.2M) and of n11320's K = 2 pairs (64.3M per card), each at
    the line's real concurrency (A: 4 processes, B: 8; the pairs: 4), build + bootstrap + 50 steps, per-card
    VRAM peak and host memory peak (a reference that does not fit becomes K = 2 pairs, see above). Batch 1 =
    representative points of every reference kind and load (BATCH_ONE) + the n8000 K = 8 soak (17,000 steps,
    V7_POOL_PEAKS=1, the v7 name of V6_POOL_PEAKS); batch 2 = the rest of F1, F2, F3 (2-D main results);
    batch 3 = the rest of F4, F5 + F6 (3-D main results); batch 4 = F7, F8, the n8000 --anatomy at K = 1 and
    K = 8 (V7_LOOP_TRACE=1: the host loop's submit / wait time per step, which the step trace does not have),
    the cube K = 8 soak (~1 h, V7_POOL_PEAKS=1, E15 developed flow) and the cross-node repeats: each line's job
    re-runs one light and one heavy point of the other line (3 trials with references, calibrated on that
    node, no traces) and skips a repeat that lands on the node of the original point (then it is resubmitted
    with --exclude).

Time model per run = construction + bootstrap + loop + post-run checks + 10 s, every term measured in the E7
smoke (job 1679191, wqd10nbj04g2, v7-rc1):
  - construction: case load from the node-local .obj cache (~0.35 s per million particles) + the simulators:
    a configuration new to the job compiles its pipelines (~17 s per simulator; the driver's shader disk cache
    is per job, node-local, so every job starts cold), a repeat in the same job hits the cache (~1.2 s per
    simulator). Cold: the calibration pilots, the first reference set of every reference case, the pre-checks,
    the anatomy / trace / soak runs and the step-traced runs (their extra device extension may change the
    cache key; counted cold to be safe); warm: the timed arms (pilot 1 = equal weights, the last pilot = the
    calibrated cuts) and later reference sets;
  - bootstrap: ~0.5 s per million particles + 2 s;
  - loop: steps / fps, fps from the K = 1 cost per particle-step (2-D 1.258 ns: 32M K = 1 at 24.8 fps; 3-D
    6.0 ns: cube K = 2 at 4.2 fps), an efficiency per K and a per-K floor for light loads (2-D 16M K = 8 ran
    110 / 146 fps at 2M per card);
  - post-run: without the seam check ~0.1 s per million + 1 s per slab (estimate); with it 2-D ~0.42 s per
    million + 8 s, 3-D (0.26 + 0.16 K) s per million (cube 43 / 71 / 118 s at K = 2 / 4 / 8); a step trace
    writes its CSV files in ~15 s more (estimate).
Job overhead 300 s (prelude, configuration print, provenance, bring-up, device check, cache stage-in at
~120 MB/s; the smoke's took 157 s with 4.6 GB staged).
The numbers are planning estimates; the timeouts of the generated scripts are 2 x the estimate + 600 s
(minimum 900 s), the job time limit 2 x the job estimate + 1 h.

After a batch, e7_refit.py fits the model's fields to the batch's measurements; --model-json FILE plans with them.

Usage:
    python docs/cluster_v6/scripts/e7_plan.py [--markdown] [--json OUT] [--emit-jobs DIR [--batch N ...]]
        [--model-json FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

# total particles (cases/aligned/*/generate.txt line 1)
PARTICLES = {
    "cavity2d_n240": 68_644, "cavity2d_n360": 145_924, "cavity2d_n520": 293_764, "cavity2d_n720": 550_564,
    "cavity2d_n1000": 1_044_484, "cavity2d_n1440": 2_137_444, "cavity2d_n2000": 4_088_484,
    "cavity2d_n2840": 8_191_044, "cavity2d_n4000": 16_176_484, "cavity2d_n5640": 32_058_244,
    "cavity2d_n8000": 64_352_484, "cavity2d_n9000": 81_396_484, "cavity2d_n11320": 128_640_964,
    "cavity2d_n1440_k2": 4_242_724, "cavity2d_n1440_k4": 8_453_284, "cavity2d_n1440_k8": 16_874_404,
    "cavity2d_n2000_k2": 8_132_484, "cavity2d_n2000_k4": 16_220_484, "cavity2d_n2000_k8": 32_396_484,
    "cavity2d_n2840_k2": 16_319_124, "cavity2d_n2840_k4": 32_575_284, "cavity2d_n2840_k8": 65_087_604,
    "cavity2d_n4000_k2": 32_264_484, "cavity2d_n4000_k4": 64_440_484, "cavity2d_n4000_k8": 128_792_484,
    "cavity2d_n5640_k2": 63_991_924, "cavity2d_n5640_k4": 127_859_284, "cavity2d_n5640_k8": 255_594_004,
    "cavity3d_n104_b9": 1_815_848, "cavity3d_n160": 4_741_632, "cavity3d_n160_k2": 9_257_472,
    "cavity3d_n160_k4": 18_289_152, "cavity3d_n160_k8": 36_352_512, "cavity3d_n200": 8_998_912,
    "cavity3d_n200_b9": 10_360_232, "cavity3d_n200_k2": 17_651_712, "cavity3d_n200_k4": 34_957_312,
    "cavity3d_n200_k8": 69_568_512, "cavity3d_n200_x96": 4_499_456, "cavity3d_n416": 76_225_024,
}
LIGHT_PER_CARD = 4.2e6          # at most this many particles per card: 5 trials (the nominal 4M class included)
BRINGUP_CASE = "cavity2d_n1000"
# pre-checks (user item 9): the large K = 1 references at each line's concurrency, n11320's pairs on line B
PRECHECKS = {"A": [("cavity2d_n8000", "k1", 4), ("cavity3d_n160_k8", "k1", 4), ("cavity3d_n200_k8", "k1", 4)],
             "B": [("cavity2d_n8000", "k1", 8), ("cavity2d_n9000", "k1", 8), ("cavity3d_n160_k8", "k1", 8),
                   ("cavity3d_n200_k8", "k1", 8), ("cavity3d_n416", "k1", 8), ("cavity2d_n11320", "pairs", 4)]}
PRECHECK_STEPS = 50
# cross-node repeats (user item 13): one light and one heavy point of the other line, 3 trials with references
CROSS = {"A": [("cavity2d_n4000", 8), ("cavity2d_n8000", 8)],          # line B points, re-run in line A's last job
         "B": [("cavity2d_n2840", 4), ("cavity2d_n8000", 4)]}          # line A points, re-run in line B's last job
CROSS_TRIALS = 3
# batch 1: representative points (user 2026-10-10): every reference kind (K = 1 at 4 and 8 cards, weak, pairs,
# weak + strong) and both loads (5-trial light points)
BATCH_ONE = {"A": [("cavity2d_n8000", 4), ("cavity2d_n2840_k4", 4), ("cavity3d_n160_k8", 4), ("cavity2d_n1440_k2", 2)],
             "B": [("cavity2d_n8000", 8), ("cavity2d_n4000", 8), ("cavity2d_n11320", 8), ("cavity3d_n160_k8", 8)]}
# batches 2-4: the points not in batch 1 whose first family is one of these
BATCH_FAMILIES = {2: ("F1", "F2", "F3"), 3: ("F4", "F5", "F6"), 4: ("F7", "F8")}
# extras per batch and line: (before the points, after the points); batch 0 runs at the start of batch 1
BATCH_EXTRAS = {1: {"A": (["precheck"], []), "B": (["selftest", "precheck"], ["soak_2d"])},
                2: {"A": ([], []), "B": ([], [])},
                3: {"A": ([], []), "B": ([], [])},
                4: {"A": ([], ["cross"]), "B": ([], ["anatomy", "cross", "soak_3d"])}}


@dataclasses.dataclass
class Point:
    family: str
    case: str
    slabs: int
    references: list        # "same" = K = 1 of the case; "pairs" = K = 2 pairs of the case; else a k1 case (weak)
    cross: bool = False

    @property
    def line(self) -> str:
        return "B" if self.slabs == 8 else "A"

    @property
    def dimension(self) -> int:
        return 3 if self.case.startswith("cavity3d") else 2

    @property
    def per_card(self) -> float:
        return PARTICLES[self.case] / self.slabs

    @property
    def trials(self) -> int:
        if self.cross:
            return CROSS_TRIALS
        return 5 if self.per_card <= LIGHT_PER_CARD else 3

    @property
    def traces(self) -> int:
        return 0 if self.cross else 2


def campaign_points() -> list[Point]:
    points: list[Point] = []
    seen = set()

    def add(family, case, slabs, reference="same"):
        key = (case, slabs)
        if key in seen:                     # one point serves every family that names it (K runs shared);
            for point in points:            # a family with another reference adds its reference set
                if (point.case, point.slabs) == key:
                    point.family += "+" + family
                    if reference not in point.references:
                        point.references.append(reference)
            return
        seen.add(key)
        points.append(Point(family, case, slabs, [reference]))

    # F1 2-D strong scaling at K = 8 (probe34 successor) + n9000 (K = 1 fits) + n11320 (pairs)
    for n in (2000, 2840, 4000, 5640, 8000, 9000):
        add("F1", f"cavity2d_n{n}", 8)
    add("F1", "cavity2d_n11320", 8, "pairs")
    # F2 2-D fixed-N K sweep (probe36): 32M and 64M at K = 2 / 4 (K = 8 = F1)
    for n in (5640, 8000):
        for slabs in (2, 4):
            add("F2", f"cavity2d_n{n}", slabs)
    # F3 2-D weak, 2 / 4 / 8 / 16 / 32M per card (probe35): k2 / k4 / k8 members, reference = the k1 case
    for n in (1440, 2000, 2840, 4000, 5640):
        for slabs in (2, 4, 8):
            add("F3", f"cavity2d_n{n}_k{slabs}", slabs, f"cavity2d_n{n}")
    # F4 3-D weak, 4M / 8M per card (probe37), reference = the k1 case
    for n in (160, 200):
        for slabs in (2, 4, 8):
            add("F4", f"cavity3d_n{n}_k{slabs}", slabs, f"cavity3d_n{n}")
    # F5 3-D strong, stretched 32M / 64M (= the k8 geometries) at K = 2 / 4 / 8, reference = K = 1 of the case
    for n in (160, 200):
        for slabs in (2, 4, 8):
            add("F5", f"cavity3d_n{n}_k8", slabs)
    # F6 cube at K = 8
    add("F6", "cavity3d_n416", 8)
    # F7 E29 per-card points, 2-D (n_B ~ 32k / 130k / 0.5M / 2M / 8M per card)
    for slabs, sizes in ((2, (240, 520, 1000, 2000, 4000)), (4, (360, 720, 1440, 2840, 5640)),
                         (8, (520, 1000, 2000, 4000, 8000))):
        for n in sizes:
            add("F7", f"cavity2d_n{n}", slabs)
    # F8 E29 per-card points, 3-D (0.7M / 4.7M per card, narrow slab), K = 2
    for case in ("cavity3d_n104_b9", "cavity3d_n200_b9", "cavity3d_n200_x96"):
        add("F8", case, 2)
    return points


def cross_points(points: list[Point]) -> dict:
    """Line -> the other line's points it re-runs (copies with cross=True)."""
    by_key = {(point.case, point.slabs): point for point in points}
    return {line: [dataclasses.replace(by_key[key], references=list(by_key[key].references), cross=True)
                   for key in keys] for line, keys in CROSS.items()}


@dataclasses.dataclass
class Model:
    cost_2d: float = 1.258e-9       # s per particle-step, K = 1 (smoke: 32M K = 1 references, 24.8 fps mean)
    cost_3d: float = 6.0e-9         # s per particle-step, K = 1 (smoke: cube 76.2M K = 2 at 4.2 fps, eta ~0.96)
    floor_k1: float = 0.0014        # s per step, dispatch-bound K = 1 (1M aligned K = 1: 651-721 fps)
    floor: dict = dataclasses.field(default_factory=lambda: {2: 0.0018, 4: 0.0045, 8: 0.0080})
    efficiency: dict = dataclasses.field(default_factory=lambda: {2: 0.95, 4: 0.90, 8: 0.87})
    efficiency_3d: dict = None      # per K for the 3-D cases; None = the 2-D values (the plan)
    cold_per_simulator: float = 17.0      # s: pipeline compilation of a configuration new to the job
    warm_per_simulator: float = 1.2       # s: driver shader disk cache hit
    load_per_million: float = 0.35        # s per million particles: .npy load + partition
    bootstrap_fixed: float = 2.0
    bootstrap_per_million: float = 0.5
    post_per_million: float = 0.1         # post-run checks without the seam check: s per million + s per slab
    post_per_slab: float = 1.0
    seam_2d_per_million: float = 0.42     # with the seam check, 2-D: s per million + fixed
    seam_2d_fixed: float = 8.0
    seam_3d_per_million: float = 0.26     # with the seam check, 3-D: (this + per slab x K) s per million
    seam_3d_per_million_per_slab: float = 0.16
    calibration_rounds: int = 2
    calibration_fixed: float = 20.0       # s per calibration besides its pilots
    pilot_steps: int = 500          # 200 warmup + 300 measured
    run_overhead: float = 10.0      # s per run: process start, parse, sync, telemetry window
    trace_write: float = 15.0       # s: a step trace's CSV files
    job_overhead: float = 300.0     # s per job: prelude, bring-up, device check, cache stage-in (smoke: 157 s)
    anatomy_steps: int = 3000       # three anatomy frames per simulator at the default defrag cadence (1000)
    soak_steps: int = 17000
    cube_soak_seconds: float = 3600.0

    def window(self, case: str) -> tuple[int, int]:
        return (1500, 500) if case.startswith("cavity3d") else (3000, 1000)

    def step_time(self, case: str, slabs: int) -> float:
        three_dimensional = case.startswith("cavity3d")
        cost = self.cost_3d if three_dimensional else self.cost_2d
        per_card = PARTICLES[case] / slabs
        if slabs == 1:
            return max(self.floor_k1, cost * per_card)
        efficiency = (self.efficiency_3d or self.efficiency) if three_dimensional else self.efficiency
        return max(self.floor[slabs], cost * per_card / efficiency[slabs])

    def setup(self, case: str, slabs: int, cold: bool, seam_check: bool = False) -> float:
        """Construction + bootstrap + post-run checks + per-run overhead of one run (a reference set is one
        process per card in parallel: one simulator's construction; a K = 2 pair two)."""
        millions = PARTICLES[case] / 1e6
        construction = ((self.cold_per_simulator if cold else self.warm_per_simulator) * slabs
                        + self.load_per_million * millions)
        bootstrap = self.bootstrap_fixed + self.bootstrap_per_million * millions
        if not seam_check:
            post = self.post_per_million * millions + self.post_per_slab * slabs
        elif case.startswith("cavity3d"):
            post = (self.seam_3d_per_million + self.seam_3d_per_million_per_slab * slabs) * millions
        else:
            post = self.seam_2d_per_million * millions + self.seam_2d_fixed
        return construction + bootstrap + post + self.run_overhead

    def run(self, case: str, slabs: int, steps: int, cold: bool, seam_check: bool = False) -> float:
        return self.setup(case, slabs, cold, seam_check) + steps * self.step_time(case, slabs)


def load_model(path) -> Model:
    """Model() with the fields of a JSON object replaced (e7_refit.py --out-model); per-K tables get int keys."""
    model = Model()
    if not path:
        return model
    overrides = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(overrides) - {field.name for field in dataclasses.fields(Model)})
    if unknown:
        raise SystemExit(f"{path}: not fields of the model: {', '.join(unknown)}")
    for name, value in overrides.items():
        setattr(model, name, {int(key): item for key, item in value.items()} if isinstance(value, dict) else value)
    return model


def reference_spec(point: Point, kind: str) -> tuple[str, int]:
    """(case, simulators per process) of one reference kind of a point."""
    if kind == "pairs":
        return point.case, 2
    return (point.case if kind == "same" else kind), 1


def point_seconds(point: Point, model: Model) -> dict:
    steps, _ = model.window(point.case)
    run_warm = model.run(point.case, point.slabs, steps, cold=False)
    references_first = references_later = 0.0
    reference_runs = {}
    for kind in point.references:
        reference_case, slabs = reference_spec(point, kind)
        first = model.run(reference_case, slabs, steps, cold=True)
        later = model.run(reference_case, slabs, steps, cold=False)
        references_first += first
        references_later += later
        reference_runs[kind] = first
    pilot = model.setup(point.case, point.slabs, cold=True) + model.pilot_steps * model.step_time(point.case,
                                                                                                 point.slabs)
    calibration = model.calibration_rounds * pilot + model.calibration_fixed
    trials_total = references_first + (point.trials - 1) * references_later + point.trials * 2 * run_warm
    traced = model.run(point.case, point.slabs, steps, cold=True, seam_check=True) + model.trace_write
    total = calibration + trials_total + point.traces * traced
    return {"run": run_warm, "reference": references_later, "reference_first": references_first,
            "reference_runs": reference_runs, "calibration": calibration, "traced": traced, "total": total}


def precheck_seconds(case: str, kind: str, model: Model) -> float:
    slabs = 2 if kind == "pairs" else 1
    return model.run(case, slabs, PRECHECK_STEPS, cold=True)


def extras(model: Model) -> dict:
    """The extra items by group: label, kind, case, K, steps, warmup, estimated seconds. selftest runs every E7
    output path once on the 1M case before the pre-checks (anatomy durations, defrag log, [loop] lines, pool
    series, stage epochs, loop-window clocks)."""
    cube_steps = int(round(model.cube_soak_seconds / model.step_time("cavity3d_n416", 8), -3))
    items = {
        "selftest": [("selftest_2d_n1000_K8", "selftest", "cavity2d_n1000", 8, 1000)],
        "soak_2d": [("soak_2d_n8000_K8", "soak", "cavity2d_n8000", 8, model.soak_steps)],
        "anatomy": [("anatomy_2d_n8000_K1", "anatomy", "cavity2d_n8000", 1, model.anatomy_steps),
                    ("anatomy_2d_n8000_K8", "anatomy", "cavity2d_n8000", 8, model.anatomy_steps)],
        "soak_3d": [("soak_3d_n416_K8", "soak", "cavity3d_n416", 8, cube_steps)],
    }
    out = {}
    for group, entries in items.items():
        out[group] = []
        for label, kind, case, slabs, steps in entries:
            seconds = model.run(case, slabs, steps, cold=True, seam_check=kind == "soak")
            warmup = 200 if kind == "selftest" else model.window(case)[1]
            out[group].append({"label": label, "kind": kind, "case": case, "slabs": slabs, "steps": steps,
                               "warmup": warmup, "seconds": seconds})
    return out


def timeout(seconds: float) -> int:
    return int(max(900, math.ceil((2.0 * seconds + 600) / 60.0) * 60))


def batch_points(points: list[Point]) -> dict:
    """Batch -> line -> points (every point in exactly one batch)."""
    by_key = {(point.case, point.slabs): point for point in points}
    batches = {batch: {"A": [], "B": []} for batch in (1, 2, 3, 4)}
    first = set()
    for line, keys in BATCH_ONE.items():
        for key in keys:
            point = by_key[key]
            if point.line != line:
                raise SystemExit(f"batch 1: {key} is a line {point.line} point")
            batches[1][line].append(point)
            first.add(key)
    for point in points:
        if (point.case, point.slabs) in first:
            continue
        batch = next(batch for batch, families in BATCH_FAMILIES.items() if point.family.split("+")[0] in families)
        batches[batch][point.line].append(point)
    return batches


def build_jobs(points: list[Point], model: Model) -> dict:
    """(batch, line) -> [items], each item a dict with its kind, arguments and estimated seconds."""
    crosses = cross_points(points)
    extra_items = extras(model)
    batches = batch_points(points)
    jobs = {}
    for batch, lines in batches.items():
        for line in ("A", "B"):
            before, after = BATCH_EXTRAS[batch][line]
            items = []

            def add_groups(groups):
                for group in groups:
                    if group == "precheck":
                        for case, kind, parties in PRECHECKS[line]:
                            items.append({"kind": "precheck", "case": case, "reference_kind": kind,
                                          "parties": parties, "steps": PRECHECK_STEPS,
                                          "seconds": precheck_seconds(case, kind, model)})
                    elif group == "cross":
                        for point in crosses[line]:
                            items.append({"kind": "cross", "point": point,
                                          "seconds": point_seconds(point, model)["total"],
                                          "original_line": "B" if line == "A" else "A"})
                    else:
                        items.extend(dict(entry) for entry in extra_items[group])

            add_groups(before)
            for point in lines[line]:
                items.append({"kind": "point", "point": point, "seconds": point_seconds(point, model)["total"]})
            add_groups(after)
            jobs[(batch, line)] = items
    assigned = sum(1 for items in jobs.values() for item in items if item["kind"] == "point")
    if assigned != len(points):
        raise SystemExit(f"{assigned} points in the batches, {len(points)} in the campaign")
    return jobs


def job_cases(items: list[dict]) -> list[str]:
    cases = {BRINGUP_CASE}
    for item in items:
        if item["kind"] in ("point", "cross"):
            point = item["point"]
            cases.add(point.case)
            for kind in point.references:
                cases.add(reference_spec(point, kind)[0])
        else:
            cases.add(item["case"])
    return sorted(cases)


def emit_item(item: dict, model: Model) -> list[str]:
    if item["kind"] == "precheck":
        return [f"e7_precheck {item['case']} {item['reference_kind']} {item['parties']} {item['steps']} "
                f"{timeout(item['seconds'])}"]
    if item["kind"] in ("point", "cross"):
        point = item["point"]
        seconds = point_seconds(point, model)
        steps, warmup = model.window(point.case)
        references = " ".join(f"{kind}:{timeout(seconds['reference_runs'][kind])}" for kind in point.references)
        arguments = (f"{point.family} {point.case} {point.slabs} {point.trials} {steps} {warmup} "
                     f"{timeout(seconds['calibration'])} {timeout(seconds['run'])} {timeout(seconds['traced'])} "
                     f"{point.traces} {references}")
        if item["kind"] == "cross":
            return [f"e7_cross {item['original_line']} {arguments}"]
        return [f"e7_point {arguments}"]
    return [f"e7_extra {item['kind']} {item['label']} {item['case']} {item['slabs']} {item['steps']} "
            f"{item['warmup']} {timeout(item['seconds'])}"]


def job_name(batch: int, line: str) -> str:
    return f"b{batch}{line}"


def emit_jobs(jobs: dict, model: Model, directory: pathlib.Path, batches) -> list[pathlib.Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for (batch, line), items in jobs.items():
        if batch not in batches or not items:
            continue
        name = job_name(batch, line)
        estimate = model.job_overhead + sum(item["seconds"] for item in items)
        limit = int(math.ceil((2.0 * estimate + 3600) / 3600.0))
        content = ", ".join(sorted({(item["point"].family if "point" in item else item["kind"]) for item in items}))
        body = [
            "#!/bin/bash",
            "#SBATCH -p hp_5090",
            "#SBATCH -A hp5090",
            "#SBATCH -N 1",
            "#SBATCH --gpus=8",
            f"#SBATCH --time={limit:02d}:00:00",
            f"#SBATCH -J e7_{line}",
            "#SBATCH --dependency=singleton",
            f"#SBATCH -o /data/run01/scxm138/logs/e7_{name}_%j.out",
            f"# E7 full campaign, batch {batch}, line {line} (generated by docs/cluster_v6/scripts/e7_plan.py "
            "--emit-jobs; do not edit by hand).",
            f"# Estimate {estimate / 3600:.2f} h, {len(items)} items: {content}. The protocol is e7_lib.sh's.",
            'source "${E30_REPO:-$HOME/run/vulkan-demo-v7rc1}/docs/cluster_v6/scripts/e7_lib.sh" '
            '|| { echo "ABORT: no e7_lib.sh"; exit 2; }',
            f"e7_job_begin {name} {line} {' '.join(job_cases(items))}",
        ]
        for item in items:
            body += emit_item(item, model)
        body.append("e7_job_end")
        path = directory / f"e7_{name}.sbatch"
        path.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 batches: job list, node-hour budget and job scripts")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", default=None, help="write points, jobs, totals and the model inputs as JSON")
    parser.add_argument("--emit-jobs", default=None, metavar="DIR", help="write one sbatch script per batch and line")
    parser.add_argument("--batch", type=int, action="append", default=None,
                        help="with --emit-jobs: only these batches (default all)")
    parser.add_argument("--model-json", default=None, metavar="FILE",
                        help="replace model fields (a JSON object, e.g. e7_refit.py --out-model); default the plan's")
    arguments = parser.parse_args()
    model = load_model(arguments.model_json)
    points = campaign_points()
    jobs = build_jobs(points, model)

    lines = ["| batch | line | family | case | K | particles | per card | trials | references | run (min) | "
             "reference set (min) | calibration (min) | traced run (min) | point total (h) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    job_rows, records = [], []
    totals: dict = {}
    for (batch, line), items in jobs.items():
        estimate = model.job_overhead + sum(item["seconds"] for item in items)
        totals[(batch, line)] = estimate
        point_count = sum(1 for item in items if item["kind"] == "point")
        content = ", ".join(sorted({(item["point"].family.split("+")[0] + (" (cross)" if item["kind"] == "cross" else "")
                                     if "point" in item else item["kind"]) for item in items}))
        job_rows.append(f"| {batch} | {line} | e7_{job_name(batch, line)} | {point_count} | {len(items)} | "
                        f"{estimate / 3600:.2f} | {content} |")
        for item in items:
            if "point" in item:
                point = item["point"]
                seconds = point_seconds(point, model)
                reference = " + ".join({"same": "K=1 of the case", "pairs": "K=2 pairs"}.get(kind, f"K=1 {kind}")
                                       for kind in point.references)
                lines.append(f"| {batch} | {line} | {'cross ' if point.cross else ''}{point.family} | "
                             f"{point.case} | {point.slabs} | {PARTICLES[point.case] / 1e6:.1f}M | "
                             f"{point.per_card / 1e6:.2f}M | {point.trials} | {reference} | "
                             f"{seconds['run'] / 60:.1f} | {seconds['reference'] / 60:.1f} | "
                             f"{seconds['calibration'] / 60:.1f} | {seconds['traced'] / 60:.1f} | "
                             f"{seconds['total'] / 3600:.2f} |")
                records.append({"batch": batch, "line": line, "job": job_name(batch, line), "family": point.family,
                                "case": point.case, "slabs": point.slabs, "particles": PARTICLES[point.case],
                                "per_card": point.per_card, "dimension": point.dimension, "trials": point.trials,
                                "traces": point.traces, "cross": point.cross, "references": point.references,
                                "window": list(model.window(point.case)),
                                "fps_estimate": 1.0 / model.step_time(point.case, point.slabs),
                                **{name: round(value, 1) for name, value in seconds.items()
                                   if not isinstance(value, dict)}})
            else:
                records.append({"batch": batch, "line": line, "job": job_name(batch, line), "kind": item["kind"],
                                "case": item["case"], "slabs": item.get("slabs", item.get("parties")),
                                "steps": item.get("steps"), "seconds": round(item["seconds"], 1),
                                **({"label": item["label"]} if "label" in item else {}),
                                **({"reference_kind": item["reference_kind"], "parties": item["parties"]}
                                   if item["kind"] == "precheck" else {})})
    batch_totals = {}
    for (batch, line), value in totals.items():
        batch_totals.setdefault(batch, {})[line] = value
    total = sum(totals.values())
    summary = [f"points: {len(points)} (line A {sum(p.line == 'A' for p in points)}, line B "
               f"{sum(p.line == 'B' for p in points)}); 5 trials: {sum(p.trials == 5 for p in points)} points "
               f"(<= {LIGHT_PER_CARD / 1e6:.1f}M per card), 3 trials: {sum(p.trials == 3 for p in points)}; "
               "2 traced runs per point; timing runs without the seam check",
               "batches (node-h; wall = the longer line): " + "; ".join(
                   f"{batch}: A {values['A'] / 3600:.1f} + B {values['B'] / 3600:.1f} = "
                   f"{(values['A'] + values['B']) / 3600:.1f} (wall {max(values.values()) / 3600:.1f} h)"
                   for batch, values in sorted(batch_totals.items())),
               f"total {total / 3600:.1f} node-h = {8 * total / 3600:.0f} GPU-h (queue not included)",
               "", "| batch | line | job | points | items | estimate (h) | content |", "|---|---|---|---|---|---|---|"] + job_rows
    print("\n".join(summary + [""] + (lines if arguments.markdown else [])))
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({
            "model": dataclasses.asdict(model), "light_per_card": LIGHT_PER_CARD,
            "batches_hours": {str(batch): {line: value / 3600 for line, value in values.items()}
                              for batch, values in batch_totals.items()},
            "total_node_hours": total / 3600, "records": records}, indent=1), encoding="utf-8")
    if arguments.emit_jobs:
        for path in emit_jobs(jobs, model, pathlib.Path(arguments.emit_jobs), set(arguments.batch or (1, 2, 3, 4))):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
