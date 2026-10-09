"""
e7_plan.py — E7 full campaign on N56 (hp_5090, whole nodes): the job list and its node-hour budget.

The job list follows the user's E7 rules (2026-10-09) and the E7 notes of E29 / E31 / E32 / E30:
  - cases/aligned/ only, simple walls (no *_adami case);
  - every point (case, K >= 2): one weight calibration on its node right before its trials
    (--weights auto --calibrate-only --weights-file, explicit --device-map), then TRIALS trials, each one
    K = 1 reference set on every participating card at the same time (same window as the K run), the
    equal-weight arm and the calibrated arm (--weights-file + --device-map), configuration order rotated
    over the trials (R E C / E C R / C R E); one step-traced run per point (E29 fields: t_tr, T_B, phase times);
  - 2-D strong: n2000 .. n8000 plus n9000 (K = 1 fits) and n11320 (128M: reference = 4 simultaneous K = 2
    pairs, eta x K/2, reported separately);
  - node A runs K <= 4 (GPUs 0-3, whole node), node B runs K = 8, in parallel;
  - windows: 2-D 3000 steps (warmup 1000), 3-D 1500 steps (warmup 500), as probe34-37.

Time model (per run): setup + steps x step time, step time from the K = 1 cost per particle-step of the
dimension (v7 on N56, from the smoke when given with --smoke, else the defaults below) divided by an
efficiency, with a floor per K for transport / dispatch-bound small loads; setup = a + b x million
particles (cached .obj). Calibration = rounds x (setup + warmup + steps of the pilot) at the K run's speed.
The numbers are planning estimates, printed with their inputs; the measured smoke rates replace the defaults.

Usage: python docs/cluster_v6/scripts/e7_plan.py [--smoke RESULTS.jsonl] [--trials 3] [--traces 1] [--markdown]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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


@dataclasses.dataclass
class Point:
    family: str
    case: str
    slabs: int
    references: list        # per trial: "same" = K = 1 of the case; "pairs" = 4 K = 2 pairs; else a k1 case

    @property
    def node(self) -> str:
        return "B" if self.slabs == 8 else "A"

    @property
    def dimension(self) -> int:
        return 3 if self.case.startswith("cavity3d") else 2


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
    # F6 cube at K = 8 (K = 1 reference fits: 24.5 GiB)
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


@dataclasses.dataclass
class Model:
    cost_2d: float = 1.30e-9        # s per particle-step, K = 1, v7 on a 5090 (E39 local 2-D 1M/16M scaled)
    cost_3d: float = 6.7e-9         # s per particle-step, K = 1, 3-D
    floor_k1: float = 0.0012        # s per step: dispatch-bound small K = 1 runs (2-D 1M: ~720 fps)
    floor: dict = dataclasses.field(default_factory=lambda: {2: 0.0015, 4: 0.0035, 8: 0.0075})
    efficiency: dict = dataclasses.field(default_factory=lambda: {2: 0.95, 4: 0.92, 8: 0.90})
    setup_fixed: float = 8.0        # s per run: imports, contexts, pipelines, bootstrap, post-run checks
    setup_per_million: float = 0.6  # s per million particles with the .obj cache (load, partition, upload)
    calibration_rounds: int = 2
    pilot_steps: int = 500          # 200 warmup + 300 measured
    run_overhead: float = 10.0      # s per run: parse, sync, telemetry window
    job_overhead: float = 600.0     # s per job: prelude, bring-up env stage, cache stage-in

    def steps(self, point: Point) -> tuple[int, int]:
        return (1500, 500) if point.dimension == 3 else (3000, 1000)

    def step_time(self, case: str, slabs: int) -> float:
        cost = self.cost_3d if case.startswith("cavity3d") else self.cost_2d
        per_card = PARTICLES[case] / slabs
        if slabs == 1:
            return max(self.floor_k1, cost * per_card)
        return max(self.floor[slabs], cost * per_card / self.efficiency[slabs])

    def setup(self, case: str) -> float:
        return self.setup_fixed + self.setup_per_million * PARTICLES[case] / 1e6


def point_seconds(point: Point, model: Model, trials: int, traces: int) -> dict:
    steps, _ = model.steps(point)
    run = model.setup(point.case) + steps * model.step_time(point.case, point.slabs) + model.run_overhead
    reference = 0.0
    for kind in point.references:
        if kind == "pairs":
            reference += model.setup(point.case) + steps * model.step_time(point.case, 2) + model.run_overhead
        else:
            reference_case = point.case if kind == "same" else kind
            reference += model.setup(reference_case) + steps * model.step_time(reference_case, 1) + model.run_overhead
    calibration = model.calibration_rounds * (model.setup(point.case)
                                              + model.pilot_steps * model.step_time(point.case, point.slabs)) + 20.0
    total = calibration + trials * (reference + 2 * run) + traces * run
    return {"run": run, "reference": reference, "calibration": calibration, "total": total}


def apply_smoke(model: Model, path: pathlib.Path) -> list[str]:
    """Replace the K = 1 costs and the setup slope by the smoke's measurements where it has them."""
    notes = []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    k1 = [row for row in rows if row["label"].startswith("k1_w32_") and row.get("steady_fps")]
    if k1:
        fps = sum(row["steady_fps"] for row in k1) / len(k1)
        model.cost_2d = 1.0 / (fps * PARTICLES["cavity2d_n5640"])
        notes.append(f"cost_2d from {len(k1)} K=1 n5640 references: {fps:.2f} fps -> {model.cost_2d * 1e9:.3f} ns")
    builds = [(row.get("loaded_particles"), row.get("build_seconds")) for row in rows
              if row.get("build_seconds") and row.get("loaded_particles")]
    if len(builds) >= 3:
        x = [count / 1e6 for count, _ in builds]
        y = [seconds for _, seconds in builds]
        mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / max(sum((a - mean_x) ** 2 for a in x), 1e-9)
        model.setup_per_million = max(slope, 0.05)
        model.setup_fixed = max(mean_y - slope * mean_x, 2.0)
        notes.append(f"setup from {len(builds)} runs: {model.setup_fixed:.1f} s + {model.setup_per_million:.3f} s/M")
    for row in rows:
        if row["label"] in ("k2_cube", "k4_cube", "k8_cube") and row.get("steady_fps"):
            slabs = int(row["label"][1])
            if slabs == 2:
                model.cost_3d = 2.0 / (row["steady_fps"] * PARTICLES["cavity3d_n416"]) * model.efficiency[2]
                notes.append(f"cost_3d from the K=2 cube: {row['steady_fps']} fps -> {model.cost_3d * 1e9:.2f} ns")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 job list and node-hour budget")
    parser.add_argument("--smoke", default=None, help="results.jsonl of the smoke job (measured rates)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--traces", type=int, default=1, help="step-traced runs per point")
    parser.add_argument("--markdown", action="store_true")
    arguments = parser.parse_args()
    model = Model()
    notes = apply_smoke(model, pathlib.Path(arguments.smoke)) if arguments.smoke else ["defaults (no smoke data)"]
    points = campaign_points()
    totals: dict = {"A": 0.0, "B": 0.0}
    lines = ["| node | family | case | K | particles | reference | run (min) | reference set (min) | "
             "calibration (min) | point total (h) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for point in sorted(points, key=lambda item: (item.node, item.family, item.case, item.slabs)):
        seconds = point_seconds(point, model, arguments.trials, arguments.traces)
        totals[point.node] += seconds["total"]
        reference = " + ".join({"same": "K=1 of the case", "pairs": "4 x K=2 pairs"}.get(kind, f"K=1 {kind}")
                               for kind in point.references)
        lines.append(f"| {point.node} | {point.family} | {point.case} | {point.slabs} | {PARTICLES[point.case] / 1e6:.1f}M | "
                     f"{reference} | {seconds['run'] / 60:.1f} | {seconds['reference'] / 60:.1f} | "
                     f"{seconds['calibration'] / 60:.1f} | {seconds['total'] / 3600:.2f} |")
    for node in totals:
        totals[node] += 2 * model.job_overhead         # two jobs per node
    summary = [f"model: {notes}",
               f"points: node A {sum(p.node == 'A' for p in points)}, node B {sum(p.node == 'B' for p in points)}; "
               f"trials {arguments.trials}, traced runs per point {arguments.traces}",
               f"node A {totals['A'] / 3600:.1f} h, node B {totals['B'] / 3600:.1f} h, "
               f"total {(totals['A'] + totals['B']) / 3600:.1f} node-h = {8 * (totals['A'] + totals['B']) / 3600:.0f} GPU-h"]
    print("\n".join(summary + [""] + (lines if arguments.markdown else [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
