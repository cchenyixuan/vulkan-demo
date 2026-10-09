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

Time model per run = construction + bootstrap + timed loop + post-run checks, every term measured in the E7
smoke (job 1679191, wqd10nbj04g2, v7-rc1):
  - construction (case load from the node-local .obj cache, partition, contexts, simulators): the first run of
    a configuration compiles its pipelines (~17 s per simulator: every slab's specialization constants are
    new), a repeat hits the driver's shader disk cache (~1.2 s per simulator); plus case load ~0.35 s per
    million particles. With the protocol only the calibration pilots and the first K = 1 reference set of a
    case are cold: pilot 1 runs the equal weights (the equal arm and the trace then hit the cache), the last
    pilot normally the calibrated cuts (the calibrated arm hits; when round 2 still moves the cuts, the
    calibrated arm's first run is cold too, and the step trace's extra device extension may change the key:
    both unmodelled, at most ~1 h over the campaign);
  - bootstrap (upload, ghost round, defrag, command recording): ~0.5 s per million particles + 2 s;
  - loop: steps / fps, fps from the K = 1 cost per particle-step (2-D 1.258 ns: 32M K = 1 at 24.8 fps; 3-D
    6.0 ns: cube K = 2 at 4.2 fps), an efficiency per K and a per-K floor for light loads (2-D 16M K = 8 ran
    110 / 146 fps, 2M per card);
  - post-run: defrag, readbacks and (default) the seam check: 2-D ~0.42 s per million particles + 8 s, 3-D
    (0.26 + 0.16 K) s per million (cube: 43 / 71 / 118 s at K = 2 / 4 / 8). Decided 2026-10-09: the timing
    runs use --no-seam-check (~0.1 s per million + 1 s per slab, estimate); the seam check runs on the traced
    run of every point.
The numbers are planning estimates; the measured inputs are listed above.

Usage: python docs/cluster_v6/scripts/e7_plan.py [--trials 3] [--traces 1] [--seam-check] [--markdown] [--json OUT]
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
    cost_2d: float = 1.258e-9       # s per particle-step, K = 1 (smoke: 32M K = 1 references, 24.8 fps mean)
    cost_3d: float = 6.0e-9         # s per particle-step, K = 1 (smoke: cube 76.2M K = 2 at 4.2 fps, eta ~0.96)
    floor_k1: float = 0.0014        # s per step, dispatch-bound K = 1 (1M aligned K = 1: 651-721 fps)
    floor: dict = dataclasses.field(default_factory=lambda: {2: 0.0018, 4: 0.0045, 8: 0.0080})
    efficiency: dict = dataclasses.field(default_factory=lambda: {2: 0.95, 4: 0.90, 8: 0.87})
    cold_per_simulator: float = 17.0      # s: pipeline compilation of a new configuration
    warm_per_simulator: float = 1.2       # s: driver shader disk cache hit
    load_per_million: float = 0.35        # s per million particles: .npy load + partition
    bootstrap_fixed: float = 2.0
    bootstrap_per_million: float = 0.5
    seam_check: bool = True
    calibration_rounds: int = 2
    pilot_steps: int = 500          # 200 warmup + 300 measured
    run_overhead: float = 10.0      # s per run: process start, parse, sync, telemetry window
    job_overhead: float = 600.0     # s per job: prelude, bring-up env stage, cache stage-in (~120 MB/s)

    def steps(self, point: Point) -> tuple[int, int]:
        return (1500, 500) if point.dimension == 3 else (3000, 1000)

    def step_time(self, case: str, slabs: int) -> float:
        cost = self.cost_3d if case.startswith("cavity3d") else self.cost_2d
        per_card = PARTICLES[case] / slabs
        if slabs == 1:
            return max(self.floor_k1, cost * per_card)
        return max(self.floor[slabs], cost * per_card / self.efficiency[slabs])

    def setup(self, case: str, slabs: int, cold: bool) -> float:
        """Construction + bootstrap + post-run checks of one run (a K = 1 reference set is one process per
        card, in parallel: one simulator's construction)."""
        millions = PARTICLES[case] / 1e6
        construction = ((self.cold_per_simulator if cold else self.warm_per_simulator) * slabs
                        + self.load_per_million * millions)
        bootstrap = self.bootstrap_fixed + self.bootstrap_per_million * millions
        if not self.seam_check:
            post = 0.1 * millions + 1.0 * slabs
        elif case.startswith("cavity3d"):
            post = (0.26 + 0.16 * slabs) * millions
        else:
            post = 0.42 * millions + 8.0
        return construction + bootstrap + post + self.run_overhead


def point_seconds(point: Point, model: Model, trials: int, traces: int) -> dict:
    steps, _ = model.steps(point)
    loop = steps * model.step_time(point.case, point.slabs)
    run_warm = model.setup(point.case, point.slabs, cold=False) + loop
    references_first = references_later = 0.0
    for kind in point.references:
        if kind == "pairs":                                   # 4 simultaneous K = 2 pairs of the case
            loop_reference = steps * model.step_time(point.case, 2)
            references_first += model.setup(point.case, 2, cold=True) + loop_reference
            references_later += model.setup(point.case, 2, cold=False) + loop_reference
        else:
            reference_case = point.case if kind == "same" else kind
            loop_reference = steps * model.step_time(reference_case, 1)
            references_first += model.setup(reference_case, 1, cold=True) + loop_reference
            references_later += model.setup(reference_case, 1, cold=False) + loop_reference
    pilot = (model.setup(point.case, point.slabs, cold=True)
             + model.pilot_steps * model.step_time(point.case, point.slabs))
    calibration = model.calibration_rounds * pilot + 20.0
    trials_total = references_first + (trials - 1) * references_later + trials * 2 * run_warm
    traced = dataclasses.replace(model, seam_check=True).setup(point.case, point.slabs, cold=False) + loop
    total = calibration + trials_total + traces * traced
    return {"run": run_warm, "reference": references_later, "reference_first": references_first,
            "calibration": calibration, "traced": traced, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 job list and node-hour budget")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--traces", type=int, default=1, help="step-traced runs per point")
    parser.add_argument("--seam-check", action="store_true",
                        help="timing runs with the post-run seam check (decided: off; the traced run keeps it)")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", default=None, help="write points, totals and the model inputs as JSON")
    arguments = parser.parse_args()
    model = Model(seam_check=arguments.seam_check)
    records = []
    points = campaign_points()
    totals: dict = {"A": 0.0, "B": 0.0}
    by_family: dict = {}
    lines = ["| node | family | case | K | particles | reference per trial | run (min) | reference set (min) | "
             "calibration (min) | point total (h) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for point in sorted(points, key=lambda item: (item.node, item.family, item.case, item.slabs)):
        seconds = point_seconds(point, model, arguments.trials, arguments.traces)
        totals[point.node] += seconds["total"]
        family = point.family.split("+")[0]
        by_family[(point.node, family)] = by_family.get((point.node, family), 0.0) + seconds["total"]
        reference = " + ".join({"same": "K=1 of the case", "pairs": "4 x K=2 pairs"}.get(kind, f"K=1 {kind}")
                               for kind in point.references)
        lines.append(f"| {point.node} | {point.family} | {point.case} | {point.slabs} | "
                     f"{PARTICLES[point.case] / 1e6:.1f}M | {reference} | {seconds['run'] / 60:.1f} | "
                     f"{seconds['reference'] / 60:.1f} | {seconds['calibration'] / 60:.1f} | "
                     f"{seconds['total'] / 3600:.2f} |")
        records.append({"node": point.node, "family": point.family, "case": point.case, "slabs": point.slabs,
                        "particles": PARTICLES[point.case], "dimension": point.dimension,
                        "references": point.references, "window": list(model.steps(point)),
                        "fps_estimate": 1.0 / model.step_time(point.case, point.slabs),
                        **{name: round(value, 1) for name, value in seconds.items()}})
    for node in totals:
        totals[node] += 2 * model.job_overhead         # two jobs per node
    summary = [f"seam check on timing runs: {model.seam_check}; trials {arguments.trials}; traced runs per point "
               f"{arguments.traces}",
               f"points: node A {sum(p.node == 'A' for p in points)}, node B {sum(p.node == 'B' for p in points)}",
               f"node A {totals['A'] / 3600:.1f} h, node B {totals['B'] / 3600:.1f} h, total "
               f"{(totals['A'] + totals['B']) / 3600:.1f} node-h = {8 * (totals['A'] + totals['B']) / 3600:.0f} GPU-h",
               "by family (h): " + ", ".join(f"{node}/{family} {value / 3600:.2f}"
                                            for (node, family), value in sorted(by_family.items()))]
    print("\n".join(summary + [""] + (lines if arguments.markdown else [])))
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({
            "model": dataclasses.asdict(model), "trials": arguments.trials, "traces": arguments.traces,
            "totals_hours": {node: value / 3600 for node, value in totals.items()},
            "by_family_hours": {f"{node}/{family}": value / 3600 for (node, family), value in by_family.items()},
            "points": records}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
