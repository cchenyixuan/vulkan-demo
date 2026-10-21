"""
e8_plan.py — E8 phase 0 item 5 (and the phase-1 budget): wall-clock estimates of the 3-D scaling matrix on N56.

Protocol = E7's (e7_lib.sh e7_point): per point one weight calibration (2 pilots of 200 + 300 steps at K, cold
build each), then TRIALS trials, each = the reference set (K simultaneous K = 1 runs of the reference case on the
participating cards behind one barrier: wall = one K = 1 run, the first trial's cold) + one calibrated K run
(warm), 1500 steps (warmup 500), timed runs without the seam check; then the traced runs (1500 steps, seam
check, step trace, cold).

Matrix (user, 2026-10-22; phase 2, estimate only):
  strong, load sweep   cavity3d_n192 / n244 / n312 / n392 (8.00 / 16.0 / 32.8 / 64.0M) at K = 8, reference = K = 1 of the case
  strong, K sweep      cavity3d_n392 at K = 2, 4 (K = 8 from the first line), reference = K = 1 of n392
  weak                 cavity3d_n192_k2 / _k4 / _k8 (2 / 4 / 8 cubes of 8M along x) at K = 2 / 4 / 8, reference =
                       K = 1 of cavity3d_n192 on the participating cards
  3 trials per point; traces (phase timing) of n192 K8, n244 K8, n392 K8, n192_k8 K8.

Time model (e7_plan.py's, with the batch-1 + batch-3 refit of its setup constants, docs/cluster_v6/data/e7/batch3/
refit_b13_model.json):
  K = 1 step          cost x N                        (cost = --cost-ns, phase 0: 6.69 ns per particle-step)
  K > 1 step          cost x N / K / eta_K + exposed  (eta_K = --eta-strong / --eta-weak: E7 batch 3, 3-D, 4.6-35M per
                      card: strong K2 0.95 / K4 0.90 / K8 0.875, weak 0.985)
  exposed             max(0, t_chain - T_B): the transport chain of one link, t_chain = t0 + B_link / beta (E29 2-D
                      fit: t0 = 122 us, beta = 3.76 GB/s; --t-chain-fixed-us / --host-bandwidth-gbs; phase 1 replaces
                      them), B_link = the host copy per link and step: 2 replica layers x 32 B x 4 n^2 particles per
                      ghost column + the list / count segments (fit of the E7 3-D traces: ~4,400 n^1.044 B), against
                      T_B = the receiver's phase B = compute x B share(N): the interior variants of correction /
                      density / force outside bands 2 / 2 / 3 on N own columns, weighted by the kernels' shares of a
                      3-D step (E5.1), scaled to the measured 0.80 of the 13-column slabs of 3d_n416 K8 and capped at
                      0.88 (KERNEL_SHARES, B_SHARE_MEASURED_13); the own columns = the equal-weight cuts (e8_vram.chain)
  setup per run       construction (cold 2.2 s or warm 1.46 s per simulator + 0.386 s per million particles read
                      from the .obj cache) + bootstrap (0.32 s + 0.547 s per million) + post-run checks (0.042 s per
                      million without the seam check; 3-D with it 0.093 x K s per million) + 2.7 s
  calibration         2 x (cold setup at K + 500 steps) - 10.4 s (the refit's fixed term)
  trace               cold run with the seam check + 15 s of CSV writing
  job                 165 s of prelude / bring-up / cache stage-in; every point's items run back to back on one
                      whole node (two lines as in E7: A = K <= 4 on cards 0-3, B = K = 8); node-hours = wall hours of
                      8-card jobs. The timeouts of a job script: 2 x estimate + 600 s (min 900 s), limit 2 x + 1 h.

Usage:
    python docs/cluster_v6/scripts/e8_plan.py [--cost-ns 6.69] [--t-chain-fixed-us 122] [--host-bandwidth-gbs 3.76]
        [--eta-strong 2:0.95,4:0.90,8:0.875] [--eta-weak 0.985] [--b-share 0.8] [--phase1] [--md OUT] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
import e8_vram  # noqa: E402

PARTICLES = {name: e8_vram.lattice(n, n_x)["total"] for name, (n, n_x) in e8_vram.CASES.items()}
EDGE = {name: n for name, (n, n_x) in e8_vram.CASES.items()}
STEPS, WARMUP = 1500, 500
TRIALS = 3
PILOT_STEPS, PILOT_ROUNDS = 500, 2
# refit_b13_model.json (E7 batches 1 + 3, 3-D heavy points included)
SETUP = {"cold_per_simulator": 2.1992, "warm_per_simulator": 1.4611, "load_per_million": 0.3864,
         "bootstrap_fixed": 0.3189, "bootstrap_per_million": 0.5470, "post_per_million": 0.0416,
         "seam_3d_per_million_per_slab": 0.0929, "calibration_fixed": -10.392, "run_overhead": 2.720,
         "trace_write": 15.0, "job_overhead": 165.5}
# Phase B of a K-slab step as a share of the slab's compute: the interior variants of correction (31.4 % of a 3-D
# step, E5.1), density (31.8 %) outside band c = 2 and of force (35.4 %) outside band f = 3 on a slab of N own
# columns, B_model(N) = 0.314 (N - 4) / N + 0.318 (N - 4) / N + 0.354 (N - 6) / N, scaled so that the 13-column
# slabs of 3d_n416 K8 give the measured 0.80 (E7 batch 3: T_B 37.5 ms of a 47 ms step), capped at 0.88 (the 2-D
# 250-column slabs of 2d_n8000 K8: T_B 9.28 of 10.8 ms)
KERNEL_SHARES = (0.314, 0.318, 0.354)
BAND_COLUMNS = (2, 2, 3)
B_SHARE_MEASURED_13 = 0.80
B_SHARE_CAP = 0.88

MATRIX = [
    # (group, case, K, reference case, traced)
    ("strong, load sweep", "cavity3d_n152", 8, "cavity3d_n152", True),
    ("strong, load sweep", "cavity3d_n192", 8, "cavity3d_n192", True),
    ("strong, load sweep", "cavity3d_n244", 8, "cavity3d_n244", True),
    ("strong, load sweep", "cavity3d_n312", 8, "cavity3d_n312", False),
    ("strong, load sweep", "cavity3d_n392", 8, "cavity3d_n392", True),
    ("strong, K sweep", "cavity3d_n392", 2, "cavity3d_n392", False),
    ("strong, K sweep", "cavity3d_n392", 4, "cavity3d_n392", False),
    ("weak", "cavity3d_n192_k2", 2, "cavity3d_n192", False),
    ("weak", "cavity3d_n192_k4", 4, "cavity3d_n192", False),
    ("weak", "cavity3d_n192_k8", 8, "cavity3d_n192", True),
]
# Phase 2 (user 2026-10-22): every point traced, 4M and 8M with 5 trials, the K = 1 traces of the five cubes, the
# correctness dumps of 4M and 8M (K = 1 twice + K = 8 once, 1500 steps, equal weights) with their comparisons, the
# v7_single comparison (3 interleaved pairs at 8M on one card), the partition record of every point (CPU).
MATRIX2 = [
    ("strong, load sweep", "cavity3d_n152", 8, "cavity3d_n152", True, 5),
    ("strong, load sweep", "cavity3d_n192", 8, "cavity3d_n192", True, 5),
    ("strong, load sweep", "cavity3d_n244", 8, "cavity3d_n244", True, 3),
    ("strong, load sweep", "cavity3d_n312", 8, "cavity3d_n312", True, 3),
    ("strong, load sweep", "cavity3d_n392", 8, "cavity3d_n392", True, 3),
    # e8_C (user 2026-10-22): the 128M cube, K = 1 does not fit one card -> reference "pairs" = four simultaneous K = 2
    # pairs of the case (E7 2-D 128M), eta = fps_K / (4 x mean pair fps); one K = 2 pair traced (trace_k2)
    ("strong, load sweep", "cavity3d_n496", 8, "pairs", True, 3),
    ("strong, K sweep", "cavity3d_n392", 2, "cavity3d_n392", True, 3),
    ("strong, K sweep", "cavity3d_n392", 4, "cavity3d_n392", True, 3),
    ("weak", "cavity3d_n192_k2", 2, "cavity3d_n192", True, 3),
    ("weak", "cavity3d_n192_k4", 4, "cavity3d_n192", True, 3),
    ("weak", "cavity3d_n192_k8", 8, "cavity3d_n192", True, 3),
]
SEQUENCE2 = {
    "B": [("point", "cavity3d_n152", 8), ("dump", "cavity3d_n152", 1), ("dump", "cavity3d_n152", 1), ("dump", "cavity3d_n152", 8),
          ("compare", "cavity3d_n152"), ("trace_k1", "cavity3d_n152"),
          ("point", "cavity3d_n192", 8), ("dump", "cavity3d_n192", 1), ("dump", "cavity3d_n192", 1), ("dump", "cavity3d_n192", 8),
          ("compare", "cavity3d_n192"), ("trace_k1", "cavity3d_n192"),
          ("point", "cavity3d_n244", 8), ("trace_k1", "cavity3d_n244"), ("point", "cavity3d_n312", 8), ("trace_k1", "cavity3d_n312"),
          ("point", "cavity3d_n392", 8), ("trace_k1", "cavity3d_n392"), ("point", "cavity3d_n192_k8", 8),
          ("single_ratio", "cavity3d_n192", 3)],
    "A": [("point", "cavity3d_n392", 2), ("point", "cavity3d_n392", 4), ("point", "cavity3d_n192_k2", 2), ("point", "cavity3d_n192_k4", 4)],
    "C": [("point", "cavity3d_n496", 8), ("trace_k2", "cavity3d_n496")],
}
PHASE1 = [
    # (label, case, K, steps, warmup, seam check, trace, kind)
    ("8m_k1", "cavity3d_n192", 1, 300, 100, False, False, "run"),
    ("8m_k2_trace", "cavity3d_n192", 2, 300, 100, True, True, "run"),
    ("dump_8m_k1", "cavity3d_n192", 1, 300, 0, False, False, "dump"),
    ("dump_8m_k2", "cavity3d_n192", 2, 300, 0, False, False, "dump"),
    ("s48_k1", "cavity3d_n48", 1, 300, 100, False, False, "run"),
    ("s48_k2", "cavity3d_n48", 2, 300, 100, True, False, "run"),
    ("dump_s48_k1", "cavity3d_n48", 1, 300, 0, False, False, "dump"),
    ("dump_s48_k2", "cavity3d_n48", 2, 300, 0, False, False, "dump"),
    ("weak2_k2", "cavity3d_n192_k2", 2, 300, 100, True, False, "run"),
    ("64m_k2", "cavity3d_n392", 2, 300, 100, True, False, "run"),
]


class Model:
    def __init__(self, cost_ns: float, t_chain_fixed_us: float, host_bandwidth_gbs: float, eta_strong: dict,
                 eta_weak: float, b_share: float, b_link_override: dict, cost_table: dict = None,
                 b_link_linear: float = 4400.0, b_link_power: float = 1.044):
        self.cost = cost_ns * 1e-9
        # phase 1 (--cost-table): the K = 1 cost per particle-step depends on the particles per card (a bigger
        # slab is cheaper per particle, E7 item 0): piecewise linear in log10(N per card) between the measured
        # points, flat outside them; None = the constant --cost-ns
        self.cost_table = sorted((math.log10(n), c * 1e-9) for n, c in cost_table.items()) if cost_table else None
        self.b_link_linear, self.b_link_power = b_link_linear, b_link_power
        self.t0 = t_chain_fixed_us * 1e-6
        self.beta = host_bandwidth_gbs * 1e9
        self.eta_strong = eta_strong
        self.eta_weak = eta_weak
        self.b_share_scale = b_share
        self.b_link_override = b_link_override

    @staticmethod
    def per_k(table: dict, slabs: int) -> float:
        if slabs in table:
            return table[slabs]
        lower = max((key for key in table if key < slabs), default=None)
        upper = min((key for key in table if key > slabs), default=None)
        if lower is None:
            return table[upper]
        if upper is None:
            return table[lower]
        weight = (slabs - lower) / (upper - lower)
        return table[lower] + weight * (table[upper] - table[lower])

    def cost_per_particle(self, per_card: float) -> float:
        if not self.cost_table:
            return self.cost
        x = math.log10(per_card)
        points = self.cost_table
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]
        for (x0, c0), (x1, c1) in zip(points, points[1:]):
            if x0 <= x <= x1:
                return c0 + (c1 - c0) * (x - x0) / (x1 - x0)
        return self.cost

    def b_link(self, case: str) -> float:
        """Host copy bytes per link and step: 2 layers x 32 B x 4 n^2 + the list / count segments (phase 0: the E7
        3-D traces' fit 4,400 n^1.044; phase 1 (--b-link-linear 4500 --b-link-power 1): 256 n^2 + 4,500 n fits the
        measured 806,912 / 10,283,280 / 41,126,502 B of n = 48 / 192 / 392 within 2 %); --b-link CASE=BYTES overrides."""
        if case in self.b_link_override:
            return self.b_link_override[case]
        n = EDGE[case]
        return 2 * 32 * 4 * n * n + self.b_link_linear * n ** self.b_link_power

    def t_chain(self, case: str) -> float:
        return self.t0 + self.b_link(case) / self.beta

    @staticmethod
    def b_model(own_columns: float, peers: int = 2) -> float:
        return sum(share * max(0.0, own_columns - peers * band) / own_columns
                   for share, band in zip(KERNEL_SHARES, BAND_COLUMNS))

    def b_share(self, case: str, slabs: int) -> float:
        """Phase B share of the narrowest interior slab (end slabs have one band side)."""
        n, n_x = e8_vram.CASES[case]
        entries = e8_vram.chain(n, n_x, slabs)
        shares = [self.b_model(entry["own_columns"], entry["peers"]) for entry in entries]
        scale = B_SHARE_MEASURED_13 / self.b_model(13, 2)
        return min(B_SHARE_CAP, min(shares) * scale) * self.b_share_scale

    def interior_fraction(self, case: str, slabs: int) -> float:
        n, n_x = e8_vram.CASES[case]
        columns = [entry["own_columns"] for entry in e8_vram.chain(n, n_x, slabs)]
        return min(max(0.0, (own - 2 * BAND_COLUMNS[2]) / own) for own in (columns[1:-1] or columns))

    def step(self, case: str, slabs: int, weak: bool) -> dict:
        particles = PARTICLES[case]
        compute = self.cost_per_particle(particles / slabs) * particles / slabs
        if slabs == 1:
            return {"step": compute, "compute": compute, "eta": 1.0, "t_chain": 0.0, "T_B": 0.0, "exposed": 0.0,
                    "r": 0.0, "fps": 1.0 / compute}
        eta = self.eta_weak if weak else self.per_k(self.eta_strong, slabs)
        t_chain = self.t_chain(case)
        T_B = compute * self.b_share(case, slabs)
        exposed = max(0.0, t_chain - T_B)
        step = compute / eta + exposed
        return {"step": step, "compute": compute, "eta": eta, "t_chain": t_chain, "T_B": T_B, "exposed": exposed,
                "r": (t_chain / T_B) if T_B > 0 else 99.0, "fps": 1.0 / step,
                "eta_model": compute / step}

    def setup(self, case: str, slabs: int, cold: bool, seam_check: bool) -> float:
        millions = PARTICLES[case] / 1e6
        construction = (SETUP["cold_per_simulator"] if cold else SETUP["warm_per_simulator"]) * slabs \
            + SETUP["load_per_million"] * millions
        bootstrap = SETUP["bootstrap_fixed"] + SETUP["bootstrap_per_million"] * millions
        post = (SETUP["seam_3d_per_million_per_slab"] * slabs if seam_check else SETUP["post_per_million"]) * millions
        return construction + bootstrap + post + SETUP["run_overhead"]

    def run(self, case: str, slabs: int, steps: int, cold: bool, seam_check: bool, weak: bool) -> float:
        return self.setup(case, slabs, cold, seam_check) + steps * self.step(case, slabs, weak)["step"]


def point_estimate(model: Model, group: str, case: str, slabs: int, reference: str, traced: bool,
                   trials: int = TRIALS, partition_record: bool = False) -> dict:
    weak = group == "weak"
    pilot = model.setup(case, slabs, cold=True, seam_check=False) + PILOT_STEPS * model.step(case, slabs, weak)["step"]
    calibration = PILOT_ROUNDS * pilot + SETUP["calibration_fixed"]
    record = (SETUP["load_per_million"] * PARTICLES[case] / 1e6 + 0.5 * PARTICLES[case] / 1e6 + 10.0) if partition_record else 0.0
    # reference "pairs" (e8_C, 128M): four simultaneous K = 2 pairs of the case itself, one pair's run time
    reference_case, reference_slabs = (case, 2) if reference == "pairs" else (reference, 1)
    reference_first = model.run(reference_case, reference_slabs, STEPS, cold=True, seam_check=False, weak=False)
    reference_later = model.run(reference_case, reference_slabs, STEPS, cold=False, seam_check=False, weak=False)
    chain_run = model.run(case, slabs, STEPS, cold=False, seam_check=False, weak=weak)
    trace = model.run(case, slabs, STEPS, cold=True, seam_check=True, weak=weak) + SETUP["trace_write"] if traced else 0.0
    trials_total = reference_first + (trials - 1) * reference_later + trials * chain_run
    total = calibration + record + trials_total + trace
    return {"group": group, "case": case, "slabs": slabs, "reference": reference, "traced": traced, "trials": trials,
            "line": "B" if slabs == 8 else "A", "particles": PARTICLES[case], "per_card": PARTICLES[case] / slabs,
            "calibration": calibration, "partition_record": record, "reference_first": reference_first,
            "reference_later": reference_later, "chain_run": chain_run, "trace": trace, "total": total,
            "reference_slabs": reference_slabs,
            "step": model.step(case, slabs, weak), "reference_step": model.step(reference_case, reference_slabs, False)}


def phase2_lines(model: Model) -> tuple[list[str], dict]:
    """The phase-2 budget: every item of both lines in the job order (SEQUENCE2), with its estimate."""
    by_key = {(case, slabs): (group, reference, traced, trials) for group, case, slabs, reference, traced, trials in MATRIX2}
    lines = ["## Phase 2 (jobs e8_A / e8_B / e8_C, the items in the job order)", "",
             "| line | item | case | K | trials | estimate | note |", "|---|---|---|---|---|---|---|"]
    totals = {line: SETUP["job_overhead"] for line in ("A", "B", "C")}
    records = {"A": [], "B": [], "C": []}
    for line in ("B", "A", "C"):
        for spec in SEQUENCE2[line]:
            kind = spec[0]
            case = spec[1]
            millions = PARTICLES[case] / 1e6
            if kind == "point":
                group, reference, traced, trials = by_key[(case, spec[2])]
                estimate = point_estimate(model, group, case, spec[2], reference, traced, trials, partition_record=True)
                seconds = estimate["total"]
                note = (f"calibration {seconds_text(estimate['calibration'])} + record {estimate['partition_record']:.0f} s, "
                        f"reference set ({'4 K = 2 pairs' if reference == 'pairs' else 'K = 1'}) {seconds_text(estimate['reference_first'])} / {seconds_text(estimate['reference_later'])} x {trials - 1}, "
                        f"chain run {seconds_text(estimate['chain_run'])} x {trials}, trace {seconds_text(estimate['trace'])}")
                records[line].append({"kind": "point", "case": case, "slabs": spec[2], "trials": trials, "seconds": seconds, "detail": estimate})
            elif kind == "dump":
                seconds = model.run(case, spec[2], STEPS, cold=True, seam_check=False, weak=False) + 0.3 * millions + 20.0
                note = f"dump_state K = {spec[2]} to step {STEPS}, equal weights, npz {0.13 * millions:.1f} GB"
                records[line].append({"kind": "dump", "case": case, "slabs": spec[2], "seconds": seconds})
            elif kind == "compare":
                seconds = 60.0 + 20.0 * millions
                note = "K = 8 vs K = 1 a + the noise floor K = 1 a vs b (fluid, by seam distance, per slab)"
                records[line].append({"kind": "compare", "case": case, "seconds": seconds})
            elif kind == "trace_k1":
                seconds = model.run(case, 1, STEPS, cold=True, seam_check=True, weak=False) + SETUP["trace_write"]
                note = "K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases)"
                records[line].append({"kind": "trace_k1", "case": case, "seconds": seconds})
            elif kind == "trace_k2":
                seconds = model.run(case, 2, STEPS, cold=True, seam_check=True, weak=False) + SETUP["trace_write"]
                note = "K = 2 pair (cards 0, 1, equal weights) step trace, every kernel's tick (the ideal-chain row = a quarter of its phases)"
                records[line].append({"kind": "trace_k2", "case": case, "seconds": seconds})
            elif kind == "single_ratio":
                pairs = spec[2]
                seconds = pairs * 2 * model.run(case, 1, STEPS, cold=False, seam_check=False, weak=False) + pairs * 30.0
                note = f"{pairs} interleaved pairs v7_single / chain K = 1 on card 0, {STEPS} steps (warmup {WARMUP}), no defrag"
                records[line].append({"kind": "single_ratio", "case": case, "pairs": pairs, "seconds": seconds})
            else:
                raise ValueError(kind)
            totals[line] += seconds
            lines.append(f"| {line} | {kind} | {case} | {spec[2] if len(spec) > 2 and isinstance(spec[2], int) and kind != 'single_ratio' else ('1' if kind in ('trace_k1', 'single_ratio') else ('2' if kind == 'trace_k2' else '-'))} | "
                         f"{by_key[(case, spec[2])][3] if kind == 'point' else '-'} | **{seconds_text(seconds)}** | {note} |")
    lines += ["", f"Job overhead {SETUP['job_overhead']:.0f} s per job. **"
              + ", ".join(f"line {line} {totals[line] / 3600:.2f} h" for line in ("A", "B", "C"))
              + f", total {sum(totals.values()) / 3600:.2f} node-h**; time limits (2 x + 1 h, rounded up): "
              + ", ".join(f"{line} {math.ceil(2 * totals[line] / 3600 + 1)} h" for line in ("A", "B", "C")) + "."]
    return lines, {"totals_hours": {line: value / 3600 for line, value in totals.items()}, "items": records}


def seconds_text(value: float) -> str:
    return f"{value / 60:.1f} min" if value >= 60 else f"{value:.0f} s"


def seconds(value: float) -> str:
    return f"{value / 60:.1f} min" if value >= 60 else f"{value:.0f} s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-ns", type=float, default=6.69, help="K = 1 ns per particle-step (phase 0: 6.69)")
    parser.add_argument("--cost-table", default=None, metavar="N:NS,...",
                        help="measured ns per particle-step by particles per card, interpolated in log N (phase 1: "
                             "175616:6.83,8000000:6.28,32000000:5.91)")
    parser.add_argument("--b-link-linear", type=float, default=4400.0, help="B_link = 256 n^2 + linear x n^power")
    parser.add_argument("--b-link-power", type=float, default=1.044)
    parser.add_argument("--t-chain-fixed-us", type=float, default=122.0)
    parser.add_argument("--host-bandwidth-gbs", type=float, default=3.76)
    parser.add_argument("--eta-strong", default="2:0.95,4:0.90,8:0.875")
    parser.add_argument("--eta-weak", type=float, default=0.985)
    parser.add_argument("--b-share", type=float, default=1.0, help="scale of the phase-B share model (1 = as fitted)")
    parser.add_argument("--b-link", action="append", default=[], metavar="CASE=BYTES",
                        help="measured host copy bytes per link and step of a case (phase 1)")
    parser.add_argument("--phase1", action="store_true", help="also the phase-1 job's budget")
    parser.add_argument("--phase2", action="store_true", help="the phase-2 jobs' budget (MATRIX2 / SEQUENCE2)")
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    eta_strong = {int(key): float(value) for key, value in (item.split(":") for item in arguments.eta_strong.split(","))}
    overrides = {}
    for item in arguments.b_link:
        case, value = item.split("=")
        overrides[case] = float(value)
    cost_table = ({float(key): float(value) for key, value in (item.split(":") for item in arguments.cost_table.split(","))}
                  if arguments.cost_table else None)
    model = Model(arguments.cost_ns, arguments.t_chain_fixed_us, arguments.host_bandwidth_gbs, eta_strong,
                  arguments.eta_weak, arguments.b_share, overrides, cost_table, arguments.b_link_linear, arguments.b_link_power)
    lines = [f"Model: cost {arguments.cost_ns} ns per particle-step (K = 1){' / table ' + arguments.cost_table if arguments.cost_table else ''}, "
             f"B_link = 256 n^2 + {arguments.b_link_linear:g} n^{arguments.b_link_power:g}, t_chain = {arguments.t_chain_fixed_us:.0f} us + "
             f"B_link / {arguments.host_bandwidth_gbs} GB/s, eta_strong {eta_strong}, eta_weak {arguments.eta_weak}, "
             f"B share {arguments.b_share}; windows {STEPS}/{WARMUP}, {TRIALS} trials, {PILOT_ROUNDS} pilots x {PILOT_STEPS} steps.", ""]
    lines += ["## Step model per point", "",
              "| point | particles | per card | own columns (interior) | step K = 1 ref (ms) | compute / card (ms) | "
              "eta assumed | B_link (MB) | t_chain (ms) | B share | T_B (ms) | r = t_chain / T_B | exposed (ms) | step K (ms) | "
              "fps K | eta model |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    points = []
    for group, case, slabs, reference, traced in MATRIX:
        estimate = point_estimate(model, group, case, slabs, reference, traced)
        points.append(estimate)
        step = estimate["step"]
        n, n_x = e8_vram.CASES[case]
        columns = [entry["own_columns"] for entry in e8_vram.chain(n, n_x, slabs)]
        lines.append(f"| {case} K{slabs} ({group}) | {PARTICLES[case]:,} | {PARTICLES[case] / slabs / 1e6:.2f}M ({model.cost_per_particle(PARTICLES[case] / slabs) * 1e9:.2f} ns) | "
                     f"{'/'.join(str(column) for column in columns)} ({model.interior_fraction(case, slabs):.2f}) | "
                     f"{estimate['reference_step']['step'] * 1e3:.1f} | {step['compute'] * 1e3:.2f} | {step['eta']:.3f} | "
                     f"{model.b_link(case) / 1e6:.1f} | {step['t_chain'] * 1e3:.2f} | {model.b_share(case, slabs):.2f} | {step['T_B'] * 1e3:.2f} | "
                     f"{step['r']:.2f} | {step['exposed'] * 1e3:.2f} | {step['step'] * 1e3:.2f} | {step['fps']:.1f} | "
                     f"{step['eta_model']:.3f} |")
    lines += ["", "## Wall clock per point (E7 protocol)", "",
              "| point | line | calibration (2 pilots) | reference set, trial 1 (cold) | reference set, trials 2-3 | "
              "chain run (per trial) | trace | point total | timeouts cal / run / trace (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    totals = {"A": 0.0, "B": 0.0}
    for estimate in points:
        totals[estimate["line"]] += estimate["total"]

        def timeout(value):
            return int(max(900, math.ceil((2.0 * value + 600) / 60.0) * 60))
        lines.append(f"| {estimate['case']} K{estimate['slabs']} | {estimate['line']} | {seconds(estimate['calibration'])} | "
                     f"{seconds(estimate['reference_first'])} | {seconds(estimate['reference_later'])} x 2 | "
                     f"{seconds(estimate['chain_run'])} x {TRIALS} | "
                     f"{seconds(estimate['trace']) if estimate['traced'] else '-'} | **{seconds(estimate['total'])}** | "
                     f"{timeout(estimate['calibration'])} / {timeout(max(estimate['reference_first'], estimate['chain_run']))} / "
                     f"{timeout(estimate['trace']) if estimate['traced'] else '-'} |")
    job_a = totals["A"] + SETUP["job_overhead"]
    job_b = totals["B"] + SETUP["job_overhead"]
    lines += ["", f"Line A (K <= 4, cards 0-3 of a whole node): {job_a / 3600:.2f} h; line B (K = 8): {job_b / 3600:.2f} h; "
              f"both on whole nodes: **{(job_a + job_b) / 3600:.2f} node-h** (one node back to back: the same). "
              f"Time limits (2 x + 1 h): A {math.ceil(2 * job_a / 3600 + 1)} h, B {math.ceil(2 * job_b / 3600 + 1)} h.",
              f"Reference runs alone: {sum(e['reference_first'] + 2 * e['reference_later'] for e in points) / 3600:.2f} h "
              f"of the {(totals['A'] + totals['B']) / 3600:.2f} h (the 64M K = 1 references of the three n392 points are "
              f"{sum(e['reference_first'] + 2 * e['reference_later'] for e in points if e['case'] == 'cavity3d_n392') / 3600:.2f} h)."]
    phase2 = {}
    if arguments.phase2:
        phase2_text, phase2 = phase2_lines(model)
        lines += [""] + phase2_text
    phase1 = []
    if arguments.phase1:
        lines += ["", "## Phase-1 job (2 cards, 300 steps each)", "",
                  "| item | case | K | steps | estimate | timeout (s) |", "|---|---|---|---|---|---|"]
        total = SETUP["job_overhead"]
        for label, case, slabs, steps, warmup, seam_check, trace, kind in PHASE1:
            weak = case.endswith("_k2")
            estimate = model.run(case, slabs, steps, cold=True, seam_check=seam_check, weak=weak)
            if trace:
                estimate += SETUP["trace_write"]
            if kind == "dump":        # the dump worker: a readback of every field + an npz write (~1 GB per 8M)
                estimate += 0.08 * PARTICLES[case] / 1e6 * 3 + 5
            phase1.append({"label": label, "case": case, "slabs": slabs, "steps": steps, "seconds": estimate})
            total += estimate
            lines.append(f"| {label} | {case} | {slabs} | {steps} | {seconds(estimate)} | "
                         f"{int(max(600, math.ceil((2 * estimate + 300) / 60) * 60))} |")
        lines.append(f"\nPhase-1 job estimate incl. {SETUP['job_overhead']:.0f} s of overhead and the field "
                     f"comparisons: **{total / 60:.1f} min** (limit 30 min).")
    text = "\n".join(lines)
    print(text)
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({"model": vars(arguments), "points": points, "phase1": phase1,
                                                            "phase2": phase2,
                                                            "line_hours": {"A": job_a / 3600, "B": job_b / 3600}},
                                                           indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
