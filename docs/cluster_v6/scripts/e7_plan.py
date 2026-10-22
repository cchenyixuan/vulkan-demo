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

Job scripts of another deployment (default: the batch jobs source e7_lib.sh from ~/run/vulkan-demo-v7rc1 and
check v7-rc1's tree): --source-repo DIR changes the source line's default deployment; --solver-tag TAG with
--expected-tree SHA (git rev-parse <commit>:experiment/v7) and --expect-commit PREFIX are exported by every
emitted batch job (E7_SOLVER_TAG, E7_EXPECTED_TREE, E30_EXPECT_COMMIT; without them no export line is written).

Revision of 2026-10-12 (--revision; user, after the batch-3 report: batches 2 and 4 on v7-rc2, E7_ARMS=C exported):
  - the calibrated arm only (R C / C R alternating), 3 trials at every point, one traced run (arm C) per strong
    point and none per weak point, cross repeats without traces;
  - batch 2 = F1 K = 8 n2000, n2840, n5640, n11320 (n9000 dropped; n4000 and n8000 K = 8 keep the validation job's
    rc2 values), F2 n5640 and n8000 at K = 2 and 4 (n8000 K = 4 re-measured: batch 1 was rc1), F3 at 2M / 8M / 32M
    per card only (n1440_k4, n1440_k8, n2840_k2 / k4 / k8, n5640_k2 / k4 / k8; n2840_k4 re-measured; n1440_k2 keeps
    the validation job's value; n2000_k* and n4000_k* dropped);
  - batch 4 = one line-B job: F7 K = 8 n520 and n1000 (F7 K = 2 / 4, F8 and the cube soak dropped), the n8000
    anatomy at K = 1 and K = 8 plus the 3-D anatomy 3d_n200 K = 1, 3d_n200_k8 K = 1 and K = 8 (per-kernel time per
    particle, cube against stretched; ANATOMY_STEPS_3D steps), the cross repeat of 2d_n8000 K = 8 only (submit with
    --exclude=<the validation job's node>).

Revision 2 (--revision 2; user, 2026-10-12 evening): 2-D only this round, 3-D postponed (revision 1's batch 4 keeps
  the 3-D anatomy for later). rc2, arm C, 3 trials, 2-D windows of 1500 steps (warmup 500): A 2d_n8000 K = 1 (a
  step-traced K = 1 run), 2, 4, 8 with traces; B K = 8 n8000, n11320, n520, n1000 with traces (n4000 K = 8 done in
  the validation job); C weak 2M / 8M / 32M per card at K = 2 / 4 / 8, no traces (n1440_k2 done); D the n8000
  anatomy at K = 1 (line A) and K = 8 (line B); E n8000 K = 8 on another node = the cross repeat of line B's job,
  with the trace (submit with --exclude=<the validation job's node>). One batch, two jobs (e7_b2A, e7_b2B).

Revision 3 (--revision 3; user, 2026-10-13 morning): the batch-2 supplement, one job e7_b2S (REVISIONS[3]).
Revision 4 (--revision 4; user, 2026-10-13 afternoon): the second supplement, one job e7_b2T: 2d_n8000 at K = 3, 5,
  6, 7 (3 trials, traced, cards 0..K-1 in bus order); the per-K model tables are interpolated in K (Model.per_k).
Revision 5 (--revision 5): the third supplement e7_b2U (cancelled before it started). Revision 6 (--revision 6; user,
  2026-10-15): the fourth supplement, jobs e7_b2V (strong, batch 8) and e7_b2W (weak at K = 3, 5, 6, 7 + one traced
  2d_n1440_k8 K = 8 run, batch 9); a point with 0 trials runs its calibration and traced runs only. Revision 7
  (--revision 7; user, 2026-10-16): b2W stopped at the 3 % rule; the weak item again as job e7_b2X (batch 10, on
  wqd10nbj04g2) or, if that has not started within 6 h, e7_b2Y (batch 11, any node but wqd10nbm13g5 / wqd10nah09g1,
  without 2d_n1440_k3 / _k5); both stop before any run unless the eight power caps are equal (E7_UNIFORM_POWER_CAP).
Revision 8 (--revision 8; user, 2026-10-19): the phases of the K = 1 reference runs for the paper's tab:phases, one job
  e7_b2Z (batch 12, submitted with --nodelist=wqd10nbj04g2): four traced reference groups (e7_trace_group: the eight
  simultaneous K = 1 runs of a case behind one start barrier, step-traced like the K = 8 traced runs), strong
  cavity2d_n8000 / _n5640 / _n4000 and the weak per-card case cavity2d_n1440, one trial of 1500 / 500 steps, no points.
Revision 9 (--revision 9; user, 2026-10-19 evening): b2Z keeps waiting for wqd10nbj04g2; a second job e7_b2AA (batch
  13) of the same four traced reference groups plus, on the same node, the K = 8 chain of each case traced once
  (points with 0 trials: calibration + the arm-C traced run, as the K = 8 traces of b2B / b2V / b2S / b2Y), on any
  node of the partition but wqd10nbm13g5 / wqd10nah09g1 (--exclude), the whole node (--exclusive), every card's
  power cap checked to be 575 W before any run (E7_POWER_CAP_W). Its sbatch name is e7_B9, not the line's
  singleton e7_B (that would wait for b2Z to end). Whichever of b2Z / b2AA starts first is kept, the other cancelled
  (a watcher outside the plan).
Revision 10 (--revision 10; user, 2026-10-20 night): the 8M-per-device weak point at K = 5 again (2d_n2840_k5 K = 5,
  eta 0.980 in b2Y 1704741, where the second pilot round measured GPU 4 3 % faster than the first and gave it
  omega 1.0354). One job e7_b2AB (batch 14, sbatch name e7_B10): (1) the point under the protocol (a fresh 2-round
  calibration, 3 trials R C / C R / R C with the K = 1 references of cavity2d_n2840 on the same cards); (2) the same
  point with b2Y's weights file (E7_WEIGHTS_FROM, labels old_*: the file copied into the job's weights/ after its
  sha256 is checked, no calibration, 3 trials with references); (3) one traced run of each (new and old weights).
  Any node but wqd10nbm13g5 / wqd10nah09g1, the whole node, every card at 575 W or the job stops before any run,
  the 3 % spread rule. The point's eta is computed from this job's own references only.

Validation job (--emit-validation DIR: e7_v1.sbatch, job e7_V, line V; user item 7): the job's deployment
(VALIDATION source repo, tag rc2, --validation-tree / --validation-commit, placeholders until rc2 exists) under
the full protocol at two batch-1 points (same labels as batch 1: they replace those points), and A/B points of
the two solvers (e7_pair_point: one calibration by the job's solver, then trials x (the reference sets + one K run
per solver back to back, the order alternating); no traces) against v7-rc1 (E7_RC1_REPO, read only).

Usage:
    python docs/cluster_v6/scripts/e7_plan.py [--markdown] [--json OUT] [--emit-jobs DIR [--batch N ...]]
        [--model-json FILE] [--source-repo DIR] [--solver-tag TAG --expected-tree SHA] [--expect-commit PREFIX]
        [--emit-validation DIR [--validation-tree SHA] [--validation-commit PREFIX]] [--revision]
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
    # revision 6 (user 2026-10-15): the weak members at K = 3, 5, 6, 7 (same generator, --n-x = n K)
    "cavity2d_n1440_k3": 6_348_004, "cavity2d_n1440_k5": 10_558_564, "cavity2d_n1440_k6": 12_663_844,
    "cavity2d_n1440_k7": 14_769_124, "cavity2d_n2840_k3": 24_447_204, "cavity2d_n2840_k5": 40_703_364,
    "cavity2d_n2840_k6": 48_831_444, "cavity2d_n2840_k7": 56_959_524, "cavity2d_n5640_k3": 95_925_604,
    "cavity2d_n5640_k5": 159_792_964, "cavity2d_n5640_k6": 191_726_644, "cavity2d_n5640_k7": 223_660_324,
    "cavity3d_n104_b9": 1_815_848, "cavity3d_n160": 4_741_632, "cavity3d_n160_k2": 9_257_472,
    "cavity3d_n160_k4": 18_289_152, "cavity3d_n160_k8": 36_352_512, "cavity3d_n200": 8_998_912,
    "cavity3d_n200_b9": 10_360_232, "cavity3d_n200_k2": 17_651_712, "cavity3d_n200_k4": 34_957_312,
    "cavity3d_n200_k8": 69_568_512, "cavity3d_n200_x96": 4_499_456, "cavity3d_n416": 76_225_024,
    # E8 (2026-10-22): the cubes by total edge (cases/aligned/cavity3d_n<n>/generate.txt line 1), the weak boxes, the stand-in
    "cavity3d_n48": 175_616, "cavity3d_n152": 4_096_000, "cavity3d_n192": 8_000_000, "cavity3d_n244": 16_003_008,
    "cavity3d_n312": 32_768_000, "cavity3d_n392": 64_000_000, "cavity3d_n192_k2": 15_680_000,
    "cavity3d_n192_k4": 31_040_000, "cavity3d_n192_k8": 61_760_000,
    # E8 e8_C (user 2026-10-22, the third job): the 128M cube (504^3; its references are K = 2 pairs)
    "cavity3d_n496": 128_024_064,
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
SOURCE_REPOSITORY = "$HOME/run/vulkan-demo-v7rc1"      # the batch jobs' source line default (--source-repo)
# validation job (user item 7): rc2 (the job's deployment) under the full protocol at two batch-1 points (their
# labels, so they replace them), and A/B points of rc2 against v7-rc1 (read only), arm C, no traces; the 3-D point
# with the per-card reference only (keeps the job small)
VALIDATION = {"job": "v1", "line": "V", "source_repository": "$HOME/run/vulkan-demo-v7rc2",
              "rc1_repository": "$HOME/run/vulkan-demo-v7rc1", "solver_tag": "rc2",
              "points": [("F3", "cavity2d_n1440_k2", 2, ["cavity2d_n1440"]), ("F1+F7", "cavity2d_n4000", 8, ["same"])],
              "pairs": [("F1+F7", "cavity2d_n8000", 8, ["same"]), ("F4", "cavity3d_n160_k8", 8, ["cavity3d_n160"])],
              "pair_trials": 3, "pair_arm": "C"}
PLACEHOLDER_TREE = "PLACEHOLDER_TREE"           # rc2's tree and commit are not known yet: the job stops at its start
PLACEHOLDER_COMMIT = "PLACEHOLDER_COMMIT"
# revision of 2026-10-12 (docstring): the arm, the trials, the traces per point kind, the batch 2 / 4 point lists
# (case, K), batch 4's extras and cross repeat
REVISIONS = {
    1: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0,
        "batches": {2: {"A": [("cavity2d_n5640", 2), ("cavity2d_n5640", 4), ("cavity2d_n8000", 2), ("cavity2d_n8000", 4),
                              ("cavity2d_n1440_k4", 4), ("cavity2d_n2840_k2", 2), ("cavity2d_n2840_k4", 4),
                              ("cavity2d_n5640_k2", 2), ("cavity2d_n5640_k4", 4)],
                        "B": [("cavity2d_n2000", 8), ("cavity2d_n2840", 8), ("cavity2d_n5640", 8), ("cavity2d_n11320", 8),
                              ("cavity2d_n1440_k8", 8), ("cavity2d_n2840_k8", 8), ("cavity2d_n5640_k8", 8)]},
                    4: {"A": [], "B": [("cavity2d_n520", 8), ("cavity2d_n1000", 8)]}},
        "extras": {2: {"A": ([], []), "B": ([], [])}, 4: {"A": ([], []), "B": ([], ["anatomy", "anatomy_3d", "cross"])}},
        "cross": {"A": [], "B": [("cavity2d_n8000", 8)]},
    },
    # revision 2 (user, 2026-10-12 evening): 2-D only this round, 3-D postponed; rc2, arm C, 3 trials, 2-D windows of
    # 1500 steps (warmup 500). A: 2d_n8000 K = 1 (a step-traced K = 1 run) / 2 / 4 / 8 with traces; B: K = 8 n4000
    # (done, validation job), n8000, n11320, n520, n1000 with traces; C: weak 2M / 8M / 32M per card at K = 2 / 4 / 8
    # without traces (n1440_k2 done, validation job); D: the n8000 anatomy at K = 1 and K = 8; E: n8000 K = 8 on
    # another node (the cross repeat, with the trace: the validation job's three trials have none).
    2: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 1, "window_2d": (1500, 500),
        "batches": {2: {"A": [("cavity2d_n8000", 2), ("cavity2d_n8000", 4), ("cavity2d_n1440_k4", 4),
                              ("cavity2d_n2840_k2", 2), ("cavity2d_n2840_k4", 4), ("cavity2d_n5640_k2", 2),
                              ("cavity2d_n5640_k4", 4)],
                        "B": [("cavity2d_n11320", 8), ("cavity2d_n520", 8), ("cavity2d_n1000", 8),
                              ("cavity2d_n1440_k8", 8), ("cavity2d_n2840_k8", 8), ("cavity2d_n5640_k8", 8)]}},
        "extras": {2: {"A": ([], ["trace_2d_k1", "anatomy_2d_k1"]), "B": ([], ["cross", "anatomy_2d_k8"])}},
        "cross": {"A": [], "B": [("cavity2d_n8000", 8)]},
    },
    # revision 3 (user, 2026-10-13: the batch-2 supplement, one whole-node job b2S): 1. the two old-window points
    # again (2d_n4000 K8 traced, 2d_n1440_k2 K2); 2. the K = 8 strong-scaling transition n1440 / n2000 / n2840,
    # traced; 3. the 2M-per-card weak points to 5 trials (n1440_k2 K2 = item 1, n1440_k4 K4, n1440_k8 K8);
    # 5. if this account may lock the SM clocks (e7_clock_lock), 2d_n8000 K8 and 2d_n5640_k8 K8 with their K = 1
    # references at one locked frequency, else skipped. Item 4 (n11320 K = 1) needs no run: 128.6M does not fit.
    3: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F1", "cavity2d_n1440", 8, "same")],
        "any_line": True,                       # the K = 2 / 4 points run in the same whole-node job
        "trials_overrides": {("cavity2d_n1440_k2", 2): 5, ("cavity2d_n1440_k4", 4): 5, ("cavity2d_n1440_k8", 8): 5},
        "batches": {5: {"B": [("cavity2d_n4000", 8), ("cavity2d_n1440_k2", 2), ("cavity2d_n1440", 8),
                              ("cavity2d_n2000", 8), ("cavity2d_n2840", 8), ("cavity2d_n1440_k4", 4),
                              ("cavity2d_n1440_k8", 8)]}},
        "extras": {5: {"A": ([], []), "B": ([], ["lock"])}},
        "lock": [("cavity2d_n8000", 8), ("cavity2d_n5640_k8", 8)], "lock_mhz": 2500, "lock_trials": 3,
        "cross": {"A": [], "B": []},
        "job_names": {5: "b2S"},
    },
    # revision 4 (user, 2026-10-13 afternoon: the second batch-2 supplement, one whole-node job b2T): 2d_n8000's
    # strong scaling at K = 3, 5, 6, 7 (3 trials, one traced run each), the slabs on cards 0..K-1 in bus order
    # (e7_device_map; cards 0-3 = NUMA node 0, 4-7 = node 1: the seam 3|4 crosses the sockets at K >= 5), the K = 1
    # references on the same cards. The model's per-K tables (K = 2, 4, 8) are interpolated in K for these points.
    4: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F2", "cavity2d_n8000", 3, "same"), ("F2", "cavity2d_n8000", 5, "same"),
                         ("F2", "cavity2d_n8000", 6, "same"), ("F2", "cavity2d_n8000", 7, "same")],
        "any_line": True,                       # K < 8 points in the whole-node job (line B: its n8000 refmode k1)
        "batches": {6: {"B": [("cavity2d_n8000", 3), ("cavity2d_n8000", 5), ("cavity2d_n8000", 6),
                              ("cavity2d_n8000", 7)]}},
        "extras": {6: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {6: "b2T"},
    },
    # revision 5 (user, 2026-10-13 afternoon: the third supplement, job b2U): revision 4's four K again, on the batch-2
    # node (submitted with -w wqd10nbj04g2 --exclude=wqd10nbm13g5; 1695731 ran on a 600 W node and its GPU 5 was slow
    # and then lost), order K = 7, 6, 5, 3; the job exports E7_REFERENCE_SPREAD_PCT=3 (e7_lib.sh: a K = 1 reference
    # group whose member deviates more than 3 % from the mean of the others stops the job).
    5: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F2", "cavity2d_n8000", 3, "same"), ("F2", "cavity2d_n8000", 5, "same"),
                         ("F2", "cavity2d_n8000", 6, "same"), ("F2", "cavity2d_n8000", 7, "same")],
        "any_line": True,
        # + (user, a few minutes later) cavity2d_n5640 at K = 8 (4.0M per card, F1 strong scaling), 3 trials, traced,
        # after the four n8000 points (the pending job 1695948 was cancelled and resubmitted with it)
        "batches": {7: {"B": [("cavity2d_n8000", 7), ("cavity2d_n8000", 6), ("cavity2d_n8000", 5),
                              ("cavity2d_n8000", 3), ("cavity2d_n5640", 8)]}},
        "extras": {7: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {7: "b2U"},
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3")],
    },
    # revision 6 (user, 2026-10-15: the fourth supplement; replaces b2U, whose job 1695962 was cancelled before it
    # started): any hp_5090 node but wqd10nbm13g5, the points of one item of the request on one node. Job b2V (batch
    # 8) = item 1, revision 5's five strong points (3 trials, one traced run each); job b2W (batch 9) = item 2, the weak
    # members cavity2d_n{1440,2840,5640}_k{3,5,6,7} at their K (reference = K = 1 of the per-card case on the same
    # cards, the n1440 members 5 trials, the others 3, no traces), and item 3, one traced run of 2d_n1440_k8 at K = 8
    # (its calibration and the traced run, no trials: b2S has the five). Both export the 3 % reference-spread rule.
    6: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F2", "cavity2d_n8000", slabs, "same") for slabs in (3, 5, 6, 7)]
                        + [("F3", f"cavity2d_n{n}_k{slabs}", slabs, f"cavity2d_n{n}")
                           for n in (1440, 2840, 5640) for slabs in (3, 5, 6, 7)],
        "any_line": True,
        "trials_overrides": {**{(f"cavity2d_n1440_k{slabs}", slabs): 5 for slabs in (3, 5, 6, 7)},
                             ("cavity2d_n1440_k8", 8): 0},
        "traces_overrides": {("cavity2d_n1440_k8", 8): 1},
        "batches": {8: {"B": [("cavity2d_n8000", 7), ("cavity2d_n8000", 6), ("cavity2d_n8000", 5),
                              ("cavity2d_n8000", 3), ("cavity2d_n5640", 8)]},
                    9: {"A": [(f"cavity2d_n{n}_k{slabs}", slabs) for n in (1440, 2840, 5640) for slabs in (3, 5, 6, 7)]
                             + [("cavity2d_n1440_k8", 8)]}},
        "extras": {8: {"A": ([], []), "B": ([], [])}, 9: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {8: "b2V", 9: "b2W"},
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3")],
    },
    # revision 7 (user, 2026-10-16): b2W (1703658, wqd10nah09g1) stopped at the 3 % rule (2d_n1440_k6 K = 6 reference:
    # GPU 5, the node's one 600 W card, +3.31 %). The whole weak item again on wqd10nbj04g2 (job b2X, batch 10 =
    # revision 6's batch 9, submitted with --nodelist=wqd10nbj04g2) so that all 2-D scaling data share one node; if it
    # has not started within 6 h it is cancelled and job b2Y (batch 11) runs on any node but wqd10nbm13g5 and
    # wqd10nah09g1, without 2d_n1440_k3 / _k5 (kept from wqd10nah09g1). Both export E7_UNIFORM_POWER_CAP=1 (e7_lib.sh:
    # the eight cards' power caps must be equal, else the job stops before any run) and the 3 % spread rule.
    7: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F3", f"cavity2d_n{n}_k{slabs}", slabs, f"cavity2d_n{n}")
                         for n in (1440, 2840, 5640) for slabs in (3, 5, 6, 7)],
        "any_line": True,
        "trials_overrides": {**{(f"cavity2d_n1440_k{slabs}", slabs): 5 for slabs in (3, 5, 6, 7)},
                             ("cavity2d_n1440_k8", 8): 0},
        "traces_overrides": {("cavity2d_n1440_k8", 8): 1},
        "batches": {10: {"A": [(f"cavity2d_n{n}_k{slabs}", slabs) for n in (1440, 2840, 5640) for slabs in (3, 5, 6, 7)]
                              + [("cavity2d_n1440_k8", 8)]},
                    11: {"A": [(f"cavity2d_n{n}_k{slabs}", slabs) for n in (1440, 2840, 5640) for slabs in (3, 5, 6, 7)
                               if (n, slabs) not in ((1440, 3), (1440, 5))] + [("cavity2d_n1440_k8", 8)]}},
        "extras": {10: {"A": ([], []), "B": ([], [])}, 11: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {10: "b2X", 11: "b2Y"},
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3"), ("E7_UNIFORM_POWER_CAP", "1")],
    },
    # revision 8 (user, 2026-10-19): the phases of the K = 1 reference runs for the paper's tab:phases. One job b2Z
    # (batch 12, line B, submitted with --nodelist=wqd10nbj04g2 = the node of every K = 8 trace) of four traced
    # reference groups (e7_trace_group: the eight simultaneous K = 1 runs of a case behind one start barrier, each
    # step-traced like the K = 8 traced runs), in the user's order: strong cavity2d_n8000 / _n5640 / _n4000 and the
    # weak per-card case cavity2d_n1440; one trial of 1500 / 500 steps; no points, no calibration. The 3 % spread rule
    # and the uniform power-cap check as in revision 7.
    8: {
        "arms": "C", "trials": 1, "traces_strong": 0, "traces_weak": 0, "cross_traces": 0, "window_2d": (1500, 500),
        "any_line": True,
        "batches": {12: {"B": []}},
        "extras": {12: {"A": ([], []), "B": (["tracegroup_k1"], [])}},
        "tracegroups": [("cavity2d_n8000", 8), ("cavity2d_n5640", 8), ("cavity2d_n4000", 8), ("cavity2d_n1440", 8)],
        "cross": {"A": [], "B": []},
        "job_names": {12: "b2Z"},
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3"), ("E7_UNIFORM_POWER_CAP", "1")],
    },
    # revision 9 (user, 2026-10-19 evening): b2Z stays queued for wqd10nbj04g2; a second job b2AA (batch 13, line B)
    # of the same four traced reference groups and then, on the same node, the K = 8 chain of each case traced once
    # (points with 0 trials: calibration + the arm-C traced run, as the K = 8 traces of b2B / b2V / b2S / b2Y were
    # made), any node but wqd10nbm13g5 / wqd10nah09g1, the whole node, every card at 575 W (E7_POWER_CAP_W) or the
    # job stops before any run. sbatch name e7_B9 (job_tag), not the line's singleton e7_B, which would wait for b2Z.
    9: {
        "arms": "C", "trials": 0, "traces_strong": 1, "traces_weak": 1, "cross_traces": 0, "window_2d": (1500, 500),
        "any_line": True,
        "batches": {13: {"B": [("cavity2d_n8000", 8), ("cavity2d_n5640", 8), ("cavity2d_n4000", 8),
                               ("cavity2d_n1440_k8", 8)]}},
        "extras": {13: {"A": ([], []), "B": (["tracegroup_k1"], [])}},
        "tracegroups": [("cavity2d_n8000", 8), ("cavity2d_n5640", 8), ("cavity2d_n4000", 8), ("cavity2d_n1440", 8)],
        "cross": {"A": [], "B": []},
        "job_names": {13: "b2AA"},
        "job_tag": "9",
        # --exclusive alone is refused by hp_5090 ("Requested node configuration is not available": select/cons_tres,
        # CR_CPU_MEMORY); --exclusive --mem=0 (the node's whole memory) is accepted and allocates all 128 CPUs
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575")],
    },
    # revision 10 (user, 2026-10-20 night): the 8M-per-device K = 5 weak point again (2d_n2840_k5 K5, eta 0.980 in b2Y
    # 1704741: its second pilot round measured GPU 4 3 % faster than the first and gave it omega 1.0354, +3.75 %
    # particles over the mean; the trials then waited on GPU 4). One job b2AB (batch 14, line B, sbatch name e7_B10):
    # (1) the point under the protocol (fresh 2-round calibration, 3 trials R C / C R / R C, the K = 1 references of
    # cavity2d_n2840 on the same cards), (2) the same point with b2Y's weights file ("fixed_weights": E7_WEIGHTS_FROM
    # copied into the job's weights/ under the prefix old_ after its sha256 is checked; no calibration; 3 trials with
    # references), (3) one traced run of each (new and old weights: traces_weak 1). Any node but wqd10nbm13g5 /
    # wqd10nah09g1, the whole node (--exclusive --mem=0), every card at 575 W (E7_POWER_CAP_W) or the job stops
    # before any run, the 3 % spread rule. eta of both points from this job's own references only.
    10: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 1, "cross_traces": 0, "window_2d": (1500, 500),
        "extra_points": [("F3", "cavity2d_n2840_k5", 5, "cavity2d_n2840")],
        "any_line": True,
        "batches": {14: {"B": [("cavity2d_n2840_k5", 5), ("cavity2d_n2840_k5", 5, "old")]}},
        # a third element names a "fixed_weights" entry: the point runs with that weights file instead of a calibration
        "fixed_weights": {"old": {"prefix": "old_",
                                  "from": "$HOME/run/logs/e7_b2Y_1704741/weights/2d_n2840_k5_K5.json",
                                  "sha256": "d2e283170ae51618b724c96bb2c5615259deb19e42d7233ba15e75a4968c4f5f"}},
        "extras": {14: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {14: "b2AB"},
        "job_tag": "10",
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_REFERENCE_SPREAD_PCT", "3"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575")],
    },
    # revision 11 (user 2026-10-22: E8 phase 2, the 3-D cubic-cavity scaling on v7-rc3 + the partitioner floor 3 = tag
    # e8-3d-p2). Points: strong load sweep 4M / 8M (5 trials) and 16M / 32M / 64M (3 trials) at K = 8 (reference = K = 1 of
    # the case), strong K sweep 64M at K = 2 / 4 (reference = K = 1 of 64M), weak 2 / 4 / 8 cubes of 8M at K = 2 / 4 / 8
    # (reference = K = 1 of the 8M cube); E7 protocol (1500 / 500, two pilot rounds, simultaneous references behind one
    # barrier, arm C), every point traced once with every kernel's tick (E7_TRACE_DETAIL=full) and V7_POOL_PEAKS=1, the
    # five cubes' K = 1 traced once (the ideal-chain rows of tab:phases), the per-step raw data of every run (E7_RAW=1),
    # the partition / calibration record of every point (E7_PARTITION_RECORD=1). Line B (K = 8, in this order): 4M ->
    # its dumps (K = 1 twice, K = 8 once, at step 1500, equal weights) + comparison -> its K = 1 trace -> 8M -> dumps +
    # comparison -> K = 1 trace -> 16M -> trace -> 32M -> trace -> 64M -> trace -> weak 8 -> the v7_single comparison
    # (3 interleaved pairs on card 0, E8_SINGLE_REPO). Line A (K <= 4, cards 0-3): 64M K2 -> 64M K4 -> weak 2 -> weak
    # 4, each point's trace right after its trials. A point's hard failure is recorded and the job goes on
    # (E7_CONTINUE_ON_FAILURE=1). Whole nodes, every card at 575 W, 600 W nodes excluded; time limit 2 x estimate + 1 h.
    11: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 1, "cross_traces": 0,
        "extra_points": [("E8S", "cavity3d_n152", 8, "same"), ("E8S", "cavity3d_n192", 8, "same"),
                         ("E8S", "cavity3d_n244", 8, "same"), ("E8S", "cavity3d_n312", 8, "same"),
                         ("E8S", "cavity3d_n392", 8, "same"), ("E8K", "cavity3d_n392", 2, "same"),
                         ("E8K", "cavity3d_n392", 4, "same"), ("E8W", "cavity3d_n192_k2", 2, "cavity3d_n192"),
                         ("E8W", "cavity3d_n192_k4", 4, "cavity3d_n192"), ("E8W", "cavity3d_n192_k8", 8, "cavity3d_n192")],
        "any_line": True,
        "trials_overrides": {("cavity3d_n152", 8): 5, ("cavity3d_n192", 8): 5},
        "batches": {15: {"A": [("cavity3d_n392", 2), ("cavity3d_n392", 4), ("cavity3d_n192_k2", 2), ("cavity3d_n192_k4", 4)],
                         "B": [("cavity3d_n152", 8), ("cavity3d_n192", 8), ("cavity3d_n244", 8), ("cavity3d_n312", 8),
                               ("cavity3d_n392", 8), ("cavity3d_n192_k8", 8)]}},
        # the items of each line in the user's order (build_jobs: a "sequence" replaces extras + points)
        "sequence": {15: {
            "B": [("point", "cavity3d_n152", 8), ("dump", "cavity3d_n152", 1, "a"), ("dump", "cavity3d_n152", 1, "b"),
                  ("dump", "cavity3d_n152", 8, ""), ("compare", "cavity3d_n152", 8), ("trace_k1", "cavity3d_n152"),
                  ("point", "cavity3d_n192", 8), ("dump", "cavity3d_n192", 1, "a"), ("dump", "cavity3d_n192", 1, "b"),
                  ("dump", "cavity3d_n192", 8, ""), ("compare", "cavity3d_n192", 8), ("trace_k1", "cavity3d_n192"),
                  ("point", "cavity3d_n244", 8), ("trace_k1", "cavity3d_n244"),
                  ("point", "cavity3d_n312", 8), ("trace_k1", "cavity3d_n312"),
                  ("point", "cavity3d_n392", 8), ("trace_k1", "cavity3d_n392"),
                  ("point", "cavity3d_n192_k8", 8), ("single_ratio", "cavity3d_n192", 3)],
            "A": [("point", "cavity3d_n392", 2), ("point", "cavity3d_n392", 4), ("point", "cavity3d_n192_k2", 2),
                  ("point", "cavity3d_n192_k4", 4)]}},
        "extras": {15: {"A": ([], []), "B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {15: {"A": "e8A", "B": "e8B"}},
        "job_prefix": "e8",
        "description": "E8 phase 2 (user 2026-10-22), 3-D cubic cavity on tag e8-3d-p2 (generated by docs/cluster_v6/scripts/e7_plan.py --revision 11 --emit-jobs",
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_RAW", "1"), ("E7_TRACE_DETAIL", "full"), ("E7_CONTINUE_ON_FAILURE", "1"),
                    ("E7_PARTITION_RECORD", "1"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575"),
                    ("E8_SINGLE_REPO", "$HOME/run/vulkan-demo-v7single")],
    },
    # revision 12 (user 2026-10-22, after the phase-2 jobs were queued: e8_C, one more point of the K = 8 load sweep at
    # w ~ 15.5 columns per card). The 128M cube cavity3d_n496 (504^3 = 128,024,064 particles, 124 fluid columns; K = 8 ->
    # four slabs of 15 and four of 16 columns) at K = 8, 3 trials, two pilot rounds. K = 1 does not fit one card, so the
    # references are E7's 2-D 128M kind: four simultaneous K = 2 pairs of the case behind one barrier ("pairs"; eta =
    # fps_K / (4 x mean pair fps), the mean fps = the harmonic mean of the pairs' step times). Traces: the K = 8 chain once
    # (the point's trace, every kernel's tick, V7_POOL_PEAKS) and one K = 2 pair once (trace_k2 = e7_extra trace at K = 2,
    # cards 0, 1, equal weights: the ideal-chain row = a quarter of the pair's phases). No dumps, no per-kernel runs, no
    # single-solver comparison. Line C = job e8_C (whole node, 8 cards, queued after e8_A); E7_RAW, E7_PARTITION_RECORD,
    # 575 W as the phase-2 jobs; time limit 2 x estimate + 1 h.
    12: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 1, "cross_traces": 0,
        "extra_points": [("E8S", "cavity3d_n496", 8, "pairs")],
        "any_line": True,
        "batches": {16: {"C": [("cavity3d_n496", 8)]}},
        "sequence": {16: {"C": [("point", "cavity3d_n496", 8), ("trace_k2", "cavity3d_n496")]}},
        "extras": {16: {"C": ([], [])}},
        "cross": {"A": [], "B": [], "C": []},
        "job_names": {16: {"C": "e8C"}},
        "job_prefix": "e8",
        "description": "E8 phase 2 addition e8_C (user 2026-10-22), the 128M cube at K = 8 with K = 2 pair references, on tag e8-3d-p2c (generated by docs/cluster_v6/scripts/e7_plan.py --revision 12 --emit-jobs",
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_RAW", "1"), ("E7_TRACE_DETAIL", "full"), ("E7_CONTINUE_ON_FAILURE", "1"),
                    ("E7_PARTITION_RECORD", "1"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575")],
    },
    # revision 13 (2026-10-22, after job 1721839): the v7_single comparison again. Every single half of e8_B's
    # single_ratio exited 1: v7_single refuses the V7_WORKER_AFFINITY the job exports for the chain workers
    # (switches.py), and e8_single_bench swallowed the message; the chain halves passed. Job e8_S = bring-up + the
    # 3 interleaved pairs on card 0 of a whole 575 W node (the user's whole-node rule), nothing else.
    13: {
        "arms": "C", "trials": 3, "traces_strong": 0, "traces_weak": 0, "cross_traces": 0,
        "extra_points": [], "any_line": True,
        "batches": {17: {"B": []}},
        "sequence": {17: {"B": [("single_ratio", "cavity3d_n192", 3)]}},
        "extras": {17: {"B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {17: {"B": "e8S"}},
        "job_tag": "S",
        "job_prefix": "e8",
        "description": "E8 phase 2 re-run of the v7_single comparison (job 1721839's single halves failed on V7_WORKER_AFFINITY), on tag e8-3d-p2s (generated by docs/cluster_v6/scripts/e7_plan.py --revision 13 --emit-jobs",
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_RAW", "1"), ("E7_CONTINUE_ON_FAILURE", "1"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575"),
                    ("E8_SINGLE_REPO", "$HOME/run/vulkan-demo-v7single")],
    },
    # revision 14 (user 2026-10-22 morning: e8_D, the integer-column partner of 64M K8). The 76M cube cavity3d_n416 (E7's
    # case: 424^3 = 76,225,024 particles, 104 fluid columns = 13 per card with no remainder) at K = 8, 3 trials, 8 simultaneous
    # K = 1 references, two pilot rounds; traces: the K = 8 chain once and K = 1 once (every kernel's tick, V7_POOL_PEAKS,
    # E7_RAW); no dumps, no per-kernel runs, no single-solver comparison. Line B job e8_D (name e8_BD), whole 575 W node,
    # queued after e8_C; time limit 2 x estimate + 1 h.
    14: {
        "arms": "C", "trials": 3, "traces_strong": 1, "traces_weak": 1, "cross_traces": 0,
        "extra_points": [("E8S", "cavity3d_n416", 8, "same")],
        "any_line": True,
        "batches": {18: {"B": [("cavity3d_n416", 8)]}},
        "sequence": {18: {"B": [("point", "cavity3d_n416", 8), ("trace_k1", "cavity3d_n416")]}},
        "extras": {18: {"B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {18: {"B": "e8D"}},
        "job_tag": "D",
        "job_prefix": "e8",
        "description": "E8 phase 2 addition e8_D (user 2026-10-22), the 76M cube (13 columns per card) at K = 8 on tag e8-3d-p2d (generated by docs/cluster_v6/scripts/e7_plan.py --revision 14 --emit-jobs",
        "sbatch": ["--exclusive", "--mem=0", "--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_RAW", "1"), ("E7_TRACE_DETAIL", "full"), ("E7_CONTINUE_ON_FAILURE", "1"),
                    ("E7_PARTITION_RECORD", "1"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575")],
    },
    # revision 15 (user 2026-10-22 afternoon: e8_S re-measured). Job 1722130's v7_single halves never defragmented (the bench
    # skipped the bootstrap defrag under --no-defrag while the chain's bootstrap_all always runs one; claude/
    # e8_single_vs_chain_k1.md). Job e8_S2 (name e8_BS2): the bench with the bootstrap defrag (branch v7-single, deployment
    # ~/run/vulkan-demo-v7single2), Sec. 5.1's protocol - both solvers at the case's defrag cadence (1000), the window from
    # step 1001 (warmup 1000 + 1000 measured), 2 in flight, 3 interleaved pairs on card 0 - plus one control pair of
    # v7_single runs under job 1722130's protocol (--no-defrag, 1500 / 500): with and without the bootstrap defrag. Two
    # cards of a 575 W node (--gpus=2, the user: "两卡或整节点短作业"), nothing else; E7_EXPECTED_DEVICES=2.
    15: {
        "arms": "C", "trials": 3, "traces_strong": 0, "traces_weak": 0, "cross_traces": 0,
        "extra_points": [], "any_line": True,
        "batches": {19: {"B": []}},
        "sequence": {19: {"B": [("single_ratio", "cavity3d_n192", 3, "cadence", 2000, 1000),
                                ("single_control", "cavity3d_n192", 1500, 500)]}},
        "extras": {19: {"B": ([], [])}},
        "cross": {"A": [], "B": []},
        "job_names": {19: {"B": "e8S2"}},
        "job_tag": "S2",
        "job_prefix": "e8",
        "gpus": 2,
        "description": "E8 phase 2 re-measurement of the v7_single comparison (job 1722130's single halves ran without any defrag), Sec. 5.1 protocol + a bootstrap-defrag control pair, on tag e8-3d-p2s2 (generated by docs/cluster_v6/scripts/e7_plan.py --revision 15 --emit-jobs",
        # hp_5090's submit filter (slurm_job_submit_lua, probed 2026-10-22 16:22): a GPU job gets 8 CPUs and 250 GB per card and
        # refuses --cpus-per-gpu / --cpus-per-task above that / --mem; --gpus=2 alone (+ --exclude) is accepted
        "sbatch": ["--exclude=wqd10nbm13g5,wqd10nah09g1"],
        "exports": [("E7_RAW", "1"), ("E7_CONTINUE_ON_FAILURE", "1"), ("E7_UNIFORM_POWER_CAP", "1"), ("E7_POWER_CAP_W", "575"),
                    ("E7_EXPECTED_DEVICES", "2"), ("E8_SINGLE_REPO", "$HOME/run/vulkan-demo-v7single2")],
    },
}
REVISION = None                                 # --revision N: REVISIONS[N]
ARMS = "EC"                                     # the timed arms per trial (--revision: REVISION["arms"])


@dataclasses.dataclass
class Point:
    family: str
    case: str
    slabs: int
    references: list        # "same" = K = 1 of the case; "pairs" = K = 2 pairs of the case; else a k1 case (weak)
    cross: bool = False
    trials_fixed: int = None        # --revision: the trials and traces set by the revision instead of the rules
    traces_fixed: int = None
    # revision 10: the point runs with an earlier job's weights file (E7_WEIGHTS_FROM, its sha256 checked) instead of
    # a calibration; its labels carry the prefix (E7_PREFIX)
    weights_from: str = None
    weights_sha256: str = None
    prefix: str = ""

    @property
    def line(self) -> str:
        return "B" if self.slabs == 8 else "A"

    @property
    def strong(self) -> bool:
        """A strong-scaling point (the reference is the case itself, K = 1 or pairs), else weak (per-card case)."""
        return any(kind in ("same", "pairs") for kind in self.references)

    @property
    def dimension(self) -> int:
        return 3 if self.case.startswith("cavity3d") else 2

    @property
    def per_card(self) -> float:
        return PARTICLES[self.case] / self.slabs

    @property
    def trials(self) -> int:
        if self.trials_fixed is not None:
            return self.trials_fixed
        if self.cross:
            return CROSS_TRIALS
        return 5 if self.per_card <= LIGHT_PER_CARD else 3

    @property
    def traces(self) -> int:
        if self.traces_fixed is not None:
            return self.traces_fixed
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
    # a revision's own points (revision 3: 2d_n1440 at K = 8), not part of the 51-point campaign
    for family, case, slabs, reference in (REVISION or {}).get("extra_points", []):
        add(family, case, slabs, reference)
    return points


def cross_points(points: list[Point], revision: bool = False) -> dict:
    """Line -> the points it re-runs (copies with cross=True): the other line's (CROSS), or the revision's (its own
    line's point on another node, 0 traces, the revision's trials)."""
    by_key = {(point.case, point.slabs): point for point in points}
    table = REVISION["cross"] if revision else CROSS
    return {line: [dataclasses.replace(by_key[key], references=list(by_key[key].references), cross=True,
                                       trials_fixed=REVISION["trials"] if revision else None,
                                       traces_fixed=REVISION.get("cross_traces", 0) if revision else None)
                   for key in keys] for line, keys in table.items()}


def revised_batches(points: list[Point], batches: dict) -> dict:
    """--revision: batches 2 and 4 replaced by REVISION's point lists (copies of the campaign's points with the
    revision's trials and traces: one traced run at a strong point, none at a weak one); batches 1 and 3 are done."""
    by_key = {(point.case, point.slabs): point for point in points}
    out = dict(batches)
    for batch, lines in REVISION["batches"].items():
        out[batch] = {}
        for line, keys in lines.items():
            out[batch][line] = []
            for spec in keys:
                # (case, K) or (case, K, name): name = a "fixed_weights" entry of the revision (revision 10)
                key, fixed = tuple(spec[:2]), (REVISION["fixed_weights"][spec[2]] if len(spec) > 2 else None)
                point = by_key[key]
                if point.line != line and not REVISION.get("any_line"):
                    raise SystemExit(f"revision batch {batch}: {key} is a line {point.line} point")
                out[batch][line].append(dataclasses.replace(
                    point, references=list(point.references),
                    trials_fixed=REVISION.get("trials_overrides", {}).get(key, REVISION["trials"]),
                    traces_fixed=REVISION.get("traces_overrides", {}).get(
                        key, REVISION["traces_strong"] if point.strong else REVISION["traces_weak"]),
                    **({"weights_from": fixed["from"], "weights_sha256": fixed["sha256"], "prefix": fixed["prefix"]}
                       if fixed else {})))
    return out


def lock_items(points: list[Point], model: Model) -> list[dict]:
    """--revision with a "lock" group (revision 3, item 5): the points run under e7_clock_lock, between a lock_begin
    and a lock_end item (the job skips the block when the account cannot lock the clocks)."""
    by_key = {(point.case, point.slabs): point for point in points}
    items = [{"kind": "lock_begin", "seconds": 0.0, "mhz": REVISION["lock_mhz"]}]
    for key in REVISION["lock"]:
        point = dataclasses.replace(by_key[key], references=list(by_key[key].references),
                                    trials_fixed=REVISION.get("lock_trials", REVISION["trials"]), traces_fixed=0)
        items.append({"kind": "lock", "point": point, "seconds": point_seconds(point, model)["total"]})
    items.append({"kind": "lock_end", "seconds": 0.0})
    return items


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
    anatomy_steps_3d: int = 2000    # the 3-D anatomy runs of the revision (two frames; 69.6M at K = 1 is ~2.4 fps)
    soak_steps: int = 17000
    cube_soak_seconds: float = 3600.0

    def window(self, case: str) -> tuple[int, int]:
        if case.startswith("cavity3d"):
            return (1500, 500)
        return tuple(REVISION["window_2d"]) if REVISION and REVISION.get("window_2d") else (3000, 1000)

    @staticmethod
    def per_k(table: dict, slabs: int) -> float:
        """A per-K table value; a K the table lacks (3, 5, 6, 7: revision 4) is interpolated linearly in K between
        its nearest keys (clamped at the ends)."""
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

    def step_time(self, case: str, slabs: int) -> float:
        three_dimensional = case.startswith("cavity3d")
        cost = self.cost_3d if three_dimensional else self.cost_2d
        per_card = PARTICLES[case] / slabs
        if slabs == 1:
            return max(self.floor_k1, cost * per_card)
        efficiency = (self.efficiency_3d or self.efficiency) if three_dimensional else self.efficiency
        return max(self.per_k(self.floor, slabs), cost * per_card / self.per_k(efficiency, slabs))

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


def reference_seconds(point: Point, model: Model) -> tuple[float, float, dict]:
    """(the first trial's reference sets, a later trial's, {kind: one cold reference run}) of a point."""
    steps, _ = model.window(point.case)
    references_first = references_later = 0.0
    reference_runs = {}
    for kind in point.references:
        reference_case, slabs = reference_spec(point, kind)
        first = model.run(reference_case, slabs, steps, cold=True)
        later = model.run(reference_case, slabs, steps, cold=False)
        references_first += first
        references_later += later
        reference_runs[kind] = first
    return references_first, references_later, reference_runs


def calibration_seconds(point: Point, model: Model) -> float:
    pilot = model.setup(point.case, point.slabs, cold=True) + model.pilot_steps * model.step_time(point.case,
                                                                                                 point.slabs)
    return model.calibration_rounds * pilot + model.calibration_fixed


def point_seconds(point: Point, model: Model) -> dict:
    steps, _ = model.window(point.case)
    run_warm = model.run(point.case, point.slabs, steps, cold=False)
    references_first, references_later, reference_runs = reference_seconds(point, model)
    # a point with a given weights file (revision 10) has no calibration
    calibration = 0.0 if point.weights_from else calibration_seconds(point, model)
    trials_total = 0.0      # a point without trials (revision 6, item 3): its calibration and traced runs only
    if point.trials:
        trials_total = references_first + (point.trials - 1) * references_later + point.trials * len(ARMS) * run_warm
    traced = model.run(point.case, point.slabs, steps, cold=True, seam_check=True) + model.trace_write
    total = calibration + trials_total + point.traces * traced
    return {"run": run_warm, "reference": references_later, "reference_first": references_first,
            "reference_runs": reference_runs, "calibration": calibration, "traced": traced, "total": total}


def pair_seconds(point: Point, model: Model, trials: int, arm: str) -> dict:
    """An e7_pair_point: one calibration (arm C) by the job's solver, then TRIALS x (the reference sets + one K run
    per solver); no traces. Every K run counts warm (the rc1 half's pipelines are new to the job only if its
    SPIR-V differs from rc2's: one simulator construction more, inside the timeout margin)."""
    steps, _ = model.window(point.case)
    run_warm = model.run(point.case, point.slabs, steps, cold=False)
    references_first, references_later, reference_runs = reference_seconds(point, model)
    calibration = calibration_seconds(point, model) if arm == "C" else 0.0
    total = calibration + references_first + (trials - 1) * references_later + trials * 2 * run_warm
    return {"run": run_warm, "reference": references_later, "reference_first": references_first,
            "reference_runs": reference_runs, "calibration": calibration, "total": total}


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
        # revision 2: the same two anatomy runs on separate lines, and a step-traced K = 1 run of n8000 (item A)
        "anatomy_2d_k1": [("anatomy_2d_n8000_K1", "anatomy", "cavity2d_n8000", 1, model.anatomy_steps)],
        "anatomy_2d_k8": [("anatomy_2d_n8000_K8", "anatomy", "cavity2d_n8000", 8, model.anatomy_steps)],
        "trace_2d_k1": [("trace_2d_n8000_K1", "trace", "cavity2d_n8000", 1, model.window("cavity2d_n8000")[0])],
        # revision of 2026-10-12: per-kernel time per particle of the cube K = 1 (per-card case), the stretched
        # K = 1 (the strong reference, one card) and the stretched K = 8
        "anatomy_3d": [("anatomy_3d_n200_K1", "anatomy", "cavity3d_n200", 1, model.anatomy_steps_3d),
                       ("anatomy_3d_n200_k8_K1", "anatomy", "cavity3d_n200_k8", 1, model.anatomy_steps_3d),
                       ("anatomy_3d_n200_k8_K8", "anatomy", "cavity3d_n200_k8", 8, model.anatomy_steps_3d)],
        "soak_3d": [("soak_3d_n416_K8", "soak", "cavity3d_n416", 8, cube_steps)],
    }
    out = {}
    for group, entries in items.items():
        out[group] = []
        for label, kind, case, slabs, steps in entries:
            seconds = model.run(case, slabs, steps, cold=True, seam_check=kind in ("soak", "trace"))
            if kind == "trace":
                seconds += model.trace_write
            warmup = 200 if kind == "selftest" else model.window(case)[1]
            out[group].append({"label": label, "kind": kind, "case": case, "slabs": slabs, "steps": steps,
                               "warmup": warmup, "seconds": seconds})
    # revision 8: the traced reference groups (REVISION["tracegroups"]: case, parties), PARTIES simultaneous K = 1
    # runs of the case with the seam check and a step trace each: one cold K = 1 run (the members run in parallel)
    out["tracegroup_k1"] = []
    for case, parties in (REVISION or {}).get("tracegroups", []):
        steps, warmup = model.window(case)
        out["tracegroup_k1"].append({"label": f"reftrace_{case.replace('cavity', '')}_K1", "kind": "tracegroup",
                                     "case": case, "slabs": 1, "parties": parties, "steps": steps, "warmup": warmup,
                                     "seconds": model.run(case, 1, steps, cold=True, seam_check=True) + model.trace_write})
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
        batch = next((batch for batch, families in BATCH_FAMILIES.items() if point.family.split("+")[0] in families), None)
        if batch is None:           # a revision's own family (E8S / E8K / E8W): placed by the revision's batches only
            continue
        batches[batch][point.line].append(point)
    return batches


def build_jobs(points: list[Point], model: Model, revision: bool = False) -> dict:
    """(batch, line) -> [items], each item a dict with its kind, arguments and estimated seconds."""
    crosses = cross_points(points, revision)
    extra_items = extras(model)
    batches = batch_points(points)
    if revision:          # batches 1 and 3 are done: only the revised batches are planned and emitted
        batches = {batch: lines for batch, lines in revised_batches(points, batches).items()
                   if batch in REVISION["batches"]}
    jobs = {}
    for batch, lines in batches.items():
        for line in ("A", "B", "C"):            # line C: revision 12's e8_C (a line without entries has no job)
            extras_by_line = (REVISION["extras"].get(batch) or BATCH_EXTRAS[batch]) if revision else BATCH_EXTRAS[batch]
            before, after = extras_by_line.get(line, ([], []))
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
                                          "original_line": point.line})
                    elif group == "lock":
                        items.extend(lock_items(points, model))
                    else:
                        items.extend(dict(entry) for entry in extra_items[group])

            sequence = ((REVISION.get("sequence", {}) or {}).get(batch) or {}).get(line) if revision else None
            if sequence is not None:
                items = sequence_items(sequence, lines.get(line, []), model)
            else:
                add_groups(before)
                for point in lines.get(line, []):
                    items.append({"kind": "point", "point": point, "seconds": point_seconds(point, model)["total"]})
                add_groups(after)
            if items:               # a line without items has no job (the revision's batch 4 line A)
                jobs[(batch, line)] = items
    assigned = sum(1 for items in jobs.values() for item in items if item["kind"] == "point")
    if assigned != len(points) and not revision:
        raise SystemExit(f"{assigned} points in the batches, {len(points)} in the campaign")
    return jobs


def short_case(case: str) -> str:
    return case.replace("cavity", "")           # e7_lib.sh e7_short: cavity3d_n192 -> 3d_n192


def sequence_items(sequence: list, points: list[Point], model: Model) -> list[dict]:
    """E8 (revision 11): the items of a line in the given order. ("point", case, K): the revised point; ("dump", case, K,
    suffix): a seam-audit dump at the window's last step (dump_<short>_K<K><suffix>, equal weights); ("compare", case,
    K): the comparison of the K dump against the two K = 1 dumps; ("trace_k1", case): one step-traced K = 1 run;
    ("single_ratio", case, pairs): the v7_single comparison, PAIRS interleaved pairs on card 0; ("trace_k2", case): one
    step-traced K = 2 pair (revision 12, the ideal-chain row of a "pairs" point)."""
    by_key = {(point.case, point.slabs): point for point in points}
    items = []
    for spec in sequence:
        kind = spec[0]
        if kind == "point":
            point = by_key[(spec[1], spec[2])]
            items.append({"kind": "point", "point": point, "seconds": point_seconds(point, model)["total"]})
        elif kind == "dump":
            case, slabs, suffix = spec[1], spec[2], spec[3]
            steps = model.window(case)[0]
            seconds = model.run(case, slabs, steps, cold=True) + 0.3 * PARTICLES[case] / 1e6 + 20.0
            items.append({"kind": "dump", "label": f"dump_{short_case(case)}_K{slabs}{suffix}", "case": case, "slabs": slabs,
                          "steps": steps, "warmup": 0, "seconds": seconds})
        elif kind == "compare":
            case, slabs = spec[1], spec[2]
            items.append({"kind": "compare", "label": f"compare_{short_case(case)}_K{slabs}", "case": case, "slabs": slabs,
                          "steps": model.window(case)[0], "warmup": 0, "seconds": 60.0 + 20.0 * PARTICLES[case] / 1e6})
        elif kind == "trace_k1":
            case = spec[1]
            steps, warmup = model.window(case)
            items.append({"kind": "trace", "label": f"trace_{short_case(case)}_K1", "case": case, "slabs": 1, "steps": steps,
                          "warmup": warmup, "seconds": model.run(case, 1, steps, cold=True, seam_check=True) + model.trace_write})
        elif kind == "trace_k2":      # revision 12 (e8_C): one step-traced K = 2 pair (cards 0, 1, equal weights = a reference pair)
            case = spec[1]
            steps, warmup = model.window(case)
            items.append({"kind": "trace", "label": f"trace_{short_case(case)}_K2", "case": case, "slabs": 2, "steps": steps,
                          "warmup": warmup, "seconds": model.run(case, 2, steps, cold=True, seam_check=True) + model.trace_write})
        elif kind == "single_ratio":
            # revision 15: optional defrag mode (off = job 1721839 / 1722130's --no-defrag on both halves; cadence = Sec. 5.1's,
            # both at the case's cadence) and window (steps, warmup); the label carries the mode when it is not off
            case, pairs = spec[1], spec[2]
            mode = spec[3] if len(spec) > 3 else "off"
            steps, warmup = (spec[4], spec[5]) if len(spec) > 5 else model.window(case)
            label = f"single_{short_case(case)}" + ("" if mode == "off" else f"_{mode}")
            items.append({"kind": "single_ratio", "label": label, "case": case, "slabs": 1, "pairs": pairs,
                          "steps": steps, "warmup": warmup, "mode": mode,
                          "seconds": pairs * 2 * model.run(case, 1, steps, cold=False) + pairs * 30.0})
        elif kind == "single_control":
            # revision 15: two v7_single runs on card 0 under the --no-defrag protocol, with and without the bootstrap defrag
            case = spec[1]
            steps, warmup = (spec[2], spec[3]) if len(spec) > 3 else model.window(case)
            items.append({"kind": "single_control", "label": f"single_{short_case(case)}_bd", "case": case, "slabs": 1,
                          "steps": steps, "warmup": warmup,
                          "seconds": 2 * model.run(case, 1, steps, cold=False) * 1.1 + 30.0})
        else:
            raise SystemExit(f"sequence item of unknown kind: {spec}")
    return items


def job_cases(items: list[dict]) -> list[str]:
    cases = {BRINGUP_CASE}
    for item in items:
        if item["kind"] in ("point", "cross", "pair", "lock"):
            point = item["point"]
            cases.add(point.case)
            for kind in point.references:
                cases.add(reference_spec(point, kind)[0])
        elif "case" in item:
            cases.add(item["case"])
    return sorted(cases)


def emit_item(item: dict, model: Model) -> list[str]:
    if item["kind"] == "precheck":
        return [f"e7_precheck {item['case']} {item['reference_kind']} {item['parties']} {item['steps']} "
                f"{timeout(item['seconds'])}"]
    if item["kind"] == "pair":
        point = item["point"]
        seconds = pair_seconds(point, model, item["trials"], item["arm"])
        steps, warmup = model.window(point.case)
        references = " ".join(f"{kind}:{timeout(seconds['reference_runs'][kind])}" for kind in point.references)
        return [f"e7_pair_point {point.family} {point.case} {point.slabs} {item['trials']} {steps} {warmup} "
                f"{timeout(seconds['calibration'])} {timeout(seconds['run'])} {item['arm']} {references}"]
    if item["kind"] == "lock_begin":
        return [f"if e7_clock_lock {item['mhz']}; then"]
    if item["kind"] == "lock_end":
        return ["    e7_clock_unlock", "fi"]
    if item["kind"] in ("point", "cross", "lock"):
        point = item["point"]
        seconds = point_seconds(point, model)
        steps, warmup = model.window(point.case)
        references = " ".join(f"{kind}:{timeout(seconds['reference_runs'][kind])}" for kind in point.references)
        arguments = (f"{point.family} {point.case} {point.slabs} {point.trials} {steps} {warmup} "
                     f"{timeout(seconds['calibration'])} {timeout(seconds['run'])} {timeout(seconds['traced'])} "
                     f"{point.traces} {references}")
        if item["kind"] == "cross":
            return [f"e7_cross {item['original_line']} {arguments}"]
        if item["kind"] == "lock":
            return [f"    E7_PREFIX=lock_ e7_point {arguments}"]
        if point.weights_from:      # revision 10: the given weights file instead of a calibration, prefixed labels
            return [f"E7_PREFIX={point.prefix} E7_WEIGHTS_FROM={point.weights_from} "
                    f"E7_WEIGHTS_SHA256={point.weights_sha256} e7_point {arguments}"]
        return [f"e7_point {arguments}"]
    if item["kind"] == "tracegroup":
        return [f"e7_trace_group {item['label']} {item['case']} {item['parties']} {item['steps']} {item['warmup']} "
                f"{timeout(item['seconds'])}"]
    if item["kind"] == "single_ratio":
        mode = item.get("mode", "off")
        return [f"e8_single_ratio {item['label']} {item['case']} {item['pairs']} {item['steps']} {item['warmup']} "
                f"{timeout(item['seconds'])}" + ("" if mode == "off" else f" {mode}")]
    if item["kind"] == "single_control":
        return [f"e8_single_control {item['label']} {item['case']} {item['steps']} {item['warmup']} {timeout(item['seconds'])}"]
    return [f"e7_extra {item['kind']} {item['label']} {item['case']} {item['slabs']} {item['steps']} "
            f"{item['warmup']} {timeout(item['seconds'])}"]


def job_name(batch: int, line: str) -> str:
    if REVISION and batch in REVISION.get("job_names", {}):
        name = REVISION["job_names"][batch]
        return name[line] if isinstance(name, dict) else name
    return f"b{batch}{line}"


def job_estimate(items: list[dict], model: Model) -> tuple[float, int]:
    """(estimated seconds, time limit in hours = 2 x the estimate + 1 h, rounded up) of one job."""
    estimate = model.job_overhead + sum(item["seconds"] for item in items)
    return estimate, int(math.ceil((2.0 * estimate + 3600) / 3600.0))


def job_script(name: str, line: str, items: list[dict], model: Model, description: str, source_repository: str,
               exports: list[tuple[str, str]], content: str = None) -> str:
    """One sbatch script: job e7_<line> (singleton per line; a revision's job_tag makes it e7_<line><tag>, its own
    singleton), the revision's extra sbatch directives, output e7_<name>_<id>.out, the EXPORTS as
    `export VARIABLE=${VARIABLE:-VALUE}` before the source line (a value set at submission wins), the source line
    defaulting to SOURCE_REPOSITORY, e7_job_begin, one line per item, e7_job_end."""
    estimate, limit = job_estimate(items, model)
    if content is None:
        content = ", ".join(sorted({(item["point"].family if "point" in item else item["kind"]) for item in items}))
    job_tag = REVISION.get("job_tag", "") if REVISION else ""
    prefix = REVISION.get("job_prefix", "e7") if REVISION else "e7"
    directives = [f"#SBATCH {option}" for option in (REVISION.get("sbatch", []) if REVISION else [])]
    body = [
        "#!/bin/bash",
        "#SBATCH -p hp_5090",
        "#SBATCH -A hp5090",
        "#SBATCH -N 1",
        f"#SBATCH --gpus={REVISION.get('gpus', 8) if REVISION else 8}",
        f"#SBATCH --time={limit:02d}:00:00",
        f"#SBATCH -J {prefix}_{line}{job_tag}",
        "#SBATCH --dependency=singleton",
        *directives,
        f"#SBATCH -o /data/run01/scxm138/logs/{prefix}_{name}_%j.out",
        f"# {description}; do not edit by hand).",
        f"# Estimate {estimate / 3600:.2f} h, {len(items)} items: {content}. The protocol is e7_lib.sh's.",
    ]
    body += [f"export {variable}=${{{variable}:-{value}}}" for variable, value in exports]
    body += [
        'source "${E30_REPO:-' + source_repository + '}/docs/cluster_v6/scripts/e7_lib.sh" '
        '|| { echo "ABORT: no e7_lib.sh"; exit 2; }',
        f"e7_job_begin {name} {line} {' '.join(job_cases(items))}",
    ]
    for item in items:
        body += emit_item(item, model)
    body.append("e7_job_end")
    return "\n".join(body) + "\n"


def emit_jobs(jobs: dict, model: Model, directory: pathlib.Path, batches, source_repository: str = SOURCE_REPOSITORY,
              exports: list = ()) -> list[pathlib.Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for (batch, line), items in jobs.items():
        if batch not in batches or not items:
            continue
        name = job_name(batch, line)
        description = ((REVISION.get("description") if REVISION else None)
                       or f"E7 full campaign, batch {batch}, line {line} (generated by docs/cluster_v6/scripts/e7_plan.py --emit-jobs")
        description = f"{description}, line {line}" if REVISION and REVISION.get("description") else description
        path = directory / f"{(REVISION.get('job_prefix', 'e7') if REVISION else 'e7')}_{name}.sbatch"
        path.write_text(job_script(name, line, items, model, description, source_repository, list(exports)),
                        encoding="utf-8", newline="\n")
        written.append(path)
    return written


def validation_items(model: Model) -> list[dict]:
    """The validation job's items: the full-protocol points (rc2), then the A/B pair points."""
    items = []
    for family, case, slabs, references in VALIDATION["points"]:
        point = Point(family, case, slabs, list(references))
        items.append({"kind": "point", "point": point, "seconds": point_seconds(point, model)["total"]})
    for family, case, slabs, references in VALIDATION["pairs"]:
        point = Point(family, case, slabs, list(references))
        trials, arm = VALIDATION["pair_trials"], VALIDATION["pair_arm"]
        items.append({"kind": "pair", "point": point, "trials": trials, "arm": arm,
                      "seconds": pair_seconds(point, model, trials, arm)["total"]})
    return items


def emit_validation(model: Model, directory: pathlib.Path, tree: str, commit: str) -> tuple[pathlib.Path, list[dict]]:
    """docs/cluster_v6/e7_jobs/e7_v1.sbatch: job e7_V (line V: no pre-checks, no refmode of another line), run from
    the rc2 deployment (its tree TREE and COMMIT prefix exported: the job stops unless both match), v7-rc1 at
    E7_RC1_REPO (read only)."""
    items = validation_items(model)
    name, line = VALIDATION["job"], VALIDATION["line"]
    content = ", ".join(f"{item['point'].family} {item['point'].case} K={item['point'].slabs}"
                        + (f" A/B rc1-{VALIDATION['solver_tag']}" if item["kind"] == "pair" else "")
                        for item in items)
    description = (f"E7 validation job {name}, line {line}: {VALIDATION['solver_tag']} (this deployment) against "
                   "v7-rc1 (E7_RC1_REPO, read only) (generated by docs/cluster_v6/scripts/e7_plan.py --emit-validation")
    exports = [("E30_EXPECT_COMMIT", commit), ("E7_SOLVER_TAG", VALIDATION["solver_tag"]),
               ("E7_EXPECTED_TREE", tree), ("E7_RC1_REPO", VALIDATION["rc1_repository"])]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"e7_{name}.sbatch"
    path.write_text(job_script(name, line, items, model, description, VALIDATION["source_repository"], exports,
                               content), encoding="utf-8", newline="\n")
    return path, items


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 batches: job list, node-hour budget and job scripts")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", default=None, help="write points, jobs, totals and the model inputs as JSON")
    parser.add_argument("--emit-jobs", default=None, metavar="DIR", help="write one sbatch script per batch and line")
    parser.add_argument("--batch", type=int, action="append", default=None,
                        help="with --emit-jobs: only these batches (default all)")
    parser.add_argument("--model-json", default=None, metavar="FILE",
                        help="replace model fields (a JSON object, e.g. e7_refit.py --out-model); default the plan's")
    parser.add_argument("--source-repo", default=SOURCE_REPOSITORY, metavar="DIR",
                        help=f"with --emit-jobs: the source line's default deployment (default {SOURCE_REPOSITORY})")
    parser.add_argument("--solver-tag", default=None,
                        help="with --emit-jobs and --expected-tree: exported as E7_SOLVER_TAG (e.g. rc2)")
    parser.add_argument("--expected-tree", default=None, metavar="SHA",
                        help="with --emit-jobs and --solver-tag: exported as E7_EXPECTED_TREE (git rev-parse "
                             "<commit>:experiment/v7)")
    parser.add_argument("--expect-commit", default=None, metavar="PREFIX",
                        help="with --emit-jobs: exported as E30_EXPECT_COMMIT (the deployment's COMMIT prefix)")
    parser.add_argument("--emit-validation", default=None, metavar="DIR",
                        help="write the validation job e7_v1.sbatch (job e7_V, line V) into DIR")
    parser.add_argument("--validation-tree", default=PLACEHOLDER_TREE, metavar="SHA",
                        help="the validation job's expected experiment/v7 tree (git rev-parse <rc2>:experiment/v7)")
    parser.add_argument("--validation-commit", default=PLACEHOLDER_COMMIT, metavar="PREFIX",
                        help="the validation job's expected COMMIT prefix (the rc2 commit)")
    parser.add_argument("--revision", type=int, choices=sorted(REVISIONS), default=None, metavar="N",
                        help="a revision of the plan (REVISIONS): 1 = 2026-10-12 noon (arm C only, 3 trials, the trimmed "
                             "batches 2 and 4), 2 = 2026-10-12 evening (2-D only, 1500-step windows, batch 2 = items "
                             "A-E), 3 = the 2026-10-13 supplement b2S, 4 = the second supplement b2T (2d_n8000 K = 3 / "
                             "5 / 6 / 7), 5 = the third (b2U), 6 = the fourth (b2V strong, b2W weak; --batch 8 "
                             "--batch 9), 7 = the weak item again (b2X on wqd10nbj04g2, fallback b2Y; --batch 10 "
                             "--batch 11), 8 = the traced K = 1 reference groups for tab:phases (b2Z; --batch 12), 9 = the "
                             "same groups plus the four K = 8 chains traced on the same node, any 575 W node (b2AA; "
                             "--batch 13), 10 = 2d_n2840_k5 K = 5 again, calibrated and with b2Y's weights file, both "
                             "traced (b2AB; --batch 14), 11 = E8 phase 2 (the 3-D cubic cavity, jobs e8_A / e8_B; --batch 15); emitted jobs "
                             "export E7_ARMS")
    arguments = parser.parse_args()
    if (arguments.solver_tag is None) != (arguments.expected_tree is None):
        parser.error("--solver-tag and --expected-tree go together (the tag names the tree's solver)")
    global ARMS, REVISION
    if arguments.revision:
        REVISION = REVISIONS[arguments.revision]
        ARMS = REVISION["arms"]
    model = load_model(arguments.model_json)
    points = campaign_points()
    jobs = build_jobs(points, model, arguments.revision)

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
                lines.append(f"| {batch} | {line} | {'cross ' if point.cross else ''}"
                             f"{'lock ' if item['kind'] == 'lock' else ''}{point.family}"
                             f"{' (weights ' + point.prefix + pathlib.Path(point.weights_from).name + ', no calibration)' if point.weights_from else ''} | "
                             f"{point.case} | {point.slabs} | {PARTICLES[point.case] / 1e6:.1f}M | "
                             f"{point.per_card / 1e6:.2f}M | {point.trials} | {reference} | "
                             f"{seconds['run'] / 60:.1f} | {seconds['reference'] / 60:.1f} | "
                             f"{seconds['calibration'] / 60:.1f} | {seconds['traced'] / 60:.1f} | "
                             f"{seconds['total'] / 3600:.2f} |")
                records.append({"batch": batch, "line": line, "job": job_name(batch, line), "family": point.family,
                                "case": point.case, "slabs": point.slabs, "particles": PARTICLES[point.case],
                                "per_card": point.per_card, "dimension": point.dimension, "trials": point.trials,
                                "traces": point.traces, "cross": point.cross, "references": point.references,
                                **({"prefix": point.prefix, "weights_from": point.weights_from,
                                    "weights_sha256": point.weights_sha256} if point.weights_from else {}),
                                "window": list(model.window(point.case)),
                                "fps_estimate": 1.0 / model.step_time(point.case, point.slabs),
                                **{name: round(value, 1) for name, value in seconds.items()
                                   if not isinstance(value, dict)}})
            elif "case" in item:
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
    planned = [item["point"] for items in jobs.values() for item in items if item["kind"] == "point"]
    summary = [f"points: {len(planned)} planned of {len(points)} in the campaign (line A {sum(p.line == 'A' for p in planned)}, "
               f"line B {sum(p.line == 'B' for p in planned)}); 5 trials: {sum(p.trials == 5 for p in planned)} points "
               f"(<= {LIGHT_PER_CARD / 1e6:.1f}M per card), 3 trials: {sum(p.trials == 3 for p in planned)}; arms {ARMS}; "
               f"traced runs per point {sorted({p.traces for p in planned})}; timing runs without the seam check"
               + (f" (revision {arguments.revision} of 2026-10-12)" if arguments.revision else ""),
               "batches (node-h; wall = the longer line): " + "; ".join(
                   f"{batch}: A {values.get('A', 0.0) / 3600:.1f} + B {values.get('B', 0.0) / 3600:.1f} = "
                   f"{sum(values.values()) / 3600:.1f} (wall {max(values.values()) / 3600:.1f} h)"
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
        exports = []
        if arguments.expect_commit:
            exports.append(("E30_EXPECT_COMMIT", arguments.expect_commit))
        if arguments.solver_tag:
            exports += [("E7_SOLVER_TAG", arguments.solver_tag), ("E7_EXPECTED_TREE", arguments.expected_tree)]
        if arguments.revision:
            exports.append(("E7_ARMS", REVISION["arms"]))
            exports += [tuple(item) for item in REVISION.get("exports", [])]
        for path in emit_jobs(jobs, model, pathlib.Path(arguments.emit_jobs), set(arguments.batch or sorted({batch for batch, _ in jobs})),
                              arguments.source_repo, exports):
            print(f"wrote {path}")
    if arguments.emit_validation:
        path, items = emit_validation(model, pathlib.Path(arguments.emit_validation), arguments.validation_tree,
                                      arguments.validation_commit)
        estimate, limit = job_estimate(items, model)
        print(f"\nvalidation job e7_{VALIDATION['job']} (line {VALIDATION['line']}, job name e7_{VALIDATION['line']}): "
              f"estimate {estimate / 3600:.2f} h = {estimate:.0f} s (job overhead {model.job_overhead:.0f} s), "
              f"time limit {limit} h; tree {arguments.validation_tree}, commit {arguments.validation_commit}")
        for item in items:
            point = item["point"]
            if item["kind"] == "pair":
                seconds = pair_seconds(point, model, item["trials"], item["arm"])
                text = (f"e7_pair_point ab_{point.case.replace('cavity', '')}_K{point.slabs}: {item['trials']} trials "
                        f"x (reference sets + rc1 + {VALIDATION['solver_tag']}), arm {item['arm']}")
            else:
                seconds = point_seconds(point, model)
                text = (f"e7_point {point.case.replace('cavity', '')}_K{point.slabs}: {point.trials} trials "
                        f"x (reference sets + E + C) + {point.traces} traced runs")
            print(f"  {text}: {seconds['total'] / 60:.1f} min (calibration {seconds['calibration'] / 60:.1f}, "
                  f"run {seconds['run'] / 60:.1f}, reference set {seconds['reference'] / 60:.1f} / first "
                  f"{seconds['reference_first'] / 60:.1f} min)")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
