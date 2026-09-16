# Inter-GPU phase-offset trace (local 2× RTX 5090, 2-D 4M, K=2 and K=4) — 2026-09-16

Question: the N56 K sweep showed the two directions of a K=2 link with ~25 ms different transport slack, so
the chain may run with a systematic phase offset between GPUs — a candidate mechanism for the light-load
run-to-run bimodality. This measures the offset frame by frame.

## Instrument

`experiment/v5/utils/phase_trace_v5.py` + chain bench `--phase-trace DIR` (+ `plot_phase_trace.py`):

- every frame, every sim: GPU timestamps of phase A start/end, B start/end, C start/end (existing
  BenchTimer ticks; the phase-C parity regions survive two frames, the A/B slots are caught on 89–100% of
  frames and fall back to c_end(n−1) otherwise);
- all GPUs on one clock through `VK_KHR_calibrated_timestamps` (device tick ↔ QueryPerformanceCounter
  pairs every 100 frames; one linear map per sim fitted afterwards: the two 5090 counters run 10–13 ppm
  fast vs QPC and 3 ppm apart from each other; fit residual 8–35 µs rms, i.e. the phase resolution);
- hook `ChainOrchestratorV5.on_frame_done(frame, sim)` called from the global loop's `_wait_frame` and from
  the readiness scheduler; ~100 µs of host time per frame per sim, no change to submission or sync.

Runs: `cases/lid_driven_cavity_2d_4m` (2M per GPU), 3000 steps, warmup 1000, cascade + band + fast submit +
full host stack (the curve-job configuration), depth 2. All drift 0, stamps 0.

| run | loop | GPUs | steady fps | period T | sim1 − sim0 phase A start (frames ≥ 1000) |
|---|---|---|---|---|---|
| k2_r1 | global | 0, 1 | 233.5 | 4.194 ms | mean +0.66 ms, std 2.44; sawtooth (ramps, ceiling +2.5 ms = 0.72 T, slips of −T) |
| k2_r2 | global | 0, 1 | 235.6 | 4.202 ms | mean +1.60 ms, std 1.42; ramps, then locked at +3.0 ms (0.72 T) for the last 900 frames |
| k2_ready | ready scheduler | 0, 1 | 228.0 | 4.193 ms | mean −2.46 ms, std 0.76; locked at −0.58 T, drift 0.15 ms per 1000 frames |
| k4_r1 / r2 | global | 0, 1, 0, 1 (two sims per GPU) | 191.4 / 190.7 | 5.03 ms | same-GPU pairs 0.4–0.6 T apart (time-sharing); cross-GPU offset flips between plateaus ≈ +1.5 and −2.5 ms in blocks of ~300 frames |

Figures: `phase_trace_4m_k2_r1.png`, `_k2_r2.png`, `_k2_ready.png`, `_k4_r1.png`, `_k4_r2.png` (panels:
offset in ms with ±T/2 and ±T lines; offset as a phase fraction; per-frame waiting; per-sim period).

## What the traces show

1. **The offset is systematic, not random.** GPU 0 is intrinsically 0.8% faster (period median 4.163 vs
   4.200 ms in r1, 4.187 vs 4.209 in r2). With the global loop the offset Δ = A1 − A0 therefore ramps
   linearly at ≈ +35 µs per frame until it hits a **ceiling at Δ ≈ +2.5 to +3.1 ms = 0.72 T** in both runs,
   where the faster GPU's phase C is gated by the slower GPU's ghost upload (its C(n) cannot start before
   A1(n) + readback + host copy + upload). At the ceiling the fast GPU waits a little every frame; every few
   hundred frames it stalls a whole period once (a "slip", Δ jumps by −T) and the ramp restarts. The chain is
   never phase-aligned; it lives on a sawtooth between 0 and 0.72 T.
2. **Two identical runs took different trajectories** (r1 mostly ramping, r2 locked at the ceiling for the
   last 900 frames), i.e. the phase state is a run-to-run variable. At this load its cost is small: waiting
   per frame (A→C span above its floor) sim0 0.14 / 0.11 ms, sim1 0.10 / 0.04 ms — 2–3% of the period,
   and the two runs differ by 1% in fps. The slack window here is wide (phase B 3.5 ms vs chain ≈ 1.3 ms).
3. **The readiness scheduler holds a fixed offset** (−0.58 T, std 0.16 T, no drift) but does not remove the
   waiting: 0.21 / 0.20 ms per frame (p90 0.7 ms), fps 228 vs 234–236. Without the global barrier the fast
   GPU runs ahead until the ghost dependency stops it, every frame, instead of every few hundred frames.
4. **K=4 on two GPUs is not a chain measurement**: the two sims sharing a GPU time-share it (offset 0.4–0.6 T
   between them, "waiting" = the other sim's execution). The K=4 trace needs four real GPUs (N56 node).

## Consequence for the fix

The period mismatch between cards (0.8% here; 1–4% between N56 cards) is the driver: the faster card has
to lose exactly that much time per frame somewhere, and where it loses it (ceiling waits vs whole-period
slips vs the ready scheduler's every-frame waits) is what changes between runs. Two fixes have a basis now:

- **rate-match instead of gate**: give the fast card that much more work (per-card weights from measured
  single-GPU periods, V3.1 static rebalancing) so the phase offset stops ramping — the chain then runs at
  a fixed offset with no slips;
- **period alignment**: a per-frame pacing target at the slowest card's period (submit-side) so the fast card
  never runs into the ceiling.

Whether the light-load bimodality (N56 2M/GPU K=2: 119 / 245 / 179 fps) is the same phenomenon is not
proven here: the local 2M/GPU K=2 runs were not bimodal (233 / 236 fps) because the slack window is wide.
The trace should be run on the N56 node at 2M/GPU with K=2 and K=8 (real GPUs, narrow slack) — the
instrument is ready and costs nothing at run time.

## Side observation

Frames 0–1000 run at a 21–22 ms period (5× the steady 4.2 ms) in every 4M run, dropping to 4.2 ms at the
first defrag (frame 1000). The anatomy confirms it is the particle order: at f1000 (last frame before the
first defrag) phase B = 23.7 ms (correction_interior 9.45, density 7.77, force 6.44 ms) vs 3.5 ms at f2000
(1.03 / 1.08 / 1.41) — the interior kernels are 5–9× slower on the generator's initial order, and
`bootstrap_all` runs no defrag (the 1M case happens to be generated in a voxel-friendly order and does not
show it). A defrag at bootstrap (or a first cadence of a few frames) would cut the warmup — and the
reference-run cost on the cluster (the 64M K=1 references spend ~50% of their wall time in the first 1000
frames) — by up to 2×. Not changed here.
