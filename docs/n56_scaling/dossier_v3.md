<!-- exported from docs/n56_scaling/dossier_v3.html; figures in the fig/ folder of the bundle -->
N56 Scaling Dossier III

 Paper-writing dossier, third edition · multi-GPU δ-plus WCSPH · Vulkan · Paratera N56 8× RTX 5090 · 2026-09-14 → 2026-09-17

# N56 Scaling Dossier III

Every number, figure and table needed to write the scaling section: 2-D and 3-D strong scaling (vs size and fixed-N K sweeps), weak scaling, the two-regime hiding-margin figure with its model, the 64M K=8 loss ledger, frame anatomy vs K, communication volumes predicted and measured, the cube boundary point, mechanisms (phase drift, band-kernel occupancy), correctness evidence and job-level provenance. 写作用完整版：图清单 §1，方法 §2，全部表 §3，机理 §4，正确性 §5，溯源 §6，写作注意事项 §7。

 code **eda4b8f** (tag **freeze-2026-09-17**; probe34–36 ran at **367706a**, identical solver path)]
 cluster **N56R5** · queue **hp_5090** · nodes **wqd10nbj04g2** (575 W) · **wqd10nbm13g5** (600 W)]
 solver **experiment/v5** · lid-driven cavity · c = 100 · γ = 7 · h = 4 dx (support radius) · Wendland C4 · leapfrog]

## At a glance

- 2-D strong K=8, 8M/GPU (64M)91.4 ± 0.3%reproduced 91.4 ± 0.9% in a second job

- 2-D strong K=8, 16M/GPU (128M)95.4 ± 0.2%reference = 4 simultaneous K=2 pairs

- 3-D strong K=8, 8.8M/GPU (64M stretched)92.8 ± 0.3%K=2 98.0, K=4 96.1

- 3-D weak K=8, 8M/GPU96.7 ± 0.3%flat 96–97% for K = 2, 4, 8

- 2-D weak K=8, 32M/GPU (256M)95.9 ± 0.4%17.4 fps on 256M particles

- 64M K=8 loss, 2-D / 3-D8.6% / 7.2%≈ 4–5% per-frame floor + ≈ 3–4% chain gaps, both dimensions

One node, host-staged transport (consumer GeForce has no P2P), 1-D slab decomposition with a one-voxel halo and cascaded interior/boundary kernels. Transport is fully hidden at ≥ 8M particles per GPU in 2-D and at every 3-D point measured; the residual is a per-frame floor that does not shrink with K plus the gaps of a chain whose cards drift in phase. Below ≈ 4M particles per GPU in 2-D the chain is transport-bound and, at 2–4M/GPU, run-to-run bimodal.

## 1. Paper figure list

 Files: `docs/n56_scaling/*.png` with the generating scripts in `docs/n56_scaling/scripts/` and the per-figure JSON tables next to them. All efficiencies follow the standardized protocol of §2 (references on the participating GPUs, simultaneously; ≥ 3 trials; error bar = std of the trial-wise η). Every run in every figure reported drift 0 and stamp errors 0.

### F1 — Strong scaling vs problem size, 2-D, K=8

**F1.** η_strong at K=8 vs particles per GPU (4M → 128M total). Three trials each; references on all 8 GPUs at once (4× K=2 pairs at 128M). Data: `scripts/plot_curve34.py`, jobs probe34 1592849/50/51. Table T1.

*Figure file: fig/eta_strong_vs_size_k8.png*

### F2 — Fixed-N K sweep, 2-D and 3-D on one axis

**F2.** η_strong vs K at fixed N: 2-D 64M / 32M (probe36) and 3-D stretched 64M / 32M (probe37); hollow diamond = 2-D 64M K=2 on the cross-NUMA GPU pair (3, 4). The 2-D 32M K=8 point (4.06M/GPU, 67.4%) is where the transport stops being hidden. Data: `ksweep_table.json`, `matrix_3d_table.json`; `scripts/plot_paper_figs.py`. Tables T2, T3.

*Figure file: fig/eta_fixed_n_ksweep_2d_3d.png*

### F3 — Weak scaling, 2-D five loads and 3-D two loads

**F3.** η_weak vs K. 2-D: 2M, 4M, 8M, 16M, 32M per GPU (probe35, dashed); 3-D: 4M and 8M per GPU (probe37, solid). Reference = the K=1 case on all 8 GPUs simultaneously; η_weak(K) = fps_K / mean of the reference over GPUs 0..K−1. Table T4.

*Figure file: fig/eta_weak_2d_3d.png*

### F4 — Two-regime figure: efficiency vs transport hiding margin, all 15 strong points

**F4.** Left: η vs r = t_transport / T_B of the binding link (log axis; p50 over sampled frames). r > 1: phase C waited on most frames (2-D 4M / 16M K=8); r ≈ 0.75: partly exposed (2-D 32M K=8); r < 0.65: hidden. Right: the twelve hidden points against model B, excess = α + β(K−1) + γ·T_ideal (filled = measured, hollow = model). Data: `hiding_margin_points.json`, `hiding_margin_fit.json`; `scripts/plot_hiding_margin.py`. Table T5.

*Figure file: fig/eta_vs_hiding_margin.png*

### F5 — 64M K=8 loss ledger, 2-D and 3-D side by side (table T6)

A table, not a plot: see T6 in §3. Both dimensions lose ≈ 4–5% of the frame to a floor that does not scale with K (band kernels, ghost_send / install / barriers, the scratch→primary copy) and ≈ 3–4% to chain gaps.

### F6 — Frame anatomy vs K, 2-D and 3-D

**F6a (2-D).** 64M per-GPU frame vs K (K=1 reference, K=2/4/8): phase A, B, C, gaps, idle; right: how long before phase C each neighbour's upload landed. probe36, `scripts/plot_ksweep36.py`.

*Figure file: fig/anatomy_vs_k_64m.png*

**F6b (3-D).** 64M and 32M stretched: top = whole frame, bottom = everything except phase B, with the gaps split into c→a (queue ran dry: the global loop gated on the slowest card) and a→b + b→c (waiting for the neighbours' uploads). probe37, `scripts/plot_3d_anatomy.py`. Table T7.

*Figure file: fig/anatomy_vs_k_3d.png*

### F7 — Communication volume, predicted vs measured (table T8)

Table: bytes per link per frame from the halo geometry (one voxel column × 140 B + the two voxel-list segments) against the pre-run analytic sizing and the measured DMA / host-copy times.

### F8 — Cube boundary point (table T9)

401³ cube at K=8 (64% of each slab inside the force band) against the stretched 64M domain: same fps, phase C 3.3× larger, transport slack 17 / 5 ms instead of 56 / 44 ms, 2.8 ms b→c wait per frame — the edge of the hiding window.

### Supplementary figures

**S1.** 3-D matrix as measured (strong left, weak right) with η_min (dashed): the per-GPU reference spread is 1–2%.

*Figure file: fig/eta_3d_matrix.png*

**S2.** 2-D fixed-N K sweep with the throughput panel (steps per second vs K against K × mean single-GPU rate).

*Figure file: fig/eta_strong_vs_k.png*

**S3.** 2-D weak scaling, the five families alone (probe35), with η_min.

*Figure file: fig/eta_weak_vs_k.png*

**S4.** Mechanism: per-frame phase offset between the two GPUs of a K=2 chain (local 2× 5090, 2-D 4M): the 0.8% faster card drifts ahead 35 µs per frame to a ceiling at 0.72 T where its phase C is gated by the slower card's upload, then slips one period. §4.

*Figure file: fig/phase_trace_4m_k2_r2.png*

**S5.** The host-side optimization campaign at 2-D 64M K=8 (legacy → cascade → band dispatch → fast submit → no-wait → readiness scheduler): 64.1 → 65.5 fps; CPU submit time 3.5 → 0.5 ms per frame; the inter-phase gaps only move.

*Figure file: fig/cpu_path_campaign_64m_k8.png*

**S6.** Cascading force (force_deep_interior in phase B) and band-voxel dispatch A/B at 64M K=8.

*Figure file: fig/cascade_band_64m_k8.png*

**S7.** Per-particle verifier: acceleration difference by voxel-column distance to the seam, noise pairs vs test pair (cascading force vs legacy, 1M K=2, 300 steps).

*Figure file: fig/verify_cascade_1m_k2_300.png*

## 2. Methods (as run)

### Efficiency definition (standardized 2026-09-15)

- **η_strong(N, K) = fps_K(N) / (K · mean_i<K fps_1(N; GPU i))** — the K=1 reference measured on *every GPU that takes part*, on all of them *simultaneously* (same power and thermal state), same case, same window. Clocks cannot be locked (no root; all cards at the node's power cap, busy SM clocks 2640–2780 MHz, recorded as a covariate every 5 s).

- **η_weak(K) = fps_K(K·n) / mean_i<K fps_1(n; GPU i)** with the K=1 case on all 8 GPUs at once as the reference.

- **η_min** = fps_K / (K · min_i fps_1): an equal-slab chain cannot beat its slowest card; η_mean − η_min is the hardware-spread term.

- **Window**: chain bench 3000 steps, steady = last 2000 (2-D, probe34–36); 1500 steps, steady = last 1000 (3-D, probe37) — reference and K-run share the window. The bootstrap defrag (2026-09-17) removes the slow first-1000 phase; before it, warmup frames ran on the generator's unsorted order (5× slower in 2-D 4M).

- **Trials**: 3 alternating (reference set, K-run) in 2-D; one 8-GPU reference per N and 3 K-runs in 3-D. Error bar = std over trials of the trial-wise η. When K=1 does not fit a 32 GB card (2-D 128M, 256M), the reference is K=2 pairs on all GPUs with factor K/2.

- **fps** = steady steps / steady seconds from each run log (the RESULT line's one-decimal fps is a 3% quantum at 1.6 fps).

### Cases

- 2-D: `lid_driven_cavity_2d_{4m,16m,32m,64m,128m}` (strong), `cavity_weak{,4,8,16,32}_k{K}_{N}m` (weak, per-GPU 2/4/8/16/32M). Walls 1% of the count.

- 3-D: stretched domains of K cubes joined along x (8M/GPU: 1609×201×201 dx = 70.6M incl. walls; 4M/GPU: 1273×159×159 = 35.7M), the same geometry serving as fixed-N strong case and as the weak family's K=8 point; cube 401³ (68.4M) as the slab-in-cube control. h = 4 dx (support radius, ≈ 268 neighbours), voxel edge = h, 128 slots per voxel, wall shell 4 layers (one support radius; verified equivalent to 9 layers on 8M K=1 and the K=2 stretched case).

### Runtime configuration of every curve job

Cascading force + band-voxel dispatch (Phase C over band voxels only) + fast cffi submission + host transport stack (count-aware worker copies, workers pinned to the destination GPU's NUMA node, split readback/upload transfer queues, 0.2 ms GIL switch interval, ghost pool ¼ in 2-D / 1.0 in 3-D), global frame loop, pipeline depth 2, pool safety 1.2, per-direction timelines. Switches are exported explicitly by every job script.

### Instruments

- **Anatomy**: GPU timestamps per kernel and phase on the compute and transfer queues (parity regions keep the previous frame's phase-C ticks so the cross-frame c→a gap is measurable); one sampled frame per defrag boundary per GPU (f2000 + f3000 in 2-D, f1000 in 3-D).

- **Hiding margin**: per link, slack = c_start − upload_end (clamped at 0 by the semaphore); per frame the binding link's margin (min slack − b→c gap) / T_B; t_transport = T_B − margin·T_B.

- **Phase trace** (local): every frame's A/B/C timestamps of every GPU on one host clock via VK_KHR_calibrated_timestamps.

- **Per-particle verifier**: four runs (2 + 2), KD-tree matching, field differences and acceleration difference by column distance to the seam vs the run-to-run noise floor.

## 3. Tables

### T1 — 2-D strong scaling vs size, K=8 (probe34, 3 trials)

| N | per GPU | node | K=8 fps (trials) | reference (mean; per-GPU range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|---|
| 4M | 0.52M | wqd10nbm13g5 | 114.5 / 140.0 / 111.2 | K=1 140.8 (138.7–144.2) | 10.8% | 1.4 | 11.0% |
| 16M | 2.05M | wqd10nbm13g5 | 122.5 / 134.0 / 125.0 | K=1 36.0 (35.2–36.7) | 44.1% | 2.1 | 45.2% |
| 32M | 4.06M | wqd10nbm13g5 | 102.4 / 98.3 / 95.5 | K=1 18.3 (18.0–18.6) | 67.4% | 2.3 | 68.7% |
| 64M | 8.05M | wqd10nbj04g2 | 65.9 / 66.2 / 65.4 | K=1 9.01 (8.9–9.1) | 91.4% | 0.3 | 92.1% |
| 128M | 16.0M | wqd10nbm13g5 | 34.2 / 34.1 / 34.2 | 4× K=2 pairs 8.95 (8.9–9.0) | 95.4% | 0.2 | 96.0% |

The 32M K=8 point measured 72.7 ± 2.6% on wqd10nbj04g2 in probe36 (same configuration): ≤ 4M/GPU points reproduce only to ≈ 5 points across nodes/jobs; ≥ 8M/GPU to ±1.

### T2 — 2-D fixed-N K sweep (probe36, wqd10nbj04g2, 3 trials, references on the participating GPUs)

| N | K | GPUs | per GPU | K-run fps (trials) | reference mean per trial (per-GPU range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|---|---|
| 64M | 2 | 0, 1 | 32.2M | 17.8 / 17.8 / 17.8 | 9.10 / 9.05 / 9.00 (9.0–9.1) | 98.3% | 0.5 | 98.5% |
| 64M | 4 | 0–3 | 16.1M | 34.1 / 34.2 / 34.3 | 9.00 / 9.07 / 9.05 | 94.6% | 0.3 | 95.0% |
| 64M | 8 | 0–7 | 8.05M | 66.3 / 65.3 / 66.3 | 9.01 / 9.04 / 9.03 | 91.4% | 0.9 | 91.6% |
| 64M | 2 | 3, 4 (cross-NUMA) | 32.2M | 17.9 / 17.9 | 9.10 / 9.10 | 98.4% | 0.0 | 98.4% |
| 32M | 2 | 0, 1 | 16.3M | 35.4 / 35.3 / 35.2 | 18.2 / 18.2 / 18.2 | 97.0% | 0.3 | 97.0% |
| 32M | 4 | 0–3 | 8.1M | 67.0 / 67.0 / 67.1 | 18.1 / 18.1 / 18.1 | 92.6% | 0.1 | 93.3% |
| 32M | 8 | 0–7 | 4.06M | 101.4 / 105.5 / 108.9 | 18.07 / 18.10 / 18.09 | 72.7% | 2.6 | 73.4% |

### T3 — 3-D fixed-N K sweep, stretched domains (probe37, wqd10nbj04g2; one 8-GPU reference per N, 3 K-runs)

| case | K | per GPU | K-run fps (trials) | reference mean (per-GPU range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|---|
| 64M 1609×201×201 (70.6M) | 2 | 35.3M | 3.18 / 3.18 / 3.18 | 1.622 (1.621–1.624) | 98.0% | 0.1 | 98.0% |
| | 4 | 17.7M | 6.20 / 6.20 / 6.22 | 1.615 (1.597–1.624) | 96.1% | 0.2 | 97.2% |
| | 8 | 8.8M | 11.96 / 12.02 / 12.03 | 1.617 (1.597–1.627) | 92.8% | 0.3 | 94.0% |
| 32M 1273×159×159 (35.7M) | 2 | 17.9M | 6.19 / 6.23 / 6.17 | 3.191 (3.186–3.197) | 97.1% | 0.5 | 97.2% |
| | 4 | 8.9M | 11.98 / 12.00 / 12.09 | 3.196 (3.178–3.222) | 94.1% | 0.5 | 94.6% |
| | 8 | 4.5M | 23.50 / 23.56 / 23.40 | 3.202 (3.178–3.231) | 91.7% | 0.3 | 92.4% |

### T4 — Weak scaling (2-D probe35 on both nodes; 3-D probe37)

| family | K=2 | K=4 | K=8 | η_min at K=8 | note |
|---|---|---|---|---|---|
| 2-D 32M/GPU (32M … 256M) | 98.3 ± 0.3 | 97.3 ± 0.0 | 95.9 ± 0.4 | 97.2 | K=8 = 256M particles, 17.4–17.5 fps |
| 2-D 16M/GPU | 98.1 ± 0.2 | 97.2 ± 0.1 | 96.1 ± 0.3 | 98.3 | |
| 2-D 8M/GPU | 95.7 ± 0.3 | 94.1 ± 0.4 | 92.2 ± 0.4 | 94.0 | |
| 2-D 4M/GPU | 88.3 ± 8.8 | 79.1 ± 9.1 | 81.6 ± 5.4 | 82.4 | bimodal run to run |
| 2-D 2M/GPU | 63.5 ± 22.2 | 62.4 ± 10.8 | 53.3 ± 4.9 | 53.9 | K=2: 119 / 245 / 179 fps on identical configs; refs 281–288 |
| 3-D 8M/GPU (9.1M incl. walls) | 96.2 ± 0.1 | 96.0 ± 0.2 | 96.7 ± 0.3 | 97.6 | refs 12.30–12.50 fps |
| 3-D 4M/GPU (4.7M) | 95.4 ± 0.2 | 94.6 ± 0.2 | 95.3 ± 0.3 | 96.1 | refs 24.4–24.8 fps; no bimodality |

### T5 — Hiding margin of every strong point and the hidden-regime model

| point | η | T_B (ms) | slack p50 (ms) | b→c gap (ms) | r = t_transport / T_B | T_ideal (ms) | excess (ms) | model B η | residual (pts) |
|---|---|---|---|---|---|---|---|---|---|
| 2-D 4M K=8 | 10.8% | 0.82 | 0.02 | 4.99 | 7.09 | 0.89 | 7.33 | — | exposed |
| 2-D 16M K=8 | 44.1% | 3.07 | 0.08 | 2.88 | 1.25 | 3.47 | 4.40 | — | exposed |
| 2-D 32M K=8 | 67.4% | 6.13 | 3.30 | 1.54 | 0.75 | 6.87 | 3.32 | — | partly exposed (p10 margin −0.64) |
| 2-D 64M K=8 | 91.4% | 12.42 | 9.82 | 0.01 | 0.42 | 13.85 | 1.30 | 89.0% | +2.4 |
| 2-D 128M K=8 | 95.4% | 24.17 | 21.42 | 0.12 | 0.29 | 55.87 | 2.69 | 95.8% | −0.4 |
| 2-D 64M K=2 | 98.3% | 50.72 | 26.49 | 0.44 | 0.48 | 55.40 | 0.93 | 97.7% | +0.7 |
| 2-D 64M K=4 | 94.6% | 25.28 | 20.10 | 0.26 | 0.64 | 27.70 | 1.59 | 95.8% | −1.3 |
| 2-D 32M K=2 | 97.0% | 25.24 | 11.57 | 0.03 | 0.54 | 27.47 | 0.86 | 97.1% | −0.1 |
| 2-D 32M K=4 | 92.6% | 12.61 | 8.98 | 0.07 | 0.65 | 13.74 | 1.10 | 93.5% | −0.9 |
| 3-D 64M K=2 | 98.0% | 302.8 | 246.8 | 0.19 | 0.18 | 309.2 | 6.41 | 97.9% | +0.1 |
| 3-D 64M K=4 | 96.1% | 147.9 | 118.8 | 0.64 | 0.12 | 154.6 | 6.24 | 96.6% | −0.5 |
| 3-D 64M K=8 | 92.8% | 69.2 | 55.7 | 0.17 | 0.21 | 77.3 | 6.02 | 93.7% | −1.0 |
| 3-D 32M K=2 | 97.1% | 153.1 | 72.6 | 0.85 | 0.53 | 156.3 | 4.72 | 96.8% | +0.3 |
| 3-D 32M K=4 | 94.1% | 74.0 | 60.6 | 0.94 | 0.60 | 78.1 | 4.91 | 94.4% | −0.4 |
| 3-D 32M K=8 | 91.7% | 33.3 | 25.7 | 0.20 | 0.24 | 39.1 | 3.54 | 89.1% | +2.6 |

**Model fits** (twelve hidden points, excess = T_ideal (1/η − 1) in ms per frame): model A, excess = α + β(K−1): 2-D α 0.75 ms, β 0.18 ms per extra GPU; 3-D α 5.82, β −0.14 — 10 of 12 residuals exceed the error bars. Model B, excess = α + β(K−1) + γ·T_ideal: 2-D α 0.15 ms, β 0.19 ms, γ 1.75% of the frame; 3-D α 3.34 ms, β 0.15 ms, γ 1.06% — 8 of 12 exceed (residual rms 1.2 points against error bars of 0.1–0.5). Points off the model beyond their error bar: 2-D 64M K=8 (+2.4), 64M K=2 (+0.7), 64M K=4 (−1.3), 32M K=4 (−0.9), 128M K=8 (−0.4); 3-D 64M K=4 (−0.5), 64M K=8 (−1.0), 32M K=8 (+2.6). The structure: K=4 points fall below any smooth floor model (largest c→a gaps), the K=8 points at 4–9M per GPU sit above it. **The two regimes are the robust statement; the model gives only the two coarse constants (≈ 1–2% of the frame plus ≈ 0.2 ms per extra GPU in 2-D; ≈ 1% plus a 3.3 ms floor in 3-D).**

### T6 — 64M K=8 loss ledger, 2-D and 3-D (ms per frame per GPU)

| item | 2-D 64M (probe36, 8.05M/GPU) | 3-D 64M stretched (probe37, 8.8M/GPU) |
|---|---|---|
| ideal period = T_ref / 8 | 13.86 | 77.28 |
| measured period | 15.16 | 83.31 |
| **η = ideal / measured** | **91.4%** | **92.8%** |
| excess over ideal | 1.30 (8.6%) | 6.02 (7.2%) |
| phase A measured / at ideal | 1.02 / 0.80 | 1.24 / 0.83 |
| phase C measured / at ideal | 1.10 / 0.27 | 10.26 / 0.30 |
| phase B measured / at ideal | 12.42 / 12.74 | 69.18 / 76.22 |
| → band work moved from B to C (B deficit) | −0.32 | −7.04 |
| → A + C excess | +1.05 | +10.37 |
| **→ net per-frame floor** | **+0.73 (4.8%)** | **+3.33 (4.0%)** |
| gap c→a (queue dry) | 0.25 | 1.90 |
| gap a→b + b→c (upload wait) | 0.02 | 0.17 |
| other idle | 0.36 | 0.57 |
| **→ gaps + idle** | **+0.57 (3.8%)** | **+2.63 (3.2%)** |

### T7 — Frame anatomy vs K (ms per frame per GPU; means over GPUs and trials)

| case | K | period | ideal T_ref/K | A | B | C | c→a | a→b + b→c | other idle | B·K | slack p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2-D 64M | 1 | 110.8 | — | 6.37 | 101.9 | 2.19 | 0.01 | — | 0.33 | 101.9 | — |
| | 2 | 56.18 | 55.4 | 3.32 | 50.72 | 1.71 | 0.45 (all gaps) | | 0.00 | 101.4 | 26.5 |
| | 4 | 29.24 | 27.7 | 1.78 | 25.28 | 1.31 | 0.50 (all gaps) | | 0.37 | 101.1 | 20.1 |
| | 8 | 15.16 | 13.85 | 1.02 | 12.42 | 1.10 | 0.25 | 0.02 | 0.36 | 99.4 | 9.8 |
| 3-D 64M stretched | 1 | 618.3 | — | 6.60 | 609.8 | 2.41 | 0.01 | 0.01 | 0.00 | 610 | — |
| | 2 | 314.6 | 309.1 | 3.58 | 302.8 | 7.43 | 0.01 | 0.20 | 0.53 | 606 | 247 |
| | 4 | 161.1 | 154.6 | 2.07 | 147.9 | 9.26 | 1.38 | 0.64 | 0.00 | 591 | 119 |
| | 8 | 83.3 | 77.3 | 1.24 | 69.2 | 10.26 | 1.90 | 0.17 | 0.57 | 553 | 56 |
| 3-D 32M stretched | 1 | 312.3 | — | 3.31 | 307.7 | 1.22 | 0.01 | 0.01 | 0.11 | 308 | — |
| | 2 | 161.4 | 156.2 | 1.93 | 153.1 | 4.93 | 0.01 | 0.85 | 0.56 | 306 | 73 |
| | 4 | 83.1 | 78.1 | 1.18 | 74.0 | 6.20 | 0.37 | 0.94 | 0.47 | 296 | 61 |
| | 8 | 42.6 | 39.0 | 0.80 | 33.3 | 6.83 | 0.96 | 0.20 | 0.44 | 267 | 26 |

### T8 — Communication volume per link per frame: predicted vs measured

| case | pre-run estimate (3d_prep, border 9) | halo column particles (border 4 / as run) | SoA bytes (140 B each) | voxel-list segments | MB per link per frame | measured readback / upload DMA (µs) | host copy p50 (µs) | upload GB/s | node-wide at K=8 |
|---|---|---|---|---|---|---|---|---|---|
| 3-D 64M stretched | 26.9 MB | 174,724 | 24.46 | 2,809 vox × 129 × 4 B = 1.45 | **25.9** | 1326–2159 / 1182–1249 | 1782–2453 | ≈ 20 | 363 MB/frame, 4.4 GB/s |
| 3-D 32M stretched | 17.5 MB (measured locally at border 9) | 111,556 | 15.62 | 1,849 × 129 × 4 = 0.95 | **16.6** | 783–1379 / 774–782 | 1048–1244 | ≈ 20 | 232 MB/frame, 5.5 GB/s |
| 2-D 128M | — | ≈ 57,800 | 8.1 | ≈ 0.9 | **≈ 9.0** | 186 / 215 | 839 | 37.6 | 126 MB/frame, 4.3 GB/s |
| 2-D 64M | 5.6 MB | 40,900 | 5.73 | 0.62 | **6.35** | 162–168 / 137–139 | 382–428 | 41.5 | 89 MB/frame, 5.8 GB/s |
| 2-D 32M | — | 28,395 | 3.98 | 0.44 | **4.42** | 124–132 / 101–103 | 268–326 | 38.9 | 62 MB/frame, 6.1 GB/s |
| 2-D 16M | — | 20,115 | 2.82 | 0.31 | **3.13** | 97 / 76 | 789 | 37.1 | 44 MB/frame, 5.6 GB/s |
| 2-D 4M | — | 10,225 | 1.43 | 0.16 | **1.59** | 269 / 263 | 809 | 5.4 | 22 MB/frame, 2.7 GB/s |

Bytes are derived from what the count-aware worker copies (the live particles of the one-voxel halo column, verified against the ghost_send counter of the local 3-D soak: 125,316 = exactly one column at border 9), because the transport stamps record timings only. Bytes per link do not depend on K. The host copy stage has a ≈ 0.8 ms floor below 3 MB — the reason the 2-D 4M / 16M chains cannot hide inside a 0.8–3 ms phase B — and runs at 12–15 GB/s for ≥ 4 MB.

### T9 — Cube boundary point (probe37, K=8, no reference)

| geometry | particles | K=8 fps (3 trials) | phase A / B / C (ms) | C / B | b→c gap (ms) | upload slack lead / trail (ms) |
|---|---|---|---|---|---|---|
| stretched 64M (8.8M per GPU, slab 201 dx = 50 voxels, halo 2% per side) | 70.6M | 11.96 / 12.02 / 12.03 | 1.24 / 69.2 / 10.3 | 0.15 | 0.17 | 56 / 44 |
| cube 401³ (8.55M per GPU, slab 50 dx = 12.5 voxels, halo 8% per side, 64% of the slab in the force band) | 68.4M | 11.96 / 11.75 / 12.00 | 1.49 / 42.6 / 34.3 | 0.81 | 2.84 | 17 / 5 |

Same fps within 1% (per particle the cube is 2% slower): the 3-D band kernels cost only 1.2–1.5× the interior per particle, so moving 64% of a slab into phase C costs little compute — but the transport slack collapses from 56 / 44 ms to 17 / 5 ms and phase C waits 2.8 ms per frame (3.4%). At K=8 the cube is at the edge of the hiding window; a 16-GPU cube would not be hidden.

## 4. Mechanisms behind the residual

### Chain phase drift (S4)

Per-frame phase timestamps of both GPUs of a local K=2 chain on one calibrated clock: GPU 0 is intrinsically 0.8% faster (period 4.163 vs 4.200 ms), so the offset between the two cards ramps at +35 µs per frame until it reaches a ceiling of 0.72 T where the fast card's phase C is gated by the slow card's upload; there it waits a little every frame, and every few hundred frames it stalls a whole period and the ramp restarts. Two identical runs took different trajectories (means +0.66 and +1.60 ms) at 2–3% cost at this load. The readiness scheduler holds a fixed offset (−0.58 T) but waits every frame instead (228 vs 234–236 fps). On the 8-GPU node this drift is what produces the c→a gaps on the fast cards and the b→c gaps on the slow ones (T6, T7) and the just-in-time frames on every point of T5 (p10 margin ≈ 0). Remedy with a basis: rate-match the cards (per-card weights from measured periods) or pace submissions to the slowest card's period. Not implemented.

### Band kernels: occupancy, not locality (Nsight GPU Trace)

| kernel | mode | SM throughput % | warps active % of peak | threads per warp-instruction (of 32) | L1 hit % | L2 hit % | cycles (M) |
|---|---|---|---|---|---|---|---|
| correction | interior (phase B) | 81 | 95 | 21.0 | 77 | 85 | 58.4 |
| correction | band (voxel, slot) | 47 | 43 | 22.0 | 78 | 88 | 3.92 |
| density | interior / band | 80 / 48 | 94 / 46 | 21.3 / 22.6 | 78 / 76 | 86 / 81 | 60.4 / 6.07 |
| force | interior / band | 70 / 54 | 62 / 43 | 18.1 / 19.6 | 78 / 78 | 81 / 87 | 77.5 / 8.11 |

Local RTX 5090, 3-D 8M single GPU with a diagnostic interior band (identical kernels and costs to the real seam band: fake band = real band at equal size in 2-D and 3-D). Divergence and cache hit rates are the same for band and interior kernels; the band kernels run at half the warp occupancy because the (voxel, slot) mapping launches 50% (3-D) / 75% (2-D) dead threads that hold their CTA's resources. A compacted-list mapping lifts occupancy to 56–69% and cuts 11–22% of the cycles but stays a single-wave kernel. Per particle, band kernels cost 1.24–1.55× (3-D) and, marginally, 2–2.7× for density / force in 2-D with a ≈ 77 µs launch floor. Worth ≈ 3–4% of a 64M K=8 frame at most; not changed under the freeze.

### Bootstrap defrag

The generator's initial order is not voxel-sorted; before 2026-09-17 the first 1000 frames of every run executed the interior kernels 5–9× slower (2-D 4M: phase B 23.7 ms at f1000 vs 3.5 ms at f2000). A defrag at bootstrap (same path as the periodic one) removed this: first 1000 frames 22 → 4.4 s (2-D 4M K=2), 188 → 83 s (3-D 8M K=1); steady throughput unchanged (interleaved A/B 226.8 / 226.7 vs 225.0 / 227.1 fps). All 3-D matrix runs used it; the 2-D curves predate it (their steady windows are unaffected by construction).

## 5. Correctness evidence

- **Every run in every table**: particle count conserved (drift 0), transport frame-stamp errors 0 on GPU and host; 2-D seam checks (overshoot < 1 dx, no duplicates, ρ within 5%, v ≤ lid) on the 4M K=2 / K=4 jobs; 3-D seam checks on the K=2 runs (overshoot 0.00 dx, dup 0, ρ ∈ [999.0, 1001.2]).

- **Per-particle verifier** (KD-tree matched fields, acceleration difference by column distance to the seam vs the run-to-run noise floor, PASS if worst ratio < 3): cascading force 1M K=2 (0.53–1.06), 1M K=4 (0.72–1.37), 4M K=2, 1M K=2 2000 steps (chaos-grown noise 0.76–1.40); band dispatch 1M K=2 / K=4 / 4M K=2 (0.50–1.39, band invariant OK); fast submit (0.58–1.42); no-wait (1.23); readiness scheduler on-cluster 4M K=4 300 / 2000 steps (1.02 / 1.24); 3-D K=2 stretched (1.11); shader refactor vs previous build (1.86); fake band (1.38); wall border 4 vs 9 on 8M K=1 (1.94) and K=2 stretched (1.35) with centreline profiles inside the run-to-run envelope.

- **Equivalence battery** (K=1, 1, 2, 2; aggregate quantities within the K=1 rerun envelope): 2-D (N56) and 3-D K=2 stretched (local) ALL PASS.

- **Soaks**: 64M K=8 17k frames on N56 drift 0 (cascade, band, fast, ready variants); 12-hour 8M dual soak locally 4.78M frames drift 0 (earlier); 3-D K=2 local soak ghost overflow 0 at pool factor 1.0.

## 6. Provenance

| job | date (CST) | node | commit | content | logs (logs/n56/…) |
|---|---|---|---|---|---|
| probe34 1592849 / 50 / 51 | 2026-09-16 00:2x–01:25 | wqd10nbm13g5 (4/16/32/128M), wqd10nbj04g2 (64M) | 367706a (pre-commit tree) | 2-D strong vs size, K=8, 3 trials, 8-GPU simultaneous refs | 2026-09-16/01–03_probe34_curve_* |
| probe35 1593607 / 08, 1594018 / 19 / 20 / 21 / 22 | 2026-09-16 11:20–15:00 | both nodes | 367706a | 2-D weak families 2 / 4 / 8 / 16 / 32M per GPU | 2026-09-16/04–10_probe35_weak_* |
| probe36 1594545 | 2026-09-16 17:24–20:38 (3 h 15) | wqd10nbj04g2 | 367706a | 2-D fixed-N K sweep 64M / 32M, K=2/4/8, anatomy, cross-NUMA control | 2026-09-16/11_probe36_ksweep_1594545 |
| probe37 1595386 | 2026-09-17 00:46–03:38 (2 h 52) | wqd10nbj04g2 | eda4b8f | 3-D matrix: strong 64M / 32M, weak 8M / 4M per GPU, cube control; 71 runs | 2026-09-17/01_probe37_3d_1595386 |
| probe38 1596186 | 2026-09-17 10:00–10:10 | wqd10nbm13g5 | eda4b8f | 2-D K=8 anatomy at 4M / 16M / 128M (transport numbers for F4 only) | 2026-09-17/02_probe38_anat_1596186 |
| campaign probes 13–32 | 2026-09-15 | wqd10nbj04g2 | uncommitted → 367706a | optimization A/Bs at 64M K=8 (S5, S6), on-cluster verifier | 2026-09-15/* |

- **repo** `vulkan-demo`, branch `v4-multigpu-orchestration`; freeze commit **eda4b8f** (tag `freeze-2026-09-17`) = cascade + band default ON; later commits touch docs/scripts only. Server checkout verified equal to eda4b8f (`git diff --ignore-cr-at-eol` empty). Experiment code (compact band dispatch, GPU-trace labels) lives on branch `exp/band-compact`, not merged.

- **scripts** `docs/n56_scaling/scripts/`: probe34_curve / probe35_weak / probe36_ksweep / probe37_3d_matrix / probe38_anatomy .sbatch; plot_curve34, plot_weak35, plot_ksweep36, plot_3d_matrix, plot_3d_anatomy, plot_hiding_margin, plot_paper_figs, band_cost_table, wall_profile_compare, plot_phase_trace .py; gen_weak_families.sh, gen_3d_families.sh, analyze_3d_prep.py.

- **write-ups** `docs/n56_scaling/`: ksweep_fixed_n.md, matrix_3d_results.md, hiding_margin_and_ledger.md, 3d_prep.md, 3d_matrix_budget.md, phase_offset_trace.md, bootstrap_defrag_fakeband_border.md, band_dispatch_experiment.md, band_gputrace_metrics.txt; earlier dossiers dossier.html (I), dossier_v2.html (II).

- **tables (JSON)** ksweep_table.json, matrix_3d_table.json, hiding_margin_points.json, hiding_margin_fit.json, anatomy_3d.json.

- **telemetry** per job `telemetry.csv` (nvidia-smi every 5 s): power at cap (574–575 W on wqd10nbj04g2, 600 W on wqd10nbm13g5), busy SM clocks 2640–2784 MHz, ≤ 72 °C.

- **cost** probe36 3.24 node-h (11 reference sets = 45%); probe37 2.87 node-h (references 29%); probe38 0.17 node-h.

## 7. Notes for writing [caveats]

- **Quote the two regimes, not the fit.** Neither a constant-floor nor a floor-plus-frame-fraction model reproduces the twelve hidden points within their error bars (T5); the defensible statements are: hidden regime for t_transport / T_B ≤ 0.65, i.e. ≥ 8M/GPU in 2-D and every 3-D point ≥ 4.5M/GPU; loss there ordered by K; exposed regime with η tracking the margin below.

- **Node mix.** 2-D sizes 4 / 16 / 32 / 128M were measured on the 600 W node and 64M on the 575 W node; each η is intra-job. The 4M/GPU point reproduces only to ≈ 5 points across nodes (67.4 vs 72.7%).

- **Reference substitutes.** 2-D 128M and 256M use K=2-pair references (factor K/2). 3-D 64M used real K=1 references (23 GB per card at border 4, 1.62 fps × 8 at once).

- **Window.** All numbers are from a cavity started from rest (steps 1000–3000 or 500–1500); the 12-hour local soak showed fps falling 120 → 109 as the flow develops. Developed-flow efficiency on N56 is not measured.

- **Light-load bimodality** (2-D 2–4M/GPU) is real and unexplained beyond the phase-drift mechanism shown locally; 3-D 4M/GPU does not show it (slack 30 ms).

- **2-D curves ran without the bootstrap defrag**; steady windows are unaffected (the first defrag at frame 1000 sorts everything), wall times in those logs are not comparable with 3-D ones.

- **Not measured**: multiple nodes (no inter-node transport exists), NVLink/P2P (consumer GeForce has none), inflow/outflow, physics validation (Ghia profiles) on the cluster runs, weights sweeps on N56, K=8 3-D cube with a reference.
