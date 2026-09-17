# Hiding margin, 3-D anatomy vs K, 64M K=8 loss ledger, bytes per link — 2026-09-17

Sources: probe34 (2-D K=8 vs size, standardized η), probe36 (2-D fixed-N K sweep, anatomy), probe37 (3-D
matrix, anatomy), probe38 (2-D K=8 anatomy at 4M / 16M / 128M, job 1596186, wqd10nbm13g5, 1500 steps, one run
each, only to supply the transport numbers that probe34 did not record). Scripts: `scripts/plot_hiding_margin.py`,
`scripts/plot_3d_anatomy.py`; data `hiding_margin_points.json`, `anatomy_3d.json`.

## 1. Unified hiding-margin figure — `eta_vs_hiding_margin.png`

Definition. Per sampled frame and GPU the anatomy gives, for each link, slack = c_start − upload_end
(how long before phase C the neighbour's ghost upload landed) and the b→c gap (how long phase C waited).
The slack is clamped at 0 by construction (phase C waits on the upload semaphore), so an exposed transport
shows up as slack ≈ 0 plus a b→c gap. The transport time of the binding link is therefore
t_transport = T_B − min_link(slack) + gap, and the margin x = (T_B − t_transport) / T_B = (min_link slack − gap) / T_B
per frame; the plotted x is the p50 over frames (x-bar down to the p10). y = standardized η with its error bar.

| point | η | T_B (ms) | slack p50 (ms) | b→c gap (ms) | margin p50 | p10 | min | samples |
|---|---|---|---|---|---|---|---|---|
| 2-D 4M K=8 | 10.8% | 0.82 | 0.02 | 4.99 | **−6.09** | −12.5 | −12.7 | 8 |
| 2-D 16M K=8 | 44.1% | 3.07 | 0.08 | 2.88 | **−0.25** | −2.44 | −2.59 | 8 |
| 2-D 32M K=8 | 67.4% | 6.13 | 3.30 | 1.54 | **0.25** | −0.64 | −0.87 | 48 |
| 2-D 64M K=8 | 91.4% | 12.42 | 9.82 | 0.01 | 0.58 | 0.00 | −0.03 | 48 |
| 2-D 128M K=8 | 95.4% | 24.17 | 21.42 | 0.12 | 0.71 | 0.01 | −0.04 | 8 |
| 2-D 64M K=2 | 98.3% | 50.72 | 26.49 | 0.44 | 0.52 | 0.04 | −0.11 | 12 |
| 2-D 64M K=4 | 94.6% | 25.28 | 20.10 | 0.26 | 0.36 | −0.03 | −0.05 | 24 |
| 2-D 32M K=2 | 97.0% | 25.24 | 11.57 | 0.03 | 0.46 | 0.03 | −0.01 | 12 |
| 2-D 32M K=4 | 92.6% | 12.61 | 8.98 | 0.07 | 0.35 | −0.02 | −0.04 | 24 |
| 3-D 64M K=2 | 98.0% | 302.8 | 246.8 | 0.19 | 0.82 | 0.06 | −0.00 | 6 |
| 3-D 64M K=4 | 96.1% | 147.9 | 118.8 | 0.64 | 0.88 | −0.00 | −0.05 | 12 |
| 3-D 64M K=8 | 92.8% | 69.2 | 55.7 | 0.17 | 0.79 | 0.01 | −0.05 | 24 |
| 3-D 32M K=2 | 97.1% | 153.1 | 72.6 | 0.85 | 0.47 | −0.02 | −0.02 | 6 |
| 3-D 32M K=4 | 94.1% | 74.0 | 60.6 | 0.94 | 0.40 | −0.03 | −0.05 | 12 |
| 3-D 32M K=8 | 91.7% | 33.3 | 25.7 | 0.20 | 0.76 | 0.14 | −0.06 | 24 |

**The points are not collinear; there are two regimes.**

- Margin < ≈ 0.3 (three 2-D K=8 points): η tracks the margin — 4M (−6.1 → 10.8%), 16M (−0.25 → 44%),
  32M (+0.25 with p10 −0.64 → 67%). Here the transport chain is longer than, or comparable to, phase B and
  phase C waits on most frames (b→c gap 1.5–5 ms per frame).
- Margin ≥ 0.35 (all twelve other points): η no longer depends on the margin at all; it is set by K.
  K=2: 97.0–98.3% at margins 0.46–0.82; K=4: 92.6–96.1% at 0.35–0.88; K=8: 91.4–95.4% at 0.58–0.79.
  Points that deviate from a single curve: 2-D 32M K=4 (margin 0.35, η 92.6%) vs 2-D 64M K=2 (0.52, 98.3%)
  and 3-D 32M K=2 (0.47, 97.1%) — nearly the same margin, 5 points apart because of K; 3-D 64M K=4 (0.88,
  96.1%) sits *below* 2-D 64M K=2 (0.52, 98.3%) despite a larger margin; 3-D 64M K=8 (0.79, 92.8%) sits below
  2-D 128M K=8 (0.71, 95.4%) — at K=8 η rises with the per-GPU size (16M vs 8.8M per GPU), not with the margin.
  What orders these points is the non-scaling floor per frame (phase A + C + gaps, §3), i.e. K and the
  per-GPU size, once the transport is hidden.
- Every point, including the fully hidden ones, has a p10 ≈ 0 and a negative minimum: on ≥ 10% of the sampled
  frames the binding link landed just in time or a little late (the sawtooth phase drift measured in
  `phase_offset_trace.md`: the faster card drifts ahead until its C waits). Those frames cost the 0.2–0.9 ms
  b→c gaps seen at K=2/4 in 3-D.

## 2. 3-D anatomy vs K — `anatomy_vs_k_3d.png`

Per-GPU frame (ms, f1000, all GPUs, 3 trials; K=1 = the 8-GPU reference set); gaps split into c→a (queue ran
dry: the global loop had not submitted the next frame) and a→b + b→c (waiting for the neighbours' uploads).

| block | K | period | ideal T_ref/K | A | B | C | c→a | b→c | other idle | B·K | non-B share of frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64M | 1 | 618.3 | — | 6.60 | 609.8 | 2.41 | 0.01 | 0.01 | 0.00 | 610 | 1.5% |
| | 2 | 314.6 | 309.1 | 3.58 | 302.8 | 7.43 | 0.01 | 0.20 | 0.53 | 606 | 3.7% |
| | 4 | 161.1 | 154.6 | 2.07 | 147.9 | 9.26 | 1.38 | 0.64 | 0.00 | 591 | 8.3% |
| | 8 | 83.3 | 77.3 | 1.24 | 69.2 | 10.26 | 1.90 | 0.17 | 0.57 | 553 | 17.0% |
| 32M | 1 | 312.3 | — | 3.31 | 307.7 | 1.22 | 0.01 | 0.01 | 0.11 | 308 | 1.5% |
| | 2 | 161.4 | 156.2 | 1.93 | 153.1 | 4.93 | 0.01 | 0.85 | 0.56 | 306 | 5.1% |
| | 4 | 83.1 | 78.1 | 1.18 | 74.0 | 6.20 | 0.37 | 0.94 | 0.47 | 296 | 11.0% |
| | 8 | 42.6 | 39.0 | 0.80 | 33.3 | 6.83 | 0.96 | 0.20 | 0.44 | 267 | 21.7% |

Reading: B·K falls from 610 to 553 ms (64M) as K grows because the band columns move out of phase B into
phase C (C grows 2.4 → 10.3 ms while B shrinks faster than 1/K): the work is conserved, only its phase changes.
The gaps are dominated by **c→a** at K=4/8 (1.4–1.9 ms at 64M, 0.4–1.0 at 32M) — the global loop gating on the
slowest card — with b→c (upload waits) at 0.2–0.9 ms; both are the two ends of the same phase-drift sawtooth.

## 3. 64M K=8 loss ledger, 2-D and 3-D side by side (ms per frame per GPU)

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
| **→ net per-frame floor (A + C excess + B deficit)** | **+0.73 (4.8% of the frame)** | **+3.33 (4.0%)** |
| gap c→a | 0.25 | 1.90 |
| gap a→b + b→c | 0.02 | 0.17 |
| other idle | 0.36 | 0.57 |
| **→ gaps + idle excess** | **+0.57 (3.8%)** | **+2.63 (3.2%)** |
| sum of the two lines | 1.30 | 5.96 |

Both dimensions lose the same two things in the same proportions: a per-frame floor that does not shrink with
K (band kernels' extra per-particle cost, ghost_send / install / barriers, the scratch→primary copy) ≈ 4–5%,
and gaps + idle ≈ 3–4% (chain phase drift: c→a on the fast cards, b→c on the slow ones). 3-D is one point
better only because its frame is 5.5× longer while the floor grew by 4.6×.

## 4. Bytes per link per frame

The transport stamps in the worker (`dequeue / source_wait / dest_guard / wait / copy / signal` timestamps)
carry timings, not sizes; the copied size is kept only as `last_copy_bytes` and not logged. The per-frame
bytes are therefore derived from what the count-aware worker copies: the live particles of the one-voxel
halo column (one column of the initial lattice, walls included; verified against the ghost_send counter of the
local 3-D soak: 125,316 = exactly one column) × 140 B per particle (9 SoA fields), plus the two voxel-indexed
segments of the ghost face (inside_particle_count + inside_particle_index: face voxels × (MPV + 1) × 4 B).

| case | halo column particles | SoA bytes | voxel-list bytes | **MB per link per frame** | links (K=8) | node-wide at K=8 |
|---|---|---|---|---|---|---|
| 3-D 64M stretched (201² face, border 4) | 174,724 | 24.46 MB | 2,809 vox × 129 × 4 = 1.45 MB | **25.9** | 7 × 2 directions | 363 MB per frame at 12.0 fps = 4.4 GB/s |
| 3-D 32M stretched (159² face) | 111,556 | 15.62 MB | 1,849 × 129 × 4 = 0.95 MB | **16.6** | 7 × 2 | 232 MB per frame at 23.5 fps = 5.5 GB/s |
| 2-D 128M | ≈ 57,800 | 8.1 MB | ≈ 2,276 × 97 × 4 = 0.9 MB | **≈ 9.0** | 7 × 2 | 126 MB at 34.2 fps = 4.3 GB/s |
| 2-D 64M | 40,900 | 5.73 MB | 1,609 × 97 × 4 = 0.62 MB | **6.35** | 7 × 2 | 89 MB at 65.6 fps = 5.8 GB/s |
| 2-D 32M | 28,395 | 3.98 MB | 0.44 MB | **4.42** | 7 × 2 | 62 MB at 99 fps = 6.1 GB/s |
| 2-D 16M | 20,115 | 2.82 MB | 0.31 MB | **3.13** | 7 × 2 | 44 MB at 127 fps = 5.6 GB/s |
| 2-D 4M | 10,225 | 1.43 MB | 0.16 MB | **1.59** | 7 × 2 | 22 MB at 122 fps = 2.7 GB/s |

The bytes per link do not depend on K (the halo is one column whichever way the domain is cut); with 3
trials per K the measured DMA and host-copy times per link were:

| point | readback DMA (µs) | upload DMA (µs) | host copy p50 (µs) | upload GB/s | host copy GB/s |
|---|---|---|---|---|---|
| 3-D 64M K=2 / 4 / 8 | 1326 / 1450 / 2159 | 1225 / 1182 / 1249 | 1782 / 1960 / 2453 | 20.0 / 20.7 / 19.6 | 13.7 / 12.5 / 10.0 |
| 3-D 32M K=2 / 4 / 8 | 794 / 783 / 1379 | 774 / 774 / 782 | 1048 / 1211 / 1244 | 20.2 / 20.2 / 20.0 | 14.9 / 12.9 / 12.6 |
| 2-D 64M K=2 / 4 / 8 | 168 / 166 / 162 | 139 / 138 / 137 | 382 / 392 / 428 | 41.2 / 41.5 / 41.7 | 15.0 / 14.6 / 13.4 |
| 2-D 32M K=2 / 4 / 8 | 132 / 132 / 124 | 101 / 103 / 103 | 268 / 289 / 326 | 39.5 / 38.6 / 38.7 | 14.8 / 13.8 / 12.2 |
| 2-D 4M / 16M / 128M K=8 | 269 / 97 / 186 | 263 / 76 / 215 | 809 / 789 / 839 | 5.4 / 37.1 / 37.6 | 1.8 / 3.6 / 9.7 |

(SoA bytes / DMA time; the 3-D uploads move at ≈ 20 GB/s and the 2-D ones at ≈ 40 GB/s because the 3-D
packet includes the 1.45 MB voxel-list segments and the count-aware copy has more, smaller segments; the host
copy stage sits at 12–15 GB/s for ≥ 4 MB payloads and has a ≈ 0.8 ms floor below 3 MB — the reason the
2-D 4M/16M chain cannot hide inside a 0.8–3 ms phase B.) Whole transport chain per link at 3-D 64M K=8:
2.2 + 2.5 + 1.2 ≈ 5.9 ms plus waits, against a 69 ms phase B — margin 0.79 as measured.
