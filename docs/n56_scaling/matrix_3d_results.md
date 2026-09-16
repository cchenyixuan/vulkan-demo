# 3-D matrix on N56 — results (job 1595386, wqd10nbj04g2, 2026-09-17 00:46–03:38 CST)

Code eda4b8f (`freeze-2026-09-17`), border-4 cases, protocol of `3d_matrix_budget.md`: chain bench 1500 steps,
warmup 500, steady = last 1000; one 8-GPU simultaneous K=1 reference per N; K-runs ×3; `--anatomy`;
`V5_GHOST_POOL_FACTOR=1.0`, `--pool-safety 1.2`; switches as the 2-D curve jobs. All 8 GPUs at the 574 W cap,
busy SM clocks 2655–2700 MHz. **All 71 runs: drift 0, stamp errors 0.** Wall 2 h 52 min (budget estimate
2.7 h). Precise fps = 1000 steps / steady seconds from each run log (the RESULT line's one-decimal fps is a 3%
quantum at 1.6 fps). Figure `eta_3d_matrix.png`, table `matrix_3d_table.json`, script
`scripts/plot_3d_matrix.py`, logs `logs/n56/2026-09-17/01_probe37_3d_1595386/`.

## Strong scaling (fixed N, stretched domain, reference = K=1 on all 8 GPUs at once)

| case | K | per GPU | K-run fps (3 trials) | reference mean (per-GPU range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|---|
| 64M stretched 1609×201×201 (70.6M total) | 2 | 35.3M | 3.18 / 3.18 / 3.18 | 1.622 (1.621–1.624) | **98.0%** | 0.1 | 98.0% |
| | 4 | 17.7M | 6.20 / 6.20 / 6.22 | 1.615 (1.597–1.624) | **96.1%** | 0.2 | 97.2% |
| | 8 | 8.8M | 11.96 / 12.02 / 12.03 | 1.617 (1.597–1.627) | **92.8%** | 0.3 | 94.0% |
| 32M stretched 1273×159×159 (35.7M total) | 2 | 17.9M | 6.19 / 6.23 / 6.17 | 3.191 (3.186–3.197) | **97.1%** | 0.5 | 97.2% |
| | 4 | 8.9M | 11.98 / 12.00 / 12.09 | 3.196 (3.178–3.222) | **94.1%** | 0.5 | 94.6% |
| | 8 | 4.5M | 23.50 / 23.56 / 23.40 | 3.202 (3.178–3.231) | **91.7%** | 0.3 | 92.4% |

## Weak scaling (reference = K=1 case on all 8 GPUs at once; K=8 = the strong blocks' K=8 runs)

| load per GPU | K | K-run fps (3 trials) | reference mean (range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|
| 8M (9.1M total per GPU) | 2 | 12.01 / 12.01 / 11.99 | 12.477 (12.458–12.495) | **96.2%** | 0.1 | 96.3% |
| | 4 | 11.95 / 11.95 / 11.91 | 12.439 (12.309–12.495) | **96.0%** | 0.2 | 97.0% |
| | 8 | 11.96 / 12.02 / 12.03 | 12.409 (12.300–12.500) | **96.7%** | 0.3 | 97.6% |
| 4M (4.7M total per GPU) | 2 | 23.60 / 23.70 / 23.65 | 24.795 (24.789–24.802) | **95.4%** | 0.2 | 95.4% |
| | 4 | 23.39 / 23.40 / 23.30 | 24.688 (24.432–24.802) | **94.6%** | 0.2 | 95.6% |
| | 8 | 23.50 / 23.56 / 23.40 | 24.642 (24.426–24.802) | **95.3%** | 0.3 | 96.1% |

## Cube control (401³, 68.4M total, K=8, no reference)

| geometry | K=8 fps (3 trials) | phase A / B / C (ms) | C / B | b→c gap | upload slack lead / trail (ms) |
|---|---|---|---|---|---|
| stretched 64M (8.8M per GPU) | 11.96 / 12.02 / 12.03 | 1.24 / 69.2 / 10.3 | 0.15 | 0.17 | 56 / 44 |
| cube 64M (8.55M per GPU, 64% of each slab in the force band) | 11.96 / 11.75 / 12.00 | 1.49 / 42.6 / 34.3 | 0.81 | 2.84 | 17 / 5 |

## Frame anatomy of the K-runs (per GPU, mean over GPUs and trials, f1000)

| block | K | A | B | C | C/B | b→c gap | c→a gap | slack lead / trail (ms) |
|---|---|---|---|---|---|---|---|---|
| strong 64M | 2 | 3.58 | 302.8 | 7.4 | 0.025 | 0.20 | 0.01 | 298 / 79 |
| | 4 | 2.07 | 147.9 | 9.3 | 0.063 | 0.64 | 1.38 | 105 / 80 |
| | 8 | 1.24 | 69.2 | 10.3 | 0.148 | 0.17 | 1.90 | 56 / 44 |
| strong 32M | 2 | 1.93 | 153.1 | 4.9 | 0.032 | 0.85 | 0.01 | 48 / 101 |
| | 4 | 1.18 | 74.0 | 6.2 | 0.084 | 0.94 | 0.37 | 47 / 41 |
| | 8 | 0.80 | 33.3 | 6.8 | 0.205 | 0.20 | 0.96 | 28 / 21 |
| weak 8M | 2 | 1.05 | 74.1 | 6.5 | 0.088 | 1.00 | 0.01 | 68 / 0 |
| | 4 | 1.18 | 71.0 | 9.1 | 0.128 | 0.62 | 1.51 | 64 / 37 |
| weak 4M | 2 | 0.63 | 36.5 | 4.6 | 0.126 | 0.53 | 0.01 | 34 / 2 |
| | 4 | 0.74 | 34.3 | 6.1 | 0.178 | 0.24 | 0.95 | 30 / 16 |

## Reading

- **3-D strong scaling on one node: 98 → 96 → 93% (64M) and 97 → 94 → 92% (32M) for K = 2 → 4 → 8**, with
  error bars ≤ 0.5 points and the reference cards within 1–2% of each other (η_min − η_mean ≈ 1 point). The
  2-D fixed-N sweep gave 98.3 / 94.6 / 91.4% at 64M — 3-D is 1–1.5 points *higher* at K=4/8, because the
  per-frame floor (phase A + C, ≈ 11.5 ms of an 83 ms frame at K=8) is a smaller fraction of a 5× more
  expensive 3-D frame than in 2-D (2.1 of 15.2 ms), even though 3-D phase C itself is larger (C/B 0.15 vs 0.09).
- **3-D weak scaling is flat: 96–97% at 8M/GPU and 95% at 4M/GPU for K = 2, 4, 8**, tight (± 0.1–0.3). The
  light-load bimodality of the 2-D weak families (2–4M/GPU) does not appear at 3-D 4M/GPU, where the 36 ms
  phase B leaves a 30 ms transport slack.
- **Transport is hidden everywhere on the stretched domains**: uploads land 20–300 ms before phase C; b→c gaps
  0.2–1.0 ms. The remaining loss is again the non-scaling floor: phase A + C grow from 11 ms (K=2, 3.5% of
  the frame) to 11.5 ms (K=8, 14%) while B shrinks 4.4×, plus 1–2 ms of gaps at K=4/8.
- **Cube control**: the 1-D slab in a cube reaches the *same* fps as the stretched domain at K=8 (11.9 vs 12.0;
  per particle 2% slower). Its phase C is 3.3× larger (34 vs 10 ms; 64% of the slab in the force band) but its
  phase B shrinks by the same amount, and the 3-D band kernels cost only 1.2–1.5× the interior per particle
  — so in 3-D the cascade's "hide transport behind the interior" premise degrades gracefully. What the cube
  does lose is the transport slack (17 / 5 ms instead of 56 / 44) and a 2.8 ms b→c gap per frame (3.4%): at
  K=8 the cube is at the edge of the hiding window; a 16-GPU cube would not be hidden.
- Cost: 2 h 52 min for 71 runs; references took 49 min (29%); the 64M K=1 references ran at 1.62 fps on all
  8 cards at once (single-card local 1.7).
