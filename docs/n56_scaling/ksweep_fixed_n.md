# Fixed-N K sweep with error bars (Task A) — job 1594545, 2026-09-16

Node wqd10nbj04g2 (8× RTX 5090, 575 W cap), commit 367706a, 3 h 15 min wall (3.24 node-hours; estimate was ≈2 h —
the eleven 64M reference sets at 7.8 min each are ≈45% of the job). Protocol: docs/n56_scaling (2026-09-15) with the
K=1 reference measured simultaneously on exactly the GPUs that take part in the K run (K=2 → 0,1; K=4 → 0–3;
K=8 → 0–7), three alternating trials (reference set, K run), chain bench 3000 steps (steady = last 2000),
`--anatomy` on every run, no solver/shader change, same host stack as the curve jobs. Cross-NUMA control: K=2 on
GPUs (3, 4), a SYS path in `nvidia-smi topo -m` (GPUs 0–3 on NUMA 0, 4–7 on NUMA 1), two trials.
All 41 runs: drift 0, stamp errors 0. Busy SM clocks 2640–2677 MHz median, all cards at 575 W, ≤72 °C.

| N | K | GPUs | per GPU | K-run fps (3 trials) | reference mean per trial (per-GPU range) | η_mean | ± std | η_min |
|---|---|---|---|---|---|---|---|---|
| 64M | 2 | 0,1 | 32.2M | 17.8 / 17.8 / 17.8 | 9.10 / 9.05 / 9.00 (9.0–9.1) | 98.3% | 0.5 | 98.5% |
| 64M | 4 | 0–3 | 16.1M | 34.1 / 34.2 / 34.3 | 9.00 / 9.07 / 9.05 (9.0–9.1) | 94.6% | 0.3 | 95.0% |
| 64M | 8 | 0–7 | 8.05M | 66.3 / 65.3 / 66.3 | 9.01 / 9.04 / 9.03 (9.0–9.1) | 91.4% | 0.9 | 91.6% |
| 64M | 2 | 3,4 cross-NUMA | 32.2M | 17.9 / 17.9 | 9.10 / 9.10 | 98.4% | 0.0 | 98.4% |
| 32M | 2 | 0,1 | 16.3M | 35.4 / 35.3 / 35.2 | 18.2 / 18.2 / 18.2 | 97.0% | 0.3 | 97.0% |
| 32M | 4 | 0–3 | 8.1M | 67.0 / 67.0 / 67.1 | 18.1 / 18.1 / 18.1 (18.0–18.2) | 92.6% | 0.1 | 93.3% |
| 32M | 8 | 0–7 | 4.06M | 101.4 / 105.5 / 108.9 | 18.07 / 18.10 / 18.09 (17.9–18.2) | 72.7% | 2.6 | 73.4% |

Acceptance: 64M K=8 reproduces the previous job's 91.4 ± 0.3% (criterion ±1%); η monotone in K for both sizes.

## Frame anatomy vs K (64M, per-GPU means over frames ≥ 2000, all GPUs, 3 trials)

| K | phase A | phase B | phase C | gaps | other idle | period (ms) | ideal period | B·K | transport slack p50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 (ref) | 6.37 | 101.90 | 2.19 | 0.01 | 0.33 | 110.80 | — | 101.9 | — |
| 2 | 3.32 | 50.72 | 1.71 | 0.45 | 0.00 | 56.18 | 55.4 | 101.4 | 26.5 ms |
| 4 | 1.78 | 25.28 | 1.31 | 0.50 | 0.37 | 29.24 | 27.7 | 101.1 | 20.1 ms |
| 8 | 1.02 | 12.42 | 1.10 | 0.26 | 0.36 | 15.16 | 13.85 | 99.4 | 9.8 ms |

Reading:

- Phase B scales ideally (B·K constant; K=8 even 2.5% under). The whole loss sits in what does not shrink:
  A + C = 8.6 → 5.0 → 3.1 → 2.1 ms (≈1 ms above the ideal (A+C)/K at K=8) and gaps + idle 0.3 → 0.45 → 0.9 → 0.6 ms.
  At K=8 the 1.31 ms excess over the ideal period = +1.05 (A+C) + 0.58 (gaps/idle) − 0.32 (B faster than ideal).
- Transport is hidden at every K (median slack 26 / 20 / 10 ms vs phase B 50 / 25 / 12 ms); the lower whisker reaches 0
  at K=4 and K=8 (a few just-in-time frames = the small b→c gaps).
- Cross-NUMA K=2 pair = same-NUMA pair (17.9 vs 17.8 fps): the link topology is irrelevant with 26 ms of slack.
- 32M K=8 (4M/GPU) is the light-load regime: 101–109 fps across trials, and 5 points above the 67.4 ± 2.3% measured
  yesterday on wqd10nbm13g5. ≤4M/GPU efficiencies reproduce only to ≈5 points across nodes/jobs; ≥8M/GPU to ±1.
- Same per-GPU load, fewer GPUs: 32M K=4 (8M/GPU) 92.6% vs 64M K=8 (8M/GPU) 91.4% → the 8-way chain costs ≈1 point
  more than the 4-way chain at equal load.

Consequence: at heavy load the strong-scaling limiter is the per-frame floor (scratch→primary copy ≈0.35 ms, band
kernels ≈0.6 ms at 2× per-particle cost, ghost_send/install/barriers ≈0.2 ms). Removing the copy and halving the band
cost would move 64M K=8 from 91 to ≈95%.

Files: `eta_strong_vs_k.png`, `anatomy_vs_k_64m.png`, `ksweep_table.json`, `scripts/probe36_ksweep.sbatch`,
`scripts/plot_ksweep36.py`; logs `logs/n56/2026-09-16/11_probe36_ksweep_1594545/` (RESULT lines in the .out,
per-run logs with anatomy in `probe36_1594545/`, `telemetry.csv`).
