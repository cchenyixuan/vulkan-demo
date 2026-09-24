# E5 — Sec. 3 / Sec. 4 supplementary measurements (2026-09-24)

Code: commit `e778a14` (V0 = `utils/sph` + `shaders/sph`; V5 with the loop-order fix `f9c660e`); the
analysis scripts (`_run_kernel_breakdown.py` density split, `_summarize_kernel_breakdown.py`,
`_plot_single_baseline_paper.py`) are added by the commit that adds this directory. Environment: one
headless RTX 5090 (nvidia-smi index 1, PCI 03:00.0, uuid ae137c66…; V0 raw index 2, V5 index 1), driver
576.88, validation layer off, alive count read back after every run (drift must be 0), nvidia-smi 1 Hz
telemetry per item.

| item | deliverable | where |
|---|---|---|
| E5.0 pre-check | V0 runs the 3-D case `cavity3d_8m` (10.5M particles): 200 steps, drift 0 → PASS, E5.1 has both columns | `e5_0_precheck/` |
| E5.1 V0 per-kernel cost (Tab kernelcost) | 2-D and 3-D 8M, density kernel and scratch→primary copy separated, 3 runs each, mean/p50/std, LaTeX rows; all acceptance checks pass (drift 0; step-total std 0.21 % / 0.06 %; 2-D +0.21 % vs the 9/23 reference) | `e5_1_kernel_breakdown/` (`summary.md` has the LaTeX rows) |
| E5.2 figures | Fig A `manuscripts/fig/single_gpu_v0_throughput.pdf` (V0 throughput + fps, 1 and 2 frames in flight); Fig B `manuscripts/fig/v5_single_vs_v0.pdf` (V5 single mode / V0 ratio); PNGs alongside; data not re-run | `e5_2_figures/`, `../../manuscripts/fig/` |
| E5.3 chain K=1 vs single mode | 2-D 8M, same card, 3 interleaved trials: chain / single = 99.91 ± 0.59 % (ratio of means 99.90 %), drift 0 | `e5_3_chain_vs_single/` |

Not done, as specified: defrag cost, 64M, encoding-swap re-measurement, V5 per-kernel (9/23 data stands).

## Headline numbers

V0 per-kernel cost, µs per step (ns per particle), sync loop:

| kernel | 2-D 8M (8,128,201 p.) | 3-D 8M (10,503,459 p.) |
|---|---|---|
| predict | 494 (0.06) | 507 (0.05) |
| update_voxel | 336 (0.04) | 360 (0.03) |
| correction | 3898 (0.48) | 26840 (2.56) |
| density kernel | 3946 (0.49) | 27100 (2.58) |
| density scratch→primary copy | 275 (0.03) | 356 (0.03) |
| force | 5150 (0.63) | 30186 (2.87) |
| step total | 14100 (1.73) | 85350 (8.13) |

The three neighbor kernels are 92 % (2-D) and 98.6 % (3-D) of a step; the copy is 2.0 % / 0.4 %.
