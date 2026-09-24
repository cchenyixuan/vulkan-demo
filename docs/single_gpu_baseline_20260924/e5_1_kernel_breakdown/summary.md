# Per-kernel cost, V0 reference solver, one NVIDIA GeForce RTX 5090 (2-D: 8,128,201 particles (3 runs), 3-D: 10,503,459 particles (3 runs))

Sync loop (1 frame in flight), warmup 1000, 2000 timed steps per run, defrag frames excluded, BOTTOM_OF_PIPE timestamps. Values: mean over runs of the per-run mean (p50 in parentheses); ns/particle uses the alive count incl. walls.

| kernel | dispatch | 2-D µs/step | 2-D ns/particle | 2-D % | 3-D µs/step | 3-D ns/particle | 3-D % |
|---|---|---|---|---|---|---|---|
| predict | per particle | 493.6 (493.5) | 0.06 | 3.5% | 507.1 (507.0) | 0.05 | 0.6% |
| update_voxel | per voxel | 336.1 (335.9) | 0.04 | 2.4% | 360.2 (360.1) | 0.03 | 0.4% |
| correction | per particle | 3898.2 (3899.8) | 0.48 | 27.6% | 26840.3 (26713.4) | 2.56 | 31.4% |
| density_kernel | per particle | 3946.5 (3949.1) | 0.49 | 28.0% | 27100.2 (26958.4) | 2.58 | 31.8% |
| density_copy | buffer copy | 275.2 (274.6) | 0.03 | 2.0% | 355.5 (355.8) | 0.03 | 0.4% |
| force | per particle | 5150.1 (5150.6) | 0.63 | 36.5% | 30186.4 (29971.0) | 2.87 | 35.4% |
| step total |  | 14099.7 (14103.9) | 1.73 | 100.0% | 85349.6 (84857.2) | 8.13 | 100.0% |

## Acceptance

- 2-D: drift = 0 on all runs: **yes**; step total 14099.7 ± 30.2 µs over 3 runs (std 0.21%, limit 1%): **ok**; CPU-side 69.91 fps.
- 2-D step total vs reference 14070 µs: +0.21% (limit ±1%): **ok**
- 3-D: drift = 0 on all runs: **yes**; step total 85349.6 ± 54.0 µs over 3 runs (std 0.06%, limit 1%): **ok**; CPU-side 11.64 fps.

## LaTeX rows

```latex
% columns: kernel & dispatch & 2-D $\mu$s/step & 2-D ns/particle & 2-D \% & 3-D $\mu$s/step & 3-D ns/particle & 3-D \% \\
% 2-D: 8,128,201 particles; 3-D: 10,503,459 particles
predict & per particle & 494 & 0.06 & 3.5 & 507 & 0.05 & 0.6 \\
update\_voxel & per voxel & 336 & 0.04 & 2.4 & 360 & 0.03 & 0.4 \\
correction & per particle & 3898 & 0.48 & 27.6 & 26840 & 2.56 & 31.4 \\
density (kernel) & per particle & 3946 & 0.49 & 28.0 & 27100 & 2.58 & 31.8 \\
density (scratch$\to$primary copy) & buffer copy & 275 & 0.03 & 2.0 & 356 & 0.03 & 0.4 \\
force & per particle & 5150 & 0.63 & 36.5 & 30186 & 2.87 & 35.4 \\
\midrule
\textbf{step total} &  & 14100 & 1.73 & 100.0 & 85350 & 8.13 & 100.0 \\
```
