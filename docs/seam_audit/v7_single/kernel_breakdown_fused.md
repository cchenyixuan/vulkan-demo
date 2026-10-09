# Per-kernel cost, v7_single, one NVIDIA GeForce RTX 5090 (2-D: 8,128,201 particles (3 runs), 3-D: 9,129,329 particles (3 runs))

Sync loop (1 frame in flight), warmup 1000, 2000 timed steps per run, defrag frames excluded, BOTTOM_OF_PIPE timestamps. Values: mean over runs of the per-run mean (p50 in parentheses); ns/particle uses the alive count incl. walls.

| kernel | dispatch | 2-D µs/step | 2-D ns/particle | 2-D % | 3-D µs/step | 3-D ns/particle | 3-D % |
|---|---|---|---|---|---|---|---|
| predict | per particle | 496.8 (496.7) | 0.06 | 4.7% | 498.4 (498.3) | 0.05 | 0.8% |
| update_voxel | per voxel | 331.3 (331.8) | 0.04 | 3.1% | 318.4 (317.4) | 0.03 | 0.5% |
| correction_density | per particle | 4564.7 (4560.6) | 0.56 | 43.2% | 31410.3 (31381.7) | 3.44 | 49.6% |
| density_copy | buffer copy | 82.0 (82.0) | 0.01 | 0.8% | 77.6 (78.0) | 0.01 | 0.1% |
| force | per particle | 5097.1 (5091.5) | 0.63 | 48.2% | 31014.4 (30974.0) | 3.40 | 49.0% |
| step total |  | 10571.9 (10561.5) | 1.30 | 100.0% | 63319.1 (63251.7) | 6.94 | 100.0% |

## Acceptance

- 2-D: drift = 0 on all runs: **yes**; step total 10571.9 ± 14.6 µs over 3 runs (std 0.14%, limit 1%): **ok**; CPU-side 46.54 fps.
- 2-D: timed GPU frame 10571.9 µs; the timed frames' wall time 10728.0 µs/step (submit + fence, fresh fence per step: +156.1 µs = the exposed host round trip); the production frames' wall time 10695.3 µs/step (interleaved, one persistent fence, no timestamps): +123.4 µs (+1.17 %) over the GPU frame, -32.7 µs against the timed frames.
- 2-D switches: {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_FUSED_CORRECTION_DENSITY': 1}
- 2-D candidates (particles listed in the 3^d voxels around each particle's voxel, from the end-of-run lists): 1,823,772,163 pairs per step = 224.4 per particle; ps per candidate of the neighbour kernels: correction_density 2.50, force 2.79
- 3-D: drift = 0 on all runs: **yes**; step total 63319.1 ± 36.1 µs over 3 runs (std 0.06%, limit 1%): **ok**; CPU-side 7.86 fps.
- 3-D: timed GPU frame 63319.1 µs; the timed frames' wall time 63567.6 µs/step (submit + fence, fresh fence per step: +248.5 µs = the exposed host round trip); the production frames' wall time 63501.7 µs/step (interleaved, one persistent fence, no timestamps): +182.6 µs (+0.29 %) over the GPU frame, -65.9 µs against the timed frames.
- 3-D switches: {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_FUSED_CORRECTION_DENSITY': 1}
- 3-D candidates (particles listed in the 3^d voxels around each particle's voxel, from the end-of-run lists): 15,054,978,746 pairs per step = 1649.1 per particle; ps per candidate of the neighbour kernels: correction_density 2.09, force 2.06

## LaTeX rows

```latex
% columns: kernel & dispatch & 2-D $\mu$s/step & 2-D ns/particle & 2-D \% & 3-D $\mu$s/step & 3-D ns/particle & 3-D \% \\
% 2-D: 8,128,201 particles; 3-D: 9,129,329 particles
predict & per particle & 497 & 0.06 & 4.7 & 498 & 0.05 & 0.8 \\
update\_voxel & per voxel & 331 & 0.04 & 3.1 & 318 & 0.03 & 0.5 \\
correction + density (fused) & per particle & 4565 & 0.56 & 43.2 & 31410 & 3.44 & 49.6 \\
density (scratch$\to$primary copy) & buffer copy & 82 & 0.01 & 0.8 & 78 & 0.01 & 0.1 \\
force & per particle & 5097 & 0.63 & 48.2 & 31014 & 3.40 & 49.0 \\
\midrule
\textbf{step total} &  & 10572 & 1.30 & 100.0 & 63319 & 6.94 & 100.0 \\
```
