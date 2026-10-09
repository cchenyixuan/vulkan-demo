# Per-kernel cost, v7_single, one NVIDIA GeForce RTX 5090 (2-D: 8,128,201 particles (3 runs), 3-D: 9,129,329 particles (3 runs))

Sync loop (1 frame in flight), warmup 1000, 2000 timed steps per run, defrag frames excluded, BOTTOM_OF_PIPE timestamps. Values: mean over runs of the per-run mean (p50 in parentheses); ns/particle uses the alive count incl. walls.

| kernel | dispatch | 2-D µs/step | 2-D ns/particle | 2-D % | 3-D µs/step | 3-D ns/particle | 3-D % |
|---|---|---|---|---|---|---|---|
| predict | per particle | 496.6 (496.6) | 0.06 | 3.6% | 499.0 (498.9) | 0.05 | 0.6% |
| update_voxel | per voxel | 331.5 (331.8) | 0.04 | 2.4% | 318.2 (317.4) | 0.03 | 0.4% |
| correction | per particle | 3858.7 (3851.3) | 0.47 | 27.9% | 24159.0 (24155.0) | 2.65 | 29.5% |
| density_kernel | per particle | 3955.5 (3948.5) | 0.49 | 28.6% | 25824.1 (25833.5) | 2.83 | 31.5% |
| density_copy | buffer copy | 57.7 (57.5) | 0.01 | 0.4% | 71.4 (71.8) | 0.01 | 0.1% |
| force | per particle | 5114.6 (5108.6) | 0.63 | 37.0% | 31003.5 (30965.6) | 3.40 | 37.9% |
| step total |  | 13814.6 (13794.0) | 1.70 | 100.0% | 81875.2 (81840.5) | 8.97 | 100.0% |

## Acceptance

- 2-D: drift = 0 on all runs: **yes**; step total 13814.6 ± 6.0 µs over 3 runs (std 0.04%, limit 1%): **ok**; CPU-side 35.73 fps.
- 2-D: timed GPU frame 13814.6 µs; the timed frames' wall time 13979.1 µs/step (submit + fence, fresh fence per step: +164.5 µs = the exposed host round trip); the production frames' wall time 13944.0 µs/step (interleaved, one persistent fence, no timestamps): +129.4 µs (+0.94 %) over the GPU frame, -35.1 µs against the timed frames.
- 2-D switches: {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_FUSED_CORRECTION_DENSITY': 0}
- 2-D candidates (particles listed in the 3^d voxels around each particle's voxel, from the end-of-run lists): 1,823,771,992 pairs per step = 224.4 per particle; ps per candidate of the neighbour kernels: correction 2.12, density_kernel 2.17, force 2.80
- 3-D: drift = 0 on all runs: **yes**; step total 81875.2 ± 80.3 µs over 3 runs (std 0.10%, limit 1%): **ok**; CPU-side 6.09 fps.
- 3-D: timed GPU frame 81875.2 µs; the timed frames' wall time 82139.6 µs/step (submit + fence, fresh fence per step: +264.4 µs = the exposed host round trip); the production frames' wall time 82067.0 µs/step (interleaved, one persistent fence, no timestamps): +191.8 µs (+0.23 %) over the GPU frame, -72.6 µs against the timed frames.
- 3-D switches: {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_FUSED_CORRECTION_DENSITY': 0}
- 3-D candidates (particles listed in the 3^d voxels around each particle's voxel, from the end-of-run lists): 15,054,967,515 pairs per step = 1649.1 per particle; ps per candidate of the neighbour kernels: correction 1.60, density_kernel 1.72, force 2.06

## LaTeX rows

```latex
% columns: kernel & dispatch & 2-D $\mu$s/step & 2-D ns/particle & 2-D \% & 3-D $\mu$s/step & 3-D ns/particle & 3-D \% \\
% 2-D: 8,128,201 particles; 3-D: 9,129,329 particles
predict & per particle & 497 & 0.06 & 3.6 & 499 & 0.05 & 0.6 \\
update\_voxel & per voxel & 331 & 0.04 & 2.4 & 318 & 0.03 & 0.4 \\
correction & per particle & 3859 & 0.47 & 27.9 & 24159 & 2.65 & 29.5 \\
density (kernel) & per particle & 3956 & 0.49 & 28.6 & 25824 & 2.83 & 31.5 \\
density (scratch$\to$primary copy) & buffer copy & 58 & 0.01 & 0.4 & 71 & 0.01 & 0.1 \\
force & per particle & 5115 & 0.63 & 37.0 & 31003 & 3.40 & 37.9 \\
\midrule
\textbf{step total} &  & 13815 & 1.70 & 100.0 & 81875 & 8.97 & 100.0 \\
```
