# E7 batch 2 tables (data only)

Jobs: e7_b2AB_1718639

η = fps_K / (K · mean fps of the K=1 references of the same trial) for strong points, fps_K / mean fps_1 for weak points (per-card case); η_min uses the slowest reference. 'no defrag' = both sides from the steady window minus the wall time of the in-window defrag events (harness b8e73dd and later; '-' = runs without the event lines). fps always includes the events. ns/particle = step time × K / loaded particles. r_own = own chain time / T_B on the worst link per step (p50 / p95) of the arm's traced run.

## 1. Points

| point | job | node | case | K | arm | trials | window | fps | fps (no defrag) | η (no defrag) | η_min (no defrag) | η (with defrag) | η_min (with defrag) | r_own p50 / p95 | SM clock K / ref | ns/particle K | ns/particle K=1 ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5 | e7_b2AB_1718639 | wqd10nbj11g2 | cavity2d_n2840_k5 | 5 | C | 3 | 1500/500 | 91.2 (91.1–91.2) | 92.5 (92.4–92.5) | 1.001 (0.998–1.003) | 1.013 (1.010–1.015) | 0.990 (0.988–0.992) | 1.002 (1.000–1.004) | 0.081 / 0.117 | 1.007 (1.006–1.008) | 1.348 (1.346–1.349) | 1.326 (1.309–1.345) (weak: cavity2d_n2840, 1500/500) |
| old_2d_n2840_k5_K5 | e7_b2AB_1718639 | wqd10nbj11g2 | cavity2d_n2840_k5 | 5 | C | 3 | 1500/500 | 89.1 (89.0–89.2) | 90.3 (90.2–90.5) | 0.978 (0.975–0.980) | 0.991 (0.987–0.993) | 0.968 (0.964–0.970) | 0.981 (0.977–0.983) | 0.099 / 0.120 | 1.015 (1.014–1.016) | 1.379 (1.377–1.381) | 1.327 (1.300–1.348) (weak: cavity2d_n2840, 1500/500) |

## 2. Same-load-normalised η (strong-scaling points)

Reference = every passing K=1 run of the case whose particle count is nearest to the K run's per-card load (tolerance 6 %); η_same = fps_K / mean fps_1 (the weak-scaling definition, per trial); ratio = η_same / η_strong (means).

| point | job | arm | K | per-card particles | same-load K=1 case (particles, mismatch) | K=1 runs (jobs; windows) | K=1 fps | η_same (with defrag) | η_strong (with defrag) | ratio | η_same (no defrag) | η_strong (no defrag) | ratio (no defrag) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## 4. Defrag events (in-window) per case, wall ms per event

| case | K=1 wall ms (events) | K=5 wall ms (events) |
|---|---|---|
| cavity2d_n2840 | 39.1 (38.2–39.7) (30) | - |
| cavity2d_n2840_k5 | - | 168.0 (154.4–204.2) (8) |

## 5. Anatomy: kernel sums against the whole-step wall time (ms, mean over the sampled frames)

(no anatomy runs)

## 6. Traced runs: per-simulator phase medians

**2d_n2840_k5_K5_C** (cavity2d_n2840_k5, warmup 500): medians over the steady steps, ms

| sim | steps | T_A | A→B | T_B | B→C | T_C | C→A | period | busy |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 999 | 0.873 | 0.002 | 9.590 | 0.003 | 0.265 | 0.006 | 10.762 | 10.727 |
| 1 | 999 | 0.877 | 0.002 | 9.525 | 0.002 | 0.406 | 0.006 | 10.818 | 10.807 |
| 2 | 999 | 0.876 | 0.002 | 9.453 | 0.002 | 0.404 | 0.006 | 10.777 | 10.734 |
| 3 | 999 | 0.863 | 0.002 | 9.501 | 0.002 | 0.406 | 0.006 | 10.804 | 10.770 |
| 4 | 999 | 0.867 | 0.002 | 9.513 | 0.002 | 0.269 | 0.144 | 10.809 | 10.648 |

**old_2d_n2840_k5_K5_C** (cavity2d_n2840_k5, warmup 500): medians over the steady steps, ms

| sim | steps | T_A | A→B | T_B | B→C | T_C | C→A | period | busy |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 999 | 0.872 | 0.002 | 9.469 | 0.002 | 0.265 | 0.448 | 11.061 | 10.605 |
| 1 | 999 | 0.867 | 0.002 | 9.300 | 0.002 | 0.404 | 0.485 | 11.061 | 10.571 |
| 2 | 999 | 0.859 | 0.002 | 9.110 | 0.002 | 0.400 | 0.689 | 11.061 | 10.371 |
| 3 | 999 | 0.865 | 0.002 | 9.475 | 0.267 | 0.406 | 0.006 | 11.021 | 10.747 |
| 4 | 999 | 0.892 | 0.002 | 9.883 | 0.002 | 0.273 | 0.006 | 11.058 | 11.047 |


## 7. Old (rc1) next to new (rc2) on the same case and K

(no --old)

## Appendix A. Every defrag event

| job | run | case | K | frame | wall ms | drain ms | idle ms | in window |
|---|---|---|---|---|---|---|---|---|
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 39.1 | 10.8 | 28.3 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 38.8 | 10.7 | 28.1 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 38.8 | 10.6 | 28.2 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.1 | 10.9 | 28.2 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 39.0 | 10.9 | 28.1 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_C | cavity2d_n2840_k5 | 5 | 1000 | 154.4 | 10.8 | 143.6 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_C | cavity2d_n2840_k5 | 5 | 1000 | 154.7 | 10.8 | 143.9 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 39.3 | 10.9 | 28.4 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 39.0 | 10.7 | 28.2 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 39.5 | 10.7 | 28.8 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.4 | 10.9 | 28.5 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 39.5 | 10.8 | 28.7 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 38.7 | 10.8 | 27.8 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 39.3 | 10.7 | 28.6 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 38.7 | 10.8 | 27.9 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.7 | 11.1 | 28.7 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 39.0 | 10.8 | 28.2 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_C | cavity2d_n2840_k5 | 5 | 1000 | 155.4 | 10.8 | 144.5 | yes |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_trace_C | cavity2d_n2840_k5 | 5 | 1000 | 202.1 | 10.9 | 191.2 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 39.3 | 10.8 | 28.5 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 38.9 | 10.9 | 28.0 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 39.0 | 10.6 | 28.4 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.3 | 10.9 | 28.3 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 38.6 | 10.8 | 27.8 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_C | cavity2d_n2840_k5 | 5 | 1000 | 158.4 | 11.1 | 147.3 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_C | cavity2d_n2840_k5 | 5 | 1000 | 157.1 | 11.1 | 146.0 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 39.2 | 10.7 | 28.5 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 38.9 | 10.8 | 28.1 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 39.4 | 10.7 | 28.7 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.5 | 11.0 | 28.4 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 38.2 | 10.8 | 27.5 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g0 | cavity2d_n2840 | 1 | 1000 | 38.4 | 10.9 | 27.6 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g1 | cavity2d_n2840 | 1 | 1000 | 39.0 | 10.8 | 28.2 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g2 | cavity2d_n2840 | 1 | 1000 | 39.3 | 10.7 | 28.6 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g3 | cavity2d_n2840 | 1 | 1000 | 39.5 | 11.0 | 28.5 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g4 | cavity2d_n2840 | 1 | 1000 | 38.8 | 10.9 | 28.0 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_C | cavity2d_n2840_k5 | 5 | 1000 | 157.8 | 11.1 | 146.7 | yes |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_trace_C | cavity2d_n2840_k5 | 5 | 1000 | 204.2 | 11.1 | 193.1 | yes |

## Appendix B. ns per particle of every run

| job | run | role | K | particles | steady steps / s | fps | step ms | ns/particle | ns/particle (no defrag) | SM MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.84 | 92.3 | 10.840 | 1.323 | 1.319 | 2745 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.79 | 92.7 | 10.790 | 1.317 | 1.313 | 2767 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.72 | 93.3 | 10.720 | 1.309 | 1.304 | 2760 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 10.96 | 91.2 | 10.960 | 1.338 | 1.333 | 2715 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.82 | 92.4 | 10.820 | 1.321 | 1.317 | 2760 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t1_C | calibrated | 5 | 40703364 | 1000 / 10.96 | 91.2 | 10.960 | 1.346 | 1.328 | 2766 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_C | calibrated | 5 | 40703364 | 1000 / 10.97 | 91.2 | 10.970 | 1.348 | 1.329 | 2758 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.88 | 91.9 | 10.880 | 1.328 | 1.324 | 2730 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.81 | 92.5 | 10.810 | 1.320 | 1.314 | 2752 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.80 | 92.6 | 10.800 | 1.319 | 1.314 | 2752 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 11.01 | 90.8 | 11.010 | 1.344 | 1.339 | 2700 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t2_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.86 | 92.1 | 10.860 | 1.326 | 1.320 | 2745 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.92 | 91.6 | 10.920 | 1.333 | 1.328 | 2730 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.84 | 92.3 | 10.840 | 1.323 | 1.318 | 2752 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.81 | 92.5 | 10.810 | 1.320 | 1.315 | 2752 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 11.02 | 90.7 | 11.020 | 1.345 | 1.341 | 2700 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.87 | 92.0 | 10.870 | 1.327 | 1.323 | 2752 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_t3_C | calibrated | 5 | 40703364 | 1000 / 10.98 | 91.1 | 10.980 | 1.349 | 1.329 | 2755 |
| e7_b2AB_1718639 | 2d_n2840_k5_K5_trace_C | trace_calibrated | 5 | 40703364 | 1000 / 11.02 | 90.7 | 11.020 | 1.354 | 1.328 | 2758 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.86 | 92.1 | 10.860 | 1.326 | 1.321 | 2737 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.83 | 92.3 | 10.830 | 1.322 | 1.317 | 2760 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.65 | 93.9 | 10.650 | 1.300 | 1.296 | 2767 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 10.97 | 91.2 | 10.970 | 1.339 | 1.335 | 2707 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.85 | 92.2 | 10.850 | 1.325 | 1.320 | 2752 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t1_C | calibrated | 5 | 40703364 | 1000 / 11.23 | 89.0 | 11.230 | 1.379 | 1.360 | 2782 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_C | calibrated | 5 | 40703364 | 1000 / 11.21 | 89.2 | 11.210 | 1.377 | 1.358 | 2784 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.87 | 92.0 | 10.870 | 1.327 | 1.322 | 2737 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.82 | 92.4 | 10.820 | 1.321 | 1.316 | 2752 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.80 | 92.6 | 10.800 | 1.319 | 1.313 | 2767 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 11.02 | 90.7 | 11.020 | 1.345 | 1.340 | 2700 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t2_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.86 | 92.1 | 10.860 | 1.326 | 1.321 | 2752 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g0 | reference | 1 | 8191044 | 1000 / 10.91 | 91.7 | 10.910 | 1.332 | 1.328 | 2730 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g1 | reference | 1 | 8191044 | 1000 / 10.83 | 92.3 | 10.830 | 1.322 | 1.317 | 2752 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g2 | reference | 1 | 8191044 | 1000 / 10.81 | 92.5 | 10.810 | 1.320 | 1.315 | 2752 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g3 | reference | 1 | 8191044 | 1000 / 11.04 | 90.6 | 11.040 | 1.348 | 1.342 | 2700 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_R1_2d_n2840_g4 | reference | 1 | 8191044 | 1000 / 10.87 | 92.0 | 10.870 | 1.327 | 1.322 | 2745 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_t3_C | calibrated | 5 | 40703364 | 1000 / 11.24 | 89.0 | 11.240 | 1.381 | 1.362 | 2779 |
| e7_b2AB_1718639 | old_2d_n2840_k5_K5_trace_C | trace_calibrated | 5 | 40703364 | 1000 / 11.26 | 88.8 | 11.260 | 1.383 | 1.358 | 2781 |
