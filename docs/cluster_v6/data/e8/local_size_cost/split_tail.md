### split-dispatch tail series: per run (kernel times = step-trace medians over steps 100-299, ms)

| run | SPLIT_CD | SPLIT_FO | step (ms) | corr+dens (ms) | force (ms) | trace period (ms) | SM MHz | W | drift |
|---|---|---|---|---|---|---|---|---|---|
| 64m_cd2_r1 | 2 | 0 | 383.350 | 188.073 | 186.573 | 381.424 | 2767 | 575 | 0 |
| 64m_cd2_r2 | 2 | 0 | 383.900 | 188.348 | 186.891 | 382.022 | 2760 | 575 | 0 |
| 64m_cd4_r1 | 4 | 0 | 383.650 | 188.565 | 186.453 | 381.826 | 2767 | 575 | 0 |
| 64m_cd4_r2 | 4 | 0 | 383.900 | 188.627 | 186.544 | 381.943 | 2767 | 575 | 0 |
| 64m_cd8_r1 | 8 | 0 | 384.700 | 189.590 | 186.404 | 382.777 | 2767 | 575 | 0 |
| 64m_cd8_r2 | 8 | 0 | 385.300 | 189.830 | 186.730 | 383.348 | 2767 | 575 | 0 |
| 64m_fo2_r1 | 0 | 2 | 383.650 | 187.998 | 187.035 | 381.808 | 2767 | 575 | 0 |
| 64m_fo2_r2 | 0 | 2 | 383.350 | 187.894 | 186.843 | 381.547 | 2767 | 575 | 0 |
| 64m_fo4_r1 | 0 | 4 | 383.750 | 187.855 | 187.321 | 381.971 | 2767 | 575 | 0 |
| 64m_fo4_r2 | 0 | 4 | 383.750 | 187.790 | 187.251 | 381.819 | 2767 | 575 | 0 |
| 64m_fo8_r1 | 0 | 8 | 384.750 | 187.806 | 188.304 | 382.884 | 2767 | 575 | 0 |
| 64m_fo8_r2 | 0 | 8 | 384.750 | 187.721 | 188.350 | 382.865 | 2767 | 575 | 0 |
| 64m_m1_plain_r1 | 0 | 0 | 383.350 | 187.937 | 186.679 | 381.399 | 2760 | 575 | 0 |
| 64m_m1_plain_r2 | 0 | 0 | 383.250 | 187.921 | 186.626 | 381.301 | 2760 | 575 | 0 |
| 64m_m1_seg_r1 | 1 | 1 | 383.250 | 187.899 | 186.618 | 381.301 | 2760 | 575 | 0 |
| 64m_m1_seg_r2 | 1 | 1 | 383.400 | 187.983 | 186.697 | 381.448 | 2760 | 575 | 0 |
| 8m_cd2_r1 | 2 | 0 | 52.600 | 26.337 | 25.189 | 52.354 | 2808 | 575 | 0 |
| 8m_cd2_r2 | 2 | 0 | 52.550 | 26.323 | 25.142 | 52.297 | 2805 | 575 | 0 |
| 8m_cd4_r1 | 4 | 0 | 53.250 | 27.033 | 25.135 | 52.989 | 2812 | 575 | 0 |
| 8m_cd4_r2 | 4 | 0 | 53.250 | 27.060 | 25.100 | 52.990 | 2812 | 575 | 0 |
| 8m_cd8_r1 | 8 | 0 | 54.500 | 28.404 | 25.006 | 54.234 | 2835 | 575 | 0 |
| 8m_cd8_r2 | 8 | 0 | 54.500 | 28.410 | 24.986 | 54.219 | 2835 | 575 | 0 |
| 8m_fo2_r1 | 0 | 2 | 52.700 | 25.979 | 25.654 | 52.464 | 2808 | 575 | 0 |
| 8m_fo2_r2 | 0 | 2 | 52.500 | 25.899 | 25.540 | 52.267 | 2805 | 575 | 0 |
| 8m_fo4_r1 | 0 | 4 | 53.300 | 25.861 | 26.382 | 53.073 | 2820 | 575 | 0 |
| 8m_fo4_r2 | 0 | 4 | 53.450 | 25.933 | 26.414 | 53.176 | 2812 | 575 | 0 |
| 8m_fo8_r1 | 0 | 8 | 54.750 | 25.635 | 27.994 | 54.457 | 2842 | 575 | 0 |
| 8m_fo8_r2 | 0 | 8 | 54.550 | 25.659 | 27.819 | 54.305 | 2835 | 575 | 0 |
| 8m_m1_plain_r1 | 0 | 0 | 52.100 | 25.915 | 25.138 | 51.883 | 2808 | 575 | 0 |
| 8m_m1_plain_r2 | 0 | 0 | 52.350 | 26.019 | 25.246 | 52.086 | 2805 | 575 | 0 |
| 8m_m1_seg_r1 | 1 | 1 | 52.200 | 25.972 | 25.183 | 51.983 | 2805 | 575 | 0 |
| 8m_m1_seg_r2 | 1 | 1 | 52.200 | 25.957 | 25.166 | 51.943 | 2805 | 575 | 0 |

### T(m) - T(1) per kernel (the other kernel unsplit); T(1) = the m1_plain runs

**cube 8M (200^3)**: T(1) corr+dens 25.967 ms, force 25.192 ms, step 52.225 ms (m1_plain x2: steps [52.1, 52.35]); m1_seg (SEGMENT pipelines, first thread 0, no split): corr+dens 25.965, force 25.175, step 52.200

| kernel | m | runs | T_k(m) (ms) | T_k(m) - T_k(1) (ms) | per boundary (ms) | other kernel (ms) | step - step(1) (ms) |
|---|---|---|---|---|---|---|---|
| correction_density_interior | 2 | 2 | 26.330 (26.337, 26.323) | +0.363 | +0.363 | 25.165 (T(1) 25.192) | +0.350 |
| correction_density_interior | 4 | 2 | 27.046 (27.033, 27.060) | +1.079 | +0.360 | 25.117 (T(1) 25.192) | +1.025 |
| correction_density_interior | 8 | 2 | 28.407 (28.404, 28.410) | +2.440 | +0.349 | 24.996 (T(1) 25.192) | +2.275 |
| **correction_density_interior slope** | | | | slope through origin **+0.350 ms per boundary** (free fit +0.345 m + +0.028; rms 0.021; from step times +0.328) | | fixed term 7.16: **+2.628 ms** | ratio slope / fixed = **0.13** |
| force_deep_interior_scratch | 2 | 2 | 25.597 (25.654, 25.540) | +0.405 | +0.405 | 25.939 (T(1) 25.967) | +0.375 |
| force_deep_interior_scratch | 4 | 2 | 26.398 (26.382, 26.414) | +1.206 | +0.402 | 25.897 (T(1) 25.967) | +1.150 |
| force_deep_interior_scratch | 8 | 2 | 27.907 (27.994, 27.819) | +2.714 | +0.388 | 25.647 (T(1) 25.967) | +2.425 |
| **force_deep_interior_scratch slope** | | | | slope through origin **+0.390 ms per boundary** (free fit +0.384 m + +0.034; rms 0.066; from step times +0.353) | | fixed term 7.16: **+2.342 ms** | ratio slope / fixed = **0.17** |

**cube 64M (400^3)**: T(1) corr+dens 187.929 ms, force 186.653 ms, step 383.300 ms (m1_plain x2: steps [383.35, 383.25]); m1_seg (SEGMENT pipelines, first thread 0, no split): corr+dens 187.941, force 186.657, step 383.325

| kernel | m | runs | T_k(m) (ms) | T_k(m) - T_k(1) (ms) | per boundary (ms) | other kernel (ms) | step - step(1) (ms) |
|---|---|---|---|---|---|---|---|
| correction_density_interior | 2 | 2 | 188.211 (188.073, 188.348) | +0.282 | +0.282 | 186.732 (T(1) 186.653) | +0.325 |
| correction_density_interior | 4 | 2 | 188.596 (188.565, 188.627) | +0.667 | +0.222 | 186.498 (T(1) 186.653) | +0.475 |
| correction_density_interior | 8 | 2 | 189.710 (189.590, 189.830) | +1.781 | +0.254 | 186.567 (T(1) 186.653) | +1.700 |
| **correction_density_interior slope** | | | | slope through origin **+0.250 ms per boundary** (free fit +0.254 m + -0.021; rms 0.120; from step times +0.231) | | fixed term 7.16: **+1.643 ms** | ratio slope / fixed = **0.15** |
| force_deep_interior_scratch | 2 | 2 | 186.939 (187.035, 186.843) | +0.286 | +0.286 | 187.946 (T(1) 187.929) | +0.200 |
| force_deep_interior_scratch | 4 | 2 | 187.286 (187.321, 187.251) | +0.633 | +0.211 | 187.823 (T(1) 187.929) | +0.450 |
| force_deep_interior_scratch | 8 | 2 | 188.327 (188.304, 188.350) | +1.675 | +0.239 | 187.764 (T(1) 187.929) | +1.450 |
| **force_deep_interior_scratch slope** | | | | slope through origin **+0.236 ms per boundary** (free fit +0.236 m + +0.001; rms 0.081; from step times +0.198) | | fixed term 7.16: **+3.883 ms** | ratio slope / fixed = **0.06** |

### summary

| cube | kernel | T(1) (ms) | slope: T(m) - T(1) per (m - 1) (ms) | fixed term T_k - a_k N (ms) | slope / fixed |
|---|---|---|---|---|---|
| cube 8M (200^3) | correction_density_interior | 25.967 | +0.350 (free +0.345, rms 0.021) | +2.628 | 0.13 |
| cube 8M (200^3) | force_deep_interior_scratch | 25.192 | +0.390 (free +0.384, rms 0.066) | +2.342 | 0.17 |
| cube 64M (400^3) | correction_density_interior | 187.929 | +0.250 (free +0.254, rms 0.120) | +1.643 | 0.15 |
| cube 64M (400^3) | force_deep_interior_scratch | 186.653 | +0.236 (free +0.236, rms 0.081) | +3.883 | 0.06 |