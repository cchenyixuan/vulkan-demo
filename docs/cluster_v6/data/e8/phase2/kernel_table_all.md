## Per-kernel time per step (E8 phase 2; medians over the steady steps, mean over the sims)

**3d_n152_K8** — 4M (160^3) K = 8, detail full, step period mean of sims 7.311 ms, G1 redundancy share 8.11 % (5.14–9.63 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 14.2 | 0.028 | 0.2 % | 8 |
| A | voxel | 23.2 | 0.045 | 0.3 % | 8 |
| A | ghost_trailing | 12.0 | 0.025 | 0.2 % | 7 |
| B | correction_density_interior | 687.4 | 1.216 | 9.4 % | 8 |
| B | force_deep_interior | 286.0 | 0.467 | 3.9 % | 8 |
| C | expand | 10.6 | 0.021 | 0.1 % | 8 |
| C | install_trailing | 2.4 | 0.005 | 0.0 % | 7 |
| C | append_departed | 2.0 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 1,779.3 | 3.626 | 24.3 % | 8 |
| C | density | 6.3 | 0.013 | 0.1 % | 8 |
| C | force | 1,793.4 | 3.658 | 24.5 % | 8 |
| A | ghost_leading | 11.4 | 0.023 | 0.2 % | 7 |
| C | install_leading | 4.1 | 0.008 | 0.1 % | 7 |

per sim (us / step): sim 0: predict 14, voxel 31, ghost_trailing 10, correction_density_interior 1543, force_deep_interior 1102, expand 9, install_trailing 4, append_departed 2, correction_density_boundary 1116, density 6, force 1052; sim 1: predict 12, voxel 20, ghost_leading 10, ghost_trailing 12, correction_density_interior 6, force_deep_interior 7, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 2121, density 6, force 2146; sim 2: predict 15, voxel 21, ghost_leading 12, ghost_trailing 12, correction_density_interior 607, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1929, density 6, force 1915; sim 3: predict 15, voxel 20, ghost_leading 12, ghost_trailing 12, correction_density_interior 585, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1924, density 6, force 1915; sim 4: predict 15, voxel 20, ghost_leading 12, ghost_trailing 12, correction_density_interior 597, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1935, density 6, force 1922; sim 5: predict 15, voxel 22, ghost_leading 12, ghost_trailing 12, correction_density_interior 581, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1967, density 6, force 2093; sim 6: predict 12, voxel 21, ghost_leading 10, ghost_trailing 12, correction_density_interior 6, force_deep_interior 7, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1958, density 6, force 1976; sim 7: predict 15, voxel 31, ghost_leading 10, correction_density_interior 1575, force_deep_interior 1156, expand 9, install_leading 4, append_departed 2, correction_density_boundary 1285, density 6, force 1328

**3d_n192_K8** — 8M (200^3) K = 8, detail full, step period mean of sims 9.980 ms, G1 redundancy share 8.82 % (5.91–10.19 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 25.5 | 0.025 | 0.3 % | 8 |
| A | voxel | 33.1 | 0.033 | 0.3 % | 8 |
| A | ghost_trailing | 15.7 | 0.016 | 0.2 % | 7 |
| B | correction_density_interior | 1,677.6 | 1.652 | 16.8 % | 8 |
| B | force_deep_interior | 461.5 | 0.413 | 4.6 % | 8 |
| C | expand | 19.2 | 0.020 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.003 | 0.0 % | 7 |
| C | append_departed | 3.8 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 2,640.5 | 2.684 | 26.5 % | 8 |
| C | density | 9.9 | 0.010 | 0.1 % | 8 |
| C | force | 2,693.2 | 2.740 | 27.0 % | 8 |
| A | ghost_leading | 14.3 | 0.015 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.004 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 32, voxel 37, ghost_trailing 14, correction_density_interior 2636, force_deep_interior 1890, expand 12, install_trailing 4, append_departed 2, correction_density_boundary 1760, density 8, force 1622; sim 1: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1328, force_deep_interior 9, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2868, density 11, force 2978; sim 2: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1410, force_deep_interior 10, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2845, density 10, force 2966; sim 3: predict 23, voxel 31, ghost_leading 14, ghost_trailing 14, correction_density_interior 1401, force_deep_interior 6, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2859, density 10, force 2993; sim 4: predict 24, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1374, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2845, density 10, force 2975; sim 5: predict 24, voxel 32, ghost_leading 14, ghost_trailing 16, correction_density_interior 1399, force_deep_interior 7, expand 23, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 3054, density 10, force 2973; sim 6: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1388, force_deep_interior 6, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 3054, density 10, force 3132; sim 7: predict 32, voxel 41, ghost_leading 14, correction_density_interior 2485, force_deep_interior 1757, expand 11, install_leading 4, append_departed 4, correction_density_boundary 1840, density 8, force 1906

**3d_n244_K8** — 16M (252^3) K = 8, detail full, step period mean of sims 15.814 ms, G1 redundancy share 8.32 % (4.81–9.56 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 86.5 | 0.043 | 0.5 % | 8 |
| A | voxel | 50.9 | 0.025 | 0.3 % | 8 |
| A | ghost_trailing | 24.3 | 0.013 | 0.2 % | 7 |
| B | correction_density_interior | 3,578.2 | 1.753 | 22.6 % | 8 |
| B | force_deep_interior | 2,165.3 | 1.035 | 13.7 % | 8 |
| C | expand | 30.7 | 0.016 | 0.2 % | 8 |
| C | install_trailing | 3.5 | 0.002 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.002 | 0.0 % | 8 |
| C | correction_density_boundary | 3,943.0 | 2.026 | 25.0 % | 8 |
| C | density | 19.9 | 0.010 | 0.1 % | 8 |
| C | force | 4,131.2 | 2.121 | 26.1 % | 8 |
| A | ghost_leading | 22.5 | 0.012 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.002 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 101, voxel 59, ghost_trailing 23, correction_density_interior 5500, force_deep_interior 4272, expand 17, install_trailing 4, append_departed 4, correction_density_boundary 2284, density 21, force 2492; sim 1: predict 72, voxel 43, ghost_leading 23, ghost_trailing 25, correction_density_interior 2527, force_deep_interior 1046, expand 37, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 4432, density 19, force 4538; sim 2: predict 92, voxel 49, ghost_leading 23, ghost_trailing 25, correction_density_interior 3210, force_deep_interior 1844, expand 33, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4497, density 21, force 4647; sim 3: predict 73, voxel 47, ghost_leading 23, ghost_trailing 25, correction_density_interior 2613, force_deep_interior 1060, expand 37, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 4495, density 19, force 4627; sim 4: predict 93, voxel 51, ghost_leading 23, ghost_trailing 25, correction_density_interior 3322, force_deep_interior 1859, expand 33, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4496, density 21, force 4659; sim 5: predict 92, voxel 50, ghost_leading 23, ghost_trailing 25, correction_density_interior 3308, force_deep_interior 1866, expand 33, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4505, density 21, force 4656; sim 6: predict 73, voxel 47, ghost_leading 23, ghost_trailing 25, correction_density_interior 2550, force_deep_interior 1075, expand 37, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4475, density 19, force 4794; sim 7: predict 96, voxel 61, ghost_leading 23, correction_density_interior 5594, force_deep_interior 4300, expand 17, install_leading 4, append_departed 4, correction_density_boundary 2359, density 21, force 2636

**3d_n312_K8** — 32M (320^3) K = 8, detail full, step period mean of sims 29.610 ms, G1 redundancy share 7.23 % (4.18–8.37 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 222.3 | 0.054 | 0.8 % | 8 |
| A | voxel | 125.5 | 0.031 | 0.4 % | 8 |
| A | ghost_trailing | 40.0 | 0.010 | 0.1 % | 7 |
| B | correction_density_interior | 8,414.0 | 2.037 | 28.4 % | 8 |
| B | force_deep_interior | 6,083.4 | 1.462 | 20.5 % | 8 |
| C | expand | 50.3 | 0.013 | 0.2 % | 8 |
| C | install_trailing | 4.1 | 0.001 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 6,411.7 | 1.592 | 21.7 % | 8 |
| C | density | 43.6 | 0.011 | 0.1 % | 8 |
| C | force | 6,441.1 | 1.600 | 21.8 % | 8 |
| A | ghost_leading | 31.4 | 0.008 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 234, voxel 143, ghost_trailing 31, correction_density_interior 11343, force_deep_interior 9285, expand 23, install_trailing 4, append_departed 4, correction_density_boundary 3751, density 43, force 3805; sim 1: predict 229, voxel 127, ghost_leading 31, ghost_trailing 41, correction_density_interior 8148, force_deep_interior 5526, expand 59, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7307, density 45, force 7335; sim 2: predict 203, voxel 105, ghost_leading 32, ghost_trailing 43, correction_density_interior 6560, force_deep_interior 4168, expand 59, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7121, density 40, force 7192; sim 3: predict 230, voxel 127, ghost_leading 31, ghost_trailing 41, correction_density_interior 8096, force_deep_interior 5538, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7372, density 45, force 7421; sim 4: predict 228, voxel 127, ghost_leading 31, ghost_trailing 41, correction_density_interior 7875, force_deep_interior 5350, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7284, density 45, force 7327; sim 5: predict 202, voxel 106, ghost_leading 31, ghost_trailing 43, correction_density_interior 6542, force_deep_interior 4151, expand 59, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7133, density 39, force 7188; sim 6: predict 229, voxel 127, ghost_leading 31, ghost_trailing 41, correction_density_interior 7970, force_deep_interior 5456, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7331, density 45, force 7386; sim 7: predict 224, voxel 143, ghost_leading 33, correction_density_interior 10778, force_deep_interior 9193, expand 23, install_leading 4, append_departed 4, correction_density_boundary 3995, density 46, force 3875

**3d_n392_K8** — 64M (400^3) K = 8, detail full, step period mean of sims 57.474 ms, G1 redundancy share 5.69 % (3.38–6.59 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 470.0 | 0.059 | 0.8 % | 8 |
| A | voxel | 278.0 | 0.035 | 0.5 % | 8 |
| A | ghost_trailing | 66.4 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 18,062.0 | 2.252 | 31.4 % | 8 |
| B | force_deep_interior | 14,338.7 | 1.785 | 25.0 % | 8 |
| C | expand | 89.7 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.3 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 9,815.9 | 1.235 | 17.1 % | 8 |
| C | density | 94.0 | 0.012 | 0.2 % | 8 |
| C | force | 10,090.0 | 1.270 | 17.5 % | 8 |
| A | ghost_leading | 47.0 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 464, voxel 293, ghost_trailing 47, correction_density_interior 21560, force_deep_interior 18366, expand 36, install_trailing 4, append_departed 4, correction_density_boundary 5854, density 91, force 5902; sim 1: predict 499, voxel 289, ghost_leading 48, ghost_trailing 70, correction_density_interior 18505, force_deep_interior 14593, expand 108, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11388, density 102, force 11562; sim 2: predict 460, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16050, force_deep_interior 12165, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11003, density 91, force 11353; sim 3: predict 462, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16449, force_deep_interior 12538, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11268, density 92, force 11461; sim 4: predict 459, voxel 266, ghost_leading 46, ghost_trailing 70, correction_density_interior 16063, force_deep_interior 12150, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11069, density 92, force 11314; sim 5: predict 498, voxel 289, ghost_leading 47, ghost_trailing 70, correction_density_interior 18267, force_deep_interior 14447, expand 108, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11143, density 102, force 11626; sim 6: predict 459, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16287, force_deep_interior 12306, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11013, density 91, force 11475; sim 7: predict 460, voxel 289, ghost_leading 47, correction_density_interior 21315, force_deep_interior 18145, expand 38, install_leading 4, append_departed 4, correction_density_boundary 5789, density 90, force 6027

**3d_n192_k8_K8** — weak 8 x 8M K = 8, detail full, step period mean of sims 53.527 ms, G1 redundancy share 1.64 % (1.00–1.87 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 446.0 | 0.058 | 0.8 % | 8 |
| A | voxel | 276.7 | 0.036 | 0.5 % | 8 |
| A | ghost_trailing | 16.5 | 0.002 | 0.0 % | 7 |
| B | correction_density_interior | 23,506.8 | 3.045 | 43.9 % | 8 |
| B | force_deep_interior | 22,339.5 | 2.893 | 41.7 % | 8 |
| C | expand | 17.7 | 0.002 | 0.0 % | 8 |
| C | install_trailing | 2.3 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 2,627.5 | 0.341 | 4.9 % | 8 |
| C | density | 84.8 | 0.011 | 0.2 % | 8 |
| C | force | 2,754.2 | 0.358 | 5.1 % | 8 |
| A | ghost_leading | 16.4 | 0.002 | 0.0 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 452, voxel 285, ghost_trailing 16, correction_density_interior 24747, force_deep_interior 23834, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1646, density 84, force 1668; sim 1: predict 444, voxel 277, ghost_leading 16, ghost_trailing 16, correction_density_interior 23293, force_deep_interior 22059, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2946, density 86, force 3214; sim 2: predict 444, voxel 279, ghost_leading 16, ghost_trailing 17, correction_density_interior 22748, force_deep_interior 21503, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2924, density 85, force 3079; sim 3: predict 435, voxel 262, ghost_leading 16, ghost_trailing 17, correction_density_interior 22902, force_deep_interior 21684, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2987, density 84, force 3114; sim 4: predict 443, voxel 277, ghost_leading 16, ghost_trailing 17, correction_density_interior 23155, force_deep_interior 21901, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2961, density 84, force 3093; sim 5: predict 454, voxel 277, ghost_leading 16, ghost_trailing 16, correction_density_interior 23459, force_deep_interior 22235, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2945, density 86, force 3067; sim 6: predict 444, voxel 279, ghost_leading 16, ghost_trailing 16, correction_density_interior 23262, force_deep_interior 21999, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2986, density 84, force 3106; sim 7: predict 451, voxel 279, ghost_leading 16, correction_density_interior 24489, force_deep_interior 23501, expand 13, install_leading 4, append_departed 4, correction_density_boundary 1624, density 84, force 1692

**trace_3d_n152_K1** — 4M (160^3) K = 1, detail full, step period mean of sims 26.808 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 224.4 | 0.055 | 0.8 % | 1 |
| A | voxel | 122.9 | 0.030 | 0.5 % | 1 |
| B | correction_density_interior | 13,221.6 | 3.228 | 49.3 % | 1 |
| B | force_deep_interior | 12,763.4 | 3.116 | 47.6 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 24.1 | 0.006 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 224, voxel 123, correction_density_interior 13222, force_deep_interior 12763, correction_density_boundary 0, density 24, force 0

**trace_3d_n192_K1** — 8M (200^3) K = 1, detail full, step period mean of sims 51.902 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 462.9 | 0.058 | 0.9 % | 1 |
| A | voxel | 287.1 | 0.036 | 0.6 % | 1 |
| B | correction_density_interior | 25,720.5 | 3.215 | 49.6 % | 1 |
| B | force_deep_interior | 24,934.0 | 3.117 | 48.0 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 61.0 | 0.008 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 463, voxel 287, correction_density_interior 25721, force_deep_interior 24934, correction_density_boundary 0, density 61, force 0

**trace_3d_n244_K1** — 16M (252^3) K = 1, detail full, step period mean of sims 101.571 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 948.7 | 0.059 | 0.9 % | 1 |
| A | voxel | 583.7 | 0.036 | 0.6 % | 1 |
| B | correction_density_interior | 50,422.1 | 3.151 | 49.6 % | 1 |
| B | force_deep_interior | 49,287.3 | 3.080 | 48.5 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 153.2 | 0.010 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 949, voxel 584, correction_density_interior 50422, force_deep_interior 49287, correction_density_boundary 0, density 153, force 0

**trace_3d_n312_K1** — 32M (320^3) K = 1, detail full, step period mean of sims 201.860 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 1,973.5 | 0.060 | 1.0 % | 1 |
| A | voxel | 1,129.9 | 0.034 | 0.6 % | 1 |
| B | correction_density_interior | 97,109.5 | 2.964 | 48.1 % | 1 |
| B | force_deep_interior | 95,720.7 | 2.921 | 47.4 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 350.8 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 1973, voxel 1130, correction_density_interior 97109, force_deep_interior 95721, correction_density_boundary 0, density 351, force 0

**trace_3d_n392_K1** — 64M (400^3) K = 1, detail full, step period mean of sims 396.443 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 3,899.5 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,144.8 | 0.034 | 0.5 % | 1 |
| B | correction_density_interior | 186,150.9 | 2.909 | 47.0 % | 1 |
| B | force_deep_interior | 185,186.7 | 2.894 | 46.7 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 721.3 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 3899, voxel 2145, correction_density_interior 186151, force_deep_interior 185187, correction_density_boundary 0, density 721, force 0

**3d_n392_K2** — 64M (400^3) K = 2, detail full, step period mean of sims 201.399 ms, G1 redundancy share 0.98 % (0.95–1.01 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 1,946.0 | 0.061 | 1.0 % | 2 |
| A | voxel | 1,084.6 | 0.034 | 0.5 % | 2 |
| A | ghost_trailing | 47.1 | 0.002 | 0.0 % | 1 |
| B | correction_density_interior | 89,232.7 | 2.790 | 44.3 % | 2 |
| B | force_deep_interior | 88,580.8 | 2.768 | 44.0 % | 2 |
| C | expand | 34.8 | 0.001 | 0.0 % | 2 |
| C | install_trailing | 4.1 | 0.000 | 0.0 % | 1 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 2 |
| C | correction_density_boundary | 5,898.8 | 0.184 | 2.9 % | 2 |
| C | density | 383.6 | 0.012 | 0.2 % | 2 |
| C | force | 5,957.9 | 0.186 | 3.0 % | 2 |
| A | ghost_leading | 50.6 | 0.002 | 0.0 % | 1 |
| C | install_leading | 4.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 1908, voxel 1064, ghost_trailing 47, correction_density_interior 89214, force_deep_interior 87338, expand 34, install_trailing 4, append_departed 4, correction_density_boundary 5971, density 376, force 5984; sim 1: predict 1984, voxel 1105, ghost_leading 51, correction_density_interior 89251, force_deep_interior 89824, expand 36, install_leading 4, append_departed 4, correction_density_boundary 5827, density 391, force 5932

**3d_n392_K4** — 64M (400^3) K = 4, detail full, step period mean of sims 107.713 ms, G1 redundancy share 2.69 % (1.82–3.54 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 961.6 | 0.060 | 0.9 % | 4 |
| A | voxel | 570.2 | 0.036 | 0.5 % | 4 |
| A | ghost_trailing | 62.1 | 0.004 | 0.1 % | 3 |
| B | correction_density_interior | 43,243.9 | 2.701 | 40.1 % | 4 |
| B | force_deep_interior | 40,278.6 | 2.514 | 37.4 % | 4 |
| C | expand | 71.8 | 0.005 | 0.1 % | 4 |
| C | install_trailing | 2.7 | 0.000 | 0.0 % | 3 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 4 |
| C | correction_density_boundary | 8,657.4 | 0.545 | 8.1 % | 4 |
| C | density | 190.8 | 0.012 | 0.2 % | 4 |
| C | force | 8,818.8 | 0.555 | 8.2 % | 4 |
| A | ghost_leading | 49.1 | 0.003 | 0.0 % | 3 |
| C | install_leading | 4.1 | 0.000 | 0.0 % | 3 |

per sim (us / step): sim 0: predict 945, voxel 560, ghost_trailing 49, correction_density_interior 45415, force_deep_interior 42380, expand 38, install_trailing 4, append_departed 4, correction_density_boundary 5861, density 186, force 5956; sim 1: predict 982, voxel 565, ghost_leading 49, ghost_trailing 68, correction_density_interior 42334, force_deep_interior 38434, expand 106, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11414, density 197, force 11574; sim 2: predict 943, voxel 530, ghost_leading 49, ghost_trailing 70, correction_density_interior 39657, force_deep_interior 35823, expand 108, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11236, density 187, force 11576; sim 3: predict 976, voxel 626, ghost_leading 49, correction_density_interior 45569, force_deep_interior 44478, expand 36, install_leading 4, append_departed 4, correction_density_boundary 6119, density 193, force 6169

**3d_n192_k2_K2** — weak 2 x 8M K = 2, detail full, step period mean of sims 53.767 ms, G1 redundancy share 1.01 % (0.98–1.04 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 451.3 | 0.058 | 0.8 % | 2 |
| A | voxel | 281.8 | 0.036 | 0.5 % | 2 |
| A | ghost_trailing | 16.1 | 0.002 | 0.0 % | 1 |
| B | correction_density_interior | 24,537.7 | 3.130 | 45.6 % | 2 |
| B | force_deep_interior | 23,562.6 | 3.005 | 43.8 % | 2 |
| C | expand | 12.1 | 0.002 | 0.0 % | 2 |
| C | install_trailing | 4.1 | 0.001 | 0.0 % | 1 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 2 |
| C | correction_density_boundary | 1,633.2 | 0.208 | 3.0 % | 2 |
| C | density | 85.1 | 0.011 | 0.2 % | 2 |
| C | force | 1,662.5 | 0.212 | 3.1 % | 2 |
| A | ghost_leading | 16.4 | 0.002 | 0.0 % | 1 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 455, voxel 285, ghost_trailing 16, correction_density_interior 24780, force_deep_interior 23853, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1664, density 84, force 1670; sim 1: predict 447, voxel 279, ghost_leading 16, correction_density_interior 24295, force_deep_interior 23272, expand 11, install_leading 4, append_departed 4, correction_density_boundary 1602, density 86, force 1655

**3d_n192_k4_K4** — weak 4 x 8M K = 4, detail full, step period mean of sims 53.951 ms, G1 redundancy share 1.41 % (0.98–1.84 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 448.1 | 0.058 | 0.8 % | 4 |
| A | voxel | 279.6 | 0.036 | 0.5 % | 4 |
| A | ghost_trailing | 16.4 | 0.002 | 0.0 % | 3 |
| B | correction_density_interior | 23,893.3 | 3.078 | 44.3 % | 4 |
| B | force_deep_interior | 22,748.3 | 2.931 | 42.2 % | 4 |
| C | expand | 16.1 | 0.002 | 0.0 % | 4 |
| C | install_trailing | 2.7 | 0.000 | 0.0 % | 3 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 4 |
| C | correction_density_boundary | 2,284.3 | 0.295 | 4.2 % | 4 |
| C | density | 84.9 | 0.011 | 0.2 % | 4 |
| C | force | 2,370.4 | 0.306 | 4.4 % | 4 |
| A | ghost_leading | 16.3 | 0.002 | 0.0 % | 3 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 3 |

per sim (us / step): sim 0: predict 454, voxel 285, ghost_trailing 16, correction_density_interior 24730, force_deep_interior 23805, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1636, density 84, force 1662; sim 1: predict 445, voxel 276, ghost_leading 16, ghost_trailing 16, correction_density_interior 23272, force_deep_interior 22010, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2935, density 85, force 3081; sim 2: predict 443, voxel 278, ghost_leading 16, ghost_trailing 16, correction_density_interior 22935, force_deep_interior 21637, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2945, density 84, force 3059; sim 3: predict 450, voxel 279, ghost_leading 16, correction_density_interior 24635, force_deep_interior 23542, expand 13, install_leading 4, append_departed 4, correction_density_boundary 1621, density 86, force 1679

**3d_n496_K8** — 128M (504^3) K = 8, detail full, step period mean of sims 110.500 ms, G1 redundancy share 4.69 % (2.72–5.45 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 974.8 | 0.061 | 0.9 % | 8 |
| A | voxel | 571.1 | 0.036 | 0.5 % | 8 |
| A | ghost_trailing | 107.3 | 0.007 | 0.1 % | 7 |
| B | correction_density_interior | 37,177.4 | 2.316 | 33.6 % | 8 |
| B | force_deep_interior | 32,746.2 | 2.037 | 29.6 % | 8 |
| C | expand | 164.1 | 0.010 | 0.1 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 8 |
| C | correction_density_boundary | 15,524.9 | 0.983 | 14.1 % | 8 |
| C | density | 202.5 | 0.013 | 0.2 % | 8 |
| C | force | 15,926.1 | 1.008 | 14.4 % | 8 |
| A | ghost_leading | 82.8 | 0.005 | 0.1 % | 7 |
| C | install_leading | 5.4 | 0.000 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 1012, voxel 633, ghost_trailing 80, correction_density_interior 44114, force_deep_interior 40790, expand 78, install_trailing 4, append_departed 4, correction_density_boundary 8997, density 208, force 9455; sim 1: predict 945, voxel 537, ghost_leading 82, ghost_trailing 111, correction_density_interior 33970, force_deep_interior 29082, expand 193, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 17658, density 197, force 17934; sim 2: predict 941, voxel 536, ghost_leading 82, ghost_trailing 112, correction_density_interior 33652, force_deep_interior 28904, expand 193, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17532, density 197, force 17910; sim 3: predict 1006, voxel 576, ghost_leading 84, ghost_trailing 115, correction_density_interior 37260, force_deep_interior 32528, expand 192, install_leading 6, install_trailing 4, append_departed 4, correction_density_boundary 18004, density 209, force 18515; sim 4: predict 941, voxel 537, ghost_leading 84, ghost_trailing 111, correction_density_interior 33832, force_deep_interior 29067, expand 193, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17576, density 197, force 17926; sim 5: predict 1006, voxel 578, ghost_leading 82, ghost_trailing 113, correction_density_interior 36789, force_deep_interior 32118, expand 192, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17685, density 209, force 18092; sim 6: predict 940, voxel 538, ghost_leading 82, ghost_trailing 111, correction_density_interior 33877, force_deep_interior 29066, expand 193, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17666, density 197, force 18291; sim 7: predict 1008, voxel 636, ghost_leading 84, correction_density_interior 43924, force_deep_interior 40414, expand 78, install_leading 4, append_departed 4, correction_density_boundary 9081, density 207, force 9287

**3d_n416_K8** — 76M (424^3) K = 8, detail full, step period mean of sims 65.167 ms, G1 redundancy share 5.63 % (3.33–6.57 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 565.9 | 0.059 | 0.9 % | 8 |
| A | voxel | 328.0 | 0.034 | 0.5 % | 8 |
| A | ghost_trailing | 73.4 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 21,739.0 | 2.277 | 33.4 % | 8 |
| B | force_deep_interior | 17,717.2 | 1.852 | 27.2 % | 8 |
| C | expand | 104.9 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 8 |
| C | correction_density_boundary | 11,013.8 | 1.166 | 16.9 % | 8 |
| C | density | 114.4 | 0.012 | 0.2 % | 8 |
| C | force | 11,359.3 | 1.202 | 17.4 % | 8 |
| A | ghost_leading | 53.6 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.000 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 570, voxel 350, ghost_trailing 53, correction_density_interior 26392, force_deep_interior 22804, expand 44, install_trailing 4, append_departed 4, correction_density_boundary 6449, density 114, force 6603; sim 1: predict 565, voxel 321, ghost_leading 53, ghost_trailing 76, correction_density_interior 20179, force_deep_interior 15851, expand 126, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12371, density 115, force 12789; sim 2: predict 566, voxel 321, ghost_leading 53, ghost_trailing 77, correction_density_interior 20641, force_deep_interior 16237, expand 125, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12623, density 115, force 13035; sim 3: predict 567, voxel 321, ghost_leading 54, ghost_trailing 77, correction_density_interior 20474, force_deep_interior 16134, expand 125, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12484, density 115, force 12837; sim 4: predict 564, voxel 321, ghost_leading 55, ghost_trailing 76, correction_density_interior 20447, force_deep_interior 16083, expand 125, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12379, density 115, force 12915; sim 5: predict 564, voxel 320, ghost_leading 53, ghost_trailing 76, correction_density_interior 20774, force_deep_interior 16362, expand 125, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12754, density 115, force 13126; sim 6: predict 566, voxel 321, ghost_leading 53, ghost_trailing 80, correction_density_interior 20297, force_deep_interior 15950, expand 125, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 12488, density 115, force 12871; sim 7: predict 565, voxel 348, ghost_leading 53, correction_density_interior 24708, force_deep_interior 22316, expand 44, install_leading 4, append_departed 4, correction_density_boundary 6563, density 113, force 6699

**trace_3d_n416_K1** — 76M (424^3) K = 1, detail full, step period mean of sims 465.389 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 4,656.6 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,545.1 | 0.033 | 0.5 % | 1 |
| B | correction_density_interior | 218,939.3 | 2.872 | 47.0 % | 1 |
| B | force_deep_interior | 218,144.0 | 2.862 | 46.9 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 865.9 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 4657, voxel 2545, correction_density_interior 218939, force_deep_interior 218144, correction_density_boundary 0, density 866, force 0
