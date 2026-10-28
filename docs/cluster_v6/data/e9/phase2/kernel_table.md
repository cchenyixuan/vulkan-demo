## Per-kernel time per step (E8 phase 2; medians over the steady steps, mean over the sims)

**3d_n392_K2** — 64M (400^3) K = 2, detail full, step period mean of sims 193.626 ms, G1 redundancy share 1.01 % (0.99–1.03 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 1,942.9 | 0.061 | 1.0 % | 2 |
| A | voxel | 1,086.1 | 0.034 | 0.6 % | 2 |
| A | ghost_trailing | 46.8 | 0.001 | 0.0 % | 1 |
| B | correction_density_interior | 85,302.6 | 2.666 | 44.1 % | 2 |
| B | force_deep_interior | 85,107.5 | 2.660 | 44.0 % | 2 |
| C | expand | 37.3 | 0.001 | 0.0 % | 2 |
| C | install_trailing | 4.1 | 0.000 | 0.0 % | 1 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 2 |
| C | correction_density_boundary | 5,845.5 | 0.183 | 3.0 % | 2 |
| C | density | 382.7 | 0.012 | 0.2 % | 2 |
| C | force | 5,985.2 | 0.187 | 3.1 % | 2 |
| A | ghost_leading | 47.3 | 0.001 | 0.0 % | 1 |
| C | install_leading | 4.1 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 1945, voxel 1094, ghost_trailing 47, correction_density_interior 86789, force_deep_interior 85482, expand 39, install_trailing 4, append_departed 4, correction_density_boundary 5905, density 383, force 6036; sim 1: predict 1940, voxel 1078, ghost_leading 47, correction_density_interior 83816, force_deep_interior 84733, expand 36, install_leading 4, append_departed 4, correction_density_boundary 5786, density 382, force 5934

**3d_n392_K4** — 64M (400^3) K = 4, detail full, step period mean of sims 98.774 ms, G1 redundancy share 2.88 % (1.95–3.83 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 962.0 | 0.060 | 1.0 % | 4 |
| A | voxel | 569.2 | 0.036 | 0.6 % | 4 |
| A | ghost_trailing | 61.0 | 0.004 | 0.1 % | 3 |
| B | correction_density_interior | 39,388.3 | 2.461 | 39.9 % | 4 |
| B | force_deep_interior | 36,918.0 | 2.305 | 37.4 % | 4 |
| C | expand | 71.7 | 0.005 | 0.1 % | 4 |
| C | install_trailing | 3.3 | 0.000 | 0.0 % | 3 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 4 |
| C | correction_density_boundary | 8,543.0 | 0.538 | 8.6 % | 4 |
| C | density | 190.8 | 0.012 | 0.2 % | 4 |
| C | force | 8,752.9 | 0.551 | 8.9 % | 4 |
| A | ghost_leading | 47.9 | 0.003 | 0.0 % | 3 |
| C | install_leading | 4.1 | 0.000 | 0.0 % | 3 |

per sim (us / step): sim 0: predict 945, voxel 561, ghost_trailing 47, correction_density_interior 41160, force_deep_interior 38893, expand 36, install_trailing 4, append_departed 4, correction_density_boundary 5781, density 186, force 5866; sim 1: predict 980, voxel 563, ghost_leading 48, ghost_trailing 66, correction_density_interior 38046, force_deep_interior 34924, expand 105, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 11228, density 197, force 11519; sim 2: predict 943, voxel 531, ghost_leading 47, ghost_trailing 69, correction_density_interior 36606, force_deep_interior 33466, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11304, density 188, force 11699; sim 3: predict 980, voxel 622, ghost_leading 49, correction_density_interior 41741, force_deep_interior 40389, expand 38, install_leading 4, append_departed 4, correction_density_boundary 5858, density 193, force 5927

**3d_n192_k2_K2** — weak 2 x 8M K = 2, detail full, step period mean of sims 46.765 ms, G1 redundancy share 1.16 % (1.14–1.17 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 443.6 | 0.057 | 0.9 % | 2 |
| A | voxel | 272.6 | 0.035 | 0.6 % | 2 |
| A | ghost_trailing | 16.1 | 0.002 | 0.0 % | 1 |
| B | correction_density_interior | 21,452.7 | 2.736 | 45.9 % | 2 |
| B | force_deep_interior | 20,526.2 | 2.618 | 43.9 % | 2 |
| C | expand | 12.2 | 0.002 | 0.0 % | 2 |
| C | install_trailing | 4.1 | 0.001 | 0.0 % | 1 |
| C | append_departed | 3.1 | 0.000 | 0.0 % | 2 |
| C | correction_density_boundary | 1,625.4 | 0.207 | 3.5 % | 2 |
| C | density | 84.1 | 0.011 | 0.2 % | 2 |
| C | force | 1,657.8 | 0.211 | 3.5 % | 2 |
| A | ghost_leading | 14.3 | 0.002 | 0.0 % | 1 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 452, voxel 275, ghost_trailing 16, correction_density_interior 21544, force_deep_interior 20630, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1648, density 84, force 1653; sim 1: predict 435, voxel 270, ghost_leading 14, correction_density_interior 21361, force_deep_interior 20422, expand 11, install_leading 4, append_departed 2, correction_density_boundary 1602, density 84, force 1663

**3d_n192_k4_K4** — weak 4 x 8M K = 4, detail full, step period mean of sims 46.760 ms, G1 redundancy share 1.63 % (1.15–2.11 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 441.6 | 0.057 | 0.9 % | 4 |
| A | voxel | 269.4 | 0.035 | 0.6 % | 4 |
| A | ghost_trailing | 16.5 | 0.002 | 0.0 % | 3 |
| B | correction_density_interior | 20,670.6 | 2.663 | 44.2 % | 4 |
| B | force_deep_interior | 19,782.4 | 2.548 | 42.3 % | 4 |
| C | expand | 16.2 | 0.002 | 0.0 % | 4 |
| C | install_trailing | 2.7 | 0.000 | 0.0 % | 3 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 4 |
| C | correction_density_boundary | 2,285.0 | 0.296 | 4.9 % | 4 |
| C | density | 83.6 | 0.011 | 0.2 % | 4 |
| C | force | 2,372.2 | 0.308 | 5.1 % | 4 |
| A | ghost_leading | 16.1 | 0.002 | 0.0 % | 3 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 3 |

per sim (us / step): sim 0: predict 452, voxel 275, ghost_trailing 16, correction_density_interior 21421, force_deep_interior 20500, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1616, density 84, force 1639; sim 1: predict 439, voxel 270, ghost_leading 16, ghost_trailing 16, correction_density_interior 20088, force_deep_interior 18979, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2935, density 84, force 3093; sim 2: predict 430, voxel 258, ghost_leading 16, ghost_trailing 17, correction_density_interior 19884, force_deep_interior 18794, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2968, density 82, force 3081; sim 3: predict 445, voxel 274, ghost_leading 16, correction_density_interior 21290, force_deep_interior 20857, expand 13, install_leading 4, append_departed 4, correction_density_boundary 1622, density 84, force 1675

**3d_n152_K8** — 4M (160^3) K = 8, detail full, step period mean of sims 6.235 ms, G1 redundancy share 9.60 % (6.01–11.58 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 14.0 | 0.028 | 0.2 % | 8 |
| A | voxel | 21.7 | 0.043 | 0.3 % | 8 |
| A | ghost_trailing | 12.0 | 0.025 | 0.2 % | 7 |
| B | correction_density_interior | 557.6 | 0.987 | 9.0 % | 8 |
| B | force_deep_interior | 228.5 | 0.374 | 3.7 % | 8 |
| C | expand | 10.6 | 0.021 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.005 | 0.0 % | 7 |
| C | append_departed | 2.0 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 1,796.2 | 3.661 | 28.8 % | 8 |
| C | density | 6.3 | 0.013 | 0.1 % | 8 |
| C | force | 1,782.5 | 3.648 | 28.6 % | 8 |
| A | ghost_leading | 11.4 | 0.023 | 0.2 % | 7 |
| C | install_leading | 4.1 | 0.008 | 0.1 % | 7 |

per sim (us / step): sim 0: predict 14, voxel 29, ghost_trailing 10, correction_density_interior 1275, force_deep_interior 888, expand 9, install_trailing 4, append_departed 2, correction_density_boundary 1110, density 7, force 1031; sim 1: predict 12, voxel 20, ghost_leading 10, ghost_trailing 12, correction_density_interior 6, force_deep_interior 7, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 2125, density 6, force 2066; sim 2: predict 15, voxel 20, ghost_leading 12, ghost_trailing 12, correction_density_interior 482, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1914, density 6, force 1908; sim 3: predict 15, voxel 20, ghost_leading 12, ghost_trailing 12, correction_density_interior 471, force_deep_interior 4, expand 11, install_leading 4, install_trailing 4, append_departed 2, correction_density_boundary 1931, density 6, force 1929; sim 4: predict 15, voxel 19, ghost_leading 12, ghost_trailing 12, correction_density_interior 478, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1907, density 6, force 1894; sim 5: predict 12, voxel 20, ghost_leading 10, ghost_trailing 12, correction_density_interior 6, force_deep_interior 7, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 1958, density 6, force 2207; sim 6: predict 15, voxel 20, ghost_leading 12, ghost_trailing 12, correction_density_interior 487, force_deep_interior 4, expand 11, install_leading 4, install_trailing 2, append_departed 2, correction_density_boundary 2163, density 6, force 1940; sim 7: predict 14, voxel 25, ghost_leading 10, correction_density_interior 1255, force_deep_interior 910, expand 9, install_leading 4, append_departed 2, correction_density_boundary 1262, density 6, force 1285

**3d_n192_K8** — 8M (200^3) K = 8, detail full, step period mean of sims 8.785 ms, G1 redundancy share 9.93 % (6.57–11.82 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 27.2 | 0.027 | 0.3 % | 8 |
| A | voxel | 30.0 | 0.030 | 0.3 % | 8 |
| A | ghost_trailing | 16.1 | 0.016 | 0.2 % | 7 |
| B | correction_density_interior | 1,382.2 | 1.355 | 15.7 % | 8 |
| B | force_deep_interior | 394.7 | 0.353 | 4.5 % | 8 |
| C | expand | 19.3 | 0.020 | 0.2 % | 8 |
| C | install_trailing | 3.2 | 0.003 | 0.0 % | 7 |
| C | append_departed | 3.8 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 2,617.4 | 2.661 | 29.8 % | 8 |
| C | density | 10.4 | 0.010 | 0.1 % | 8 |
| C | force | 2,721.7 | 2.767 | 31.0 % | 8 |
| A | ghost_leading | 14.3 | 0.015 | 0.2 % | 7 |
| C | install_leading | 4.1 | 0.004 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 33, voxel 33, ghost_trailing 14, correction_density_interior 2294, force_deep_interior 1554, expand 12, install_trailing 4, append_departed 2, correction_density_boundary 1725, density 11, force 1734; sim 1: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1103, force_deep_interior 10, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2883, density 10, force 3012; sim 2: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1077, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2813, density 10, force 2936; sim 3: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1083, force_deep_interior 7, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 2869, density 10, force 3004; sim 4: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1067, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2837, density 10, force 2971; sim 5: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1091, force_deep_interior 6, expand 23, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 2905, density 10, force 3034; sim 6: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1099, force_deep_interior 6, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 3112, density 10, force 3171; sim 7: predict 35, voxel 35, ghost_leading 14, correction_density_interior 2245, force_deep_interior 1561, expand 11, install_leading 4, append_departed 4, correction_density_boundary 1795, density 10, force 1911

**3d_n244_K8** — 16M (252^3) K = 8, detail full, step period mean of sims 14.280 ms, G1 redundancy share 9.26 % (5.33–10.64 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 91.4 | 0.046 | 0.6 % | 8 |
| A | voxel | 45.1 | 0.022 | 0.3 % | 8 |
| A | ghost_trailing | 24.3 | 0.013 | 0.2 % | 7 |
| B | correction_density_interior | 3,179.5 | 1.558 | 22.3 % | 8 |
| B | force_deep_interior | 1,911.8 | 0.915 | 13.4 % | 8 |
| C | expand | 30.8 | 0.016 | 0.2 % | 8 |
| C | install_trailing | 3.2 | 0.002 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.002 | 0.0 % | 8 |
| C | correction_density_boundary | 3,967.5 | 2.040 | 27.8 % | 8 |
| C | density | 18.4 | 0.009 | 0.1 % | 8 |
| C | force | 4,139.4 | 2.125 | 29.0 % | 8 |
| A | ghost_leading | 24.6 | 0.013 | 0.2 % | 7 |
| C | install_leading | 4.1 | 0.002 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 105, voxel 55, ghost_trailing 23, correction_density_interior 4883, force_deep_interior 3727, expand 17, install_trailing 4, append_departed 4, correction_density_boundary 2272, density 21, force 2486; sim 1: predict 83, voxel 39, ghost_leading 25, ghost_trailing 25, correction_density_interior 2274, force_deep_interior 948, expand 37, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 4486, density 17, force 4614; sim 2: predict 99, voxel 45, ghost_leading 25, ghost_trailing 25, correction_density_interior 2923, force_deep_interior 1640, expand 34, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 4525, density 19, force 4637; sim 3: predict 84, voxel 39, ghost_leading 25, ghost_trailing 25, correction_density_interior 2290, force_deep_interior 964, expand 37, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4537, density 17, force 4651; sim 4: predict 98, voxel 47, ghost_leading 25, ghost_trailing 25, correction_density_interior 2926, force_deep_interior 1638, expand 34, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4508, density 19, force 4795; sim 5: predict 99, voxel 45, ghost_leading 25, ghost_trailing 25, correction_density_interior 2960, force_deep_interior 1659, expand 34, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 4555, density 19, force 4710; sim 6: predict 83, voxel 39, ghost_leading 25, ghost_trailing 25, correction_density_interior 2281, force_deep_interior 953, expand 37, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 4505, density 17, force 4641; sim 7: predict 80, voxel 51, ghost_leading 25, correction_density_interior 4900, force_deep_interior 3764, expand 16, install_leading 4, append_departed 4, correction_density_boundary 2352, density 21, force 2581

**3d_n312_K8** — 32M (320^3) K = 8, detail full, step period mean of sims 27.316 ms, G1 redundancy share 7.72 % (4.52–8.89 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 224.4 | 0.055 | 0.8 % | 8 |
| A | voxel | 122.9 | 0.030 | 0.4 % | 8 |
| A | ghost_trailing | 38.3 | 0.010 | 0.1 % | 7 |
| B | correction_density_interior | 7,410.8 | 1.793 | 27.1 % | 8 |
| B | force_deep_interior | 5,374.5 | 1.291 | 19.7 % | 8 |
| C | expand | 50.5 | 0.013 | 0.2 % | 8 |
| C | install_trailing | 4.1 | 0.001 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 6,323.6 | 1.572 | 23.2 % | 8 |
| C | density | 41.9 | 0.010 | 0.2 % | 8 |
| C | force | 6,405.2 | 1.591 | 23.5 % | 8 |
| A | ghost_leading | 31.4 | 0.008 | 0.1 % | 7 |
| C | install_leading | 4.4 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 236, voxel 137, ghost_trailing 31, correction_density_interior 9972, force_deep_interior 8138, expand 23, install_trailing 4, append_departed 4, correction_density_boundary 3694, density 41, force 3750; sim 1: predict 233, voxel 125, ghost_leading 31, ghost_trailing 39, correction_density_interior 7067, force_deep_interior 4867, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7291, density 43, force 7306; sim 2: predict 208, voxel 104, ghost_leading 31, ghost_trailing 41, correction_density_interior 5669, force_deep_interior 3588, expand 58, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7032, density 39, force 7075; sim 3: predict 233, voxel 125, ghost_leading 31, ghost_trailing 39, correction_density_interior 6926, force_deep_interior 4792, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7167, density 43, force 7206; sim 4: predict 232, voxel 125, ghost_leading 31, ghost_trailing 39, correction_density_interior 6957, force_deep_interior 4807, expand 59, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7192, density 43, force 7245; sim 5: predict 209, voxel 104, ghost_leading 33, ghost_trailing 41, correction_density_interior 5830, force_deep_interior 3670, expand 59, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7214, density 39, force 7269; sim 6: predict 233, voxel 125, ghost_leading 31, ghost_trailing 39, correction_density_interior 6998, force_deep_interior 4825, expand 60, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 7249, density 43, force 7274; sim 7: predict 213, voxel 137, ghost_leading 32, correction_density_interior 9868, force_deep_interior 8309, expand 23, install_leading 6, append_departed 4, correction_density_boundary 3751, density 43, force 4116

**3d_n392_K8** — 64M (400^3) K = 8, detail full, step period mean of sims 53.609 ms, G1 redundancy share 6.05 % (3.60–7.00 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 470.8 | 0.059 | 0.9 % | 8 |
| A | voxel | 276.3 | 0.035 | 0.5 % | 8 |
| A | ghost_trailing | 66.3 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 16,017.4 | 1.997 | 29.9 % | 8 |
| B | force_deep_interior | 12,866.7 | 1.601 | 24.0 % | 8 |
| C | expand | 89.0 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 9,745.9 | 1.227 | 18.2 % | 8 |
| C | density | 93.9 | 0.012 | 0.2 % | 8 |
| C | force | 9,971.2 | 1.255 | 18.6 % | 8 |
| A | ghost_leading | 46.0 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 463, voxel 291, ghost_trailing 47, correction_density_interior 19044, force_deep_interior 16270, expand 38, install_trailing 4, append_departed 4, correction_density_boundary 5743, density 91, force 5775; sim 1: predict 461, voxel 264, ghost_leading 45, ghost_trailing 70, correction_density_interior 14480, force_deep_interior 11069, expand 105, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11057, density 92, force 11457; sim 2: predict 500, voxel 287, ghost_leading 46, ghost_trailing 69, correction_density_interior 16240, force_deep_interior 12917, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11132, density 102, force 11402; sim 3: predict 462, voxel 264, ghost_leading 47, ghost_trailing 70, correction_density_interior 14162, force_deep_interior 10896, expand 106, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 10931, density 92, force 11128; sim 4: predict 460, voxel 264, ghost_leading 45, ghost_trailing 70, correction_density_interior 14205, force_deep_interior 10934, expand 105, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 10998, density 92, force 11212; sim 5: predict 502, voxel 287, ghost_leading 47, ghost_trailing 69, correction_density_interior 16350, force_deep_interior 13016, expand 107, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 11205, density 102, force 11513; sim 6: predict 461, voxel 264, ghost_leading 47, ghost_trailing 69, correction_density_interior 14317, force_deep_interior 11036, expand 106, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11073, density 92, force 11226; sim 7: predict 458, voxel 289, ghost_leading 45, correction_density_interior 19341, force_deep_interior 16796, expand 38, install_leading 4, append_departed 4, correction_density_boundary 5828, density 88, force 6056

**3d_n192_k8_K8** — weak 8 x 8M K = 8, detail full, step period mean of sims 46.898 ms, G1 redundancy share 1.86 % (1.15–2.21 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 440.1 | 0.057 | 0.9 % | 8 |
| A | voxel | 268.5 | 0.035 | 0.6 % | 8 |
| A | ghost_trailing | 16.4 | 0.002 | 0.0 % | 7 |
| B | correction_density_interior | 20,388.9 | 2.641 | 43.5 % | 8 |
| B | force_deep_interior | 19,344.2 | 2.505 | 41.2 % | 8 |
| C | expand | 17.8 | 0.002 | 0.0 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 2,618.6 | 0.340 | 5.6 % | 8 |
| C | density | 84.1 | 0.011 | 0.2 % | 8 |
| C | force | 2,719.3 | 0.353 | 5.8 % | 8 |
| A | ghost_leading | 15.7 | 0.002 | 0.0 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 453, voxel 276, ghost_trailing 16, correction_density_interior 21440, force_deep_interior 20565, expand 13, install_trailing 4, append_departed 4, correction_density_boundary 1612, density 84, force 1637; sim 1: predict 429, voxel 258, ghost_leading 16, ghost_trailing 17, correction_density_interior 19603, force_deep_interior 18566, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 3109, density 82, force 3223; sim 2: predict 448, voxel 273, ghost_leading 16, ghost_trailing 16, correction_density_interior 20398, force_deep_interior 19283, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2903, density 86, force 3026; sim 3: predict 439, voxel 268, ghost_leading 16, ghost_trailing 16, correction_density_interior 19918, force_deep_interior 18833, expand 19, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 2897, density 84, force 3034; sim 4: predict 438, voxel 268, ghost_leading 15, ghost_trailing 16, correction_density_interior 20096, force_deep_interior 18998, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2916, density 84, force 3057; sim 5: predict 440, voxel 266, ghost_leading 16, ghost_trailing 16, correction_density_interior 19916, force_deep_interior 18848, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2965, density 84, force 3044; sim 6: predict 439, voxel 268, ghost_leading 16, ghost_trailing 16, correction_density_interior 20152, force_deep_interior 19048, expand 19, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2931, density 84, force 3064; sim 7: predict 435, voxel 270, ghost_leading 15, correction_density_interior 21588, force_deep_interior 20613, expand 13, install_leading 4, append_departed 4, correction_density_boundary 1616, density 84, force 1670

**trace_3d_n152_K1** — 4M (160^3) K = 1, detail full, step period mean of sims 23.476 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 225.5 | 0.055 | 1.0 % | 1 |
| A | voxel | 123.3 | 0.030 | 0.5 % | 1 |
| B | correction_density_interior | 11,524.0 | 2.813 | 49.1 % | 1 |
| B | force_deep_interior | 11,002.1 | 2.686 | 46.9 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 26.3 | 0.006 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 225, voxel 123, correction_density_interior 11524, force_deep_interior 11002, correction_density_boundary 0, density 26, force 0

**trace_3d_n192_K1** — 8M (200^3) K = 1, detail full, step period mean of sims 45.870 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 464.1 | 0.058 | 1.0 % | 1 |
| A | voxel | 278.9 | 0.035 | 0.6 % | 1 |
| B | correction_density_interior | 22,660.4 | 2.833 | 49.4 % | 1 |
| B | force_deep_interior | 21,837.3 | 2.730 | 47.6 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 67.3 | 0.008 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 464, voxel 279, correction_density_interior 22660, force_deep_interior 21837, correction_density_boundary 0, density 67, force 0

**trace_3d_n244_K1** — 16M (252^3) K = 1, detail full, step period mean of sims 92.327 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 947.6 | 0.059 | 1.0 % | 1 |
| A | voxel | 579.0 | 0.036 | 0.6 % | 1 |
| B | correction_density_interior | 44,656.9 | 2.791 | 48.4 % | 1 |
| B | force_deep_interior | 43,130.8 | 2.695 | 46.7 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 154.3 | 0.010 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 948, voxel 579, correction_density_interior 44657, force_deep_interior 43131, correction_density_boundary 0, density 154, force 0

**trace_3d_n312_K1** — 32M (320^3) K = 1, detail full, step period mean of sims 189.607 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 1,970.6 | 0.060 | 1.0 % | 1 |
| A | voxel | 1,129.9 | 0.034 | 0.6 % | 1 |
| B | correction_density_interior | 90,965.7 | 2.776 | 48.0 % | 1 |
| B | force_deep_interior | 90,003.4 | 2.747 | 47.5 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 350.9 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 1971, voxel 1130, correction_density_interior 90966, force_deep_interior 90003, correction_density_boundary 0, density 351, force 0

**trace_3d_n392_K1** — 64M (400^3) K = 1, detail full, step period mean of sims 381.840 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 3,899.8 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,136.5 | 0.033 | 0.6 % | 1 |
| B | correction_density_interior | 179,043.9 | 2.798 | 46.9 % | 1 |
| B | force_deep_interior | 178,712.5 | 2.792 | 46.8 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 720.7 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 3900, voxel 2136, correction_density_interior 179044, force_deep_interior 178713, correction_density_boundary 0, density 721, force 0

**3d_n496_K8** — 128M (504^3) K = 8, detail full, step period mean of sims 101.096 ms, G1 redundancy share 5.11 % (2.93–5.88 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 974.2 | 0.061 | 1.0 % | 8 |
| A | voxel | 567.6 | 0.035 | 0.6 % | 8 |
| A | ghost_trailing | 105.2 | 0.007 | 0.1 % | 7 |
| B | correction_density_interior | 33,457.3 | 2.083 | 33.1 % | 8 |
| B | force_deep_interior | 29,514.2 | 1.835 | 29.2 % | 8 |
| C | expand | 163.3 | 0.010 | 0.2 % | 8 |
| C | install_trailing | 3.1 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 8 |
| C | correction_density_boundary | 15,485.2 | 0.981 | 15.3 % | 8 |
| C | density | 202.3 | 0.013 | 0.2 % | 8 |
| C | force | 15,820.5 | 1.002 | 15.7 % | 8 |
| A | ghost_leading | 81.3 | 0.005 | 0.1 % | 7 |
| C | install_leading | 5.4 | 0.000 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 1010, voxel 632, ghost_trailing 80, correction_density_interior 40522, force_deep_interior 37694, expand 77, install_trailing 4, append_departed 4, correction_density_boundary 8995, density 208, force 9215; sim 1: predict 940, voxel 532, ghost_leading 80, ghost_trailing 109, correction_density_interior 29925, force_deep_interior 25672, expand 192, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 17488, density 197, force 17783; sim 2: predict 941, voxel 531, ghost_leading 82, ghost_trailing 109, correction_density_interior 30525, force_deep_interior 26144, expand 193, install_leading 6, install_trailing 4, append_departed 4, correction_density_boundary 17702, density 197, force 18084; sim 3: predict 1005, voxel 577, ghost_leading 83, ghost_trailing 112, correction_density_interior 33110, force_deep_interior 28786, expand 191, install_leading 6, install_trailing 4, append_departed 4, correction_density_boundary 17705, density 209, force 18087; sim 4: predict 1004, voxel 577, ghost_leading 82, ghost_trailing 111, correction_density_interior 32791, force_deep_interior 28587, expand 191, install_leading 6, install_trailing 4, append_departed 4, correction_density_boundary 17679, density 209, force 18073; sim 5: predict 940, voxel 531, ghost_leading 80, ghost_trailing 109, correction_density_interior 30199, force_deep_interior 25890, expand 192, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17703, density 197, force 18075; sim 6: predict 942, voxel 533, ghost_leading 80, ghost_trailing 109, correction_density_interior 30171, force_deep_interior 26009, expand 192, install_leading 6, install_trailing 2, append_departed 4, correction_density_boundary 17611, density 197, force 17908; sim 7: predict 1011, voxel 628, ghost_leading 82, correction_density_interior 40416, force_deep_interior 37331, expand 79, install_leading 4, append_departed 4, correction_density_boundary 8998, density 206, force 9340

**3d_n416_K8** — 76M (424^3) K = 8, detail full, step period mean of sims 60.602 ms, G1 redundancy share 6.07 % (3.51–7.03 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 566.6 | 0.060 | 0.9 % | 8 |
| A | voxel | 326.1 | 0.034 | 0.5 % | 8 |
| A | ghost_trailing | 72.2 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 19,576.3 | 2.049 | 32.3 % | 8 |
| B | force_deep_interior | 16,041.4 | 1.677 | 26.4 % | 8 |
| C | expand | 104.3 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.000 | 0.0 % | 8 |
| C | correction_density_boundary | 11,025.9 | 1.167 | 18.2 % | 8 |
| C | density | 114.4 | 0.012 | 0.2 % | 8 |
| C | force | 11,354.6 | 1.202 | 18.8 % | 8 |
| A | ghost_leading | 52.9 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.000 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 570, voxel 347, ghost_trailing 51, correction_density_interior 23636, force_deep_interior 20478, expand 44, install_trailing 4, append_departed 4, correction_density_boundary 6411, density 114, force 6629; sim 1: predict 566, voxel 319, ghost_leading 53, ghost_trailing 76, correction_density_interior 18390, force_deep_interior 14562, expand 124, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12645, density 115, force 13001; sim 2: predict 565, voxel 319, ghost_leading 53, ghost_trailing 76, correction_density_interior 18128, force_deep_interior 14423, expand 124, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12426, density 115, force 12862; sim 3: predict 567, voxel 319, ghost_leading 53, ghost_trailing 76, correction_density_interior 18248, force_deep_interior 14509, expand 124, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12528, density 115, force 12806; sim 4: predict 565, voxel 320, ghost_leading 53, ghost_trailing 76, correction_density_interior 18282, force_deep_interior 14523, expand 124, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12562, density 115, force 12935; sim 5: predict 568, voxel 319, ghost_leading 53, ghost_trailing 76, correction_density_interior 18348, force_deep_interior 14560, expand 125, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 12573, density 115, force 13022; sim 6: predict 566, voxel 319, ghost_leading 53, ghost_trailing 76, correction_density_interior 18311, force_deep_interior 14519, expand 124, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 12480, density 115, force 12919; sim 7: predict 567, voxel 344, ghost_leading 51, correction_density_interior 23268, force_deep_interior 20758, expand 46, install_leading 4, append_departed 4, correction_density_boundary 6583, density 113, force 6662

**trace_3d_n416_K1** — 76M (424^3) K = 1, detail full, step period mean of sims 452.590 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 4,657.7 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,542.0 | 0.033 | 0.6 % | 1 |
| B | correction_density_interior | 213,429.6 | 2.800 | 47.2 % | 1 |
| B | force_deep_interior | 212,799.1 | 2.792 | 47.0 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 866.0 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 4658, voxel 2542, correction_density_interior 213430, force_deep_interior 212799, correction_density_boundary 0, density 866, force 0
