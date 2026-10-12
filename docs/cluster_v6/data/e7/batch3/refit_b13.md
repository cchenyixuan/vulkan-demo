## Time model refit (batch report item 10)

- job b1A: elapsed 1.75 h (sacct), prelude 179 s, 7 items, 57 run / calibration groups (..\vulkan-demo\logs\n56\2026-10-10\01_e7_b1A_1681465\e7_b1A_1681465.out)
- job b1B: elapsed 2.90 h (sacct), prelude 225 s, 12 items, 65 run / calibration groups (..\vulkan-demo\logs\n56\2026-10-10\02_e7_b1B_1681466\e7_b1B_1681466.out)
- job b3A2: elapsed 4.60 h (sacct), prelude 121 s, 5 items, 60 run / calibration groups (..\vulkan-demo\logs\n56\2026-10-11\01_e7_b3A2_1689810\e7_b3A2_1689810.out)
- job b3B: elapsed 2.22 h (sacct), prelude 111 s, 2 items, 27 run / calibration groups (..\vulkan-demo\logs\n56\2026-10-10\04_e7_b3B_1683994\e7_b3B_1683994.out)

### Model fields: plan and refit

| field | plan | refit | data |
|---|---|---|---|
| cost_2d (ns per particle-step) | 1.258 | 1.301 | 98 K = 1 reference runs (2d_n1440, 2d_n2840, 2d_n4000, 2d_n8000) |
| cost_3d (ns per particle-step) | 6.000 | 6.179 | 174 K = 1 reference runs (3d_n160, 3d_n160_k8, 3d_n200, 3d_n200_k8, 3d_n416) |
| floor_k1 (s per step) | 0.0014 | 0.0014 | not measured: the plan value |
| floor (ms per step) | 2: 1.8, 4: 4.5, 8: 8.0 | 2: 4.7, 4: 4.5, 8: 6.8 | K=2: 10 runs (2d_n1440_k2); K=8: 10 runs (2d_n4000) |
| efficiency | 2: 0.950, 4: 0.900, 8: 0.870 | 2: 1.034, 4: 0.967, 8: 0.954 | K=2: 12 runs (2d_n11320); K=4: 12 runs (2d_n2840_k4, 2d_n8000); K=8: 12 runs (2d_n11320, 2d_n8000) |
| efficiency_3d | - | 2: 0.929, 4: 0.894, 8: 0.881 | K=2: 18 runs (3d_n160_k8, 3d_n200_k2, 3d_n200_k8); K=4: 18 runs (3d_n160_k8, 3d_n200_k4, 3d_n200_k8); K=8: 18 runs (3d_n160_k8, 3d_n200_k8, 3d_n416) |
| cold_per_simulator | 17 | 2.199 | 86 groups (calibration pilots 28) |
| warm_per_simulator | 1.2 | 1.461 | 136 groups |
| load_per_million | 0.35 | 0.3864 | 194 groups (intercept 0.00 s -> run_overhead) |
| bootstrap_fixed | 2 | 0.3189 | 222 groups and pilots |
| bootstrap_per_million | 0.5 | 0.547 | 222 groups and pilots |
| post_per_million | 0.1 | 0.04164 | 163 groups without the seam check |
| post_per_slab | 1 | 0 | 163 groups without the seam check |
| seam_2d_per_million | 0.42 | 0.241 | 13 seam-checked groups, 0.57 x the plan's terms |
| seam_2d_fixed | 8 | 4.59 | 13 seam-checked groups, 0.57 x the plan's terms |
| seam_3d_per_million | 0.26 | 0 | least squares over 18 3-D seam-checked groups (trace write 15.0 s off) |
| seam_3d_per_million_per_slab | 0.16 | 0.09293 | least squares over 18 3-D seam-checked groups (trace write 15.0 s off) |
| calibration_rounds | 2 | 2 | 15 calibrations, pilots 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1 |
| calibration_fixed | 20 | -10.39 | 15 calibrations, pilots 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1 |
| pilot_steps | 500 | 500 | not measured: the plan value |
| run_overhead | 10 | 2.72 | 194 groups, std 16.5 s; outside the bench 5.5 s, read-in intercept 0.0 s |
| trace_write | 15 | 15 | not measured: the plan value |
| job_overhead | 300 | 165.5 | 4 jobs: 187 s, 233 s, 127 s, 115 s |
| anatomy_steps | 3000 | 3000 | not measured: the plan value |
| soak_steps | 17000 | 17000 | not measured: the plan value |
| cube_soak_seconds | 3600 | 3600 | not measured: the plan value |

### Step times (ms): measured mean over the passing runs, the plan's and the refit's model

| case | K | runs | kind | per card (M) | measured | plan | refit |
|---|---|---|---|---|---|---|---|
| cavity2d_n11320 | 2 | 12 | pair reference | 64.32 | 80.93 | 85.17 | 80.93 |
| cavity2d_n11320 | 8 | 6 | timing arm | 16.08 | 21.22 | 23.25 | 21.91 |
| cavity2d_n1440 | 1 | 10 | K = 1 reference | 2.14 | 2.76 | 2.69 | 2.78 |
| cavity2d_n1440_k2 | 2 | 10 | timing arm | 2.12 | 4.69 | 2.81 | 4.69 |
| cavity2d_n2840 | 1 | 12 | K = 1 reference | 8.19 | 10.89 | 10.30 | 10.65 |
| cavity2d_n2840_k4 | 4 | 6 | timing arm | 8.14 | 11.11 | 11.38 | 10.95 |
| cavity2d_n4000 | 1 | 40 | K = 1 reference | 16.18 | 21.34 | 20.35 | 21.04 |
| cavity2d_n4000 | 8 | 10 | timing arm | 2.02 | 6.78 | 8.00 | 6.78 |
| cavity2d_n8000 | 1 | 36 | K = 1 reference | 64.35 | 83.29 | 80.96 | 83.69 |
| cavity2d_n8000 | 4 | 6 | timing arm | 16.09 | 21.34 | 22.49 | 21.63 |
| cavity2d_n8000 | 8 | 6 | timing arm | 8.04 | 11.34 | 11.63 | 10.96 |
| cavity3d_n160 | 1 | 24 | K = 1 reference | 4.74 | 31.22 | 28.45 | 29.30 |
| cavity3d_n160_k8 | 1 | 42 | K = 1 reference | 36.35 | 229.76 | 218.12 | 224.61 |
| cavity3d_n160_k8 | 2 | 6 | timing arm | 18.18 | 123.46 | 114.80 | 120.91 |
| cavity3d_n160_k8 | 4 | 6 | timing arm | 9.09 | 64.02 | 60.59 | 62.82 |
| cavity3d_n160_k8 | 8 | 6 | timing arm | 4.54 | 32.27 | 31.34 | 31.85 |
| cavity3d_n200 | 1 | 42 | K = 1 reference | 9.00 | 59.77 | 53.99 | 55.60 |
| cavity3d_n200_k2 | 2 | 6 | timing arm | 8.83 | 60.43 | 55.74 | 58.71 |
| cavity3d_n200_k4 | 4 | 6 | timing arm | 8.74 | 60.76 | 58.26 | 60.41 |
| cavity3d_n200_k8 | 1 | 42 | K = 1 reference | 69.57 | 423.35 | 417.41 | 429.84 |
| cavity3d_n200_k8 | 2 | 6 | timing arm | 34.78 | 220.55 | 219.69 | 231.38 |
| cavity3d_n200_k8 | 4 | 6 | timing arm | 17.39 | 117.36 | 115.95 | 120.22 |
| cavity3d_n200_k8 | 8 | 6 | timing arm | 8.70 | 60.63 | 59.97 | 60.96 |
| cavity3d_n416 | 1 | 24 | K = 1 reference | 76.23 | 464.07 | 457.35 | 470.96 |
| cavity3d_n416 | 8 | 6 | timing arm | 9.53 | 66.29 | 65.71 | 66.79 |

### Items (min): the plan's estimate, the refit's (as planned, and on what the item ran), the timeline

| job | item | plan | refit | refit as run | measured | measured / plan |
|---|---|---|---|---|---|---|
| b1A | pre_2d_n8000_k1_x4 | 1.6 | 1.2 | 1.2 | 1.2 | 0.78 |
| b1A | pre_3d_n160_k8_k1_x4 | 1.3 | 0.9 | 0.9 | 0.9 | 0.76 |
| b1A | pre_3d_n200_k8_k1_x4 | 1.9 | 1.6 | 1.6 | 1.6 | 0.82 |
| b1A | 2d_n8000_K4 | 45.1 | 38.0 | 38.0 | 36.0 | 0.80 |
| b1A | 2d_n2840_k4_K4 | 21.8 | 14.5 | 14.5 | 14.0 | 0.64 |
| b1A | 3d_n160_k8_K4 | 47.3 | 40.8 | 40.8 | 40.8 | 0.86 |
| b1A | 2d_n1440_k2_K2 | 12.0 | 7.1 | 7.1 | 7.5 | 0.63 |
| b1A | **job (overhead 5.0 / 2.8 min)** | 135.9 | 106.7 | - | 105.2 | 0.77 |
| b1B | selftest_2d_n1000_K8 | 2.7 | 0.5 | 0.5 | 0.4 | 0.16 |
| b1B | pre_2d_n8000_k1_x8 | 1.6 | 1.2 | 1.2 | 2.0 | 1.24 |
| b1B | pre_2d_n9000_k1_x8 | 1.9 | 1.5 | 1.5 | 2.4 | 1.27 |
| b1B | pre_3d_n160_k8_k1_x8 | 1.3 | 0.9 | 0.9 | 1.3 | 1.01 |
| b1B | pre_3d_n200_k8_k1_x8 | 1.9 | 1.6 | 1.6 | 2.4 | 1.24 |
| b1B | pre_3d_n416_k1_x8 | 2.1 | 1.7 | 1.7 | 2.7 | 1.29 |
| b1B | pre_2d_n11320_pairs_x4 | 2.9 | 2.3 | 2.3 | 2.8 | 0.95 |
| b1B | 2d_n8000_K8 | 46.1 | 34.7 | 34.7 | 36.0 | 0.78 |
| b1B | 2d_n4000_K8 | 32.7 | 19.2 | 19.2 | 18.6 | 0.57 |
| b1B | 2d_n11320_K8 | 65.9 | 53.0 | 53.0 | 57.1 | 0.87 |
| b1B | 3d_n160_k8_K8 | 50.6 | 38.4 | 38.4 | 39.4 | 0.78 |
| b1B | soak_2d_n8000_K8 | 7.3 | 4.8 | 4.8 | 5.2 | 0.71 |
| b1B | **job (overhead 5.0 / 2.8 min)** | 221.9 | 162.4 | - | 174.1 | 0.78 |
| b3A2 | 3d_n200_k2_K2 | 26.2 | 21.9 | 21.9 | 23.4 | 0.89 |
| b3A2 | 3d_n200_k4_K4 | 32.9 | 26.0 | 26.0 | 27.8 | 0.84 |
| b3A2 | 3d_n160_k8_K2 | 55.9 | 52.6 | 50.8 | 51.4 | 0.92 |
| b3A2 | 3d_n200_k8_K2 | 100.9 | 99.1 | 99.1 | 94.6 | 0.94 |
| b3A2 | 3d_n200_k8_K4 | 82.0 | 76.0 | 76.0 | 76.3 | 0.93 |
| b3A2 | **job (overhead 5.0 / 2.8 min)** | - | - | - | 275.7 | - |
| b3B | 3d_n200_k8_K8 | 82.3 | 70.0 | 70.0 | 65.3 | 0.79 |
| b3B | 3d_n416_K8 | 82.8 | 71.1 | 69.0 | 65.9 | 0.80 |
| b3B | **job (overhead 5.0 / 2.8 min)** | 170.1 | 143.9 | - | 133.1 | 0.78 |

### Budgets (node-h; a batch's wall = its longer line)

| batch | plan A | plan B | plan total | refit A | refit B | refit total | refit wall (h) |
|---|---|---|---|---|---|---|---|
| 1 | 2.27 | 3.70 | 5.96 | 1.78 | 2.71 | 4.49 | 2.71 |
| 2 | 5.66 | 6.37 | 12.03 | 4.42 | 4.42 | 8.84 | 4.42 |
| 3 | 5.69 | 2.83 | 8.53 | 5.09 | 2.40 | 7.49 | 5.09 |
| 4 | 3.91 | 2.90 | 6.80 | 2.43 | 2.16 | 4.59 | 2.43 |

Batches 2-4: plan 27.4 node-h, refit 20.9 node-h (queue time not included).
