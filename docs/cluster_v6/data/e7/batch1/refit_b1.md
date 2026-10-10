## Time model refit (batch report item 10)

- job b1A: elapsed 1.75 h (sacct), prelude 179 s, 7 items, 57 run / calibration groups (C:\Users\cchen\PycharmProjects\vulkan-demo\logs\n56\2026-10-10\01_e7_b1A_1681465\e7_b1A_1681465.out)
- job b1B: elapsed 2.90 h (sacct), prelude 225 s, 12 items, 65 run / calibration groups (C:\Users\cchen\PycharmProjects\vulkan-demo\logs\n56\2026-10-10\02_e7_b1B_1681466\e7_b1B_1681466.out)

### Model fields: plan and refit

| field | plan | refit | data |
|---|---|---|---|
| cost_2d (ns per particle-step) | 1.258 | 1.301 | 98 K = 1 reference runs (2d_n1440, 2d_n2840, 2d_n4000, 2d_n8000) |
| cost_3d (ns per particle-step) | 6.000 | 6.336 | 60 K = 1 reference runs (3d_n160, 3d_n160_k8) |
| floor_k1 (s per step) | 0.0014 | 0.0014 | not measured: the plan value |
| floor (ms per step) | 2: 1.8, 4: 4.5, 8: 8.0 | 2: 4.7, 4: 4.5, 8: 6.8 | K=2: 10 runs (2d_n1440_k2); K=8: 10 runs (2d_n4000) |
| efficiency | 2: 0.950, 4: 0.900, 8: 0.870 | 2: 1.034, 4: 0.967, 8: 0.954 | K=2: 12 runs (2d_n11320); K=4: 12 runs (2d_n2840_k4, 2d_n8000); K=8: 12 runs (2d_n11320, 2d_n8000) |
| efficiency_3d | - | 2: 0.950, 4: 0.900, 8: 0.892 | K=4: 6 runs (3d_n160_k8); K=8: 6 runs (3d_n160_k8) |
| cold_per_simulator | 17 | 1.94 | 52 groups (calibration pilots 16) |
| warm_per_simulator | 1.2 | 0.9994 | 78 groups |
| load_per_million | 0.35 | 0.4314 | 114 groups (intercept 0.00 s -> run_overhead) |
| bootstrap_fixed | 2 | 0.512 | 130 groups and pilots |
| bootstrap_per_million | 0.5 | 0.5848 | 130 groups and pilots |
| post_per_million | 0.1 | 0.03055 | 97 groups without the seam check |
| post_per_slab | 1 | 0 | 97 groups without the seam check |
| seam_2d_per_million | 0.42 | 0.241 | 13 seam-checked groups, 0.57 x the plan's terms |
| seam_2d_fixed | 8 | 4.59 | 13 seam-checked groups, 0.57 x the plan's terms |
| seam_3d_per_million | 0.26 | 0 | least squares over 4 3-D seam-checked groups (trace write 15.0 s off) |
| seam_3d_per_million_per_slab | 0.16 | 0.05333 | least squares over 4 3-D seam-checked groups (trace write 15.0 s off) |
| calibration_rounds | 2 | 2 | 8 calibrations, pilots 2, 2, 2, 2, 2, 2, 2, 2 |
| calibration_fixed | 20 | -12.6 | 8 calibrations, pilots 2, 2, 2, 2, 2, 2, 2, 2 |
| pilot_steps | 500 | 500 | not measured: the plan value |
| run_overhead | 10 | 4.566 | 114 groups, std 14.2 s; outside the bench 6.2 s, read-in intercept 0.0 s |
| trace_write | 15 | 15 | not measured: the plan value |
| job_overhead | 300 | 210 | 2 jobs: 187 s, 233 s |
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
| cavity3d_n160 | 1 | 24 | K = 1 reference | 4.74 | 31.22 | 28.45 | 30.04 |
| cavity3d_n160_k8 | 1 | 36 | K = 1 reference | 36.35 | 229.55 | 218.12 | 230.34 |
| cavity3d_n160_k8 | 4 | 6 | timing arm | 9.09 | 64.02 | 60.59 | 64.02 |
| cavity3d_n160_k8 | 8 | 6 | timing arm | 4.54 | 32.27 | 31.34 | 32.27 |

### Items (min): the plan's estimate, the refit's (as planned, and on what the item ran), the timeline

| job | item | plan | refit | refit as run | measured | measured / plan |
|---|---|---|---|---|---|---|
| b1A | pre_2d_n8000_k1_x4 | 1.6 | 1.3 | 1.3 | 1.2 | 0.78 |
| b1A | pre_3d_n160_k8_k1_x4 | 1.3 | 0.9 | 0.9 | 0.9 | 0.76 |
| b1A | pre_3d_n200_k8_k1_x4 | 1.9 | 1.7 | 1.7 | 1.6 | 0.82 |
| b1A | 2d_n8000_K4 | 45.1 | 39.1 | 39.1 | 36.0 | 0.80 |
| b1A | 2d_n2840_k4_K4 | 21.8 | 15.0 | 15.0 | 14.0 | 0.64 |
| b1A | 3d_n160_k8_K4 | 47.3 | 42.0 | 42.0 | 40.8 | 0.86 |
| b1A | 2d_n1440_k2_K2 | 12.0 | 7.6 | 7.6 | 7.5 | 0.63 |
| b1A | **job (overhead 5.0 / 3.5 min)** | 135.9 | 111.2 | - | 105.2 | 0.77 |
| b1B | selftest_2d_n1000_K8 | 2.7 | 0.5 | 0.5 | 0.4 | 0.16 |
| b1B | pre_2d_n8000_k1_x8 | 1.6 | 1.3 | 1.3 | 2.0 | 1.24 |
| b1B | pre_2d_n9000_k1_x8 | 1.9 | 1.6 | 1.6 | 2.4 | 1.27 |
| b1B | pre_3d_n160_k8_k1_x8 | 1.3 | 0.9 | 0.9 | 1.3 | 1.01 |
| b1B | pre_3d_n200_k8_k1_x8 | 1.9 | 1.7 | 1.7 | 2.4 | 1.24 |
| b1B | pre_3d_n416_k1_x8 | 2.1 | 1.8 | 1.8 | 2.7 | 1.29 |
| b1B | pre_2d_n11320_pairs_x4 | 2.9 | 2.5 | 2.5 | 2.8 | 0.95 |
| b1B | 2d_n8000_K8 | 46.1 | 35.6 | 35.6 | 36.0 | 0.78 |
| b1B | 2d_n4000_K8 | 32.7 | 19.4 | 19.4 | 18.6 | 0.57 |
| b1B | 2d_n11320_K8 | 65.9 | 54.9 | 54.9 | 57.1 | 0.87 |
| b1B | 3d_n160_k8_K8 | 50.6 | 39.1 | 39.1 | 39.4 | 0.78 |
| b1B | soak_2d_n8000_K8 | 7.3 | 4.9 | 4.9 | 5.2 | 0.71 |
| b1B | **job (overhead 5.0 / 3.5 min)** | 221.9 | 167.7 | - | 174.1 | 0.78 |

### Budgets (node-h; a batch's wall = its longer line)

| batch | plan A | plan B | plan total | refit A | refit B | refit total | refit wall (h) |
|---|---|---|---|---|---|---|---|
| 1 | 2.27 | 3.70 | 5.96 | 1.85 | 2.80 | 4.65 | 2.80 |
| 2 | 5.66 | 6.37 | 12.03 | 4.59 | 4.55 | 9.14 | 4.59 |
| 3 | 5.69 | 2.83 | 8.53 | 5.23 | 2.46 | 7.69 | 5.23 |
| 4 | 3.91 | 2.90 | 6.80 | 2.54 | 2.18 | 4.72 | 2.54 |

Batches 2-4: plan 27.4 node-h, refit 21.5 node-h (queue time not included).
