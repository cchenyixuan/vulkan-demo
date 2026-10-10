## 01_e7_b1A_1681465

- host wqd10nah10g5, slurm job 1681465, harness 3d6182b6ca5b, experiment/v7 tree 82dd6a740fa2 (= v7-rc1), driver 580.82.07
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1681465; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 8 files, 55 MiB in /dev/shm/scxm138/glcache_1681465; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 14; equal pass 14; precheck pass 12; reference pass 46; trace_calibrated pass 4; trace_equal pass 4

### Pre-checks

| case | kind | processes | verdict | statuses | VRAM peak per card (MiB, min–max) | host: largest process / sum (GiB) | node in use peak (GiB) | build max (s) |
|---|---|---|---|---|---|---|---|---|
| cavity2d_n8000 | k1 | 4 | fits | pass | 21514–21578 | 9.3 / 37.1 | 52.4 | 62.9 |
| cavity3d_n160_k8 | k1 | 4 | fits | pass | 11934–11998 | 5.4 / 21.5 | 38.0 | 38.0 |
| cavity3d_n200_k8 | k1 | 4 | fits | pass | 22780–22844 | 10.0 / 40.0 | 54.4 | 67.5 |

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n8000_K4 | F2 | 4 | E | 46.51 ± 0.06 (med 46.52; 46.45–46.55; n=3) | strong:cavity2d_n8000: 0.972 ± 0.001 (med 0.972; 0.971–0.973; n=3) | strong:cavity2d_n8000: 0.986 ± 0.001 (med 0.986; 0.985–0.987; n=3) |
| 2d_n8000_K4 | F2 | 4 | C | 47.23 ± 0.05 (med 47.20; 47.19–47.28; n=3) | strong:cavity2d_n8000: 0.987 ± 0.003 (med 0.987; 0.985–0.990; n=3) | strong:cavity2d_n8000: 1.002 ± 0.003 (med 1.002; 0.999–1.004; n=3) |
| 2d_n2840_k4_K4 | F3 | 4 | E | 89.75 ± 0.14 (med 89.77; 89.61–89.89; n=3) | weak:cavity2d_n2840: 0.977 ± 0.004 (med 0.977; 0.974–0.981; n=3) | weak:cavity2d_n2840: 0.991 ± 0.002 (med 0.992; 0.988–0.992; n=3) |
| 2d_n2840_k4_K4 | F3 | 4 | C | 90.32 ± 0.15 (med 90.25; 90.21–90.50; n=3) | weak:cavity2d_n2840: 0.983 ± 0.005 (med 0.985; 0.978–0.987; n=3) | weak:cavity2d_n2840: 0.997 ± 0.004 (med 0.996; 0.993–1.002; n=3) |
| 3d_n160_k8_K4 | F5 | 4 | E | 15.58 ± 0.04 (med 15.58; 15.54–15.62; n=3) | strong:cavity3d_n160_k8: 0.896 ± 0.003 (med 0.895; 0.893–0.899; n=3) | strong:cavity3d_n160_k8: 0.909 ± 0.002 (med 0.909; 0.908–0.911; n=3) |
| 3d_n160_k8_K4 | F5 | 4 | C | 15.66 ± 0.02 (med 15.66; 15.64–15.68; n=3) | strong:cavity3d_n160_k8: 0.901 ± 0.001 (med 0.901; 0.899–0.901; n=3) | strong:cavity3d_n160_k8: 0.914 ± 0.001 (med 0.914; 0.913–0.914; n=3) |
| 2d_n1440_k2_K2 | F3 | 2 | E | 199.78 ± 44.16 (med 201.10; 132.71–255.40; n=5) **two modes: 1× ~132.71 / 4× ~207.40** | weak:cavity2d_n1440: 0.552 ± 0.122 (med 0.556; 0.366–0.706; n=5) **two modes: 1× ~0.366 / 4× ~0.572** | weak:cavity2d_n1440: 0.556 ± 0.123 (med 0.558; 0.369–0.710; n=5) **two modes: 1× ~0.369 / 4× ~0.576** |
| 2d_n1440_k2_K2 | F3 | 2 | C | 254.16 ± 54.44 (med 274.80; 161.20–295.80; n=5) **two modes: 1× ~161.20 / 4× ~280.70** | weak:cavity2d_n1440: 0.702 ± 0.151 (med 0.757; 0.444–0.817; n=5) **two modes: 1× ~0.444 / 4× ~0.775** | weak:cavity2d_n1440: 0.707 ± 0.151 (med 0.764; 0.448–0.820; n=5) **two modes: 1× ~0.448 / 4× ~0.780** |

### Points: calibration, arm difference, traces

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r p50/p95, t_tr p50/p95 µs, T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n8000_K4 | 1.0221, 1.0009, 0.9792, 0.9978 | [412, 812, 1204] | [403, 803, 1203]→[413, 813, 1203] changed; [413, 813, 1203]→[412, 812, 1204] changed | 1.54 ± 0.22 (med 1.46; 1.37–1.80; n=3) **two modes: 2× ~1.42 / 1× ~1.80** % | pass: r 1.007/1.447, t_tr 2039/27205, T_B 18466 (max 18752), fps -0.04 % | pass: r 0.899/1.388, t_tr 3383/25942, T_B 18483 (max 18658), fps -0.00 % |
| 2d_n2840_k4_K4 | 1.0206, 1.0027, 0.9790, 0.9978 | [583, 1152, 1708] | [571, 1139, 1707]→[585, 1154, 1707] changed; [585, 1154, 1707]→[583, 1152, 1708] changed | 0.63 ± 0.32 (med 0.50; 0.41–1.00; n=3) **two modes: 2× ~0.45 / 1× ~1.00** % | pass: r 1.200/2.040, t_tr 1117/19606, T_B 9489 (max 9596), fps -0.65 % | pass: r 1.220/1.967, t_tr 887/18497, T_B 9480 (max 9609), fps 0.29 % |
| 3d_n160_k8_K4 | 1.0211, 0.9969, 0.9860, 0.9961 | [84, 163, 242] | [82, 162, 242]→[84, 164, 242] changed; [84, 164, 242]→[84, 163, 242] changed | 0.52 ± 0.25 (med 0.63; 0.23–0.69; n=3) **two modes: 1× ~0.23 / 2× ~0.66** % | pass: r 0.994/1.205, t_tr 5704/68231, T_B 57185 (max 58970), fps -0.14 % | pass: r 0.992/1.191, t_tr 12638/66964, T_B 57384 (max 58701), fps 0.43 % |
| 2d_n1440_k2_K2 | 1.0087, 0.9913 | [293] | [291]→[293] changed; [293]→[293] kept | 34.12 ± 48.58 (med 28.78; -24.57–107.06; n=5) **two modes: 4× ~20.50 / 1× ~107.06** % | pass: r 0.967/5.058, t_tr 393/12051, T_B 2401 (max 2407), fps 31.99 % | pass: r 0.962/5.000, t_tr 525/10209, T_B 2401 (max 2409), fps 4.62 % |

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| pre_2d_n8000_k1_x4 | 4 | ok | 0.3 | 16 | 16 | 26 |
| pre_3d_n160_k8_k1_x4 | 4 | ok | 0.1 | 19 | 19 | 31 |
| pre_3d_n200_k8_k1_x4 | 4 | ok | 0.0 | 18 | 18 | 94 |
| 2d_n8000_K4_t1_R1_2d_n8000 | 4 | ok | 0.6 | 16 | 16 | 2466 |
| 2d_n8000_K4_t2_R1_2d_n8000 | 4 | ok | 0.1 | 13 | 13 | 2680 |
| 2d_n8000_K4_t3_R1_2d_n8000 | 4 | ok | 11.2 | 19 | 18 | 2334 |
| 2d_n2840_k4_K4_t1_R1_2d_n2840 | 4 | ok | 0.2 | 17 | 17 | 304 |
| 2d_n2840_k4_K4_t2_R1_2d_n2840 | 4 | ok | 0.2 | 19 | 19 | 287 |
| 2d_n2840_k4_K4_t3_R1_2d_n2840 | 4 | ok | 0.2 | 20 | 20 | 350 |
| 3d_n160_k8_K4_t1_R1_3d_n160_k8 | 4 | ok | 0.2 | 17 | 17 | 3274 |
| 3d_n160_k8_K4_t2_R1_3d_n160_k8 | 4 | ok | 0.2 | 15 | 15 | 3060 |
| 3d_n160_k8_K4_t3_R1_3d_n160_k8 | 4 | ok | 0.1 | 18 | 18 | 3296 |
| 2d_n1440_k2_K2_t1_R1_2d_n1440 | 2 | ok | 0.0 | 19 | 19 | 4 |
| 2d_n1440_k2_K2_t2_R1_2d_n1440 | 2 | ok | 0.0 | 9 | 8 | 20 |
| 2d_n1440_k2_K2_t3_R1_2d_n1440 | 2 | ok | 0.0 | 18 | 18 | 27 |
| 2d_n1440_k2_K2_t4_R1_2d_n1440 | 2 | ok | 0.0 | 18 | 18 | 56 |
| 2d_n1440_k2_K2_t5_R1_2d_n1440 | 2 | ok | 0.0 | 11 | 11 | 22 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n8000_K4 | 2750 / 575 | 2687 / 575 | 63 MHz / -0 W |
| 2d_n2840_k4_K4 | 2774 / 575 | 2735 / 575 | 39 MHz / -0 W |
| 3d_n160_k8_K4 | 2811 / 575 | 2738 / 575 | 72 MHz / -0 W |
| 2d_n1440_k2_K2 | 2897 / 403 | 2791 / 575 | 106 MHz / -172 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n8000_K4 | K runs | 4 | 6 | 1.96 | 14.70 | 3.83 | 20.49 | 5.96 | 33.42 |
| 2d_n8000_K4 | references | 1 | 8 | 2.02 | 8.55 | 3.28 | 13.85 | 3.02 | 36.82 |
| 2d_n2840_k4_K4 | K runs | 4 | 6 | 1.03 | 7.25 | 3.58 | 11.87 | 5.04 | 18.70 |
| 2d_n2840_k4_K4 | references | 1 | 8 | 0.28 | 1.07 | 4.35 | 5.70 | 1.28 | 5.82 |
| 3d_n160_k8_K4 | K runs | 4 | 6 | 0.98 | 8.17 | 6.44 | 15.59 | 4.98 | 20.46 |
| 3d_n160_k8_K4 | references | 1 | 8 | 1.02 | 4.80 | 6.72 | 12.54 | 2.02 | 22.57 |
| 2d_n1440_k2_K2 | K runs | 2 | 10 | 0.16 | 0.57 | 2.51 | 3.24 | 2.16 | 2.23 |
| 2d_n1440_k2_K2 | references | 1 | 8 | 0.09 | 0.22 | 3.25 | 3.56 | 1.09 | 1.24 |

## 02_e7_b1B_1681466

- host wqd10nbj02g1, slurm job 1681466, harness 3d6182b6ca5b, experiment/v7 tree 82dd6a740fa2 (= v7-rc1), driver 580.82.07
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1681466; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 16 files, 144 MiB in /dev/shm/scxm138/glcache_1681466; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 14; equal pass 14; precheck pass 44; reference pass 124; selftest pass 1; soak pass 1; trace_calibrated pass 4; trace_equal pass 4

### Pre-checks

| case | kind | processes | verdict | statuses | VRAM peak per card (MiB, min–max) | host: largest process / sum (GiB) | node in use peak (GiB) | build max (s) |
|---|---|---|---|---|---|---|---|---|
| cavity2d_n8000 | k1 | 8 | fits | pass | 21514–21636 | 9.3 / 74.1 | 73.3 | 102.5 |
| cavity2d_n9000 | k1 | 8 | fits | pass | 27202–27324 | 11.7 / 93.3 | 90.2 | 115.5 |
| cavity3d_n160_k8 | k1 | 8 | fits | pass | 11931–12058 | 5.4 / 42.9 | 54.2 | 55.6 |
| cavity3d_n200_k8 | k1 | 8 | fits | pass | 22780–22907 | 10.0 / 80.0 | 79.6 | 111.2 |
| cavity3d_n416 | k1 | 8 | fits | pass | 25020–25147 | 10.9 / 87.5 | 85.2 | 126.9 |
| cavity2d_n11320 | pairs | 4 | fits | pass | 22433–22492 | 12.8 / 51.2 | 68.6 | 151.4 |

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n8000_K8 | F1+F7 | 8 | E | 87.63 ± 0.09 (med 87.68; 87.53–87.68; n=3) | strong:cavity2d_n8000: 0.910 ± 0.002 (med 0.911; 0.908–0.912; n=3) | strong:cavity2d_n8000: 0.918 ± 0.001 (med 0.918; 0.917–0.919; n=3) |
| 2d_n8000_K8 | F1+F7 | 8 | C | 88.80 ± 0.23 (med 88.93; 88.53–88.93; n=3) | strong:cavity2d_n8000: 0.922 ± 0.003 (med 0.923; 0.920–0.925; n=3) | strong:cavity2d_n8000: 0.930 ± 0.003 (med 0.932; 0.927–0.933; n=3) |
| 2d_n4000_K8 | F1+F7 | 8 | E | 154.55 ± 23.17 (med 145.90; 140.45–195.80; n=5) **two modes: 4× ~145.30 / 1× ~195.80** | strong:cavity2d_n4000: 0.412 ± 0.062 (med 0.389; 0.374–0.522; n=5) **two modes: 4× ~0.388 / 1× ~0.522** | strong:cavity2d_n4000: 0.415 ± 0.062 (med 0.392; 0.377–0.526; n=5) **two modes: 4× ~0.391 / 1× ~0.526** |
| 2d_n4000_K8 | F1+F7 | 8 | C | 147.75 ± 26.81 (med 157.60; 103.84–174.80; n=5) **two modes: 1× ~103.84 / 4× ~157.85** | strong:cavity2d_n4000: 0.394 ± 0.071 (med 0.420; 0.277–0.465; n=5) **two modes: 1× ~0.277 / 4× ~0.421** | strong:cavity2d_n4000: 0.397 ± 0.072 (med 0.423; 0.279–0.469; n=5) **two modes: 1× ~0.279 / 4× ~0.425** |
| 2d_n11320_K8 | F1 | 8 | E | 47.00 ± 0.02 (med 47.01; 46.98–47.01; n=3) | pairs:cavity2d_n11320: 0.951 ± 0.001 (med 0.951; 0.950–0.951; n=3) | pairs:cavity2d_n11320: 0.955 ± 0.001 (med 0.955; 0.954–0.955; n=3) |
| 2d_n11320_K8 | F1 | 8 | C | 47.27 ± 0.05 (med 47.29; 47.21–47.29; n=3) | pairs:cavity2d_n11320: 0.956 ± 0.001 (med 0.956; 0.956–0.957; n=3) | pairs:cavity2d_n11320: 0.960 ± 0.001 (med 0.960; 0.960–0.961; n=3) |
| 3d_n160_k8_K8 | F4+F5 | 8 | E | 31.04 ± 0.04 (med 31.04; 31.00–31.08; n=3) | weak:cavity3d_n160: 0.969 ± 0.003 (med 0.970; 0.965–0.971; n=3)<br>strong:cavity3d_n160_k8: 0.890 ± 0.000 (med 0.889; 0.889–0.890; n=3) | weak:cavity3d_n160: 0.978 ± 0.001 (med 0.978; 0.977–0.979; n=3)<br>strong:cavity3d_n160_k8: 0.898 ± 0.002 (med 0.899; 0.897–0.899; n=3) |
| 3d_n160_k8_K8 | F4+F5 | 8 | C | 30.95 ± 0.05 (med 30.94; 30.90–31.00; n=3) | weak:cavity3d_n160: 0.966 ± 0.002 (med 0.966; 0.964–0.969; n=3)<br>strong:cavity3d_n160_k8: 0.887 ± 0.002 (med 0.888; 0.885–0.888; n=3) | weak:cavity3d_n160: 0.975 ± 0.002 (med 0.976; 0.973–0.977; n=3)<br>strong:cavity3d_n160_k8: 0.896 ± 0.002 (med 0.896; 0.894–0.898; n=3) |

### Points: calibration, arm difference, traces

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r p50/p95, t_tr p50/p95 µs, T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n8000_K8 | 1.0049, 1.0184, 0.9795, 1.0000, 0.9959, 0.9903, 1.0004, 1.0106 | [204, 408, 604, 804, 1003, 1201, 1401] | [203, 403, 603, 803, 1003, 1203, 1403]→[204, 407, 603, 803, 1002, 1200, 1400] changed; [204, 407, 603, 803, 1002, 1200, 1400]→[204, 408, 604, 804, 1003, 1201, 1401] changed | 1.33 ± 0.32 (med 1.42; 0.97–1.60; n=3) **two modes: 1× ~0.97 / 2× ~1.51** % | pass: r 1.294/1.994, t_tr 1684/18219, T_B 9268 (max 9385), fps 0.28 % | pass: r 1.175/2.032, t_tr 1636/18207, T_B 9192 (max 9399), fps 0.19 % |
| 2d_n4000_K8 | 1.0139, 0.9887, 0.9918, 1.0010, 1.0009, 0.9930, 0.9876, 1.0231 | [104, 203, 302, 403, 503, 602, 701] | [103, 203, 303, 403, 503, 603, 703]→[105, 204, 302, 403, 503, 602, 701] changed; [105, 204, 302, 403, 503, 602, 701]→[104, 203, 302, 403, 503, 602, 701] changed | -3.14 ± 21.39 (med -0.21; -28.83–24.46; n=5) **two modes: 2× ~-24.17 / 3× ~8.36** % | pass: r 4.831/8.351, t_tr 1372/18453, T_B 2296 (max 2374), fps -6.31 % | pass: r 8.196/8.839, t_tr 1350/20026, T_B 2264 (max 2405), fps -23.00 % |
| 2d_n11320_K8 | 1.0011, 1.0160, 0.9887, 0.9975, 0.9905, 0.9970, 1.0003, 1.0089 | [286, 574, 854, 1136, 1416, 1698, 1981] | [286, 569, 852, 1135, 1418, 1701, 1984]→[287, 576, 854, 1136, 1416, 1698, 1982] changed; [287, 576, 854, 1136, 1416, 1698, 1982]→[286, 574, 854, 1136, 1416, 1698, 1981] changed | 0.56 ± 0.06 (med 0.59; 0.50–0.59; n=3) **two modes: 1× ~0.50 / 2× ~0.59** % | pass: r 1.218/1.523, t_tr 2644/27419, T_B 17878 (max 18048), fps -0.30 % | pass: r 0.958/1.437, t_tr 1400/26320, T_B 17819 (max 18236), fps -0.28 % |
| 3d_n160_k8_K8 | 0.9820, 1.0104, 0.9953, 1.0168, 1.0022, 1.0094, 1.0003, 0.9836 | [41, 82, 122, 162, 202, 243, 283] | [42, 82, 122, 162, 202, 242, 282]→[41, 82, 122, 162, 202, 242, 282] changed; [41, 82, 122, 162, 202, 242, 282]→[41, 82, 122, 162, 202, 243, 283] changed | -0.29 ± 0.23 (med -0.19; -0.56–-0.12; n=3) **two modes: 1× ~-0.56 / 2× ~-0.15** % | pass: r 1.267/1.321, t_tr 2321/35712, T_B 25940 (max 28392), fps -0.12 % | pass: r 1.305/1.322, t_tr 1890/35624, T_B 26343 (max 27650), fps -0.18 % |

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| pre_2d_n8000_k1_x8 | 8 | ok | 43.3 | 19 | 19 | 284 |
| pre_2d_n9000_k1_x8 | 8 | ok | 44.2 | 17 | 17 | 306 |
| pre_3d_n160_k8_k1_x8 | 8 | ok | 18.6 | 18 | 18 | 260 |
| pre_3d_n200_k8_k1_x8 | 8 | ok | 42.5 | 19 | 19 | 338 |
| pre_3d_n416_k1_x8 | 8 | ok | 52.9 | 18 | 18 | 313 |
| pre_2d_n11320_pairs_x4 | 4 | ok | 48.7 | 14 | 14 | 264 |
| 2d_n8000_K8_t1_R1_2d_n8000 | 8 | ok | 23.6 | 19 | 19 | 1877 |
| 2d_n8000_K8_t2_R1_2d_n8000 | 8 | ok | 51.3 | 16 | 16 | 1866 |
| 2d_n8000_K8_t3_R1_2d_n8000 | 8 | ok | 43.6 | 16 | 16 | 3240 |
| 2d_n4000_K8_t1_R1_2d_n4000 | 8 | ok | 0.2 | 19 | 19 | 562 |
| 2d_n4000_K8_t2_R1_2d_n4000 | 8 | ok | 8.8 | 20 | 20 | 534 |
| 2d_n4000_K8_t3_R1_2d_n4000 | 8 | ok | 0.2 | 20 | 20 | 538 |
| 2d_n4000_K8_t4_R1_2d_n4000 | 8 | ok | 5.3 | 12 | 13 | 777 |
| 2d_n4000_K8_t5_R1_2d_n4000 | 8 | ok | 0.7 | 18 | 18 | 633 |
| 2d_n11320_K8_t1_RP_2d_n11320 | 4 | ok | 45.0 | 18 | 18 | 738 |
| 2d_n11320_K8_t2_RP_2d_n11320 | 4 | ok | 43.5 | 15 | 15 | 908 |
| 2d_n11320_K8_t3_RP_2d_n11320 | 4 | ok | 45.4 | 15 | 15 | 861 |
| 3d_n160_k8_K8_t1_R1_3d_n160 | 8 | ok | 0.3 | 14 | 14 | 324 |
| 3d_n160_k8_K8_t1_R1_3d_n160_k8 | 8 | ok | 14.2 | 18 | 18 | 2182 |
| 3d_n160_k8_K8_t2_R1_3d_n160 | 8 | ok | 2.4 | 17 | 17 | 305 |
| 3d_n160_k8_K8_t2_R1_3d_n160_k8 | 8 | ok | 5.1 | 16 | 16 | 1978 |
| 3d_n160_k8_K8_t3_R1_3d_n160 | 8 | ok | 0.3 | 16 | 16 | 299 |
| 3d_n160_k8_K8_t3_R1_3d_n160_k8 | 8 | ok | 11.2 | 16 | 16 | 1895 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n8000_K8 | 2803 / 574 | 2707 / 575 | 96 MHz / -1 W |
| 2d_n4000_K8 | 2805 / 282 | 2734 / 575 | 71 MHz / -293 W |
| 2d_n11320_K8 | 2767 / 575 | 2709 / 575 | 58 MHz / 0 W |
| 3d_n160_k8_K8 | 2829 / 570 | 2781 / 575 | 48 MHz / -5 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n8000_K8 | K runs | 8 | 6 | 2.02 | 21.89 | 4.73 | 28.65 | 10.02 | 37.32 |
| 2d_n8000_K8 | references | 1 | 16 | 4.09 | 15.45 | 3.83 | 23.37 | 5.09 | 39.30 |
| 2d_n4000_K8 | K runs | 8 | 10 | 0.53 | 4.88 | 3.51 | 8.91 | 8.53 | 10.71 |
| 2d_n4000_K8 | references | 1 | 32 | 0.56 | 2.14 | 3.09 | 5.80 | 1.56 | 11.86 |
| 2d_n11320_K8 | K runs | 8 | 6 | 7.52 | 60.26 | 4.39 | 72.17 | 15.52 | 74.92 |
| 2d_n11320_K8 | references | 2 | 8 | 10.63 | 33.27 | 3.60 | 47.50 | 12.63 | 75.84 |
| 3d_n160_k8_K8 | K runs | 8 | 6 | 0.99 | 12.03 | 6.79 | 19.81 | 8.99 | 22.58 |
| 3d_n160_k8_K8 | references | 1 | 32 | 0.77 | 3.81 | 5.98 | 10.56 | 1.77 | 14.12 |

### selftest selftest_2d_n1000_K8

- status pass ; steps 1000, fps 222.9; drift 0, overflow 0, far migration 0, stamps (0, 0), seam checks None; anatomy frames 8, [loop] intervals 1, defrag times 24
- pool inner (capacity 6691): peak per 1000 frames, max over links: 5111 (max 0.7639 of capacity)
- pool outer (capacity 6691): peak per 1000 frames, max over links: 5111 (max 0.7639 of capacity)
- pool migrant (capacity 165): peak per 1000 frames, max over links: 1 (max 0.0061 of capacity)
- own pool watermark per defrag (max over slabs: used fraction / interval migration): f1000: 0.8332/11
- host loop f1000 (1000 frames, V7_LOOP_TRACE): period p50 2.016 ms (max 21.728); in _submit_frame p50 1.023 / mean 1.167 / max 21.48 ms; blocked in frame waits p50 0.934 / mean 3.388 ms; cpu share 0.257
- defrag sim0 (every call, bootstrap included): wall_ms mean 1.2 (max 1.7, n=3); gpu_us mean 155.3 (max 192.5, n=3)
- defrag sim1 (every call, bootstrap included): wall_ms mean 1.2 (max 1.6, n=3); gpu_us mean 158.6 (max 197.1, n=3)
- defrag sim2 (every call, bootstrap included): wall_ms mean 1.2 (max 1.5, n=3); gpu_us mean 160.7 (max 196.1, n=3)
- defrag sim3 (every call, bootstrap included): wall_ms mean 1.8 (max 3.4, n=3); gpu_us mean 153.3 (max 199.9, n=3)
- defrag sim4 (every call, bootstrap included): wall_ms mean 1.2 (max 1.6, n=3); gpu_us mean 160.0 (max 199.2, n=3)
- defrag sim5 (every call, bootstrap included): wall_ms mean 2.0 (max 3.2, n=3); gpu_us mean 141.5 (max 155.1, n=3)
- defrag sim6 (every call, bootstrap included): wall_ms mean 1.1 (max 1.4, n=3); gpu_us mean 149.2 (max 159.5, n=3)
- defrag sim7 (every call, bootstrap included): wall_ms mean 1.0 (max 1.0, n=3); gpu_us mean 134.0 (max 139.3, n=3)

GPU µs per frame, mean over the sampled frames (n = 1 per simulator):

| item | sim0 | sim1 | sim2 | sim3 | sim4 | sim5 | sim6 | sim7 |
|---|---|---|---|---|---|---|---|---|
| predict | 7.9 | 11.5 | 11.3 | 16.1 | 11.5 | 11.3 | 15.9 | 15.9 |
| update_voxel | 10.2 | 12.3 | 12.3 | 16.1 | 14.3 | 12.3 | 16.4 | 16.4 |
| ghost_send_trailing | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 5.9 | - |
| phase_a | 24.3 | 36.1 | 35.8 | 44.5 | 38.1 | 35.8 | 44.3 | 38.4 |
| a_to_b_gap | 2.3 | 2.3 | 2.3 | 2.6 | 2.3 | 2.3 | 2.6 | 2.3 |
| correction_density_interior | 126.0 | 119.8 | 117.5 | 115.5 | 113.4 | 113.7 | 111.4 | 125.7 |
| force_deep_interior | 114.7 | 108.5 | 100.6 | 96.3 | 104.4 | 104.4 | 106.5 | 118.8 |
| phase_b | 240.6 | 228.4 | 218.1 | 211.7 | 217.9 | 218.1 | 217.9 | 244.5 |
| b_to_c_gap | 22007.0 | 19747.8 | 17799.2 | 15110.4 | 12381.4 | 9711.9 | 7144.4 | 3856.9 |
| expand_lists | 9.7 | 11.0 | 9.7 | 9.7 | 10.8 | 9.7 | 10.8 | 9.5 |
| install_trailing | 6.1 | 6.1 | 6.1 | 6.1 | 5.6 | 5.4 | 4.1 | - |
| append_departed | 6.1 | 5.6 | 6.1 | 6.1 | 3.8 | 4.1 | 4.1 | 4.1 |
| correction_density_boundary | 87.0 | 83.5 | 87.3 | 87.0 | 79.9 | 79.6 | 77.6 | 77.6 |
| density_copy | 2.3 | 4.4 | 2.0 | 2.3 | 4.1 | 4.4 | 4.4 | 2.3 |
| force | 69.4 | 73.5 | 73.7 | 71.4 | 71.7 | 71.4 | 71.4 | 69.6 |
| phase_c | 180.7 | 190.2 | 191.2 | 188.9 | 182.0 | 180.7 | 177.9 | 168.4 |
| readback_trailing_dma | 78.8 | 78.1 | 79.9 | 78.1 | 79.9 | 78.1 | 78.1 | - |
| readback_trailing_barrier | 2.6 | 2.6 | 3.3 | 2.6 | 2.6 | 2.6 | 2.6 | - |
| upload_trailing_dma | 86.5 | 86.5 | 86.5 | 87.3 | 85.8 | 86.8 | 86.0 | - |
| readback_trailing_sched_gap | 20565.8 | 18712.6 | 16252.7 | 13759.5 | 11234.8 | 8448.3 | 5675.0 | - |
| upload_trailing_to_c_gap | 9.7 | 13.1 | 9.7 | 13.8 | 10.2 | 9.7 | 10.0 | - |
| install_sum | 22.0 | 28.9 | 28.2 | 28.2 | 26.4 | 25.3 | 24.6 | 18.9 |
| ghost_send_leading | - | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 |
| install_leading | - | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 5.6 | 5.4 |
| readback_leading_dma | - | 75.5 | 78.8 | 81.2 | 83.2 | 82.4 | 80.1 | 80.4 |
| readback_leading_barrier | - | 2.3 | 2.0 | 3.1 | 1.8 | 2.6 | 2.0 | 1.8 |
| upload_leading_dma | - | 85.8 | 87.0 | 87.8 | 86.5 | 86.8 | 86.3 | 87.3 |
| readback_leading_sched_gap | - | 18635.5 | 16170.8 | 13675.0 | 11149.8 | 8363.8 | 5593.3 | 2807.3 |
| upload_leading_to_c_gap | - | 122.1 | 97.3 | 423.7 | 493.3 | 337.9 | 97.0 | 13.3 |

### soak soak_2d_n8000_K8

- status pass ; steps 17000, fps 86.9; drift 0, overflow 0, far migration 0, stamps (0, 0), seam checks True; anatomy frames 0, [loop] intervals 0, defrag times 152
- pool inner (capacity 52163): peak per 1000 frames, max over links: 40111, 40112, 40112, 40113, 40113, 40113, 40113, 40115, 40114, 40115, 40115, 40115, 40115, 40124, 40122, 40131, 40124 (max 0.7693 of capacity)
- pool outer (capacity 52163): peak per 1000 frames, max over links: 40111, 40112, 40112, 40113, 40113, 40113, 40113, 40114, 40113, 40115, 40114, 40114, 40124, 40126, 40134, 40141, 40149 (max 0.7697 of capacity)
- pool migrant (capacity 1285): peak per 1000 frames, max over links: 2, 2, 3, 2, 4, 3, 4, 3, 3, 3, 3, 3, 3, 5, 4, 4, 4 (max 0.0039 of capacity)
- own pool watermark per defrag (max over slabs: used fraction / interval migration): f1000: 0.8333/37, f2000: 0.8333/75, f3000: 0.8333/98, f4000: 0.8334/114, f5000: 0.8334/133, f6000: 0.8334/147, f7000: 0.8334/164, f8000: 0.8334/170, f9000: 0.8334/185, f10000: 0.8335/192, f11000: 0.8335/204, f12000: 0.8335/212, f13000: 0.8335/238, f14000: 0.8335/293, f15000: 0.8336/284, f16000: 0.8336/330, f17000: 0.8336/347
- defrag sim0 (every call, bootstrap included): wall_ms mean 32.3 (max 170.5, n=19)
- defrag sim1 (every call, bootstrap included): wall_ms mean 30.1 (max 132.9, n=19)
- defrag sim2 (every call, bootstrap included): wall_ms mean 30.5 (max 140.8, n=19)
- defrag sim3 (every call, bootstrap included): wall_ms mean 33.2 (max 195.4, n=19)
- defrag sim4 (every call, bootstrap included): wall_ms mean 35.3 (max 234.3, n=19)
- defrag sim5 (every call, bootstrap included): wall_ms mean 38.9 (max 303.4, n=19)
- defrag sim6 (every call, bootstrap included): wall_ms mean 37.9 (max 284.6, n=19)
- defrag sim7 (every call, bootstrap included): wall_ms mean 35.1 (max 224.9, n=19)

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| b1A | 1681465 | wqd10nah10g5 | 2026-10-10T10:09:59 | 2026-10-10T11:55:13 | COMPLETED | 1.75 | 2.27 |
| b1B | 1681466 | wqd10nbj02g1 | 2026-10-10T10:10:16 | 2026-10-10T13:04:19 | COMPLETED | 2.90 | 3.70 |
