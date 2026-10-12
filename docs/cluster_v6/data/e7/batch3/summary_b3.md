## e7_b3A_1683993

- host wqd10nah10g2, slurm job 1683993, harness 524ace2b34a8, driver 580.82.07
- job deployment: experiment/v7 tree 82dd6a740fa2 (v7-rc1) = expected, commit d0c8dcb7c848
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1683993; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 6; equal pass 6; reference pass 18; trace_calibrated pass 2; trace_equal pass 2

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 3d_n160_k2_K2 | F4 | 2 | E | 31.78 ± 0.02 (med 31.79; 31.77–31.80; n=3) | weak:cavity3d_n160: 0.973 ± 0.003 (med 0.974; 0.970–0.976; n=3) | weak:cavity3d_n160: 0.976 ± 0.006 (med 0.977; 0.970–0.983; n=3) |
| 3d_n160_k2_K2 | F4 | 2 | C | 31.67 ± 0.06 (med 31.68; 31.61–31.72; n=3) | weak:cavity3d_n160: 0.970 ± 0.005 (med 0.971; 0.965–0.974; n=3) | weak:cavity3d_n160: 0.973 ± 0.007 (med 0.975; 0.965–0.979; n=3) |
| 3d_n160_k4_K4 | F4 | 4 | E | 31.19 ± 0.06 (med 31.16; 31.15–31.27; n=3) | weak:cavity3d_n160: 0.958 ± 0.000 (med 0.958; 0.958–0.958; n=3) | weak:cavity3d_n160: 0.965 ± 0.004 (med 0.966; 0.961–0.968; n=3) |
| 3d_n160_k4_K4 | F4 | 4 | C | 30.92 ± 0.07 (med 30.92; 30.85–30.99; n=3) | weak:cavity3d_n160: 0.950 ± 0.000 (med 0.950; 0.949–0.950; n=3) | weak:cavity3d_n160: 0.957 ± 0.003 (med 0.957; 0.954–0.959; n=3) |
| 3d_n200_k2_K2 | F4 | 2 | E | - | - | - |
| 3d_n200_k2_K2 | F4 | 2 | C | - | - | - |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 3d_n160_k2_K2 | 0.9985, 1.0015 | [42] | [42]→[44] changed; [44]→[42] changed | -0.37 ± 0.14 (med -0.38; -0.51–-0.22; n=3) % | pass: r_chain 0.047/0.243, t_chain 1106/5947; r_own 0.045/0.235, t_own 1089/5368; r 0.804/1.198, t_tr 9558/32702; T_B 28028 (max 28036), fps -0.27 % | pass: r_chain 0.050/0.272, t_chain 1087/7031; r_own 0.047/0.214, t_own 902/5263; r 0.980/1.323, t_tr 10632/36782; T_B 27991 (max 28026), fps -0.13 % |
| 3d_n160_k4_K4 | 1.0164, 1.0031, 0.9951, 0.9854 | [43, 83, 123] | [42, 82, 122]→[42, 83, 123] changed; [42, 83, 123]→[43, 83, 123] changed | -0.88 ± 0.09 (med -0.90; -0.96–-0.77; n=3) % | pass: r_chain 0.056/0.379, t_chain 1134/8665; r_own 0.053/0.271, t_own 1119/4849; r 0.931/1.268, t_tr 1182/36981; T_B 26793 (max 28421), fps 0.08 % | pass: r_chain 0.895/1.138, t_chain 1358/28668; r_own 0.060/0.282, t_own 1080/5741; r 1.363/1.434, t_tr 4882/36459; T_B 26499 (max 28730), fps 0.28 % |
| 3d_n200_k2_K2 | - | - | - | - | - | - |

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 3d_n160_k2_K2_t1_R1_3d_n160 | 2 | ok | 0.0 | 18 | 18 | 1588 |
| 3d_n160_k2_K2_t2_R1_3d_n160 | 2 | ok | 0.0 | 19 | 19 | 241 |
| 3d_n160_k2_K2_t3_R1_3d_n160 | 2 | ok | 0.1 | 0 | 0 | 160 |
| 3d_n160_k4_K4_t1_R1_3d_n160 | 4 | ok | 1.2 | 18 | 18 | 769 |
| 3d_n160_k4_K4_t2_R1_3d_n160 | 4 | ok | 0.1 | 18 | 18 | 472 |
| 3d_n160_k4_K4_t3_R1_3d_n160 | 4 | ok | 0.0 | 15 | 15 | 277 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 3d_n160_k2_K2 | 2868 / 596 | 2856 / 597 | 12 MHz / -2 W |
| 3d_n160_k4_K4 | 2848 / 572 | 2832 / 588 | 17 MHz / -16 W |
| 3d_n200_k2_K2 | - / - | - / - | - MHz / - W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 3d_n160_k2_K2 | K runs | 2 | 6 | 0.25 | 1.30 | 3.27 | 4.82 | 2.25 | 5.36 |
| 3d_n160_k2_K2 | references | 1 | 4 | 0.15 | 0.50 | 3.87 | 4.52 | 1.15 | 2.65 |
| 3d_n160_k4_K4 | K runs | 4 | 6 | 0.48 | 3.62 | 4.59 | 8.68 | 4.48 | 10.84 |
| 3d_n160_k4_K4 | references | 1 | 8 | 0.16 | 0.52 | 3.50 | 4.17 | 1.16 | 2.93 |

## e7_b3A2_1689810

- host wqd10nbj04g1, slurm job 1689810, harness 524ace2b34a8, driver 580.82.07
- job deployment: experiment/v7 tree 82dd6a740fa2 (v7-rc1) = expected, commit d0c8dcb7c848
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1689810; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 8 files, 42 MiB in /dev/shm/scxm138/glcache_1689810; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 15; equal pass 15; reference pass 42; trace_calibrated pass 5; trace_equal pass 5

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 3d_n200_k2_K2 | F4 | 2 | E | 16.44 ± 0.01 (med 16.44; 16.42–16.45; n=3) | weak:cavity3d_n200: 0.985 ± 0.001 (med 0.985; 0.984–0.985; n=3) | weak:cavity3d_n200: 0.989 ± 0.002 (med 0.989; 0.987–0.991; n=3) |
| 3d_n200_k2_K2 | F4 | 2 | C | 16.66 ± 0.06 (med 16.66; 16.60–16.71; n=3) | weak:cavity3d_n200: 0.997 ± 0.005 (med 0.998; 0.992–1.002; n=3) | weak:cavity3d_n200: 1.002 ± 0.006 (med 1.002; 0.996–1.008; n=3) |
| 3d_n200_k4_K4 | F4 | 4 | E | 16.44 ± 0.03 (med 16.44; 16.41–16.48; n=3) | weak:cavity3d_n200: 0.984 ± 0.001 (med 0.985; 0.983–0.985; n=3) | weak:cavity3d_n200: 0.988 ± 0.001 (med 0.989; 0.987–0.989; n=3) |
| 3d_n200_k4_K4 | F4 | 4 | C | 16.47 ± 0.01 (med 16.47; 16.47–16.48; n=3) | weak:cavity3d_n200: 0.986 ± 0.002 (med 0.987; 0.984–0.987; n=3) | weak:cavity3d_n200: 0.990 ± 0.003 (med 0.991; 0.986–0.993; n=3) |
| 3d_n160_k8_K2 | F5 | 2 | E | 8.10 ± 0.01 (med 8.10; 8.09–8.11; n=3) | strong:cavity3d_n160_k8: 0.936 ± 0.001 (med 0.936; 0.935–0.936; n=3) | strong:cavity3d_n160_k8: 0.939 ± 0.001 (med 0.938; 0.938–0.940; n=3) |
| 3d_n160_k8_K2 | F5 | 2 | C | 8.10 ± 0.00 (med 8.10; 8.10–8.10; n=3) | strong:cavity3d_n160_k8: 0.936 ± 0.001 (med 0.936; 0.935–0.936; n=3) | strong:cavity3d_n160_k8: 0.939 ± 0.001 (med 0.939; 0.938–0.939; n=3) |
| 3d_n200_k8_K2 | F5 | 2 | E | 4.50 ± 0.00 (med 4.50; 4.50–4.50; n=3) | strong:cavity3d_n200_k8: 0.949 ± 0.001 (med 0.949; 0.949–0.950; n=3) | strong:cavity3d_n200_k8: 0.954 ± 0.001 (med 0.954; 0.953–0.954; n=3) |
| 3d_n200_k8_K2 | F5 | 2 | C | 4.57 ± 0.01 (med 4.57; 4.56–4.57; n=3) | strong:cavity3d_n200_k8: 0.963 ± 0.002 (med 0.964; 0.961–0.965; n=3) | strong:cavity3d_n200_k8: 0.967 ± 0.002 (med 0.968; 0.965–0.969; n=3) |
| 3d_n200_k8_K4 | F5 | 4 | E | 8.49 ± 0.01 (med 8.49; 8.48–8.49; n=3) | strong:cavity3d_n200_k8: 0.898 ± 0.000 (med 0.898; 0.898–0.899; n=3) | strong:cavity3d_n200_k8: 0.901 ± 0.002 (med 0.902; 0.900–0.902; n=3) |
| 3d_n200_k8_K4 | F5 | 4 | C | 8.56 ± 0.03 (med 8.57; 8.52–8.58; n=3) | strong:cavity3d_n200_k8: 0.906 ± 0.002 (med 0.907; 0.903–0.907; n=3) | strong:cavity3d_n200_k8: 0.909 ± 0.004 (med 0.910; 0.905–0.912; n=3) |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 3d_n200_k2_K2 | 0.9804, 1.0196 | [51] | [52]→[51] changed; [51]→[51] kept | 1.32 ± 0.44 (med 1.28; 0.90–1.77; n=3) % | pass: r_chain 0.040/0.133, t_chain 1751/6456; r_own 0.038/0.122, t_own 1726/6101; r 0.987/1.165, t_tr 28253/66208; T_B 54372 (max 54756), fps -0.10 % | pass: r_chain 0.039/0.137, t_chain 1457/7197; r_own 0.037/0.122, t_own 1410/5521; r 0.971/1.164, t_tr 17179/66189; T_B 53649 (max 53664), fps 0.03 % |
| 3d_n200_k4_K4 | 0.9865, 1.0075, 0.9980, 1.0081 | [51, 102, 152] | [52, 102, 152]→[51, 101, 152] changed; [51, 101, 152]→[51, 102, 152] changed | 0.17 ± 0.23 (med 0.18; -0.07–0.40; n=3) % | pass: r_chain 1.013/1.067, t_chain 2430/55423; r_own 0.047/0.162, t_own 1658/6535; r 1.220/1.268, t_tr 7462/65878; T_B 52716 (max 55106), fps -0.40 % | pass: r_chain 0.528/1.022, t_chain 1677/51767; r_own 0.045/0.131, t_own 1617/5146; r 1.193/1.225, t_tr 20279/63453; T_B 52583 (max 54517), fps 0.08 % |
| 3d_n160_k8_K2 | 0.9992, 1.0008 | [162] | [162]→[162] kept | 0.01 ± 0.15 (med 0.06; -0.15–0.13; n=3) % | pass: r_chain 0.012/0.055, t_chain 1015/5538; r_own 0.012/0.054, t_own 945/5529; r 0.478/1.036, t_tr 18041/126197; T_B 116173 (max 116187), fps -0.13 % | pass: r_chain 0.012/0.053, t_chain 1155/5352; r_own 0.012/0.049, t_own 1140/5339; r 0.988/1.062, t_tr 58228/125262; T_B 115907 (max 116131), fps 0.16 % |
| 3d_n200_k8_K2 | 0.9834, 1.0166 | [199] | [202]→[198] changed; [198]→[199] changed | 1.44 ± 0.16 (med 1.53; 1.26–1.54; n=3) % | pass: r_chain 0.010/0.031, t_chain 1620/5491; r_own 0.010/0.031, t_own 1605/5477; r 1.008/1.045, t_tr 109256/226561; T_B 205906 (max 211084), fps 0.09 % | pass: r_chain 0.010/0.033, t_chain 1755/6624; r_own 0.010/0.033, t_own 1741/6611; r 0.997/1.033, t_tr 106736/223585; T_B 206228 (max 207885), fps 0.09 % |
| 3d_n200_k8_K4 | 0.9897, 0.9985, 1.0023, 1.0095 | [101, 201, 301] | [102, 202, 302]→[101, 201, 302] changed; [101, 201, 302]→[101, 201, 301] changed | 0.83 ± 0.23 (med 0.89; 0.57–1.02; n=3) % | pass: r_chain 1.010/1.042, t_chain 2428/112414; r_own 0.023/0.083, t_own 1892/7381; r 1.127/1.152, t_tr 7780/123413; T_B 105897 (max 109108), fps 0.23 % | pass: r_chain 0.937/1.031, t_chain 2281/111688; r_own 0.023/0.082, t_own 1711/7123; r 1.116/1.143, t_tr 7866/122710; T_B 105399 (max 108578), fps -0.02 % |

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 3d_n200_k2_K2_t1_R1_3d_n200 | 2 | ok | 0.0 | 11 | 10 | 140 |
| 3d_n200_k2_K2_t2_R1_3d_n200 | 2 | ok | 0.0 | 18 | 18 | 258 |
| 3d_n200_k2_K2_t3_R1_3d_n200 | 2 | ok | 0.1 | 4 | 4 | 436 |
| 3d_n200_k4_K4_t1_R1_3d_n200 | 4 | ok | 1.1 | 15 | 15 | 176 |
| 3d_n200_k4_K4_t2_R1_3d_n200 | 4 | ok | 0.1 | 18 | 18 | 280 |
| 3d_n200_k4_K4_t3_R1_3d_n200 | 4 | ok | 2.2 | 17 | 17 | 227 |
| 3d_n160_k8_K2_t1_R1_3d_n160_k8 | 2 | ok | 0.0 | 9 | 9 | 749 |
| 3d_n160_k8_K2_t2_R1_3d_n160_k8 | 2 | ok | 0.0 | 8 | 8 | 522 |
| 3d_n160_k8_K2_t3_R1_3d_n160_k8 | 2 | ok | 0.1 | 15 | 15 | 345 |
| 3d_n200_k8_K2_t1_R1_3d_n200_k8 | 2 | ok | 0.2 | 4 | 4 | 1636 |
| 3d_n200_k8_K2_t2_R1_3d_n200_k8 | 2 | ok | 0.2 | 8 | 7 | 1633 |
| 3d_n200_k8_K2_t3_R1_3d_n200_k8 | 2 | ok | 1.1 | 15 | 15 | 1655 |
| 3d_n200_k8_K4_t1_R1_3d_n200_k8 | 4 | ok | 0.0 | 12 | 12 | 1911 |
| 3d_n200_k8_K4_t2_R1_3d_n200_k8 | 4 | ok | 0.1 | 17 | 17 | 1176 |
| 3d_n200_k8_K4_t3_R1_3d_n200_k8 | 4 | ok | 0.2 | 17 | 18 | 1603 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 3d_n200_k2_K2 | 2763 / 575 | 2751 / 575 | 12 MHz / -0 W |
| 3d_n200_k4_K4 | 2779 / 573 | 2753 / 575 | 26 MHz / -2 W |
| 3d_n160_k8_K2 | 2759 / 575 | 2736 / 575 | 24 MHz / 0 W |
| 3d_n200_k8_K2 | 2738 / 575 | 2708 / 575 | 30 MHz / 0 W |
| 3d_n200_k8_K4 | 2747 / 575 | 2700 / 574 | 46 MHz / 0 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 3d_n200_k2_K2 | K runs | 2 | 6 | 10.72 | 2.83 | 4.03 | 17.58 | 12.71 | 10.04 |
| 3d_n200_k2_K2 | references | 1 | 4 | 0.28 | 1.10 | 3.91 | 5.29 | 1.28 | 5.45 |
| 3d_n200_k4_K4 | K runs | 4 | 6 | 13.79 | 7.76 | 6.05 | 27.60 | 17.79 | 19.73 |
| 3d_n200_k4_K4 | references | 1 | 8 | 0.29 | 1.13 | 4.85 | 6.27 | 1.28 | 6.05 |
| 3d_n160_k8_K2 | K runs | 2 | 6 | 7.78 | 6.05 | 5.74 | 19.57 | 9.78 | 18.94 |
| 3d_n160_k8_K2 | references | 1 | 4 | 1.01 | 4.75 | 6.95 | 12.71 | 2.01 | 20.06 |
| 3d_n200_k8_K2 | K runs | 2 | 6 | 8.30 | 11.81 | 9.08 | 29.19 | 10.30 | 35.71 |
| 3d_n200_k8_K2 | references | 1 | 4 | 1.88 | 9.10 | 9.99 | 20.97 | 2.88 | 36.97 |
| 3d_n200_k8_K4 | K runs | 4 | 6 | 19.36 | 15.75 | 9.26 | 44.37 | 23.36 | 36.26 |
| 3d_n200_k8_K4 | references | 1 | 8 | 1.97 | 9.26 | 12.90 | 24.13 | 2.97 | 46.52 |

## e7_b3B_1683994

- host wqd10nah09g5, slurm job 1683994, harness 524ace2b34a8, driver 580.82.07
- job deployment: experiment/v7 tree 82dd6a740fa2 (v7-rc1) = expected, commit d0c8dcb7c848
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1683994; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 16 files, 51 MiB in /dev/shm/scxm138/glcache_1683994; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 6; equal pass 6; reference pass 72; trace_calibrated pass 2; trace_equal pass 2

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 3d_n200_k8_K8 | F4+F5 | 8 | E | 16.48 ± 0.05 (med 16.50; 16.43–16.51; n=3) | weak:cavity3d_n200: 0.984 ± 0.002 (med 0.983; 0.983–0.986; n=3)<br>strong:cavity3d_n200_k8: 0.873 ± 0.002 (med 0.874; 0.870–0.874; n=3) | weak:cavity3d_n200: 1.001 ± 0.001 (med 1.000; 1.000–1.002; n=3)<br>strong:cavity3d_n200_k8: 0.892 ± 0.002 (med 0.893; 0.890–0.894; n=3) |
| 3d_n200_k8_K8 | F4+F5 | 8 | C | 16.50 ± 0.03 (med 16.50; 16.48–16.53; n=3) | weak:cavity3d_n200: 0.985 ± 0.004 (med 0.985; 0.981–0.989; n=3)<br>strong:cavity3d_n200_k8: 0.874 ± 0.001 (med 0.874; 0.873–0.876; n=3) | weak:cavity3d_n200: 1.002 ± 0.004 (med 1.001; 0.999–1.006; n=3)<br>strong:cavity3d_n200_k8: 0.893 ± 0.002 (med 0.893; 0.892–0.895; n=3) |
| 3d_n416_K8 | F6 | 8 | E | 15.08 ± 0.03 (med 15.08; 15.05–15.11; n=3) | strong:cavity3d_n416: 0.875 ± 0.002 (med 0.875; 0.873–0.876; n=3) | strong:cavity3d_n416: 0.896 ± 0.001 (med 0.897; 0.895–0.897; n=3) |
| 3d_n416_K8 | F6 | 8 | C | 15.09 ± 0.03 (med 15.09; 15.06–15.12; n=3) | strong:cavity3d_n416: 0.875 ± 0.002 (med 0.875; 0.873–0.877; n=3) | strong:cavity3d_n416: 0.897 ± 0.002 (med 0.896; 0.895–0.899; n=3) |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 3d_n200_k8_K8 | 0.9966, 0.9902, 0.9994, 1.0105, 1.0142, 0.9954, 0.9874, 1.0064 | [52, 101, 151, 202, 253, 302, 352] | [52, 102, 152, 202, 252, 302, 352]→[52, 101, 151, 201, 251, 301, 352] changed; [52, 101, 151, 201, 251, 301, 352]→[52, 101, 151, 202, 253, 302, 352] changed | 0.13 ± 0.44 (med -0.08; -0.16–0.63; n=3) % | pass: r_chain 0.997/1.030, t_chain 2304/54186; r_own 0.072/0.097, t_own 1867/4273; r 1.216/1.236, t_tr 3011/65216; T_B 50996 (max 54563), fps -0.14 % | pass: r_chain 0.814/1.044, t_chain 2191/51449; r_own 0.059/0.090, t_own 1913/4081; r 1.166/1.255, t_tr 3447/63220; T_B 51330 (max 54383), fps 0.14 % |
| 3d_n416_K8 | 0.9994, 0.9862, 0.9957, 1.0046, 0.9960, 1.0000, 0.9919, 1.0263 | [15, 28, 41, 54, 67, 80, 93] | [15, 28, 41, 54, 67, 80, 93]→[15, 28, 41, 54, 67, 80, 93] kept | 0.07 ± 0.18 (med 0.05; -0.11–0.26; n=3) % | pass: r_chain 0.973/1.142, t_chain 11299/43666; r_own 0.365/0.414, t_own 9048/15462; r 1.706/1.887, t_tr 13550/75233; T_B 37545 (max 50052), fps -0.02 % | pass: r_chain 0.923/1.108, t_chain 8809/41351; r_own 0.368/0.412, t_own 8605/15492; r 1.711/1.888, t_tr 13059/75223; T_B 37556 (max 50003), fps 0.48 % |

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 3d_n200_k8_K8_t1_R1_3d_n200 | 8 | ok | 4.9 | 17 | 17 | 823 |
| 3d_n200_k8_K8_t1_R1_3d_n200_k8 | 8 | ok | 0.2 | 17 | 17 | 5816 |
| 3d_n200_k8_K8_t2_R1_3d_n200 | 8 | ok | 6.8 | 9 | 9 | 1035 |
| 3d_n200_k8_K8_t2_R1_3d_n200_k8 | 8 | ok | 0.3 | 16 | 16 | 6119 |
| 3d_n200_k8_K8_t3_R1_3d_n200 | 8 | ok | 3.1 | 18 | 18 | 932 |
| 3d_n200_k8_K8_t3_R1_3d_n200_k8 | 8 | ok | 0.2 | 15 | 15 | 6050 |
| 3d_n416_K8_t1_R1_3d_n416 | 8 | ok | 0.2 | 15 | 15 | 6612 |
| 3d_n416_K8_t2_R1_3d_n416 | 8 | ok | 0.2 | 17 | 17 | 6874 |
| 3d_n416_K8_t3_R1_3d_n416 | 8 | ok | 0.2 | 17 | 18 | 7172 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 3d_n200_k8_K8 | 2786 / 574 | 2730 / 575 | 57 MHz / -1 W |
| 3d_n416_K8 | 2763 / 575 | 2692 / 575 | 71 MHz / -0 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 3d_n200_k8_K8 | K runs | 8 | 6 | 1.52 | 19.01 | 9.00 | 29.53 | 9.52 | 28.18 |
| 3d_n200_k8_K8 | references | 1 | 32 | 0.96 | 4.06 | 7.12 | 12.14 | 1.96 | 20.86 |
| 3d_n416_K8 | K runs | 8 | 6 | 1.67 | 20.96 | 9.41 | 32.03 | 9.67 | 32.23 |
| 3d_n416_K8 | references | 1 | 16 | 1.78 | 8.00 | 9.98 | 19.75 | 2.78 | 39.39 |

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| b3A | 1683993 | wqd10nah10g2 | 2026-10-11T16:25:47 | 2026-10-11T17:24:17 | FAILED | 0.97 | 5.23 |
| b3A2 | 1689810 | wqd10nbj04g1 | 2026-10-12T10:07:15 | 2026-10-12T14:42:58 | COMPLETED | 4.60 | - |
| b3B | 1683994 | wqd10nah09g5 | 2026-10-12T09:13:13 | 2026-10-12T11:26:21 | COMPLETED | 2.22 | 2.46 |
