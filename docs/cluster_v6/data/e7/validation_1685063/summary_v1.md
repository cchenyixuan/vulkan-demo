## e7_v1_1685063

- host wqd10nbj04g1, slurm job 1685063, harness cd03f1129393, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- rc1 deployment (solver rc1): root /data/run01/scxm138/vulkan-demo-v7rc1, experiment/v7 tree 82dd6a740fa2 (v7-rc1) = expected, commit d0c8dcb7c848, harness 524ace2b34a8, manifest OK, scripts OK
- provenance at job end: unchanged
- runs tagged rc1: 6, experiment/v7 tree 82dd6a740fa2 (v7-rc1) = rc1_expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc1
- runs tagged rc2: 128, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1685063; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 16 files, 81 MiB in /dev/shm/scxm138/glcache_1685063; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 22; equal pass 10; reference pass 98; trace_calibrated pass 2; trace_equal pass 2

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n1440_k2_K2 | F3 | 2 | E | 327.86 ± 3.29 (med 328.10; 323.40–331.40; n=5) | weak:cavity2d_n1440: 0.922 ± 0.008 (med 0.924; 0.912–0.932; n=5) | weak:cavity2d_n1440: 0.929 ± 0.007 (med 0.929; 0.919–0.936; n=5) |
| 2d_n1440_k2_K2 | F3 | 2 | C | 326.56 ± 1.44 (med 326.60; 325.30–328.80; n=5) | weak:cavity2d_n1440: 0.918 ± 0.007 (med 0.915; 0.911–0.926; n=5) | weak:cavity2d_n1440: 0.925 ± 0.010 (med 0.919; 0.917–0.936; n=5) |
| 2d_n4000_K8 | F1+F7 | 8 | E | 320.68 ± 0.35 (med 320.70; 320.20–321.10; n=5) | strong:cavity2d_n4000: 0.857 ± 0.001 (med 0.857; 0.856–0.859; n=5) | strong:cavity2d_n4000: 0.865 ± 0.001 (med 0.865; 0.863–0.866; n=5) |
| 2d_n4000_K8 | F1+F7 | 8 | C | 324.02 ± 0.80 (med 324.20; 322.80–324.80; n=5) | strong:cavity2d_n4000: 0.866 ± 0.002 (med 0.867; 0.862–0.869; n=5) | strong:cavity2d_n4000: 0.874 ± 0.002 (med 0.875; 0.871–0.876; n=5) |
| ab_2d_n8000_K8 | F1+F7 | 8 | C_rc1 | 88.44 ± 0.16 (med 88.50; 88.26–88.57; n=3) | strong:cavity2d_n8000: 0.925 ± 0.002 (med 0.926; 0.923–0.927; n=3) | strong:cavity2d_n8000: 0.932 ± 0.002 (med 0.932; 0.930–0.934; n=3) |
| ab_2d_n8000_K8 | F1+F7 | 8 | C_rc2 | 89.23 ± 0.02 (med 89.25; 89.21–89.25; n=3) | strong:cavity2d_n8000: 0.933 ± 0.000 (med 0.933; 0.933–0.934; n=3) | strong:cavity2d_n8000: 0.941 ± 0.000 (med 0.941; 0.940–0.941; n=3) |
| ab_3d_n160_k8_K8 | F4 | 8 | C_rc1 | 30.79 ± 0.07 (med 30.75; 30.74–30.87; n=3) | weak:cavity3d_n160: 0.968 ± 0.005 (med 0.965; 0.965–0.974; n=3) | weak:cavity3d_n160: 0.974 ± 0.004 (med 0.973; 0.971–0.979; n=3) |
| ab_3d_n160_k8_K8 | F4 | 8 | C_rc2 | 30.78 ± 0.05 (med 30.77; 30.73–30.83; n=3) | weak:cavity3d_n160: 0.968 ± 0.003 (med 0.967; 0.965–0.971; n=3) | weak:cavity3d_n160: 0.974 ± 0.002 (med 0.973; 0.972–0.975; n=3) |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n1440_k2_K2 | 0.9994, 1.0006 | [291] | [291]→[291] kept | -0.39 ± 1.11 (med -0.18; -1.84–0.99; n=5) % | pass: r_chain 0.066/1.297, t_chain 157/3214; r_own 0.060/1.274, t_own 144/3131; r 0.270/1.366, t_tr 422/3375; T_B 2493 (max 2504), fps 0.68 % | pass: r_chain 0.067/1.242, t_chain 159/3072; r_own 0.061/1.237, t_own 145/3062; r 0.240/1.306, t_tr 397/3209; T_B 2489 (max 2503), fps 1.27 % |
| 2d_n4000_K8 | 1.0149, 0.9963, 0.9936, 0.9887, 0.9811, 1.0046, 0.9963, 1.0244 | [104, 204, 303, 402, 500, 601, 701] | [103, 203, 303, 403, 503, 603, 703]→[105, 205, 304, 403, 500, 600, 700] changed; [105, 205, 304, 403, 500, 600, 700]→[104, 204, 303, 402, 500, 601, 701] changed | 1.04 ± 0.24 (med 1.09; 0.65–1.28; n=5) % | pass: r_chain 0.914/1.264, t_chain 825/2848; r_own 0.564/0.815, t_own 725/1617; r 1.227/1.722, t_tr 1062/3958; T_B 2312 (max 2400), fps -0.56 % | pass: r_chain 0.781/1.075, t_chain 825/2342; r_own 0.547/0.834, t_own 775/1576; r 1.093/1.713, t_tr 1258/3926; T_B 2303 (max 2435), fps -0.47 % |
| ab_2d_n8000_K8 | 1.0024, 1.0046, 0.9886, 1.0058, 0.9907, 0.9907, 1.0009, 1.0163 | [203, 404, 602, 803, 1001, 1200, 1400] | [203, 403, 603, 803, 1003, 1203, 1403]→[203, 405, 602, 803, 1001, 1199, 1399] changed; [203, 405, 602, 803, 1001, 1199, 1399]→[203, 404, 602, 803, 1001, 1200, 1400] changed | - | - | - |
| ab_3d_n160_k8_K8 | 0.9883, 1.0133, 1.0007, 0.9952, 0.9930, 1.0066, 1.0106, 0.9924 | [42, 82, 122, 162, 202, 242, 282] | [42, 82, 122, 162, 202, 242, 282]→[42, 82, 122, 162, 202, 242, 282] kept | - | - | - |

### rc1 vs rc2 pairs

Per trial, both runs against the same reference set: Δfps = (fps_rc2 − fps_rc1) / fps_rc1, Δeta = eta_rc2 − eta_rc1; over the trials mean ± std (med; min–max; n) and the standard error SE = std / √n, no mode test on the differences. First − second K slot: the mean of the K run right after the reference set minus the mean of the one after it (the position term). The rc1 runs ran the rc1 deployment's own wrapper (its run_chain_v6.py, harness 524ace2b34a8), the rc2 runs the job's (harness cd03f1129393): a wrapper-side difference is part of Δ.

| point | K | arm | quantity | rc1 | rc2 | rc2 − rc1 | per trial | first − second K slot |
|---|---|---|---|---|---|---|---|---|
| ab_2d_n8000_K8 | 8 | C | fps | 88.44 ± 0.16 (med 88.50; 88.26–88.57; n=3) | 89.23 ± 0.02 (med 89.25; 89.21–89.25; n=3) | 0.892 ± 0.205 (med 0.848; 0.714–1.116; n=3) %; SE 0.118 % | t1 +1.116, t2 +0.848, t3 +0.714 % | -0.29 fps (-0.33 %) |
| ab_2d_n8000_K8 | 8 | C | eta strong:cavity2d_n8000 | 0.9251 ± 0.0018 (med 0.9257; 0.9231–0.9266; n=3) | 0.9334 ± 0.0002 (med 0.9334; 0.9332–0.9335; n=3) | 0.0083 ± 0.0019 (med 0.0078; 0.0066–0.0103; n=3); SE 0.0011 | t1 +0.0103, t2 +0.0078, t3 +0.0066 | -0.0030 |
| ab_2d_n8000_K8 | 8 | C | eta_min strong:cavity2d_n8000 | 0.9322 ± 0.0017 (med 0.9323; 0.9304–0.9339; n=3) | 0.9405 ± 0.0003 (med 0.9406; 0.9402–0.9408; n=3) | 0.0083 ± 0.0019 (med 0.0079; 0.0067–0.0104; n=3); SE 0.0011 | t1 +0.0104, t2 +0.0079, t3 +0.0067 | -0.0030 |
| ab_3d_n160_k8_K8 | 8 | C | fps | 30.79 ± 0.07 (med 30.75; 30.74–30.87; n=3) | 30.78 ± 0.05 (med 30.77; 30.73–30.83; n=3) | -0.041 ± 0.293 (med -0.031; -0.338–0.247; n=3) %; SE 0.169 % | t1 +0.247, t2 -0.338, t3 -0.031 % | -0.06 fps (-0.19 %) |
| ab_3d_n160_k8_K8 | 8 | C | eta weak:cavity3d_n160 | 0.9682 ± 0.0054 (med 0.9655; 0.9647–0.9745; n=3) | 0.9678 ± 0.0031 (med 0.9670; 0.9652–0.9712; n=3) | -0.0004 ± 0.0028 (med -0.0003; -0.0033–0.0024; n=3); SE 0.0016 | t1 +0.0024, t2 -0.0033, t3 -0.0003 | -0.0018 |
| ab_3d_n160_k8_K8 | 8 | C | eta_min weak:cavity3d_n160 | 0.9741 ± 0.0040 (med 0.9726; 0.9711–0.9787; n=3) | 0.9737 ± 0.0015 (med 0.9735; 0.9723–0.9754; n=3) | -0.0004 ± 0.0029 (med -0.0003; -0.0033–0.0024; n=3); SE 0.0016 | t1 +0.0024, t2 -0.0033, t3 -0.0003 | -0.0018 |

Resolution over the trials (95 % = t(n − 1) × SE). With unequal leads the position term enters the mean Δ with weight (leads of rc2 − leads of rc1) / n; the lead-balanced Δ weighs both orders equally (Δ = mean when rc2 led + mean when rc1 led, halved; position = half their difference).

- ab_2d_n8000_K8 C fps: Δ +0.892 % ± 0.118 % SE (n 3; 95 % ± 0.508 %, t 4.30); first − second K slot -0.29 fps (-0.33 %); leads t1 rc1, t2 rc2, t3 rc1; lead-balanced Δ +0.881 %, position -0.033 %
- ab_2d_n8000_K8 C eta strong:cavity2d_n8000: Δ +0.0083 ± 0.0011 SE (n 3; 95 % ± 0.0047, t 4.30); eta SD over the trials rc1 0.0018, rc2 0.0002; first − second K slot -0.0030; leads t1 rc1, t2 rc2, t3 rc1; lead-balanced Δ +0.0082, position -0.0003
- ab_3d_n160_k8_K8 C fps: Δ -0.041 % ± 0.169 % SE (n 3; 95 % ± 0.727 %, t 4.30); first − second K slot -0.06 fps (-0.19 %); leads t1 rc1, t2 rc2, t3 rc1; lead-balanced Δ -0.115 %, position -0.223 %
- ab_3d_n160_k8_K8 C eta weak:cavity3d_n160: Δ -0.0004 ± 0.0016 SE (n 3; 95 % ± 0.0071, t 4.30); eta SD over the trials rc1 0.0054, rc2 0.0031; first − second K slot -0.0018; leads t1 rc1, t2 rc2, t3 rc1; lead-balanced Δ -0.0011, position -0.0022

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n1440_k2_K2 E: 6 of 6 runs (6 with V7_DEST_GUARD=relay): precheck 20256, relay 15744, fallback 0, blocking 0 (relay slept 15744, released 0)
- 2d_n1440_k2_K2 C: 6 of 6 runs (6 with V7_DEST_GUARD=relay): precheck 20084, relay 15916, fallback 0, blocking 0 (relay slept 15916, released 0)
- 2d_n4000_K8 E: 6 of 6 runs (6 with V7_DEST_GUARD=relay): precheck 155984, relay 96016, fallback 0, blocking 0 (relay slept 95880, released 0)
- 2d_n4000_K8 C: 6 of 6 runs (6 with V7_DEST_GUARD=relay): precheck 145087, relay 106913, fallback 0, blocking 0 (relay slept 106745, released 0)
- ab_2d_n8000_K8 C_rc2: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 64669, relay 61331, fallback 0, blocking 0 (relay slept 61315, released 0)
- ab_3d_n160_k8_K8 C_rc2: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 33242, relay 29758, fallback 0, blocking 0 (relay slept 29720, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n1440_k2_K2_t1_R1_2d_n1440 | 2 | ok | 0.0 | 13 | 13 | 24 |
| 2d_n1440_k2_K2_t2_R1_2d_n1440 | 2 | ok | 0.0 | 8 | 8 | 5 |
| 2d_n1440_k2_K2_t3_R1_2d_n1440 | 2 | ok | 0.0 | 7 | 7 | 31 |
| 2d_n1440_k2_K2_t4_R1_2d_n1440 | 2 | ok | 0.0 | 7 | 7 | 40 |
| 2d_n1440_k2_K2_t5_R1_2d_n1440 | 2 | ok | 0.0 | 8 | 8 | 22 |
| 2d_n4000_K8_t1_R1_2d_n4000 | 8 | ok | 84.5 | 19 | 19 | 564 |
| 2d_n4000_K8_t2_R1_2d_n4000 | 8 | ok | 82.9 | 17 | 17 | 598 |
| 2d_n4000_K8_t3_R1_2d_n4000 | 8 | ok | 93.2 | 19 | 19 | 444 |
| 2d_n4000_K8_t4_R1_2d_n4000 | 8 | ok | 97.7 | 19 | 19 | 326 |
| 2d_n4000_K8_t5_R1_2d_n4000 | 8 | ok | 102.0 | 18 | 19 | 553 |
| ab_2d_n8000_K8_t1_R1_2d_n8000 | 8 | ok | 258.3 | 19 | 19 | 1364 |
| ab_2d_n8000_K8_t2_R1_2d_n8000 | 8 | ok | 44.4 | 19 | 19 | 2760 |
| ab_2d_n8000_K8_t3_R1_2d_n8000 | 8 | ok | 45.7 | 15 | 15 | 1429 |
| ab_3d_n160_k8_K8_t1_R1_3d_n160 | 8 | ok | 6.0 | 16 | 16 | 657 |
| ab_3d_n160_k8_K8_t2_R1_3d_n160 | 8 | ok | 4.4 | 19 | 19 | 539 |
| ab_3d_n160_k8_K8_t3_R1_3d_n160 | 8 | ok | 4.0 | 16 | 16 | 321 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n1440_k2_K2 | 2799 / 558 | 2746 / 575 | 53 MHz / -17 W |
| 2d_n4000_K8 | 2799 / 530 | 2731 / 575 | 69 MHz / -45 W |
| ab_2d_n8000_K8 | 2769 / 570 | 2689 / 575 | 79 MHz / -5 W |
| ab_3d_n160_k8_K8 | 2813 / 561 | 2784 / 568 | 29 MHz / -7 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n1440_k2_K2 | K runs | 2 | 10 | 0.16 | 0.57 | 2.47 | 3.20 | 2.16 | 2.22 |
| 2d_n1440_k2_K2 | references | 1 | 8 | 0.09 | 0.22 | 3.12 | 3.43 | 1.09 | 1.23 |
| 2d_n4000_K8 | K runs | 8 | 10 | 3.23 | 4.88 | 4.07 | 12.18 | 11.23 | 10.53 |
| 2d_n4000_K8 | references | 1 | 32 | 8.45 | 21.92 | 1.99 | 32.37 | 9.45 | 26.12 |
| ab_2d_n8000_K8 | K runs | 8 | 6 | 9.21 | 24.04 | 4.35 | 37.60 | 17.21 | 36.94 |
| ab_2d_n8000_K8 | references | 1 | 16 | 6.20 | 26.85 | 3.57 | 36.62 | 7.20 | 43.31 |
| ab_3d_n160_k8_K8 | K runs | 8 | 6 | 2.56 | 12.45 | 6.06 | 21.07 | 10.55 | 22.79 |
| ab_3d_n160_k8_K8 | references | 1 | 16 | 0.23 | 0.78 | 3.27 | 4.28 | 1.23 | 2.90 |

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| v1 | 1685063 | wqd10nbj04g1 | 2026-10-12T14:42:58 | 2026-10-12T16:04:18 | COMPLETED | 1.36 | - |
