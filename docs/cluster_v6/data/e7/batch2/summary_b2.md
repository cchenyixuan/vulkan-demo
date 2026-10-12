## e7_b2A_1694327

- host wqd10nbj04g2, slurm job 1694327, harness 99fee6d9f46e, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- provenance at job end: unchanged
- runs tagged rc2: 91, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1694327; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 8 files, 76 MiB in /dev/shm/scxm138/glcache_1694327; new files in ~/.cache/nvidia: 0
- runs: anatomy pass 1; calibrated pass 21; reference pass 66; trace pass 1; trace_calibrated pass 2

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n8000_K2 | F2 | 2 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K2 | F2 | 2 | C | 24.42 ± 0.02 (med 24.42; 24.40–24.44; n=3) | strong:cavity2d_n8000: 1.013 ± 0.001 (med 1.013; 1.011–1.014; n=3) | strong:cavity2d_n8000: 1.014 ± 0.001 (med 1.014; 1.013–1.015; n=3) |
| 2d_n8000_K4 | F2 | 4 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K4 | F2 | 4 | C | 47.50 ± 0.03 (med 47.48; 47.48–47.53; n=3) | strong:cavity2d_n8000: 0.988 ± 0.001 (med 0.989; 0.987–0.989; n=3) | strong:cavity2d_n8000: 0.997 ± 0.001 (med 0.998; 0.996–0.998; n=3) |
| 2d_n1440_k4_K4 | F3 | 4 | E | - | weak:cavity2d_n1440: - | weak:cavity2d_n1440: - |
| 2d_n1440_k4_K4 | F3 | 4 | C | 322.00 ± 13.95 (med 329.70; 305.90–330.40; n=3) **two modes: 1× ~305.90 / 2× ~330.05** | weak:cavity2d_n1440: 0.890 ± 0.040 (med 0.911; 0.844–0.916; n=3) **two modes: 1× ~0.844 / 2× ~0.913** | weak:cavity2d_n1440: 0.899 ± 0.040 (med 0.920; 0.852–0.924; n=3) **two modes: 1× ~0.852 / 2× ~0.922** |
| 2d_n2840_k2_K2 | F3 | 2 | E | - | weak:cavity2d_n2840: - | weak:cavity2d_n2840: - |
| 2d_n2840_k2_K2 | F3 | 2 | C | 92.76 ± 0.15 (med 92.68; 92.68–92.94; n=3) | weak:cavity2d_n2840: 1.001 ± 0.002 (med 1.001; 1.000–1.003; n=3) | weak:cavity2d_n2840: 1.002 ± 0.004 (med 1.001; 1.000–1.006; n=3) |
| 2d_n2840_k4_K4 | F3 | 4 | E | - | weak:cavity2d_n2840: - | weak:cavity2d_n2840: - |
| 2d_n2840_k4_K4 | F3 | 4 | C | 92.08 ± 0.22 (med 92.00; 91.91–92.34; n=3) | weak:cavity2d_n2840: 0.995 ± 0.002 (med 0.995; 0.993–0.996; n=3) | weak:cavity2d_n2840: 1.003 ± 0.005 (med 1.001; 1.000–1.008; n=3) |
| 2d_n5640_k2_K2 | F3 | 2 | E | - | weak:cavity2d_n5640: - | weak:cavity2d_n5640: - |
| 2d_n5640_k2_K2 | F3 | 2 | C | 24.78 ± 0.01 (med 24.78; 24.76–24.79; n=3) | weak:cavity2d_n5640: 1.015 ± 0.002 (med 1.016; 1.013–1.016; n=3) | weak:cavity2d_n5640: 1.015 ± 0.002 (med 1.016; 1.013–1.016; n=3) |
| 2d_n5640_k4_K4 | F3 | 4 | E | - | weak:cavity2d_n5640: - | weak:cavity2d_n5640: - |
| 2d_n5640_k4_K4 | F3 | 4 | C | 24.56 ± 0.03 (med 24.56; 24.53–24.58; n=3) | weak:cavity2d_n5640: 1.010 ± 0.002 (med 1.011; 1.008–1.011; n=3) | weak:cavity2d_n5640: 1.020 ± 0.002 (med 1.021; 1.017–1.021; n=3) |

Without the defrag events (user 2026-10-12): fps = steady_frames / (steady_s - the wall time of the events in the window), the same on the K run and on every K = 1 member; eta from these. The defrag event table lists the events themselves.

| point | K | arm | fps without defrag | eta without defrag (type: reference) | eta_min without defrag |
|---|---|---|---|---|---|
| 2d_n8000_K2 | 2 | E | - | - | - |
| 2d_n8000_K2 | 2 | C | 24.57 ± 0.02 (med 24.56; 24.55–24.60; n=3) | strong:cavity2d_n8000: 1.015 ± 0.001 (med 1.016; 1.014–1.017; n=3) | strong:cavity2d_n8000: 1.016 ± 0.001 (med 1.017; 1.015–1.017; n=3) |
| 2d_n8000_K4 | 4 | E | - | - | - |
| 2d_n8000_K4 | 4 | C | 48.04 ± 0.04 (med 48.01; 48.01–48.09; n=3) | strong:cavity2d_n8000: 0.996 ± 0.001 (med 0.997; 0.995–0.997; n=3) | strong:cavity2d_n8000: 1.005 ± 0.002 (med 1.006; 1.003–1.007; n=3) |
| 2d_n1440_k4_K4 | 4 | E | - | - | - |
| 2d_n1440_k4_K4 | 4 | C | 326.77 ± 14.28 (med 334.93; 310.28–335.11; n=3) **two modes: 1× ~310.28 / 2× ~335.02** | weak:cavity2d_n1440: 0.899 ± 0.041 (med 0.921; 0.852–0.923; n=3) **two modes: 1× ~0.852 / 2× ~0.922** | weak:cavity2d_n1440: 0.908 ± 0.042 (med 0.931; 0.859–0.932; n=3) **two modes: 1× ~0.859 / 2× ~0.932** |
| 2d_n2840_k2_K2 | 2 | E | - | - | - |
| 2d_n2840_k2_K2 | 2 | C | 93.40 ± 0.12 (med 93.41; 93.28–93.52; n=3) | weak:cavity2d_n2840: 1.004 ± 0.002 (med 1.004; 1.002–1.005; n=3) | weak:cavity2d_n2840: 1.004 ± 0.002 (med 1.005; 1.003–1.006; n=3) |
| 2d_n2840_k4_K4 | 4 | E | - | - | - |
| 2d_n2840_k4_K4 | 4 | C | 93.17 ± 0.15 (med 93.17; 93.03–93.33; n=3) | weak:cavity2d_n2840: 1.003 ± 0.001 (med 1.003; 1.002–1.004; n=3) | weak:cavity2d_n2840: 1.011 ± 0.004 (med 1.010; 1.008–1.015; n=3) |
| 2d_n5640_k2_K2 | 2 | E | - | - | - |
| 2d_n5640_k2_K2 | 2 | C | 24.93 ± 0.02 (med 24.93; 24.91–24.94; n=3) | weak:cavity2d_n5640: 1.017 ± 0.002 (med 1.018; 1.015–1.019; n=3) | weak:cavity2d_n5640: 1.018 ± 0.002 (med 1.019; 1.016–1.019; n=3) |
| 2d_n5640_k4_K4 | 4 | E | - | - | - |
| 2d_n5640_k4_K4 | 4 | C | 24.83 ± 0.04 (med 24.84; 24.80–24.87; n=3) | weak:cavity2d_n5640: 1.018 ± 0.001 (med 1.018; 1.016–1.018; n=3) | weak:cavity2d_n5640: 1.028 ± 0.002 (med 1.028; 1.026–1.029; n=3) |

### Defrag events (wall time per event: the drain before the defrag, the readbacks, every slab's defrag, up to the next submit)

| point | runs | events per run | wall ms per event (mean; min–max) | drain ms | idle ms | share of the steady window |
|---|---|---|---|---|---|---|
| 2d_n8000_K2 K=1 2d_n8000 | 6 | 1.0 (1.0 in the window) | 284.1; 281.6–289.7 | 82.6 | 201.5 | 0.34 % |
| 2d_n8000_K2 K=2 C | 3 | 1.0 (1.0 in the window) | 247.1; 243.4–254.0 | 40.8 | 206.3 | 0.60 % |
| 2d_n8000_K4 K=1 2d_n8000 | 12 | 1.0 (1.0 in the window) | 283.7; 281.6–290.8 | 83.1 | 200.6 | 0.34 % |
| 2d_n8000_K4 K=4 C | 3 | 1.0 (1.0 in the window) | 234.9; 228.7–247.1 | 20.8 | 214.1 | 1.12 % |
| 2d_n1440_k4_K4 K=1 2d_n1440 | 12 | 1.0 (1.0 in the window) | 13.9; 11.8–19.4 | 2.8 | 11.2 | 0.50 % |
| 2d_n1440_k4_K4 K=4 C | 3 | 1.0 (1.0 in the window) | 45.4; 41.0–49.2 | 0.6 | 44.8 | 1.46 % |
| 2d_n2840_k2_K2 K=1 2d_n2840 | 6 | 1.0 (1.0 in the window) | 45.0; 37.7–76.5 | 10.8 | 34.2 | 0.42 % |
| 2d_n2840_k2_K2 K=2 C | 3 | 1.0 (1.0 in the window) | 70.2; 64.9–80.5 | 10.7 | 59.5 | 0.65 % |
| 2d_n2840_k4_K4 K=1 2d_n2840 | 12 | 1.0 (1.0 in the window) | 38.9; 37.6–44.2 | 10.8 | 28.1 | 0.36 % |
| 2d_n2840_k4_K4 K=4 C | 3 | 1.0 (1.0 in the window) | 129.3; 118.0–151.3 | 10.7 | 118.5 | 1.19 % |
| 2d_n5640_k2_K2 K=1 2d_n5640 | 6 | 1.0 (1.0 in the window) | 143.6; 141.1–153.0 | 40.9 | 102.7 | 0.35 % |
| 2d_n5640_k2_K2 K=2 C | 3 | 1.0 (1.0 in the window) | 244.9; 241.7–251.1 | 40.1 | 204.8 | 0.61 % |
| 2d_n5640_k4_K4 K=1 2d_n5640 | 12 | 1.0 (1.0 in the window) | 142.8; 141.0–153.3 | 41.0 | 101.8 | 0.35 % |
| 2d_n5640_k4_K4 K=4 C | 3 | 1.0 (1.0 in the window) | 452.9; 440.0–474.5 | 40.2 | 412.7 | 1.11 % |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n8000_K2 | 0.9996, 1.0004 | [803] | [803]→[803] kept | - | - | pass: r_chain 0.011/0.099, t_chain 406/3644; r_own 0.011/0.098, t_own 393/3625; r 0.126/0.218, t_tr 1338/7817; T_B 36769 (max 36785), fps -0.15 % |
| 2d_n8000_K4 | 1.0084, 1.0013, 0.9845, 1.0058 | [406, 807, 1201] | [403, 803, 1203]→[407, 805, 1198] changed; [407, 805, 1198]→[406, 807, 1201] changed | - | - | pass: r_chain 0.903/1.015, t_chain 487/18495; r_own 0.031/0.226, t_own 472/2647; r 1.172/1.269, t_tr 8574/23186; T_B 18417 (max 18443), fps -0.36 % |
| 2d_n1440_k4_K4 | 0.9950, 1.0102, 1.0025, 0.9923 | [290, 580, 869] | [291, 579, 867]→[289, 581, 868] changed; [289, 581, 868]→[290, 580, 869] changed | - | - | - |
| 2d_n2840_k2_K2 | 0.9977, 1.0023 | [570] | [571]→[573] changed; [573]→[570] changed | - | - | - |
| 2d_n2840_k4_K4 | 1.0081, 1.0041, 0.9843, 1.0035 | [576, 1146, 1705] | [571, 1139, 1707]→[578, 1144, 1703] changed; [578, 1144, 1703]→[576, 1146, 1705] changed | - | - | - |
| 2d_n5640_k2_K2 | 1.0011, 0.9989 | [1132] | [1131]→[1132] changed; [1132]→[1132] kept | - | - | - |
| 2d_n5640_k4_K4 | 1.0045, 1.0047, 0.9898, 1.0010 | [1136, 2269, 3386] | [1131, 2259, 3387]→[1140, 2271, 3386] changed; [1140, 2271, 3386]→[1136, 2269, 3386] changed | - | - | - |

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n8000_K2 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 6008, relay 5992, fallback 0, blocking 0 (relay slept 5992, released 0)
- 2d_n8000_K4 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 18040, relay 17960, fallback 0, blocking 0 (relay slept 17960, released 0)
- 2d_n1440_k4_K4 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 15180, relay 11820, fallback 0, blocking 0 (relay slept 11811, released 0)
- 2d_n2840_k2_K2 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 4505, relay 4495, fallback 0, blocking 0 (relay slept 4495, released 0)
- 2d_n2840_k4_K4 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 13635, relay 13365, fallback 0, blocking 0 (relay slept 13365, released 0)
- 2d_n5640_k2_K2 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 4511, relay 4489, fallback 0, blocking 0 (relay slept 4489, released 0)
- 2d_n5640_k4_K4 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 13525, relay 13475, fallback 0, blocking 0 (relay slept 13475, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n8000_K2_t1_R1_2d_n8000 | 2 | ok | 0.0 | 5 | 4 | 34 |
| 2d_n8000_K2_t2_R1_2d_n8000 | 2 | ok | 0.0 | 0 | 0 | 30 |
| 2d_n8000_K2_t3_R1_2d_n8000 | 2 | ok | 0.0 | 14 | 14 | 16 |
| 2d_n8000_K4_t1_R1_2d_n8000 | 4 | ok | 0.1 | 10 | 10 | 611 |
| 2d_n8000_K4_t2_R1_2d_n8000 | 4 | ok | 0.5 | 13 | 13 | 603 |
| 2d_n8000_K4_t3_R1_2d_n8000 | 4 | ok | 0.0 | 15 | 15 | 613 |
| 2d_n1440_k4_K4_t1_R1_2d_n1440 | 4 | ok | 0.1 | 8 | 8 | 23 |
| 2d_n1440_k4_K4_t2_R1_2d_n1440 | 4 | ok | 0.0 | 18 | 18 | 40 |
| 2d_n1440_k4_K4_t3_R1_2d_n1440 | 4 | ok | 0.0 | 11 | 11 | 16 |
| 2d_n2840_k2_K2_t1_R1_2d_n2840 | 2 | ok | 0.0 | 8 | 8 | 8 |
| 2d_n2840_k2_K2_t2_R1_2d_n2840 | 2 | ok | 0.1 | 4 | 4 | 14 |
| 2d_n2840_k2_K2_t3_R1_2d_n2840 | 2 | ok | 0.1 | 0 | 0 | 10 |
| 2d_n2840_k4_K4_t1_R1_2d_n2840 | 4 | ok | 1.6 | 10 | 10 | 144 |
| 2d_n2840_k4_K4_t2_R1_2d_n2840 | 4 | ok | 0.1 | 19 | 19 | 61 |
| 2d_n2840_k4_K4_t3_R1_2d_n2840 | 4 | ok | 1.7 | 15 | 15 | 115 |
| 2d_n5640_k2_K2_t1_R1_2d_n5640 | 2 | ok | 0.0 | 11 | 11 | 1 |
| 2d_n5640_k2_K2_t2_R1_2d_n5640 | 2 | ok | 0.1 | 18 | 18 | 18 |
| 2d_n5640_k2_K2_t3_R1_2d_n5640 | 2 | ok | 0.0 | 11 | 11 | 11 |
| 2d_n5640_k4_K4_t1_R1_2d_n5640 | 4 | ok | 0.9 | 17 | 17 | 316 |
| 2d_n5640_k4_K4_t2_R1_2d_n5640 | 4 | ok | 0.1 | 17 | 17 | 313 |
| 2d_n5640_k4_K4_t3_R1_2d_n5640 | 4 | ok | 0.1 | 15 | 15 | 323 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n8000_K2 | 2738 / 575 | 2713 / 575 | 25 MHz / -0 W |
| 2d_n8000_K4 | 2755 / 575 | 2701 / 575 | 54 MHz / -0 W |
| 2d_n1440_k4_K4 | 2818 / 536 | 2775 / 575 | 42 MHz / -39 W |
| 2d_n2840_k2_K2 | 2771 / 575 | 2762 / 575 | 9 MHz / -0 W |
| 2d_n2840_k4_K4 | 2770 / 575 | 2758 / 575 | 12 MHz / 0 W |
| 2d_n5640_k2_K2 | 2738 / 575 | 2731 / 575 | 7 MHz / 0 W |
| 2d_n5640_k4_K4 | 2733 / 575 | 2715 / 575 | 18 MHz / 0 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n8000_K2 | K runs | 2 | 3 | 1.94 | 11.75 | 2.86 | 16.55 | 3.94 | 33.04 |
| 2d_n8000_K2 | references | 1 | 4 | 1.98 | 8.51 | 3.61 | 14.10 | 2.98 | 34.61 |
| 2d_n8000_K4 | K runs | 4 | 3 | 1.96 | 15.61 | 4.58 | 22.15 | 5.96 | 33.48 |
| 2d_n8000_K4 | references | 1 | 8 | 2.04 | 8.59 | 4.89 | 15.51 | 3.04 | 39.25 |
| 2d_n1440_k4_K4 | K runs | 4 | 3 | 0.28 | 1.69 | 4.22 | 6.19 | 4.28 | 5.18 |
| 2d_n1440_k4_K4 | references | 1 | 8 | 0.10 | 0.25 | 3.97 | 4.31 | 1.10 | 1.44 |
| 2d_n2840_k2_K2 | K runs | 2 | 3 | 0.52 | 2.78 | 3.96 | 7.25 | 2.52 | 9.34 |
| 2d_n2840_k2_K2 | references | 1 | 4 | 0.29 | 1.03 | 3.40 | 4.71 | 1.29 | 4.92 |
| 2d_n2840_k4_K4 | K runs | 4 | 3 | 1.01 | 7.60 | 4.46 | 13.06 | 5.01 | 18.64 |
| 2d_n2840_k4_K4 | references | 1 | 8 | 0.30 | 1.04 | 3.11 | 4.44 | 1.30 | 5.52 |
| 2d_n5640_k2_K2 | K runs | 2 | 3 | 1.92 | 11.72 | 2.81 | 16.45 | 3.92 | 32.70 |
| 2d_n5640_k2_K2 | references | 1 | 4 | 1.01 | 4.23 | 3.39 | 8.64 | 2.01 | 17.36 |
| 2d_n5640_k4_K4 | K runs | 4 | 3 | 3.82 | 30.87 | 4.70 | 39.39 | 7.82 | 65.34 |
| 2d_n5640_k4_K4 | references | 1 | 8 | 1.04 | 4.28 | 3.81 | 9.12 | 2.04 | 20.13 |

### anatomy anatomy_2d_n8000_K1

- status pass ; steps 3000, fps 12.1; drift 0, overflow 0, far migration 0, stamps (0, 0), seam checks None; anatomy frames 3, [loop] intervals 3, defrag times 5
- loop rows 3000 in 3 segments (/dev/shm/scxm138/e7_b2A_1694327/traces/anatomy_2d_n8000_K1)
- own pool watermark per defrag (max over slabs: used fraction / interval migration): f1000: 0.0000/0, f2000: 0.0000/0, f3000: 0.0000/0
- host loop f1000 (1000 frames, V7_LOOP_TRACE): period p50 81.984 ms (max 82.954); in _submit_frame p50 0.013 / mean 0.022 / max 2.988 ms; blocked in frame waits p50 81.963 / mean 81.707 ms; cpu share 0.0
- host loop f2000 (1000 frames, V7_LOOP_TRACE): period p50 82.557 ms (max 83.068); in _submit_frame p50 0.013 / mean 0.017 / max 2.944 ms; blocked in frame waits p50 82.544 / mean 82.516 ms; cpu share 0.0
- host loop f3000 (1000 frames, V7_LOOP_TRACE): period p50 82.66 ms (max 83.264); in _submit_frame p50 0.013 / mean 0.019 / max 2.944 ms; blocked in frame waits p50 82.646 / mean 82.696 ms; cpu share 0.0
- defrag sim0 (every call, bootstrap included): wall_ms mean 198.0 (max 204.7, n=5); gpu_us mean 197141.2 (max 203433.5, n=5)

GPU µs per frame, mean over the sampled frames (n = 3 per simulator):

| item | sim0 |
|---|---|
| predict | 4122.5 |
| update_voxel | 2250.8 |
| phase_a | 6373.2 |
| a_to_b_gap | 2.2 |
| correction_density_interior | 36886.3 |
| force_deep_interior | 38883.0 |
| phase_b | 75769.3 |
| b_to_c_gap | 3.7 |
| correction_density_boundary | 0.3 |
| density_copy | 722.8 |
| force | 0.0 |
| phase_c | 723.0 |

## e7_b2B_1694328

- host wqd10nbj04g2, slurm job 1694328, harness 99fee6d9f46e, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- provenance at job end: unchanged
- runs tagged rc2: 182, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1694328; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 16 files, 192 MiB in /dev/shm/scxm138/glcache_1694328; new files in ~/.cache/nvidia: 0
- runs: anatomy pass 1; calibrated pass 21; reference pass 156; trace_calibrated pass 4

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n11320_K8 | F1 | 8 | E | - | pairs:cavity2d_n11320: - | pairs:cavity2d_n11320: - |
| 2d_n11320_K8 | F1 | 8 | C | 47.79 ± 0.06 (med 47.80; 47.73–47.85; n=3) | pairs:cavity2d_n11320: 0.964 ± 0.001 (med 0.964; 0.963–0.965; n=3) | pairs:cavity2d_n11320: 0.969 ± 0.001 (med 0.969; 0.968–0.969; n=3) |
| 2d_n520_K8 | F7 | 8 | E | - | strong:cavity2d_n520: - | strong:cavity2d_n520: - |
| 2d_n520_K8 | F7 | 8 | C | 492.53 ± 3.40 (med 491.20; 490.00–496.40; n=3) | strong:cavity2d_n520: 0.029 ± 0.000 (med 0.029; 0.029–0.029; n=3) | strong:cavity2d_n520: 0.029 ± 0.000 (med 0.029; 0.029–0.030; n=3) |
| 2d_n1000_K8 | F7 | 8 | E | - | strong:cavity2d_n1000: - | strong:cavity2d_n1000: - |
| 2d_n1000_K8 | F7 | 8 | C | 458.40 ± 9.53 (med 456.60; 449.90–468.70; n=3) | strong:cavity2d_n1000: 0.078 ± 0.002 (med 0.077; 0.076–0.079; n=3) | strong:cavity2d_n1000: 0.078 ± 0.001 (med 0.078; 0.078–0.080; n=3) **two modes: 2× ~0.078 / 1× ~0.080** |
| 2d_n1440_k8_K8 | F3 | 8 | E | - | weak:cavity2d_n1440: - | weak:cavity2d_n1440: - |
| 2d_n1440_k8_K8 | F3 | 8 | C | 326.10 ± 0.10 (med 326.10; 326.00–326.20; n=3) | weak:cavity2d_n1440: 0.902 ± 0.001 (med 0.902; 0.901–0.903; n=3) | weak:cavity2d_n1440: 0.910 ± 0.000 (med 0.910; 0.910–0.911; n=3) |
| 2d_n2840_k8_K8 | F3 | 8 | E | - | weak:cavity2d_n2840: - | weak:cavity2d_n2840: - |
| 2d_n2840_k8_K8 | F3 | 8 | C | 91.05 ± 0.05 (med 91.07; 90.99–91.07; n=3) | weak:cavity2d_n2840: 0.982 ± 0.000 (med 0.982; 0.982–0.983; n=3) | weak:cavity2d_n2840: 0.994 ± 0.002 (med 0.993; 0.993–0.995; n=3) |
| 2d_n5640_k8_K8 | F3 | 8 | E | - | weak:cavity2d_n5640: - | weak:cavity2d_n5640: - |
| 2d_n5640_k8_K8 | F3 | 8 | C | 24.31 ± 0.01 (med 24.30; 24.30–24.32; n=3) | weak:cavity2d_n5640: 0.998 ± 0.003 (med 0.999; 0.995–1.000; n=3) | weak:cavity2d_n5640: 1.008 ± 0.003 (med 1.009; 1.005–1.010; n=3) |
| x_2d_n8000_K8 | F1+F7 | 8 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| x_2d_n8000_K8 | F1+F7 | 8 | C | 90.42 ± 0.14 (med 90.50; 90.25–90.50; n=3) | strong:cavity2d_n8000: 0.940 ± 0.001 (med 0.940; 0.938–0.941; n=3) | strong:cavity2d_n8000: 0.948 ± 0.001 (med 0.948; 0.946–0.949; n=3) |

Without the defrag events (user 2026-10-12): fps = steady_frames / (steady_s - the wall time of the events in the window), the same on the K run and on every K = 1 member; eta from these. The defrag event table lists the events themselves.

| point | K | arm | fps without defrag | eta without defrag (type: reference) | eta_min without defrag |
|---|---|---|---|---|---|
| 2d_n11320_K8 | 8 | E | - | - | - |
| 2d_n11320_K8 | 8 | C | 48.81 ± 0.06 (med 48.83; 48.74–48.86; n=3) | pairs:cavity2d_n11320: 0.978 ± 0.001 (med 0.979; 0.977–0.979; n=3) | pairs:cavity2d_n11320: 0.984 ± 0.001 (med 0.984; 0.983–0.984; n=3) |
| 2d_n520_K8 | 8 | E | - | - | - |
| 2d_n520_K8 | 8 | C | 498.04 ± 3.84 (med 496.48; 495.23–502.42; n=3) | strong:cavity2d_n520: 0.029 ± 0.000 (med 0.029; 0.029–0.029; n=3) | strong:cavity2d_n520: 0.029 ± 0.000 (med 0.029; 0.029–0.030; n=3) |
| 2d_n1000_K8 | 8 | E | - | - | - |
| 2d_n1000_K8 | 8 | C | 463.72 ± 9.54 (med 461.98; 455.16–474.01; n=3) | strong:cavity2d_n1000: 0.078 ± 0.002 (med 0.078; 0.077–0.080; n=3) | strong:cavity2d_n1000: 0.079 ± 0.001 (med 0.078; 0.078–0.081; n=3) **two modes: 2× ~0.078 / 1× ~0.081** |
| 2d_n1440_k8_K8 | 8 | E | - | - | - |
| 2d_n1440_k8_K8 | 8 | C | 335.73 ± 0.00 (med 335.73; 335.72–335.73; n=3) | weak:cavity2d_n1440: 0.924 ± 0.001 (med 0.924; 0.924–0.925; n=3) | weak:cavity2d_n1440: 0.933 ± 0.001 (med 0.933; 0.933–0.934; n=3) |
| 2d_n2840_k8_K8 | 8 | E | - | - | - |
| 2d_n2840_k8_K8 | 8 | C | 93.07 ± 0.03 (med 93.07; 93.05–93.11; n=3) | weak:cavity2d_n2840: 1.001 ± 0.001 (med 1.001; 1.000–1.001; n=3) | weak:cavity2d_n2840: 1.012 ± 0.002 (med 1.011; 1.011–1.014; n=3) |
| 2d_n5640_k8_K8 | 8 | E | - | - | - |
| 2d_n5640_k8_K8 | 8 | C | 24.82 ± 0.01 (med 24.82; 24.81–24.83; n=3) | weak:cavity2d_n5640: 1.016 ± 0.003 (med 1.017; 1.012–1.018; n=3) | weak:cavity2d_n5640: 1.026 ± 0.003 (med 1.027; 1.022–1.027; n=3) |
| x_2d_n8000_K8 | 8 | E | - | - | - |
| x_2d_n8000_K8 | 8 | C | 92.28 ± 0.12 (med 92.32; 92.14–92.38; n=3) | strong:cavity2d_n8000: 0.956 ± 0.001 (med 0.956; 0.955–0.956; n=3) | strong:cavity2d_n8000: 0.964 ± 0.001 (med 0.965; 0.963–0.965; n=3) |

### Defrag events (wall time per event: the drain before the defrag, the readbacks, every slab's defrag, up to the next submit)

| point | runs | events per run | wall ms per event (mean; min–max) | drain ms | idle ms | share of the steady window |
|---|---|---|---|---|---|---|
| 2d_n11320_K8 K=1 2d_n11320 | 12 | 1.0 (1.0 in the window) | 480.9; 479.8–483.8 | 80.3 | 400.6 | 0.60 % |
| 2d_n11320_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 435.9; 435.5–436.7 | 20.5 | 415.4 | 2.08 % |
| 2d_n520_K8 K=1 2d_n520 | 24 | 1.0 (1.0 in the window) | 5.7; 3.1–10.3 | 0.0 | 5.7 | 1.21 % |
| 2d_n520_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 22.5; 21.6–24.2 | 0.7 | 21.8 | 1.11 % |
| 2d_n1000_K8 K=1 2d_n1000 | 24 | 1.0 (1.0 in the window) | 7.5; 5.3–11.0 | 0.2 | 7.3 | 0.55 % |
| 2d_n1000_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 25.0; 23.8–25.8 | 1.3 | 23.7 | 1.15 % |
| 2d_n1440_k8_K8 K=1 2d_n1440 | 24 | 1.0 (1.0 in the window) | 12.8; 11.6–15.8 | 2.5 | 10.3 | 0.46 % |
| 2d_n1440_k8_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 88.0; 87.1–88.4 | 1.4 | 86.6 | 2.87 % |
| 2d_n2840_k8_K8 K=1 2d_n2840 | 24 | 1.0 (1.0 in the window) | 38.9; 37.8–42.7 | 10.8 | 28.2 | 0.36 % |
| 2d_n2840_k8_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 239.7; 236.0–241.5 | 10.7 | 228.9 | 2.18 % |
| 2d_n5640_k8_K8 K=1 2d_n5640 | 24 | 1.0 (1.0 in the window) | 142.1; 141.1–145.4 | 41.0 | 101.2 | 0.35 % |
| 2d_n5640_k8_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 847.1; 843.3–849.7 | 40.3 | 806.8 | 2.06 % |
| x_2d_n8000_K8 K=1 2d_n8000 | 24 | 1.0 (1.0 in the window) | 282.4; 281.6–283.4 | 83.0 | 199.4 | 0.34 % |
| x_2d_n8000_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 222.8; 221.0–225.4 | 7.9 | 214.9 | 2.01 % |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n11320_K8 | 1.0098, 1.0032, 0.9880, 0.9984, 1.0013, 0.9905, 1.0017, 1.0071 | [289, 573, 852, 1135, 1418, 1699, 1982] | [286, 569, 852, 1135, 1418, 1701, 1984]→[290, 576, 854, 1136, 1419, 1698, 1981] changed; [290, 576, 854, 1136, 1419, 1698, 1981]→[289, 573, 852, 1135, 1418, 1699, 1982] changed | - | - | pass: r_chain 0.858/1.015, t_chain 799/18119; r_own 0.064/0.094, t_own 756/1336; r 1.018/1.224, t_tr 1074/21838; T_B 17843 (max 18014), fps -0.60 % |
| 2d_n520_K8 | 1.0151, 0.9901, 0.9916, 0.9888, 0.9864, 1.0316, 0.9799, 1.0164 | [16, 29, 42, 55, 68, 81, 94] | [16, 29, 42, 55, 68, 81, 94]→[16, 29, 42, 55, 68, 81, 94] kept | - | - | pass: r_chain 11.149/12.473, t_chain 1461/1959; r_own 10.458/11.488, t_own 1298/1854; r 11.461/12.945, t_tr 1558/2024; T_B 167 (max 172), fps -5.91 % |
| 2d_n1000_K8 | 0.9452, 1.0203, 1.0006, 0.9890, 1.0547, 0.9908, 1.0029, 0.9965 | [27, 52, 77, 102, 128, 153, 178] | [28, 53, 78, 103, 128, 153, 178]→[27, 53, 78, 102, 128, 153, 178] changed; [27, 53, 78, 102, 128, 153, 178]→[27, 52, 77, 102, 128, 153, 178] changed | - | - | pass: r_chain 9.169/10.684, t_chain 1625/2239; r_own 8.529/9.238, t_own 1484/1944; r 9.406/10.885, t_tr 1718/2321; T_B 220 (max 238), fps -2.38 % |
| 2d_n1440_k8_K8 | 1.0100, 1.0058, 0.9925, 0.9998, 0.9994, 0.9986, 0.9916, 1.0023 | [294, 584, 869, 1157, 1445, 1733, 2018] | [291, 579, 867, 1155, 1443, 1731, 2019]→[293, 586, 869, 1157, 1448, 1730, 2017] changed; [293, 586, 869, 1157, 1448, 1730, 2017]→[294, 584, 869, 1157, 1445, 1733, 2018] changed | - | - | - |
| 2d_n2840_k8_K8 | 1.0118, 1.0017, 0.9911, 1.0010, 1.0038, 0.9942, 0.9934, 1.0031 | [578, 1147, 1710, 2278, 2848, 3413, 3977] | [571, 1139, 1707, 2275, 2843, 3411, 3979]→[576, 1144, 1707, 2276, 2847, 3411, 3974] changed; [576, 1144, 1707, 2276, 2847, 3411, 3974]→[578, 1147, 1710, 2278, 2848, 3413, 3977] changed | - | - | - |
| 2d_n5640_k8_K8 | 1.0045, 1.0012, 0.9926, 0.9998, 1.0059, 0.9961, 0.9972, 1.0027 | [1136, 2265, 3385, 4513, 5648, 6771, 7896] | [1131, 2259, 3387, 4515, 5643, 6771, 7899]→[1139, 2270, 3387, 4514, 5653, 6770, 7892] changed; [1139, 2270, 3387, 4514, 5653, 6770, 7892]→[1136, 2265, 3385, 4513, 5648, 6771, 7896] changed | - | - | - |
| x_2d_n8000_K8 | 1.0124, 1.0073, 0.9860, 1.0006, 1.0013, 0.9812, 1.0014, 1.0097 | [205, 407, 604, 804, 1005, 1201, 1401] | [203, 403, 603, 803, 1003, 1203, 1403]→[206, 407, 602, 803, 1003, 1198, 1400] changed; [206, 407, 602, 803, 1003, 1198, 1400]→[205, 407, 604, 804, 1005, 1201, 1401] changed | - | - | pass: r_chain 0.489/0.952, t_chain 560/8766; r_own 0.080/0.174, t_own 529/1309; r 0.783/1.120, t_tr 2256/10213; T_B 9242 (max 9407), fps -0.54 % |

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n11320_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 48325, relay 35675, fallback 0, blocking 0 (relay slept 35620, released 0)
- 2d_n520_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 83730, relay 270, fallback 0, blocking 0 (relay slept 268, released 0)
- 2d_n1000_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 83209, relay 791, fallback 0, blocking 0 (relay slept 785, released 0)
- 2d_n1440_k8_K8 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 36579, relay 26421, fallback 0, blocking 0 (relay slept 26405, released 0)
- 2d_n2840_k8_K8 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 33081, relay 29919, fallback 0, blocking 0 (relay slept 29903, released 0)
- 2d_n5640_k8_K8 C: 3 of 3 runs (3 with V7_DEST_GUARD=relay): precheck 35265, relay 27735, fallback 0, blocking 0 (relay slept 27708, released 0)
- x_2d_n8000_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 43179, relay 40821, fallback 0, blocking 0 (relay slept 40804, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n11320_K8_t1_RP_2d_n11320 | 4 | ok | 0.2 | 5 | 5 | 550 |
| 2d_n11320_K8_t2_RP_2d_n11320 | 4 | ok | 0.2 | 13 | 14 | 545 |
| 2d_n11320_K8_t3_RP_2d_n11320 | 4 | ok | 0.1 | 18 | 19 | 582 |
| 2d_n520_K8_t1_R1_2d_n520 | 8 | ok | 3.0 | 16 | 16 | 15 |
| 2d_n520_K8_t2_R1_2d_n520 | 8 | ok | 4.6 | 18 | 18 | 26 |
| 2d_n520_K8_t3_R1_2d_n520 | 8 | ok | 1.4 | 17 | 17 | 22 |
| 2d_n1000_K8_t1_R1_2d_n1000 | 8 | ok | 7.0 | 20 | 20 | 166 |
| 2d_n1000_K8_t2_R1_2d_n1000 | 8 | ok | 2.2 | 18 | 18 | 66 |
| 2d_n1000_K8_t3_R1_2d_n1000 | 8 | ok | 1.3 | 20 | 20 | 25 |
| 2d_n1440_k8_K8_t1_R1_2d_n1440 | 8 | ok | 3.7 | 19 | 19 | 260 |
| 2d_n1440_k8_K8_t2_R1_2d_n1440 | 8 | ok | 3.3 | 19 | 19 | 97 |
| 2d_n1440_k8_K8_t3_R1_2d_n1440 | 8 | ok | 2.6 | 18 | 18 | 223 |
| 2d_n2840_k8_K8_t1_R1_2d_n2840 | 8 | ok | 5.0 | 18 | 17 | 261 |
| 2d_n2840_k8_K8_t2_R1_2d_n2840 | 8 | ok | 0.6 | 18 | 19 | 135 |
| 2d_n2840_k8_K8_t3_R1_2d_n2840 | 8 | ok | 0.4 | 20 | 19 | 226 |
| 2d_n5640_k8_K8_t1_R1_2d_n5640 | 8 | ok | 0.2 | 19 | 19 | 349 |
| 2d_n5640_k8_K8_t2_R1_2d_n5640 | 8 | ok | 0.1 | 19 | 19 | 388 |
| 2d_n5640_k8_K8_t3_R1_2d_n5640 | 8 | ok | 0.4 | 19 | 20 | 304 |
| x_2d_n8000_K8_t1_R1_2d_n8000 | 8 | ok | 0.3 | 16 | 16 | 686 |
| x_2d_n8000_K8_t2_R1_2d_n8000 | 8 | ok | 0.3 | 12 | 13 | 591 |
| x_2d_n8000_K8_t3_R1_2d_n8000 | 8 | ok | 0.2 | 19 | 19 | 625 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n11320_K8 | 2764 / 575 | 2713 / 575 | 51 MHz / 0 W |
| 2d_n520_K8 | 2718 / 93 | 2361 / 77 | 356 MHz / 16 W |
| 2d_n1000_K8 | 2663 / 110 | 2786 / 451 | -123 MHz / -341 W |
| 2d_n1440_k8_K8 | 2814 / 534 | 2778 / 575 | 36 MHz / -42 W |
| 2d_n2840_k8_K8 | 2773 / 575 | 2753 / 575 | 20 MHz / -0 W |
| 2d_n5640_k8_K8 | 2743 / 575 | 2720 / 575 | 24 MHz / -0 W |
| x_2d_n8000_K8 | 2779 / 575 | 2705 / 575 | 74 MHz / -0 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n11320_K8 | K runs | 8 | 3 | 3.92 | 45.90 | 6.05 | 55.87 | 11.92 | 66.72 |
| 2d_n11320_K8 | references | 2 | 8 | 4.02 | 20.85 | 5.09 | 29.96 | 6.02 | 82.94 |
| 2d_n520_K8 | K runs | 8 | 3 | 0.04 | 0.06 | 3.05 | 3.15 | 8.04 | 0.37 |
| 2d_n520_K8 | references | 1 | 16 | 0.04 | 0.04 | 3.20 | 3.28 | 1.04 | 0.27 |
| 2d_n1000_K8 | K runs | 8 | 3 | 0.06 | 0.23 | 4.33 | 4.62 | 8.06 | 0.77 |
| 2d_n1000_K8 | references | 1 | 16 | 0.07 | 0.12 | 3.32 | 3.50 | 1.07 | 0.74 |
| 2d_n1440_k8_K8 | K runs | 8 | 3 | 0.52 | 5.26 | 5.92 | 11.70 | 8.52 | 10.96 |
| 2d_n1440_k8_K8 | references | 1 | 16 | 0.10 | 0.24 | 3.16 | 3.50 | 1.10 | 1.31 |
| 2d_n2840_k8_K8 | K runs | 8 | 3 | 1.97 | 22.87 | 6.48 | 31.32 | 9.97 | 37.35 |
| 2d_n2840_k8_K8 | references | 1 | 16 | 0.29 | 1.05 | 3.53 | 4.86 | 1.28 | 7.58 |
| 2d_n5640_k8_K8 | K runs | 8 | 3 | 7.67 | 91.87 | 7.42 | 106.96 | 15.67 | 130.11 |
| 2d_n5640_k8_K8 | references | 1 | 16 | 1.04 | 4.27 | 3.33 | 8.64 | 2.04 | 25.01 |
| x_2d_n8000_K8 | K runs | 8 | 3 | 1.92 | 22.68 | 6.59 | 31.19 | 9.92 | 36.97 |
| x_2d_n8000_K8 | references | 1 | 16 | 2.01 | 8.58 | 4.23 | 14.81 | 3.01 | 49.17 |

### anatomy anatomy_2d_n8000_K8

- status pass ; steps 3000, fps 88.7; drift 0, overflow 0, far migration 0, stamps (0, 0), seam checks None; anatomy frames 24, [loop] intervals 3, defrag times 40
- loop rows 3000 in 3 segments (/dev/shm/scxm138/e7_b2B_1694328/traces/anatomy_2d_n8000_K8)
- own pool watermark per defrag (max over slabs: used fraction / interval migration): f1000: 0.8333/37, f2000: 0.8333/75, f3000: 0.8333/98
- host loop f1000 (1000 frames, V7_LOOP_TRACE): period p50 10.98 ms (max 67.603); in _submit_frame p50 0.315 / mean 0.508 / max 20.292 ms; blocked in frame waits p50 10.666 / mean 10.587 ms; cpu share 0.046
- host loop f2000 (1000 frames, V7_LOOP_TRACE): period p50 10.984 ms (max 25.148); in _submit_frame p50 0.31 / mean 0.501 / max 24.86 ms; blocked in frame waits p50 10.673 / mean 10.476 ms; cpu share 0.046
- host loop f3000 (1000 frames, V7_LOOP_TRACE): period p50 11.047 ms (max 26.579); in _submit_frame p50 0.282 / mean 0.541 / max 25.111 ms; blocked in frame waits p50 10.762 / mean 10.5 ms; cpu share 0.049
- defrag sim0 (every call, bootstrap included): wall_ms mean 47.2 (max 137.8, n=5); gpu_us mean 46029.3 (max 135305.5, n=5)
- defrag sim1 (every call, bootstrap included): wall_ms mean 48.9 (max 147.4, n=5); gpu_us mean 47818.4 (max 145354.8, n=5)
- defrag sim2 (every call, bootstrap included): wall_ms mean 54.8 (max 176.8, n=5); gpu_us mean 53314.6 (max 172571.4, n=5)
- defrag sim3 (every call, bootstrap included): wall_ms mean 64.6 (max 225.9, n=5); gpu_us mean 62641.2 (max 219238.4, n=5)
- defrag sim4 (every call, bootstrap included): wall_ms mean 41.1 (max 108.7, n=5); gpu_us mean 39781.3 (max 105077.5, n=5)
- defrag sim5 (every call, bootstrap included): wall_ms mean 74.0 (max 272.8, n=5); gpu_us mean 72923.4 (max 270784.8, n=5)
- defrag sim6 (every call, bootstrap included): wall_ms mean 63.4 (max 219.4, n=5); gpu_us mean 61629.7 (max 214141.2, n=5)
- defrag sim7 (every call, bootstrap included): wall_ms mean 51.1 (max 156.7, n=5); gpu_us mean 49315.1 (max 151485.4, n=5)

GPU µs per frame, mean over the sampled frames (n = 3 per simulator):

| item | sim0 | sim1 | sim2 | sim3 | sim4 | sim5 | sim6 | sim7 |
|---|---|---|---|---|---|---|---|---|
| predict | 514.7 | 508.1 | 509.1 | 509.4 | 510.0 | 508.6 | 509.1 | 512.3 |
| update_voxel | 336.5 | 331.0 | 333.7 | 338.8 | 337.8 | 338.9 | 331.7 | 336.0 |
| ghost_send_trailing | 9.0 | 8.2 | 8.2 | 9.0 | 8.2 | 8.2 | 8.8 | - |
| phase_a | 860.2 | 856.9 | 860.3 | 867.5 | 865.5 | 865.2 | 859.8 | 858.6 |
| a_to_b_gap | 2.5 | 2.3 | 2.1 | 2.2 | 2.1 | 2.0 | 2.0 | 2.0 |
| correction_density_interior | 4384.4 | 4334.6 | 4489.7 | 4377.7 | 4327.6 | 4471.6 | 4337.2 | 4390.9 |
| force_deep_interior | 4854.8 | 4807.7 | 4970.2 | 4861.7 | 4819.0 | 4948.7 | 4831.6 | 4897.8 |
| phase_b | 9239.2 | 9142.3 | 9459.9 | 9239.4 | 9146.6 | 9420.3 | 9168.8 | 9288.7 |
| b_to_c_gap | 3109.8 | 2497.4 | 4.2 | 1458.1 | 698.4 | 3.6 | 82.9 | 3.8 |
| expand_lists | 7.3 | 9.0 | 11.2 | 9.9 | 9.5 | 10.3 | 9.6 | 8.5 |
| install_trailing | 4.1 | 2.0 | 2.0 | 2.0 | 2.7 | 2.0 | 2.0 | - |
| append_departed | 2.7 | 2.7 | 4.1 | 3.4 | 4.1 | 3.4 | 4.1 | 3.4 |
| correction_density_boundary | 175.2 | 251.2 | 256.7 | 253.0 | 250.3 | 253.0 | 257.1 | 177.5 |
| density_copy | 74.7 | 79.9 | 80.6 | 79.4 | 78.8 | 79.8 | 79.4 | 75.8 |
| force | 179.6 | 327.7 | 335.2 | 332.2 | 329.5 | 331.5 | 331.5 | 183.3 |
| phase_c | 443.6 | 677.4 | 693.8 | 684.1 | 678.9 | 684.3 | 688.0 | 453.3 |
| readback_trailing_dma | 113.4 | 115.1 | 108.0 | 115.5 | 96.0 | 99.9 | 88.7 | - |
| readback_trailing_barrier | 1.4 | 0.9 | 0.8 | 0.9 | 0.8 | 0.9 | 0.8 | - |
| upload_trailing_dma | 77.2 | 87.1 | 86.8 | 74.8 | 74.3 | 73.7 | 74.2 | - |
| readback_trailing_sched_gap | 5679.1 | 4918.4 | 2152.9 | 4004.8 | 3064.3 | 2243.5 | 1453.0 | - |
| upload_trailing_to_c_gap | 4765.6 | 2565.6 | 6023.5 | 2066.2 | 64.4 | 6203.6 | 1206.7 | - |
| install_sum | 14.1 | 18.6 | 21.4 | 19.5 | 20.4 | 20.0 | 19.9 | 16.7 |
| ghost_send_leading | - | 9.6 | 9.3 | 10.2 | 9.6 | 9.6 | 10.2 | 10.3 |
| install_leading | - | 4.8 | 4.1 | 4.1 | 4.1 | 4.2 | 4.1 | 4.8 |
| readback_leading_dma | - | 124.5 | 109.1 | 126.4 | 91.6 | 101.4 | 102.1 | 102.0 |
| readback_leading_barrier | - | 1.1 | 1.3 | 1.0 | 1.4 | 1.0 | 1.1 | 1.1 |
| upload_leading_dma | - | 78.4 | 77.0 | 87.5 | 74.5 | 74.3 | 74.3 | 73.6 |
| readback_leading_sched_gap | - | 4798.3 | 2047.9 | 3883.5 | 2976.9 | 2146.6 | 1356.0 | 541.9 |
| upload_leading_to_c_gap | - | 5535.1 | 6133.8 | 2680.1 | 5090.4 | 6352.1 | 1281.5 | 7032.8 |

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| b2A | 1694327 | wqd10nbj04g2 | 2026-10-13T02:49:07 | 2026-10-13T04:19:28 | COMPLETED | 1.51 | 1.66 |
| b2B | 1694328 | wqd10nbj04g2 | 2026-10-13T04:19:28 | 2026-10-13T05:56:18 | COMPLETED | 1.61 | 1.74 |
