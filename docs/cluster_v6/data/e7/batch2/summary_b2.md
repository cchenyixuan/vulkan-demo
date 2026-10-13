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

## e7_b2S_1695411

- host wqd10nbj04g2, slurm job 1695411, harness 1592f3cd9db7, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- provenance at job end: unchanged
- runs tagged rc2: 197, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1695411; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 16 files, 138 MiB in /dev/shm/scxm138/glcache_1695411; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 27; reference pass 166; trace_calibrated pass 4

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n4000_K8 | F1+F7 | 8 | E | - | strong:cavity2d_n4000: - | strong:cavity2d_n4000: - |
| 2d_n4000_K8 | F1+F7 | 8 | C | 326.80 ± 2.01 (med 327.70; 324.50–328.20; n=3) | strong:cavity2d_n4000: 0.866 ± 0.003 (med 0.867; 0.863–0.868; n=3) | strong:cavity2d_n4000: 0.874 ± 0.003 (med 0.875; 0.871–0.877; n=3) |
| 2d_n1440_k2_K2 | F3 | 2 | E | - | weak:cavity2d_n1440: - | weak:cavity2d_n1440: - |
| 2d_n1440_k2_K2 | F3 | 2 | C | 335.98 ± 2.98 (med 335.70; 331.60–339.50; n=5) | weak:cavity2d_n1440: 0.926 ± 0.007 (med 0.925; 0.915–0.934; n=5) | weak:cavity2d_n1440: 0.927 ± 0.007 (med 0.927; 0.916–0.935; n=5) |
| 2d_n1440_K8 | F1 | 8 | E | - | strong:cavity2d_n1440: - | strong:cavity2d_n1440: - |
| 2d_n1440_K8 | F1 | 8 | C | 468.10 ± 3.32 (med 469.60; 464.30–470.40; n=3) | strong:cavity2d_n1440: 0.162 ± 0.001 (med 0.162; 0.160–0.162; n=3) | strong:cavity2d_n1440: 0.163 ± 0.001 (med 0.164; 0.162–0.164; n=3) |
| 2d_n2000_K8 | F1+F7 | 8 | E | - | strong:cavity2d_n2000: - | strong:cavity2d_n2000: - |
| 2d_n2000_K8 | F1+F7 | 8 | C | 451.90 ± 6.78 (med 451.20; 445.50–459.00; n=3) | strong:cavity2d_n2000: 0.308 ± 0.005 (med 0.308; 0.304–0.313; n=3) | strong:cavity2d_n2000: 0.311 ± 0.005 (med 0.310; 0.307–0.316; n=3) |
| 2d_n2840_K8 | F1 | 8 | E | - | strong:cavity2d_n2840: - | strong:cavity2d_n2840: - |
| 2d_n2840_K8 | F1 | 8 | C | 429.10 ± 2.95 (med 430.00; 425.80–431.50; n=3) | strong:cavity2d_n2840: 0.579 ± 0.005 (med 0.579; 0.574–0.583; n=3) | strong:cavity2d_n2840: 0.583 ± 0.005 (med 0.582; 0.580–0.589; n=3) |
| 2d_n1440_k4_K4 | F3 | 4 | E | - | weak:cavity2d_n1440: - | weak:cavity2d_n1440: - |
| 2d_n1440_k4_K4 | F3 | 4 | C | 331.04 ± 1.88 (med 331.00; 328.60–333.70; n=5) | weak:cavity2d_n1440: 0.914 ± 0.005 (med 0.915; 0.909–0.921; n=5) | weak:cavity2d_n1440: 0.925 ± 0.006 (med 0.924; 0.917–0.934; n=5) |
| 2d_n1440_k8_K8 | F3 | 8 | E | - | weak:cavity2d_n1440: - | weak:cavity2d_n1440: - |
| 2d_n1440_k8_K8 | F3 | 8 | C | 328.42 ± 0.50 (med 328.20; 327.90–329.00; n=5) | weak:cavity2d_n1440: 0.908 ± 0.002 (med 0.908; 0.905–0.910; n=5) | weak:cavity2d_n1440: 0.917 ± 0.001 (med 0.916; 0.915–0.919; n=5) |

Without the defrag events (user 2026-10-12): fps = steady_frames / (steady_s - the wall time of the events in the window), the same on the K run and on every K = 1 member; eta from these. The defrag event table lists the events themselves.

| point | K | arm | fps without defrag | eta without defrag (type: reference) | eta_min without defrag |
|---|---|---|---|---|---|
| 2d_n4000_K8 | 8 | E | - | - | - |
| 2d_n4000_K8 | 8 | C | 334.44 ± 2.18 (med 335.59; 331.93–335.81; n=3) | strong:cavity2d_n4000: 0.883 ± 0.003 (med 0.884; 0.879–0.886; n=3) | strong:cavity2d_n4000: 0.892 ± 0.003 (med 0.893; 0.888–0.894; n=3) |
| 2d_n1440_k2_K2 | 2 | E | - | - | - |
| 2d_n1440_k2_K2 | 2 | C | 338.29 ± 3.03 (med 338.06; 333.75–341.88; n=5) | weak:cavity2d_n1440: 0.928 ± 0.008 (med 0.928; 0.915–0.936; n=5) | weak:cavity2d_n1440: 0.929 ± 0.008 (med 0.929; 0.915–0.937; n=5) |
| 2d_n1440_K8 | 8 | E | - | - | - |
| 2d_n1440_K8 | 8 | C | 477.52 ± 3.41 (med 478.98; 473.62–479.95; n=3) | strong:cavity2d_n1440: 0.164 ± 0.001 (med 0.165; 0.163–0.165; n=3) | strong:cavity2d_n1440: 0.166 ± 0.001 (med 0.166; 0.165–0.167; n=3) |
| 2d_n2000_K8 | 8 | E | - | - | - |
| 2d_n2000_K8 | 8 | C | 457.82 ± 6.88 (med 456.97; 451.41–465.08; n=3) | strong:cavity2d_n2000: 0.311 ± 0.005 (med 0.310; 0.307–0.316; n=3) | strong:cavity2d_n2000: 0.314 ± 0.005 (med 0.313; 0.310–0.319; n=3) |
| 2d_n2840_K8 | 8 | E | - | - | - |
| 2d_n2840_K8 | 8 | C | 439.60 ± 3.23 (med 440.70; 435.96–442.13; n=3) | strong:cavity2d_n2840: 0.591 ± 0.005 (med 0.591; 0.586–0.596; n=3) | strong:cavity2d_n2840: 0.596 ± 0.005 (med 0.594; 0.591–0.602; n=3) |
| 2d_n1440_k4_K4 | 4 | E | - | - | - |
| 2d_n1440_k4_K4 | 4 | C | 336.05 ± 1.51 (med 335.64; 334.63–338.38; n=5) | weak:cavity2d_n1440: 0.924 ± 0.004 (med 0.923; 0.920–0.930; n=5) | weak:cavity2d_n1440: 0.935 ± 0.005 (med 0.933; 0.930–0.944; n=5) |
| 2d_n1440_k8_K8 | 8 | E | - | - | - |
| 2d_n1440_k8_K8 | 8 | C | 338.14 ± 0.50 (med 338.22; 337.40–338.63; n=5) | weak:cavity2d_n1440: 0.930 ± 0.002 (med 0.931; 0.927–0.932; n=5) | weak:cavity2d_n1440: 0.940 ± 0.002 (med 0.939; 0.938–0.942; n=5) |

### Defrag events (wall time per event: the drain before the defrag, the readbacks, every slab's defrag, up to the next submit)

| point | runs | events per run | wall ms per event (mean; min–max) | drain ms | idle ms | share of the steady window |
|---|---|---|---|---|---|---|
| 2d_n4000_K8 K=1 2d_n4000 | 24 | 1.0 (1.0 in the window) | 73.6; 73.2–74.2 | 21.2 | 52.5 | 0.35 % |
| 2d_n4000_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 69.8; 68.6–71.7 | 1.3 | 68.6 | 2.28 % |
| 2d_n1440_k2_K2 K=1 2d_n1440 | 10 | 1.0 (1.0 in the window) | 13.2; 11.6–20.2 | 2.7 | 10.5 | 0.48 % |
| 2d_n1440_k2_K2 K=2 C | 5 | 1.0 (1.0 in the window) | 20.1; 18.7–21.9 | 2.1 | 18.0 | 0.68 % |
| 2d_n1440_K8 K=1 2d_n1440 | 24 | 1.0 (1.0 in the window) | 12.5; 11.4–13.8 | 2.5 | 10.0 | 0.45 % |
| 2d_n1440_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 42.2; 41.9–42.4 | 1.4 | 40.7 | 1.97 % |
| 2d_n2000_K8 K=1 2d_n2000 | 24 | 1.0 (1.0 in the window) | 21.0; 20.5–23.6 | 5.4 | 15.5 | 0.38 % |
| 2d_n2000_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 28.6; 28.0–29.2 | 1.3 | 27.2 | 1.29 % |
| 2d_n2840_K8 K=1 2d_n2840 | 24 | 1.0 (1.0 in the window) | 38.4; 37.8–39.5 | 10.8 | 27.7 | 0.36 % |
| 2d_n2840_K8 K=8 C | 3 | 1.0 (1.0 in the window) | 55.7; 55.0–56.2 | 1.4 | 54.3 | 2.39 % |
| 2d_n1440_k4_K4 K=1 2d_n1440 | 20 | 1.0 (1.0 in the window) | 13.4; 11.7–21.1 | 2.8 | 10.7 | 0.49 % |
| 2d_n1440_k4_K4 K=4 C | 5 | 1.0 (1.0 in the window) | 44.9; 41.0–54.8 | 1.3 | 43.6 | 1.48 % |
| 2d_n1440_k8_K8 K=1 2d_n1440 | 40 | 1.0 (1.0 in the window) | 12.7; 11.2–18.2 | 2.5 | 10.2 | 0.46 % |
| 2d_n1440_k8_K8 K=8 C | 5 | 1.0 (1.0 in the window) | 87.6; 85.8–91.2 | 1.3 | 86.3 | 2.88 % |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n4000_K8 | 1.0266, 0.9869, 0.9761, 1.0029, 1.0004, 0.9959, 0.9855, 1.0258 | [106, 204, 302, 402, 502, 602, 700] | [103, 203, 303, 403, 503, 603, 703]→[106, 205, 303, 404, 504, 603, 701] changed; [106, 205, 303, 404, 504, 603, 701]→[106, 204, 302, 402, 502, 602, 700] changed | - | - | pass: r_chain 0.603/0.890, t_chain 810/1932; r_own 0.554/0.813, t_own 770/1588; r 0.886/1.074, t_tr 1224/2463; T_B 2283 (max 2463), fps -0.64 % |
| 2d_n1440_k2_K2 | 0.9999, 1.0001 | [291] | [291]→[291] kept | - | - | - |
| 2d_n1440_K8 | 0.9989, 1.0244, 1.0099, 0.9685, 0.9924, 1.0399, 0.9883, 0.9776 | [39, 76, 112, 147, 183, 220, 256] | [39, 75, 111, 147, 183, 219, 255]→[39, 76, 112, 147, 183, 220, 256] changed; [39, 76, 112, 147, 183, 220, 256]→[39, 76, 112, 147, 183, 220, 256] kept | - | - | pass: r_chain 5.473/6.500, t_chain 1619/2256; r_own 5.102/5.672, t_own 1476/1980; r 5.629/6.663, t_tr 1724/2341; T_B 369 (max 388), fps -6.32 % |
| 2d_n2000_K8 | 0.9999, 1.0086, 0.9989, 1.0061, 1.0178, 0.9973, 0.9838, 0.9876 | [53, 103, 153, 204, 255, 304, 354] | [53, 103, 153, 203, 253, 303, 353]→[53, 103, 153, 203, 254, 304, 353] changed; [53, 103, 153, 203, 254, 304, 353]→[53, 103, 153, 204, 255, 304, 354] changed | - | - | pass: r_chain 3.212/3.697, t_chain 1567/2175; r_own 3.025/3.347, t_own 1441/1941; r 3.289/3.833, t_tr 1670/2242; T_B 616 (max 637), fps -1.88 % |
| 2d_n2840_K8 | 1.0646, 0.9934, 0.9701, 0.9751, 0.9871, 0.9872, 0.9790, 1.0435 | [79, 149, 218, 287, 357, 427, 497] | [74, 145, 216, 287, 358, 429, 500]→[78, 149, 218, 288, 358, 428, 497] changed; [78, 149, 218, 288, 358, 428, 497]→[79, 149, 218, 287, 357, 427, 497] changed | - | - | pass: r_chain 1.874/2.262, t_chain 1582/2325; r_own 1.781/2.048, t_own 1464/2133; r 1.954/2.348, t_tr 1729/2470; T_B 1111 (max 1247), fps -6.25 % |
| 2d_n1440_k4_K4 | 0.9964, 1.0090, 0.9931, 1.0014 | [290, 581, 867] | [291, 579, 867]→[288, 581, 869] changed; [288, 581, 869]→[290, 581, 867] changed | - | - | - |
| 2d_n1440_k8_K8 | 0.9999, 1.0092, 0.9876, 1.0035, 1.0064, 0.9952, 1.0011, 0.9972 | [291, 582, 866, 1155, 1445, 1731, 2020] | [291, 579, 867, 1155, 1443, 1731, 2019]→[291, 584, 868, 1156, 1445, 1731, 2019] changed; [291, 584, 868, 1156, 1445, 1731, 2019]→[291, 582, 866, 1155, 1445, 1731, 2020] changed | - | - | - |

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n4000_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 54229, relay 29771, fallback 0, blocking 0 (relay slept 29722, released 0)
- 2d_n1440_k2_K2 C: 5 of 5 runs (5 with V7_DEST_GUARD=relay): precheck 8197, relay 6803, fallback 0, blocking 0 (relay slept 6803, released 0)
- 2d_n1440_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 82689, relay 1311, fallback 0, blocking 0 (relay slept 1287, released 0)
- 2d_n2000_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 81898, relay 2102, fallback 0, blocking 0 (relay slept 2086, released 0)
- 2d_n2840_K8 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 72055, relay 11945, fallback 0, blocking 0 (relay slept 11829, released 0)
- 2d_n1440_k4_K4 C: 5 of 5 runs (5 with V7_DEST_GUARD=relay): precheck 25007, relay 19993, fallback 0, blocking 0 (relay slept 19985, released 0)
- 2d_n1440_k8_K8 C: 5 of 5 runs (5 with V7_DEST_GUARD=relay): precheck 60271, relay 44729, fallback 0, blocking 0 (relay slept 44686, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n4000_K8_t1_R1_2d_n4000 | 8 | ok | 0.3 | 19 | 19 | 187 |
| 2d_n4000_K8_t2_R1_2d_n4000 | 8 | ok | 6.5 | 15 | 15 | 288 |
| 2d_n4000_K8_t3_R1_2d_n4000 | 8 | ok | 0.2 | 14 | 14 | 172 |
| 2d_n1440_k2_K2_t1_R1_2d_n1440 | 2 | ok | 0.1 | 13 | 13 | 15 |
| 2d_n1440_k2_K2_t2_R1_2d_n1440 | 2 | ok | 0.0 | 19 | 19 | 9 |
| 2d_n1440_k2_K2_t3_R1_2d_n1440 | 2 | ok | 0.0 | 8 | 8 | 11 |
| 2d_n1440_k2_K2_t4_R1_2d_n1440 | 2 | ok | 0.0 | 16 | 16 | 19 |
| 2d_n1440_k2_K2_t5_R1_2d_n1440 | 2 | ok | 0.0 | 8 | 8 | 0 |
| 2d_n1440_K8_t1_R1_2d_n1440 | 8 | ok | 3.0 | 16 | 16 | 164 |
| 2d_n1440_K8_t2_R1_2d_n1440 | 8 | ok | 1.9 | 19 | 19 | 50 |
| 2d_n1440_K8_t3_R1_2d_n1440 | 8 | ok | 3.1 | 16 | 16 | 212 |
| 2d_n2000_K8_t1_R1_2d_n2000 | 8 | ok | 4.5 | 17 | 17 | 198 |
| 2d_n2000_K8_t2_R1_2d_n2000 | 8 | ok | 1.2 | 14 | 14 | 50 |
| 2d_n2000_K8_t3_R1_2d_n2000 | 8 | ok | 3.7 | 19 | 19 | 239 |
| 2d_n2840_K8_t1_R1_2d_n2840 | 8 | ok | 0.5 | 19 | 19 | 87 |
| 2d_n2840_K8_t2_R1_2d_n2840 | 8 | ok | 3.5 | 15 | 14 | 133 |
| 2d_n2840_K8_t3_R1_2d_n2840 | 8 | ok | 4.3 | 19 | 19 | 175 |
| 2d_n1440_k4_K4_t1_R1_2d_n1440 | 4 | ok | 0.0 | 18 | 18 | 32 |
| 2d_n1440_k4_K4_t2_R1_2d_n1440 | 4 | ok | 0.0 | 18 | 19 | 16 |
| 2d_n1440_k4_K4_t3_R1_2d_n1440 | 4 | ok | 0.0 | 10 | 10 | 36 |
| 2d_n1440_k4_K4_t4_R1_2d_n1440 | 4 | ok | 0.0 | 18 | 18 | 32 |
| 2d_n1440_k4_K4_t5_R1_2d_n1440 | 4 | ok | 0.0 | 18 | 17 | 31 |
| 2d_n1440_k8_K8_t1_R1_2d_n1440 | 8 | ok | 3.2 | 19 | 19 | 41 |
| 2d_n1440_k8_K8_t2_R1_2d_n1440 | 8 | ok | 2.5 | 12 | 12 | 135 |
| 2d_n1440_k8_K8_t3_R1_2d_n1440 | 8 | ok | 1.3 | 16 | 17 | 26 |
| 2d_n1440_k8_K8_t4_R1_2d_n1440 | 8 | ok | 4.6 | 19 | 19 | 208 |
| 2d_n1440_k8_K8_t5_R1_2d_n1440 | 8 | ok | 1.1 | 19 | 19 | 20 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n4000_K8 | 2813 / 517 | 2747 / 575 | 66 MHz / -58 W |
| 2d_n1440_k2_K2 | 2835 / 553 | 2788 / 575 | 47 MHz / -22 W |
| 2d_n1440_K8 | 2739 / 138 | 2781 / 575 | -42 MHz / -438 W |
| 2d_n2000_K8 | 2866 / 214 | 2765 / 575 | 101 MHz / -361 W |
| 2d_n2840_K8 | 2848 / 343 | 2757 / 575 | 91 MHz / -232 W |
| 2d_n1440_k4_K4 | 2813 / 546 | 2774 / 575 | 39 MHz / -28 W |
| 2d_n1440_k8_K8 | 2817 / 531 | 2776 / 575 | 41 MHz / -44 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n4000_K8 | K runs | 8 | 3 | 0.51 | 5.09 | 5.94 | 11.53 | 8.51 | 10.62 |
| 2d_n4000_K8 | references | 1 | 16 | 0.54 | 2.14 | 3.26 | 5.93 | 1.53 | 11.93 |
| 2d_n1440_k2_K2 | K runs | 2 | 5 | 0.16 | 0.60 | 2.43 | 3.19 | 2.16 | 2.24 |
| 2d_n1440_k2_K2 | references | 1 | 8 | 0.09 | 0.22 | 3.01 | 3.32 | 1.09 | 1.23 |
| 2d_n1440_K8 | K runs | 8 | 3 | 0.09 | 0.48 | 2.76 | 3.34 | 8.09 | 1.29 |
| 2d_n1440_K8 | references | 1 | 16 | 0.11 | 0.24 | 3.51 | 3.86 | 1.11 | 1.33 |
| 2d_n2000_K8 | K runs | 8 | 3 | 0.15 | 0.96 | 4.79 | 5.90 | 8.15 | 2.52 |
| 2d_n2000_K8 | references | 1 | 16 | 0.16 | 0.50 | 3.67 | 4.33 | 1.16 | 2.56 |
| 2d_n2840_K8 | K runs | 8 | 3 | 0.27 | 2.33 | 5.42 | 8.02 | 8.27 | 5.06 |
| 2d_n2840_K8 | references | 1 | 16 | 0.29 | 1.05 | 4.03 | 5.37 | 1.29 | 6.40 |
| 2d_n1440_k4_K4 | K runs | 4 | 5 | 0.28 | 1.69 | 3.48 | 5.45 | 4.28 | 4.93 |
| 2d_n1440_k4_K4 | references | 1 | 16 | 0.10 | 0.25 | 4.20 | 4.55 | 1.10 | 1.48 |
| 2d_n1440_k8_K8 | K runs | 8 | 5 | 0.53 | 5.29 | 4.58 | 10.39 | 8.53 | 10.79 |
| 2d_n1440_k8_K8 | references | 1 | 32 | 0.10 | 0.24 | 3.75 | 4.09 | 1.10 | 1.35 |

## e7_b2T_1695731

- host wqd10nbm13g5, slurm job 1695731, harness a96a2c2c3807, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- provenance at job end: unchanged (checked in the exit cleanup: the job stopped before its end)
- runs tagged rc2: 77, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1695731; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 11; reference hard_failure 1, pass 62; trace_calibrated pass 3
  - **2d_n8000_K7_t3_R1_2d_n8000_g5** (reference): hard_failure rc=1 errors: Traceback (most recent call last),VkError; no final line

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n8000_K3 | F2 | 3 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K3 | F2 | 3 | C | 36.81 ± 0.03 (med 36.83; 36.78–36.83; n=3) | strong:cavity2d_n8000: 0.997 ± 0.001 (med 0.998; 0.995–0.998; n=3) | strong:cavity2d_n8000: 1.011 ± 0.002 (med 1.011; 1.009–1.012; n=3) |
| 2d_n8000_K5 | F2 | 5 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K5 | F2 | 5 | C | 59.75 ± 0.11 (med 59.77; 59.63–59.84; n=3) | strong:cavity2d_n8000: 0.972 ± 0.001 (med 0.972; 0.972–0.974; n=3) | strong:cavity2d_n8000: 0.987 ± 0.001 (med 0.987; 0.986–0.988; n=3) |
| 2d_n8000_K6 | F2 | 6 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K6 | F2 | 6 | C | 70.13 ± 0.05 (med 70.13; 70.08–70.18; n=3) | strong:cavity2d_n8000: 0.966 ± 0.011 (med 0.963; 0.957–0.978; n=3) | strong:cavity2d_n8000: 1.047 ± 0.066 (med 1.024; 0.996–1.122; n=3) **two modes: 2× ~1.010 / 1× ~1.122** |
| 2d_n8000_K7 | F2 | 7 | E | - | strong:cavity2d_n8000: - | strong:cavity2d_n8000: - |
| 2d_n8000_K7 | F2 | 7 | C | 80.78 ± 0.18 (med 80.78; 80.65–80.91; n=2) | strong:cavity2d_n8000: 0.948 ± 0.002 (med 0.948; 0.947–0.949; n=2) | strong:cavity2d_n8000: 0.996 ± 0.001 (med 0.996; 0.995–0.997; n=2) |

Without the defrag events (user 2026-10-12): fps = steady_frames / (steady_s - the wall time of the events in the window), the same on the K run and on every K = 1 member; eta from these. The defrag event table lists the events themselves.

| point | K | arm | fps without defrag | eta without defrag (type: reference) | eta_min without defrag |
|---|---|---|---|---|---|
| 2d_n8000_K3 | 3 | E | - | - | - |
| 2d_n8000_K3 | 3 | C | 37.14 ± 0.03 (med 37.15; 37.11–37.17; n=3) | strong:cavity2d_n8000: 1.002 ± 0.001 (med 1.003; 1.001–1.003; n=3) | strong:cavity2d_n8000: 1.016 ± 0.001 (med 1.017; 1.014–1.017; n=3) |
| 2d_n8000_K5 | 5 | E | - | - | - |
| 2d_n8000_K5 | 5 | C | 60.60 ± 0.11 (med 60.63; 60.48–60.69; n=3) | strong:cavity2d_n8000: 0.983 ± 0.001 (med 0.982; 0.982–0.984; n=3) | strong:cavity2d_n8000: 0.997 ± 0.001 (med 0.997; 0.996–0.999; n=3) |
| 2d_n8000_K6 | 6 | E | - | - | - |
| 2d_n8000_K6 | 6 | C | 71.33 ± 0.05 (med 71.34; 71.27–71.37; n=3) | strong:cavity2d_n8000: 0.979 ± 0.010 (med 0.976; 0.970–0.991; n=3) | strong:cavity2d_n8000: 1.062 ± 0.067 (med 1.038; 1.010–1.137; n=3) **two modes: 2× ~1.024 / 1× ~1.137** |
| 2d_n8000_K7 | 7 | E | - | - | - |
| 2d_n8000_K7 | 7 | C | 82.32 ± 0.16 (med 82.32; 82.20–82.43; n=2) | strong:cavity2d_n8000: 0.963 ± 0.002 (med 0.963; 0.962–0.964; n=2) | strong:cavity2d_n8000: 1.012 ± 0.001 (med 1.012; 1.011–1.012; n=2) |

### Defrag events (wall time per event: the drain before the defrag, the readbacks, every slab's defrag, up to the next submit)

| point | runs | events per run | wall ms per event (mean; min–max) | drain ms | idle ms | share of the steady window |
|---|---|---|---|---|---|---|
| 2d_n8000_K3 K=1 2d_n8000 | 9 | 1.0 (1.0 in the window) | 281.9; 280.3–285.4 | 81.0 | 200.9 | 0.35 % |
| 2d_n8000_K3 K=3 C | 3 | 1.0 (1.0 in the window) | 243.7; 235.6–250.6 | 26.9 | 216.8 | 0.90 % |
| 2d_n8000_K5 K=1 2d_n8000 | 15 | 1.0 (1.0 in the window) | 285.6; 280.5–305.4 | 81.2 | 204.4 | 0.35 % |
| 2d_n8000_K5 K=5 C | 3 | 1.0 (1.0 in the window) | 233.3; 232.1–234.8 | 16.5 | 216.8 | 1.39 % |
| 2d_n8000_K6 K=1 2d_n8000 | 18 | 1.0 (1.0 in the window) | 283.4; 280.8–289.3 | 82.1 | 201.3 | 0.34 % |
| 2d_n8000_K6 K=6 C | 3 | 1.0 (1.0 in the window) | 238.8; 234.5–243.8 | 14.0 | 224.8 | 1.67 % |
| 2d_n8000_K7 K=1 2d_n8000 | 20 | 1.0 (1.0 in the window) | 282.5; 280.6–285.7 | 81.7 | 200.9 | 0.34 % |
| 2d_n8000_K7 K=7 C | 2 | 1.0 (1.0 in the window) | 231.5; 231.2–231.8 | 8.8 | 222.7 | 1.87 % |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n8000_K3 | 1.0096, 0.9872, 1.0032 | [541, 1068] | [536, 1070]→[542, 1067] changed; [542, 1067]→[541, 1068] changed | - | - | pass: r_chain 0.032/0.320, t_chain 654/7192; r_own 0.030/0.319, t_own 633/7174; r 0.739/1.005, t_tr 3438/24126; T_B 23954 (max 24018), fps -0.06 % |
| 2d_n8000_K5 | 1.0112, 0.9821, 1.0053, 0.9941, 1.0073 | [327, 641, 963, 1281] | [323, 643, 963, 1283]→[328, 641, 961, 1279] changed; [328, 641, 961, 1279]→[327, 641, 963, 1281] changed | - | - | pass: r_chain 0.508/0.848, t_chain 723/11933; r_own 0.059/0.526, t_own 701/5119; r 1.169/1.299, t_tr 3940/18692; T_B 14433 (max 14506), fps -0.50 % |
| 2d_n8000_K6 | 1.0202, 0.9928, 0.9991, 1.0026, 0.9986, 0.9868 | [275, 540, 806, 1074, 1340] | [270, 536, 803, 1070, 1336]→[275, 538, 806, 1072, 1340] changed; [275, 538, 806, 1072, 1340]→[275, 540, 806, 1074, 1340] changed | - | - | pass: r_chain 0.932/1.039, t_chain 717/12368; r_own 0.087/0.610, t_own 651/4322; r 1.180/1.318, t_tr 1344/15403; T_B 12136 (max 12320), fps -0.42 % |
| 2d_n8000_K7 | 1.0173, 0.9926, 1.0016, 1.0050, 0.9953, 0.9842, 1.0039 | [236, 462, 691, 921, 1149, 1374] | [232, 460, 689, 917, 1146, 1374]→[236, 462, 692, 921, 1150, 1373] changed; [236, 462, 692, 921, 1150, 1373]→[236, 462, 691, 921, 1149, 1374] changed | - | - | - |

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n8000_K3 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 12091, relay 11909, fallback 0, blocking 0 (relay slept 11909, released 0)
- 2d_n8000_K5 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 24360, relay 23640, fallback 0, blocking 0 (relay slept 23640, released 0)
- 2d_n8000_K6 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 33562, relay 26438, fallback 0, blocking 0 (relay slept 26432, released 0)
- 2d_n8000_K7 C: 2 of 2 runs (2 with V7_DEST_GUARD=relay): precheck 21692, relay 14308, fallback 0, blocking 0 (relay slept 14305, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n8000_K3_t1_R1_2d_n8000 | 3 | ok | 0.2 | 14 | 14 | 720 |
| 2d_n8000_K3_t2_R1_2d_n8000 | 3 | ok | 0.0 | 18 | 18 | 812 |
| 2d_n8000_K3_t3_R1_2d_n8000 | 3 | ok | 0.0 | 20 | 20 | 821 |
| 2d_n8000_K5_t1_R1_2d_n8000 | 5 | ok | 0.2 | 9 | 9 | 709 |
| 2d_n8000_K5_t2_R1_2d_n8000 | 5 | ok | 0.2 | 19 | 19 | 789 |
| 2d_n8000_K5_t3_R1_2d_n8000 | 5 | ok | 0.1 | 13 | 13 | 913 |
| 2d_n8000_K6_t1_R1_2d_n8000 | 6 | ok | 0.2 | 19 | 19 | 1431 |
| 2d_n8000_K6_t2_R1_2d_n8000 | 6 | ok | 0.2 | 15 | 15 | 1556 |
| 2d_n8000_K6_t3_R1_2d_n8000 | 6 | ok | 0.2 | 20 | 20 | 1697 |
| 2d_n8000_K7_t1_R1_2d_n8000 | 7 | ok | 0.3 | 18 | 18 | 1583 |
| 2d_n8000_K7_t2_R1_2d_n8000 | 7 | ok | 0.1 | 18 | 18 | 1617 |
| 2d_n8000_K7_t3_R1_2d_n8000 | 7 | ok | 0.2 | 19 | 19 | 858 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n8000_K3 | 2799 / 600 | 2771 / 600 | 28 MHz / 0 W |
| 2d_n8000_K5 | 2816 / 600 | 2766 / 600 | 49 MHz / -0 W |
| 2d_n8000_K6 | 2811 / 600 | 2748 / 596 | 63 MHz / 4 W |
| 2d_n8000_K7 | 2814 / 600 | 2747 / 596 | 67 MHz / 4 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n8000_K3 | K runs | 3 | 3 | 1.79 | 11.21 | 2.91 | 15.91 | 4.79 | 24.27 |
| 2d_n8000_K3 | references | 1 | 6 | 1.80 | 6.88 | 3.48 | 12.16 | 2.80 | 26.14 |
| 2d_n8000_K5 | K runs | 5 | 3 | 1.80 | 14.10 | 3.92 | 19.82 | 6.80 | 25.36 |
| 2d_n8000_K5 | references | 1 | 10 | 1.78 | 6.82 | 4.62 | 13.22 | 2.78 | 29.04 |
| 2d_n8000_K6 | K runs | 6 | 3 | 1.83 | 15.60 | 3.89 | 21.32 | 7.83 | 25.56 |
| 2d_n8000_K6 | references | 1 | 12 | 1.85 | 6.85 | 4.35 | 13.04 | 2.85 | 30.78 |
| 2d_n8000_K7 | K runs | 7 | 2 | 1.82 | 17.09 | 4.54 | 23.44 | 8.82 | 26.64 |
| 2d_n8000_K7 | references | 1 | 14 | 1.89 | 6.88 | 4.85 | 13.62 | 2.89 | 32.10 |

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| b2A | 1694327 | wqd10nbj04g2 | 2026-10-13T02:49:07 | 2026-10-13T04:19:28 | COMPLETED | 1.51 | 1.66 |
| b2B | 1694328 | wqd10nbj04g2 | 2026-10-13T04:19:28 | 2026-10-13T05:56:18 | COMPLETED | 1.61 | 1.74 |
| b2S | 1695411 | wqd10nbj04g2 | 2026-10-13T07:21:49 | 2026-10-13T07:49:23 | COMPLETED | 0.46 | - |
| b2T | 1695731 | wqd10nbm13g5 | 2026-10-13T10:26:43 | 2026-10-13T11:29:23 | FAILED | 1.04 | - |
