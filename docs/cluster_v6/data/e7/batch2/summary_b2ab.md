## e7_b2AB_1718639

- host wqd10nbj11g2, slurm job 1718639, harness 720e83fe657a, driver 580.82.07
- job deployment (solver rc2): root /data/run01/scxm138/vulkan-demo-v7rc2, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected, commit c56d07863dd3
- provenance at job end: unchanged
- runs tagged rc2: 38, experiment/v7 tree a4b3a8318a83 (v7-rc2) = expected_tree; root /data/run01/scxm138/vulkan-demo-v7rc2
- shader cache (after_bringup): CONFIRMED, 2 files, 3 MiB in /dev/shm/scxm138/glcache_1718639; new files in ~/.cache/nvidia: 0
- shader cache (job_end): CONFIRMED, 10 files, 27 MiB in /dev/shm/scxm138/glcache_1718639; new files in ~/.cache/nvidia: 0
- runs: calibrated pass 6; reference pass 30; trace_calibrated pass 2

### Points: fps and efficiency

| point | family | K | arm | fps | eta (type: reference) | eta_min |
|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5 | F3 | 5 | E | - | weak:cavity2d_n2840: - | weak:cavity2d_n2840: - |
| 2d_n2840_k5_K5 | F3 | 5 | C | 91.16 ± 0.08 (med 91.16; 91.07–91.24; n=3) | weak:cavity2d_n2840: 0.990 ± 0.002 (med 0.991; 0.988–0.992; n=3) | weak:cavity2d_n2840: 1.002 ± 0.002 (med 1.004; 1.000–1.004; n=3) |
| old_2d_n2840_k5_K5 | F3 | 5 | E | - | weak:cavity2d_n2840: - | weak:cavity2d_n2840: - |
| old_2d_n2840_k5_K5 | F3 | 5 | C | 89.07 ± 0.12 (med 89.05; 88.97–89.21; n=3) | weak:cavity2d_n2840: 0.968 ± 0.003 (med 0.969; 0.964–0.970; n=3) | weak:cavity2d_n2840: 0.981 ± 0.003 (med 0.982; 0.977–0.983; n=3) |

Without the defrag events (user 2026-10-12): fps = steady_frames / (steady_s - the wall time of the events in the window), the same on the K run and on every K = 1 member; eta from these. The defrag event table lists the events themselves.

| point | K | arm | fps without defrag | eta without defrag (type: reference) | eta_min without defrag |
|---|---|---|---|---|---|
| 2d_n2840_k5_K5 | 5 | E | - | - | - |
| 2d_n2840_k5_K5 | 5 | C | 92.46 ± 0.05 (med 92.46; 92.42–92.51; n=3) | weak:cavity2d_n2840: 1.001 ± 0.003 (med 1.001; 0.998–1.003; n=3) | weak:cavity2d_n2840: 1.013 ± 0.003 (med 1.014; 1.010–1.015; n=3) |
| old_2d_n2840_k5_K5 | 5 | E | - | - | - |
| old_2d_n2840_k5_K5 | 5 | C | 90.33 ± 0.13 (med 90.32; 90.21–90.47; n=3) | weak:cavity2d_n2840: 0.978 ± 0.003 (med 0.979; 0.975–0.980; n=3) | weak:cavity2d_n2840: 0.991 ± 0.003 (med 0.992; 0.987–0.993; n=3) |

### Defrag events (wall time per event: the drain before the defrag, the readbacks, every slab's defrag, up to the next submit)

| point | runs | events per run | wall ms per event (mean; min–max) | drain ms | idle ms | share of the steady window |
|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5 K=1 2d_n2840 | 15 | 1.0 (1.0 in the window) | 39.1; 38.7–39.7 | 10.8 | 28.3 | 0.36 % |
| 2d_n2840_k5_K5 K=5 C | 3 | 1.0 (1.0 in the window) | 154.8; 154.4–155.4 | 10.8 | 144.0 | 1.41 % |
| old_2d_n2840_k5_K5 K=1 2d_n2840 | 15 | 1.0 (1.0 in the window) | 39.0; 38.2–39.5 | 10.8 | 28.2 | 0.36 % |
| old_2d_n2840_k5_K5 K=5 C | 3 | 1.0 (1.0 in the window) | 157.8; 157.1–158.4 | 11.1 | 146.7 | 1.41 % |

### Points: calibration, arm difference, traces

r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's two waits); r_own = t_own / T_B also leaves out the transfer queues' idle waits (t_own: t_chain minus the readback's wait for the previous frame's consumed-ack and the upload's wait behind an earlier upload that waited: the chain's own work); r = t_tr / T_B contains the receiver's waits and with them the late wake-ups. All per step on the worse link; t_chain, t_own, t_tr: median of the links' p50 / largest p95.

| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain p50/p95, t_chain p50/p95 µs; r_own p50/p95, t_own p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | trace C: same |
|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5 | 1.0033, 1.0071, 1.0050, 0.9850, 0.9996 | [573, 1145, 1716, 2275] | [571, 1139, 1707, 2275]→[574, 1143, 1714, 2272] changed; [574, 1143, 1714, 2272]→[573, 1145, 1716, 2275] changed | - | - | pass: r_chain 0.633/1.021, t_chain 671/9669; r_own 0.081/0.117, t_own 577/1050; r 1.024/1.222, t_tr 2378/11570; T_B 9513 (max 9590), fps -0.45 % |
| old_2d_n2840_k5_K5 | 1.0010, 0.9937, 0.9800, 0.9900, 1.0354 | [572, 1136, 1693, 2255] | [571, 1139, 1707, 2275]→[577, 1144, 1703, 2270] changed; [577, 1144, 1703, 2270]→[572, 1136, 1693, 2255] changed | - | - | pass: r_chain 0.102/0.122, t_chain 768/1082; r_own 0.099/0.120, t_own 747/1058; r 1.023/1.066, t_tr 852/10528; T_B 9469 (max 9883), fps -0.30 % |

### Dest guards (E7 B2, rc2's transport)

Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest guards should not sleep in the driver.

- 2d_n2840_k5_K5 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 26640, relay 21360, fallback 0, blocking 0 (relay slept 21359, released 0)
- old_2d_n2840_k5_K5 C: 4 of 4 runs (4 with V7_DEST_GUARD=relay): precheck 35283, relay 12717, fallback 0, blocking 0 (relay slept 12713, released 0)

### Start barrier (simultaneous groups)

| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | steady start spread (ms) |
|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5_t1_R1_2d_n2840 | 5 | ok | 4.4 | 17 | 17 | 166 |
| 2d_n2840_k5_K5_t2_R1_2d_n2840 | 5 | ok | 3.0 | 19 | 19 | 86 |
| 2d_n2840_k5_K5_t3_R1_2d_n2840 | 5 | ok | 3.5 | 18 | 18 | 192 |
| old_2d_n2840_k5_K5_t1_R1_2d_n2840 | 5 | ok | 3.9 | 18 | 18 | 257 |
| old_2d_n2840_k5_K5_t2_R1_2d_n2840 | 5 | ok | 2.1 | 19 | 19 | 82 |
| old_2d_n2840_k5_K5_t3_R1_2d_n2840 | 5 | ok | 3.3 | 6 | 6 | 84 |

### Clocks and power (loop window medians, mean over GPUs)

| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |
|---|---|---|---|
| 2d_n2840_k5_K5 | 2760 / 575 | 2741 / 575 | 19 MHz / -0 W |
| old_2d_n2840_k5_K5 | 2782 / 575 | 2741 / 575 | 41 MHz / -0 W |

### Construction of warm runs (mean over the runs; s since the bench started)

| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | read-in + 1 s × K | bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2d_n2840_k5_K5 | K runs | 5 | 3 | 1.06 | 8.91 | 3.36 | 13.32 | 6.06 | 17.25 |
| 2d_n2840_k5_K5 | references | 1 | 10 | 0.24 | 0.82 | 2.76 | 3.82 | 1.24 | 3.09 |
| old_2d_n2840_k5_K5 | K runs | 5 | 3 | 1.13 | 8.98 | 3.64 | 13.75 | 6.13 | 16.74 |
| old_2d_n2840_k5_K5 | references | 1 | 10 | 0.25 | 0.82 | 2.31 | 3.38 | 1.25 | 3.00 |

## Node-hours

| job | slurm job | node | start | end | state | used (h) | planned (h) |
|---|---|---|---|---|---|---|---|
| b2AB | 1718639 | wqd10nbj11g2 | 2026-10-20T19:51:19 | 2026-10-20T20:05:09 | COMPLETED | 0.23 | - |
