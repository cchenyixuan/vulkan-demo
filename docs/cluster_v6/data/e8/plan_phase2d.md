Model: cost 6.69 ns per particle-step (K = 1) / table 175616:6.83,8000000:6.28,32000000:5.91, B_link = 256 n^2 + 4500 n^1, t_chain = 122 us + B_link / 8.38 GB/s, eta_strong {2: 0.95, 4: 0.9, 8: 0.875}, eta_weak 0.985, B share 1.0; windows 1500/500, 3 trials, 2 pilots x 500 steps.

## Step model per point

| point | particles | per card | own columns (interior) | step K = 1 ref (ms) | compute / card (ms) | eta assumed | B_link (MB) | t_chain (ms) | B share | T_B (ms) | r = t_chain / T_B | exposed (ms) | step K (ms) | fps K | eta model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity3d_n152 K8 (strong, load sweep) | 4,096,000 | 0.51M (6.68 ns) | 7/4/5/5/5/4/5/7 (0.00) | 26.1 | 3.42 | 0.875 | 6.6 | 0.91 | 0.00 | 0.00 | 99.00 | 0.91 | 4.82 | 207.7 | 0.710 |
| cavity3d_n192 K8 (strong, load sweep) | 8,000,000 | 1.00M (6.58 ns) | 8/6/6/6/6/6/6/8 (0.00) | 50.2 | 6.58 | 0.875 | 10.3 | 1.35 | 0.27 | 1.77 | 0.77 | 0.00 | 7.52 | 133.0 | 0.875 |
| cavity3d_n244 K8 (strong, load sweep) | 16,003,008 | 2.00M (6.48 ns) | 10/7/8/7/8/8/7/10 (0.14) | 97.5 | 12.96 | 0.875 | 16.3 | 2.07 | 0.41 | 5.31 | 0.39 | 0.00 | 14.81 | 67.5 | 0.875 |
| cavity3d_n312 K8 (strong, load sweep) | 32,768,000 | 4.10M (6.38 ns) | 12/9/10/10/10/9/10/12 (0.33) | 193.7 | 26.12 | 0.875 | 26.3 | 3.26 | 0.60 | 15.60 | 0.21 | 0.00 | 29.85 | 33.5 | 0.875 |
| cavity3d_n392 K8 (strong, load sweep) | 64,000,000 | 8.00M (6.28 ns) | 14/12/13/12/12/12/13/14 (0.50) | 378.2 | 50.24 | 0.875 | 41.1 | 5.03 | 0.76 | 38.28 | 0.13 | 0.00 | 57.42 | 17.4 | 0.875 |
| cavity3d_n392 K2 (strong, K sweep) | 64,000,000 | 32.00M (5.91 ns) | 51/51 (0.88) | 378.2 | 189.12 | 0.950 | 41.1 | 5.03 | 0.88 | 166.43 | 0.03 | 0.00 | 199.07 | 5.0 | 0.950 |
| cavity3d_n392 K4 (strong, K sweep) | 64,000,000 | 16.00M (6.10 ns) | 26/25/24/27 (0.75) | 378.2 | 97.52 | 0.900 | 41.1 | 5.03 | 0.88 | 85.82 | 0.06 | 0.00 | 108.36 | 9.2 | 0.900 |
| cavity3d_n192_k2 K2 (weak) | 15,680,000 | 7.84M (6.28 ns) | 50/50 (0.88) | 50.2 | 49.26 | 0.985 | 10.3 | 1.35 | 0.88 | 43.35 | 0.03 | 0.00 | 50.01 | 20.0 | 0.985 |
| cavity3d_n192_k4 K4 (weak) | 31,040,000 | 7.76M (6.28 ns) | 50/48/48/50 (0.88) | 50.2 | 48.77 | 0.985 | 10.3 | 1.35 | 0.88 | 42.91 | 0.03 | 0.00 | 49.51 | 20.2 | 0.985 |
| cavity3d_n192_k8 K8 (weak) | 61,760,000 | 7.72M (6.29 ns) | 50/48/48/48/48/48/48/50 (0.88) | 50.2 | 48.52 | 0.985 | 10.3 | 1.35 | 0.88 | 42.70 | 0.03 | 0.00 | 49.26 | 20.3 | 0.985 |

## Wall clock per point (E7 protocol)

| point | line | calibration (2 pilots) | reference set, trial 1 (cold) | reference set, trials 2-3 | chain run (per trial) | trace | point total | timeouts cal / run / trace (s) |
|---|---|---|---|---|---|---|---|---|
| cavity3d_n152 K8 | B | 44 s | 48 s | 48 s x 2 | 26 s x 3 | 50 s | **5.2 min** | 900 / 900 / 900 |
| cavity3d_n192 K8 | B | 54 s | 1.5 min | 1.5 min x 2 | 34 s x 3 | 1.0 min | **8.0 min** | 900 / 900 / 900 |
| cavity3d_n244 K8 | B | 1.3 min | 2.8 min | 2.8 min x 2 | 53 s x 3 | 1.4 min | **13.7 min** | 900 / 960 / 900 |
| cavity3d_n312 K8 | B | 2.1 min | 5.5 min | 5.4 min x 2 | 1.5 min x 3 | - | **23.0 min** | 900 / 1260 / - |
| cavity3d_n392 K8 | B | 3.6 min | 10.6 min | 10.6 min x 2 | 2.7 min x 3 | 3.8 min | **47.3 min** | 1080 / 1920 / 1080 |
| cavity3d_n392 K2 | A | 5.5 min | 10.6 min | 10.6 min x 2 | 6.1 min x 3 | - | **55.5 min** | 1260 / 1920 / - |
| cavity3d_n392 K4 | A | 4.1 min | 10.6 min | 10.6 min x 2 | 3.9 min x 3 | - | **47.5 min** | 1140 / 1920 / - |
| cavity3d_n192_k2 K2 | A | 1.4 min | 1.5 min | 1.5 min x 2 | 1.6 min x 3 | - | **10.6 min** | 900 / 900 / - |
| cavity3d_n192_k4 K4 | A | 2.1 min | 1.5 min | 1.5 min x 2 | 1.9 min x 3 | - | **12.1 min** | 900 / 900 / - |
| cavity3d_n192_k8 K8 | B | 3.3 min | 1.5 min | 1.5 min x 2 | 2.5 min x 3 | 3.6 min | **18.7 min** | 1020 / 900 / 1080 |

Line A (K <= 4, cards 0-3 of a whole node): 2.14 h; line B (K = 8): 1.98 h; both on whole nodes: **4.12 node-h** (one node back to back: the same). Time limits (2 x + 1 h): A 6 h, B 5 h.
Reference runs alone: 2.33 h of the 4.03 h (the 64M K = 1 references of the three n392 points are 1.59 h).

## Phase 2 (jobs e8_A / e8_B / e8_C / e8_D, the items in the job order)

| line | item | case | K | trials | estimate | note |
|---|---|---|---|---|---|---|
| B | point | cavity3d_n152 | 8 | 5 | **7.9 min** | calibration 44 s + record 14 s, reference set (K = 1) 48 s / 48 s x 4, chain run 26 s x 5, trace 50 s |
| B | dump | cavity3d_n152 | 1 | - | **1.2 min** | dump_state K = 1 to step 1500, equal weights, npz 0.5 GB |
| B | dump | cavity3d_n152 | 1 | - | **1.2 min** | dump_state K = 1 to step 1500, equal weights, npz 0.5 GB |
| B | dump | cavity3d_n152 | 8 | - | **53 s** | dump_state K = 8 to step 1500, equal weights, npz 0.5 GB |
| B | compare | cavity3d_n152 | - | - | **2.4 min** | K = 8 vs K = 1 a + the noise floor K = 1 a vs b (fluid, by seam distance, per slab) |
| B | trace_k1 | cavity3d_n152 | 1 | - | **1.1 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |
| B | point | cavity3d_n192 | 8 | 5 | **12.3 min** | calibration 54 s + record 17 s, reference set (K = 1) 1.5 min / 1.5 min x 4, chain run 34 s x 5, trace 1.0 min |
| B | dump | cavity3d_n192 | 1 | - | **1.8 min** | dump_state K = 1 to step 1500, equal weights, npz 1.0 GB |
| B | dump | cavity3d_n192 | 1 | - | **1.8 min** | dump_state K = 1 to step 1500, equal weights, npz 1.0 GB |
| B | dump | cavity3d_n192 | 8 | - | **1.0 min** | dump_state K = 8 to step 1500, equal weights, npz 1.0 GB |
| B | compare | cavity3d_n192 | - | - | **3.7 min** | K = 8 vs K = 1 a + the noise floor K = 1 a vs b (fluid, by seam distance, per slab) |
| B | trace_k1 | cavity3d_n192 | 1 | - | **1.7 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |
| B | point | cavity3d_n244 | 8 | 3 | **14.1 min** | calibration 1.3 min + record 24 s, reference set (K = 1) 2.8 min / 2.8 min x 2, chain run 53 s x 3, trace 1.4 min |
| B | trace_k1 | cavity3d_n244 | 1 | - | **3.0 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |
| B | point | cavity3d_n312 | 8 | 3 | **25.9 min** | calibration 2.1 min + record 39 s, reference set (K = 1) 5.5 min / 5.4 min x 2, chain run 1.5 min x 3, trace 2.3 min |
| B | trace_k1 | cavity3d_n312 | 1 | - | **5.7 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |
| B | point | cavity3d_n392 | 8 | 3 | **48.4 min** | calibration 3.6 min + record 67 s, reference set (K = 1) 10.6 min / 10.6 min x 2, chain run 2.7 min x 3, trace 3.8 min |
| B | trace_k1 | cavity3d_n392 | 1 | - | **10.9 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |
| B | point | cavity3d_n192_k8 | 8 | 3 | **19.8 min** | calibration 3.3 min + record 65 s, reference set (K = 1) 1.5 min / 1.5 min x 2, chain run 2.5 min x 3, trace 3.6 min |
| B | single_ratio | cavity3d_n192 | 1 | - | **10.3 min** | 3 interleaved pairs v7_single / chain K = 1 on card 0, 1500 steps (warmup 500), no defrag |
| A | point | cavity3d_n392 | 2 | 3 | **63.2 min** | calibration 5.5 min + record 67 s, reference set (K = 1) 10.6 min / 10.6 min x 2, chain run 6.1 min x 3, trace 6.5 min |
| A | point | cavity3d_n392 | 4 | 3 | **53.2 min** | calibration 4.1 min + record 67 s, reference set (K = 1) 10.6 min / 10.6 min x 2, chain run 3.9 min x 3, trace 4.5 min |
| A | point | cavity3d_n192_k2 | 2 | 3 | **12.9 min** | calibration 1.4 min + record 24 s, reference set (K = 1) 1.5 min / 1.5 min x 2, chain run 1.6 min x 3, trace 1.9 min |
| A | point | cavity3d_n192_k4 | 4 | 3 | **15.1 min** | calibration 2.1 min + record 38 s, reference set (K = 1) 1.5 min / 1.5 min x 2, chain run 1.9 min x 3, trace 2.4 min |
| C | point | cavity3d_n496 | 8 | 3 | **67.3 min** | calibration 6.5 min + record 123 s, reference set (4 K = 2 pairs) 12.2 min / 12.1 min x 2, chain run 5.1 min x 3, trace 7.0 min |
| C | trace_k2 | cavity3d_n496 | 2 | - | **12.7 min** | K = 2 pair (cards 0, 1, equal weights) step trace, every kernel's tick (the ideal-chain row = a quarter of its phases) |
| D | point | cavity3d_n416 | 8 | 3 | **57.1 min** | calibration 4.1 min + record 78 s, reference set (K = 1) 12.6 min / 12.6 min x 2, chain run 3.2 min x 3, trace 4.4 min |
| D | trace_k1 | cavity3d_n416 | 1 | - | **12.9 min** | K = 1 step trace, every kernel's tick (the ideal-chain row of tab:phases) |

Job overhead 166 s per job. **line A 2.45 h, line B 2.96 h, line C 1.38 h, line D 1.21 h, total 8.01 node-h**; time limits (2 x + 1 h, rounded up): A 6 h, B 7 h, C 4 h, D 4 h.
