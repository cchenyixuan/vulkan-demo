Model: cost 6.69 ns per particle-step (K = 1) / table 175616:6.83,8000000:6.28,32000000:5.91, B_link = 256 n^2 + 4500 n^1, t_chain = 122 us + B_link / 8.38 GB/s, eta_strong {2: 0.95, 4: 0.9, 8: 0.875}, eta_weak 0.985, B share 1.0; windows 1500/500, 3 trials, 2 pilots x 500 steps.

## Step model per point

| point | particles | per card | own columns (interior) | step K = 1 ref (ms) | compute / card (ms) | eta assumed | B_link (MB) | t_chain (ms) | B share | T_B (ms) | r = t_chain / T_B | exposed (ms) | step K (ms) | fps K | eta model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
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
| cavity3d_n192 K8 | B | 54 s | 1.5 min | 1.5 min x 2 | 34 s x 3 | 1.0 min | **8.0 min** | 900 / 900 / 900 |
| cavity3d_n244 K8 | B | 1.3 min | 2.8 min | 2.8 min x 2 | 53 s x 3 | 1.4 min | **13.7 min** | 900 / 960 / 900 |
| cavity3d_n312 K8 | B | 2.1 min | 5.5 min | 5.4 min x 2 | 1.5 min x 3 | - | **23.0 min** | 900 / 1260 / - |
| cavity3d_n392 K8 | B | 3.6 min | 10.6 min | 10.6 min x 2 | 2.7 min x 3 | 3.8 min | **47.3 min** | 1080 / 1920 / 1080 |
| cavity3d_n392 K2 | A | 5.5 min | 10.6 min | 10.6 min x 2 | 6.1 min x 3 | - | **55.5 min** | 1260 / 1920 / - |
| cavity3d_n392 K4 | A | 4.1 min | 10.6 min | 10.6 min x 2 | 3.9 min x 3 | - | **47.5 min** | 1140 / 1920 / - |
| cavity3d_n192_k2 K2 | A | 1.4 min | 1.5 min | 1.5 min x 2 | 1.6 min x 3 | - | **10.6 min** | 900 / 900 / - |
| cavity3d_n192_k4 K4 | A | 2.1 min | 1.5 min | 1.5 min x 2 | 1.9 min x 3 | - | **12.1 min** | 900 / 900 / - |
| cavity3d_n192_k8 K8 | B | 3.3 min | 1.5 min | 1.5 min x 2 | 2.5 min x 3 | 3.6 min | **18.7 min** | 1020 / 900 / 1080 |

Line A (K <= 4, cards 0-3 of a whole node): 2.14 h; line B (K = 8): 1.89 h; both on whole nodes: **4.03 node-h** (one node back to back: the same). Time limits (2 x + 1 h): A 6 h, B 5 h.
Reference runs alone: 2.29 h of the 3.94 h (the 64M K = 1 references of the three n392 points are 1.59 h).
