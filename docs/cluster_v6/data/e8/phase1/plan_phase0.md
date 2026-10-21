Model: cost 6.69 ns per particle-step (K = 1), t_chain = 122 us + B_link / 3.76 GB/s, eta_strong {2: 0.95, 4: 0.9, 8: 0.875}, eta_weak 0.985, B share 1.0; windows 1500/500, 3 trials, 2 pilots x 500 steps.

## Step model per point

| point | particles | per card | own columns (interior) | step K = 1 ref (ms) | compute / card (ms) | eta assumed | B_link (MB) | t_chain (ms) | B share | T_B (ms) | r = t_chain / T_B | exposed (ms) | step K (ms) | fps K | eta model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity3d_n192 K8 (strong, load sweep) | 8,000,000 | 1.00M | 8/6/6/6/6/6/6/8 (0.00) | 53.5 | 6.69 | 0.875 | 10.5 | 2.92 | 0.27 | 1.79 | 1.62 | 1.12 | 8.77 | 114.1 | 0.763 |
| cavity3d_n244 K8 (strong, load sweep) | 16,003,008 | 2.00M | 10/7/8/7/8/8/7/10 (0.14) | 107.1 | 13.38 | 0.875 | 16.6 | 4.54 | 0.41 | 5.48 | 0.83 | 0.00 | 15.29 | 65.4 | 0.875 |
| cavity3d_n312 K8 (strong, load sweep) | 32,768,000 | 4.10M | 12/9/10/10/10/9/10/12 (0.33) | 219.2 | 27.40 | 0.875 | 26.7 | 7.22 | 0.60 | 16.37 | 0.44 | 0.00 | 31.32 | 31.9 | 0.875 |
| cavity3d_n392 K8 (strong, load sweep) | 64,000,000 | 8.00M | 14/12/13/12/12/12/13/14 (0.50) | 428.2 | 53.52 | 0.875 | 41.6 | 11.18 | 0.76 | 40.78 | 0.27 | 0.00 | 61.17 | 16.3 | 0.875 |
| cavity3d_n392 K2 (strong, K sweep) | 64,000,000 | 32.00M | 51/51 (0.88) | 428.2 | 214.08 | 0.950 | 41.6 | 11.18 | 0.88 | 188.39 | 0.06 | 0.00 | 225.35 | 4.4 | 0.950 |
| cavity3d_n392 K4 (strong, K sweep) | 64,000,000 | 16.00M | 26/25/24/27 (0.75) | 428.2 | 107.04 | 0.900 | 41.6 | 11.18 | 0.88 | 94.20 | 0.12 | 0.00 | 118.93 | 8.4 | 0.900 |
| cavity3d_n192_k2 K2 (weak) | 15,680,000 | 7.84M | 50/50 (0.88) | 53.5 | 52.45 | 0.985 | 10.5 | 2.92 | 0.88 | 46.16 | 0.06 | 0.00 | 53.25 | 18.8 | 0.985 |
| cavity3d_n192_k4 K4 (weak) | 31,040,000 | 7.76M | 50/48/48/50 (0.88) | 53.5 | 51.91 | 0.985 | 10.5 | 2.92 | 0.88 | 45.68 | 0.06 | 0.00 | 52.70 | 19.0 | 0.985 |
| cavity3d_n192_k8 K8 (weak) | 61,760,000 | 7.72M | 50/48/48/48/48/48/48/50 (0.88) | 53.5 | 51.65 | 0.985 | 10.5 | 2.92 | 0.88 | 45.45 | 0.06 | 0.00 | 52.43 | 19.1 | 0.985 |

## Wall clock per point (E7 protocol)

| point | line | calibration (2 pilots) | reference set, trial 1 (cold) | reference set, trials 2-3 | chain run (per trial) | trace | point total | timeouts cal / run / trace (s) |
|---|---|---|---|---|---|---|---|---|
| cavity3d_n192 K8 | B | 55 s | 1.6 min | 1.5 min x 2 | 36 s x 3 | 1.0 min | **8.4 min** | 900 / 900 / 900 |
| cavity3d_n244 K8 | B | 1.3 min | 3.0 min | 3.0 min x 2 | 53 s x 3 | 1.4 min | **14.4 min** | 900 / 1020 / 900 |
| cavity3d_n312 K8 | B | 2.1 min | 6.1 min | 6.1 min x 2 | 1.6 min x 3 | - | **25.1 min** | 900 / 1380 / - |
| cavity3d_n392 K8 | B | 3.6 min | 11.8 min | 11.8 min x 2 | 2.8 min x 3 | 3.9 min | **51.4 min** | 1080 / 2040 / 1080 |
| cavity3d_n392 K2 | A | 5.9 min | 11.8 min | 11.8 min x 2 | 6.8 min x 3 | - | **61.7 min** | 1320 / 2040 / - |
| cavity3d_n392 K4 | A | 4.3 min | 11.8 min | 11.8 min x 2 | 4.2 min x 3 | - | **52.2 min** | 1140 / 2040 / - |
| cavity3d_n192_k2 K2 | A | 1.5 min | 1.6 min | 1.5 min x 2 | 1.7 min x 3 | - | **11.2 min** | 900 / 900 / - |
| cavity3d_n192_k4 K4 | A | 2.1 min | 1.6 min | 1.5 min x 2 | 2.0 min x 3 | - | **12.7 min** | 900 / 900 / - |
| cavity3d_n192_k8 K8 | B | 3.4 min | 1.6 min | 1.5 min x 2 | 2.6 min x 3 | 3.6 min | **19.3 min** | 1020 / 960 / 1080 |

Line A (K <= 4, cards 0-3 of a whole node): 2.34 h; line B (K = 8): 2.02 h; both on whole nodes: **4.37 node-h** (one node back to back: the same). Time limits (2 x + 1 h): A 6 h, B 6 h.
Reference runs alone: 2.54 h of the 4.27 h (the 64M K = 1 references of the three n392 points are 1.77 h).

## Phase-1 job (2 cards, 300 steps each)

| item | case | K | steps | estimate | timeout (s) |
|---|---|---|---|---|---|
| 8m_k1 | cavity3d_n192 | 1 | 300 | 29 s | 600 |
| 8m_k2_trace | cavity3d_n192 | 2 | 300 | 40 s | 600 |
| dump_8m_k1 | cavity3d_n192 | 1 | 300 | 36 s | 600 |
| dump_8m_k2 | cavity3d_n192 | 2 | 300 | 31 s | 600 |
| s48_k1 | cavity3d_n48 | 1 | 300 | 6 s | 600 |
| s48_k2 | cavity3d_n48 | 2 | 300 | 8 s | 600 |
| dump_s48_k1 | cavity3d_n48 | 1 | 300 | 11 s | 600 |
| dump_s48_k2 | cavity3d_n48 | 2 | 300 | 13 s | 600 |
| weak2_k2 | cavity3d_n192_k2 | 2 | 300 | 41 s | 600 |
| 64m_k2 | cavity3d_n392 | 2 | 300 | 2.4 min | 600 |

Phase-1 job estimate incl. 166 s of overhead and the field comparisons: **8.8 min** (limit 30 min).
