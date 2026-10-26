## 2. tab:phases, 3-D row blocks (ideal chain = the per-card-size cube's K = 1 trace, not divided)

Rules of e8_phases_table / e7_phases_table: steady steps = step >= warmup with the next step present, minus the defrag step; A / B / C and the step median = each card's median, then the mean over the cards (min–max); step mean (window) = (sum of the steady periods − in-window defrag wall time) / steps; step mean (skip) = the mean of the kept steps. ms.

**3d_n392_K8** — strong 64M (400^3) K = 8 (02_e8_e8B_1721839, wqd10nbj11g2); ideal chain = K = 1 trace of the per-card-size cube 8M (200^3) (8M cube = the per-card size; trace_3d_n192_K1 of 02_e8_e8B_1721839, one card, not divided)

| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |
|---|---|---|---|---|---|---|---|
| strong 64M (400^3) | ideal chain: K = 1 trace of 8M (200^3), not divided (8M cube = the per-card size) | 0.750 | 50.872 | 0.061 | 51.734 | 52.703 | 52.756 |
| strong 64M (400^3) | chain K = 8 (02_e8_e8B_1721839, wqd10nbj11g2) | 0.848 (0.798–0.906) | 33.283 (29.156–40.694) | 20.161 (11.985–23.199) | 57.405 (57.040–58.932) | 58.375 | 58.396 |
| strong 64M (400^3) | for comparison: K = 1 trace of 64M (400^3) ÷ 8 (the 7.3 ideal row) | 0.756 | 47.558 | 0.090 | 48.416 | 48.897 | 48.946 |

ratio ideal / chain (window − defrag means) = **0.9028** (per-card-size reference, the eta' of the three-factor decomposition); whole-cube ÷ 8 / chain = 0.8376 (the trace's eta; timed eta 0.8405); 1 defrag event(s) in the chain window, 1 in the ideal's.

**3d_n192_K8** — strong 8M (200^3) K = 8 (02_e8_e8B_1721839, wqd10nbj11g2); ideal chain = K = 1 trace of the per-card-size cube 4M (160^3) (4M cube as the stand-in for 1M per card; trace_3d_n152_K1 of 02_e8_e8B_1721839, one card, ÷ 4.096 = x 1,000,000 / 4,096,000 (the per-card load at the cube's per-particle cost))

| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |
|---|---|---|---|---|---|---|---|
| strong 8M (200^3) | ideal chain: K = 1 trace of 4M (160^3), ÷ 4.096 = x 1,000,000 / 4,096,000 (the per-card load at the cube's per-particle cost) (4M cube as the stand-in for 1M per card) | 0.085 | 6.397 | 0.006 | 6.496 | 6.601 | 6.608 |
| strong 8M (200^3) | chain K = 8 (02_e8_e8B_1721839, wqd10nbj11g2) | 0.088 (0.087–0.092) | 2.217 (1.388–4.676) | 5.440 (3.438–6.259) | 9.979 (9.928–10.012) | 10.113 | 10.110 |
| strong 8M (200^3) | for comparison: K = 1 trace of 8M (200^3) ÷ 8 (the 7.3 ideal row) | 0.094 | 6.359 | 0.008 | 6.467 | 6.588 | 6.594 |
| strong 8M (200^3) | for reference: the 4M (160^3) K = 1 trace raw, not divided | 0.346 | 26.202 | 0.024 | 26.608 | 27.038 | 27.065 |

ratio ideal / chain (window − defrag means) = **0.6528** (per-card-size reference, the eta' of the three-factor decomposition); whole-cube ÷ 8 / chain = 0.6514 (the trace's eta; timed eta 0.6634); 1 defrag event(s) in the chain window, 1 in the ideal's.
