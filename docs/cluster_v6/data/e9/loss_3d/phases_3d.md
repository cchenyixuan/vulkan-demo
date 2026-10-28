## 2. tab:phases, 3-D row blocks (ideal chain = the per-card-size cube's K = 1 trace, not divided)

Rules of e8_phases_table / e7_phases_table: steady steps = step >= warmup with the next step present, minus the defrag step; A / B / C and the step median = each card's median, then the mean over the cards (min–max); step mean (window) = (sum of the steady periods − in-window defrag wall time) / steps; step mean (skip) = the mean of the kept steps. ms.

**3d_n392_K8** — strong 64M (400^3) K = 8 (02_e9_e9B_1741917, wqd10nbj04g4); ideal chain = K = 1 trace of the per-card-size cube 8M (200^3) (8M cube = the per-card size; trace_3d_n192_K1 of 02_e9_e9B_1741917, one card, not divided)

| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |
|---|---|---|---|---|---|---|---|
| strong 64M (400^3) | ideal chain: K = 1 trace of 8M (200^3), not divided (8M cube = the per-card size) | 0.743 | 44.777 | 0.068 | 45.594 | 46.287 | 46.333 |
| strong 64M (400^3) | chain K = 8 (02_e9_e9B_1741917, wqd10nbj04g4) | 0.845 (0.793–0.906) | 29.205 (25.507–36.204) | 19.936 (11.736–23.033) | 53.509 (53.067–53.975) | 54.053 | 54.069 |
| strong 64M (400^3) | for comparison: K = 1 trace of 64M (400^3) ÷ 8 (the 7.3 ideal row) | 0.754 | 45.801 | 0.090 | 46.647 | 47.143 | 47.191 |

ratio ideal / chain (window − defrag means) = **0.8563** (per-card-size reference, the eta' of the three-factor decomposition); whole-cube ÷ 8 / chain = 0.8722 (the trace's eta; timed eta 0.8776); 1 defrag event(s) in the chain window, 1 in the ideal's.

**3d_n192_K8** — strong 8M (200^3) K = 8 (02_e9_e9B_1741917, wqd10nbj04g4); ideal chain = K = 1 trace of the per-card-size cube 4M (160^3) (4M cube as the stand-in for 1M per card; trace_3d_n152_K1 of 02_e9_e9B_1741917, one card, ÷ 4.096 = x 1,000,000 / 4,096,000 (the per-card load at the cube's per-particle cost))

| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |
|---|---|---|---|---|---|---|---|
| strong 8M (200^3) | ideal chain: K = 1 trace of 4M (160^3), ÷ 4.096 = x 1,000,000 / 4,096,000 (the per-card load at the cube's per-particle cost) (4M cube as the stand-in for 1M per card) | 0.085 | 5.568 | 0.007 | 5.662 | 5.731 | 5.737 |
| strong 8M (200^3) | chain K = 8 (02_e9_e9B_1741917, wqd10nbj04g4) | 0.084 (0.080–0.085) | 1.786 (1.073–3.864) | 5.405 (3.441–6.279) | 8.784 (8.744–8.807) | 8.902 | 8.909 |
| strong 8M (200^3) | for comparison: K = 1 trace of 8M (200^3) ÷ 8 (the 7.3 ideal row) | 0.093 | 5.597 | 0.008 | 5.699 | 5.786 | 5.792 |
| strong 8M (200^3) | for reference: the 4M (160^3) K = 1 trace raw, not divided | 0.348 | 22.807 | 0.027 | 23.192 | 23.475 | 23.498 |

ratio ideal / chain (window − defrag means) = **0.6438** (per-card-size reference, the eta' of the three-factor decomposition); whole-cube ÷ 8 / chain = 0.6500 (the trace's eta; timed eta 0.6609); 1 defrag event(s) in the chain window, 1 in the ideal's.
