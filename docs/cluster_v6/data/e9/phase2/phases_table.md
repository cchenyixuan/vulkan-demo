## tab:phases (E8 phase 2)

**3d_n192_k2_K2** — weak 2 x 8M K = 2 (01_e9_e9A_1741918; chain trace fps 21.07 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 2 (per card median, mean over cards (min–max)) | 0.732 (0.720–0.744) | 41.969 (41.788–42.150) | 3.391 (3.367–3.415) | 46.756 (46.735–46.777) | 47.401 | 47.430 |
| ideal chain = K = 1 trace of 3d_n192 (per-card cube, not divided) | 0.743 (0.743–0.743) | 44.777 (44.777–44.777) | 0.068 (0.068–0.068) | 45.594 (45.594–45.594) | 46.287 | 46.333 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9765** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.744 / 42.150 / 3.415 / 46.777; 1: 0.720 / 41.788 / 3.367 / 46.735

**3d_n392_K2** — 64M (400^3) K = 2 (01_e9_e9A_1741918; chain trace fps 5.17 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 2 (per card median, mean over cards (min–max)) | 3.078 (3.069–3.086) | 173.024 (171.916–174.131) | 12.348 (12.161–12.535) | 192.577 (189.943–195.212) | 193.178 | 193.288 |
| ideal chain = K = 1 trace of 3d_n392 ÷ 2 | 3.018 (3.018–3.018) | 183.204 (183.204–183.204) | 0.361 (0.361–0.361) | 186.586 (186.586–186.586) | 188.573 | 188.762 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9762** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 3.086 / 174.131 / 12.535 / 189.943; 1: 3.069 / 171.916 / 12.161 / 195.212

**3d_n192_k4_K4** — weak 4 x 8M K = 4 (01_e9_e9A_1741918; chain trace fps 21.14 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 4 (per card median, mean over cards (min–max)) | 0.735 (0.720–0.744) | 40.455 (38.685–42.147) | 4.808 (3.407–6.159) | 46.738 (46.668–46.763) | 47.256 | 47.271 |
| ideal chain = K = 1 trace of 3d_n192 (per-card cube, not divided) | 0.743 (0.743–0.743) | 44.777 (44.777–44.777) | 0.068 (0.068–0.068) | 45.594 (45.594–45.594) | 46.287 | 46.333 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9795** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.744 / 41.930 / 3.522 / 46.760; 1: 0.740 / 39.059 / 6.146 / 46.760; 2: 0.720 / 38.685 / 6.159 / 46.763; 3: 0.735 / 42.147 / 3.407 / 46.668

**3d_n392_K4** — 64M (400^3) K = 4 (01_e9_e9A_1741918; chain trace fps 9.96 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 4 (per card median, mean over cards (min–max)) | 1.614 (1.554–1.659) | 76.509 (70.467–82.341) | 17.641 (11.970–23.349) | 98.706 (97.828–99.891) | 100.270 | 100.310 |
| ideal chain = K = 1 trace of 3d_n392 ÷ 4 | 1.509 (1.509–1.509) | 91.602 (91.602–91.602) | 0.180 (0.180–0.180) | 93.293 (93.293–93.293) | 94.287 | 94.381 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9403** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 1.554 / 80.111 / 11.970 / 98.993; 1: 1.659 / 73.120 / 23.130 / 97.828; 2: 1.591 / 70.467 / 23.349 / 99.891; 3: 1.650 / 82.341 / 12.114 / 98.113

**3d_n152_K8** — 4M (160^3) K = 8 (02_e9_e9B_1741917; chain trace fps 156.67 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.056 (0.049–0.060) | 0.787 (0.013–2.170) | 3.591 (2.168–4.226) | 6.234 (6.160–6.273) | 6.377 | 6.383 |
| ideal chain = K = 1 trace of 3d_n152 ÷ 8 | 0.044 (0.044–0.044) | 2.851 (2.851–2.851) | 0.003 (0.003–0.003) | 2.899 (2.899–2.899) | 2.934 | 2.937 |
| ratio ideal / chain (window − defrag means) | | | | | **0.4601** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.053 / 2.170 / 2.168 / 6.160; 1: 0.054 / 0.013 / 4.226 / 6.256; 2: 0.059 / 0.487 / 3.844 / 6.273; 3: 0.060 / 0.476 / 3.882 / 6.252; 4: 0.059 / 0.483 / 3.829 / 6.237; 5: 0.055 / 0.013 / 4.197 / 6.244; 6: 0.059 / 0.491 / 4.170 / 6.225; 7: 0.049 / 2.166 / 2.409 / 6.223

**3d_n192_K8** — 8M (200^3) K = 8 (02_e9_e9B_1741917; chain trace fps 112.24 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.084 (0.080–0.085) | 1.786 (1.073–3.864) | 5.405 (3.441–6.279) | 8.784 (8.744–8.807) | 8.902 | 8.909 |
| ideal chain = K = 1 trace of 3d_n192 ÷ 8 | 0.093 (0.093–0.093) | 5.597 (5.597–5.597) | 0.008 (0.008–0.008) | 5.699 (5.699–5.699) | 5.786 | 5.792 |
| ratio ideal / chain (window − defrag means) | | | | | **0.6500** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.080 / 3.864 / 3.441 / 8.753; 1: 0.084 / 1.119 / 5.971 / 8.793; 2: 0.084 / 1.093 / 5.799 / 8.807; 3: 0.085 / 1.110 / 5.924 / 8.800; 4: 0.084 / 1.073 / 6.076 / 8.807; 5: 0.084 / 1.098 / 6.010 / 8.794; 6: 0.084 / 1.114 / 6.279 / 8.775; 7: 0.084 / 3.818 / 3.740 / 8.744

**3d_n192_k8_K8** — weak 8 x 8M K = 8 (02_e9_e9B_1741917; chain trace fps 20.96 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.737 (0.719–0.753) | 39.743 (38.167–42.195) | 5.461 (3.387–6.394) | 46.888 (46.814–46.946) | 47.667 | 47.681 |
| ideal chain = K = 1 trace of 3d_n192 (per-card cube, not divided) | 0.743 (0.743–0.743) | 44.777 (44.777–44.777) | 0.068 (0.068–0.068) | 45.594 (45.594–45.594) | 46.287 | 46.333 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9711** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.744 / 42.039 / 3.387 / 46.884; 1: 0.719 / 38.167 / 6.394 / 46.924; 2: 0.753 / 39.759 / 6.062 / 46.880; 3: 0.740 / 38.742 / 6.046 / 46.925; 4: 0.738 / 39.089 / 6.092 / 46.946; 5: 0.739 / 38.759 / 6.182 / 46.814; 6: 0.739 / 39.195 / 6.115 / 46.883; 7: 0.720 / 42.195 / 3.408 / 46.852

**3d_n244_K8** — 16M (252^3) K = 8 (02_e9_e9B_1741917; chain trace fps 69.23 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.179 (0.156–0.194) | 5.112 (3.221–8.746) | 8.192 (4.805–9.349) | 14.279 (14.205–14.334) | 14.428 | 14.440 |
| ideal chain = K = 1 trace of 3d_n244 ÷ 8 | 0.191 (0.191–0.191) | 11.151 (11.151–11.151) | 0.019 (0.019–0.019) | 11.361 (11.361–11.361) | 11.555 | 11.567 |
| ratio ideal / chain (window − defrag means) | | | | | **0.8009** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.182 / 8.746 / 4.805 / 14.205; 1: 0.171 / 3.221 / 9.169 / 14.244; 2: 0.192 / 4.552 / 9.232 / 14.256; 3: 0.171 / 3.252 / 9.252 / 14.294; 4: 0.194 / 4.559 / 9.349 / 14.334; 5: 0.194 / 4.615 / 9.346 / 14.274; 6: 0.171 / 3.234 / 9.234 / 14.314; 7: 0.156 / 8.715 / 5.152 / 14.314

**3d_n312_K8** — 32M (320^3) K = 8 (02_e9_e9B_1741917; chain trace fps 36.15 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.408 (0.381–0.428) | 12.818 (9.252–18.306) | 12.847 (7.512–14.711) | 27.309 (27.216–27.542) | 27.631 | 27.644 |
| ideal chain = K = 1 trace of 3d_n312 ÷ 8 | 0.387 (0.387–0.387) | 22.943 (22.943–22.943) | 0.044 (0.044–0.044) | 23.375 (23.375–23.375) | 23.752 | 23.776 |
| ratio ideal / chain (window − defrag means) | | | | | **0.8596** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.404 / 18.265 / 7.512 / 27.232; 1: 0.427 / 11.925 / 14.711 / 27.331; 2: 0.384 / 9.252 / 14.215 / 27.313; 3: 0.428 / 11.717 / 14.487 / 27.240; 4: 0.426 / 11.761 / 14.552 / 27.282; 5: 0.387 / 9.497 / 14.695 / 27.216; 6: 0.428 / 11.821 / 14.639 / 27.317; 7: 0.381 / 18.306 / 7.968 / 27.542

**3d_n392_K8** — 64M (400^3) K = 8 (02_e9_e9B_1741917; chain trace fps 18.48 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 0.845 (0.793–0.906) | 29.205 (25.507–36.204) | 19.936 (11.736–23.033) | 53.509 (53.067–53.975) | 54.053 | 54.069 |
| ideal chain = K = 1 trace of 3d_n392 ÷ 8 | 0.754 (0.754–0.754) | 45.801 (45.801–45.801) | 0.090 (0.090–0.090) | 46.647 (46.647–46.647) | 47.143 | 47.191 |
| ratio ideal / chain (window − defrag means) | | | | | **0.8722** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 0.801 / 35.931 / 11.736 / 53.141; 1: 0.839 / 25.796 / 22.718 / 53.261; 2: 0.902 / 29.353 / 22.745 / 53.975; 3: 0.842 / 25.616 / 22.336 / 53.856; 4: 0.839 / 25.507 / 22.430 / 53.737; 5: 0.906 / 29.596 / 23.033 / 53.317; 6: 0.841 / 25.638 / 22.514 / 53.720; 7: 0.793 / 36.204 / 11.979 / 53.067

**3d_n496_K8** — 128M (504^3) K = 8 (03_e9_e9C_1741921; chain trace fps 9.77 without defrag, 1 defrag event(s) in the window)

| row | T_A (ms) | T_B (ms) | T_C (ms) | step median (ms) | step mean window − defrag (ms) | step mean skip (ms) |
|---|---|---|---|---|---|---|
| chain K = 8 (per card median, mean over cards (min–max)) | 1.705 (1.660–1.775) | 63.332 (56.028–78.364) | 31.704 (18.637–36.199) | 101.070 (100.285–102.195) | 102.216 | 102.247 |
| ideal chain = K = 2 pair trace of 3d_n496 ÷ 4 | 1.542 (1.541–1.543) | 86.488 (86.460–86.516) | 4.848 (4.836–4.861) | 93.406 (92.900–93.912) | 94.147 | 94.200 |
| ratio ideal / chain (window − defrag means) | | | | | **0.9211** | |

per card (sim: device): T_A / T_B / T_C / period medians (ms): 0: 1.722 / 78.364 / 18.638 / 100.341; 1: 1.661 / 56.028 / 35.693 / 101.516; 2: 1.661 / 56.963 / 36.199 / 101.677; 3: 1.775 / 62.007 / 36.197 / 100.313; 4: 1.774 / 61.781 / 36.125 / 101.439; 5: 1.660 / 56.797 / 36.160 / 100.792; 6: 1.664 / 56.578 / 35.980 / 100.285; 7: 1.722 / 78.139 / 18.637 / 102.195
