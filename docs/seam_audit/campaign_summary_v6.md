# Seam audit campaign summary

- matrix: `experiment\seam_audit\matrix_v6.json`; generated 2026-10-02T22:25:56
- reference: v5 K=1 devices 1 (reference_v5_dev1), 2 trials; horizons [300, 2000]
- ratio = rms(|B1 - A1|) / rms(|A1 - A2|) on the same matched particles; d=0 = the seam column on both sides; flagged = a particle that crossed a seam in the last step lies within h

## Seam column (d=0) rms ratio test/noise

| case | test | N | triplets | acceleration all | acceleration flagged | acceleration unflagged | shift all | shift flagged | shift unflagged | density all | density flagged | density unflagged | far worst acceleration | B2-A2 acceleration d0 | B1-B2 noise acceleration d0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v6_keep0_layers1_K2 | 300 | 1,002,001 | 1.24 | - | 1.24 | 1.5 | - | 1.5 | 1.03 | - | 1.03 | 1.19 (d=13) | 1.79 | 1.73 |
| cavity2d_1m | v6_keep0_layers1_K2 | 2000 | 1,001,406 | 1.16 | - | 1.16 | 29.1 | - | 29.1 | 1.36 | - | 1.36 | 1.13 (d=11) | 1.11 | 1.1 |
| cavity2d_1m | v6_keep0_layers1_K4 | 300 | 1,002,001 | 0.992 | - | 0.992 | 0.865 | - | 0.865 | 1.05 | - | 1.05 | 1.28 (d=8) | 1.15 | 0.972 |
| cavity2d_1m | v6_keep0_layers1_K4 | 2000 | 1,001,505 | 1.1 | - | 1.1 | 2.2 | - | 2.2 | 1.22 | - | 1.22 | 1.01 (d=11) | 1.2 | 1.05 |
| cavity2d_1m | v6_keep1_layers1_K2 | 300 | 1,002,001 | 0.929 | - | 0.929 | 1.14 | - | 1.14 | 0.966 | - | 0.966 | 1.38 (d=8) | 1.74 | 1.19 |
| cavity2d_1m | v6_keep1_layers1_K2 | 2000 | 1,001,489 | 1.05 | - | 1.05 | 0.996 | - | 0.996 | 2.04 | - | 2.04 | 1.1 (d=12) | 1.07 | 1.08 |
| cavity2d_1m | v6_keep1_layers1_K4 | 300 | 1,002,001 | 0.955 | - | 0.955 | 0.936 | - | 0.936 | 1 | - | 1 | 1.41 (d=8) | 1.17 | 1.12 |
| cavity2d_1m | v6_keep1_layers1_K4 | 2000 | 1,001,494 | 1.04 | - | 1.04 | 1.08 | - | 1.08 | 1.5 | - | 1.5 | 1.04 (d=15) | 1.04 | 1.1 |
| cavity2d_1m | v6_keep1_layers2_K2 | 300 | 1,002,001 | 1.22 | - | 1.22 | 1.2 | - | 1.2 | 1.18 | - | 1.18 | 1.09 (d=interior) | 1.27 | 1.38 |
| cavity2d_1m | v6_keep1_layers2_K2 | 2000 | 1,001,438 | 1.04 | - | 1.04 | 1.05 | - | 1.05 | 1.47 | - | 1.47 | 1.18 (d=11) | 1.07 | 1.05 |
| cavity2d_1m | v6_keep1_layers2_K4 | 300 | 1,002,001 | 1.29 | - | 1.29 | 0.952 | - | 0.952 | 1.3 | - | 1.3 | 1.28 (d=8) | 0.851 | 1.28 |
| cavity2d_1m | v6_keep1_layers2_K4 | 2000 | 1,001,437 | 1.03 | - | 1.03 | 1.18 | - | 1.18 | 1.26 | - | 1.26 | 1.01 (d=11) | 1.04 | 1.09 |
| cavity2d_4m | v6_keep0_layers1_K2 | 300 | 4,004,001 | 1.58 | - | 1.58 | 10.5 | - | 10.5 | 1.92 | - | 1.92 | 1.38 (d=8) | 2.26 | 1.65 |
| cavity2d_4m | v6_keep0_layers1_K2 | 2000 | 4,002,467 | 1.76 | - | 1.76 | 67.8 | - | 67.8 | 1.28 | - | 1.28 | 1.08 (d=9) | 1.76 | 1.07 |
| cavity2d_4m | v6_keep0_layers1_K4 | 300 | 4,004,001 | 1.06 | - | 1.06 | 8.27 | - | 8.27 | 0.551 | - | 0.551 | 1.12 (d=8) | 1.29 | 0.547 |
| cavity2d_4m | v6_keep0_layers1_K4 | 2000 | 4,002,320 | 1.86 | - | 1.86 | 22.8 | - | 22.8 | 0.556 | - | 0.556 | 1.06 (d=interior) | - | - |
| cavity2d_4m | v6_keep1_layers1_K2 | 300 | 4,004,001 | 1.14 | - | 1.14 | 0.705 | - | 0.705 | 1.54 | - | 1.54 | 1.04 (d=14) | 1.63 | 1.29 |
| cavity2d_4m | v6_keep1_layers1_K2 | 2000 | 4,002,393 | 1.06 | - | 1.06 | 1.2 | - | 1.2 | 0.838 | - | 0.838 | 1.04 (d=interior) | 0.997 | 1.02 |
| cavity2d_4m | v6_keep1_layers1_K4 | 300 | 4,004,001 | 0.665 | - | 0.665 | 0.904 | - | 0.904 | 0.426 | - | 0.426 | 1.05 (d=interior) | 1.02 | 0.569 |
| cavity2d_4m | v6_keep1_layers1_K4 | 2000 | 4,002,382 | 0.996 | 1.41 | 0.996 | 0.847 | 1.17 | 0.839 | 0.406 | 4.75 | 0.405 | 1.04 (d=interior) | 1.01 | 1.02 |
| cavity2d_4m | v6_keep1_layers2_K2 | 300 | 4,004,001 | 0.897 | - | 0.897 | 0.718 | - | 0.718 | 1.01 | - | 1.01 | 0.983 (d=10) | 1.64 | 1.12 |
| cavity2d_4m | v6_keep1_layers2_K2 | 2000 | 4,002,193 | 1.05 | - | 1.05 | 0.958 | - | 0.958 | 1.21 | - | 1.21 | 1.05 (d=interior) | 0.966 | 1.01 |
| cavity2d_4m | v6_keep1_layers2_K4 | 300 | 4,004,001 | 0.436 | - | 0.436 | 0.951 | - | 0.951 | 0.227 | - | 0.227 | 1.22 (d=interior) | 1.04 | 0.844 |
| cavity2d_4m | v6_keep1_layers2_K4 | 2000 | 4,002,382 | 0.968 | 1.22 | 0.968 | 1.63 | 1.13 | 1.63 | 0.579 | 3.73 | 0.578 | 1.06 (d=interior) | 1.01 | 1.06 |

## Matching and invariants

| case | test | N | column 0 all/flagged/unflagged/crossed | unmatched B1-A1 (kd) | unmatched A1-A2 (kd) | id agreement B1-A1 | max abs drift | stamps gpu/host | overflow | all runs valid | report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v6_keep0_layers1_K2 | 300 | 10,010/0/10,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep0_layers1_K2_N300/report.md) |
| cavity2d_1m | v6_keep0_layers1_K2 | 2000 | 10,010/0/10,010/0 | 471 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep0_layers1_K2_N2000/report.md) |
| cavity2d_1m | v6_keep0_layers1_K4 | 300 | 30,030/0/30,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep0_layers1_K4_N300/report.md) |
| cavity2d_1m | v6_keep0_layers1_K4 | 2000 | 30,030/0/30,030/0 | 342 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep0_layers1_K4_N2000/report.md) |
| cavity2d_1m | v6_keep1_layers1_K2 | 300 | 10,010/0/10,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers1_K2_N300/report.md) |
| cavity2d_1m | v6_keep1_layers1_K2 | 2000 | 10,010/0/10,010/0 | 346 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers1_K2_N2000/report.md) |
| cavity2d_1m | v6_keep1_layers1_K4 | 300 | 30,030/0/30,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers1_K4_N300/report.md) |
| cavity2d_1m | v6_keep1_layers1_K4 | 2000 | 30,030/0/30,030/0 | 333 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers1_K4_N2000/report.md) |
| cavity2d_1m | v6_keep1_layers2_K2 | 300 | 10,010/0/10,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers2_K2_N300/report.md) |
| cavity2d_1m | v6_keep1_layers2_K2 | 2000 | 10,010/0/10,010/0 | 352 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers2_K2_N2000/report.md) |
| cavity2d_1m | v6_keep1_layers2_K4 | 300 | 30,030/0/30,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers2_K4_N300/report.md) |
| cavity2d_1m | v6_keep1_layers2_K4 | 2000 | 30,030/0/30,030/0 | 405 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v6_keep1_layers2_K4_N2000/report.md) |
| cavity2d_4m | v6_keep0_layers1_K2 | 300 | 20,010/0/20,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep0_layers1_K2_N300/report.md) |
| cavity2d_4m | v6_keep0_layers1_K2 | 2000 | 20,011/0/20,011/0 | 1,048 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep0_layers1_K2_N2000/report.md) |
| cavity2d_4m | v6_keep0_layers1_K4 | 300 | 60,030/0/60,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep0_layers1_K4_N300/report.md) |
| cavity2d_4m | v6_keep0_layers1_K4 | 2000 | 60,031/0/60,031/0 | 784 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep0_layers1_K4_N2000/report.md) |
| cavity2d_4m | v6_keep1_layers1_K2 | 300 | 20,010/0/20,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers1_K2_N300/report.md) |
| cavity2d_4m | v6_keep1_layers1_K2 | 2000 | 20,010/0/20,010/0 | 733 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers1_K2_N2000/report.md) |
| cavity2d_4m | v6_keep1_layers1_K4 | 300 | 60,030/0/60,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers1_K4_N300/report.md) |
| cavity2d_4m | v6_keep1_layers1_K4 | 2000 | 60,030/69/59,961/1 | 641 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers1_K4_N2000/report.md) |
| cavity2d_4m | v6_keep1_layers2_K2 | 300 | 20,010/0/20,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers2_K2_N300/report.md) |
| cavity2d_4m | v6_keep1_layers2_K2 | 2000 | 20,010/0/20,010/0 | 1,031 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers2_K2_N2000/report.md) |
| cavity2d_4m | v6_keep1_layers2_K4 | 300 | 60,030/0/60,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers2_K4_N300/report.md) |
| cavity2d_4m | v6_keep1_layers2_K4 | 2000 | 60,030/69/59,961/1 | 812 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v6_keep1_layers2_K4_N2000/report.md) |

## Crossing window (column 0, pooled over the window frames)

Groups of test column-0 fluid particles in frames with a seam crossing: departed = a particle that LEFT this particle's slab this frame lies within h (V5 defect 2); arrived = only arrivals within h; control = between h and 1.5 h of a crossing. Ratio = rms |B1-A1| / rms |A1-A2| on the same particles and frame.

| case | test | N | frames w/ crossings | crossing particles | departed n | departed accel | departed density | departed kernel_sum | arrived n | arrived accel | control n | control accel | unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v6_keep0_layers1_K2 | 2000 | 26 | 30 | 290 | 149 | 53.8 | 1.4e+04 | 290 | 33.8 | 334 | 12.2 | 0 |
| cavity2d_1m | v6_keep0_layers1_K4 | 2000 | 27 | 30 | 867 | 176 | 48.4 | 9.6e+03 | 872 | 28.6 | 1003 | 13.9 | 0 |
| cavity2d_1m | v6_keep1_layers1_K2 | 2000 | 26 | 30 | 290 | 1.03 | 1.01 | 1.31 | 281 | 1.11 | 343 | 1.01 | 0 |
| cavity2d_1m | v6_keep1_layers1_K4 | 2000 | 26 | 30 | 861 | 1.07 | 1.03 | 0.996 | 844 | 1.12 | 1036 | 1.1 | 0 |
| cavity2d_1m | v6_keep1_layers2_K2 | 2000 | 26 | 30 | 286 | 1 | 1.11 | 1.38 | 281 | 1.2 | 347 | 1.01 | 0 |
| cavity2d_1m | v6_keep1_layers2_K4 | 2000 | 26 | 30 | 860 | 1.01 | 0.944 | 1.13 | 845 | 1.04 | 1037 | 1.04 | 0 |
| cavity2d_4m | v6_keep0_layers1_K2 | 2000 | 13 | 20 | 199 | 98.9 | 30.3 | 1.36e+04 | 198 | 19.3 | 214 | 12.2 | 0 |
| cavity2d_4m | v6_keep0_layers1_K4 | 2000 | 10 | 21 | 590 | 124 | 14.2 | 3.92e+03 | 594 | 24.3 | 666 | 24 | 0 |
| cavity2d_4m | v6_keep1_layers1_K2 | 2000 | 10 | 20 | 209 | 1.11 | 1.33 | 0.997 | 207 | 0.976 | 286 | 1.06 | 0 |
| cavity2d_4m | v6_keep1_layers1_K4 | 2000 | 11 | 21 | 613 | 1.12 | 0.97 | 1.26 | 610 | 1.01 | 746 | 1.04 | 0 |
| cavity2d_4m | v6_keep1_layers2_K2 | 2000 | 11 | 21 | 208 | 0.773 | 0.615 | 0.982 | 207 | 0.819 | 287 | 0.818 | 0 |
| cavity2d_4m | v6_keep1_layers2_K4 | 2000 | 9 | 20 | 575 | 1.08 | 0.915 | 1 | 572 | 1.07 | 682 | 1.12 | 0 |

## Missing or failed

- cavity2d_4m v6_keep0_layers1_K4 N=2000: ok (partial inputs) ['cavity2d_4m/v6_keep0_layers1_K4_t2']
