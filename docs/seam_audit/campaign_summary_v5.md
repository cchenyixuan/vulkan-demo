# Seam audit campaign summary

- matrix: `experiment\seam_audit\matrix_v5_baseline.json`; generated 2026-10-02T22:19:22
- reference: v5 K=1 devices 1 (reference_v5_dev1), 2 trials; horizons [300, 2000]
- ratio = rms(|B1 - A1|) / rms(|A1 - A2|) on the same matched particles; d=0 = the seam column on both sides; flagged = a particle that crossed a seam in the last step lies within h

## Seam column (d=0) rms ratio test/noise

| case | test | N | triplets | acceleration all | acceleration flagged | acceleration unflagged | shift all | shift flagged | shift unflagged | density all | density flagged | density unflagged | far worst acceleration | B2-A2 acceleration d0 | B1-B2 noise acceleration d0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v5_K2 | 300 | 1,002,001 | 1.25 | - | 1.25 | 1.01 | - | 1.01 | 1.17 | - | 1.17 | 1.04 (d=15) | 1.53 | 1.7 |
| cavity2d_1m | v5_K2 | 2000 | 1,001,194 | 1.06 | - | 1.06 | 29 | - | 29 | 1.81 | - | 1.81 | 1.23 (d=15) | 1.2 | 1.04 |
| cavity2d_1m | v5_K4 | 300 | 1,002,001 | 1.03 | - | 1.03 | 0.881 | - | 0.881 | 1.24 | - | 1.24 | 1.2 (d=8) | 1.12 | 0.741 |
| cavity2d_1m | v5_K4 | 2000 | 1,001,389 | 1.09 | - | 1.09 | 2.23 | - | 2.23 | 1.58 | - | 1.58 | 1.04 (d=15) | 1.19 | 1.06 |
| cavity2d_4m | v5_K2 | 300 | 4,004,001 | 2.11 | - | 2.11 | 10.5 | - | 10.5 | 2.22 | - | 2.22 | 1.65 (d=10) | 2.1 | 0.781 |
| cavity2d_4m | v5_K2 | 2000 | 4,002,292 | 1.79 | - | 1.79 | 67.9 | - | 67.9 | 1.56 | - | 1.56 | 1.08 (d=8) | 1.76 | 1.1 |
| cavity2d_4m | v5_K4 | 300 | 4,004,001 | 1.14 | - | 1.14 | 8.31 | - | 8.31 | 0.862 | - | 0.862 | 1.27 (d=8) | 1.2 | 1.06 |
| cavity2d_4m | v5_K4 | 2000 | 4,002,347 | 1.85 | - | 1.85 | 22.8 | - | 22.8 | 0.635 | - | 0.635 | 1.04 (d=interior) | 1.87 | 1.03 |

## Matching and invariants

| case | test | N | column 0 all/flagged/unflagged/crossed | unmatched B1-A1 (kd) | unmatched A1-A2 (kd) | id agreement B1-A1 | max abs drift | stamps gpu/host | overflow | all runs valid | report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v5_K2 | 300 | 10,010/0/10,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v5_K2_N300/report.md) |
| cavity2d_1m | v5_K2 | 2000 | 10,010/0/10,010/0 | 606 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v5_K2_N2000/report.md) |
| cavity2d_1m | v5_K4 | 300 | 30,030/0/30,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v5_K4_N300/report.md) |
| cavity2d_1m | v5_K4 | 2000 | 30,030/0/30,030/0 | 265 | 400 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_1m/v5_K4_N2000/report.md) |
| cavity2d_4m | v5_K2 | 300 | 20,010/0/20,010/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v5_K2_N300/report.md) |
| cavity2d_4m | v5_K2 | 2000 | 20,011/0/20,011/0 | 1,046 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v5_K2_N2000/report.md) |
| cavity2d_4m | v5_K4 | 300 | 60,030/0/60,030/0 | 0 | 0 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v5_K4_N300/report.md) |
| cavity2d_4m | v5_K4 | 2000 | 60,031/0/60,031/0 | 884 | 1,311 | 1 | 0 | 0/0 | 0 | True | [report](analysis/cavity2d_4m/v5_K4_N2000/report.md) |

## Crossing window (column 0, pooled over the window frames)

Groups of test column-0 fluid particles in frames with a seam crossing: departed = a particle that LEFT this particle's slab this frame lies within h (V5 defect 2); arrived = only arrivals within h; control = between h and 1.5 h of a crossing. Ratio = rms |B1-A1| / rms |A1-A2| on the same particles and frame.

| case | test | N | frames w/ crossings | crossing particles | departed n | departed accel | departed density | departed kernel_sum | arrived n | arrived accel | control n | control accel | unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity2d_1m | v5_K2 | 2000 | 26 | 30 | 290 | 149 | 54.3 | 1.28e+04 | 290 | 33.5 | 334 | 12.1 | 0 |
| cavity2d_1m | v5_K4 | 2000 | 26 | 30 | 867 | 176 | 48.4 | 9.47e+03 | 872 | 28.4 | 1003 | 14 | 0 |
| cavity2d_4m | v5_K2 | 2000 | 13 | 21 | 199 | 98.5 | 30.5 | 1.25e+04 | 198 | 20 | 214 | 12.5 | 0 |
| cavity2d_4m | v5_K4 | 2000 | 10 | 21 | 590 | 122 | 13.9 | 4e+03 | 595 | 24.3 | 666 | 24.5 | 0 |
