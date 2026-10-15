points: 18 planned of 67 in the campaign (line A 16, line B 2); 5 trials: 4 points (<= 4.2M per card), 3 trials: 13; arms C; traced runs per point [0, 1]; timing runs without the seam check (revision 6 of 2026-10-12)
batches (node-h; wall = the longer line): 8: A 0.0 + B 1.5 = 1.5 (wall 1.5 h); 9: A 2.4 + B 0.0 = 2.4 (wall 2.4 h)
total 3.9 node-h = 31 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 8 | B | e7_b2V | 5 | 5 | 1.54 | F1, F2 |
| 9 | A | e7_b2W | 13 | 13 | 2.37 | F3 |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | B | F2 | cavity2d_n8000 | 7 | 64.4M | 9.19M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
| 8 | B | F2 | cavity2d_n8000 | 6 | 64.4M | 10.73M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
| 8 | B | F2 | cavity2d_n8000 | 5 | 64.4M | 12.87M | 3 | K=1 of the case | 1.7 | 3.2 | 2.7 | 2.3 | 0.33 |
| 8 | B | F2 | cavity2d_n8000 | 3 | 64.4M | 21.45M | 3 | K=1 of the case | 1.9 | 3.2 | 2.7 | 2.4 | 0.34 |
| 8 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 3 | K=1 of the case | 0.9 | 1.6 | 1.7 | 1.5 | 0.18 |
| 9 | A | F3 | cavity2d_n1440_k3 | 3 | 6.3M | 2.12M | 5 | K=1 cavity2d_n1440 | 0.3 | 0.2 | 0.4 | 0.7 | 0.05 |
| 9 | A | F3 | cavity2d_n1440_k5 | 5 | 10.6M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.5 | 0.2 | 0.7 | 0.9 | 0.07 |
| 9 | A | F3 | cavity2d_n1440_k6 | 6 | 12.7M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.5 | 0.2 | 0.9 | 1.0 | 0.07 |
| 9 | A | F3 | cavity2d_n1440_k7 | 7 | 14.8M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.6 | 0.2 | 1.0 | 1.1 | 0.08 |
| 9 | A | F3 | cavity2d_n2840_k3 | 3 | 24.4M | 8.15M | 3 | K=1 cavity2d_n2840 | 0.8 | 0.5 | 1.1 | 1.2 | 0.08 |
| 9 | A | F3 | cavity2d_n2840_k5 | 5 | 40.7M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.1 | 0.5 | 1.8 | 1.6 | 0.11 |
| 9 | A | F3 | cavity2d_n2840_k6 | 6 | 48.8M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.3 | 0.5 | 2.1 | 1.8 | 0.12 |
| 9 | A | F3 | cavity2d_n2840_k7 | 7 | 57.0M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.4 | 0.5 | 2.5 | 2.0 | 0.14 |
| 9 | A | F3 | cavity2d_n5640_k3 | 3 | 95.9M | 31.98M | 3 | K=1 cavity2d_n5640 | 2.7 | 1.6 | 4.0 | 3.4 | 0.28 |
| 9 | A | F3 | cavity2d_n5640_k5 | 5 | 159.8M | 31.96M | 3 | K=1 cavity2d_n5640 | 3.8 | 1.6 | 6.2 | 4.8 | 0.38 |
| 9 | A | F3 | cavity2d_n5640_k6 | 6 | 191.7M | 31.95M | 3 | K=1 cavity2d_n5640 | 4.4 | 1.6 | 7.3 | 5.4 | 0.42 |
| 9 | A | F3 | cavity2d_n5640_k7 | 7 | 223.7M | 31.95M | 3 | K=1 cavity2d_n5640 | 4.9 | 1.6 | 8.4 | 6.1 | 0.47 |
| 9 | A | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 0 | K=1 cavity2d_n1440 | 0.7 | 0.2 | 1.2 | 1.2 | 0.04 |
wrote docs\cluster_v6\e7_jobs\e7_b2V.sbatch
wrote docs\cluster_v6\e7_jobs\e7_b2W.sbatch
