points: 7 planned of 52 in the campaign (line A 2, line B 5); 5 trials: 3 points (<= 4.2M per card), 3 trials: 4; arms C; traced runs per point [0, 1]; timing runs without the seam check (revision 3 of 2026-10-12)
batches (node-h; wall = the longer line): 5: A 0.0 + B 1.4 = 1.4 (wall 1.4 h)
total 1.4 node-h = 11 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 5 | B | e7_b2S | 7 | 11 | 1.36 | F1, F3, lock_begin, lock_end |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | B | F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 3 | K=1 of the case | 0.7 | 0.9 | 1.2 | 1.2 | 0.12 |
| 5 | B | F3 | cavity2d_n1440_k2 | 2 | 4.2M | 2.12M | 5 | K=1 cavity2d_n1440 | 0.3 | 0.2 | 0.3 | 0.7 | 0.04 |
| 5 | B | F1 | cavity2d_n1440 | 8 | 2.1M | 0.27M | 3 | K=1 of the case | 0.4 | 0.2 | 0.7 | 0.9 | 0.06 |
| 5 | B | F1+F7 | cavity2d_n2000 | 8 | 4.1M | 0.51M | 3 | K=1 of the case | 0.5 | 0.3 | 0.8 | 0.9 | 0.07 |
| 5 | B | F1 | cavity2d_n2840 | 8 | 8.2M | 1.02M | 3 | K=1 of the case | 0.5 | 0.5 | 0.9 | 1.0 | 0.08 |
| 5 | B | F3 | cavity2d_n1440_k4 | 4 | 8.5M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.4 | 0.2 | 0.6 | 0.8 | 0.06 |
| 5 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.7 | 0.2 | 1.2 | 1.2 | 0.09 |
| 5 | B | lock F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.6 | 3.2 | 2.8 | 2.2 | 0.29 |
| 5 | B | lock F3 | cavity2d_n5640_k8 | 8 | 255.6M | 31.95M | 3 | K=1 cavity2d_n5640 | 5.5 | 1.6 | 9.5 | 6.8 | 0.52 |
wrote docs\cluster_v6\e7_jobs\e7_b2S.sbatch
