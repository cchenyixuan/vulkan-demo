points: 4 planned of 51 in the campaign (line A 0, line B 4); 5 trials: 0 points (<= 4.2M per card), 3 trials: 0; arms C; traced runs per point [1]; timing runs without the seam check (revision 9 of 2026-10-12)
batches (node-h; wall = the longer line): 13: A 0.0 + B 0.9 = 0.9 (wall 0.9 h)
total 0.9 node-h = 8 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 13 | B | e7_b2AA | 4 | 8 | 0.94 | F1, F3, tracegroup |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | B | F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 0 | K=1 of the case | 1.8 | 3.3 | 7.8 | 4.5 | 0.20 |
| 13 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 0 | K=1 of the case | 1.2 | 1.8 | 6.7 | 3.7 | 0.17 |
| 13 | B | F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 0 | K=1 of the case | 0.9 | 1.0 | 6.2 | 3.4 | 0.16 |
| 13 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 0 | K=1 cavity2d_n1440 | 1.0 | 0.3 | 6.2 | 3.4 | 0.16 |
wrote docs\cluster_v6\e7_jobs\e7_b2AA.sbatch
