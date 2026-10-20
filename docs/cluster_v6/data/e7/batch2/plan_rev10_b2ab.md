points: 2 planned of 52 in the campaign (line A 2, line B 0); 5 trials: 0 points (<= 4.2M per card), 3 trials: 2; arms C; traced runs per point [1]; timing runs without the seam check (revision 10 of 2026-10-12)
batches (node-h; wall = the longer line): 14: A 0.0 + B 0.5 = 0.5 (wall 0.5 h)
total 0.5 node-h = 4 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 14 | B | e7_b2AB | 2 | 2 | 0.48 | F3 |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | B | F3 | cavity2d_n2840_k5 | 5 | 40.7M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.3 | 0.6 | 5.2 | 3.1 | 0.24 |
| 14 | B | F3 (weights old_2d_n2840_k5_K5.json, no calibration) | cavity2d_n2840_k5 | 5 | 40.7M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.3 | 0.6 | 0.0 | 3.1 | 0.15 |
wrote docs\cluster_v6\e7_jobs\e7_b2AB.sbatch
