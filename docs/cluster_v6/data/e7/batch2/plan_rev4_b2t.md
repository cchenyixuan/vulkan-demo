points: 4 planned of 55 in the campaign (line A 4, line B 0); 5 trials: 0 points (<= 4.2M per card), 3 trials: 4; arms C; traced runs per point [1]; timing runs without the seam check (revision 4 of 2026-10-12)
batches (node-h; wall = the longer line): 6: A 0.0 + B 1.4 = 1.4 (wall 1.4 h)
total 1.4 node-h = 11 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 6 | B | e7_b2T | 4 | 4 | 1.36 | F2 |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | B | F2 | cavity2d_n8000 | 3 | 64.4M | 21.45M | 3 | K=1 of the case | 1.9 | 3.2 | 2.7 | 2.4 | 0.34 |
| 6 | B | F2 | cavity2d_n8000 | 5 | 64.4M | 12.87M | 3 | K=1 of the case | 1.7 | 3.2 | 2.7 | 2.3 | 0.33 |
| 6 | B | F2 | cavity2d_n8000 | 6 | 64.4M | 10.73M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
| 6 | B | F2 | cavity2d_n8000 | 7 | 64.4M | 9.19M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
wrote docs\cluster_v6\e7_jobs\e7_b2T.sbatch
