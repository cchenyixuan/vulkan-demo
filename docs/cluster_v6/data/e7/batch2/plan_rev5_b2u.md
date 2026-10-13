points: 5 planned of 55 in the campaign (line A 4, line B 1); 5 trials: 0 points (<= 4.2M per card), 3 trials: 5; arms C; traced runs per point [1]; timing runs without the seam check (revision 5 of 2026-10-12)
batches (node-h; wall = the longer line): 7: A 0.0 + B 1.5 = 1.5 (wall 1.5 h)
total 1.5 node-h = 12 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 7 | B | e7_b2U | 5 | 5 | 1.54 | F1, F2 |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | B | F2 | cavity2d_n8000 | 7 | 64.4M | 9.19M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
| 7 | B | F2 | cavity2d_n8000 | 6 | 64.4M | 10.73M | 3 | K=1 of the case | 1.6 | 3.2 | 2.7 | 2.2 | 0.32 |
| 7 | B | F2 | cavity2d_n8000 | 5 | 64.4M | 12.87M | 3 | K=1 of the case | 1.7 | 3.2 | 2.7 | 2.3 | 0.33 |
| 7 | B | F2 | cavity2d_n8000 | 3 | 64.4M | 21.45M | 3 | K=1 of the case | 1.9 | 3.2 | 2.7 | 2.4 | 0.34 |
| 7 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 3 | K=1 of the case | 0.9 | 1.6 | 1.7 | 1.5 | 0.18 |
wrote docs\cluster_v6\e7_jobs\e7_b2U.sbatch
