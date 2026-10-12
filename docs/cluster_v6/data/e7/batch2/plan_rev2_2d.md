points: 13 planned of 51 in the campaign (line A 7, line B 6); 5 trials: 0 points (<= 4.2M per card), 3 trials: 13; arms C; traced runs per point [0, 1]; timing runs without the seam check (revision 2 of 2026-10-12)
batches (node-h; wall = the longer line): 2: A 1.7 + B 1.7 = 3.4 (wall 1.7 h)
total 3.4 node-h = 27 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 2 | A | e7_b2A | 7 | 9 | 1.66 | F2, F3, anatomy, trace |
| 2 | B | e7_b2B | 6 | 8 | 1.74 | F1, F1 (cross), F3, F7, anatomy |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | A | F2 | cavity2d_n8000 | 2 | 64.4M | 32.18M | 3 | K=1 of the case | 2.2 | 3.2 | 2.8 | 2.7 | 0.36 |
| 2 | A | F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 1.7 | 3.2 | 2.7 | 2.3 | 0.33 |
| 2 | A | F3 | cavity2d_n1440_k4 | 4 | 8.5M | 2.11M | 3 | K=1 cavity2d_n1440 | 0.4 | 0.2 | 0.6 | 0.8 | 0.04 |
| 2 | A | F3 | cavity2d_n2840_k2 | 2 | 16.3M | 8.16M | 3 | K=1 cavity2d_n2840 | 0.6 | 0.5 | 0.8 | 1.0 | 0.07 |
| 2 | A | F3 | cavity2d_n2840_k4 | 4 | 32.6M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.0 | 0.5 | 1.5 | 1.4 | 0.10 |
| 2 | A | F3 | cavity2d_n5640_k2 | 2 | 64.0M | 32.00M | 3 | K=1 cavity2d_n5640 | 2.1 | 1.6 | 2.8 | 2.7 | 0.24 |
| 2 | A | F3 | cavity2d_n5640_k4 | 4 | 127.9M | 31.96M | 3 | K=1 cavity2d_n5640 | 3.3 | 1.6 | 5.1 | 4.1 | 0.33 |
| 2 | B | F1 | cavity2d_n11320 | 8 | 128.6M | 16.08M | 3 | K=2 pairs | 2.9 | 4.2 | 5.1 | 3.7 | 0.50 |
| 2 | B | F7 | cavity2d_n520 | 8 | 0.3M | 0.04M | 3 | K=1 of the case | 0.4 | 0.1 | 0.6 | 0.8 | 0.05 |
| 2 | B | F7 | cavity2d_n1000 | 8 | 1.0M | 0.13M | 3 | K=1 of the case | 0.4 | 0.1 | 0.7 | 0.9 | 0.05 |
| 2 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 3 | K=1 cavity2d_n1440 | 0.7 | 0.2 | 1.2 | 1.2 | 0.06 |
| 2 | B | F3 | cavity2d_n2840_k8 | 8 | 65.1M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.6 | 0.5 | 2.8 | 2.2 | 0.15 |
| 2 | B | F3 | cavity2d_n5640_k8 | 8 | 255.6M | 31.95M | 3 | K=1 cavity2d_n5640 | 5.5 | 1.6 | 9.5 | 6.8 | 0.52 |
| 2 | B | cross F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.6 | 3.2 | 2.8 | 2.2 | 0.32 |
