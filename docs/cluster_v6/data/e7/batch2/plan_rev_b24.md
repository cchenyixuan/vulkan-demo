points: 18 planned of 51 in the campaign (line A 9, line B 9); 5 trials: 0 points (<= 4.2M per card), 3 trials: 18; arms C; traced runs per point [0, 1]; timing runs without the seam check (revision of 2026-10-12)
batches (node-h; wall = the longer line): 2: A 2.6 + B 2.0 = 4.6 (wall 2.6 h); 4: A 0.0 + B 1.1 = 1.1 (wall 1.1 h)
total 5.7 node-h = 45 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 2 | A | e7_b2A | 9 | 9 | 2.61 | F2, F3 |
| 2 | B | e7_b2B | 7 | 7 | 2.00 | F1, F3 |
| 4 | B | e7_b4B | 2 | 8 | 1.05 | F1 (cross), F7, anatomy |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | A | F2 | cavity2d_n5640 | 2 | 32.1M | 16.03M | 3 | K=1 of the case | 1.6 | 2.7 | 1.5 | 2.1 | 0.27 |
| 2 | A | F2+F7 | cavity2d_n5640 | 4 | 32.1M | 8.01M | 3 | K=1 of the case | 1.2 | 2.7 | 1.4 | 1.7 | 0.25 |
| 2 | A | F2 | cavity2d_n8000 | 2 | 64.4M | 32.18M | 3 | K=1 of the case | 3.2 | 5.3 | 2.8 | 3.7 | 0.53 |
| 2 | A | F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 2.3 | 5.3 | 2.7 | 2.9 | 0.47 |
| 2 | A | F3 | cavity2d_n1440_k4 | 4 | 8.5M | 2.11M | 3 | K=1 cavity2d_n1440 | 0.5 | 0.2 | 0.6 | 0.9 | 0.05 |
| 2 | A | F3 | cavity2d_n2840_k2 | 2 | 16.3M | 8.16M | 3 | K=1 cavity2d_n2840 | 0.9 | 0.7 | 0.8 | 1.3 | 0.09 |
| 2 | A | F3 | cavity2d_n2840_k4 | 4 | 32.6M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.2 | 0.7 | 1.5 | 1.7 | 0.12 |
| 2 | A | F3 | cavity2d_n5640_k2 | 2 | 64.0M | 32.00M | 3 | K=1 cavity2d_n5640 | 3.2 | 2.7 | 2.8 | 3.7 | 0.34 |
| 2 | A | F3 | cavity2d_n5640_k4 | 4 | 127.9M | 31.96M | 3 | K=1 cavity2d_n5640 | 4.4 | 2.7 | 5.1 | 5.2 | 0.44 |
| 2 | B | F1+F7 | cavity2d_n2000 | 8 | 4.1M | 0.51M | 3 | K=1 of the case | 0.7 | 0.4 | 0.8 | 1.1 | 0.08 |
| 2 | B | F1 | cavity2d_n2840 | 8 | 8.2M | 1.02M | 3 | K=1 of the case | 0.7 | 0.7 | 0.9 | 1.2 | 0.11 |
| 2 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 3 | K=1 of the case | 1.1 | 2.7 | 1.7 | 1.6 | 0.24 |
| 2 | B | F1 | cavity2d_n11320 | 8 | 128.6M | 16.08M | 3 | K=2 pairs | 3.4 | 6.2 | 5.1 | 4.3 | 0.64 |
| 2 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 3 | K=1 cavity2d_n1440 | 0.9 | 0.2 | 1.2 | 1.3 | 0.08 |
| 2 | B | F3 | cavity2d_n2840_k8 | 8 | 65.1M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.9 | 0.7 | 2.8 | 2.5 | 0.18 |
| 2 | B | F3 | cavity2d_n5640_k8 | 8 | 255.6M | 31.95M | 3 | K=1 cavity2d_n5640 | 6.6 | 2.7 | 9.5 | 7.9 | 0.62 |
| 4 | B | F7 | cavity2d_n520 | 8 | 0.3M | 0.04M | 3 | K=1 of the case | 0.6 | 0.1 | 0.6 | 1.0 | 0.06 |
| 4 | B | F7 | cavity2d_n1000 | 8 | 1.0M | 0.13M | 3 | K=1 of the case | 0.6 | 0.2 | 0.7 | 1.0 | 0.07 |
| 4 | B | cross F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.8 | 5.3 | 2.8 | 2.5 | 0.40 |
