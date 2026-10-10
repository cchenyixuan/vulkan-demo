points: 51 (line A 34, line B 17); 5 trials: 22 points (<= 4.2M per card), 3 trials: 29; 2 traced runs per point; timing runs without the seam check
batches (node-h; wall = the longer line): 1: A 1.9 + B 2.8 = 4.6 (wall 2.8 h); 2: A 4.6 + B 4.6 = 9.1 (wall 4.6 h); 3: A 5.2 + B 2.5 = 7.7 (wall 5.2 h); 4: A 2.5 + B 2.2 = 4.7 (wall 2.5 h)
total 26.2 node-h = 210 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 1 | A | e7_b1A | 4 | 7 | 1.85 | F2, F3, F5, precheck |
| 1 | B | e7_b1B | 4 | 12 | 2.80 | F1, F4, precheck, selftest, soak |
| 2 | A | e7_b2A | 11 | 11 | 4.59 | F2, F3 |
| 2 | B | e7_b2B | 9 | 9 | 4.55 | F1, F3 |
| 3 | A | e7_b3A | 7 | 7 | 5.23 | F4, F5 |
| 3 | B | e7_b3B | 2 | 2 | 2.46 | F4, F6 |
| 4 | A | e7_b4A | 12 | 14 | 2.54 | F1 (cross), F7, F8 |
| 4 | B | e7_b4B | 2 | 7 | 2.18 | F2 (cross), F7, F7 (cross), anatomy, soak |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 2.4 | 5.4 | 2.8 | 3.0 | 0.65 |
| 1 | A | F3 | cavity2d_n2840_k4 | 4 | 32.6M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.3 | 0.8 | 1.5 | 1.8 | 0.25 |
| 1 | A | F5 | cavity3d_n160_k8 | 4 | 36.4M | 9.09M | 3 | K=1 of the case | 2.4 | 6.5 | 2.6 | 2.8 | 0.70 |
| 1 | A | F3 | cavity2d_n1440_k2 | 2 | 4.2M | 2.12M | 5 | K=1 cavity2d_n1440 | 0.4 | 0.3 | 0.3 | 0.8 | 0.13 |
| 1 | B | F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.9 | 5.4 | 2.9 | 2.6 | 0.59 |
| 1 | B | F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 5 | K=1 of the case | 0.8 | 1.4 | 1.2 | 1.3 | 0.32 |
| 1 | B | F1 | cavity2d_n11320 | 8 | 128.6M | 16.08M | 3 | K=2 pairs | 3.6 | 6.4 | 5.3 | 4.5 | 0.91 |
| 1 | B | F4+F5 | cavity3d_n160_k8 | 8 | 36.4M | 4.54M | 3 | K=1 cavity3d_n160 + K=1 of the case | 1.7 | 7.4 | 2.3 | 2.3 | 0.65 |
| 2 | A | F2 | cavity2d_n5640 | 2 | 32.1M | 16.03M | 3 | K=1 of the case | 1.7 | 2.7 | 1.5 | 2.2 | 0.40 |
| 2 | A | F2+F7 | cavity2d_n5640 | 4 | 32.1M | 8.01M | 3 | K=1 of the case | 1.2 | 2.7 | 1.5 | 1.8 | 0.35 |
| 2 | A | F2 | cavity2d_n8000 | 2 | 64.4M | 32.18M | 3 | K=1 of the case | 3.3 | 5.4 | 3.0 | 3.8 | 0.78 |
| 2 | A | F3 | cavity2d_n1440_k4 | 4 | 8.5M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.5 | 0.3 | 0.6 | 0.9 | 0.15 |
| 2 | A | F3 | cavity2d_n2000_k2 | 2 | 8.1M | 4.07M | 5 | K=1 cavity2d_n2000 | 0.5 | 0.4 | 0.5 | 0.9 | 0.16 |
| 2 | A | F3 | cavity2d_n2000_k4 | 4 | 16.2M | 4.06M | 5 | K=1 cavity2d_n2000 | 0.7 | 0.4 | 0.9 | 1.2 | 0.21 |
| 2 | A | F3 | cavity2d_n2840_k2 | 2 | 16.3M | 8.16M | 3 | K=1 cavity2d_n2840 | 0.9 | 0.8 | 0.8 | 1.3 | 0.19 |
| 2 | A | F3 | cavity2d_n4000_k2 | 2 | 32.3M | 16.13M | 3 | K=1 cavity2d_n4000 | 1.7 | 1.4 | 1.6 | 2.2 | 0.34 |
| 2 | A | F3 | cavity2d_n4000_k4 | 4 | 64.4M | 16.11M | 3 | K=1 cavity2d_n4000 | 2.4 | 1.4 | 2.8 | 3.0 | 0.45 |
| 2 | A | F3 | cavity2d_n5640_k2 | 2 | 64.0M | 32.00M | 3 | K=1 cavity2d_n5640 | 3.2 | 2.7 | 3.0 | 3.8 | 0.64 |
| 2 | A | F3 | cavity2d_n5640_k4 | 4 | 127.9M | 31.96M | 3 | K=1 cavity2d_n5640 | 4.5 | 2.7 | 5.4 | 5.4 | 0.86 |
| 2 | B | F1+F7 | cavity2d_n2000 | 8 | 4.1M | 0.51M | 5 | K=1 of the case | 0.6 | 0.4 | 0.7 | 1.1 | 0.19 |
| 2 | B | F1 | cavity2d_n2840 | 8 | 8.2M | 1.02M | 5 | K=1 of the case | 0.7 | 0.8 | 0.9 | 1.2 | 0.24 |
| 2 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 5 | K=1 of the case | 1.1 | 2.7 | 1.7 | 1.7 | 0.50 |
| 2 | B | F1 | cavity2d_n9000 | 8 | 81.4M | 10.17M | 3 | K=1 of the case | 2.3 | 6.8 | 3.5 | 3.1 | 0.74 |
| 2 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.9 | 0.3 | 1.2 | 1.4 | 0.23 |
| 2 | B | F3 | cavity2d_n2000_k8 | 8 | 32.4M | 4.05M | 5 | K=1 cavity2d_n2000 | 1.1 | 0.4 | 1.7 | 1.7 | 0.31 |
| 2 | B | F3 | cavity2d_n2840_k8 | 8 | 65.1M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.9 | 0.8 | 2.9 | 2.6 | 0.37 |
| 2 | B | F3 | cavity2d_n4000_k8 | 8 | 128.8M | 16.10M | 3 | K=1 cavity2d_n4000 | 3.6 | 1.4 | 5.3 | 4.5 | 0.67 |
| 2 | B | F3 | cavity2d_n5640_k8 | 8 | 255.6M | 31.95M | 3 | K=1 cavity2d_n5640 | 6.9 | 2.7 | 10.1 | 8.2 | 1.26 |
| 3 | A | F4 | cavity3d_n160_k2 | 2 | 9.3M | 4.63M | 3 | K=1 cavity3d_n160 | 1.1 | 0.9 | 0.9 | 1.3 | 0.21 |
| 3 | A | F4 | cavity3d_n160_k4 | 4 | 18.3M | 4.57M | 3 | K=1 cavity3d_n160 | 1.3 | 0.9 | 1.4 | 1.6 | 0.25 |
| 3 | A | F4 | cavity3d_n200_k2 | 2 | 17.7M | 8.83M | 3 | K=1 cavity3d_n200 | 1.9 | 1.7 | 1.7 | 2.2 | 0.38 |
| 3 | A | F4 | cavity3d_n200_k4 | 4 | 35.0M | 8.74M | 3 | K=1 cavity3d_n200 | 2.3 | 1.7 | 2.5 | 2.7 | 0.45 |
| 3 | A | F5 | cavity3d_n160_k8 | 2 | 36.4M | 18.18M | 3 | K=1 of the case | 3.8 | 6.5 | 3.4 | 4.1 | 0.90 |
| 3 | A | F5 | cavity3d_n200_k8 | 2 | 69.6M | 34.78M | 3 | K=1 of the case | 7.1 | 12.3 | 6.4 | 7.5 | 1.69 |
| 3 | A | F5 | cavity3d_n200_k8 | 4 | 69.6M | 17.39M | 3 | K=1 of the case | 4.4 | 12.3 | 4.7 | 5.0 | 1.30 |
| 3 | B | F4+F5 | cavity3d_n200_k8 | 8 | 69.6M | 8.70M | 3 | K=1 cavity3d_n200 + K=1 of the case | 3.0 | 14.0 | 3.9 | 3.8 | 1.19 |
| 3 | B | F6 | cavity3d_n416 | 8 | 76.2M | 9.53M | 3 | K=1 of the case | 3.2 | 13.5 | 4.3 | 4.1 | 1.21 |
| 4 | A | F7 | cavity2d_n240 | 2 | 0.1M | 0.03M | 5 | K=1 of the case | 0.4 | 0.2 | 0.2 | 0.7 | 0.10 |
| 4 | A | F7 | cavity2d_n520 | 2 | 0.3M | 0.15M | 5 | K=1 of the case | 0.4 | 0.2 | 0.2 | 0.7 | 0.10 |
| 4 | A | F7 | cavity2d_n1000 | 2 | 1.0M | 0.52M | 5 | K=1 of the case | 0.4 | 0.2 | 0.2 | 0.7 | 0.11 |
| 4 | A | F7 | cavity2d_n2000 | 2 | 4.1M | 2.04M | 5 | K=1 of the case | 0.4 | 0.4 | 0.3 | 0.8 | 0.14 |
| 4 | A | F7 | cavity2d_n4000 | 2 | 16.2M | 8.09M | 3 | K=1 of the case | 0.9 | 1.4 | 0.8 | 1.3 | 0.22 |
| 4 | A | F7 | cavity2d_n360 | 4 | 0.1M | 0.04M | 5 | K=1 of the case | 0.4 | 0.2 | 0.3 | 0.8 | 0.11 |
| 4 | A | F7 | cavity2d_n720 | 4 | 0.6M | 0.14M | 5 | K=1 of the case | 0.4 | 0.2 | 0.3 | 0.8 | 0.11 |
| 4 | A | F7 | cavity2d_n1440 | 4 | 2.1M | 0.53M | 5 | K=1 of the case | 0.4 | 0.3 | 0.4 | 0.8 | 0.13 |
| 4 | A | F7 | cavity2d_n2840 | 4 | 8.2M | 2.05M | 5 | K=1 of the case | 0.5 | 0.8 | 0.6 | 0.9 | 0.19 |
| 4 | A | F8 | cavity3d_n104_b9 | 2 | 1.8M | 0.91M | 5 | K=1 of the case | 0.3 | 0.4 | 0.3 | 0.6 | 0.11 |
| 4 | A | F8 | cavity3d_n200_b9 | 2 | 10.4M | 5.18M | 3 | K=1 of the case | 1.2 | 1.9 | 1.0 | 1.5 | 0.28 |
| 4 | A | F8 | cavity3d_n200_x96 | 2 | 4.5M | 2.25M | 5 | K=1 of the case | 0.6 | 0.9 | 0.5 | 0.9 | 0.21 |
| 4 | A | cross F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 3 | K=1 of the case | 0.8 | 1.4 | 1.2 | 1.3 | 0.18 |
| 4 | A | cross F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.9 | 5.4 | 2.9 | 2.6 | 0.51 |
| 4 | B | F7 | cavity2d_n520 | 8 | 0.3M | 0.04M | 5 | K=1 of the case | 0.6 | 0.2 | 0.6 | 1.0 | 0.15 |
| 4 | B | F7 | cavity2d_n1000 | 8 | 1.0M | 0.13M | 5 | K=1 of the case | 0.6 | 0.2 | 0.6 | 1.0 | 0.16 |
| 4 | B | cross F7 | cavity2d_n2840 | 4 | 8.2M | 2.05M | 3 | K=1 of the case | 0.5 | 0.8 | 0.6 | 0.9 | 0.10 |
| 4 | B | cross F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 2.4 | 5.4 | 2.8 | 3.0 | 0.55 |
