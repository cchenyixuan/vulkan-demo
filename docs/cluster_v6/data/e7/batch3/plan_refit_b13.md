points: 51 (line A 34, line B 17); 5 trials: 22 points (<= 4.2M per card), 3 trials: 29; 2 traced runs per point; timing runs without the seam check
batches (node-h; wall = the longer line): 1: A 1.8 + B 2.7 = 4.5 (wall 2.7 h); 2: A 4.4 + B 4.4 = 8.8 (wall 4.4 h); 3: A 5.1 + B 2.4 = 7.5 (wall 5.1 h); 4: A 2.4 + B 2.2 = 4.6 (wall 2.4 h)
total 25.4 node-h = 203 GPU-h (queue not included)

| batch | line | job | points | items | estimate (h) | content |
|---|---|---|---|---|---|---|
| 1 | A | e7_b1A | 4 | 7 | 1.78 | F2, F3, F5, precheck |
| 1 | B | e7_b1B | 4 | 12 | 2.71 | F1, F4, precheck, selftest, soak |
| 2 | A | e7_b2A | 11 | 11 | 4.42 | F2, F3 |
| 2 | B | e7_b2B | 9 | 9 | 4.42 | F1, F3 |
| 3 | A | e7_b3A | 7 | 7 | 5.09 | F4, F5 |
| 3 | B | e7_b3B | 2 | 2 | 2.40 | F4, F6 |
| 4 | A | e7_b4A | 12 | 14 | 2.43 | F1 (cross), F7, F8 |
| 4 | B | e7_b4B | 2 | 7 | 2.16 | F2 (cross), F7, F7 (cross), anatomy, soak |

| batch | line | family | case | K | particles | per card | trials | references | run (min) | reference set (min) | calibration (min) | traced run (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 2.3 | 5.3 | 2.7 | 2.9 | 0.63 |
| 1 | A | F3 | cavity2d_n2840_k4 | 4 | 32.6M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.2 | 0.7 | 1.5 | 1.7 | 0.24 |
| 1 | A | F5 | cavity3d_n160_k8 | 4 | 36.4M | 9.09M | 3 | K=1 of the case | 2.3 | 6.3 | 2.4 | 2.8 | 0.68 |
| 1 | A | F3 | cavity2d_n1440_k2 | 2 | 4.2M | 2.12M | 5 | K=1 cavity2d_n1440 | 0.4 | 0.2 | 0.3 | 0.8 | 0.12 |
| 1 | B | F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.8 | 5.3 | 2.8 | 2.5 | 0.58 |
| 1 | B | F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 5 | K=1 of the case | 0.8 | 1.4 | 1.2 | 1.3 | 0.32 |
| 1 | B | F1 | cavity2d_n11320 | 8 | 128.6M | 16.08M | 3 | K=2 pairs | 3.4 | 6.2 | 5.1 | 4.3 | 0.88 |
| 1 | B | F4+F5 | cavity3d_n160_k8 | 8 | 36.4M | 4.54M | 3 | K=1 cavity3d_n160 + K=1 of the case | 1.6 | 7.2 | 2.2 | 2.4 | 0.64 |
| 2 | A | F2 | cavity2d_n5640 | 2 | 32.1M | 16.03M | 3 | K=1 of the case | 1.6 | 2.7 | 1.5 | 2.1 | 0.39 |
| 2 | A | F2+F7 | cavity2d_n5640 | 4 | 32.1M | 8.01M | 3 | K=1 of the case | 1.2 | 2.7 | 1.4 | 1.7 | 0.34 |
| 2 | A | F2 | cavity2d_n8000 | 2 | 64.4M | 32.18M | 3 | K=1 of the case | 3.2 | 5.3 | 2.8 | 3.7 | 0.75 |
| 2 | A | F3 | cavity2d_n1440_k4 | 4 | 8.5M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.5 | 0.2 | 0.6 | 0.9 | 0.15 |
| 2 | A | F3 | cavity2d_n2000_k2 | 2 | 8.1M | 4.07M | 5 | K=1 cavity2d_n2000 | 0.5 | 0.4 | 0.4 | 0.9 | 0.15 |
| 2 | A | F3 | cavity2d_n2000_k4 | 4 | 16.2M | 4.06M | 5 | K=1 cavity2d_n2000 | 0.7 | 0.4 | 0.8 | 1.1 | 0.20 |
| 2 | A | F3 | cavity2d_n2840_k2 | 2 | 16.3M | 8.16M | 3 | K=1 cavity2d_n2840 | 0.9 | 0.7 | 0.8 | 1.3 | 0.18 |
| 2 | A | F3 | cavity2d_n4000_k2 | 2 | 32.3M | 16.13M | 3 | K=1 cavity2d_n4000 | 1.6 | 1.4 | 1.5 | 2.1 | 0.33 |
| 2 | A | F3 | cavity2d_n4000_k4 | 4 | 64.4M | 16.11M | 3 | K=1 cavity2d_n4000 | 2.3 | 1.4 | 2.7 | 2.9 | 0.44 |
| 2 | A | F3 | cavity2d_n5640_k2 | 2 | 64.0M | 32.00M | 3 | K=1 cavity2d_n5640 | 3.2 | 2.7 | 2.8 | 3.7 | 0.62 |
| 2 | A | F3 | cavity2d_n5640_k4 | 4 | 127.9M | 31.96M | 3 | K=1 cavity2d_n5640 | 4.4 | 2.7 | 5.1 | 5.2 | 0.83 |
| 2 | B | F1+F7 | cavity2d_n2000 | 8 | 4.1M | 0.51M | 5 | K=1 of the case | 0.7 | 0.4 | 0.8 | 1.1 | 0.19 |
| 2 | B | F1 | cavity2d_n2840 | 8 | 8.2M | 1.02M | 5 | K=1 of the case | 0.7 | 0.7 | 0.9 | 1.2 | 0.24 |
| 2 | B | F1 | cavity2d_n5640 | 8 | 32.1M | 4.01M | 5 | K=1 of the case | 1.1 | 2.7 | 1.7 | 1.6 | 0.49 |
| 2 | B | F1 | cavity2d_n9000 | 8 | 81.4M | 10.17M | 3 | K=1 of the case | 2.3 | 6.7 | 3.4 | 3.0 | 0.72 |
| 2 | B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | 2.11M | 5 | K=1 cavity2d_n1440 | 0.9 | 0.2 | 1.2 | 1.3 | 0.23 |
| 2 | B | F3 | cavity2d_n2000_k8 | 8 | 32.4M | 4.05M | 5 | K=1 cavity2d_n2000 | 1.1 | 0.4 | 1.7 | 1.6 | 0.30 |
| 2 | B | F3 | cavity2d_n2840_k8 | 8 | 65.1M | 8.14M | 3 | K=1 cavity2d_n2840 | 1.9 | 0.7 | 2.8 | 2.5 | 0.35 |
| 2 | B | F3 | cavity2d_n4000_k8 | 8 | 128.8M | 16.10M | 3 | K=1 cavity2d_n4000 | 3.4 | 1.4 | 5.1 | 4.3 | 0.64 |
| 2 | B | F3 | cavity2d_n5640_k8 | 8 | 255.6M | 31.95M | 3 | K=1 cavity2d_n5640 | 6.6 | 2.7 | 9.5 | 7.9 | 1.21 |
| 3 | A | F4 | cavity3d_n160_k2 | 2 | 9.3M | 4.63M | 3 | K=1 cavity3d_n160 | 1.0 | 0.9 | 0.9 | 1.3 | 0.21 |
| 3 | A | F4 | cavity3d_n160_k4 | 4 | 18.3M | 4.57M | 3 | K=1 cavity3d_n160 | 1.2 | 0.9 | 1.3 | 1.6 | 0.24 |
| 3 | A | F4 | cavity3d_n200_k2 | 2 | 17.7M | 8.83M | 3 | K=1 cavity3d_n200 | 1.9 | 1.6 | 1.6 | 2.2 | 0.37 |
| 3 | A | F4 | cavity3d_n200_k4 | 4 | 35.0M | 8.74M | 3 | K=1 cavity3d_n200 | 2.2 | 1.6 | 2.4 | 2.7 | 0.43 |
| 3 | A | F5 | cavity3d_n160_k8 | 2 | 36.4M | 18.18M | 3 | K=1 of the case | 3.7 | 6.3 | 3.3 | 4.1 | 0.88 |
| 3 | A | F5 | cavity3d_n200_k8 | 2 | 69.6M | 34.78M | 3 | K=1 of the case | 7.0 | 12.0 | 6.2 | 7.5 | 1.65 |
| 3 | A | F5 | cavity3d_n200_k8 | 4 | 69.6M | 17.39M | 3 | K=1 of the case | 4.3 | 12.0 | 4.5 | 5.0 | 1.27 |
| 3 | B | F4+F5 | cavity3d_n200_k8 | 8 | 69.6M | 8.70M | 3 | K=1 cavity3d_n200 + K=1 of the case | 2.9 | 13.6 | 3.8 | 4.1 | 1.17 |
| 3 | B | F6 | cavity3d_n416 | 8 | 76.2M | 9.53M | 3 | K=1 of the case | 3.2 | 13.1 | 4.1 | 4.4 | 1.18 |
| 4 | A | F7 | cavity2d_n240 | 2 | 0.1M | 0.03M | 5 | K=1 of the case | 0.3 | 0.1 | 0.2 | 0.7 | 0.09 |
| 4 | A | F7 | cavity2d_n520 | 2 | 0.3M | 0.15M | 5 | K=1 of the case | 0.3 | 0.1 | 0.2 | 0.7 | 0.09 |
| 4 | A | F7 | cavity2d_n1000 | 2 | 1.0M | 0.52M | 5 | K=1 of the case | 0.4 | 0.2 | 0.2 | 0.7 | 0.10 |
| 4 | A | F7 | cavity2d_n2000 | 2 | 4.1M | 2.04M | 5 | K=1 of the case | 0.4 | 0.4 | 0.3 | 0.8 | 0.13 |
| 4 | A | F7 | cavity2d_n4000 | 2 | 16.2M | 8.09M | 3 | K=1 of the case | 0.9 | 1.4 | 0.8 | 1.3 | 0.21 |
| 4 | A | F7 | cavity2d_n360 | 4 | 0.1M | 0.04M | 5 | K=1 of the case | 0.4 | 0.1 | 0.3 | 0.8 | 0.11 |
| 4 | A | F7 | cavity2d_n720 | 4 | 0.6M | 0.14M | 5 | K=1 of the case | 0.4 | 0.2 | 0.3 | 0.8 | 0.11 |
| 4 | A | F7 | cavity2d_n1440 | 4 | 2.1M | 0.53M | 5 | K=1 of the case | 0.4 | 0.2 | 0.4 | 0.8 | 0.12 |
| 4 | A | F7 | cavity2d_n2840 | 4 | 8.2M | 2.05M | 5 | K=1 of the case | 0.5 | 0.7 | 0.6 | 0.9 | 0.19 |
| 4 | A | F8 | cavity3d_n104_b9 | 2 | 1.8M | 0.91M | 5 | K=1 of the case | 0.3 | 0.4 | 0.2 | 0.6 | 0.10 |
| 4 | A | F8 | cavity3d_n200_b9 | 2 | 10.4M | 5.18M | 3 | K=1 of the case | 1.1 | 1.8 | 1.0 | 1.4 | 0.27 |
| 4 | A | F8 | cavity3d_n200_x96 | 2 | 4.5M | 2.25M | 5 | K=1 of the case | 0.5 | 0.8 | 0.5 | 0.8 | 0.20 |
| 4 | A | cross F1+F7 | cavity2d_n4000 | 8 | 16.2M | 2.02M | 3 | K=1 of the case | 0.8 | 1.4 | 1.2 | 1.3 | 0.17 |
| 4 | A | cross F1+F7 | cavity2d_n8000 | 8 | 64.4M | 8.04M | 3 | K=1 of the case | 1.8 | 5.3 | 2.8 | 2.5 | 0.50 |
| 4 | B | F7 | cavity2d_n520 | 8 | 0.3M | 0.04M | 5 | K=1 of the case | 0.6 | 0.1 | 0.6 | 1.0 | 0.16 |
| 4 | B | F7 | cavity2d_n1000 | 8 | 1.0M | 0.13M | 5 | K=1 of the case | 0.6 | 0.2 | 0.7 | 1.0 | 0.16 |
| 4 | B | cross F7 | cavity2d_n2840 | 4 | 8.2M | 2.05M | 3 | K=1 of the case | 0.5 | 0.7 | 0.6 | 0.9 | 0.10 |
| 4 | B | cross F2 | cavity2d_n8000 | 4 | 64.4M | 16.09M | 3 | K=1 of the case | 2.3 | 5.3 | 2.7 | 2.9 | 0.54 |
