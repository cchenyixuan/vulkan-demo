### fps (pipelined depth 2, no GPU timers)

| case | particles | transport | trials | fps median | fps min–max | two/three (medians) | ranges overlap |
|---|---:|---|---:|---:|---|---:|---|
| 1m | 1,046,529 | three_hop | 3 | 759.3 | 758.5–763.1 |  |  |
| 1m | 1,046,529 | two_hop | 3 | 758.2 | 757.2–759.4 | 99.86% | yes |
| 2m | 2,064,969 | three_hop | 3 | 479.2 | 477.0–479.6 |  |  |
| 2m | 2,064,969 | two_hop | 3 | 478.4 | 477.8–479.1 | 99.82% | yes |
| 4m | 4,182,025 | three_hop | 3 | 242.6 | 242.3–242.8 |  |  |
| 4m | 4,182,025 | two_hop | 3 | 242.5 | 242.2–242.6 | 99.95% | yes |
| 8m | 8,128,201 | three_hop | 3 | 133.3 | 133.3–133.5 |  |  |
| 8m | 8,128,201 | two_hop | 3 | 133.2 | 132.9–133.2 | 99.86% | no |
| 16m | 16,184,529 | three_hop | 3 | 67.5 | 67.0–68.0 |  |  |
| 16m | 16,184,529 | two_hop | 3 | 67.8 | 66.9–67.8 | 100.34% | yes |

### mechanism (µs)

GPU columns: instrumented depth-1 run, median over the two GPUs of the per-GPU p50; `gap mean` and `exposed frames` are over all post-warmup frames of both GPUs (exposed = b_to_c_gap > 20 µs). Worker columns: pipelined depth-2 trials, median over trials and the two workers.

| case | transport | staging KB/dir | phase B | readback sched gap | readback DMA | worker copy | worker signal | upload DMA | b_to_c_gap p50 | b_to_c_gap mean | exposed frames | upload→C slack p50 | worker upload wait | consumed signal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 862 | 890.1 | 41.7 | 36.7 | 104.4 | 12.2 | 51.6 | 5.9 | 21.1 | 0.53% | 583.6 | — | — |
| 1m | two_hop | 862 | 891.5 | 42.4 | 38.4 | 0.9 | 6.8 | 55.9 | 5.4 | 19.6 | 0.53% | 688.4 | 95.5 | 5.7 |
| 2m | three_hop | 1212 | 1576.3 | 43.4 | 54.5 | 143.2 | 13.4 | 70.8 | 5.8 | 20.4 | 0.53% | 1179.3 | — | — |
| 2m | two_hop | 1212 | 1576.7 | 43.9 | 55.8 | 0.9 | 8.0 | 74.2 | 5.4 | 16.8 | 0.47% | 1324.7 | 116.6 | 5.7 |
| 4m | three_hop | 1725 | 3383.4 | 43.0 | 86.5 | 203.7 | 15.9 | 99.1 | 5.8 | 10.5 | 0.30% | 2860.3 | — | — |
| 4m | two_hop | 1725 | 3382.2 | 43.3 | 86.1 | 1.4 | 10.8 | 105.2 | 5.5 | 10.0 | 0.33% | 3057.6 | 150.9 | 6.8 |
| 8m | three_hop | 2402 | 6346.2 | 43.0 | 124.5 | 285.8 | 16.2 | 141.2 | 5.8 | 6.6 | 0.05% | 5666.0 | — | — |
| 8m | two_hop | 2402 | 6338.8 | 43.4 | 128.4 | 1.5 | 11.2 | 147.1 | 5.6 | 6.8 | 0.05% | 5935.4 | 193.7 | 6.8 |
| 16m | three_hop | 3391 | 12938.4 | 44.0 | 185.5 | 410.6 | 18.1 | 200.6 | 5.8 | 5.9 | 0.03% | 11986.8 | — | — |
| 16m | two_hop | 3391 | 12889.0 | 44.2 | 186.5 | 1.7 | 11.1 | 210.6 | 5.5 | 5.6 | 0.03% | 12344.5 | 257.9 | 7.7 |

### correctness (all runs of the group)

| case | transport | runs | drift ≠ 0 | GPU stamp errors | host stamp errors | overwrite-during-upload | install drops | seam check failed | invalid runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
