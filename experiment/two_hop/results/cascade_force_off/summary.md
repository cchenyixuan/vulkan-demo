### fps (pipelined depth 2, no GPU timers)

| case | particles | transport | trials | fps median | fps min–max | two/three (medians) | ranges overlap |
|---|---:|---|---:|---:|---|---:|---|
| 1m | 1,046,529 | three_hop | 3 | 795.0 | 794.8–797.1 |  |  |
| 1m | 1,046,529 | two_hop | 3 | 797.3 | 796.2–798.0 | 100.30% | yes |
| 2m | 2,064,969 | three_hop | 3 | 488.6 | 488.4–489.5 |  |  |
| 2m | 2,064,969 | two_hop | 3 | 489.6 | 488.2–492.8 | 100.20% | yes |
| 4m | 4,182,025 | three_hop | 3 | 246.7 | 246.1–246.9 |  |  |
| 4m | 4,182,025 | two_hop | 3 | 247.2 | 246.8–248.2 | 100.20% | yes |

### mechanism (µs)

GPU columns: instrumented depth-1 run, median over the two GPUs of the per-GPU p50; `gap mean` and `exposed frames` are over all post-warmup frames of both GPUs (exposed = b_to_c_gap > 20 µs). Worker columns: pipelined depth-2 trials, median over trials and the two workers.

| case | transport | staging KB/dir | phase B | readback sched gap | readback DMA | worker copy | worker signal | upload DMA | b_to_c_gap p50 | b_to_c_gap mean | exposed frames | upload→C slack p50 | worker upload wait | consumed signal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 862 | 550.3 | 41.3 | 37.2 | 103.3 | 11.9 | 50.2 | 6.1 | 23.5 | 1.27% | 247.6 | — | — |
| 1m | two_hop | 862 | 550.3 | 41.0 | 36.2 | 0.8 | 6.5 | 54.9 | 5.8 | 21.4 | 0.53% | 350.6 | 95.2 | 5.5 |
| 2m | three_hop | 1212 | 963.8 | 41.3 | 52.0 | 143.2 | 12.9 | 71.8 | 5.9 | 20.2 | 0.45% | 574.3 | — | — |
| 2m | two_hop | 1212 | 964.9 | 42.5 | 53.8 | 0.9 | 7.3 | 73.1 | 5.8 | 19.3 | 0.45% | 718.5 | 115.2 | 5.7 |
| 4m | three_hop | 1725 | 2050.6 | 42.9 | 86.1 | 205.4 | 13.9 | 99.8 | 5.9 | 15.2 | 0.47% | 1532.9 | — | — |
| 4m | two_hop | 1725 | 2047.2 | 43.1 | 84.2 | 1.1 | 9.6 | 105.3 | 5.8 | 14.4 | 0.45% | 1733.6 | 149.0 | 6.1 |

### correctness (all runs of the group)

| case | transport | runs | drift ≠ 0 | GPU stamp errors | host stamp errors | overwrite-during-upload | install drops | seam check failed | invalid runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
