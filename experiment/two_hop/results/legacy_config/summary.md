### fps (pipelined depth 2, no GPU timers)

| case | particles | transport | trials | fps median | fps min–max | two/three (medians) | ranges overlap |
|---|---:|---|---:|---:|---|---:|---|
| 1m | 1,046,529 | three_hop | 3 | 610.5 | 596.7–612.7 |  |  |
| 1m | 1,046,529 | two_hop | 3 | 787.6 | 771.8–791.1 | 129.03% | no |
| 2m | 2,064,969 | three_hop | 3 | 420.1 | 418.8–420.4 |  |  |
| 2m | 2,064,969 | two_hop | 3 | 483.6 | 482.6–485.1 | 115.11% | no |
| 4m | 4,182,025 | three_hop | 3 | 245.2 | 245.0–245.9 |  |  |
| 4m | 4,182,025 | two_hop | 3 | 245.5 | 245.4–246.3 | 100.12% | yes |

### mechanism (µs)

GPU columns: instrumented depth-1 run, median over the two GPUs of the per-GPU p50; `gap mean` and `exposed frames` are over all post-warmup frames of both GPUs (exposed = b_to_c_gap > 20 µs). Worker columns: pipelined depth-2 trials, median over trials and the two workers.

| case | transport | staging KB/dir | phase B | readback sched gap | readback DMA | worker copy | worker signal | upload DMA | b_to_c_gap p50 | b_to_c_gap mean | exposed frames | upload→C slack p50 | worker upload wait | consumed signal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 3217 | 540.3 | 42.6 | 175.6 | 412.0 | 12.8 | 190.1 | 389.8 | 426.1 | 99.95% | 35.2 | — | — |
| 1m | two_hop | 3217 | 551.2 | 42.2 | 178.8 | 0.8 | 6.6 | 189.6 | 6.0 | 39.7 | 23.03% | 70.9 | 236.9 | 5.7 |
| 2m | three_hop | 4519 | 967.7 | 41.9 | 246.5 | 580.9 | 15.0 | 267.6 | 310.0 | 343.5 | 99.67% | 34.2 | — | — |
| 2m | two_hop | 4519 | 976.6 | 43.0 | 249.1 | 1.0 | 7.8 | 280.6 | 6.8 | 23.8 | 0.92% | 330.0 | 324.1 | 6.1 |
| 4m | three_hop | 6433 | 2063.4 | 42.5 | 371.2 | 826.7 | 15.2 | 395.4 | 7.3 | 27.3 | 2.23% | 352.0 | — | — |
| 4m | two_hop | 6433 | 2062.0 | 44.0 | 375.2 | 1.1 | 9.4 | 396.7 | 5.8 | 16.7 | 0.45% | 1163.3 | 450.3 | 6.2 |

### correctness (all runs of the group)

| case | transport | runs | drift ≠ 0 | GPU stamp errors | host stamp errors | overwrite-during-upload | install drops | seam check failed | invalid runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | three_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4m | two_hop | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
