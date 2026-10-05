### fps (pipelined depth 2, no GPU timers)

| case | particles | transport | trials | fps median | fps min–max | two/three (medians) | ranges overlap |
|---|---:|---|---:|---:|---|---:|---|
| 50k | 61,009 | three_hop | 5 | 1510.1 | 1509.4–1510.9 |  |  |
| 50k | 61,009 | two_hop | 5 | 1524.3 | 1518.5–1528.8 | 100.94% | no |
| 100k | 117,649 | three_hop | 5 | 1476.4 | 1473.4–1484.3 |  |  |
| 100k | 117,649 | two_hop | 5 | 1497.4 | 1495.2–1500.5 | 101.43% | no |
| 250k | 273,529 | three_hop | 5 | 1338.2 | 1334.9–1345.8 |  |  |
| 250k | 273,529 | two_hop | 5 | 1352.2 | 1345.7–1354.9 | 101.04% | yes |
| 500k | 534,361 | three_hop | 5 | 1073.5 | 1073.3–1076.4 |  |  |
| 500k | 534,361 | two_hop | 5 | 1079.4 | 1078.4–1081.5 | 100.54% | no |
| 1m_gen | 1,046,529 | three_hop | 5 | 756.7 | 754.4–760.2 |  |  |
| 1m_gen | 1,046,529 | two_hop | 5 | 757.3 | 754.9–760.8 | 100.07% | yes |

### mechanism (µs)

GPU columns: instrumented depth-1 run, median over the two GPUs of the per-GPU p50; `gap mean` and `exposed frames` are over all post-warmup frames of both GPUs (exposed = b_to_c_gap > 20 µs). Worker columns: pipelined depth-2 trials, median over trials and the two workers.

| case | transport | staging KB/dir | phase B | readback sched gap | readback DMA | worker copy | worker signal | upload DMA | b_to_c_gap p50 | b_to_c_gap mean | exposed frames | upload→C slack p50 | worker upload wait | consumed signal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | three_hop | 210 | 266.6 | 39.7 | 9.3 | 32.4 | 11.7 | 17.0 | 6.0 | 29.4 | 7.71% | 102.9 | — | — |
| 50k | two_hop | 210 | 265.9 | 39.9 | 9.3 | 0.8 | 6.3 | 18.9 | 5.9 | 25.7 | 3.67% | 134.8 | 57.5 | 5.5 |
| 100k | three_hop | 294 | 280.8 | 40.1 | 12.5 | 40.4 | 11.6 | 20.9 | 6.1 | 31.1 | 9.02% | 99.2 | — | — |
| 100k | two_hop | 294 | 282.4 | 39.9 | 12.5 | 0.8 | 6.3 | 23.0 | 5.9 | 25.2 | 3.08% | 143.4 | 61.9 | 5.5 |
| 250k | three_hop | 446 | 351.5 | 40.8 | 17.7 | 57.0 | 11.5 | 28.5 | 6.0 | 27.2 | 5.03% | 138.4 | — | — |
| 250k | two_hop | 446 | 349.6 | 40.8 | 17.7 | 0.8 | 6.5 | 32.4 | 5.8 | 23.8 | 1.60% | 193.4 | 71.1 | 5.5 |
| 500k | three_hop | 618 | 524.7 | 41.6 | 24.4 | 76.8 | 11.8 | 39.4 | 6.0 | 22.7 | 0.98% | 270.5 | — | — |
| 500k | two_hop | 618 | 525.7 | 41.3 | 24.4 | 0.8 | 6.5 | 42.5 | 5.8 | 22.1 | 0.61% | 349.1 | 81.7 | 5.6 |
| 1m_gen | three_hop | 867 | 899.3 | 43.1 | 33.3 | 103.9 | 12.2 | 50.0 | 5.9 | 20.4 | 0.53% | 592.1 | — | — |
| 1m_gen | two_hop | 867 | 896.8 | 42.9 | 34.8 | 0.9 | 6.9 | 55.8 | 5.4 | 19.0 | 0.53% | 691.6 | 95.2 | 5.7 |

### correctness (all runs of the group)

| case | transport | runs | drift ≠ 0 | GPU stamp errors | host stamp errors | overwrite-during-upload | install drops | seam check failed | invalid runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | three_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50k | two_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100k | three_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100k | two_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 250k | three_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 250k | two_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500k | three_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500k | two_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1m_gen | three_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1m_gen | two_hop | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
