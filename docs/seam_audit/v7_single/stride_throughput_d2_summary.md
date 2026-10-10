# v7_single measurement A: throughput and host overhead on one RTX 5090

Builds: S = v7_single, V = v7 single-GPU mode, Vx = V with V7_CONCURRENT_BUFFERS=0, C = v7 chain K = 1 (depth 2). fps = measured steps / wall time (empty queue at both ends; C: the orchestrator's steady window); warmup 1000; defrag every 1000 steps inside the window; mean ± std over interleaved trials.

## Throughput (fps)

| size | particles | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|---|
| 2m | 2,064,969 | n/a | 372.58 ± 0.67 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 4m_gen | 4,182,025 | n/a | 175.02 ± 0.08 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n406 | 4,092,529 | n/a | 182.07 ± 0.26 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n414 | 4,272,489 | n/a | 181.11 ± 0.21 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 6m | 6,105,841 | n/a | 126.61 ± 0.12 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 8m | 8,128,201 | n/a | 94.45 ± 0.08 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 10m | 10,144,225 | n/a | 77.98 ± 0.07 (n=3) | n/a | n/a | n/a | n/a | n/a |
| 14m | 14,160,169 | n/a | 52.75 ± 0.04 (n=3) | n/a | n/a | n/a | n/a | n/a |

## Particle throughput (million particle-steps per second, mean ± std)

| size | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|
| 2m | n/a | 769.4 ± 1.4 | n/a | n/a | n/a | n/a | n/a |
| 4m_gen | n/a | 731.9 ± 0.4 | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n406 | n/a | 745.1 ± 1.1 | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n414 | n/a | 773.8 ± 0.9 | n/a | n/a | n/a | n/a | n/a |
| 6m | n/a | 773.0 ± 0.7 | n/a | n/a | n/a | n/a | n/a |
| 8m | n/a | 767.7 ± 0.7 | n/a | n/a | n/a | n/a | n/a |
| 10m | n/a | 791.0 ± 0.7 | n/a | n/a | n/a | n/a | n/a |
| 14m | n/a | 747.0 ± 0.6 | n/a | n/a | n/a | n/a | n/a |

## Ratios (trial-wise, mean ± std; a run against the run of the same trial)

| size | V/S d1 | V/S d2 | Vx/S d1 | Vx/S d2 | C/S | Vx/V d1 | Vx/V d2 |
|---|---|---|---|---|---|---|---|
| 2m | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4m_gen | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n406 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4m_gen_n414 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6m | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8m | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 10m | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 14m | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Depth-1 loss (fps at 1 in flight / fps at 2 in flight - 1, trial-wise)

| size | S | V | Vx |
|---|---|---|---|
| 2m | n/a | n/a | n/a |
| 4m_gen | n/a | n/a | n/a |
| 4m_gen_n406 | n/a | n/a | n/a |
| 4m_gen_n414 | n/a | n/a | n/a |
| 6m | n/a | n/a | n/a |
| 8m | n/a | n/a | n/a |
| 10m | n/a | n/a | n/a |
| 14m | n/a | n/a | n/a |

## Covariates (nvidia-smi 1 Hz inside each window, mean over trials) and invariants

| size | build | SM MHz | W | max °C | max |drift| | overflow |
|---|---|---|---|---|---|---|
| 2m | S d2 | 2766 | 600 | 68 | 0 | none |
| 4m_gen | S d2 | 2756 | 600 | 68 | 0 | none |
| 4m_gen_n406 | S d2 | 2759 | 600 | 68 | 0 | none |
| 4m_gen_n414 | S d2 | 2764 | 600 | 68 | 0 | none |
| 6m | S d2 | 2763 | 600 | 68 | 0 | none |
| 8m | S d2 | 2766 | 600 | 68 | 0 | none |
| 10m | S d2 | 2765 | 600 | 69 | 0 | none |
| 14m | S d2 | 2759 | 600 | 69 | 0 | none |

## Paper numbers

```json
{
 "S_fps_range": {
  "2m": {
   "2": 372.58466666666664
  },
  "4m_gen": {
   "2": 175.0227
  },
  "4m_gen_n406": {
   "2": 182.07236666666665
  },
  "4m_gen_n414": {
   "2": 181.11173333333332
  },
  "6m": {
   "2": 126.60806666666667
  },
  "8m": {
   "2": 94.44536666666666
  },
  "10m": {
   "2": 77.97576666666667
  },
  "14m": {
   "2": 52.75043333333334
  }
 },
 "S_mparticle_steps_per_s_range": [
  731.9493069675,
  791.0037216141667
 ],
 "depth1_loss_S": {},
 "C_over_S": {},
 "V_over_S_d2": {},
 "Vx_over_S_d2": {}
}
```
