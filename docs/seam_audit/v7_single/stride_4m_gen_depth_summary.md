# v7_single measurement A: throughput and host overhead on one RTX 5090

Builds: S = v7_single, V = v7 single-GPU mode, Vx = V with V7_CONCURRENT_BUFFERS=0, C = v7 chain K = 1 (depth 2). fps = measured steps / wall time (empty queue at both ends; C: the orchestrator's steady window); warmup 1000; defrag every 1000 steps inside the window; mean ± std over interleaved trials.

## Throughput (fps)

| size | particles | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|---|
| 4m_gen | 4,182,025 | 171.83 ± 0.03 (n=3) | 174.78 ± 0.16 (n=3) | n/a | n/a | n/a | n/a | n/a |

## Particle throughput (million particle-steps per second, mean ± std)

| size | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|
| 4m_gen | 718.6 ± 0.1 | 731.0 ± 0.6 | n/a | n/a | n/a | n/a | n/a |

## Ratios (trial-wise, mean ± std; a run against the run of the same trial)

| size | V/S d1 | V/S d2 | Vx/S d1 | Vx/S d2 | C/S | Vx/V d1 | Vx/V d2 |
|---|---|---|---|---|---|---|---|
| 4m_gen | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Depth-1 loss (fps at 1 in flight / fps at 2 in flight - 1, trial-wise)

| size | S | V | Vx |
|---|---|---|---|
| 4m_gen | -1.69 % ± 0.11 | n/a | n/a |

## Covariates (nvidia-smi 1 Hz inside each window, mean over trials) and invariants

| size | build | SM MHz | W | max °C | max |drift| | overflow |
|---|---|---|---|---|---|---|
| 4m_gen | S d1 | 2769 | 600 | 69 | 0 | none |
| 4m_gen | S d2 | 2753 | 600 | 68 | 0 | none |

## Paper numbers

```json
{
 "S_fps_range": {
  "4m_gen": {
   "1": 171.82736666666668,
   "2": 174.7843333333333
  }
 },
 "S_mparticle_steps_per_s_range": [
  718.5863430841667,
  730.9524516083333
 ],
 "depth1_loss_S": {
  "4m_gen": {
   "mean": -0.01691717306460694,
   "std": 0.0010620597976476857
  }
 },
 "C_over_S": {},
 "V_over_S_d2": {},
 "Vx_over_S_d2": {}
}
```
