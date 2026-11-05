# Re 5000, 2500²: RTX 5090 + RX 7900 XTX against 2× RTX 5090

Mixed run `n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000`, continued from `n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1` at step 33500000 (t = 100.50).

## 1. Run

- devices (pilot record): ['NVIDIA GeForce RTX 5090', 'AMD Radeon RX 7900 XTX']; weights [1.351579, 0.648421] (5090 : XTX = 2.084), cut [341], slab particles [4297822, 2067707], 2 pilot round(s)
- samples after the start: 14, t 100.701 → 103.314 (step 33567000 → 34438000)
- steps/s median 131.8 (range 130.1–131.9)
- alive at every sample: [6365529] (expected 6365529); MLS fallbacks max 0; min neighbours 74; density 999.474–1000.639
- segment 0 from step 33500000: complete, totals {'overflow': 0, 'far_migration': 0, 'stamp_gpu': 0, 'stamp_host': 0, 'drift': 0}
- result.json: complete; steady.json: {'steady_time': 100.49999999999999, 'decided_at_step': 33500000, 'profile_change': 0.0007553750698335212, 'kinetic_energy_change': 0.0011929779521037932, 'steady_tol': 0.0012, 'decided_by': 're5000_pipeline --mixed-steady-tol from n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1/windows.jsonl'}

## 2. Same global step, mixed − 2×5090

14 common samples, t 100.701–103.314. Relative L2 in the Marchi norm.

| quantity | mean | max | last |
|---|---|---|---|
| u, 28 points (%) | 0.0185 | 0.0274 | 0.0274 |
| v, 28 points (%) | 0.0261 | 0.0472 | 0.0127 |
| max |Δu|, 28 points | 0.000199 | 0.000332 | 0.00032 |
| max |Δv|, 28 points | 0.000393 | 0.000862 | 0.000125 |
| max |Δu|, 1001 points | 0.000589 | 0.000976 | 0.000603 |
| max |Δv|, 1001 points | 0.000755 | 0.00112 | 0.001 |
| ΔKE / KE | -1.8e-05 | 3.27e-05 | -3.27e-05 |

Same-vendor scale: 2×5090 − K = 1 at 225 common steps (t 20.3–65.7): u 0.0084–0.0700 % (mean 0.0309), v 0.0122–0.0563 % (mean 0.0292).

## 3. Against Marchi 2021 (relative L2 %, Linf in lid units)

| | t | samples | L2 u | L2 v | Linf u | Linf v |
|---|---|---|---|---|---|---|
| mixed, newest sample | 103.314 | 1 | 0.276 | 0.366 | | |
| mixed, all samples | 100.70–103.31 | 14 | 0.281 | 0.370 | 2.69e-03 | 4.74e-03 |
| mixed, common steps | 100.70–103.31 | 14 | 0.281 | 0.370 | 2.69e-03 | 4.74e-03 |
| 2×5090, same steps | 100.70–103.31 | 14 | 0.281 | 0.360 | 2.72e-03 | 4.50e-03 |

## 4. Extrema of the window-mean dense lines vs Marchi 2021 Table 27

| | value (Marchi) | position (Marchi) | mixed, all samples value | mixed, all samples position | mixed, common steps value | mixed, common steps position | 2×5090, same steps value | 2×5090, same steps position |
|---|---|---|---|---|---|---|---|---|
| u_min | -0.44731 | 0.0743 | -0.44797 (-0.15 %) | 0.0735 (-0.0008) | -0.44797 (-0.15 %) | 0.0735 (-0.0008) | -0.44795 (-0.14 %) | 0.0735 (-0.0008) |
| v_max | 0.44751 | 0.0797 | 0.44798 (+0.11 %) | 0.0791 (-0.0006) | 0.44798 (+0.11 %) | 0.0791 (-0.0006) | 0.44799 (+0.11 %) | 0.0792 (-0.0005) |
| v_min | -0.57614 | 0.9575 | -0.57706 (-0.16 %) | 0.9579 (+0.0004) | -0.57706 (-0.16 %) | 0.9579 (+0.0004) | -0.57712 (-0.17 %) | 0.9579 (+0.0004) |

## 5. Window tests of the mixed run after the start

- none: the run stopped at step 34500000, before the next window end at step 36850000 (window 3350000 steps); the steady decision is the 2×5090 run's window at the start
