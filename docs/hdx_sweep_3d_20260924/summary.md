# 3-D h/dx sweep (E6) — V0, one RTX 5090, 1,295,029 particles (101^3 fluid + 4 wall layers), T = 1.200 s

| h/dx | neighbors | dt (s) | steps to T | us/step | s per T | speed-up | fallback | rho std/rho0 | min spacing/dx | pairs < 0.5 dx | u(y) L2 | v(x) L2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 268 | 6.00e-05 | 20,000 | 12058 | 241.2 | 1.00x | 0 | 1.20e-05 | 0.865 | 0 | 0.00 % | 0.00 % | PASS |
| 3.50 | 180 | 5.25e-05 | 22,858 | 8487 | 194.0 | 1.24x | 0 | 9.53e-06 | 0.902 | 0 | 0.07 % | 0.01 % | PASS |
| 3.00 | 113 | 4.50e-05 | 26,667 | 6024 | 160.6 | 1.50x | 0 | 6.87e-06 | 0.917 | 0 | 0.21 % | 0.01 % | FAIL |
| 2.50 | 65 | 3.75e-05 | 32,000 | 4576 | 146.4 | 1.65x | 0 | 4.16e-06 | 0.927 | 0 | 0.21 % | 0.05 % | FAIL |
| 2.00 | 34 | 3.00e-05 | 40,000 | 3537 | 141.5 | 1.70x | 0 | 2.04e-05 | 0.000 | 217503 | 9.76 % | 2.46 % | FAIL |

Resolution-sensitivity baseline (h/dx 4 vs 3.5): u L2 = 0.07 %, v L2 = 0.01 % of U.

## Secondary metrics

| h/dx | max|rho-rho0|/rho0 | interior p std/(rho0 U^2) | W_sum mean (int.) | W_sum 1 % (int.) | max|v|/U (series / final) | KE max rel. jump | dead | overflow in/inc | drift | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 1.25e-03 | 0.0064 | 0.9997 | 0.9985 | 0.865 / 0.821 | 0.063 | 0 | 0/0 | 0 | 0 |
| 3.50 | 9.61e-04 | 0.0069 | 0.9996 | 0.9965 | 0.828 / 0.828 | 0.056 | 0 | 0/0 | 0 | 0 |
| 3.00 | 7.73e-04 | 0.0086 | 0.9984 | 0.9903 | 0.832 / 0.832 | 0.047 | 0 | 0/0 | 0 | 0 |
| 2.50 | 5.28e-04 | 0.0095 | 0.9981 | 0.9756 | 0.824 / 0.824 | 0.039 | 0 | 0/0 | 0 | 0 |
| 2.00 | 3.75e-03 | 0.2039 | 0.6563 | 0.0444 | 7.519 / 3.930 | 0.202 | 0 | 0/17328 | 0 | 0 |

## Per-kernel ns per particle (sync loop, warmup 1000 + 2000 timed steps)

| h/dx | predict | update_voxel | correction | density_kernel | density_copy | force | step |
|---|---|---|---|---|---|---|---|
| 4.00 | 0.021 | 0.028 | 2.569 | 2.821 | 0.036 | 3.142 | 8.616 |
| 3.50 | 0.022 | 0.027 | 1.987 | 2.125 | 0.036 | 2.545 | 6.743 |
| 3.00 | 0.021 | 0.018 | 1.450 | 1.579 | 0.036 | 1.774 | 4.878 |
| 2.50 | 0.023 | 0.019 | 1.006 | 1.350 | 0.036 | 1.295 | 3.729 |
| 2.00 | 0.018 | 0.013 | 0.978 | 1.024 | 0.035 | 0.619 | 2.686 |

## Checks per case

- h/dx 4.00: PASS
- h/dx 3.50: PASS
- h/dx 3.00: FAIL (l2_u_ok)
- h/dx 2.50: FAIL (l2_u_ok, l2_v_ok)
- h/dx 2.00: FAIL (no_close_pairs, rho_std_within_1.5x, l2_u_ok, l2_v_ok, conserved, no_divergence)

**Recommendation: h/dx = 3.5, 1.24x faster per unit physical time than h/dx = 4 (194.0 s vs 241.2 s for T = 1.20 s).**
