### E9 local validation (GPU 1, 575 W, the E9 tree of the e8 worktree: prefix defrag order as the production path)

#### item 4: K = 1, 300 steps (window 100-299), pid-order audit after the bootstrap defrag

| run | defrag order | step (ms) | ns / particle | corr+dens (ms) | force (ms) | 7.18 prefix step (ms) | diff | audit (bootstrap defrag) | drift |
|---|---|---|---|---|---|---|---|---|---|
| e9_8m_k1_audit | prefix | 46.250 | 5.781 | 23.040 | 22.147 | 46.325 | -0.16 % | 125,000 blocks = 125,000 voxels, 0 descents, 0 without voxel -> EXACT | 0 |
| e9_64m_k1_audit | prefix | 375.400 | 5.866 | 183.729 | 183.092 | 375.500 | -0.03 % | 1,000,000 blocks = 1,000,000 voxels, 0 descents, 0 without voxel -> EXACT | 0 |
| e9_8m_k1 | prefix | 46.400 | 5.800 | 23.107 | 22.230 | 46.325 | +0.16 % | - | 0 |
| e9_64m_k1 | prefix | 375.300 | 5.864 | 183.632 | 183.042 | 375.500 | -0.05 % | - | 0 |

#### item 5: K = 2 on one card (8M cube, --device-map 1,1), 300 steps

- `e9_8m_k2_trace`: 55.500 ms/step (18.02 fps, two slabs on one card), drift 0, overflow_total 0, seam check OK; audits (bootstrap defrag, both slabs, then the end-of-run defrag): EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents)
  - [chain_v7] seam 0 (col 26): L_overshoot=-0.19dx R_overshoot=-0.46dx dup=0 OK

field comparison K = 2 vs K = 1 at step 300 (`e8_field_compare.py`, phase-1 format):

K = 2 (cuts [26], devices [1, 1]) vs K = 1 at step 300: 7,077,888 fluid particles compared (ids missing in the test 0, in the reference 0); max |dv|/U = 3.943e-05, rms 2.480e-06; max |drho|/rho0 = 1.831e-07, rms 2.052e-08; max |dx| 1.156e-05 m

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 294,912 | 1.941e-05 | 4.317e-06 | 1.221e-07 | 2.693e-08 | 2.772e-06 |
| 1 | 294,912 | 1.840e-05 | 4.174e-06 | 1.221e-07 | 2.757e-08 | 2.625e-06 |
| 2 | 294,912 | 1.873e-05 | 3.511e-06 | 1.221e-07 | 2.490e-08 | 2.594e-06 |
| 3 | 294,912 | 1.648e-05 | 2.767e-06 | 1.221e-07 | 2.228e-08 | 2.209e-06 |
| 4 | 294,912 | 1.665e-05 | 2.290e-06 | 1.221e-07 | 2.069e-08 | 2.623e-06 |
| 5 | 294,912 | 1.408e-05 | 2.092e-06 | 1.221e-07 | 2.016e-08 | 2.385e-06 |
| 6 | 294,912 | 1.276e-05 | 1.967e-06 | 1.831e-07 | 1.970e-08 | 2.297e-06 |
| 7 | 294,912 | 1.755e-05 | 1.856e-06 | 1.221e-07 | 1.933e-08 | 3.398e-06 |
| 8 | 294,912 | 2.032e-05 | 1.851e-06 | 1.221e-07 | 1.913e-08 | 3.249e-06 |
| 9 | 294,912 | 2.013e-05 | 1.762e-06 | 1.221e-07 | 1.895e-08 | 3.428e-06 |
| 10 | 294,912 | 2.054e-05 | 1.744e-06 | 1.831e-07 | 1.909e-08 | 2.777e-06 |
| 11 | 294,912 | 1.438e-05 | 1.706e-06 | 1.831e-07 | 1.923e-08 | 4.831e-06 |
| 12 | 294,912 | 2.348e-05 | 1.762e-06 | 1.221e-07 | 1.910e-08 | 4.503e-06 |
| 13 | 294,912 | 2.563e-05 | 1.825e-06 | 1.221e-07 | 1.884e-08 | 4.232e-06 |
| 14 | 294,912 | 2.349e-05 | 1.990e-06 | 1.221e-07 | 1.894e-08 | 4.984e-06 |
| 15 | 294,912 | 3.288e-05 | 2.165e-06 | 1.831e-07 | 1.937e-08 | 5.254e-06 |
| interior | 2,359,296 | 3.943e-05 | 2.426e-06 | 1.831e-07 | 1.934e-08 | 1.156e-05 |

| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |
|---|---|---|---|---|---|---|
| 0 | 1 | 3,538,944 | 2.269e-05 | 2.127e-06 | 1.831e-07 | 1.839e-08 |
| 1 | 1 | 3,538,944 | 3.943e-05 | 2.789e-06 | 1.831e-07 | 2.245e-08 |

- `e9_8m_k2_cad100` (defrag cadence 100, periodic defrags at frames 100 and 200 inside the run): step 70.450 ms, drift 0, seam check OK, defrag events [('100', '1088.521'), ('200', '1082.605'), ('300', '1081.697')] (frame, wall ms); audits ['EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT']

| window (steps) | median step period (ms) |
|---|---|
| 50-99 | 54.799 |
| 101-149 | 55.067 |
| 150-199 | 54.818 |
| 201-249 | 54.378 |
| 250-299 | 55.109 |
| first step after each defrag | f100+1: 54.866, f200+1: 54.916, f300+1: - |
- before the first periodic defrag (50-99) 54.799 ms, after the second (201-249) 54.378 ms: -0.77 % -> no jump

#### item 6: 2-D sensitivity, K = 1, 1300 steps (window 300-1299, one periodic defrag at 1000 inside; fps = steady_fps_nodefrag)

| case | particles | atomic (ms/step) | prefix (ms/step) | prefix - atomic | atomic audit | prefix audit |
|---|---|---|---|---|---|---|
| cavity2d_n1000 (1M) | 1,044,484 | 1.3621 | 1.3509 | **-0.82 %** | 642 descents | 0 descents |
| cavity2d_n2840 (8M) | 8,191,044 | 10.8929 | 10.2980 | **-5.46 %** | 5,015 descents | 0 descents |