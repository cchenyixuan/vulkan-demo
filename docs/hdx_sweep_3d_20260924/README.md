# E6 — 3-D kernel support radius h/Δx sweep (V0 path, code e778a14)

**One-line result: h/Δx = 3.5 is the smallest value that passes every E6.3 criterion; it is 1.24× faster per unit
physical time than h/Δx = 4 (194.0 s vs 241.2 s for T = 1.2 s). h/Δx = 3.0 and 2.5 are numerically healthy
(fallback 0, no close pairs, ρ std lower than the reference, W_sum ≥ 0.99, conservation exact) and would give
1.50× / 1.65×, but their centerline deviation from h/Δx = 4 exceeds 2× the 4-vs-3.5 baseline. h/Δx = 2.0
collapses by particle pairing and fails outright.**

## Setup

| item | value |
|---|---|
| solver | V0 (`utils/sph` + `shaders/sph`), commit `e778a14`; production code unchanged, hooks live in `_run_hdx_sweep_3d.py` |
| GPU | headless RTX 5090 (V0 raw index 2, nvidia-smi index 1, uuid ae137c66…), driver 576.88, validation off, sync loop (1 frame in flight) |
| cases | `cases/hdx_sweep_3d/hdx{4.0,3.5,3.0,2.5,2.0}/` from `utils/geometry/_demo_cavity_case_3d.py --half 50 --hdx H --border 4 --target-time 1.2` (yaml copies in `cases/`): 101³ fluid = 1,030,301 + 223,924 wall (4 layers) + 40,804 lid = 1,295,029 particles; L = 1 m, Δx = 0.01, c₀ = 100, γ = 7, CFL = 0.15, ν = 10⁻³ (Re = 1000), same as cavity3d_8m except the 4-layer shell (cavity3d_8m was built with 2·hdx+1 = 9 layers; 4 ≥ h/Δx covers one support radius for every case) |
| derived per h | Δt = CFL·h/c₀; voxel edge = h; max_per_voxel = ceil(1.3 × ceil(√2 (h/Δx)³)) rounded up to 32 → 128/96/64/32/32; max_incoming = max_per_voxel/4 (min 8) → 32/24/16/8/8; ε_h² = 0.01h²; δ = 0.1 and PST coefficients unchanged |
| protocol | T = 20,000 × Δt(h=4) = 1.2 s; steps 20,000 / 22,858 / 26,667 / 32,000 / 40,000; one full run per h with status + buffer readback every 1000 steps (`hdxH/run.jsonl`), final snapshot at t = T (`snapshots/hdxH.npz`, verifier layout, local only — 125 MB, gitignored); separately `_run_kernel_breakdown.py --solver v0` (warmup 1000 + 2000 timed) per h (`hdxH/breakdown.json`, `kernel_breakdown_results.jsonl`) |
| card | SM 2812–2955 MHz (mean 2855), 550 W mean, ≤ 70 °C (`telemetry.csv`) |
| conservation | drift = 0 and overflow = 0 on 4.0 / 3.5 / 3.0 / 2.5; h/Δx = 2.0: drift 0 but 17,328 incoming-list overflows |

## Main table (`summary.md` has the secondary metrics and the per-kernel ns/particle)

| h/Δx | neighbors | Δt (s) | steps to T | µs/step | s per T | speed-up | fallback | ρ std/ρ₀ | min spacing/Δx | pairs < 0.5Δx | u(y) L2 | v(x) L2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.0 | 268 | 6.00e-5 | 20,000 | 12058 | 241.2 | 1.00× | 0 | 1.20e-5 | 0.865 | 0 | — | — | PASS (ref.) |
| 3.5 | 180 | 5.25e-5 | 22,858 | 8487 | 194.0 | 1.24× | 0 | 9.53e-6 | 0.902 | 0 | 0.07 % | 0.01 % | **PASS** |
| 3.0 | 113 | 4.50e-5 | 26,667 | 6024 | 160.6 | 1.50× | 0 | 6.87e-6 | 0.917 | 0 | 0.21 % | 0.01 % | FAIL (u L2 > 2× baseline) |
| 2.5 | 65 | 3.75e-5 | 32,000 | 4576 | 146.4 | 1.65× | 0 | 4.16e-6 | 0.927 | 0 | 0.21 % | 0.05 % | FAIL (u, v L2 > 2× baseline) |
| 2.0 | 34 | 3.00e-5 | 40,000 | 3537 | 141.5 | 1.70× | 0 | 2.04e-5 | 0.000 | 217,503 | 9.76 % | 2.46 % | FAIL (pairing, overflow, divergence) |

L2 = RMS over 201 mid-plane samples, normalised by U. Baseline (4 vs 3.5): u 0.07 %, v 0.01 % of U.

Per-kernel ns per particle (step): 8.62 (4.0) → 6.74 (3.5) → 4.88 (3.0) → 3.73 (2.5) → 2.69 (2.0); the three
neighbor kernels scale with the neighbor count, predict/update_voxel/copy are flat (0.02–0.04).

## Reading the verdict

- The accuracy criterion is what separates 3.5 from 3.0/2.5, and it is relative to a very small baseline: at
  T = 1.2 s (1.2 lid transits) the cavity flow is still spinning up (kinetic energy rising ~3 % per 1000 steps at
  the end), the interior velocity is only 0.02–0.04 U, and the reference profile RMS is 0.106 U (u) / 0.016 U (v).
  Signal-relative, the deviations are: 3.5 → 0.7 % of u RMS; 3.0 → 2.0 %; 2.5 → 1.9 % (u) and 3.3 % (v). All
  four are below the absolute 2 % U bound by an order of magnitude. Smaller h also dissipates less (fluid kinetic energy at T:
  6.80 / 7.20 / 7.83 / 8.88 J for 4.0 / 3.5 / 3.0 / 2.5, still rising 2.7 / 2.0 / 1.3 / 1.7 % per 1000 steps at
  the end), which is the physical origin of the drift in the profiles.
  If the paper accepts "≤ 2 % of U" alone, 2.5 is the cut; with the stricter "≤ 2 × baseline" rule, 3.5.
- h/Δx = 2.0 fails within the first 1000 steps (max|v| 2.9 U, KE 5× the reference at the same time). The failure is
  particle pairing: min spacing → 0, 217,503 pairs closer than 0.5Δx, interior W_sum 1 % quantile 0.04, interior
  p std 0.20 ρ₀U² (vs 0.006–0.01 for the others). It is not a capacity artifact: `diag_hdx2.0_inc32/` reruns
  h/Δx = 2.0 with max_incoming 32 / max_per_voxel 64 for 3000 steps — zero overflow, same collapse
  (219,685 close pairs, W_sum 1 % = 0.05, max|v| 4.5 U). The incoming overflow in the main run is a symptom.
- correction_fallback_count stayed 0 for every h, so the KCG matrix never hit the regularisation fallback even at
  34 neighbors; the pairing at 2.0 is a kernel-sampling instability, not an ill-conditioning one.
- 3.0 passed → no 2.75 case was needed (E6.3 asks for 2.75 only if 3.0 passes and 2.5 fails; 3.0 did not pass).

## Commands

```
# cases (E6.0)
.venv/Scripts/python.exe utils/geometry/_demo_cavity_case_3d.py --half 50 --hdx H --border 4 --target-time 1.2 \
    --out cases/hdx_sweep_3d/hdxH --no-preview          # H in 4.0 3.5 3.0 2.5 2.0
# runs (E6.1): VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
.venv/Scripts/python.exe _run_hdx_sweep_3d.py --case cases/hdx_sweep_3d/hdxH/case.yaml --steps N --record-every 1000 \
    --device 2 --out-dir docs/hdx_sweep_3d_20260924/hdxH --snapshot docs/hdx_sweep_3d_20260924/snapshots/hdxH.npz --tag hdxH
.venv/Scripts/python.exe _run_kernel_breakdown.py --solver v0 --device 2 --case cases/hdx_sweep_3d/hdxH/case.yaml \
    --warmup 1000 --measure 2000 --tag hdxH
# analysis (E6.2-E6.4)
.venv/Scripts/python.exe _analyze_hdx_sweep_3d.py --dir docs/hdx_sweep_3d_20260924 --fig manuscripts/fig/hdx_sweep_3d
```

Files: `hdxH/{run.jsonl,result.json,run.log,breakdown.json,breakdown.log}`, `kernel_breakdown_results.jsonl`,
`snapshots/hdxH.npz` (local), `profiles.csv` (u(y), v(x) at 201 points per h), `summary.md` / `summary.json`
(tables, checks, verdict), `hdx_sweep_3d.png`, `telemetry.csv`, `cases/*.case.yaml`, `diag_hdx2.0_inc32/`.
Figure for the paper: `manuscripts/fig/hdx_sweep_3d.pdf` (+ `.png`): (a) s per T and (b) µs/step vs h/Δx,
(c) ρ std and (d) fallback vs h/Δx, (e) u(y) and (f) v(x) centerlines, five curves.
