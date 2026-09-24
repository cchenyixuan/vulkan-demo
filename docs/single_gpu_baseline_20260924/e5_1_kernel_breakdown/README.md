# E5.1 — V0 per-kernel cost (Tab kernelcost), 2-D and 3-D, density kernel and copy separated

Solver: **V0 reference** (`utils/sph/simulator.py` + `shaders/sph/`, commit `e778a14`; V0 code unchanged since
`250751e`). One headless RTX 5090 (V0 raw device index 2 = nvidia-smi index 1, PCI 03:00.0), validation off,
sync loop (1 frame in flight), warmup 1000 + 2000 timed steps per run, defrag frames (every 1000 steps)
excluded, BOTTOM_OF_PIPE timestamps after every stage. 3 independent runs (separate processes) per case,
2-D and 3-D interleaved: 2-D run1, 3-D run1, 2-D run2, ...

Cases: 2-D `cases/lid_driven_cavity_2d_8m` (8,128,201 particles incl. walls), 3-D `cases/cavity3d_8m`
(201³ lattice, 10,503,459 particles incl. walls; E5.0 pre-check passed, see `../e5_0_precheck/`).

## How the density kernel / copy split is measured

V0 has no timestamp facility of its own (the V5 `_bench_tick` mechanism does not exist in `utils/sph`), so
instead of inserting a tick into V0's production step cmd, `_run_kernel_breakdown.py` records its own copy
of V0's exact step sequence through V0's private recorder helpers (`_record_compute_barrier`,
`_bind_pipeline_and_sets`, `_record_density_scratch_to_primary_copy`, the same dispatch counts) and writes a
timestamp after every stage: leading barrier | predict | update_voxel | correction | **density kernel** |
**scratch→primary copy** | force. The "density_kernel" and "density_copy" columns are therefore exactly
the `density_kernel_end` / `density_end` pair the task describes, without modifying V0. The V0 production cmd
itself is untouched (it is what `_run_single_baseline_bench.py` times).

## Result (mean over 3 runs of the per-run mean; p50 in `summary.md`)

| kernel | dispatch | 2-D µs/step | 2-D ns/particle | 2-D % | 3-D µs/step | 3-D ns/particle | 3-D % |
|---|---|---|---|---|---|---|---|
| predict | per particle | 494 | 0.06 | 3.5 | 507 | 0.05 | 0.6 |
| update_voxel | per voxel | 336 | 0.04 | 2.4 | 360 | 0.03 | 0.4 |
| correction | per particle | 3898 | 0.48 | 27.6 | 26840 | 2.56 | 31.4 |
| density (kernel) | per particle | 3946 | 0.49 | 28.0 | 27100 | 2.58 | 31.8 |
| density (scratch→primary copy) | buffer copy | 275 | 0.03 | 2.0 | 356 | 0.03 | 0.4 |
| force | per particle | 5150 | 0.63 | 36.5 | 30186 | 2.87 | 35.4 |
| **step total** | | **14100** | **1.73** | 100 | **85350** | **8.13** | 100 |

ns/particle = µs/step × 1000 / alive particles (incl. walls). The LaTeX rows are in `summary.md`.

## Acceptance

| check | 2-D | 3-D |
|---|---|---|
| drift = 0 on every run | yes (3/3) | yes (3/3) |
| std of the step total over 3 runs ≤ 1 % | 0.21 % (14070 / 14131 / 14098 µs) | 0.06 % (85405 / 85346 / 85297 µs) |
| 2-D step total vs 9/23 V0 8M reference 14069.7 µs, ≤ 1 % | +0.21 % | — |

Card state inside the runs (nvidia-smi 1 Hz, `telemetry.csv`, index 1): SM 2707–2955 MHz (mean 2717),
598 W mean (600 W limit), max 70 °C. CPU-side throughput of the same runs: 69.9 fps (2-D), 11.64 fps (3-D).

## Commands

```
VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
for run in 1 2 3; do for case in lid_driven_cavity_2d_8m cavity3d_8m; do
  .venv/Scripts/python.exe _run_kernel_breakdown.py --solver v0 --device 2 \
      --case cases/$case/case.yaml --warmup 1000 --measure 2000 \
      --obj-cache logs/_obj_npy_cache --tag $case --run $run > ${case}_run$run.log
  # the RESULT {...} line of each log is appended to results.jsonl
done; done
.venv/Scripts/python.exe _summarize_kernel_breakdown.py --results results.jsonl \
    --reference-2d-us 14069.7 --out-md summary.md
```

Files: `results.jsonl` (6 records: per-stage mean/p50/std µs, alive, drift, cpu_fps, device, timestamp
period), `*_run*.log` (full stdout/stderr), `summary.md` (table, acceptance, LaTeX rows), `telemetry.csv`.
