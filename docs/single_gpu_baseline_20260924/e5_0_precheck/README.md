# E5.0 — pre-check: V0 loads and runs the 3-D case

Question: can `utils.sph.case.load_case` + V0's pre-recorded step cmd (`utils/sph/simulator.py`)
load and run `cases/cavity3d_8m` (201³ lattice, 8,120,601 fluid + 2,019,249 wall + 363,609 lid =
10,503,459 particles)? Pass criterion: 200 steps, drift = 0.

Result: **PASS** → E5.1 runs both the 2-D and the 3-D column. No V0 code was changed for this.

| item | value |
|---|---|
| code | commit `e778a14` (V0 = `utils/sph` + `shaders/sph`, unchanged since `250751e`) |
| GPU | headless RTX 5090, V0 raw device index 2 (nvidia-smi index 1, PCI 03:00.0, uuid ae137c66…) |
| steps | 100 warmup + 100 measured (200 total), 1 frame in flight, validation off |
| alive | 10,503,459 = expected, drift 0, overflow 0 |
| VRAM | 1814 MB set-0/1/3 buffers + 1613 MB defrag scratch |
| fps | 12.23 (cold, 100 measured steps; not a benchmark number) |

Command:

```
VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \
.venv/Scripts/python.exe _run_single_baseline_bench.py --solver v0 --in-flight 1 \
    --case cases/cavity3d_8m/case.yaml --device 2 --expect-gpu "RTX 5090" \
    --warmup 100 --measure 100 --obj-cache logs/_obj_npy_cache --tag e5.0_precheck
```

Log: `v0_cavity3d_8m_200steps.log` (the `RESULT {...}` line at the end is the machine-readable record).
