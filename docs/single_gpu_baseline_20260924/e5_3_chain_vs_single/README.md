# E5.3 — V5 chain mode K=1 vs V5 single-GPU mode, same card, 2-D 8M (for Sec. 5)

Question: does the chain (phased A/B/C, timeline-semaphore, depth-2 pipelined) orchestration cost anything
when it runs with a single slab on one GPU, compared with the plain single-GPU mode (one combined step cmd,
fence ring, 2 frames in flight)?

**Answer: chain K=1 / single = 99.9 % (mean of trial-wise ratios 99.91 ± 0.59 %, ratio of means 99.90 %).
Within the expected ≤ 0.5 %, so no `--anatomy` run was needed.**

| trial | single mode (fps) | chain K=1 steady (fps) | chain / single |
|---|---|---|---|
| 1 | 71.705 | 71.4 | 99.57 % |
| 2 | 71.380 | 71.8 | 100.59 % |
| 3 | 71.318 | 71.0 | 99.55 % |
| mean ± std | 71.47 ± 0.21 | 71.40 ± 0.40 | 99.91 ± 0.59 % |

All six runs: alive = 8,128,201 = expected, drift 0, stamp errors 0. The chain bench prints its steady fps
with one decimal (±0.07 %), which is below the trial-to-trial spread.

Setup: commit `e778a14` (V5 kernels with the loop-order fix `f9c660e`), one headless RTX 5090 (V5 device
index 1 = nvidia-smi index 1, PCI 03:00.0), validation off, `V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1`
(the cluster settings; both are also the defaults). Case `cases/lid_driven_cavity_2d_8m` (8,128,201
particles). Both sides: 3000 steps, the last 2000 timed (single: warmup 1000 + measure 2000 with an empty
queue at both ends; chain: `STEADY (post-warmup 1000)` window of `orch.run_pipelined`). Trials interleaved
on the same card with the order rotated (t1 single→chain, t2 chain→single, t3 single→chain).
Card inside the runs (`telemetry.csv`, index 1): SM 2700–2955 MHz (mean 2739), 595 W mean, max 71 °C.

## Commands

```
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1
# (a) single-GPU mode, 2 frames in flight
.venv/Scripts/python.exe _run_single_baseline_bench.py --solver v5 --in-flight 2 \
    --case cases/lid_driven_cavity_2d_8m/case.yaml --device 1 --expect-gpu "RTX 5090" \
    --warmup 1000 --measure 2000 --obj-cache logs/_obj_npy_cache --tag e5.3_single --trial <t>
# (b) chain mode, K=1 on the same card, depth 2
.venv/Scripts/python.exe experiment/v5/_run_v5_chain_bench.py --case cases/lid_driven_cavity_2d_8m/case.yaml \
    --weights 1 --device-map 1 --depth 2 --pool-safety 1.2 --max-steps 3000 --warmup 1000
```

Driver script and log parser: `_tools/run_e5_3.sh`, `_tools/e5_3_parse_chain.py`. Files: `results.jsonl`
(3 single records with `solver: v5`, 3 chain records with `kind: chain_k1`), `single_t*.log`,
`chain_k1_t*.log`, `telemetry.csv`.
