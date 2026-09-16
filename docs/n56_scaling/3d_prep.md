# 3-D case preparation for the N56 cluster (Task B) — 2026-09-16

Scope: bring the locally validated 3-D lid-driven cavity to a cloud-ready state under the standardized
efficiency protocol (docs/n56_scaling, 2026-09-15). Everything below was measured on the local 2× RTX 5090
(Windows, PCIe x8/x8) unless stated. **No cloud job was submitted for 3-D.**

## B.3 — quantities checked

| item | finding | action |
|---|---|---|
| **h/dx** | In this solver `h` *is* the support radius (kernels cut at `distance >= SMOOTHING_LENGTH`; voxel edge = h). 3-D cases: h = 4 dx (cavity3d_*: h 0.02, dx 0.005) → neighbours ≈ (4/3)π·4³ ≈ **268** (2-D: π·4² ≈ 50). The "2h = 8 dx → 2000 neighbours" concern does not apply: there is no 2h convention here. | keep h/dx = 4 |
| **voxel slot capacity** | 3-D cases use `max_per_voxel 128`, `max_incoming 32` (2-D: 96/16). At rest 64 particles per voxel (4³); closest-packing bound printed by the generator = 91. Measured 3-D K=2 8M: own `pool_used` 83% of a 1.2× pool, `overflow_ghost 0`, `drops 0`. | keep 128/32 |
| **band definition in 3-D** | `in_boundary_band`, `band_thread_voxel/particle` (face = NY·NZ), partition (x slabs) and the verifier's column distance all use the x index only → an x "column" is a y–z **plane** in 3-D; no hidden 2-D assumption. Proven by the 3-D K=2 verifier below. | none |
| **wall layers** | Generator default `border = 2·hdx+1 = 9` layers (validated 3-D cases). Wall + lid fraction of the fluid count: 2M cube 49%, 4M cube 38%, 8M cube 29%, 8M stretched (319×159×159) 31%; with `--border 4` (one support radius) ≈ 15% / 13% / 12% / 13%. Convention as in 2-D: cases are named by fluid count, `alive` totals include walls. | decide border for the cloud family (recommend 4 for scaling; 9 for parity with validated cases) |
| **ghost pool factor** | Measured 3-D K=2 (4M/GPU stretched): 125,316 ghosts per direction per frame = **17.5 MB per link per frame** (140 B/ghost). Ghost pool per side = face voxels × (128+32) slots: 45² × 160 = 324k → live/pool = **0.39**. The 2-D factor 0.25 (81k slots) would overflow. | use **1.0** (0.5 is the floor) |
| **speed of sound / dt** | c = 100, γ = 7, CFL 0.15 as in 2-D; dt = CFL·h/c. Validation cases at lower c are a separate track. | keep |
| **bytes per particle** | device-local + defrag-scratch buffers vs pool slots: 2M 994 MB / 3.51M, 4M 1808 MB / 6.38M, 8M 3427 MB / 12.08M → **284 B per pool slot** (no new fields in 3-D; vec3 already stored as vec4). Per *total* particle (pool = 1.15 × total) ≈ 326 B; per *fluid* particle with border-9 walls ≈ 420 B. | N_max per 32 GB card ≈ **105M pool slots ≈ 90M total particles** → ≈ 70M fluid (border 9) / ≈ 80M fluid (border 4) |

### Cube vs stretched sizing (analytic, `scripts/analyze_3d_prep.py`; band cost ratio from measurement = 1.4×)

| geometry | K | fluid N | walls (border 9 / 4) | slab width dx (voxels) | halo per side | ghost MB / link / frame | force-band share of slab | measured or est. C/B | ghost pool factor needed → use |
|---|---|---|---|---|---|---|---|---|---|
| stretched 8M/GPU (1608×201×201) | 2, 4, 8 | 16–65M | 19–27% / 8–12% | 201 (50) | 2.0% | 26.9 | 16% | est. 0.20 (1.4× band cost) | 0.40 → 1.0 |
| stretched 4M/GPU (1272×159×159) | 2, 4, 8 | 8–32M | 24–34% / 11–15% | 159 (40) | 2.5% | 17.5 (measured) | 20% | **0.115 measured (K=2)** | 0.39 → 1.0 |
| cube 400³ (64M) | 8 | 64.5M | 13% / 6% | 50 (12.5) | 8.0% | 98 | 64% | est. 1.4–2 (cascade window collapses) | 0.40 → 1.0 |
| cube 318³ (32M) | 8 | 32.5M | 17% / 8% | 40 (10) | 10% | 64 | 80% | est. 2.4–3.4 | 0.39 → 1.0 |
| 2-D 64M K=8 for reference | 8 | 64.4M | 1% | 205 columns | 0.5% | 5.6 | 3.9% | 0.09 measured | 0.25 (live/pool 0.22) |

Reading: the stretched families keep the halo at 2–2.5% per side and phase C at 10–20% of phase B, so the
cascade window still hides the transport comfortably (transport chain ≈ 17–27 MB per link per frame ≈ 5–7 ms at
the measured ~4 GB/s host hop vs phase B ≈ 40–90 ms). The cube at K=8 puts 64–80% of each slab inside the force
band; it is the "1-D slab in a cube" limit and is only worth a single K=8 control point, as planned.

## B.4 — local experiments (2× RTX 5090)

### K=1 (chain bench, 3000 steps, steady = last 2000, `--anatomy`)

| case | fluid | total (walls incl.) | fps | phase A / B / C (ms) | buffers (device + defrag scratch) |
|---|---|---|---|---|---|
| cavity3d_2m (127³) | 2.05M | 3.05M | 41.3 | 0.21 / 25.1 / 0.11 | 526 + 468 MB |
| cavity3d_4m (159³) | 4.02M | 5.55M | 22.3 | 0.45 / 46.1 / 0.19 | 957 + 851 MB |
| cavity3d_8m (201³) | 8.12M | 10.50M | 11.6 | 0.90 / 88.5 / 0.36 | 1814 + 1613 MB |

Per-particle cost ≈ 8.4 ns (total particles) vs 1.57 ns in 2-D: the 5.3× 3-D cost factor of the earlier 3-D
readiness note is reproduced.

### K=2, 4M/GPU stretched (cavity3d_weak4_k2_8m = 319×159×159 dx, 8.06M fluid, 10.56M total), ghost pool 1.0

| check | result |
|---|---|
| seam check (chain bench 3000 steps) | total 10,557,873 conserved, drift 0, stamp errors 0, seam overshoot 0.00 dx, dup 0, ρ ∈ [999.0, 1001.2], v_max 1.000 |
| equivalence battery (K=1, K=1, K=2, K=2; 2000 steps) | ALL PASS (K=2 within the K=1 run-to-run envelope on every metric) |
| per-particle verifier (lanes 0 vs 32, 300 steps) | PASS, worst column ratio 1.11, band invariant OK, 0 unmatched of 5.2M + 5.36M |
| fps (3 runs, lanes 0) | 21.1 / 22.1 fps steady (η vs K=1 4M cube 22.3 fps ≈ 95–99%; the stretched slab is not the cube, so this is indicative only) |
| anatomy (per GPU, f3000) | A 0.70 / B 40.2–41.6 / C 4.64 ms; C = correction_boundary 0.98 + density_boundary(+copy) 1.65 + force_boundary 2.0; interior: correction 12.7–13.8, density 12.6–13.3, force 14.6–15.1 |
| **C / B** | **0.115** (2-D 64M K=8: 0.09) |
| band per-particle cost vs interior | force: band 2.0 ms for ~10% of the slab vs deep 14.6 ms for ~90% → **≈ 1.4×** (2-D: ≈ 2×) |
| transport slack | upload landed 15–34 ms before Phase C (phase B ≈ 41 ms) → hidden with a wide margin |

### Band-kernel lane packing A/B (V3.8, `V5_BAND_SLOT_LANES`)

| lanes | correction_boundary | density(+copy) | force_boundary | phase C | steady fps |
|---|---|---|---|---|---|
| 0 (one thread per slot, V3.4) | 0.97–0.99 ms | 1.64–1.67 | 1.99–2.02 | 4.64 | 21.1 / 22.1 |
| 32 | 1.29–1.34 | 2.06–2.11 | 2.43–2.50 | 5.8–5.96 | 21.6 |
| 64 | 1.17–1.22 | 1.79–1.86 | 2.01–2.21 | 4.98–5.29 | 22.0 |

Lane packing makes every band kernel slower (fewer threads: 130–260k, below what the GPU needs to hide
latency); fps differences are within run-to-run noise. The lane-utilization hypothesis for the 1.4–2× band
per-particle cost is rejected; keep `V5_BAND_SLOT_LANES=0`. The remaining candidates are ghost-slot cache
misses for column-0 neighbours and the gather through `inside_particle_index`; neither was tested.

### Regression proof for the shared shader refactor

The three boundary-capable kernels were restructured (`main()` → `process_particle(pid)` + a new `main()`);
the per-particle verifier compared the committed shader build (git HEAD .spv via `V5_SPV_DIR`) against the
refactored build on 2-D 1M K=4 (300 steps): PASS, worst ratio 1.86, band invariant OK; 2-D 1M K=2 seam checks
clean at 659 / 673 fps. The 2-D path is unchanged in behaviour.

## B.5 — proposed cloud matrix (needs approval; nothing submitted)

Geometry: stretched families from `gen_3d_families.sh` (uploaded to `~/run/tools`, **not run**);
K=8 weak geometry doubles as the fixed-N strong geometry.

| block | cases | K | reference | trials | est. node time |
|---|---|---|---|---|---|
| strong 3-D, 64M (8M/GPU) | cavity3d_weak8_k8_64m (1608×201×201) | 2, 4, 8 | K=2 pairs on the participating GPUs (76M total incl. walls does not fit one 32 GB card at border 9; ~66M at border 4 is borderline) | 3 | refs 3 × (8M/GPU K=2 ≈ 5.5 min) + runs → ≈ 60 min |
| strong 3-D, 32M (4M/GPU) | cavity3d_weak4_k8_32m (1272×159×159) | 2, 4, 8 | K=1 on participating GPUs (39M total fits) | 3 | ≈ 45 min |
| cube control | cavity3d_cube_64m (401³) | 8 only | 8 × K=1? does not fit (73M total) → K=2 pairs | 3 | ≈ 25 min |
| weak 3-D, 4M/GPU | cavity3d_weak4_k{1,2,4,8} | 1, 2, 4, 8 | K=1 on all 8 GPUs simultaneously | 3 | ≈ 35 min |
| weak 3-D, 8M/GPU | cavity3d_weak8_k{1,2,4,8} | 1, 2, 4, 8 | same | 3 | ≈ 60 min |

Total ≈ 3.8 node-hours plus ~1.5 h of login-node case generation (64M stretched ≈ 15 min, 401³ ≈ 15 min).
Frame times: 8M/GPU 3-D ≈ 11.6 fps single → 3000 steps ≈ 4.3 min per run; the K=8 64M 3-D run ≈ 5 min plus
~3 min case load. Recommended switches: `V5_GHOST_POOL_FACTOR=1.0` (3-D), otherwise identical to the 2-D curve
jobs; `V5_BAND_SLOT_LANES=0`.

Open decisions before submission: wall border (9 vs 4) and whether the 64M reference uses K=2 pairs at border 9
or K=1 at border 4.

## Files

- generator: `utils/geometry/_demo_cavity_case_3d.py` (`--half-x`, x-chunked, box frame; cubic output bit-identical to before)
- families: `docs/n56_scaling/scripts/gen_3d_families.sh` (`BORDER` env; markers `cases/.cavity3d_*_ready`)
- sizing: `docs/n56_scaling/scripts/analyze_3d_prep.py`
- local logs: scratchpad `b3d/` (K=1 anatomy logs, soak3d_k2), `logs/verify3d_lanes_8m_k2/`, `logs/verify_lanes_regress_1m_k4/`, `logs/verify_lanes32_1m_k4/`
