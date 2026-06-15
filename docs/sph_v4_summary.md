# SPH V4 — Summary & Entry Point

> **Purpose of this doc:** a cold-start entry point for a new session/model picking up the V4
> multi-GPU SPH solver. It states what V4 is, what has been done & measured, where the bottleneck
> now is, and the single highest-value next step. Read this first, then `docs/sph_v2_design.md`
> (V4 inherits V2's architecture) and the `memory/` notes referenced at the bottom.

Last updated: 2026-06-15. Hardware: AMD Radeon RX 7900 XTX (device 0) + NVIDIA RTX 4060 Ti
(device 1), cross-vendor single machine. Case: `cases/lid_driven_cavity_2d/` (1M, 1,046,529 particles)
and `cases/lid_driven_cavity_2d_4m/` (4M, 4,182,025 particles).

---

## 0. TL;DR

- **V4 = a clean copy of the V2 "Path A+" architecture** (the fast CPU-worker + transfer-queue design),
  renamed `v2`→`v4`, plus four self-contained improvements (below). **V3's shared-host-transport
  architecture was measured ~7% slower and is NOT adopted** — do not build on `experiment/v3/`.
- **Orchestration is essentially optimized.** Strong-scaling efficiency: **1M 81.6%→89.6%**, **4M
  87%→92.2%**, all `drift=0`. The three orchestration levers (per-slab pool sizing + K_split
  rebalance + submit-ahead/depth-2 pipelining) are done and validated at both scales.
- **The remaining bottleneck is inside the compute kernels.** The transfer chain is only ~850µs
  (phase_b has ~700µs of headroom — NOT transfer-floored), and `ncu/nsys` profiling shows the
  neighbor kernels (correction/density/force) are **latency-bound** (irregular gather), not
  compute- or bandwidth-bound.
- **Highest-value next step:** reduce the per-neighbor *gather* — the biggest lever is **fusing the
  three neighbor traversals** (correction/density/force each independently re-gather the same
  neighbors) so each neighbor is loaded once and reused. Constraint: it must coexist with the
  Path A+ cascading split bands. First sub-task: assess fusing correction+density (both in phase_b).
- **Do NOT pursue:** the 2D mat3→2×2 inverse micro-opt (compute reduction; kernels aren't
  compute-bound — confirmed by profiling). Don't shrink ghost fields to speed the *compute* kernels
  (they aren't DRAM-bandwidth-bound; ghost shrink only helps the separate PCIe transfer chain).

---

## 1. Architecture recap (V4 = V2 Path A+)

Per-GPU 1D slab decomposition along x, 1-voxel ghost, voxel-sorted single own-particle SoA. δ-plus
WCSPH, leapfrog integration, persistent uniform voxel grid. Full design in `docs/sph_v2_design.md`.

**Per-frame timeline (5N timeline semaphore, one per sim):**
- `phase_a` (compute Q): predict → update_voxel → ghost_send. Signals `5N+1`.
- transfer Q: readback DMA (device→sender_staging). Signals `5N+2`.
- CPU worker thread: memcpy sender_staging→peer's receiver_staging. Host-signals `5N+3`.
- transfer Q: upload DMA (receiver_staging→device). Signals `5N+4`.
- `phase_b` (compute Q, queue-ordered after A): correction_interior + density_deep_interior.
  **Runs in parallel with the transfer chain to hide it.**
- `phase_c` (compute Q, waits `5N+4`): install_migrations → correction_boundary → density_boundary →
  density-scratch copy-back → force_all. Signals `5N+5` (frame done).

**Cascading interior/boundary split** (to hide the transfer chain behind phase_b): correction band=2
voxels, density band=3, force band=4 (force is NOT split in production — runs `force_all` in phase_c).

---

## 2. What was changed in V4 (vs the V2 copy)

All four are isolated to `experiment/v4/`; V2 stays frozen.

1. **set-3 buffer cleanup.** Removed 5 buffers that were declared but **never read/written by any
   kernel** (`inlet_template`, `dispatch_indirect`, `ghost_out_packet`, `ghost_in_staging`,
   `diagnostic`). Reclaimed binding 3,1 (the never-implemented `overflow_log`) → `PoolHealthBuffer`.
   (`maximum_velocity` in global_status is also dead but left in place to keep the 16-uint/64B
   cache-line layout + readback offsets stable.)

2. **Pool-health watermark** (`PoolHealthBuffer`, set 3 binding 1, 16B). Two never-reset `atomicMax`
   in `install_migrations.comp`: `peak_tail_high_water = max(slot_n+1+alive)` (closest the migrant
   tail ever got to `OWN_POOL_SIZE`) and `peak_migration_count = max(slot_n+1)`. Host reads via
   `simulator_v4.readback_pool_health()`. Gives data-driven pool sizing + a pre-overflow warning
   (warn before a particle is silently dropped). Migrants install at the own-pool tail descending and
   `migration_install_count` resets every defrag, so without this watermark the cross-run worst case
   is invisible.

3. **Per-slab pool sizing.** `partition_v4.compute_dual_gpu_partition(case, weights, pool_safety=None)`.
   Default `None` = legacy global whole-domain pool on both slabs (wastes empty-slot dispatch — NV
   owned 24% but scanned the full 1.2M pool). `pool_safety=1.2` sizes each slab `own_pool =
   ceil(slab_particles × 1.2)`, workgroup-rounded, capped at the global pool. **The pid-offset uses
   slot-0's shrunk pool** (offsets depend only on slot 0's `own_pool_size`), so AMD is safely
   shrinkable too. Self-contained: no shader changes.

4. **Submit-ahead pipelining + migration logging.** `orchestrator_v4.run_pipelined(max_steps, depth=2,
   warmup, on_defrag)` keeps `depth` frames in flight so the GPU never idles on the CPU submit/wait
   round-trip the synchronous `step()` pays every frame. **Safe with single buffers at any depth**:
   the 5N timeline makes worker(n)'s host-signal `5n+3` a prerequisite for frame n+1's readback `5n+7`,
   so no staging buffer is reused before its reader finishes and no host-signal goes backwards
   (validated drift=0 at depth 2 and 3). `step()` (depth-1) is kept for the instrumented bench — under
   pipelining the per-kernel GPU-timestamp slots get overwritten by the in-flight next frame.
   `_collect_defrag_report()` snapshots `migration_install_count` before each defrag for the
   per-interval migration time series.

---

## 3. Measured results (all drift=0, zero overflow)

### Throughput (cavity, steady-state)

| config | 1M fps | 4M fps |
|---|---:|---:|
| baseline (depth-1, no pool shrink, 3.2:1/2.8:1) | ~283 | ~79.3 |
| + per-slab pool + K_split rebalance (depth-1) | 292.9 | 80.7 |
| **+ submit-ahead (depth-2) — full V4 stack** | **310.9** | **83.6** |
| single AMD (warmed, full problem) | 232 | 61.2 |
| single NV (warmed, full problem) | 115 | 29.5 |

Optimal weights: 1M `2.9,1.0`, 4M `2.8,1.0` (hardware-pair-specific). Submit-ahead alone: +16.8 fps
@1M (+5.9%), +2.9 @4M (+3.6%) — smaller % at 4M because the ~200µs CPU bubble is a smaller fraction
of the ~12ms 4M frame. Pool+rebalance alone: +9.6 @1M.

### Strong-scaling parallel efficiency

`η_strong = dual_fps(P) / [single_AMD_fps(P) + single_NV_fps(P)]`, fixed total problem P, references =
warmed single-GPU runs on the **full** problem. (This is **strong** scaling — fixed problem, split
across the GPUs. The 4M>1M trend is the strong-scaling signature: bigger per-GPU slab → fixed overheads
amortized → higher efficiency.)

- **1M: 310.9 / (232+115) = 89.6%** (baseline 283/347 = 81.6%).
- **4M: 83.6 / (61.2+29.5) = 92.2%** (baseline 79.3/90.7 = 87.5%).

`η_weak` (fixed per-GPU slab, isolates pure coordination overhead) is **not yet measured** — needs a
helper to run each slab alone (slab particles + slab grid, ghost=0, transport=None) through the single
bench. Recipe: `η_weak = dual_frame_time / max(AMD_solo(slab_A)_frame_time, NV_solo(slab_N)_frame_time)`.

### Migration flux (PoolHealthBuffer time series)

Per 1000-step defrag interval, migrants installed at the own-pool tail ramp from ~10 (transient) to a
**~80-113/interval plateau** (1M) / **~62/interval** (4M) at steady state. Tiny (≤0.03% of a slab).
So `own_pool_size` is dominated by `alive`; migrant headroom is negligible → **`pool_safety=1.2` is
~1000× more than needed; 1.05× would still be safe.** Size from the never-reset `peak_migration`
watermark at the deployment weights (flux ramps, so short runs under-estimate).

### Transfer chain (direct GPU-timestamp probe, isolated)

| DMA leg | AMD | NV |
|---|---:|---:|
| readback (dev→sender) | 122µs | **482µs** |
| upload (recv→dev) | 124µs | **515µs** |
| worker memcpy (isolated/prod) | ~110 / 238µs | — |

Chain gating each GPU's phase_c ≈ **845-875µs** (with production memcpy). **phase_b is 1547/1630µs →
~700µs of compression room** before the transfer floor. The chain is PCIe/host-bandwidth bound (NV's
host-visible DMA is the bulk and ~4× slower than AMD's), so it barely contends with the VRAM/compute-bound
phase_b → isolated ≈ in-flight. **phase_b is NOT transfer-floored** (this overturned an earlier
b_to_c_gap heuristic that wrongly suggested it was).

### Roofline (NV, via nsys GPU-metrics — `ncu` does NOT hook this Vulkan app)

Heavy-kernel windows (correction/density/force): **DRAM read+write ~14%** (not bandwidth-bound),
**SM-issue ~63%** (not compute-bound), **occupancy ~86%** (warps resident but stalling). →
**latency-bound** (irregular per-neighbor dependent-load chain; L2 absorbs traffic). Implications:
math-reduction won't help, ghost-shrink won't speed compute; the lever is reducing the gather.

---

## 4. How to run / reproduce

```bash
# compile shaders (emits experiment/v4/shaders/spv/, gitignored)
.venv/Scripts/python.exe experiment/v4/compile_shaders_v4.py

# single-GPU bench (full problem on one device; warmup matters — AMD needs ~5000 frames)
.venv/Scripts/python.exe experiment/v4/_run_v4_single_bench.py --device 0 --max-steps 12000 --bench-window 3000

# dual bench — INSTRUMENTED, depth-1 (per-kernel GPU timestamps + migration series + pool_health)
.venv/Scripts/python.exe experiment/v4/_run_v4_dual_bench.py \
    --weights 2.9,1.0 --pool-safety 1.2 --max-steps 9000 --warmup 5000

# dual pipeline — SUBMIT-AHEAD (depth-2), throughput + migration series (no per-kernel timestamps)
.venv/Scripts/python.exe experiment/v4/_run_v4_dual_pipeline.py \
    --depth 2 --weights 2.9,1.0 --pool-safety 1.2 --max-steps 18000 --warmup 5000

# transfer-chain probe (isolated DMA leg timing)
.venv/Scripts/python.exe experiment/v4/_probe_transfer_chain.py --weights 2.9,1.0 --pool-safety 1.2 --iters 200

# 4M: add --case cases/lid_driven_cavity_2d_4m/case.yaml --weights 2.8,1.0
```

**Profiling note:** Nsight Compute (`ncu`) reports "No kernels profiled" on this Vulkan app (it targets
CUDA; Vulkan compute roofline needs Nsight Graphics, not installed). Use **Nsight Systems GPU-metrics**:
`nsys profile --gpu-metrics-devices=0 --trace=vulkan ...`, then `nsys export --type sqlite` and query
`GPU_METRICS` + `TARGET_INFO_GPU_METRICS` (metric names: "SM Issue", "DRAM Read/Write Bandwidth",
"Compute Warps in Flight"). Tools under `C:/Program Files/NVIDIA Corporation/Nsight {Compute,Systems}/`.

---

## 5. Key files (`experiment/v4/`)

| file | role |
|---|---|
| `utils/simulator_v4.py` | per-GPU sim: buffers, pipelines, phase A/B/C + transfer cmds, defrag, `readback_pool_health` |
| `utils/orchestrator_v4.py` | `DualGpuOrchestratorV4`: `step()` (depth-1 instrumented), **`run_pipelined(depth)`** (submit-ahead), `_collect_defrag_report` |
| `utils/partition_v4.py` | `compute_dual_gpu_partition(case, weights, pool_safety=)` — per-slab pool + offsets |
| `utils/transport_v4.py` | `GhostMigrationWorker` (CPU-staged 3-hop transport, one thread per direction) |
| `utils/{case_v4,case_loader_v4,vulkan_context_v4,bench_v4,renderer_v4,debug_log_v4}.py` | case model / loader / VK ctx / GPU-timestamp bench / viewer / debug snapshots |
| `shaders/common.glsl` | spec consts + descriptor layout + **`PoolHealthBuffer`** |
| `shaders/install_migrations.comp` | migrant install + the two pool-health `atomicMax` |
| `shaders/{predict,update_voxel,ghost_send,correction,density,force,defrag,...}.comp` | the kernels |
| `_run_v4_single_bench.py` / `_run_v4_dual_bench.py` / `_run_v4_dual_pipeline.py` | runners |
| `_probe_transfer_chain.py` | transfer-chain DMA-leg probe |

---

## 6. Next steps (priority order)

1. **Reduce the neighbor gather (latency-bound).** Biggest lever. correction/density/force each do an
   independent 27-voxel scattered gather of the same neighbors. **Fuse them** so each neighbor's data
   is loaded once and reused across the three computations (~3× less gather-latency exposure).
   Constraint: the Path A+ cascading split uses different bands per kernel (correction 2 / density 3 /
   force 4) and density/force have a data dependency (density needs M⁻¹; force needs ρ_{n+1}). **First
   sub-task: assess fusing correction+density** (both already in phase_b; density only needs self's M⁻¹
   which correction just computed). Validate: depth-1 per-kernel delta + depth-2 fps + drift=0.
2. **Secondary gather wins:** fewer wasted candidate loads (~9 voxels × ~25 ≈ 225 candidates, ~2/3
   distance-rejected); better locality (Morton `VOXEL_ORDER=1` is reserved but unused); more ILP/prefetch.
3. **Measure η_weak** (build the slab-only single-run helper) to quantify the residual pure
   coordination overhead — confirms how much orchestration headroom is truly left.
4. **Fundamental ceiling (not recoverable in cross-vendor V4):** NV's cross-family timeline-semaphore
   regression + the ghost/transfer tax. Only same-vendor 2×5090 P2P (a different transport backend)
   removes the regression. This is paper "portability section" material, not a V4 tuning target.

---

## 7. `memory/` notes (also auto-loaded each session)

- `project_v4_branch.md` — V4 is the forward branch; V3 architecture abandoned.
- `project_v3_0_submit_ahead_validated.md` — submit-ahead + pool sizing measured (the +9.6%/+5.4%).
- `project_transfer_chain_measured.md` — transfer chain ~850µs, phase_b not floored.
- `project_kernel_roofline.md` — kernels are latency-bound; profiling method (nsys, not ncu).
- `project_nv_concurrent_regression_negative.md` — NV cross-family regression, 9 experiments falsified.
- `project_v2_baseline_cavity_1m.md` — V2 baseline numbers + throughput-efficiency methodology.
