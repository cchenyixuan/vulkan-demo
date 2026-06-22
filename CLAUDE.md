# Vulkan Multi-GPU SPH Project

## Overview

Vulkan-based rendering/compute project migrating from an existing OpenGL δ-plus WCSPH codebase. Multi-GPU SPH fluid simulation at scale.

- **Production target**: 2× RTX 5090 on one machine (same-vendor, PCIe 5.0 x16 P2P). Primary benchmark configuration for the final paper's scaling results.
- **Stress test (paper portability section)**: NV 4060 Ti + AMD 7900 XTX, cross-vendor single machine.

Cross-vendor is an intentional robustness test, not the primary deployment constraint. Both configs share the same codepath; the transport layer is pluggable. See `docs/sph_design.md` for the full design.

## Current State

**Phase 1 complete** (`main_multigpu_particles.py`): 10M particle cross-GPU migration demo. CPU-staged migration via host-visible buffers. Ping-pong particle buffers, atomic counters, `vkCmdDrawIndirect` for variable alive count. Resize-resilient (recovered from acquire/present failures without state corruption). Validated: total particle count conserved; visible asymmetric GPU load (4060Ti ~38fps vs 7900XTX ~140fps at peak).

**Phase 2 (shared memory)** result is hardware-pair-dependent:
- **NV+AMD**: probed with `probe_external.py` / `probe_interop.py` — OPAQUE_WIN32 handles rejected by both drivers (`VK_ERROR_OUT_OF_DEVICE_MEMORY` / `VK_ERROR_UNKNOWN`). CPU staging is the only path for this pair. Do not re-attempt.
- **NV+NV (2×5090)**: not yet probed. Expected to succeed (same-vendor handle import usually works on NV). Re-run `probe_interop.py` on the 2×5090 rig when available.

**SPH V2 Path A+ complete (`commit 1214643` + `f66b607`, 2026-05-22)**: cross-vendor dual GPU now BEATS single-AMD by +17%. Cavity 1M @ 2× AMD 7900 XTX + NV 4060 Ti, weights=3.2:1.0 (new default): **273.9 fps avg (285 peak) over 50k steps**, alive perfectly conserved. Scaling efficiency 78% of theoretical max (349 fps). V2 baseline at 2.6:1.0 was 228 fps (3% slower than single-AMD); Path A+ adds transfer queue + 5N timeline + density cascading split (correction_interior + density_deep_interior in Phase B, density_boundary in Phase C). See `docs/sph_v2_design.md` for the original V2.0 design; Path A+ overlay extensions are documented in commit `1214643`'s message and in `docs/sph_v3_design.md`.

**Next: V3** — further optimization beyond Path A+ (target ~290 fps cross-vendor + 500+ fps same-vendor 2×5090). Phases: V3.0 CPU orchestrator overhead elimination → V3.1 dynamic load balancing → V3.2 same-vendor P2P backend → V3.3 conditional cascading force. Design in **`docs/sph_v3_design.md`**.

## Layout

```
vulkan-demo/
├── main_multigpu_particles.py    # Phase 1 reference demo (kept for SPH cross-GPU pattern reuse)
├── probe_external.py              # cross-vendor memory/semaphore capability probe
├── probe_interop.py               # actual OPAQUE_WIN32 export/import test
├── compile_shaders.py             # glslc batch compile for particle shaders
├── shaders/
│   ├── particle_update.comp       # Phase 1 migration + integrate
│   ├── particle.vert / .frag      # instanced quad particle rendering
│   └── *.spv                      # compiled SPIR-V
└── utils/
    ├── instance.py                # VkInstance + validation + debug messenger
    ├── device.py                  # queue family discovery + logical device
    ├── swapchain.py               # swapchain + image views (IMMEDIATE present preferred)
    ├── pipeline.py                # render pass + particle graphics pipeline
    ├── compute_pipeline.py        # compute pipeline + descriptor set helpers
    ├── commands.py                # command pool + framebuffers
    ├── sync.py                    # semaphores + fences
    ├── particle_buffer.py         # device-local / host-visible buffer helpers
    └── shaders.py                 # SPIR-V loading
```

## Code Conventions

- **No abbreviations in identifiers.** Use full words in variables, buffers, struct fields, and function names. Examples:
  - `correction_matrix` / `correction_matrix_inverse`, not `M` / `M_inv`
  - `neighbor_particle_id`, not `pid_j`
  - `smoothing_length`, not `h`
  - `kernel_gradient`, not `gW`
  - `PositionVoxelIdBuffer`, not `PosVidBuf`
  - Loop counters: `row_index` / `slot_index`, not `i` / `k`
- Math symbols (W, ρ, ∇, M, ξ) are **allowed in comments** explaining derivation; **never in identifiers**.
- Verbose names acceptable even when long: `smoothing_length_power_dimension_plus_one` beats `h_dim_p1`.

## Key Architectural Decisions (accumulated)

### SPH multi-GPU (implemented through V2 Path A+, see `docs/sph_v2_design.md` + `docs/sph_v3_design.md`)

- **Leapfrog integration** (half-step velocity storage), not explicit Euler. Algebraically equivalent to velocity Verlet (KDK) but collapses the two half-kicks into a single full-step kick per iteration — **5 kernels per step** instead of Verlet KDK's 6. The drift-before-force ordering still holds, letting end-of-step multi-GPU sync pre-compute everything the next step needs, giving **1 sync per step**.
- **Migration merged into ghost flow** via bit-exact Kernel A on ghost particles. No separate migration pack/kernel. Controlled by `STRICT_BIT_EXACT` spec constant (`constant_id=10`, default `true`, cost <2% on integration kernel).
- **Single own-particle SoA buffer**, voxel-sorted by `voxel_id`. Defrag keeps boundary-column particles clustered in contiguous pid range, so the cascading interior/boundary splits land on contiguous workgroups with near-zero divergence.
- **Ghost buffer carries `{x, v, ρ, a, shift}`** (~64B/particle padded) so both GPUs can locally do Kernel A + density + force for the boundary. Bandwidth ~1.5× classical ghost but eliminates migration flow.
- **Transport backend is pluggable**: `CpuStagingBackend` (cross-vendor, current implementation), `P2PBackend` (same-vendor `vkCmdCopyBuffer`, V3.2 future), `SharedMemoryBackend` (conditional on probe).
- **Per-particle globals → specialization constants**, not per-particle storage. Audit OpenGL's `ParticleRuntimeData` — most intermediate fields can become local variables inside a single kernel.

### V2 Path A+ specifics (`commit 1214643`)

- **5N timeline per sim** (was 3N in V2.0): values are phase_a_done (5N+1), readback_done (5N+2 — transfer Q signal), worker_done (5N+3 — host signal), upload_done (5N+4 — transfer Q signal), frame_done (5N+5). Worker waits source.readback_done + dest.readback_done; host-signals dest.worker_done.
- **Dedicated transfer queue** picked by `_find_transfer_queue_family()` (`VulkanContextV2`): prefers a transfer-only family (AMD family[2], NV family[1/3/4/5]). Fallback to compute family with stderr warning.
- **12 device-local buffers declared `SHARING_MODE_CONCURRENT`** (`_CONCURRENT_BUFFER_NAMES` in `simulator_v2.py`): 9 set 0 SoA fields + inside_particle_count + inside_particle_index + global_status. Required for cross-queue access; eliminates the need for queue family ownership transfer barriers. Theoretical cost 3-5%, observed ~+76% per-particle on NV correction_interior (under investigation, but net gains far exceed this regression).
- **Cascading boundary band widths** (in voxel x-columns): correction = 2 (must be ≥2 to account for install_migration adding migrants to column 0, which column 1 reads), density = 3 (correction's 2 + 1 for neighbor reach), force = 4 (density's 3 + 1). Force is NOT cascaded in current production design: pipelines exist (`force_deep_interior` + `force_boundary` built in P3.B) but unused; enabling them requires adding `FORCE_DENSITY_SOURCE` spec const to `force.comp` so force_deep_interior can read density from scratch instead of primary. Decision deferred until same-vendor optimization (V3.3) where the trade-off may matter.
- **Default K_split weights = 3.2:1.0** for cross-vendor AMD+NV cavity 1M. Specific to this hardware pair; other configurations should sweep first.

### Phase 1 (multi-GPU migration demo)

- **Cross-vendor shared memory does NOT work** on N 4060Ti + A 7900XTX pair. Migration is always CPU-staged there. (NV+NV 2×5090 pair is a separate question — see `probe_interop.py` re-run TODO.)
- **Ping-pong particle buffers** with parity swap inside `submit()` after successful queue submission (so parity doesn't flip on acquire failure).
- **Counter reset inside `submit()`** only after acquire succeeds — prevents state corruption on resize events.
- **`read_and_clear_outgoing()` + `append_incoming()`** pattern prevents double-routing and lost particles during acquire-failure retry paths.
- **Sentinel-based state recovery:** `(out_count + outgoing_count) > 0` indicates compute ran last frame; otherwise fall back to initial counts.
- **`vkCmdDrawIndirect`** lets compute atomically write `instanceCount` → render draws exactly the live particle count without CPU readback.

## Performance Benchmarks (current, measured 2026-05-22)

| Configuration | fps | wall_time | vs single AMD | scaling efficiency |
|---|---|---|---|---|
| Single AMD 7900 XTX (steady) | 234 | 4274 µs | 1.00× | — |
| Single NV 4060 Ti (steady) | 115 | 8696 µs | 0.49× | — |
| V2.0 baseline dual @ 2.6:1.0 | 228 | 4385 µs | 0.97× ← was a regression | 65% |
| **V2 Path A+ dual @ 3.2:1.0 (50k)** | **273.9 avg / 285.5 peak** | **3662 µs** | **1.17×** | **78%** |
| Theoretical perfect parallel | 349 | 2866 µs | 1.50× | 100% |

Cavity 1M (1,046,529 particles), `cases/lid_driven_cavity_2d/case.yaml`, alive perfectly conserved across all configurations. Detailed kernel-level breakdown in `memory/project_v2_baseline_cavity_1m.md`.

**Caveat**: AMD 7900 XTX needs ~5000 frames warmup to reach steady state. Any dual bench with `max_steps < 5000` systematically under-reports dual fps. Use `--warmup 5000 --max-steps >= 10000`.

### Second rig — RTX 5090 + 7900 XTX (cross-vendor, measured 2026-06-17)

A different machine from the table above. **The FAST card is now the NV 5090, and Vulkan device order INVERTS: `device[0]`=5090, `device[1]`=7900 XTX** (`device[2]`=AMD iGPU, ignore). So with default `--device-a 0 --device-b 1` the **`--weights` are `NV:AMD`** (primary rig: AMD:NV); the runners' `AMD/a` / `NV/b` print labels are swapped here (cosmetic only). The full Vulkan SDK is installed, so the dual runners' default `enable_validation=True` would activate — prefix every bench with `VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`.

**Full 7-point scaling curve (measured 2026-06-18; all drift=0, no overflow; dual depth-2 except 1M/4M depth-3):**

| scale | single 5090 | single AMD | ceiling | best w (NV:AMD) | dual fps | η_strong |
|---|---|---|---|---|---|---|
| 1M  | 494  | 254  | 748   | 2.6 | 531 | 71.0% |
| 4M  | 126  | 70   | 196   | 2.0 | 180 | 91.9% |
| 6M  | 89.4 | 47.6 | 137   | 1.8 | 129.5 | 94.5% |
| 8M  | 68.1 | 33.9 | 101.9 | 1.8 | 99.2 | 97.3% |
| 10M | 55.9 | 27.3 | 83.1  | 1.8 | 81.9 | 98.5% |
| 14M | 38.7 | 19.2 | 57.9  | 1.9 | 56.8 | 98.1% |
| 16M | 34.6 | 16.9 | 51.5  | 1.9 | 50.9 | **98.9%** |

**Headline: η climbs 71% → 99% and saturates near ideal by ~10M** — the cross-vendor CPU-staged transport is a fixed overhead (∝ boundary ~√N) that amortizes against per-GPU compute (∝ N), so the **1M low efficiency (71%) is a small-problem artifact, NOT an architecture limit**. The optimal weight converges 2.6 → 1.8–1.9 (slightly below the single-GPU ratio ~2.0, because the 5090 loses some per-particle throughput in dual mode — NV CONCURRENT-buffer regression + slower host-DMA on the transfer critical path). depth-3 helps only at ≤4M (≥6M the ~200µs CPU bubble is negligible). 6M–16M cases generated by `utils/geometry/_demo_cavity_case.py`; figures `docs/scaling_eta.png` + `docs/scaling_fps.png`; campaign `experiment/v4/_run_scaling_campaign.py`. Full breakdown + the 98.9% analysis in `docs/sph_v4_summary.md` §3b and `memory/project-v4-5090-amd-1m-4m-benchmarks.md`.

### Third rig — 2× RTX 5090 (SAME-VENDOR, paper-primary, measured 2026-06-19)

7900 XTX swapped for a 2nd 5090 (riser; **PCIe x8/x8** Gen5 bifurcation, ReBAR on). `device[0]`/`[1]` = the 5090s (iGPU `[2]`), weights symmetric.

**NO P2P on consumer GeForce** (the gate failed): OPAQUE_WIN32 external-memory import fails even NV→NV (`vkGetMemoryWin32HandlePropertiesKHR`→INITIALIZATION_FAILED, alloc→OUT_OF_DEVICE_MEMORY; correct probe `experiment/v5/_probe_p2p_interop.py`) AND `VK_KHR_device_group` puts the two 5090s in separate groups of 1 (no NVLink/SLI). **So the V3.2 P2P-backend premise is dead — even 2×5090 uses the same host-staging as cross-vendor; nothing to build.** "Consumer GeForce has no usable P2P" is itself a paper finding.

2×5090 curve (host-staged, symmetric w=1.0 from 4M up, all drift=0): 1M 612fps/η60.3% · 2M 422/80.5% · 4M 235/93.6% · 6M 168/93.6% · 8M 126/92.7% · 10M 104/92.8% · 14M 71.6/93.0% · 16M 64.8/94.0%. **Headline — the η curves CROSS:** 2×5090 has higher absolute fps everywhere (~1.15–1.3×; 2M=422 smashes the 350 target) BUT its strong-scaling η **plateaus ~93–94% and never reaches the cross-vendor pair's ~99%** (and at 1M is worse, 60% vs 71%). Instrumented `b_to_c_gap` (w1.0, depth-1) splits this into two regimes: **1M = ~330µs exposed (transport-FLOORED → η60%); 8M = ~5µs (transport HIDDEN → η93%).** So small-N is genuinely transport-floored (the ~418µs host worker memcpy dominates), but **the large-N plateau is NOT transport** — it's mostly a η_strong metric artifact: η_strong=dual(N)/(2·single(N)) but each 5090 runs only N/2, and the 5090 is strongly sublinear (517 M/s@1M→553@8M), so per-GPU half-problems sit in a less-efficient regime than the single-full reference (~5–7% "loss", a reference choice not overhead; dual ≈100% vs single-on-half). Cross-vendor's 99% is partly metric-flattered (its small ~36% AMD share lands in AMD's efficient small-problem regime). Real squeezables: small-N transport (`PCIe x8/x8` riser+bifurcation → clean x16 / shared-host to skip the memcpy), `device[0]`=DISPLAY GPU (~6% extra mem-BW → move display to iGPU), depth-2 bubble. Binning/thermals NOT a factor (both boost ~2835–2895 MHz, 600W limit, no throttle). Curve dev: `experiment/v5/`; figures `docs/compare_{eta,fps}.png`; details `docs/sph_v4_summary.md` §3c + `memory/project-v5-2x5090-plan.md`.

## SPH Architecture Plan

Full design: **`docs/sph_v2_design.md`** (V2.0 baseline implemented) + **`docs/sph_v3_design.md`** (V3 future optimizations including DLB + P2P backend).

Summary:
- **Method**: δ-plus WCSPH, explicit **leapfrog** integration (half-step velocity; algebraically equivalent to velocity Verlet KDK but 5 kernels/step instead of 6). Upgrade from OpenGL baseline's explicit Euler.
- **Neighbor search**: persistent uniform voxel grid (existing OpenGL approach retained).
- **Shader constants**: `VkSpecializationInfo` + `layout(constant_id=N) const` (replaces OpenGL `#define` injection).
- **Multi-GPU**: 1D slab decomposition, static K_split (default tuned for AMD+NV cross-vendor cavity 1M; V3.1 will add wait-time-driven adaptive partition). 1-voxel-thick ghost. Voxel-sorted own buffer; cascading interior/boundary split for correction (band=2) + density (band=3).
- **Crossing/migration**: merged into ghost flow via bit-exact Kernel A. No separate migration pipeline.
- **Transport**: pluggable backend abstraction planned (V3.2). Current implementation = CPU staging on host RAM + dedicated transfer queue for device↔staging DMA (V2 Path A+). Worker thread bridges source→dest sender/receiver staging.
- **Sync cadence**: 1 sync per step at end of step. V2 Path A+ hides the entire transfer chain (readback + worker memcpy + upload) behind correction_interior + density_deep_interior on the compute queue.
- **V0 → V1 → V2 → V2 Path A+ → V3 progression**: see `docs/sph_v1_design.md`, `docs/sph_v2_design.md`, `docs/sph_v3_design.md`. V2 Path A+ is the current production state (commit 1214643).

**Performance targets**:
- **Same-vendor 2× RTX 5090** (paper primary): 350 fps @ 2M particles. Estimated achievable via V3.2 P2P backend; transfer chain drops from ~1300 µs (cross-vendor PCIe DMA) to ~100 µs (same-vendor P2P), eliminating need for cascading split.
- **Cross-vendor NV+AMD** (paper portability section): already at 273 fps for 1M with V2 Path A+; V3.0 + V3.1 should push to ~290 fps (83% efficiency).
- **OpenGL baseline reference**: 1M @ 330 fps on single 5090.

## Performance Notes (OpenGL baseline, pre-migration)

- Headless compute: ~350 fps
- With rendering (60-vertex instanced spheres): ~170 fps
- With rendering (2-vertex lines): ~350 fps
- Bottleneck: fragment fill rate + overdraw, not vertex count

## Environment

- Python 3.13 venv at `.venv/`
- Vulkan SDK at `C:/VulkanSDK/1.4.341.1/` (`glslc.exe` used for shader compilation)
- GLFW for windowing
- `vulkan` (python-vulkan) package for Vulkan bindings
- `numpy` for particle initialization

## Running

```bash
# Compile shaders
.venv/Scripts/python.exe compile_shaders.py

# Run Phase 1 demo (10M particles, no-vsync, IMMEDIATE present)
.venv/Scripts/python.exe main_multigpu_particles.py

# Cross-vendor capability probes
.venv/Scripts/python.exe probe_external.py
.venv/Scripts/python.exe probe_interop.py
```
