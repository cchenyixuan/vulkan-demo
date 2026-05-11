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

**Next: SPH rewrite** — existing OpenGL code (10k lines, δ-plus WCSPH with persistent voxel grid neighbor search) will be rewritten fresh in Vulkan, not line-by-line ported. Full design in **`docs/sph_design.md`**.

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

### SPH multi-GPU (design stage, see `docs/sph_design.md`)

- **Leapfrog integration** (half-step velocity storage), not explicit Euler. Algebraically equivalent to velocity Verlet (KDK) but collapses the two half-kicks into a single full-step kick per iteration — **5 kernels per step** instead of Verlet KDK's 6. The drift-before-force ordering still holds, letting end-of-step multi-GPU sync pre-compute everything the next step needs, giving **1 sync per step**.
- **Migration merged into ghost flow** via bit-exact Kernel A on ghost particles. No separate migration pack/kernel. Controlled by `STRICT_BIT_EXACT` spec constant (`constant_id=10`, default `true`, cost <2% on integration kernel).
- **Single own-particle SoA buffer**, voxel-sorted by `voxel_id`. Defrag keeps boundary-column particles clustered in contiguous pid range, so V2's interior/boundary correction split (inline `in_boundary_band(coord)` check in `correction.comp`) lands on contiguous workgroups with near-zero divergence.
- **Ghost buffer carries `{x, v, ρ, a, shift}`** (~64B/particle padded) so both GPUs can locally do Kernel A + density + force for the boundary. Bandwidth ~1.5× classical ghost but eliminates migration flow.
- **Transport backend is pluggable**: `CpuStagingBackend` (cross-vendor, Phase 1 pattern), `P2PBackend` (same-vendor `vkCmdCopyBuffer`), `SharedMemoryBackend` (conditional on probe).
- **Per-particle globals → specialization constants**, not per-particle storage. Audit OpenGL's `ParticleRuntimeData` — most intermediate fields can become local variables inside a single kernel.

### Phase 1 (multi-GPU migration demo)

- **Cross-vendor shared memory does NOT work** on N 4060Ti + A 7900XTX pair. Migration is always CPU-staged there. (NV+NV 2×5090 pair is a separate question — see `probe_interop.py` re-run TODO.)
- **Ping-pong particle buffers** with parity swap inside `submit()` after successful queue submission (so parity doesn't flip on acquire failure).
- **Counter reset inside `submit()`** only after acquire succeeds — prevents state corruption on resize events.
- **`read_and_clear_outgoing()` + `append_incoming()`** pattern prevents double-routing and lost particles during acquire-failure retry paths.
- **Sentinel-based state recovery:** `(out_count + outgoing_count) > 0` indicates compute ran last frame; otherwise fall back to initial counts.
- **`vkCmdDrawIndirect`** lets compute atomically write `instanceCount` → render draws exactly the live particle count without CPU readback.

## SPH Architecture Plan (upcoming)

Full design: **`docs/sph_design.md`** (integration scheme, buffer layout, transport backends, invariants, performance targets, implementation phases).

Summary:
- **Method**: δ-plus WCSPH, explicit **leapfrog** integration (half-step velocity; algebraically equivalent to velocity Verlet KDK but 5 kernels/step instead of 6). Upgrade from OpenGL baseline's explicit Euler.
- **Neighbor search**: persistent uniform voxel grid (existing OpenGL approach retained).
- **Shader constants**: `VkSpecializationInfo` + `layout(constant_id=N) const` (replaces OpenGL `#define` injection).
- **Multi-GPU**: 1D slab decomposition, partition adjusts on wait-time feedback (V2; static weight in V1). 1-voxel-thick ghost. Voxel-sorted own buffer; V2 splits `correction.comp` into interior/boundary passes via inline check.
- **Crossing/migration**: merged into ghost flow via bit-exact Kernel A. No separate migration pipeline.
- **Transport**: pluggable backend (CPU staging / P2P / shared memory). V1.0a uses CPU staging on both cross-vendor and same-vendor; V3 layers `P2PBackend` on top for same-vendor.
- **Sync cadence**: 1 sync per step at end of step. V2 hides the sync window behind `correction(interior)` running concurrent with the CPU memcpy pathway.
- **V1 → V2 jump**: timeline semaphores replace fences, two CPU sync pathways run on worker threads, `correction.comp` adds inline `in_boundary_band` check, partition feedback loop adjusts `partition.x` on wait-time imbalance. See `docs/sph_v2_design.md`.

**Performance target**: 2M particles @ 350 fps on 2× RTX 5090 (V2 same-vendor + adaptive partition). Cross-vendor (NV+AMD) V2 estimate ~200–260 fps for 1M particles, limited by `max(per-GPU compute)` + PCIe transport. V1 cross-vendor measured baseline (after `HOST_CACHED` staging fix) ~60 fps. OpenGL baseline 1M @ 330 fps on single 5090.

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
