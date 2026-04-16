# Vulkan Multi-GPU SPH Project

## Overview

Vulkan-based rendering/compute project migrating from an existing OpenGL δ-plus WCSPH codebase. Target: cross-vendor multi-GPU (NVIDIA RTX 4060 Ti + AMD RX 7900 XTX) SPH fluid simulation at scale.

**Cross-vendor is an intentional stress test**, not a constraint.

## Current State

**Phase 1 complete** (`main_multigpu_particles.py`): 10M particle cross-GPU migration demo. CPU-staged migration via host-visible buffers. Ping-pong particle buffers, atomic counters, `vkCmdDrawIndirect` for variable alive count. Resize-resilient (recovered from acquire/present failures without state corruption). Validated: total particle count conserved; visible asymmetric GPU load (4060Ti ~38fps vs 7900XTX ~140fps at peak).

**Phase 2 (shared memory) ruled out:** Probed with `probe_external.py` / `probe_interop.py`. Cross-vendor OPAQUE_WIN32 handles rejected by both drivers (`VK_ERROR_OUT_OF_DEVICE_MEMORY` / `VK_ERROR_UNKNOWN`). D3D12 interop route is theoretically available but not pursued — accepting CPU-staged cost instead.

**Next: SPH rewrite** — existing OpenGL code (10k lines, δ-plus WCSPH with persistent voxel grid neighbor search) will be rewritten fresh in Vulkan, not line-by-line ported. See memory `sph_design.md` for architecture plan.

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

## Key Architectural Decisions (accumulated)

- **Cross-vendor shared memory does NOT work** on this hardware pair (N 4060Ti + A 7900XTX). Migration is always CPU-staged. Do not re-attempt.
- **Ping-pong particle buffers** with parity swap inside `submit()` after successful queue submission (so parity doesn't flip on acquire failure).
- **Counter reset inside `submit()`** only after acquire succeeds — prevents state corruption on resize events.
- **`read_and_clear_outgoing()` + `append_incoming()`** pattern prevents double-routing and lost particles during acquire-failure retry paths.
- **Sentinel-based state recovery:** `(out_count + outgoing_count) > 0` indicates compute ran last frame; otherwise fall back to initial counts.
- **`vkCmdDrawIndirect`** lets compute atomically write `instanceCount` → render draws exactly the live particle count without CPU readback.

## SPH Architecture Plan (upcoming)

- **Method:** δ-plus WCSPH (explicit time integration, no iterative solver) — multi-GPU-friendly.
- **Neighbor search:** persistent uniform voxel grid (existing OpenGL approach retained; superior to per-step radix sort for WCSPH CFL regime).
- **Shader constant injection:** via `VkSpecializationInfo` + `layout(constant_id=N) const` in SPIR-V (replaces OpenGL `#define` string injection).
- **Multi-GPU:** static voxel assignment to GPUs (no dynamic rebalancing initially); ghost layer of voxels near split; every step re-sync ghost state.
- **Migration transport:** reuse Phase 1 CPU-staged pattern from `main_multigpu_particles.py`.
- **Async compute queue optimization** (interior compute overlaps with CPU routing) is deferred — will be added once SPH is functional.

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
