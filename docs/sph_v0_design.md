# SPH V0 Design: Buffer Layout + Verlet Pipeline

V0 scope = single-GPU δ-plus WCSPH rewrite in Vulkan. Match or exceed OpenGL baseline (1M @ 330 fps on 5090). Lays the data-layout and pipeline-skeleton foundation that V1+ multi-GPU builds on. See `sph_design.md` for the V0–V3 arc and multi-GPU rationale.

## Scope Decisions Already Made

- Integration: **velocity Verlet (KDK)**, replacing OpenGL baseline's Euler.
- Neighbor search: persistent uniform voxel grid with **incremental cell list** (in/out event flow), not per-step radix sort.
- Algorithm: δ-plus WCSPH with KCG (kernel gradient correction) and PST (particle shifting).
- Extensions deferred: micropolar, multi-resolution, thermal. Base δ-SPH only for V0.
- Bit-packing rejected: separate bindings over packed high-bit fields (see `feedback_no_bitpack` in memory). Readability > marginal savings.

## Buffer Layout (Final, V0)

All buffers are `std430` SSBOs on a single descriptor set. Each particle is 132 B core + optional diagnostic.

### Particle SoA (unified pool — fluid, boundary, inlet, rotor share slots, differentiated by `Material`)

| binding | Name | Type | Size/particle | Access pattern |
|---|---|---|---|---|
| 0 | `PosVid` | `vec4` (x, y, z, voxel_id_as_float) | 16 B | Hot (every neighbor loop) |
| 1 | `RhoP_A` | `vec2` (ρ, P) | 8 B | ping-pong partner A |
| 2 | `RhoP_B` | `vec2` (ρ, P) | 8 B | ping-pong partner B |
| 3 | `VelMass` | `vec4` (vx, vy, vz, mass) | 16 B | Force read, Kick write |
| 4 | `Acc` | `vec4` (ax, ay, az, _) | 16 B | **Persistent across step** (Verlet needs a_n) |
| 5 | `Shift` | `vec4` (sx, sy, sz, _) | 16 B | **Persistent across step** (Drift needs shift_n) |
| 6 | `Material` | `uint` | 4 B | Branch (fluid/boundary/inlet/rotor) |
| 7 | `CorrInv` | `vec4 × 2` (M⁻¹ symmetric, 6 components) | 32 B | Correction writes; density + force read |
| 8 | `GradKsd` | `vec4` (∇ρ.xyz, kernel_sum_delta) | 16 B | Correction writes; density reads |

**Layout notes:**
- `voxel_id` stored as float-reinterpret-of-uint in `PosVid.w`. Decode: `uint vid = floatBitsToUint(PosVid[pid].w)`. Single hot load gets pos + vid.
- `RhoP` ping-pong: density writes to the "next" buffer; descriptor binding swaps at end of step. Shader always writes `RhoP_write[self]` and reads `RhoP_read[neighbor]`.
- `CorrInv` stores the **inverse** matrix (KCG correction, symmetric 3×3 → 6 floats). Two pad slots left blank (no cross-stage field with matching lifetime was identified).
- `GradKsd` packs density gradient (3) + `kernel_sum_delta` (1). Same producer (correction) and consumer (density).
- `Acc` and `Shift` kept as **separate bindings** even though both are force-kernel outputs: Kick1 only reads acc, Drift only reads shift — splitting avoids loading a field you don't need.
- `mass` kept per-particle (user decision, future-proofing for multi-phase / inlet-size variation).

### Voxel Buffers (SoA, no embedded header)

| binding | Name | Type | Size/voxel | Access |
|---|---|---|---|---|
| 10 | `InsideCount` | `uint` | 4 B | Hot (every neighbor query reads this) |
| 11 | `InCount` | `uint` (atomic) | 4 B | Predict writes (atomic), update_voxel consumes+resets |
| 12 | `InsideBuf` | `uint × 96` | 384 B | Per-voxel particle ID list. `CAP_inside = 96` |
| 13 | `InBuf` | `uint × 16` | 64 B | Per-voxel incoming events. `CAP_in = 16` |

**Design notes:**
- **No stored header**: `voxel_id`, x/y/z offset, and 26 neighbor IDs are all derivable from the array index. `coord_of(vid) = (vid % Dx, (vid/Dx) % Dy, vid/(Dx·Dy))`. Neighbors computed on-the-fly with bounds check.
- **No out_buffer**: the canonical "which voxel am I in" is `particle.voxel_id`. Compaction reads each inside_buf entry's particle and keeps iff `particle.voxel_id == self`. Rejected multi-source-of-truth in favor of single canonical field.
- **Voxel encoding**: linear z-major by default. Spec constant switch (`VOXEL_ORDER`, `constant_id=20`) reserved for future Morton / 4×4×4 tile layouts if cache miss rate demands.
- **Irregular domain**: dense grid with dead voxels (`inside_count=0` auto-skipped). Sparse hash deferred until waste > 50%.
- `CAP_inside = 96` chosen with 1.5× headroom over observed OpenGL peak of ~64. Overflow flag atomic counter to be added for V0 validation.
- `CAP_in = 16` is a guess (~2× safety over CFL-limited per-step crossing rate); validate with overflow counter.

### Special / Global

| binding | Name | Purpose |
|---|---|---|
| 20 | `InletTemplate` | Read-only prototype particles for inlet spawn kernel |
| 21 | `GlobalStatus` | n_particles_alive, frame_counter, max_velocity, inlet counters, overflow flags |
| 30 | `Diagnostic` (optional) | Only bound in debug/render build. Holds curl_u, FTLE, w, etc. for visualization |

### Spec Constants (replace OpenGL `#define` injection)

Globals that were per-particle `SubData` in OpenGL — promoted to compile-time constants:

```glsl
layout(constant_id = 0)  const float SMOOTHING_LENGTH = 0.05;
layout(constant_id = 1)  const float REST_DENSITY     = 1000.0;
layout(constant_id = 2)  const float VISCOSITY        = 1e-6;
layout(constant_id = 3)  const float EOS_CONSTANT     = 2.15e6;
layout(constant_id = 4)  const float EOS_GAMMA        = 7.0;
layout(constant_id = 5)  const float DELTA_COEFF      = 0.1;   // δ-plus diffusion
layout(constant_id = 6)  const float DT               = 5e-4;
layout(constant_id = 7)  const float GRID_ORIGIN_X    = 0.0;
layout(constant_id = 8)  const float GRID_ORIGIN_Y    = 0.0;
layout(constant_id = 9)  const float GRID_ORIGIN_Z    = 0.0;
layout(constant_id = 10) const bool  STRICT_BIT_EXACT = true;  // for multi-GPU V1+
layout(constant_id = 11) const uint  GRID_DIM_X       = 128;
layout(constant_id = 12) const uint  GRID_DIM_Y       = 128;
layout(constant_id = 13) const uint  GRID_DIM_Z       = 128;
layout(constant_id = 20) const uint  VOXEL_ORDER      = 0;     // 0=linear, 1=Morton, 2=tile (future)
```

## Pipeline Stages (Verlet KDK)

Entering step `n`: `x_n, v_n, a_n, ρ_n, shift_n` all valid from previous step's end.

```
1. predict          (Kick1 + Drift, per-particle dispatch)
     reads:  PosVid, Vel, Acc, Shift
     writes: PosVid (new x, new voxel_id), Vel (v_half), InBuf (atomic append on crossing)
     logic:  v_half = v_n + 0.5 * a_n * dt
             x_{n+1} = x_n + v_half * dt + shift_n
             new_vid = voxel_of(x_{n+1})
             if (new_vid != old_vid) atomicAdd(InCount[new_vid], 1) then append InBuf
             PosVid[pid].w = uintBitsToFloat(new_vid)

2. update_voxel     (per-voxel dispatch, one workgroup per voxel)
     reads:  old InsideBuf, InsideCount, InBuf, InCount, PosVid (for vid match)
     writes: new InsideBuf (compacted + in_buf appended), new InsideCount, reset InCount=0
     logic:  for each slot i in [0, InsideCount[v]):
                  pid = InsideBuf[v][i]
                  keep iff floatBitsToUint(PosVid[pid].w) == v
             append InBuf[v][0..InCount[v]) to tail
     MUST run before correction (correction's neighbor search uses inside_buf).

3. correction       (per-particle, 27-neighbor loop)
     reads:  PosVid, VelMass, RhoP_read, InsideCount, InsideBuf
     writes: CorrInv (M⁻¹ symmetric 6), GradKsd (∇ρ + ksd)

4. density          (per-particle, 27-neighbor loop)
     reads:  PosVid, VelMass, RhoP_read, CorrInv, GradKsd, InsideCount, InsideBuf
     writes: RhoP_write (new ρ, new P via EOS)

5. force            (per-particle, 27-neighbor loop)
     reads:  PosVid, VelMass, RhoP_write (now canonical ρ), CorrInv, InsideCount, InsideBuf
     writes: Acc (a_{n+1}), Shift (shift_{n+1} via PST)

6. kick2            (per-particle, trivial)
     reads:  Vel (holds v_half), Acc
     writes: Vel (v_{n+1})
     logic:  v = v_half + 0.5 * a_{n+1} * dt

END: x_{n+1}, v_{n+1}, a_{n+1}, ρ_{n+1}, shift_{n+1} ready for step n+2.
Descriptor ping-pong: swap RhoP_A ↔ RhoP_B bindings for next step.
```

### Periodic: Defrag Pass (every N steps, N≈256)

Re-sort particles so each voxel's `InsideBuf` holds contiguous integers → restores cache coherence for neighbor scatter loads. Triggered externally (CPU counter or GPU-side fragmentation metric).

```
defrag.comp:
  1) exclusive prefix sum over InsideCount[] → CellStart[]
  2) per-voxel scatter: new_pid = CellStart[v] + i, copy all particle fields old→new
  3) rewrite InsideBuf to contiguous integers
  4) swap old ↔ new particle-pool bindings
```

### Bootstrap (one-time at t=0)

First step has no `a_0`. Recommended: run a preliminary `correction → density → force` pass on initial positions to populate `Acc` and `Shift`, then enter the normal KDK loop. Cost: one extra kernel sequence at startup, zero impact on steady-state.

## Key Differences vs OpenGL Baseline

| Aspect | OpenGL (current) | Vulkan V0 |
|---|---|---|
| Integrator | Euler | Velocity Verlet (KDK) |
| Particle buffer | 3 types × 3 sub-buffers = 9 SSBOs | 1 unified SoA pool, SoA-split by field lifetime |
| Per-particle memory | ~512 B (ParticleData + SubData + RuntimeData mat4) | **132 B** (4× reduction) |
| Voxel size | 2912 ints = 11.4 KB/voxel | **~456 B/voxel** (25× reduction), no stored header |
| Shader constants | GLSL `#define` injected at compile | `VkSpecializationInfo` |
| Kernel dispatch | One giant compute shader + `stage` uniform branch | One `VkComputePipeline` per stage |
| Stage count | 5 (Euler) | 6 (Verlet: predict + update_voxel + correction + density + force + kick2) |
| update_voxel position | Step tail | After predict, before correction (Verlet requires) |
| Boundary dispatch | Sign-of-index sentinel (`particle_index = -particle_index`) | `Material` field branch |

## Cross-Stage Field Audit

From OpenGL `ParticleRuntimeData`, fields retained as persistent cross-stage state in V0:

- **Kept**: M⁻¹ (→ `CorrInv`), ∇ρ (→ `GradKsd`), kernel_sum_delta (→ `GradKsd.w`), new_ρ / new_P (→ `RhoP_write`), particle_shift (→ `Shift`)
- **Cut** (kernel-local, do in registers): `kernel_gradient_sum`, `kernel_sum`, `kernel_normalize_matrix` accumulation
- **Cut** (extensions, deferred): `curl_u`, `w`, `overlap`, `aw`, `mass_transfer_target_particle`, `FTLE`
- **Moved to diagnostic buffer** (optional, debug/render builds only): visualization quantities

## Open Items (for next session on 2×5090 rig)

1. **Kernel-level dispatch design** per stage:
   - `predict`: workgroup size (64 vs 128), atomic contention in crossing append
   - `update_voxel`: per-voxel workgroup vs thread-per-slot, shared memory for inside_buf staging
   - `correction / density / force`: 27-neighbor loop structure, shared-memory tile-caching decision
   - `kick2`: whether to inline into tail of `force` kernel (save a dispatch)
   - `defrag`: prefix-sum scan implementation (subgroup intrinsics vs two-pass)

2. **Density ping-pong mechanics**: descriptor set swap vs push constant index. Need to pick and implement.

3. **Bootstrap pass**: implement the one-time `correction → density → force` sequence before main loop.

4. **Overflow validation**: add atomic counters for CAP_inside and CAP_in overflow; run with representative scene to validate 96/16 headroom.

5. **CAP tuning once measured**: revisit 96 (inside) and 16 (in) based on live profiling data.

6. **Defrag trigger policy**: static N=256 for V0, migrate to fragmentation-metric-driven later.

7. **Numerical validation gate** (see `feedback_api_rewrite`): same initial conditions on OpenGL vs V0 Vulkan, compare after N steps. No architectural progression until parity holds.

## File Organization Plan (proposed)

```
shaders/sph/
  predict.comp
  update_voxel.comp
  correction.comp
  density.comp
  force.comp
  kick2.comp
  defrag.comp
  common.glsl        # shared: spec constants, coord_of(), neighbor iter, EOS, kernel funcs
  symmetric_mat3.glsl  # M⁻¹ pack/unpack helpers

utils/sph/
  buffers.py         # SSBO allocation + binding descriptors
  pipeline.py        # compute pipeline creation per stage
  descriptor.py      # descriptor set layout + ping-pong swap
  spec_constants.py  # VkSpecializationInfo assembly

main_sph_v0.py       # main loop, binding setup, stage dispatch ordering
```
