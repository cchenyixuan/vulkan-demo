# SPH V0 Design: Buffer Layout + Leapfrog Pipeline

V0 scope = single-GPU δ-plus WCSPH rewrite in Vulkan. Match or exceed OpenGL baseline (1M @ 330 fps on 5090). Lays the data-layout and pipeline-skeleton foundation that V1+ multi-GPU builds on. See `sph_design.md` for the V0–V3 arc and multi-GPU rationale.

**Authoritative sources for values**: `shaders/sph/common.glsl` for spec constants + descriptor bindings, `shaders/sph/README.md` for per-kernel invariants. When this document and those files disagree, trust them — this doc describes intent and rationale.

## Scope Decisions Already Made

- Integration: **Leapfrog** (half-step velocity, single full-step kick per iteration). Algebraically equivalent to velocity Verlet KDK but 5 kernels/step instead of 6. Replaces OpenGL baseline's Euler.
- Neighbor search: persistent uniform voxel grid with **incremental cell list** (in/out event flow), not per-step radix sort.
- Algorithm: δ-plus WCSPH with KCG (kernel gradient correction) and PST (particle shifting).
- Extensions deferred: micropolar, multi-resolution, thermal. Base δ-SPH only for V0.
- Bit-packing rejected: separate bindings over packed high-bit fields (see `feedback_no_bitpack` in memory). Readability > marginal savings.
- **1-based indexing throughout**: particle_id ∈ [1, POOL_SIZE], voxel_id ∈ [1, TOTAL_VOXEL_COUNT], slot entries ∈ [1, POOL_SIZE]. Slot `0` is the universal empty / dead sentinel. Particle buffers sized `POOL_SIZE + 1`, voxel buffers `TOTAL_VOXEL_COUNT + 1` in the count dimension.

## Descriptor Set Layout

Four descriptor sets, organized by update frequency:

- **set 0** — Own particle SoA (canonical `density_pressure` + transient `density_pressure_scratch`; the simulator step cmd copies scratch → canonical inside each step)
- **set 1** — Own voxel cell structures (bound once)
- **set 2** — Ghost particles + ghost voxel structures (V1 multi-GPU; V0-a: bound to dummy, `GHOST_DIMENSION_*=0` makes all ghost branches dead-code-eliminated)
- **set 3** — Global status, transport, material parameters, diagnostics (bound once)

Density uses **scratch + copy** (not ping-pong). `density.comp` writes its new ρ/P into the transient `density_pressure_scratch` (binding 2); the simulator's step cmd then issues `vkCmdCopyBuffer scratch → primary` inside the same submission, so by `force.comp` the canonical `density_pressure` (binding 1) already holds ρ_{n+1}, P_{n+1}. Single descriptor set, no parity bookkeeping.

## Buffer Layout (V0)

### set 0: Particle SoA

Unified pool — fluid, boundary, inlet, rotor share slots, differentiated by `material[pid]` indexing into `material_parameters[]` (set 3 binding 7) to get `.kind`.

| binding | Buffer | Type | Size/particle | Access pattern |
|---|---|---|---|---|
| 0 | `position_voxel_id` | `vec4` (x, y, z, voxel_id_as_float) | 16 B | Hot (every neighbor loop) |
| 1 | `density_pressure` | `vec2` (ρ, P) | 8 B | canonical (post-copy each step) |
| 2 | `density_pressure_scratch` | `vec2` (ρ, P) | 8 B | density.comp's transient write target |
| 3 | `velocity_mass` | `vec4` (vx, vy, vz, mass) | 16 B | Force / density read v; predict writes v; mass preserved |
| 4 | `acceleration` | `vec4` (ax, ay, az, _) | 16 B | **Persistent** (next step's kick consumes a_n) |
| 5 | `shift` | `vec4` (sx, sy, sz, _) | 16 B | **Persistent** (drift needs shift_n) |
| 6 | `material` | `uint` | 4 B | group_id into `material_parameters[]` |
| 7 | `correction_inverse` | `vec4 × 2` (symmetric M⁻¹: m00, m11, m22, m01, m02, m12) | 32 B | Correction writes; density + force read |
| 8 | `density_gradient_kernel_sum` | `vec4` (∇ρ.xyz, kernel_sum) | 16 B | Correction writes; density + force read |
| 9 | `extension_fields` | `vec4` (reserved: temperature, etc.) | 16 B | V0 untouched; future scalar diagnostics |

Core = **132 B/particle** (binding 0–8), +16 B for `extension_fields` if used.

**Layout notes:**

- `voxel_id` stored as `float(vid)` in `position_voxel_id.w` (1-based, 0 = dead). Decode: `uint vid = uint(round(position_voxel_id[pid].w))`. Single hot load gets `pos + vid`. Using `round()` not `floatBitsToUint` because the small 1-based integer fits exactly in a float and `round()` is cheaper to reason about.
- `density_pressure` staging: density.comp writes the scratch buffer at binding 2; the simulator's step cmd issues `vkCmdCopyBuffer scratch → primary` (with surrounding compute↔transfer barriers) inside the same submission, so force.comp reads the freshly-written ρ from binding 1.
- `correction_inverse` packs symmetric 3×3 M⁻¹ into 2 vec4:
  - `[pid*2]     = (m00, m11, m22, m01)`
  - `[pid*2 + 1] = (m02, m12, _, _)`
- `acceleration` and `shift` kept separate — predict needs both, but correction / density / force don't need `shift`. Splitting avoids wasted loads.
- `mass` kept per-particle (future-proofing for multi-phase / inlet-size variation).

### set 1: Voxel Buffers (SoA, no embedded header)

| binding | Buffer | Type | Size/voxel | Access |
|---|---|---|---|---|
| 0 | `inside_particle_count` | `uint` | 4 B | Hot (every neighbor query) |
| 1 | `incoming_particle_count` | `uint` (atomic) | 4 B | Predict atomic-appends; update_voxel consumes + resets |
| 2 | `inside_particle_index` | `uint × MAX_PARTICLES_PER_VOXEL` (V0: 96) | 384 B | Per-voxel particle_id list (1-based entries) |
| 3 | `incoming_particle_index` | `uint × MAX_INCOMING_PER_VOXEL` (V0: 16) | 64 B | Per-voxel incoming-crossing events |

Per-voxel total ≈ **456 B**.

**Design notes:**

- **No stored header**: `voxel_id ↔ (x, y, z)` derivable from linear index via `helpers.glsl::own_coord_of` / `own_voxel_id_of`.
- **No separate out_buffer**: the canonical "which voxel is this particle in" is `position_voxel_id[pid].w`. `update_voxel` compacts by filtering `decode(PosVid[pid].w) == self_voxel_id`; particles that moved out or died (vid=0) naturally fail the check. Rejected multi-source-of-truth.
- **Voxel encoding**: linear z-major (`VOXEL_ORDER=0`, constant_id=20). Future Morton / 4×4×4 tile reserved.
- **Irregular domain**: V0 allocates dense grid over `frame.obj` bounding box; dead voxels (`inside_count=0`) are zero-compute and cost 456 B each. Sparse hash / active mask deferred.
- `MAX_PARTICLES_PER_VOXEL = 96` with ~1.5× headroom over observed OpenGL peak ~64. Overflow counter in `global_status.overflow_inside_count` for validation.
- `MAX_INCOMING_PER_VOXEL = 16` guess (~2× CFL-limited per-step crossing rate); validate with `overflow_incoming_count`.

### set 2: Ghost (V1 multi-GPU, V0-a: dummy-bound)

Parallels set 0 layout for ghost particles received from peer GPU, plus per-ghost-voxel inside lists:

- `ghost_position_voxel_id`, `ghost_density_pressure`, `ghost_velocity_mass`, `ghost_acceleration`, `ghost_shift`, `ghost_material` (bindings 0, 1, 3, 4, 5, 6)
- `ghost_inside_particle_count`, `ghost_inside_particle_index` (bindings 10, 12)

V0-a: `GHOST_DIMENSION_* = 0` → all ghost branches dead-code-eliminated by `-O`. Set 2 can bind a single dummy buffer.

### set 3: Global / Transport / Materials / Diagnostics

| binding | Buffer | Purpose |
|---|---|---|
| 0 | `global_status` | `alive_particle_count`, `frame_counter`, `maximum_velocity`, overflow counters, `correction_fallback_count` (64 B = 1 cache line) |
| 1 | `overflow_log` | Ring buffer of overflow events `(voxel_id, step, kind, lost_pid)` |
| 2 | `inlet_template` | Template particle states for inlet spawn (V0+) |
| 3 | `dispatch_indirect` | For `vkCmdDispatchIndirect` (V0+, when alive_count varies) |
| 4 | `ghost_out_packet` | V1+ send-side pack buffer |
| 5 | `ghost_in_staging` | V1+ receive-side staging |
| 6 | `diagnostic` | Optional debug build: curl, FTLE, vorticity |
| 7 | `material_parameters` | Per-group SPH parameters (48 B × N_groups), indexed by `material[pid]` |

## Specialization constants

Python side (`utils/sph/config.py` + `case.py`) assembles a `VkSpecializationInfo` from a `SimulationConfig` parsed from `cases/*/case.yaml`, merged with `materials/standard.yaml`. Per-material parameters (rest_density, viscosity, eos_constant, radius, volume, ...) are **not** spec constants — they live in `material_parameters[]` buffer indexed by per-particle `material` field.

ID range plan (authoritative in `common.glsl`):

```
0  - 9   : core physics scalars + own grid origin   (id=3 reserved)
10       : multi-GPU bit-exactness toggle (V1)
11 - 13  : own grid dimensions
14 - 16  : correction regularization tunables
17 - 19  : gravity
20 - 29  : voxel layout / micropolar (V0+ reserved)
30 - 33  : dimension + kernel coefficients
34 - 39  : reserved
40 - 49  : SPH numerical parameters (ε_h², PST main, PST anti, ...)
50 - 53  : capacities + workgroup size + pool size
54 - 79  : reserved
80 - 88  : multi-GPU ghost grid (V1)
89 - 127 : reserved
```

## Pipeline Stages (Leapfrog, 5 stages)

Entering step `n`: `x_n`, `v_{n-1/2}`, `a_n`, `ρ_n`, `shift_n` all valid from previous step's end. Stored velocity is at half-step time.

```
1. predict          (per-particle dispatch)
     reads:  position_voxel_id, velocity_mass, acceleration, shift, material
     writes: position_voxel_id (new x, new vid), velocity_mass (new v_half),
             incoming_particle_index + incoming_particle_count (atomic append on crossing)
     logic:  v_{n+1/2} = v_{n-1/2} + a_n · dt              (full-step kick)
             x_{n+1}   = x_n + v_{n+1/2} · dt + shift_n    (drift with PST shift)
             new_vid   = voxel_of(x_{n+1})
             if (new_vid != old_vid) atomic-append pid into incoming_particle_index[new_vid]
             Open boundary: drift outside grid → kill (voxel_id=0, pos+vel zeroed)
             Incoming overflow: log + kill

2. update_voxel     (per-voxel dispatch, one thread per voxel)
     reads:  inside_particle_index + inside_particle_count (old),
             incoming_particle_index + incoming_particle_count,
             position_voxel_id (for vid match)
     writes: inside_particle_index (compacted + merged), inside_particle_count (new),
             incoming_particle_count = 0
     MUST run before correction (correction's neighbor search uses inside_particle_index).

3. correction       (per-particle, 27-voxel neighbor loop)
     reads:  position_voxel_id, velocity_mass (mass via .w), density_pressure,
             inside_particle_count, inside_particle_index
     writes: correction_inverse (symmetric M⁻¹),
             density_gradient_kernel_sum (∇ρ.xyz, kernel_sum)

4. density          (per-particle, 27-voxel neighbor loop)
     reads:  position_voxel_id, velocity_mass, density_pressure,
             correction_inverse, density_gradient_kernel_sum,
             inside_particle_count, inside_particle_index, material_parameters
     writes: density_pressure_scratch (new ρ_{n+1}, new P_{n+1} via Tait EOS).
             Simulator immediately copies scratch → density_pressure inside
             the step cmd (compute→transfer→compute barriers).

5. force            (per-particle, 27-voxel neighbor loop)
     reads:  position_voxel_id, velocity_mass, density_pressure (canonical ρ_{n+1}),
             correction_inverse, density_gradient_kernel_sum (kernel_sum via .w),
             material, material_parameters,
             inside_particle_count, inside_particle_index
     writes: acceleration (a_{n+1}), shift (shift_{n+1} via PST)

END: x_{n+1}, v_{n+1/2}, a_{n+1}, ρ_{n+1}, shift_{n+1} all ready for step n+1.
No descriptor parity to swap: density_pressure is canonical at binding 1 throughout.
```

There is **no step 6** (no post-force kick). Leapfrog absorbs Verlet KDK's second half-kick into the next step's predict: `v_{n+1/2} = v_{n-1/2} + a_n · dt` combines (kick2 of step n−1) + (kick1 of step n) into one full-step kick.

### Bootstrap (one-time at t = 0)

Main loop needs `a_0` and `v_{-1/2}` (**backward** half-step velocity). Initial conditions from `case.yaml` + `.obj` geometry give `x_0` and `v_0` (integer-step velocity, typically 0 for dam-break).

Startup sequence:

1. **Python-side initial voxelization**: fill `inside_particle_index` / `inside_particle_count` from `x_0` (skip inlet particles per the InsideParticleIndexBuffer invariant). Done at simulator load time, **not** via `update_voxel.comp`.
2. `correction.comp` on x_0 → M_0⁻¹, ∇ρ_0, kernel_sum_0
3. `density.comp` on x_0 → ρ_0, P_0
4. `force.comp` on x_0, v_0, ρ_0 → a_0, shift_0
5. `bootstrap_half_kick.comp` → v_{-1/2} = v_0 − 0.5 · a_0 · dt

Then enter the main 5-stage loop at step 1. Step 1's predict computes:

```
v_{1/2} = v_{-1/2} + a_0 · dt = v_0 + 0.5 · a_0 · dt    (correct half-step)
x_1     = x_0 + v_{1/2} · dt + shift_0
        = x_0 + v_0·dt + 0.5·a_0·dt² + shift_0          (Taylor 2nd order ✓)
```

The **backward** offset is critical: predict's kick consumes `acceleration = a_n` (from previous force), so step 1's kick consumes `a_0`. Forward bootstrap (`v_{1/2}`) would feed step 1 a stale `a_0` producing `v_{3/2}` instead of `v_{1/2}`, permanent O(dt²) trajectory error.

### Periodic: Defrag Pass (`defrag.comp` — implemented)

Re-sort particle SoA so each voxel's particles occupy a contiguous range `[base_v, base_v + count_v)` → restores spatial locality after particles diffuse. Per-voxel dispatch (one thread per voxel, 1-based vid).

Implementation reads the current SoA on `set 0` and writes a scratch SoA on `set 4` that mirrors set 0's layout. CPU-side bank swap after the dispatch promotes set 4 to be the next step's set 0.

Two base-index strategies via `USE_PREFIX_SUM_DEFRAG` spec constant (id=46):

- `true` — `base = voxel_base_offset[voxel_id]`, deterministic voxel-id order. Requires a prefix-sum pass over `inside_particle_count[]` first.
- `false` — `base = atomicAdd(defrag_scratch_counter, count)`. Non-deterministic per-voxel order but voxel-internal contiguity preserved (sufficient for the cache-locality goal).

Both paths preserve correctness (total count, voxel→particle assignment); path A additionally guarantees inter-voxel contiguity for stronger cache behavior.

Trigger policy: TBD — static cadence (~256 steps) vs GPU-side fragmentation metric. Currently driver-side from Python.

## Key Differences vs OpenGL Baseline

| Aspect | OpenGL | Vulkan V0 |
|---|---|---|
| Integrator | Euler | Leapfrog (half-step velocity) |
| Particle buffer | 3 types × 3 sub-buffers = 9 SSBOs | 1 unified SoA pool, SoA-split by field lifetime |
| Per-particle memory | ~512 B (ParticleData + SubData + RuntimeData mat4) | **132 B core** (4× reduction; +16 B optional extension) |
| Voxel size | 2912 ints = 11.4 KB/voxel | **~456 B/voxel** (25× reduction), no stored header |
| Shader constants | GLSL `#define` injection | `VkSpecializationInfo` |
| Kernel dispatch | One giant compute shader + `stage` uniform branch | One `VkComputePipeline` per stage (6 total incl. bootstrap) |
| Stage count | 5 (Euler) | 5 (Leapfrog) + 1 one-time bootstrap |
| `update_voxel` position | Step tail | After predict, before correction (drift moved it up) |
| Boundary dispatch | Sign-of-index sentinel (`pid = -pid`) | `material[pid] → MaterialParameters.kind` branch |
| Indexing | 0-based | 1-based (0 = universal sentinel) |
| Material parameters | Per-particle spec-const injected | Per-group `material_parameters[]`, indexed by `material[pid]` |

## Cross-Stage Field Audit

From OpenGL `ParticleRuntimeData`, fields retained as persistent cross-stage state in V0:

- **Kept**: M⁻¹ (→ `correction_inverse`), ∇ρ + kernel_sum (→ `density_gradient_kernel_sum`), new_ρ / new_P (→ `density_pressure_scratch`, copied to `density_pressure` each step), particle_shift (→ `shift`)
- **Cut** (kernel-local, computed in registers): `kernel_gradient_sum`, per-particle `kernel_sum` accumulator outside correction, `kernel_normalize_matrix` accumulation
- **Cut** (extensions, deferred): `curl_u`, `w`, `overlap`, `aw`, `mass_transfer_target_particle`, `FTLE`
- **Reserved slots** (debug/render builds only): `extension_fields` (set 0 binding 9) + optional `diagnostic` buffer (set 3 binding 6)

## Open Items — Status

| Item | Status |
|---|---|
| Kernel workgroup size | ✓ `WORKGROUP_SIZE = 128` (id=51, per-kernel overridable via `local_size_x_id`) |
| Density staging | ✓ scratch+copy: density.comp writes scratch (binding 2), step cmd copies scratch → primary (binding 1) inside same submission. No parity. |
| Bootstrap pass implementation | ✓ `bootstrap_half_kick.comp` ready; Python orchestrator pending in `utils/sph/simulator.py` |
| Overflow validation counters | ✓ `global_status.overflow_inside_count`, `overflow_incoming_count`, first-offending-voxel via `atomicCompSwap` |
| CAP tuning (96 / 16) | ⏳ Pending live profiling on real scenes |
| Defrag | ✓ `defrag.comp` implemented (per-voxel scatter, prefix-sum or atomic-counter base via `USE_PREFIX_SUM_DEFRAG` spec const, bank-swap from set 4 → set 0); trigger policy still TBD |
| Numerical validation (Vulkan vs OpenGL) | ⏳ Gated on Python pipeline + scene loader |
| 27-neighbor loop shared-memory tiling | ⏳ Deferred to V0 perf pass |
| Irregular-domain active-voxel mask | ⏳ V0+ deferred; V0 uses full frame-bbox grid |
| Inlet spawn kernel | ⏳ V0+ (pool includes inlet templates, static in V0) |

## File Organization (current)

```
shaders/sph/
  common.glsl                  # spec constants, descriptor bindings, kind tags, sentinels
  helpers.glsl                 # Wendland C4, voxel_id↔coord, correction_inverse unpack
  predict.comp                 # stage 1
  update_voxel.comp            # stage 2
  correction.comp              # stage 3
  density.comp                 # stage 4
  force.comp                   # stage 5
  bootstrap_half_kick.comp     # one-time backward half-kick
  _test_common.comp            # smoke test
  README.md                    # handoff document

utils/sph/                     # Python pipeline glue — PENDING
  config.py                    # SimulationConfig + VkSpecializationInfo assembly
  case.py                      # yaml + obj loader + derived quantities
  vulkan_context.py            # instance / device / queues / command pool
  shader_loader.py             # SPV → VkShaderModule
  buffers.py                   # SoA allocation + staging upload
  scene.py                     # initial particle pool + voxelization
  descriptors.py               # 4 set layouts + pool + single set 0 instance
  pipelines.py                 # 6 compute pipelines
  dispatch.py                  # command buffer recording
  simulator.py                 # top-level driver

cases/
  lid_driven_cavity_2d/
    case.yaml                  # parameters + obj references
    domain.obj                 # vertex-as-particle (fluid)
    wall.obj                   # vertex-as-particle (U-shaped static wall)
    wall_top.obj               # vertex-as-particle (lid: kind=boundary + initial_velocity)
    frame.obj                  # computation bbox (V0: only bbox used)

materials/
  standard.yaml                # reusable material library (water, wall, ...)

utils/phase1/                  # Archived: Phase 1 cross-GPU migration demo
```
