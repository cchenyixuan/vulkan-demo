# SPH V0 Compute Shaders

Vulkan compute shader implementation of δ-plus WCSPH for the V0 single-GPU
milestone. See `docs/sph_design.md` (multi-GPU architecture) and
`docs/sph_v0_design.md` (this milestone's buffer layout + pipeline) for the
full design.

This README is a quick orientation for anyone (including future Claude
sessions) opening this directory. Authoritative details live in the design
docs and in `common.glsl`.

---

## Pipeline (Leapfrog, 5 main stages + bootstrap)

```
Bootstrap (once at t = 0)
    ┌────────────────────────────────┐
    │ Python: voxelize x_0 → buffers │
    │ correction → density → force   │  produces a_0, ρ_0, P_0, shift_0, M_0⁻¹
    │ bootstrap_half_kick            │  v_0 → v_{-1/2}
    └────────────────────────────────┘
                 │
                 ▼
Main loop (uniform, every step):
    1. predict        kick (v_{n-1/2} → v_{n+1/2}) + drift + voxel-crossing detection
    2. update_voxel   compact inside_particle_index + merge incoming events
    3. correction     KCG matrix M⁻¹ + ∇ρ + kernel_sum
    4. density        continuity (ρ_{n+1}) + Tait EOS (P_{n+1})
    5. force          pressure (TIC) + viscosity (Morris) + gravity + PST shift
                          → a_{n+1}, shift_{n+1}
```

Stage order matters. density.comp writes the new ρ/P to the transient
`density_pressure_scratch`; the simulator immediately copies that back to
the canonical `density_pressure` (set 0 binding 1) inside the same step
cmd, so force.comp sees the freshly-written values. The next step's
predict consumes force's outputs (`acceleration`, `shift`).

Multi-GPU ghost handling and inlet/outlet kernels are deferred (V0+ work). `defrag.comp` is implemented (separate path from the per-step main loop, triggered periodically — see file inventory below).

---

## File inventory

| File | Purpose |
|---|---|
| `common.glsl` | All spec constants, descriptor set bindings, scalar constants (kind tags, sentinels). Single source of truth. |
| `helpers.glsl` | Shared device functions: Wendland C4 kernel + gradient, voxel coord conversions, symmetric mat3 unpack. |
| `bootstrap_half_kick.comp` | One-time backward half-kick at t=0 (`v_0 → v_{-1/2}`). |
| `predict.comp` | Stage 1 — kick + drift + crossing detection. |
| `update_voxel.comp` | Stage 2 — per-voxel compaction + incoming merge. |
| `correction.comp` | Stage 3 — KCG correction matrix + density gradient + kernel sum. |
| `density.comp` | Stage 4 — continuity equation + EOS. |
| `correction_density.comp` | E39 B1 (`V7_FUSED_CORRECTION_DENSITY`, default 1): stages 3 + 4 in one neighbour traversal (correction.comp's code, then density's rate as L_i : S_i with the L_i just written); every site where correction and density run back to back, on slabs whose two kernels take the same particles. B4 variant `correction_density_deep_wall_skip` (`-DDEEP_WALL_SKIP`). |
| `force.comp` | Stage 5 — pressure + viscosity + gravity + PST shift. |
| `wall_extrapolate.comp` | E37 (`numerics.wall_boundary: adami`): the Adami wall pass between density and force (wall pressure + dummy velocity from the fluid neighbours); every slab without a peer (K = 1), the kernel of v7-rc2 byte for byte. |
| `wall_extrapolate_band.comp` | E40: the same computation with the interior / boundary-band modes and the band-voxel dispatch, for an adami slab with a peer (three-layer ghost, `partition_v7.chain_ghost_layers`): the interior pass in phase B, the band pass + the inner ghost column as self in phase C. |
| `wall_boundary.glsl` | The wall option (`numerics.wall_boundary`, spec const 100): simple, adami, crossing. Its last section (behind `#ifdef WALL_CROSSING`) is the crossing mode: the rules, spec consts 120-127, the transient `wall_crossing_fluid_inverse` buffer (set 0 binding 12) and the two functions of the geometry (`wall_crossing_distance_to_wall`, `wall_crossing_depth_behind_wall`: an axis-aligned box). With `numerics.wall_crossing.geometry: pairwise` (spec const 128, off by default) the geometry is one function, `wall_crossing_pair_geometry` (the face the segment i -> j leaves the box through: its normal, the pair's d_ij / phi_ij, the wall particle's mirror point), which the two reflections (`pressure_wall: reflection`, spec const 129; `delta_wall: reflection`, spec const 116 = 3) build on. |
| `correction_density_wall_crossing` / `force_wall_crossing` (variants) | `numerics.wall_boundary: crossing`: `correction_density.comp` / `force.comp` compiled with `-DWALL_CROSSING` (`compile_shaders_v7.py` `SHADER_VARIANTS`). A crossing slab builds its `correction_density_*` and `force_*` pipelines from them; the plain builds (what simple and adami run) keep their bytes. No wall pass: a wall particle keeps rho0 and its prescribed velocity, a fluid particle continues its own fields across the wall. |
| `_test_common.comp` | Smoke test: empty kernel including `common.glsl`, used to detect regressions in bindings/spec constants. |

Compiled `.spv` artifacts coexist alongside source. Build via
`compile_shaders.py` at repo root. `_test_common.comp` is auto-skipped from
optimized build (compiled once unoptimized as a sanity check).

---

## Conventions

**1-based indexing throughout the SPH pipeline.** Every id is in `[1, N]`,
with `0` reserved as the universal "unallocated / dead / empty" sentinel:

- `particle_id ∈ [1, POOL_SIZE]` — slot 0 of every particle buffer is unused
- `voxel_id ∈ [1, TOTAL_VOXEL_COUNT]` — slot 0 of every voxel buffer is unused
- `slot_entry ∈ [1, POOL_SIZE]` (when storing particle refs in voxel buffers)

Shader entry: `uint self_particle_id = gl_GlobalInvocationID.x + 1u;`
followed by `if (self_particle_id > POOL_SIZE) return;` for padding threads.

Per-voxel kernels (currently only `update_voxel`) use `TOTAL_VOXEL_COUNT`
in the bound check.

**Voxel coords stay 0-based** (spatial coord `0..GRID_DIM-1`). The 1-based
voxel_id is encoded as `coord.x + coord.y·Dx + coord.z·Dx·Dy + 1` (the +1
makes 0 a valid sentinel). `helpers.glsl` provides `own_coord_of(vid)` and
`own_voxel_id_of(coord)` to bridge the two.

**Shared helpers go in `helpers.glsl`**. Per-shader local helpers (functions
only one shader uses) stay local to that .comp file.

**No abbreviations in identifiers**. `correction_matrix` not `M`,
`smoothing_length` not `h`, `neighbor_particle_id` not `pid_j`. Math
notation (W, ρ, ∇, ξ) is fine in comments.

**`#include "common.glsl"` then `#include "helpers.glsl"`** at the top of
every `.comp` file. Both are header-guarded.

---

## Material / kind handling per kernel

| Kernel | FLUID | BOUNDARY | ROTOR | INLET | DEAD (vid=0) |
|---|---|---|---|---|---|
| `bootstrap_half_kick` | run | skip | skip | skip | skip |
| `predict` | run | skip | skip | skip | skip |
| `update_voxel` | per-voxel kernel; not per-particle | | | | |
| `correction` | run | run | run | run* | skip |
| `density` | run | run | run | skip | skip |
| `force` | run | run | run | skip | skip |
| `defrag` (TBD) | — | — | — | — | — |

\* `correction` does not skip INLET because doing so would require pulling in
`material[pid] + material_parameters[group]` (~52 B/particle) just for the
kind check — for typical inlet population (< 1% of pool) this costs more than
it saves. INLET particles' correction outputs are written but unread (they
never appear as neighbors).

ROTOR particles compute the full force/density/correction in V0 even though
predict skips them (no motion). This is a deliberate carry-over from legacy;
rotor coupling will be revisited in V0+.

---

## Buffer access matrix

Reads (R) and Writes (W) per kernel. See `common.glsl` for binding numbers.

| Buffer (set 0 unless noted) | bootstrap | predict | update_voxel | correction | density | force |
|---|---|---|---|---|---|---|
| `position_voxel_id` | R | R/W | R/W (kill) | R | R | R |
| `density_pressure` | — | — | — | R | R | R |
| `density_pressure_scratch` | — | — | — | — | W | — |
| `velocity_mass` | R/W | R/W | R/W (kill) | R (mass via .w) | R | R |
| `acceleration` | R | R | — | — | — | W |
| `shift` | — | R | — | — | — | W |
| `material` (group_id) | R | R | — | — | R | R |
| `correction_inverse` | — | — | — | W | R | R (self) |
| `density_gradient_kernel_sum` | — | — | — | W | R | R (.w only, self) |
| `inside_particle_count` (set 1) | — | — | R/W | R | R | R |
| `inside_particle_index` (set 1) | — | — | R/W | R | R | R |
| `incoming_particle_count` (set 1) | — | atomic R/W | R/W (reset) | — | — | — |
| `incoming_particle_index` (set 1) | — | atomic W | R | — | — | — |
| `material_parameters` (set 3) | R (kind) | R (kind) | — | — | R (kind, eos, ρ_0) | R (kind, ν, R, V) |
| `global_status` (set 3, atomics) | — | atomic W on overflow | atomic W on overflow | atomic W on fallback | — | — |

Notes:
- Density staging is **scratch+copy**: density.comp writes
  `density_pressure_scratch` (binding 2); the simulator's step cmd then
  issues `vkCmdCopyBuffer scratch → primary` so by force.comp the
  canonical `density_pressure` (binding 1) holds ρ_{n+1}, P_{n+1}.
  No ping-pong, no descriptor parity.
- `correction` does not read `material` to skip INLET (see notes above).
- `force` is the most expensive: reads neighbor M, ∇ρ, mass, density, vel for
  pair-averaged formulas; uniform V0 lets us compute pair quantities from
  self only (see `force.comp` header for what's hoisted).

---

## Invariants (consumed across kernels)

These are written into `common.glsl`'s buffer comments where relevant.
Summarized here for orientation:

1. **Contiguous packing** in `inside_particle_index[v * CAP + slot]`: valid
   particle_ids occupy slots `[0, count)`; trailing slots are 0.
   Maintained by `update_voxel.comp`'s Phase 3.

2. **No inlet/dead in voxel buffers**. `inside_particle_index` and
   `incoming_particle_index` never contain inlet or dead particle_ids.
   - `predict.comp` never atomic-appends inlet (kind check) or dead
     (skipped early).
   - Initial Python-side voxelization must skip inlet particles when
     populating `inside_particle_index`.
   - `update_voxel.comp`'s Phase 1 filter (voxel_id mismatch) automatically
     drops dead particles (their voxel_id = 0 ≠ self).

3. **Incoming counter may exceed `MAX_INCOMING_PER_VOXEL`** when predict
   overflows. `update_voxel.comp` clamps via `min(count, CAP)` before
   reading.

4. **1-based id everywhere** with 0 = sentinel (see Conventions above).

5. **Bit-exact integration on Kernel A on ghost** (V1 multi-GPU only;
   not active in V0).

---

## Bootstrap convention (backward half-kick)

Predict's kick formula `v_{n+1/2} = v_{n-1/2} + a_n · dt` consumes the
acceleration left in the buffer by the previous step's force. For step 1 to
produce correct `v_{1/2}` from buffer-stored `a_0`, the buffer must hold
**`v_{-1/2}` (backward offset)**, computed as:

```
v_{-1/2} = v_0 - 0.5 · a_0 · dt
```

`bootstrap_half_kick.comp` does exactly this. Forward bootstrap
(`v_{1/2} = v_0 + 0.5 a_0 dt`) would feed step 1 a stale `a_0` instead of
fresh `a_1`, introducing a permanent O(dt²) trajectory error from step 1.

Full bootstrap sequence (run once before main loop):

1. Python: populate particle buffers from scene IC; run initial voxelization
   to fill `inside_particle_index` (skip inlet).
2. `correction.comp` on x_0
3. `density.comp` on x_0 → ρ_0, P_0
4. `force.comp` on x_0, v_0, ρ_0 → a_0, shift_0
5. `bootstrap_half_kick.comp` → v_{-1/2}
6. Enter main loop at step 1.

---

## Compile

`python compile_shaders.py` at repo root produces `.comp.spv` for every
`.comp` (and the smoke test). Uses `glslc` from VULKAN_SDK with:

- `-O` optimization (dead-code-eliminates unused bindings declared in
  `common.glsl`)
- `-I shaders/sph` so `#include` resolves
- `--target-env=vulkan1.2`

Underscore-prefixed `.comp` (currently only `_test_common.comp`) compile as
smoke tests, unoptimized, useful as regression check after `common.glsl`
edits.

---

## Spec constant ID range plan (see `common.glsl`)

```
0  - 9   : core physics scalars + own grid origin  (id=3 reserved)
10       : multi-GPU bit-exactness toggle (V1)
11 - 13  : own grid dimensions (and TOTAL_VOXEL_COUNT derived)
14 - 16  : correction regularization tunables
17 - 19  : gravity
20 - 29  : voxel layout / micropolar (V0+ reserved)
30 - 33  : dimension + kernel coefficients
34 - 39  : reserved
40 - 49  : SPH numerical parameters (ε_h², PST main, PST anti, ...)
50 - 53  : capacities + workgroup size + POOL_SIZE
54 - 79  : reserved
80 - 88  : multi-GPU ghost grid (V1)
89 - 127 : reserved
```

The plan above is the V0 one; `common.glsl`'s header has the current table. Ids in use in this tree
(`grep -rhoE "constant_id *= *[0-9]+" *.glsl *.comp`): 0-21, 30-33, 40-62, 80-109, 110 (`local_size_x_id` of
`ghost_send_lanes.comp`), 111-115, and for `numerics.wall_boundary: crossing` (declared in `wall_boundary.glsl`,
compiled only into the `WALL_CROSSING` variants):

| id | name | meaning |
|---|---|---|
| 100 | `WALL_BOUNDARY` | 0 simple, 1 adami, **2 crossing** |
| 120 | `WALL_CROSSING_CONTINUITY` | wall neighbour's velocity difference in the continuity equation: 0 extrapolate, 1 wall_velocity |
| 121 | `WALL_CROSSING_DISTANCE_FLOOR` | lower bound of phi_i in metres (`distance_floor_factor` x particle spacing) |
| 122 - 124 | `WALL_CROSSING_BOX_LOWER_X/Y/Z` | the fluid box, lower corner (world coordinates) |
| 125 - 127 | `WALL_CROSSING_BOX_UPPER_X/Y/Z` | the fluid box, upper corner |
| 116 | `WALL_CROSSING_DELTA_WALL` | **diagnostic**, not part of the mode (`numerics.wall_crossing.delta_wall`): a wall neighbour's density diffusion, 0 extension (the mode: the density continued across the wall, applied after the loop), 1 rest_density (simple's pair term rho0 - rho_i, nothing after the loop), 2 body_force (pair term rho_i g . (r_j - r_i) / c^2 with c^2 = dp/drho of the Tait EOS at rho_i and a_wall = 0, nothing after the loop; no wall contribution without gravity); **3 reflection** (not a diagnostic; needs geometry pairwise): the continued density at the wall particle's mirror point r_j' plus rho_i g . (r_j - r_j') / c^2, i.e. value 0's term and, after the loop, -2 delta h c0 (G_i - rho_i g / c^2) . sum_wall V_j |grad W_ij| r / (r^2 + eps^2) (r_j - r_j') (zero when A_f fell back) |
| 117 | `WALL_CROSSING_CONTINUITY_WALL_COMPONENT` | `numerics.wall_crossing.continuity_wall_component`: which part of a wall neighbour's velocity difference enters the continuity equation, **1 normal (the default since 2026-11-18)**: its component along the wall's inward normal, the unit vector from the wall particle to the nearest point of the box, before the continuity rule; 0 full: the whole vector (the default until then, kept as a switch). The viscous term takes the whole difference with either value |
| 118 | `WALL_CROSSING_VISCOUS_EXTENSION` | `numerics.wall_crossing.viscous_extension`: the continuation of the velocity in force's viscous term, **0 linear (the default)**: (v_w - v_i) (1 + a_j), a_j = d_j / phi_i; 1 quadratic: (v_w - v_i) (1 + a_j)^2 - a_j grad v_i (r_j - r_i), grad v_i = M_0 L_i (I - K_raw L_i)^-1 solved per particle in the fused pass and stored in `wall_crossing_velocity_gradient` (set 0 binding 14: three vec4 per particle, column b of grad v in `.xyz`, the state 0 no wall neighbour / 1 solved / 2 fallback in the first `.w`; a 48-byte stub when both extensions are linear). A particle falls back to the first-order rule for the step when det(I - K_raw L_i) < 0.5, the solution is not finite or L_i itself fell back. |
| 119 | `WALL_CROSSING_CONTINUITY_EXTENSION` | `numerics.wall_crossing.continuity_extension`: the same for the continuity equation, **0 linear (the default)**: the pair terms of constants 120 and 117; 1 quadratic: the wall neighbours' drift is -rho_i tr[(M_w + grad v_i K_raw) L_i] after the loop (the whole vector; a particle that falls back takes the first-order pair terms). Refused by the loader with continuity wall_velocity or without viscous_extension quadratic. |
| 128 | `WALL_CROSSING_GEOMETRY` | `numerics.wall_crossing.geometry`: where a pair's crossing fraction and wall normal come from, **0 nearest_wall (the default)**: phi_i = the distance of the fluid particle to its nearest wall plane, d_j = the distance of the wall particle to the box, n_j = the unit vector from the wall particle to the nearest point of the box; 1 pairwise: from the face the segment i -> j leaves the box through (the out-of-box axis of j with the smallest s_a = phi_a / (phi_a + d_a)): phi_ij = the distance of i to that face (floored), d_ij = the depth of j behind it, n_ij = its inward normal, 1 / t_ij = 1 + d_ij / phi_ij (`wall_crossing_pair_geometry`). Refused by the loader together with a quadratic extension. |
| 129 | `WALL_CROSSING_PRESSURE_WALL` | `numerics.wall_crossing.pressure_wall`: a wall neighbour's pressure in force, **0 extension (the default)**: P_i + G_i . (r_j - r_i); 1 reflection (needs geometry pairwise): the continued pressure at the mirror point plus the body force, P_i + G_i . (r_j' - r_i) + rho_i g . (r_j - r_j'), i.e. force adds (1 / rho_i) L_i M_d (G_i - rho_i g) to the pressure acceleration after the loop, M_d = sum_wall V_j grad W_ij (x) (r_j - r_j') (zero when A_f fell back). |
| 130 | `WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED` | `numerics.volume_constraint.enabled` (**0 = off, the default**; crossing only, `utils/volume_constraint_v7.py`): the constraint on the fluid's total volume in a closed container. With 1 the fused pass adds rho_i x rate to every fluid particle's density rate, rate = `volume_constraint_state.x` (set 0 binding 15, 16 bytes: (rate, theta, 0, 0)), which the host writes at every update: theta = (V_now - V_target) / V_target with V = sum of m_i / rho_i over the fluid particles, rate = theta / relaxation_time. The buffer exists on every crossing slab and is zero until the first update. |

Free: none in 116-130. `_test_seam_layout.py` (`check_packed_shader_layout`) fails when one id names two constants.

---

## V0 simplifications (to be revisited in V0+)

- **Globally uniform mass / viscosity / radius / volume**: force.comp does
  NOT read neighbor MaterialParameters (uses self's values for all
  pair-averaged quantities). To enable multi-phase, restore neighbor reads
  inside the loop and compute harmonic means — see `force.comp` header.

- **No multi-phase rest_density scaling** in density's ψ_ij: legacy form
  `ρ_j · (ρ_0i / ρ_0j) - ρ_i` simplified to `ρ_j - ρ_i`.

- **No micropolar terms** (a_transfer, aw_*, curl_u in legacy force).

- **No rotor analytical motion**: rotor particles run all kernels but
  don't actually move (predict skips them).

- **No inlet spawn kernel**: inlet templates exist as static particles in
  the pool; no new particles are emitted at runtime. ⚠ Defrag does not
  preserve inlet pool slots — `defrag.comp` only walks
  `inside_particle_index`, which by invariant excludes inlet kind. Any case
  that has INLET particles in the main pool will lose them after the first
  defrag. The current lid case has no INLET so this is dormant; V0+ inlet
  spawn must redesign the defrag path (e.g. dedicated inlet pool list, or
  rebuild inlet slots from `inlet_template` post-defrag).

- **No async multi-GPU**: ghost bindings declared in `common.glsl` (set 2)
  but `GHOST_DIMENSION_*` spec constants are 0, so all ghost-related branches
  are dead-code-eliminated.

- **Per-particle dispatch over POOL_SIZE**: inlet/dead particles are not
  excluded at the dispatch level. Each kernel does its own bounds + kind
  check. Adding indirect dispatch (with maintained `alive_count`) deferred
  until inlet/outlet logic is implemented.

- **No bit-exact integration**: `STRICT_BIT_EXACT` spec constant exists but
  is unused in V0 (no `precise` qualifiers in main kernels). V1 multi-GPU
  ghost integration will need this.

---

## What's NOT here yet

- `update_dispatch.comp` — 1-thread helper to update DispatchIndirectBuffer.
  Skipped in V0 because all kernels dispatch over POOL_SIZE directly.
- `inlet_spawn.comp` — spawn new fluid particles from inlet templates. V0+.
- Ghost-related kernels (pack/unpack/sync) — V1 multi-GPU.
- Python pipeline glue (buffer allocation, descriptor set build, dispatch
  scheduling) — next step.
- Numerical validation against legacy OpenGL — gated on Python pipeline.

## Diagnostic: density budget (`V7_DENSITY_BUDGET=1`)

`density_budget.glsl`, included by `correction_density.comp` in its `DENSITY_BUDGET` compile variants only (`correction_density_budget.comp.spv`, `correction_density_wall_crossing_budget.comp.spv`; no other SPIR-V file changes, no production path loads them). With `V7_DENSITY_BUDGET=1` (one slab, the fused pass, no deep-wall skip) the fused pass also adds, every step and for every fluid particle, `TIMESTEP` x (drift fluid, drift wall, diffusion fluid, diffusion wall) of its density rate to `density_budget[(pid * 12 + region) * 3]` (set 0 binding 13, which exists only then; the next entry holds the change the stored density took and the count of second-order fallbacks, the third, for wall_boundary crossing, the smallest det(I - K_raw L_i), the largest (1 + a_j)^2, the largest wall viscous diffusion number and the steps counted; region = which walls of the fluid box lie within the support radius, x and y only). The pass's results are those of the build without the budget: the `WALL_CROSSING` build takes the four sums beside its own in its loop, the plain build (simple / adami) in a second neighbour traversal after the pass has stored its results. The buffer is not permuted by defrag: only its sum over all slots is read (`SphSimulatorV7.readback_density_budget`).
