# Micropolar terms in the v7 chain solver: design note (Stage 0)

Branch `v9-micropolar`, base tag `micropolar-base` = ab50c10 (merge of `origin/bootstrap-density-fix` eda5b90 and
`origin/v7-adami-multi` 8c33192). Sections 0 to 7 are the Stage 0 plan, with the decisions of 2026-11-10 entered in
Section 2; Section 8 is the Stage 1 report (one slab), Section 9 the Stage 2 report (chains), Section 10 the checks A to E.

Marked **[read]** = inferred from reading the code, **[run]** = verified by running on this machine.

## 0. State of the base

- **[run]** The 19 plain kernels of `experiment/v7/shaders/*.comp` rebuild byte-identical to the tracked
  `spv/*.comp.spv` with `C:/VulkanSDK/1.4.350.0` glslc (shaderc v2026.2), the compiler of `MANIFEST.txt`. The
  working tree is checked out CRLF; that does not change the SPIR-V. T0 can be checked here.
- **[run]** `_test_seam_layout.py` on `micropolar-base`: exit 1, **858 failures**. The list is saved (
  `logs/micropolar/seam_layout_base.log`, not tracked) and is the reference set for T0.
- **[run]** This machine has one RTX 5090 and the RX 7900 XTX (`nvidia-smi` shows one NVIDIA card). Every K = 2
  test below is therefore a cross-vendor pair (device map 0 = 5090, 1 = XTX), compared with the same pair on
  `micropolar-base`.

## 1. Points in the task description I would change

1. **Restart needs `angular_acceleration` too.** `restart_init` runs no force pass: the first `predict` after a
   restart reads `acceleration` (a^n) from the state, and would read alpha^n the same way. `RESTART_FIELD_LAYOUT`
   gains both `angular_velocity` (w^{n-1/2}) and `angular_acceleration` (alpha^n) when micropolar is on. The nine
   Newtonian fields and existing checkpoints are untouched. Consequence: a resume of a micropolar run from a
   Newtonian checkpoint is refused; `--initial-state` (bootstrap path, t = 0) accepts a state without w and uses
   w = 0.
2. **The two curl terms are not a wave system.** For a transverse Fourier mode (k along x, v along y, w along z):

   ```
   d/dt (v, w) = [[-nu k^2, -i c k], [ i c k / Theta, -(2 c + xi k^2) / Theta ]] (v, w)
   ```

   The product of the off-diagonal terms is +c^2 k^2 / Theta, so both eigenvalues are real. They are negative for
   all k exactly when `c (2 nu - c) + nu xi k^2 > 0`, which for long waves is the admissible range
   0 <= c <= 2 nu. Outside it (c > 2 nu, the Bender-type use) modes with `k^2 < c (c - 2 nu) / (nu xi)` grow
   exponentially in the continuous model itself. So:
   - `c dt / (sqrt(Theta) dx)` is still printed, as the coupling rate number, not as a CFL number of a wave.
   - Outside the admissible range the log also prints the shortest growing wavelength
     `2 pi sqrt(nu xi / (c (c - 2 nu)))`, so one can see whether it fits in the domain. With xi = 0 every
     wavelength grows.
3. **Time integration of the new terms is forward Euler in effect.** `force` evaluates alpha from w^{n+1/2} and
   `predict` adds it to w^{n+1/2}, exactly as the viscous term does with v. For T4 this means
   `w^{n+1/2} = w^0 (1 - lambda dt / 2) (1 - lambda dt)^n`, `lambda = 2 c / Theta`. T4 compares against this
   product to rounding, and against `exp(-lambda t)` with the expected first-order difference
   (about n (lambda dt)^2 / 2). The explicit limit of each damping term is 2, so the 0.1 warning is conservative.
4. **T3 "to rounding" needs `regularization.xi = 0`.** Every corrected operator carries `M (M + xi I)^-1`
   (`common.glsl`, spec 14): 0.999 at the default 0.001. With the default the two curls are exact for linear
   fields only to about 9e-4 relative. T3 runs its exactness rows with xi = 0 and reports the default-xi
   deviation as a separate row.
5. **No stub buffers when micropolar is off.** `wall_dummy_velocity` needs a stub because the one `force.comp`
   module always declares binding 10. The micropolar bindings are declared only by the `MICROPOLAR` variants, so
   a Newtonian run needs no buffer and no binding at all. A Newtonian slab then has the same buffer list,
   descriptor layouts, pool size and allocation log as `micropolar-base`, which is stricter than the stub.
6. **Wall value on the particle, not in `MaterialParameters`.** The struct has no free vec3 and changing its
   48 B layout would change every Newtonian module. The material key `wall_angular_velocity: [x, y, z]`
   (default zero) is host-side only: the host writes it into `angular_velocity` of every particle of that
   material at upload, as the lid speed is written into `velocity_mass`. `simple` walls are then read like any
   neighbour, replicas carry the value, and the adami pass reads the wall's own slot as `w_wall`.

## 2. Decisions

Decided 2026-11-10: **D1 = the separate buffer** (`wall_dummy_angular_velocity`, set 0 binding 14: one design for
both dummy fields); D2, D3, D4 as recommended.

| | question | recommendation at Stage 0 |
|---|---|---|
| D1 | Where the adami dummy micro-rotation lives | In the wall particle's own `angular_acceleration` slot (a wall never has alpha). Saves 16 B per pool slot against a separate `wall_dummy_angular_velocity` buffer. Coverage, lifetime and transport are those of `wall_dummy_velocity`; the only difference is that defrag carries it along, unused. Alternative: a separate buffer at set 0 binding 14. **Decided: the separate buffer.** |
| D2 | Replica record for w | 12 B (three words) per replica in a separate packed block, only for the layers that need w (Section 5). Alternative: one word in 2-D (w_z only), 4 B instead of 12 B, at the cost of a dimension branch in pack and unpack. |
| D3 | Transport layouts supported with micropolar | Only the release layout: `V7_GHOST_LAYERS` >= 2, lean transport, compact ghost lists, packed replicas. The legacy layouts (mixed one-layer pool, SoA replicas, non-lean) are refused at start-up with micropolar on. Also refused: `V7_BAND_OVERLAP` (default 0; it would need a `MICROPOLAR` + `FORCE_SEGMENT` double variant) and `V7_BAND_COMPACT_DISPATCH`. |
| D4 | Points 1 to 6 of Section 1 | As written there. |

## 3. Model in the code's terms

Per pair, inside `force.comp`'s existing neighbour loop (`MICROPOLAR` variant), with the variables that loop
already has:

```
curl_angular_velocity += neighbor_volume_dynamic * cross(self_angular_velocity - neighbor_angular_velocity,
                                                         kernel_gradient_corrected)
curl_velocity         += neighbor_volume_dynamic * cross(self_velocity - viscous_neighbor_velocity,
                                                         kernel_gradient_corrected)
laplacian_angular_velocity += 2 * neighbor_volume_dynamic * (self_angular_velocity - neighbor_angular_velocity)
                              * dot(position_difference_neighbor_to_self, kernel_gradient_corrected)
                              / (distance * distance + EPS_H_SQUARED)
```

After the loop:

```
coupling_coefficient = self_params.viscosity + self_params.viscosity_transfer        (c, float32 sum)
total_acceleration  += coupling_coefficient * curl_angular_velocity
angular_acceleration = (self_params.viscosity_rotation * laplacian_angular_velocity
                        + coupling_coefficient * (curl_velocity - 2 * self_angular_velocity)) / MICROPOLAR_THETA
```

- `neighbor_angular_velocity` = `angular_velocity[j]` for a fluid neighbour and for a `simple` wall (its stored
  `w_wall`); `wall_dummy_angular_velocity[j]` for a wall neighbour under `adami`.
- `viscous_neighbor_velocity` is the variable the Morris term already uses, so the curl of v sees the dummy
  velocity under `adami`.
- c is summed in the shader from the two material fields. With `nu_t = -nu` the float32 sum is exactly 0 (T2).
  The host computes the same c in float64 for the log and the checks.
- Sign: rigid rotation v = Omega x r gives `curl_velocity = +2 Omega` (T3 row 1).
- The viscous term keeps `self_params.viscosity`, which is nu = (mu + kappa) / rho in Eringen's notation.

Start-up log (host, float64, per fluid material): nu, nu_t, c, xi, Theta; K = c / (nu - c),
N_c^2 = c / (2 nu - c), l^2 = xi nu / (c (2 nu - c)) where defined; admissible or not (and the growing
wavelength, Section 1 point 2); the four step numbers `nu dt / dx^2`, `(xi / Theta) dt / dx^2`, `2 c dt / Theta`,
`c dt / (sqrt(Theta) dx)` with a warning above 0.1. Rejected: c < 0, xi < 0, Theta <= 0. dt is not changed.

For the T5 parameters (dt = 0.15 h / c0 = 3.0e-5 s, dx = 4 mm): c = 1.875e-3, Theta = 1.6e-3, xi = 3.75e-6;
K = 3, N_c^2 = 0.6, l = 0.04 m; step numbers 4.7e-3, 4.4e-3, 7.0e-5, 3.5e-4. No warning.

## 4. Storage, bindings, spec constants

| item | where | size | notes |
|---|---|---|---|
| `angular_velocity` | set 0 binding 12, vec4 | 16 B per pool slot | xyz = w^{n+1/2}; w free. Fluid: integrated. Wall: prescribed `w_wall`. Defragged, transported (replicas, migrants, departed copies), in the restart state. `SHARING_MODE_CONCURRENT` (the transfer queue copies its migrant region). |
| `angular_acceleration` | set 0 binding 13, vec4 | 16 B per pool slot | xyz = alpha, written by force for the particles it writes `acceleration` for. Defragged, in the restart state, never transported, exclusive. |
| `wall_dummy_angular_velocity` | set 0 binding 14, vec4 | 16 B per pool slot with adami, a 16 B stub with simple | (2 w_wall - w~, Shepard sum) per wall particle, written by the wall pass. The rules of `wall_dummy_velocity`: transient, not defragged, not transported, not in the restart state. |
| defrag twins | set 4 bindings 12, 13 | 32 B per pool slot | Only in `defrag_micropolar`. |
| packed w block | `ghost_packed_words`, after the existing blocks | 12 B per replica of a w layer | Section 5. |

Binding 11 of set 0 stays reserved for the global id buffer, as `common.glsl` notes. `extension_fields` is not
touched.

Bytes per pool slot, micropolar on: +32 B live, +32 B defrag twins = **+64 B** on 288 B with simple walls;
with adami walls +16 B for the dummy buffer = **+80 B** on 304 B (set 0 plus defrag twins today).

Spec constants:

- 21 `MICROPOLAR_THETA`: already declared in `common.glsl`. The host adds `(21, 'f', micro_inertia)` to the
  spec entries only when micropolar is on, so a Newtonian pipeline gets the same specialization blob as today.
- 116 `MICROPOLAR_REPLICA_LAYERS` (new, declared in `micropolar.glsl`): how many replica layers carry w,
  counted from the inner one. 1 for a simple chain, 2 for an adami chain (Section 5).
- Id plan: `common.glsl`'s comment gets the 116 line only if all 23 tracked modules still rebuild byte-identical
  with it (glslc without `-g` should not care; I will check, and otherwise the plan line stays in
  `micropolar.glsl` and `README.md`).

`common.glsl`, `helpers.glsl`, `wall_boundary.glsl` and every `.comp` keep their Newtonian text outside
`#ifdef MICROPOLAR`.

## 5. The seam

Notation of `v6_design.md` / `v7_adami_chain.md`: column 0 = own column next to the seam, G1 / G2 / G3 = the
peer's columns 0 / 1 / 2, band widths c / d / f (and w for the adami wall pass). Time levels within step n -> n+1:
x^{n+1}, v^{n+1/2}, w^{n+1/2} are written by `predict` in phase A; rho^{n+1} by density in B / C.

### 5.1 Who reads w, at which level, written by whom **[read]**

| sweep (phase) | self | reads | level | written by |
|---|---|---|---|---|
| `predict` (A) | own fluid, all columns | own `angular_velocity`, `angular_acceleration` | w^{n-1/2}, alpha^n | step n's `predict` (or `install_migrations` for a particle that arrived in step n); step n's force. A migrant of step n was installed in column 0, inside the force band, so the band force wrote its alpha^n. |
| `ghost_send` (A) | own columns 0 .. L-1 | `angular_velocity` of replicas of the w layers, of migrants | w^{n+1/2} | this step's `predict` |
| wall pass, interior (B, adami chain) | own walls, column >= w = 3 | fluid w of columns >= 2 | w^{n+1/2} | `predict` (A). Columns >= 1 receive no migrant, so their lists and w are final after phase A. |
| `force_deep_interior` (B) | own fluid, column >= f | fluid w of columns >= f - 1; wall neighbours of columns >= f - 1: stored `w_wall` (simple) or dummy (adami) | w^{n+1/2}; dummy of this step | `predict` (A); the interior wall pass above (walls of column >= w = f - 1) |
| `expand_ghost_lists` (C) | G1 .. | packed w block | w^{n+1/2} | peer's `ghost_send` (A) |
| `install_migrations` (C) | migrant slots | migrant's `angular_velocity` | w^{n+1/2} | peer's `predict`, via the packet |
| correction / density band (C) | own 0 .. c-1, G1 (, G2) as self | no w | | |
| wall band pass (C, adami chain) | own walls of columns 0 .. 2, G1 walls as self | own stored `w_wall`; fluid w of own columns 0 .. 3, of G1, of G2, and of the departed copies listed in G1 | w^{n+1/2} | own: `predict` or the migrant packet; G1, G2: replicas; departed: the copy `ghost_send` kept |
| `force_boundary_band` (C) | own fluid, columns 0 .. f-1 | fluid w of own columns 0 .. f, of G1 and of the departed copies; wall neighbours: own (stored, or dummy from the B pass for column >= 3 and from the C band pass for columns 0 .. 2), G1 walls (stored value from the replica, or dummy from the C band pass) | w^{n+1/2}; dummy of this step | as above |
| bootstrap | as one step without phase split | w^0 from the bootstrap ghost round (same `ghost_send`), dummy from the bootstrap wall passes | w^0 | host upload |

Conclusions:

1. **w^{n+1/2} has the provenance of v^{n+1/2} everywhere**: written by `predict` in phase A, read by B and C,
   never rewritten in the step. The dummy micro-rotation has the provenance of the dummy velocity: the same wall
   sweep, the same inputs' columns. No sweep waits for a w that it does not already wait for through v.
   **The band widths do not change: 2 / 2 / 3 for a simple chain, 2 / 2 / 4 with wall band 3 for an adami
   chain.** This is the expectation of the task, now checked row by row above, by reading.
2. **alpha does not travel.** Its only reader is the next `predict` on the same particle, and every installed
   migrant is rewritten by the band force first. Same argument as `acceleration` under lean transport.
3. **Which replica layers carry w:** the layers whose fluid state feeds this slab's force, directly or through
   the wall pass. These are exactly the layers recomputed as self by the density band, for the same reason.
   - simple chain (two layers): **G1 only**. G2 is read only by correction / density of G1 as self (x, v, rho).
   - adami chain (three layers): **G1 and G2**. The band wall pass on a G1 wall averages the fluid w of G2.
     G3 is read only by correction / density of G2 as self.
   So `MICROPOLAR_REPLICA_LAYERS = GHOST_LAYERS - 1`; the host passes it explicitly and checks the relation.
4. **Departed copies carry w.** A departed migrant stays listed in G1 for this step's column 0 force and the G1
   wall pass, which read its w.
5. **G2 walls need no dummy value**: only a force on G1 fluid would read it, and G1 is not a force self here.

### 5.2 Records and bytes per link

Replica record, layers 0 .. `MICROPOLAR_REPLICA_LAYERS` - 1: the existing 32 B record (x y z rho | vx vy vz
material bits) unchanged, plus 12 B (w_x w_y w_z) in a new block of `ghost_packed_words` after the existing
ones: per direction and w layer `3 R` words, R = `REPLICA_REGION_SIZE`. The existing block offsets
(`packed_layer_base`) do not move. The transport gets one more count-aware segment per direction and w layer
(stride 12, bounded by that layer's replica counter, as the two 16 B blocks are).

Migrant packet: 44 B -> **60 B** (the `angular_velocity` slot travels as its 16 B SoA segment of the migrant
region, like the other four fields).

Host copy per link and step, in the form of the paper's `B_link` (N_l = replicas of layer l, N_m = migrants,
V_g = ghost voxels of the link = L x face, L = ghost layers):

```
Newtonian:   B_link = 32 (N_1 + ... + N_L) + 44 N_m + 8 V_g + 4 (L + 2)
micropolar:  B_link = 32 (N_1 + ... + N_L) + 12 (N_1 + ... + N_{L-1}) + 60 N_m + 8 V_g + 4 (L + 2)
```

(8 V_g = `inside_particle_count` + `ghost_voxel_first_particle_id`; L + 2 words = the migrant counter, L
replica counters and the frame stamp. This reproduces the E40 third-layer increment
6,691 x 32 + 206 x 8 + 4 B.) With equal layers N_l = N: replica bytes per face particle 64 -> 76 (+18.75 %) for
a simple chain, 96 -> 120 (+25 %) for an adami chain; the 3-D leading term `256 n^2` becomes `304 n^2`. The DMA
(capacity based) grows by `12 R (L - 1) + 16 M` per direction, M = migrant region slots.

Whether the larger link stays hidden behind phase B is a performance question outside this task; phase B also
grows (the force sweep does more per pair).

## 6. Change list by file

Shaders (`experiment/v7/shaders/`). Every change sits behind `#ifdef MICROPOLAR`; the plain build of each source
is unchanged.

| file | change |
|---|---|
| `micropolar.glsl` (new) | Bindings 12 / 13 of set 0, spec 116, the packed w block helpers (base offset, three-word load / store), the model and id-plan comment. Included only under `#ifdef MICROPOLAR`. |
| `predict.comp` | `angular_velocity += angular_acceleration * TIMESTEP` next to the velocity kick, fluid only. |
| `bootstrap_half_kick.comp` | `angular_velocity -= 0.5 * angular_acceleration * TIMESTEP`. |
| `force.comp` | Section 3. Wall neighbour under adami reads `wall_dummy_angular_velocity[j]`. Writes `angular_acceleration[self]`. |
| `wall_extrapolate.comp`, `wall_extrapolate_band.comp` | One more Shepard sum over the fluid neighbours' w; writes `2 w_wall - w~` (or `w_wall` without fluid neighbour) to `wall_dummy_angular_velocity`. |
| `defrag.comp` | Set 4 bindings 12 / 13, two more copies. |
| `ghost_send.comp`, `ghost_send_lanes.comp` | Packed w for layers < `MICROPOLAR_REPLICA_LAYERS`; `angular_velocity` into the migrant slot and the departed copy. Only the release layout paths (D3). |
| `expand_ghost_lists.comp` | Unpack w into the ghost slot of the w layers. |
| `install_migrations.comp` | Copy `angular_velocity` to the own slot. |
| `append_departed.comp`, correction / density kernels, `update_voxel`, `initialize_voxelization`, `band_compact`, `density_scratch_copy`, `deep_wall_marker` | none |
| `README.md` | The variants, the two fields in the access matrix. |
| `spv/` | Ten new modules: `predict_micropolar`, `bootstrap_half_kick_micropolar`, `force_micropolar`, `wall_extrapolate_micropolar`, `wall_extrapolate_band_micropolar`, `defrag_micropolar`, `ghost_send_micropolar`, `ghost_send_lanes_micropolar`, `expand_ghost_lists_micropolar`, `install_migrations_micropolar`. The 23 existing files byte-identical. `MANIFEST.txt` regenerated (33 files). |

Host:

| file | change |
|---|---|
| `compile_shaders_v7.py` | Ten `("<name>_micropolar", "<name>.comp", ("MICROPOLAR",))` entries in `SHADER_VARIANTS`. |
| `utils/case_v7.py` | `PhysicsConstants.micropolar_enabled`, `micro_inertia`; `MaterialParameter.wall_angular_velocity` (host-side, like `initial_velocity`); optional initial angular velocities in the initial state. |
| `utils/case_loader_v7.py` | `physics.micropolar: {enabled, micro_inertia}`; material keys `viscosity_transfer`, `viscosity_rotation`, `wall_angular_velocity`; the checks and the start-up log of Section 3. Absent keys = off. |
| `utils/partition_v7.py` | Slab cases carry the new fields and the per-slab rows of the initial w; `transported_particle_fields(micropolar)` adds `angular_velocity`; the refusals of D3. |
| `utils/simulator_v7.py` | `_micropolar` property; the two buffer specs only when on (as `deep_wall_specs`); per-instance defrag binding list and concurrent-name set (the module constants keep their values); module name = `<name>_micropolar` for the ten kernels, pipeline keys unchanged, so the recording code and its streams are shared; spec 21 / 116 entries only when on; packed buffer size and the w segments; initial upload of w (zero, `w_wall` on walls, or the provided state); `RESTART_FIELD_LAYOUT` per instance (+2 fields when on). |
| `_run_v7_equivalence.py`, `experiment/validation/cavity_runner.py` | `angular_velocity` in the compared / saved fields when on; `read_restart_state` follows the per-instance layout; `--initial-state` accepts an optional `angular_velocity`; centreline sampling of w_z for T5. |
| `_test_seam_layout.py` | A `check_micropolar` block: Newtonian slab unchanged (buffer specs, spec blobs, module names, recorded streams equal with the switch absent), the w segments tile the staging, `MICROPOLAR_REPLICA_LAYERS` = layers - 1, shader text pins. Existing checks untouched. |
| `materials/` , `cases/` | A micropolar material file and the T2 to T5 cases as new files; existing files unchanged. Note: `materials/standard.yaml` is modified and uncommitted in the main worktree; I will not touch that copy. |
| new test drivers | `experiment/validation/micropolar_operator_test.py` (T3: prescribed state, bootstrap, read back `acceleration` / `angular_acceleration`; the bootstrap force runs on the uploaded uniform rho^0 with P = 0), `micropolar_relaxation_test.py` (T4), a T1 / T2 / T5 comparison driver following `data/e9/equivalence_light`. |

Newtonian invariance is by construction: with micropolar absent the simulator loads the same module files, builds
the same buffer list and spec blobs, and runs the same recording code. T0 and T1 then test it.

## 7. Tests and what can run here

| test | runs here | note |
|---|---|---|
| T0 | yes | 23 files byte-identical; failure set of `_test_seam_layout.py` equal to the saved 858. |
| T1 | yes, K = 1 on the 5090, K = 2 on 5090 + XTX | Floor from two repeats of the base run on the same devices. |
| T2 | yes | Different SPIR-V than T1, so equality is within the floor, not bitwise; w must be exactly 0. |
| T3 | yes, K = 1 | xi = 0 rows and a default-xi row (Section 1 point 4). Fluid block without walls, particles farther than h from the edge. |
| T4 | yes, K = 1 | Against the discrete product and against the exponential (Section 1 point 3). |
| T5 | yes, cross-vendor K = 2 | Same-vendor K = 2 is not available on this machine. |
| 3-D | not planned | The kernels are dimension independent, but no test above exercises w_x, w_y or a 3-D seam. I will say so in the reports. |

## 8. Stage 1 report (one slab, 2026-11-10)

Everything in this section was run on this machine (RTX 5090, device 0) unless marked **[read]**. Logs and dumps:
`logs/micropolar/` (not tracked). Driver: `experiment/validation/micropolar_tests.py`.

### What changed

- Shaders: `micropolar.glsl` (new) and `#ifdef MICROPOLAR` blocks in `predict`, `bootstrap_half_kick`, `force`,
  `wall_extrapolate`, `wall_extrapolate_band`, `defrag`; six `<kernel>_micropolar` entries in `SHADER_VARIANTS`.
- Host: `physics.micropolar: {enabled, micro_inertia}`; material keys `viscosity_transfer`, `viscosity_rotation`,
  `wall_angular_velocity`; the start-up report and checks (`case_loader_v7.micropolar_coefficient_report`); the
  three buffers, the variant modules, spec entries 21 / 116, the w upload and the eleven-field restart layout in
  `simulator_v7.py`; the partition slices a provided initial w; `cavity_runner.py` and `_run_v7_equivalence.py`
  follow the per-case restart layout / read w.
- A slab with a peer refuses micropolar until Stage 2 (`NotImplementedError` in the constructor).
- New files: `materials/micropolar.yaml`, `cases/micropolar_cavity_n250_null`, `cases/micropolar_cavity_n250_re400`.

### Tests

| test | result | measured |
|---|---|---|
| T0 build | pass | 23 of 23 SPIR-V files of `micropolar-base` byte-identical; 6 new files; `MANIFEST.txt` regenerated (29). `_test_seam_layout.py`: 858 failures before, 858 after, the same set (sorted diff empty), with the new `check_micropolar` contributing none. Three existing pins were extended for the new files and fields (the two "SPIR-V files beside" lists, the v5 field strip list, the rc2 text pin of `wall_extrapolate.comp`, now compared without its MICROPOLAR blocks). |
| T1 Newtonian, K = 1 | pass | 3,000 steps of the n = 250 adami cavity, fields sampled at 2,401 points. Ten repeats of `micropolar-base`: pair rms of u 3.0e-5 to 1.5e-4 m/s, v 3.9e-5 to 1.6e-4, rho 6.6e-5 to 1.6e-3 kg/m^3. Nine runs of this tree against all ten: largest rms / largest base pair rms = 0.73 to 1.04 (u), 0.65 to 1.04 (v), 0.67 to 1.09 (rho). |
| T1, K = 2 | not run at this stage | Stage 2 (cross-vendor pair). |
| T2 null coupling, K = 1 | pass | Five runs: ratios 0.73 to 0.94 (u), 0.64 to 1.03 (v), 0.65 to 1.06 (rho); max abs w over all particles exactly 0 in all five. |
| T3 rigid rotation | pass | Regularization xi = 0: Theta alpha_z / c = +1.9999997 (lattice), +2.0000000 (jittered 0.1 dx); relative error max 8.2e-7 / 6.1e-7; a max 3.2e-8 m/s^2. Default xi = 0.001: +1.99823 (-8.9e-4), as predicted. |
| T3 linear w_z | pass | xi = 0: a / c against curl w, relative error max 1.4e-6 (lattice), 1.5e-6 (jittered). Theta alpha + 2 c w = xi lap w: lattice 5.8e-5 of grad w / dx (rounding of the pair differences); jittered max 7.1e-2, rms 1.8e-2 of grad w / dx (not exact on disordered particles, as expected). Default xi: curl off by 8.9e-4. |
| T3 sin(k x) | pass | lap w / (-k^2 w) = 0.975506, 0.979438, 0.980441 at dx = 4, 2, 1 mm (k h = 0.63, 0.31, 0.16): Richardson order 1.97, limit 0.980784, i.e. a constant factor -1.92 % from EPS_H_SQUARED = 0.0025 h^2. With EPS_H_SQUARED = 0: 0.994644, 0.998622, 0.999637, order 1.97, limit 0.999985. |
| T4 relaxation | pass | 2 c dt / Theta = 0.005: after 100 / 300 steps w_z = 1.8218766 / 0.6685520 against the scheme's product 1.8218775 / 0.6685522 (max relative deviation 4.6e-7 / 2.9e-7, all particles, spread 0); against the exponential -1.244e-3 / -3.746e-3, equal to the expected first-order difference. v and w_x, w_y stay exactly 0. |
| restart (not in the task list) | pass | 100 steps, state saved by `cavity_runner.read_restart_state` (eleven fields), restart in a new simulator, 100 more: 2.8e-7 from the uninterrupted product (a lost kick would be 5e-3); a nine-field state is refused. |
| validation layer | clean | T3, T4 and the restart check ran with the Khronos validation layer on: no message. |

### Bytes

Per pool slot with micropolar on: +64 B (simple walls), +80 B (adami walls), as planned. Per link: nothing yet
(Stage 2).

### Surprises

1. **The run-to-run floor is bimodal.** After 3,000 steps of the n = 250 cavity started from rest, repeats of one
   build fall into two groups: within a group the sampled density differs by about 2e-4 kg/m^3 rms, between the
   groups by about 1.5e-3 (max 1.5e-2). 2 of 10 base runs and 2 of 9 runs of this tree were in the second group. A
   floor taken from one pair of repeats can therefore be 7 times too low; the comparison now takes the largest of
   all pairs of ten reference repeats. I did not look for the cause (a property of `micropolar-base`).
2. **Particles cannot be matched one to one after 3,000 steps**: between two repeats single particles near the lid
   corners are up to 0.7 dx apart. The comparison is made on Shepard-sampled fields instead.
3. **The angular diffusion carries the EPS_H_SQUARED factor**, -1.92 % at h / dx = 5: the effective xi is 0.981 xi
   (the same mechanism that sets the effective viscosity). If the thesis comparisons need xi to better than 2 %,
   the case can compensate or lower `epsilon_squared_factor`; I changed no default.
4. The git-ignored cavity `.obj` files are absent from a fresh worktree; I copied the n250 / n500 / n1000 sets
   from `vulkan-demo-bootstrap` into this worktree and into the base checkout `vulkan-demo-v9-base`.

## 9. Stage 2 report (chains, 2026-11-10)

Run on this machine: RTX 5090 (device 0) and RX 7900 XTX (device 1). K = 2 was run two ways: on the pair
(cross-vendor, device map 0,1) and with both slabs on the 5090 (device map 0,0), which separates the seam from the
second card's rounding. Logs and dumps: `logs/micropolar/` (not tracked).

### What changed

- Shaders: MICROPOLAR blocks in `ghost_send`, `ghost_send_lanes`, `expand_ghost_lists`, `install_migrations` (four
  more variants, ten in all) and the packed w helpers in `micropolar.glsl`.
- Host (`simulator_v7.py`): spec 116 = ghost layers - 1 on a slab with a peer; the w block of `ghost_packed_words`
  and one transport segment per w layer and direction; `angular_velocity` in the migrant region's segments and
  CONCURRENT; micropolar with a peer refuses every transport layout but the release one.
- Tests: `micropolar_tests.py seam / advection / cavity / cavity-compare`; `check_micropolar` covers the link
  layout.

### Seam, as built **[run]** unless marked

- Band widths unchanged: 2 / 2 / 3 (simple), 2 / 2 / 4 with wall band 3 (adami). No recording changed: the
  micropolar slab binds the same pipeline keys in the same order.
- w layers: G1 (simple chain), G1 + G2 (adami chain); G3 never. Departed copies and migrants carry w; alpha does
  not travel.
- Bytes per link and step, n = 250 cavity, K = 2 (R = 1,819, M = 64, 1,365 replicas per layer, 112 / 168 ghost
  voxels), measured from the segments and the counters after 2,000 steps; they equal the Section 5.2 formula:

| chain | host copy, Newtonian | host copy, micropolar | DMA, Newtonian | DMA, micropolar |
|---|---|---|---|---|
| simple (2 layers) | 88,272 B | 104,652 B (+18.6 %) | 120,144 B | 142,996 B |
| adami (3 layers) | 132,436 B | 165,196 B (+24.7 %) | 178,804 B | 223,484 B |

### Tests

| test | result | measured |
|---|---|---|
| T0 build | pass | 23 of 23 base SPIR-V files byte-identical, 10 new, `MANIFEST.txt` regenerated (33). `_test_seam_layout.py`: 858 failures, the base's set (sorted diff empty). `_test_chain_cuts.py` ALL PASS. `_test_partition_chain.py`: 3,093 passed, 2 failed, the same two on `micropolar-base` (minimum slab width). |
| T1 Newtonian, K = 2 (5090 + XTX) | pass | 3,000 steps. 40 repeats of `micropolar-base` (780 pairs): pair rms u 2.1e-5 to 2.1e-4, v 2.7e-5 to 2.0e-4, rho 4.8e-5 to 1.9e-3. 30 runs of this tree against all 40: largest rms / largest base pair rms 0.67 to 1.01 (u), 0.68 to 1.01 (v), 0.60 to 1.00 (rho). Runs in the far group (rho rms above 1e-3 to the first base run): base 4 of 39, this tree 6 of 30. |
| T2 null coupling, K = 2 | pass | Five runs: ratios 0.74 to 0.92 (u), 0.86 to 0.94 (v), 0.78 to 0.94 (rho); max abs w exactly 0. |
| seam, K = 2 against K = 1 per particle (not in the task list) | pass | Prescribed smooth v, w and non-zero wall values on the T5 cavity; particles matched by position. After the bootstrap force: acceleration and angular_acceleration differ by 1.0e-7 to 1.5e-7 rms (max 5e-6) of the field rms near the cut, the same as a K = 1 repeat (0.9e-7 to 1.1e-7), simple and adami. After 20 steps, K = 2 on one card: acceleration 1.5e-5 near the cut against 1.2e-5 to 1.3e-5 for the K = 1 repeat; angular_acceleration 1.3e-6 to 1.6e-6 against 1.0e-6 to 1.4e-6; adami wall dummy micro-rotation 4.2e-8 against 3.7e-8. On the pair the acceleration differs by 7e-5 to 9e-5, which is what a K = 1 run on the XTX gives (9e-5 to 1.2e-4): the second card's rounding, not the seam. |
| migrants (not in the task list) | pass | Jittered fluid block with uniform w crossing the cut at 1 m/s, 1,400 steps: 935 particles changed slab, count conserved, all counters zero, every particle's w within 1.3e-6 of the relaxation product (one lost kick would be 5e-4, a lost w would be 1). Validation layer on, both cards: no message. |
| T5 micropolar cavity | equal within the K = 1 repeat scatter; four digits not reached by either | See below. |

### T5

`cases/micropolar_cavity_n250_re400`, 1,340,000 steps (t = 40.2), centreline profiles sampled every 5,000 steps
(Shepard, radius h, 201 points), mean over the last 5 time units (34 samples). Particle count conserved at every
sample and all overflow / far-migration / stamp counters zero in the four runs. Speeds: K = 1 2,715 and 2,827
steps/s, K = 2 on the pair 1,610, K = 2 on one card 776.

| extremum | K = 1 (a) | K = 1 (b) | K = 2, 5090 + XTX | K = 2, one card |
|---|---|---|---|---|
| min u on x = 0 | -0.403434 | -0.403518 | -0.403522 | -0.403503 |
| min v on y = 0 | -0.526033 | -0.525978 | -0.525906 | -0.526069 |
| max v on y = 0 | 0.383100 | 0.383141 | 0.383109 | 0.383137 |
| min w_z on x = 0 | -2.107771 | -2.107946 | -2.108683 | -2.108489 |
| max w_z on x = 0 | 0.817413 | 0.816826 | 0.816659 | 0.817163 |
| min w_z on y = 0 | -1.646534 | -1.646742 | -1.646443 | -1.646564 |
| max w_z on y = 0 | 1.768059 | 1.768169 | 1.767883 | 1.768479 |

Relative differences to K = 1 (a): the K = 1 repeat 6e-5 to 7e-4, K = 2 on the pair 3e-5 to 9e-4, K = 2 on one
card 2e-5 to 3e-4. Profile differences (max over the 201 points): u and v 1.6e-4 to 3.4e-4 for all three, w
1.0e-3 to 4.2e-3, against a standard error of the window mean of 1e-4 (u, v) and 1e-3 to 2e-3 (w).

Reading: K = 2 agrees with K = 1 as well as K = 1 agrees with itself; the values agree to three digits and
differ in the fourth. Four digits are not reached, for two reasons that are the same for K = 1: the flow is not
steady at t = 40 (the window mean still moves by up to 1e-3 relative between the last two windows), and the
sampling noise of a 5-unit window is 1e-4 in u, v and 1e-3 in w. Reaching four digits needs the method of
paper 2's correctness section: a K = 2 continuation from the K = 1 checkpoint and a longer window, on a finer
case. The runner's eleven-field checkpoints support that; I did not run it.

### Not run

- Same-vendor K = 2 on two cards (one 5090 here); K = 2 on one card stands in for it in the seam test and T5.
- K > 2, and anything in 3-D: the kernels and the layout are dimension independent, `check_micropolar` covers the
  2-D layout only, and no test exercises w_x, w_y.
- The legacy transport layouts with micropolar (refused at start-up, D3).

### Surprises

1. The bimodal run-to-run floor of Section 8 also shows at K = 2, and ten repeats were not enough to sample it:
   the first K = 2 comparison failed on a floor of ten base repeats that all fell in the near group.
2. Cross-vendor rounding is visible per particle after 20 steps (about 1e-4 of the acceleration rms), six times
   the same-card repeat floor. It is not a seam effect: a K = 1 run on the XTX shows it.
3. A lattice-aligned block crossing the cut sends a whole lattice line in one step (101 migrants against a
   migrant region of 64): the link overflows and counts it. This is the base behaviour and only concerns
   synthetic tests; the advection test jitters the block.
4. The wall dummy buffers cannot be read back meaningfully right after a defrag (they are not permuted, by
   design); the seam test compares them after steps only.
5. `_test_partition_chain.py` and two more cases need git-ignored `.obj` files; copied from the main worktree.

## 10. Checks A to E (2026-11-11)

All run on this machine (5090 = device 0, XTX = device 1); no solver or shader file changed, so T0 / T1 were not
re-run. Commands: `micropolar_tests.py label / seam --device-map 0,0,0 / factors / paired / operators3d`. Logs:
`logs/micropolar/a_*`, `b_*`, `c_*`, `d/`, `e_*` (not tracked). No defect found.

| check | result | measured |
|---|---|---|
| A. w as a passive label | pass | c = 0, xi = 0, w_0 of each fluid particle derived from its global id (three distinct non-zero components), id in `extension_fields` with `V7_TRANSPORT_EXTENSION=1`. Start: the K = 1 state at t = 40.2; 1,500 steps, state written and restarted, 1,500 steps. 63,001 fluid particles compared, **0 bitwise mismatches** in K = 1, K = 2 (5090 + XTX), K = 2 (one card), each with simple and adami walls; 11,528 wall particles unchanged; ids unique and complete. Covered per run: 4 defrags, 1 restart, 1,444 seam crossings at K = 2 (0 at K = 1). u, v, rho against Newtonian runs of the same two phases: largest rms / largest reference-pair rms 0.67 to 1.35 with three references (2.96 once at K = 1 simple, 1.02 when repeated with ten). |
| B. a slab with two peers | pass | K = 3 runs here (device maps 0,0,0 and 0,0,1; cuts at x = -0.176 and +0.164, middle slab 21,840 particles). Check A: 0 mismatches of 63,001 for both maps and wall types, 2,488 (simple) / 2,476 (adami) crossings, 4 defrags, 1 restart. Per-particle seam check on the prescribed state (0,0,0): after the bootstrap force 1.0e-7 to 1.1e-7 rms near the cuts against 0.9e-7 to 1.1e-7 for the K = 1 repeat; after 20 steps acceleration 1.3e-5 to 1.7e-5 against 1.1e-5 to 2.0e-5, angular_acceleration 1.7e-6 to 2.0e-6 against 1.7e-6 to 1.9e-6, wall dummy micro-rotation 4.9e-8 against 4.9e-8. |
| C. factors, 2-D, h / dx = 5 | measured | Table below. The factor on nu is not the factor on xi. |
| D. paired continuation | pass | Table below; every quantity within 3 K = 1 standard deviations (largest 2.5). |
| E. operators in 3-D, h / dx = 4 | pass | Rigid rotation, Omega = (0.7, -1.1, 1.6): Theta alpha / c = (1.400000, -2.199999, 3.199998) on the lattice, max relative error per component 1.5e-6 (lattice and jittered 0.1 dx), a max 0.9e-6 / 1.7e-6 m/s^2. Linear w, nine gradient components: a / c = (1.9, 1.1, 3.1) = curl w, max relative error per component 1.1e-6 (lattice), 1.5e-6 (jittered). 12,167 interior particles of a 33^3 block. |

### C and E: the constant factors

Regular lattice, Richardson limit of three resolutions (2-D: dx = 4, 2, 1 mm; wave vector along x) or three
wavelengths on one lattice (direction rows and 3-D; the factors depend on k h only); every order 1.99 to 2.00.

| case | factor on nu | factor on xi |
|---|---|---|
| 2-D, h / dx = 5, wave vector along a lattice axis | 0.9587 (-4.13 %) | 0.9808 (-1.92 %) |
| 2-D, wave vector at 22.5 degrees | 0.9808 | 0.9808 |
| 2-D, wave vector at 45 degrees | 1.0029 (+0.29 %) | 0.9808 |
| 2-D, along an axis, EPS_H_SQUARED = 0 | 0.9747 (-2.53 %) | 1.0000 |
| 3-D, h / dx = 4, wave vector along (2, 3, 6) / 7, v along (3, -2, 0) | 0.9735 (-2.65 %) | 0.9862 (-1.38 %) |
| 3-D, same, EPS_H_SQUARED = 0 | 0.9863 (-1.37 %) | 1.0000 |

- The xi operator (component-wise Laplacian) is isotropic and its only constant error is the EPS_H_SQUARED
  denominator.
- The viscous operator carries the same denominator factor on the angle average, plus a lattice anisotropy of
  about +-2.2 % in 2-D at h / dx = 5 (period 90 degrees) that does not vanish with EPS_H_SQUARED = 0. It comes from
  the (v_ij . r_ij) r form of the term, which needs an isotropic fourth moment of the particle arrangement.
- So one constant per case compensates xi, but on a lattice-like arrangement the effective nu depends on the
  local shear direction: 0.959 to 1.003 in 2-D. With mu = rho (nu - c): at the axis value a nominal K = 3 is 3.59,
  at the angle mean 3.25, at 45 degrees 2.97. Which value a developed flow sees depends on how far the particles
  have left the lattice; on the jittered lattice (0.1 dx) the single-resolution ratios follow the same pattern
  but the extrapolation is noise (orders -0.1 to 4.4), so I give no number for it.
- The 3-D value is for one direction only.

### D: paired continuations

Start: the K = 1 state at t = 40.2 (`cavity --save-state`). 200,000 steps each, profiles sampled every 1,000 steps,
first time unit discarded, mean of the next five (167 samples). K = 2 cut at x = +0.0036 (column boundary; 0.18 h
from the line x = 0) in all ten K = 2 runs. Counters zero and particle count conserved in all fifteen runs.

Per run (K = 1 | K = 2 on 5090 + XTX | K = 2 on one card):

| quantity | K = 1 runs 1-5 | K = 2 pair, runs 1-5 | K = 2 one card, runs 1-5 |
|---|---|---|---|
| min u on x = 0 | -0.403591 -0.403519 -0.403447 -0.403466 -0.403540 | -0.403460 -0.403458 -0.403400 -0.403501 -0.403468 | -0.403470 -0.403558 -0.403507 -0.403503 -0.403452 |
| min v on y = 0 | -0.525896 -0.525953 -0.525888 -0.525935 -0.525930 | -0.525893 -0.525923 -0.525998 -0.526008 -0.525883 | -0.525988 -0.525919 -0.525994 -0.525940 -0.526046 |
| max v on y = 0 | 0.383129 0.383196 0.383208 0.383192 0.383159 | 0.383165 0.383126 0.383143 0.383179 0.383190 | 0.383168 0.383124 0.383191 0.383141 0.383127 |
| min w on x = 0 | -2.108031 -2.108075 -2.107877 -2.107777 -2.108000 | -2.108000 -2.108112 -2.107964 -2.108100 -2.107663 | -2.107919 -2.107808 -2.108221 -2.107727 -2.108001 |
| max w on x = 0 | 0.817155 0.816621 0.817948 0.816098 0.816025 | 0.816938 0.817036 0.815546 0.817048 0.816634 | 0.816448 0.816860 0.815612 0.815991 0.815790 |
| min w on y = 0 | -1.646708 -1.646805 -1.646757 -1.646919 -1.646940 | -1.646838 -1.646785 -1.646768 -1.646932 -1.646862 | -1.646953 -1.647082 -1.647015 -1.646480 -1.647123 |
| max w on y = 0 | 1.767671 1.768211 1.767213 1.767774 1.767322 | 1.767760 1.767456 1.768222 1.767777 1.767610 | 1.767888 1.767452 1.767228 1.767753 1.768214 |
| L2 u(y) from the K = 1 mean | 4.6e-5 2.4e-5 7.8e-5 3.9e-5 5.9e-5 | 6.7e-5 3.5e-5 5.2e-5 8.2e-5 5.3e-5 | 3.7e-5 5.8e-5 3.3e-5 3.4e-5 9.8e-5 |
| L2 v(x) | 3.9e-5 3.2e-5 3.9e-5 4.7e-5 5.0e-5 | 4.5e-5 4.1e-5 8.5e-5 3.3e-5 3.2e-5 | 4.9e-5 3.9e-5 5.8e-5 6.1e-5 8.9e-5 |
| L2 w(y) on x = 0 | 3.0e-4 1.3e-4 5.1e-4 2.9e-4 4.2e-4 | 4.0e-4 2.3e-4 3.7e-4 4.9e-4 3.0e-4 | 2.4e-4 4.0e-4 4.5e-4 2.2e-4 6.9e-4 |
| L2 w(x) on y = 0 | 2.5e-4 2.1e-4 1.8e-4 3.1e-4 4.1e-4 | 4.3e-4 3.4e-4 3.6e-4 2.6e-4 3.3e-4 | 3.2e-4 3.5e-4 3.2e-4 4.2e-4 5.9e-4 |

Summary (difference of means in K = 1 standard deviations):

| quantity | K = 1 mean | K = 1 std | K = 2 pair mean | std | diff | K = 2 one card mean | std | diff |
|---|---|---|---|---|---|---|---|---|
| min u on x = 0 | -0.403513 | 5.8e-5 | -0.403457 | 3.6e-5 | +0.96 | -0.403498 | 4.1e-5 | +0.25 |
| min v on y = 0 | -0.525920 | 2.8e-5 | -0.525941 | 5.9e-5 | -0.76 | -0.525978 | 5.0e-5 | -2.08 |
| max v on y = 0 | 0.383177 | 3.2e-5 | 0.383161 | 2.6e-5 | -0.50 | 0.383150 | 2.9e-5 | -0.83 |
| min w on x = 0 | -2.107952 | 1.2e-4 | -2.107968 | 1.8e-4 | -0.13 | -2.107935 | 1.9e-4 | +0.14 |
| max w on x = 0 | 0.816769 | 8.0e-4 | 0.816640 | 6.3e-4 | -0.16 | 0.816140 | 5.1e-4 | -0.79 |
| min w on y = 0 | -1.646826 | 1.0e-4 | -1.646837 | 6.5e-5 | -0.11 | -1.646931 | 2.6e-4 | -1.04 |
| max w on y = 0 | 1.767638 | 4.0e-4 | 1.767765 | 2.9e-4 | +0.32 | 1.767707 | 3.8e-4 | +0.17 |
| L2 u(y) | 4.9e-5 | 2.0e-5 | 5.8e-5 | 1.7e-5 | +0.40 | 5.2e-5 | 2.8e-5 | +0.12 |
| L2 v(x) | 4.1e-5 | 7.3e-6 | 4.7e-5 | 2.2e-5 | +0.81 | 5.9e-5 | 1.9e-5 | +2.46 |
| L2 w(y) on x = 0 | 3.3e-4 | 1.5e-4 | 3.6e-4 | 9.9e-5 | +0.22 | 4.0e-4 | 1.9e-4 | +0.51 |
| L2 w(x) on y = 0 | 2.7e-4 | 8.9e-5 | 3.5e-4 | 6.0e-5 | +0.82 | 4.0e-4 | 1.1e-4 | +1.42 |

- **min w on x = 0, the quantity on the seam: the offset of the old table does not persist.** K = 1 -2.107952
  +- 1.2e-4, K = 2 on the pair -2.107968 (-0.13 std), K = 2 on one card -2.107935 (+0.14 std); the K = 2 runs
  scatter on both sides of the K = 1 mean.
- No bimodality in any group: the fifteen runs form one cloud per quantity.
- The L2 rows are biased low for the K = 1 group, because each K = 1 run is part of the mean it is compared with
  (a factor of about sqrt(4/5) = 0.89 for five runs); the K = 2 differences in those rows (+0.1 to +2.5 std) are
  consistent with that bias and with five-run standard deviations. The largest, L2 v(x) of K = 2 on one card, is
  5.9e-5 against 4.1e-5, i.e. 1.8e-5 of a profile of rms 0.29.
- Standard deviations are from five runs each and uncertain by about a third.

### Unexpected

1. The viscous term's factor (check C) is direction dependent on the lattice; see above.
2. The crossings counted in check A are equal for simple and adami walls at K = 2 (1,444) and nearly equal at
   K = 3 (2,488 / 2,476): over 3,000 steps the crossings near the cuts are set by the starting velocity field.
3. A three-reference floor was again too small once (K = 1 simple, ratio 2.96; 1.02 with ten references).
