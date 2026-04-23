# SPH Multi-GPU Design

Authoritative design for the Vulkan SPH rewrite. Complements `CLAUDE.md` at project root.

## Scope

- δ-plus WCSPH with persistent uniform voxel grid neighbor search
- Multi-GPU via 1D slab domain decomposition, static partition
- **Production target**: 2× RTX 5090 on one machine (same-vendor, PCIe 5.0 x16 P2P)
- **Stress test (paper scaling/portability)**: NV 4060 Ti + AMD 7900 XTX, cross-vendor
- Shared codepath; transport layer pluggable for both configs

## Integration: Leapfrog

Upgrades from OpenGL baseline's explicit Euler. Leapfrog stores velocity at half-step times (`v_{n-1/2}`) so the two Verlet KDK half-kicks collapse into a single full-step kick per iteration — **5 kernels per step** instead of 6. Algebraically identical trajectory to velocity Verlet. The drift-before-force ordering still holds, letting end-of-step sync pre-compute everything the next step's integration needs → **1 sync per step**.

Per-step pipeline on each GPU (entering with `x_n, v_{n-1/2}, a_n, ρ_n, shift_n`):

```
[predict]               kick + drift + crossing, on own + ghost (precise under STRICT_BIT_EXACT):
    v_{n+1/2} = v_{n-1/2} + a_n * dt               (full-step kick, consumes a_n)
    x_{n+1}   = x_n + v_{n+1/2} * dt + shift_n     (drift with δ-plus shift)

[Crossing detection]    geometric check on x_{n+1}:
    - own particle drifted into peer territory → mark for removal
    - ghost particle drifted into own territory → adopt into own boundary layer

[update_voxel]          rebuild own voxel cell structure (incl. adopted particles)

[correction]            on own, reads own + ghost x_{n+1} → M⁻¹, ∇ρ, ksd

[density]               on own, reads own + ghost x_{n+1}, ρ_n → ρ_{n+1}, P_{n+1}

[force]                 on own, reads own + ghost x, v_{n+1/2}, ρ → a_{n+1}, shift_{n+1}

[Sync]                  pack own boundary [K, N), send to peer ghost buffer
                        content: {x_{n+1}, v_{n+1/2}, a_{n+1}, ρ_{n+1}, shift_{n+1}}
```

Note: no explicit post-force kick kernel. The "second half-kick" of velocity Verlet KDK is absorbed into the next step's predict (it becomes part of the full-step kick `v_{n+1/2} = v_{n-1/2} + a_n * dt`). Stored velocity is always at half-step times.

Special cases (preserved from OpenGL code):
- **Inlet particles** (`voxel_id == 0`): skip integration
- **Rotating particles** (rotor group, identified via `MaterialParameters.kind == MATERIAL_ROTOR`): prescribed velocity `ω × r`, skip leapfrog kick. Keep rotor region entirely within one GPU partition to avoid cross-vendor `cos`/`sin` FP divergence.

## Buffer layout

### Main buffer (own)

Single SoA buffer per GPU, voxel-sorted. Voxel order: **interior voxels first, boundary voxels last**.

- Interior voxel = distance from GPU partition boundary ≥ 2 layers
- Boundary voxel = distance < 2 layers
- Split index `K` (separating the two ranges) is static, precomputed from voxel topology

This ordering enables async overlap: dispatch `[0, K)` doesn't touch ghost and can run while sync is in flight.

SoA fields (baseline):
- `pos_vid` — xyz, voxel_id (vec4)
- `vel_mass` — vxyz, mass (vec4)
- `rho_p` — ρ, P, new_ρ, new_P (vec4)
- `acc` — axyz, temperature (vec4)
- `shift` — δ-plus correction xyz, padding (vec4)
- `material` — optional, only if per-particle parameters differ

**Per-particle global constants** (smoothing_length, eos_constant, viscosity, rest_density) go to `VkSpecializationInfo` / `layout(constant_id=N) const`, NOT per-particle storage.

**Per-step scratch** (kernel_sum_grad, normalize_matrix, density_gradient, etc.) — promote to buffer only if the field crosses a kernel boundary; otherwise keep as local variables inside the kernel. The OpenGL `ParticleRuntimeData` layout should be audited aggressively; many of its fields likely don't need persistent storage.

### Ghost buffer (peer's boundary layer, local copy)

Per GPU, holds remote boundary layer (1 voxel thick). Fields:
- `x` (pos), `v` (vel), `ρ`, `a` (acceleration), `shift`
- ~52B unpadded, **64B padded** under std430

Separate voxel cell structure: `ghost_cell_start`, `ghost_cell_count`, `ghost_sorted_idx`. Covers only the boundary layer so it's small and cheap to rebuild each step.

### Transport buffers

- `ghost_out_packet` — own boundary layer packed for send
- `ghost_in_staging` — receive destination (for CPU staging path)
- No separate migration buffers — merged into ghost flow.

## Crossing handling (merged into ghost)

When a particle's integrated position crosses the GPU partition:

- **Owning GPU**: removes particle from own buffer during re-bucketing (skips it when writing new voxel slots).
- **Peer GPU**: during predict kernel on its ghost buffer, detects that a ghost particle's new position lies in its own territory; adopts it into own boundary layer and removes it from the ghost buffer.

Both GPUs reach the same decision because predict kernel is bit-exact — see invariants I1/I2. No separate migration pack/send. One data flow, not two.

## Bit-exact control: `STRICT_BIT_EXACT`

Specialization constant, default `true`:

```glsl
layout(constant_id = 10) const bool STRICT_BIT_EXACT = true;
```

- When `true`: `precise` qualifier applied to predict kernel integration expressions. Compiler forbidden to fuse FMAs or reorder. Both GPUs produce bit-exact x_{n+1} from identical synced inputs.
- When `false`: compiler free to optimize. Small mass drift at GPU boundary.

**Cost** (estimate): integration kernel is memory-bound; `precise` adds a few ALU cycles typically absorbed by memory stalls. <2% on integration kernel, <0.1% on total step.

**Cost of disabling**:
- Same-vendor (NV+NV): same driver/compiler → usually still bit-exact anyway. Mass drift estimated <0.05%.
- Cross-vendor (NV+AMD): FMA fusion and codegen differ. Mass drift estimated 0.1–1%.

**Recommendation**: keep `true` by default. Paper can include a table comparing mass conservation with/without — a nice robustness plot for reviewers.

## Sync protocol

**One sync per step, at end of step.**

Content (own's boundary layer `[K, N)`):
- `{x_{n+1}, v_{n+1}, a_{n+1}, ρ_{n+1}, shift_{n+1}}` — ~64B per particle (padded)

Typical bandwidth: 100k boundary particles × 64B = **6.4 MB per direction per step**.

Timing:
- Sync starts after force completes (the last compute stage of a step)
- Sync must complete before peer's next-step predict kernel on ghost
- Interior compute on the next step does NOT depend on sync → async overlap opportunity (V2)

## Transport backends (pluggable)

```
Transport interface:
    pack_out(src_boundary_device_buffer, dst_packet_buffer)
    exchange_with_peer(out_packet) -> in_packet
    unpack_in(in_packet, dst_ghost_device_buffer)
```

| Backend | Platform | Path | Status |
|---|---|---|---|
| `CpuStagingBackend` | any | device → host → peer host → device | ready (Phase 1 pattern reused) |
| `P2PBackend` | same-vendor NV | `vkCmdCopyBuffer` device → device | V2, likely works on 2×5090 |
| `SharedMemoryBackend` | if probe passes | mapped external memory | conditional; re-run `probe_interop.py` on 2×5090 rig |

All backends expose the same interface. Switching is a one-line instantiation decision.

## Invariants

| # | Invariant | Enforcement |
|---|---|---|
| I1 | predict kernel on own particle P and ghost copy of P produces bit-exact x, v when `STRICT_BIT_EXACT=true` | `precise` qualifier + identical `{x, v, a, shift, dt}` from sync |
| I2 | Crossing decision agrees across both GPUs | I1 + geometric check on bit-exact positions |
| I3 | Ghost layer covers max single-step drift | 1 voxel thick; CFL × h ≤ h guaranteed by timestep choice |
| I4 | Ghost carries all fields for next step's predict kernel + density + force | Sync packet = `{x, v, a, ρ, shift}` |
| I5 | Boundary state complete at sync time | Sync scheduled after force |
| I6 | Adopted ghost particle not double-counted | Re-bucketing excludes adopted particles from ghost's next-step cell structure |

## Async overlap (V2 optimization)

Voxel-sorted layout enables dispatch range split:
- Interior dispatch `[0, K)`: predict + correction + density + force. **No ghost dependency**.
- Boundary dispatch `[K, N)`: same pipeline. **Depends on ghost**.

Scheduling:

```
Step n        [int A][int D][int F][int B][sync out──┐
                                                     ↓
Step n+1      [int A][int D][int F][int B]  ←── compute queue independent of sync
              (own interior reads own only)            [bdy A][bdy D][bdy F][bdy B]
                                                        ↑ waits for sync
```

Hiding succeeds when `interior_compute_time ≥ sync_time`. At target scale (1M/GPU, interior ~2-3 ms, P2P sync ~0.1-0.5 ms), sync is fully hidden.

**Residual non-hideable overhead** (~1 ms total):
- Ghost predict kernel (~0.3-0.5 ms) — requires fresh ghost, can't start before sync
- Pack / unpack kernels (~0.1-0.3 ms)
- Command submission / fence wait (~0.05 ms)

## Non-goals (explicitly excluded)

- Separate migration kernel / buffer (replaced by bit-exact adoption via ghost)
- Multiple syncs per step
- Dynamic load balancing (deferred beyond V2)
- Ghost ρ recomputation on receiver (ρ is always synced, never derived locally)
- Interior/boundary split at the own-particle buffer level (we keep one unified buffer, partitioned by voxel-sort order)

## Open knobs

- **Ghost layer thickness**: 1 voxel default. Widening to 2+ would enable sync decimation (sync every M steps) — not V1/V2; maybe V3+.
- **Particle shifting**: δ-plus shift can be disabled via spec constant for V0 debugging.
- **Ghost field compression**: float32 default; float16 positions possible later if ghost bandwidth becomes critical.
- ~~**Verlet variant**: kick-drift-kick vs leapfrog~~ **Decided: leapfrog** (5 kernels/step, stored velocity at half-step).

## Performance targets

| Config | Particles | fps | Status |
|---|---|---|---|
| OpenGL baseline (1× 5090) | 1M | 330 | existing reference |
| V1 Vulkan (2× 5090) | 2M | 150-250 | no async overlap, correctness validation |
| V2 Vulkan (2× 5090) | 2M | 300-380 | async overlap + best backend |
| Stress test (NV+AMD) | ~1M | TBD, slower | CPU staging, shows cross-vendor penalty for paper |

## Implementation phases

- **V0**: single-GPU SPH. Voxel grid + leapfrog predict + density + force. Validates integration scheme + SoA layout without cross-GPU complexity. Target: match or exceed OpenGL single-GPU fps.
- **V1**: multi-GPU with `CpuStagingBackend`, no async overlap. Validates cross-GPU correctness — bit-exact predict kernel, adoption/removal, mass conservation.
- **V2**: async overlap + `P2PBackend` (if same-vendor). Performance push toward target fps.
- **V3**: paper experiments — all backends, weak/strong scaling runs, stress-test comparison.
