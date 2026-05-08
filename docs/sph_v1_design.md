# SPH V1 Design: Dual-GPU Cross-Vendor Pure Baseline

V1 scope = **two GPUs, single host process, cross-vendor (NV 4060 Ti + AMD 7900 XTX), CPU-staged transport, no async overlap, no bit-exactness optimisation**. Deliberately the slowest correct dual-GPU layout. Establishes the data-flow skeleton, descriptor wiring for set 2 (ghost), migration channel, and per-phase timing baseline. V1.1+ optimisations (multi-thread submission, P2P, overlap) layer on top of this without changing semantics.

**Authoritative sources for values**: `shaders/sph/common.glsl` for spec constants + descriptor bindings, `shaders/sph/README.md` for per-kernel invariants, `docs/sph_v0_design.md` for set 0/1/3 layout (unchanged in V1). When this document and those files disagree, trust them.

## Scope Decisions Already Made

- **No bit-exactness across GPUs** in V1. `STRICT_BIT_EXACT` (constant_id=10) stays `false`. Cross-vendor floating-point divergence will not match exactly; numerical equivalence is verified at simulation level (energy / mass conservation, dam-break-style smoke tests), not per-particle ULP. Bit-exact ghost-Kernel-A stays as a future option for same-vendor P2P.
- **CPU-staged transport only**. `CpuStagingBackend` from Phase 1's pattern. P2P / shared-memory deferred to V1.1+ pending NV+NV 2×5090 rig probe.
- **1D X-axis static partition**, computed once at startup from GPU compute weights. No dynamic re-balancing. 1-voxel-thick ghost layer.
- **Pure single-threaded blocking submission**. Each phase: GPU0 submit → wait → GPU1 submit → wait. No queue overlap, no double-buffered staging, no compute/transfer concurrency. This is the V1.0 baseline that subsequent milestones improve against.
- **Double sync per step (Option D)**. One CPU↔CPU exchange after `update_voxel` (carries position/velocity/migration), one after `density.comp` (carries ρ/P). Yields zero ghost-vs-own mismatch despite no bit-exactness — see "Sync Cadence" below.
- **Migration as a separate channel**, not folded into ghost. Cross-partition crossings are appended atomically inside `update_voxel` to a dedicated `outgoing_to_peer` packet buffer; the peer drains `incoming_from_peer` next step into a dead slot. Original `sph_design.md` "merge migration into ghost via bit-exact Kernel A" is deferred to same-vendor V2.

## Hardware Targets and Practical Bandwidth

| Pair | Topology | Practical bidi BW | Transport |
|---|---|---|---|
| 4060 Ti + 7900 XTX (this machine) | PCIe 4.0 x8/x8 split | ~14 GB/s shared | CPU staging via host-visible buffer |
| 2× RTX 5090 (target) | PCIe 5.0 x16+x16, P2P | ~50+ GB/s direct | P2P `vkCmdCopyBuffer` (V1.1+) |

V1 budget assumes the cross-vendor pair. Ghost SoA is ~64 B per ghost-particle; with 1M particles split per partition and ~5% as boundary residents, ghost traffic is ~3 MB/exchange × 2 syncs × 2 directions = ~12 MB/step → ~1% of practical BW headroom at 350 fps. Migration packets are O(particles_crossing/step) = trivial.

## Partition Strategy

1D X-axis static split based on GPU SPH weight. No Y/Z splits in V1 (deferred until weight imbalance benchmarks demand it).

### Compute weight lookup

`experiment/v1/utils/gpu_capability.py` (calibrated 2026-05-06):

```python
KNOWN_GPU_SPH_WEIGHT = {
    "AMD Radeon RX 7900 XTX":      2.088,  # measured: 293.6 M part-step/s
    "NVIDIA GeForce RTX 4060 Ti":  1.000,  # baseline: 140.6 M part-step/s
    "NVIDIA GeForce RTX 5090":     6.0,    # provisional spec-sheet, validate when 2x5090 rig exists
    "AMD Radeon(TM) Graphics":     0.3,    # iGPU lower bound, unmeasured
}
```

Numbers from `tools/benchmark_calibration.py` (lid_driven_cavity_2d, 1M particles, 60s wall-time post-warmup, headless). 7900 XTX measured ratio 2.09 vs spec-sheet TFLOPS ratio 2.78 — SPH on AMD pays a ~25% occupancy / driver tax relative to NVIDIA, which is exactly why we calibrate rather than trust the datasheet.

The calibration is intentionally **whole-pipeline median**, not per-kernel. Per-kernel breakdown (KCG correction is ALU-heavy, density is bandwidth-heavy, etc.) is collected by V1.2's per-phase timestamp queries and feeds a future per-kernel weight model. V1.0 uses one global weight per GPU.

### case.yaml override

```yaml
partition:
  split_axis: x          # V1 only x
  weights: null          # null = lookup from KNOWN_GPU_SPH_WEIGHT; otherwise list of floats
  ghost_layers: 1
```

If `weights` is explicit (e.g. `[1.0, 2.5]`), it bypasses the lookup table. Lookup failure on an unknown GPU name surfaces a clear error message ("Add device to KNOWN_GPU_SPH_WEIGHT or set partition.weights explicitly").

### K_split computation

The partition unit is a **voxel column** (all voxels sharing the same `x_index`), not an arbitrary X coordinate. A single voxel cannot be co-owned — neighbour search scans whole voxels, atomic ops on per-voxel slot lists assume single-writer ownership. The split line is always voxel-aligned.

Particle counts decide *which* voxel column to cut at; the cut itself is integer-indexed:

1. `fractions = weights / sum(weights)`  → e.g. `[0.25, 0.75]` for 4060 Ti + 7900 XTX
2. `target_count_gpu0 = floor(N_fluid * fractions[0])`
3. Bin all initial fluid particles into voxel columns by `x_index = floor((p.x - origin_x) / voxel_size)`
4. Cumulative-sum the column counts; `K_split_voxel_x = searchsorted(cumulative, target_count_gpu0)`

Result: GPU 0 owns voxel columns `[0, K_split_voxel_x)`, GPU 1 owns `[K_split_voxel_x, GRID_NX)`. Particle counts per side approximate the weight ratio to within one voxel-column's worth of particles.

Why bin-then-search rather than `sorted_x[floor(N * f0)]`: a sorted-particle quantile gives a real-valued X that may fall mid-voxel, then we'd snap to the nearest voxel boundary and the actual partition no longer matches the requested fraction exactly. Binning makes the snap explicit and the count error one voxel column's-worth, which is what we'd get anyway.

Why particle-count-based rather than domain-bounding-box midpoint: handles non-uniform initial distributions automatically (dam break with 90% of fluid in one corner gets a deep cut into that corner instead of an even domain split that gives one GPU almost nothing).

Boundary / inlet / rotor particles partition by the same voxel rule — whichever voxel column they sit in, that GPU owns them. They are static so this is decided once at startup and never re-evaluated.

### Per-GPU voxel grid extent

Each GPU's local voxel grid covers its own columns plus a 1-voxel ghost layer on the interior side only:

```
GPU 0:  own columns      x_index ∈ [0, K_split_voxel_x)
        ghost columns    x_index = K_split_voxel_x         (peer's leftmost own)
GPU 1:  own columns      x_index ∈ [K_split_voxel_x, GRID_NX)
        ghost columns    x_index = K_split_voxel_x - 1     (peer's rightmost own)
```

Ghost on the outward (boundary-touching) side is empty — there is no peer there, only the wall. Each GPU's allocated grid is `(own_nx + 1) × ny × nz` voxels in V1 (single peer); the extra column is its ghost slab.

## Buffer Layout: Merged Scheme + x-Slowest Encoding

V1 (after the design pivot recorded in `experiment/v1/shaders/`) **abandons the separate set 2 ghost SoA** in favour of merging ghost into set 0 (particles) and set 1 (voxel structures). Rationale:

- Hot kernels (correction / density / force) need **zero ghost branches** in their neighbour iteration — extended voxel grid covers own + ghost uniformly.
- Multi-GPU extension (3+ GPUs in 1D, future 2D) just changes spec const values; no descriptor set explosion.
- Concept unifies: ghost is "an extended slice of the same grid" not "a separate buffer that looks like the grid".

### Voxel encoding: x-slowest

V0 used `voxel_id = x + y*NX + z*NX*NY + 1` (x-fastest). V1 switches to **`voxel_id = y + z*NY + x*NY*NZ + 1` (x-slowest)** in `experiment/v1/shaders/helpers.glsl`. With the partition cut along X, x-slowest makes own / ghost ranges contiguous in voxel_id space — dispatch ranges and `is_own_voxel(vid)` checks reduce to single comparisons.

V0 (`shaders/sph/`) keeps x-fastest, untouched.

### Pid and voxel_id layout (symmetric)

```
  voxel_id      [1, M] | [M+1, T-N] | [T-N+1, T]
                 ^^^^^   ^^^^^^^^^^   ^^^^^^^^^^^^^
                 leading      own      trailing      ghost ↔ extended grid

  pid           [1, M_pid] | [M_pid+1, M_pid+OWN_POOL_SIZE] | [...+1, ...+TRAILING_POOL]
                 ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 leading                own                            trailing      ghost particles
```

- `M = LEADING_GHOST_VOXEL_COUNT` (id=80); `N = TRAILING_GHOST_VOXEL_COUNT` (id=81); `T = GRID_DIMENSION_X * NY * NZ` (extended).
- `M_pid = LEADING_GHOST_POOL_SIZE` (id=54); `TRAILING_GHOST_POOL_SIZE` (id=55).
- End-of-chain GPU: corresponding ghost dim/pool = 0; sub-range collapses to empty, own naturally starts at 1.
- Middle GPU (1D 3+): all 5 sub-ranges populated; same code, different spec const values.

helpers.glsl exposes `own_first_pid()` / `own_last_pid()` / `leading_ghost_first_pid()` / `trailing_ghost_first_pid()` / `is_own_voxel(vid)` etc. Hot kernels dispatch over `[own_first_pid(), own_last_pid()]`; ghost_send writes into the corresponding ghost sub-ranges.

### ghost_send.comp: all-in-one packer (no recv kernel)

V1 ditches separate `ghost_recv` and `clear_ghost_voxel_count` kernels. **`ghost_send.comp` does everything**: per-voxel dispatch (NY*NZ threads), each thread atomic-adds `voxel_count` consecutive slots in one shot, writes both:

1. **set 0 ghost-pid range** — 7 SoA fields per particle (`position_voxel_id`, `velocity_mass`, `density_pressure`, `material`, `correction_inverse[*2]`, `density_gradient_kernel_sum`).
2. **set 1 ghost voxel structures** — `inside_particle_count` + `inside_particle_index` for the ghost voxel column.

The atomic counter and `inside_particle_count` are written *unconditionally* per thread, so previous-step's data is implicitly overwritten — no clear pass needed.

**Pid / voxel_id values written are pre-translated to PEER's frame** via two host-computed signed-int spec consts:

- `GHOST_PID_OFFSET_TO_RECEIVER` (id=93): `peer.dest_first_pid - my.dest_first_pid`.
- `GHOST_VOXEL_ID_OFFSET_TO_RECEIVER` (id=94): `peer.dest_first_vid - my.dest_first_vid`.

Receiver does a pure byte-level `vkCmdCopyBuffer` of MY ghost-pid range into PEER's matching ghost-pid range (and same for voxel range), then runs hot kernels directly. No per-particle translation on receiver side.

Five per-pipeline spec consts for ghost_send (id 90–94):

| id | name | meaning |
|---|---|---|
| 90 | `GHOST_DIRECTION` | 0 = leading send, 1 = trailing send |
| 91 | `BOUNDARY_VOXEL_X_LOCAL` | own column to read (extended-grid x) |
| 92 | `GHOST_VOXEL_X_LOCAL` | MY ghost column to write voxel structures |
| 93 | `GHOST_PID_OFFSET_TO_RECEIVER` | signed int |
| 94 | `GHOST_VOXEL_ID_OFFSET_TO_RECEIVER` | signed int |

### Per-direction dispatch

End-of-chain GPU runs ghost_send once per step (one direction). Middle 1D GPU runs twice (leading + trailing). 2D 4-peer middle (V3++) runs four times. **Each dispatch is independent**; same SPV, different spec const values, no contention (writes to disjoint sub-ranges, each direction has its own atomic counter in `global_status`).

## Migration Channel (V1.0a, folded into ghost SoA — path 2)

Cross-partition migration = an own particle drifts into the ghost voxel range. V1.0a integrates migration into the merged-buffer scheme: there is **NO separate Channel B**. Instead, ghost SoA carries two kinds of payload, distinguished by the `voxel_id` field of each slot:

```
ghost-pid slot k:
  .position_voxel_id.w = voxel_id (in receiver's frame)
        ├─ in receiver's GHOST range  → REPLICA   (used as ghost neighbour by hot kernels)
        └─ in receiver's OWN range    → MIGRATION (installed as own particle by install_migrations.comp)
```

Sender's `ghost_send.comp` packs both kinds in the same byte stream using the same `GHOST_VOXEL_ID_OFFSET_TO_RECEIVER` spec const (lucky property: replica and migration offsets are equal because their physical voxel offset cancels out — see derivation below).

Receiver's new `install_migrations.comp` walks the ghost-pid range after transport, splits replica vs migration based on the voxel_id tag.

Re-add `ghost_acceleration` (set 2 binding 4) and `ghost_shift` (set 2 binding 5) to ghost SoA — the migrated particle needs these for the receiver's NEXT step's predict integration.

### Sender (predict + ghost_send)

`predict.comp`: when an own particle's new voxel_id is in ghost range, it atomically appends to `incoming_particle_index[ghost_voxel_id]` just like any voxel change. update_voxel.comp's `is_own_voxel` filter skips this entry (ghost voxels are not its responsibility), but ghost_send picks it up.

`ghost_send.comp`: per-(y, z) thread reads two sources:
1. **Own boundary voxel's `inside_particle_index`** (post-update_voxel) → packs as REPLICA, voxel_id_for_peer = own_boundary_vid + GHOST_VOXEL_ID_OFFSET_TO_RECEIVER (in peer's ghost range).
2. **Adjacent ghost voxel's `incoming_particle_index`** (predict's own→ghost crossings this step) → packs as MIGRATION, voxel_id_for_peer = ghost_vid + GHOST_VOXEL_ID_OFFSET_TO_RECEIVER (lucky property: same offset; lands in peer's own range). Sender marks the migrating own pid as dead (`voxel_id = 0`, `mass = 0`).

After packing, ghost_send resets `incoming_particle_count[ghost_voxel_id] = 0` so it doesn't double-count next step.

### Receiver (install_migrations.comp)

```
per-ghost-pid thread (dispatch over GHOST_POOL_SIZE):
  vid = ghost_position_voxel_id[gpid].w
  if vid == 0:                       return   # dead/consumed slot
  if !is_own_voxel(vid):             return   # replica, leave for hot kernel
  
  # migration arrival: install as own
  own_pid = allocate_dead_own_slot()          # see "slot allocation" below
  copy 9 fields ghost_<*>[gpid] → <*>[own_pid]
  atomic-append own_pid to inside_particle_index[vid]
  ghost_position_voxel_id[gpid].w = 0          # consume
```

### Migration slot allocation: strategy 1 (V1.0a) and strategy 2 (V1.x)

`install_migrations` needs a dead slot in the own pid range. Two strategies:

#### Strategy 1: end-allocate (V1.0a default)

Atomic counter `migration_install_count` in `global_status`, init to 0 at simulator start. Each install: `slot_n = atomicAdd(migration_install_count, 1)`, `own_pid = own_last_pid() - slot_n`.

Migrations install at the **tail** of own pid range, indices descending. Mid-range dead slots (left over from predict kills, ghost_send migration-out kills) are NOT recycled — they stay dead until next defrag, when V0 defrag's compaction packs all alive into [own_first_pid, alive_count] and frees the rest.

- Per install: 1 atomicAdd on `migration_install_count` + 1 atomicAdd on `inside_particle_count[vid]` + 9 SoA writes.
- Per step overhead: ~80 installs × (~150 cycles avg per atomic) ≈ **4-10 μs/step** at lid scale.
- Storage: 1 uint counter, no buffer.
- Reset cadence (V1.0a default): **only after each defrag dispatch**. Counter monotonically grows across steps within a defrag cycle. End-pointer descends through the post-defrag dead region. Defrag re-packs alive particles to the front and resets the counter, starting a fresh end-allocate cycle from `own_last_pid()`.
- Per-step reset (alternative, V1.x candidate): resetting `migration_install_count = 0` at each step start would re-use the same tail slots every step. This works ONLY if those slots are reliably re-killed each step (e.g., by ghost_send marking previous step's installs as dead). Without that guarantee, per-step reset would overwrite live migrations from prior steps. Documented here as a possible future micro-optimization; V1.0a sticks with the safer post-defrag-only reset.

**Risk**: end pointer drops monotonically (~80/step at lid scale, up to 1000+/step at high-flux). Over `defrag_cadence` (default 1000 steps), pointer falls by ~80K to ~1M. If the gap from the start of the dead region (= alive_count) is smaller than this drop, install fails and overflow_migration_install increments. Lid case has ~860K headroom → safe by ~10×. High-flux dam-break-style cases may exhaust earlier and need either smaller `defrag_cadence` or strategy 2.

#### Strategy 2: free-slot stack (V1.x upgrade)

Maintain explicit free-pid stack — `free_pid_stack[FREE_STACK_SIZE]` + `free_pid_count`:
- `predict.comp` and `ghost_send.comp` push killed pids onto the stack (atomicAdd to count + write slot).
- `install_migrations` pops from the stack (atomicSub to count + read slot).
- After each defrag, simulator runs a one-shot rebuild kernel: parallel prefix-sum over own pid range to identify dead slots, write them to stack.

- Per install: 1 atomicSub + 1 stack read + same downstream writes.
- Per kill (predict / ghost_send): +1 atomicAdd + 1 stack write.
- Per step overhead: ~10-16 μs/step (~80 push + ~80 pop + the inside_count atomic).
- Storage: 1 uint counter + free_pid_stack (uint × OWN_POOL_SIZE potential = 4.8 MB worst case at 1.2M; in practice cap at ~50% expected = ~2.4 MB).
- Defrag interaction: extra ~50 μs dispatch to rebuild stack.

Net difference vs strategy 1: ~5-10 μs/step (~0.07-0.14% of step time).

**Use strategy 2 when**: V1.0 lid passes but high-flux validation cases trip `overflow_migration_install`, or `defrag_cadence` becomes too small for stable behaviour.

### Why this works (and bit-exact concerns ducked)

Migration is **authoritative on the source GPU**: only one side runs predict on the particle (= the side that owns it at the start of the step). That side's predict is the single source of truth for the particle's new voxel. If the new voxel falls in ghost range → migration. Receiver passively installs.

Cross-vendor floating-point ULP differences don't cause ambiguity here — only ONE GPU computes the new position, and that GPU's classification (own / ghost) is final. No double-count, no loss.

This is why we explicitly **don't run predict on ghost particles**. Doing so (mechanism B in the design discussion) would require bit-exact agreement between two GPUs on each ghost particle's new position, which fails cross-vendor and creates ~0.025-0.5% particle count drift.

## Sync Cadence

Sync options vs ghost staleness in hot kernels:

| Option | Sync after | Mismatches per step | Notes |
|---|---|---|---|
| A | force only | 4 (correction, density read stale x/v/ρ; force reads stale x) | obsolete |
| **B** | **update_voxel** | **1** (force reads stale ρ — ~1e-4 error) | **V1.0 baseline** |
| C | density only | 4 (correction uses 1-step-old positions) | worse than B for KCG |
| D | update_voxel + density | 0 | V1.0a / V1.1 if smoke validation demands |

**V1.0 = Option B** (single sync after update_voxel). Trade-off: force.comp reads ghost ρ/P that's one density step stale → ~1e-4 relative error on pressure gradient at the boundary band. Acceptable for cross-vendor portability section of the paper; not bit-exact anyway.

**Option D upgrade is V1.0a/V1.1** because it requires SLOT-STABLE ghost packing across the two syncs. ghost_send's per-voxel atomicAdd order is non-deterministic across threads, so a second sync would write ρ/P into different slots than sync 1 — ghost data inconsistent. Solving needs a deterministic allocator (prefix-sum) or a per-particle persistent slot map; both are V1.0a complexity.

V1 explicitly does not bit-match across vendors; "small mismatch count" means how many kernels per step read ghost data computed in a *different* step than their own data. Floating-point ULP differences cross-vendor are accepted regardless.

## Pipeline: V1.0 Single-Sync Per Step

Pure single-threaded blocking submission. `wait` = `vkQueueWaitIdle`. Each phase finishes on both GPUs before the next begins.

```
Phase 1  GPU0 [predict + update_voxel + ghost_send (1 or 2 dispatches per direction)]
         GPU1 [same]
         wait

Phase 2  GPU0 [vkCmdCopyBuffer set 0 ghost-pid range + set 1 ghost-voxel slice
              + ghost_send_*_count → host_visible_send_buffer]
         GPU1 [same]
         wait

Phase 3  CPU memcpy: swap host buffers between GPU pair, with leading↔trailing
         direction crossing (MY leading_send → PEER trailing_recv, etc.)

Phase 4  GPU0 [vkCmdCopyBuffer host_visible_recv_buffer → set 0 ghost-pid range
              + set 1 ghost-voxel slice + ghost_recv_*_count]
         GPU1 [same]
         wait
                                            ── sync complete ──

Phase 5  GPU0 [correction + density (scratch+copy) + force]
         GPU1 [same]
         wait
                                            ── step complete ──
```

Five blocking phases per step instead of nine. No second sync (V1.0 = Option B).

Each transport copies three byte-contiguous slices per direction:

1. set 0 ghost-pid range bytes (~1.5 MB / direction at 16K particles, 92 B each)
2. set 1 ghost-voxel slice (`inside_particle_count` + `inside_particle_index`, ~80 KB)
3. `ghost_send_*_count` uint (4 B)

Total per-step transport: ~3.2 MB (both directions × full pack), ≈ 0.23 ms over 14 GB/s practical PCIe.

V1.1 replaces `vkQueueWaitIdle` with timeline semaphores + interleaved submission.

## Boundary Identification

ghost_send.comp dispatches per ghost voxel (NY*NZ threads). Each thread reads its own's outermost voxel (at `BOUNDARY_VOXEL_X_LOCAL`) and walks `inside_particle_index` of that one voxel to pick which particles to pack. **No persistent `boundary_pid_list` is needed** — the boundary set is implicit in the dispatch shape and the per-thread voxel lookup.

vkCmdCopyBuffer transports the contiguous ghost-pid range (capacity = `LEADING_GHOST_POOL_SIZE` or `TRAILING_GHOST_POOL_SIZE`, sized at observed-particles × ~3 headroom). Sender-side overflow is logged in `global_status.overflow_ghost_count`. Active particles are front-packed in the ghost range by per-voxel atomic-allocation; trailing capacity bytes carry stale data that the receiver's hot kernels never reference (because `inside_particle_count` for ghost voxels caps the loop at the actual count).

## Numerical Validation Plan

Cross-vendor will not bit-match V0 single-GPU. Two-tier validation:

1. **Smoke**: lid-driven cavity case, run 10k steps, compare vs V0 single-GPU:
   - Total kinetic energy curves: max relative deviation < 0.5%
   - Particle count per X-band over time: drift < 1%
   - No NaNs, no overflow flags
2. **Reproducibility within V1**: same seed + same partition weights → same trajectory (same machine). Detects nondeterminism from `vkQueueWaitIdle` ordering or atomic ordering.

Tier 2 first (much cheaper to set up), then tier 1 once V1 is stable. The legacy OpenGL gate from `feedback_api_rewrite.md` remains separate — V0 should pass it, and V1 inherits "approximately equivalent to V0" as its bar.

## Milestones

| Milestone | Definition of done |
|---|---|
| **V1.0** | 5-phase blocking pipeline (single sync, Option B) runs the lid case across both GPUs without crash. Tier-2 reproducibility passes. Phase timing CSV instrumented. force.comp boundary band has ~1e-4 ρ/P stale bias from Option B (known + bounded). |
| **V1.0a** | Add **Option A sync 2** (full-rerun ghost_send between density and force) — eliminates the ρ/P bias. Costs ~300 μs/step (~4% slowdown) at this stage; still blocking. Tier-1 smoke validation passes. |
| **V1.1** | Replace `vkQueueWaitIdle` with timeline semaphores + interleaved phase submission. Same numerical output as V1.0a; transport queues still on the same compute queue (no async DMA overlap yet). |
| **V1.2** | Per-kernel timestamp queries collected; per-kernel weight model replaces single global weight if kernel-mix imbalance is significant. (Whole-pipeline calibration is already done — see `KNOWN_GPU_SPH_WEIGHT` measurements above.) |
| **V1.3** (deferred) | NV+NV P2P backend swapped in; ghost via `vkCmdCopyBuffer` instead of CPU staging. Probe `probe_interop.py` first. |

V1.0 is the only required milestone for the paper's portability section. V1.0a is highly recommended (sync 2 = numerically clean baseline). V1.1+ are perf optimisations.

## Sync Strategy Decision (V1.0a → V2 Path)

**V1.0a = Option A full re-run for sync 2** (not B/C selective ρ/P refresh). Rationale:

force.comp's only stale field on ghost neighbours is ρ/P. A "ρ/P-only sync 2" looks attractive (10× less bandwidth) but breaks slot stability — sync 1's per-voxel atomicAdd allocates ghost-pid slots non-deterministically, so a second selective write can't reliably target the same slots. Workarounds (fixed-stride or prefix-sum allocators) require shader rewrites and add complexity.

Option A simply re-runs ghost_send.comp unchanged. New atomic-add allocation in sync 2 produces a different slot mapping than sync 1, but inside_particle_index is also re-written → consistent end state. Zero shader changes, zero slot-stability concerns.

**Bandwidth cost looks bad on paper**: ~3 MB/step extra transport vs ~316 KB for ρ/P-only. But under V2's async architecture (transfer queue concurrent + interior/boundary force split, see `Out-of-Scope for V1`), sync 2 transport runs in parallel with `force_interior` compute (~2.7 ms) and is fully hidden in wall time. **Net wall-time cost approaches zero** when force_interior dominates the step.

So the long-term answer is **A + V2 async overlap, not B/C**. We don't bypass via clever allocators because the transport cost evaporates anyway once async is wired up. The B/C alternatives' costs (memory bloat or extra prefix-sum kernel) are real and remain even after V2.

V1.0a / V1.1 timeline:
- V1.0a: Option A sync 2, blocking, ~300 μs/step. Wall-time slowdown ~4%.
- V1.1: Timeline semaphores remove redundant queue-idle barriers; sync 2 still on compute queue. Slowdown drops to ~3%.
- V2: Transfer queue + interior/boundary force split. Sync 2 transport hides behind force_interior. Slowdown approaches 0%.

## Out-of-Scope for V1

- Async compute queue (V2)
- Bit-exact ghost Kernel-A (V2 / same-vendor)
- Dynamic re-partitioning (V3+; see "Future: wait-time-driven K_split adjustment" below)
- 3+ GPU configurations (V3+)
- Y/Z split axes (V2 if benchmarks demand)
- Multi-resolution / micropolar / thermal extensions

## Future: wait-time-driven K_split adjustment (V3+)

Static partition based on GPU compute weights gives a good initial split but doesn't adapt to mid-run imbalance — e.g., one side of the cavity develops more turbulence and ends up with denser per-voxel particle counts, or thermal throttling on one card shifts the effective ratio. Dynamic re-partitioning closes this gap.

Mechanism sketch:

1. **Instrument per-phase wait time.** The 9-phase blocking loop already inserts `vkQueueWaitIdle` between GPU 0 and GPU 1 at each phase boundary. Sample wall-clock time at "GPU i finished phase k" (`t_finish[i][k]`) for each `(i, k)`. Idle time of the earlier-completing GPU at phase k = `max_i(t_finish[i][k]) - t_finish[earlier][k]`.
2. **Aggregate over a window.** Rolling N-step (e.g. N=100) average of total per-step idle time per GPU. `wait[0] = Σ_k (max_i t_finish[i][k] - t_finish[0][k])` and similarly for GPU 1. Exactly one of `wait[0]` / `wait[1]` is positive at each phase (the earlier-finisher waits); aggregating across phases tells you who's globally faster.
3. **Decision rule.** If `wait[0] > wait[1] + threshold` (GPU 0 chronically idle waiting for GPU 1), GPU 1 is overloaded → shift K_split by 1 voxel column toward GPU 1 (give GPU 0 more particles). Threshold should be > one-step jitter (e.g. 5% of mean step time × N).
4. **When to act.** Schedule re-balance to coincide with periodic defrag (every `defrag_cadence` steps, currently 1000). Defrag is already a "natural break" where particle ordering is reshuffled. K_split shift causes one column's worth of particles to change ownership — handled by the existing migration channel (Channel B) for that one step.
5. **Convergence.** Step-1-at-a-time shifts give a conservative integral controller. Over hundreds of defrag cycles the system finds the wait-minimising K_split. No PID complexity needed.

Why this is V3+ not V1: needs migration channel (V1 work), needs 9-phase pipeline timing instrumentation (V1.0 baseline gathers it but doesn't act on it), and the actual perf payoff requires a real workload imbalance to show up. V1 / V2 milestones use static K_split from `KNOWN_GPU_SPH_WEIGHT`; V3 layers this on once the substrate is solid.

Per-kernel weighting (V1.2 milestone, see Milestones table) is a related but distinct optimisation: instead of one global weight per GPU, learn separate weights for each kernel (correction is ALU-heavy, density is BW-heavy) and dispatch each kernel to whichever GPU has more headroom for that pattern. Wait-time adjustment operates on K_split (data partition); per-kernel weighting operates on dispatch routing — they compose.

## Cross-References

- Multi-GPU rationale + V0-V3 arc: `docs/sph_design.md`
- V0 buffer layout (set 0/1/3, unchanged in V1): `docs/sph_v0_design.md`
- Per-kernel invariants + spec constant table: `shaders/sph/common.glsl`, `shaders/sph/README.md`
- Phase 1 CPU-staged migration reference (transport pattern reuse): `main_multigpu_particles.py`
- Cross-vendor probe results (negative): `probe_interop.py` output, memory `phase2_findings`
