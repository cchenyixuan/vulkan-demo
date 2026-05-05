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

## Descriptor Set 2: Ghost Activation

V0-a kept set 2 dummy-bound and `GHOST_DIMENSION_* = 0` to dead-code-eliminate ghost branches. V1 activates them. Layout already declared in `common.glsl` (parallels set 0 + set 1):

| binding | Buffer | Type | Purpose |
|---|---|---|---|
| 0 | `ghost_position_voxel_id` | `vec4` | x, y, z, ghost_voxel_id_as_float |
| 1 | `ghost_density_pressure` | `vec2` | ρ, P (refreshed at sync 2) |
| 3 | `ghost_velocity_mass` | `vec4` | refreshed at sync 1 |
| 4 | `ghost_acceleration` | `vec4` | refreshed at sync 1 (used by predict on next step's ghost re-eval if any; in V1 pure path only own particles run predict, so this is mostly read-only) |
| 5 | `ghost_shift` | `vec4` | refreshed at sync 1 |
| 6 | `ghost_material` | `uint` | refreshed when ghost identity changes |
| 10 | `ghost_inside_particle_count` | `uint` | per ghost-voxel |
| 12 | `ghost_inside_particle_index` | `uint × MAX_PARTICLES_PER_VOXEL` | per ghost-voxel |

Spec constants flipped in V1:

- `GHOST_DIMENSION_X/Y/Z` (constant_id=80,81,82): peer's grid dimension covering the 1-voxel ghost extent
- `GHOST_ORIGIN_X/Y/Z` (id=83,84,85): peer's ghost grid origin in world space
- `OWN_TO_GHOST_OFFSET_X/Y/Z` (id=86,87,88): translation from own coord to ghost coord (used in cross-partition voxel lookup)

Ghost particle count cap: `GHOST_POOL_SIZE` (id=53 region — final id TBD, declare in `common.glsl` when implementing) sized at ~10% of own pool; overflow logged into `global_status.overflow_ghost_count`.

## Migration Channel (set 3)

Two new bindings on set 3:

- binding 8 — `outgoing_to_peer` — packed particle records (x, v, ρ, a, shift, material) appended atomically inside `update_voxel` when a particle crosses `K_split_x`. `outgoing_count` lives in the first slot.
- binding 9 — `incoming_from_peer` — peer's previous step's outgoing, copied in by host before this step's `update_voxel`.

Packet format = same SoA struct as own particle (matches set 0 bindings 0-6 exactly, ~80 B/record). Pool sized at observed-cross-rate × 4 safety; overflow logged.

Migration is **a Channel B distinct from ghost (Channel A)**:
- Channel A (ghost SoA): replicated state of peer's boundary-side particles, refreshed every sync. Used by own correction/density/force when their neighbour iteration crosses into ghost voxels. **Read-only on receiver side.**
- Channel B (migration packets): one-way handoff of ownership when a particle's new voxel crosses the partition. Receiver installs into a dead slot in own SoA on next step's `update_voxel`. **Mutates receiver's own state.**

Why separate channels: a ghost-side particle should not have force-or-density computed by the receiver, only consumed as a neighbour. A migrated particle becomes fully-owned and runs every kernel locally. Folding them would require a `is_ghost_or_owned` flag on every particle and conditional integration — exactly the kind of branch the V0 single-pool design avoided. Bit-exact Kernel-A unification (the original `sph_design.md` plan) achieves this elegantly but only same-vendor; cross-vendor needs the explicit channel.

## Sync Cadence (Option D, double sync)

Every step has **two CPU↔CPU exchanges**, one after `update_voxel`, one after `density.comp`. Each exchange is bidirectional: GPU0 reads back its boundary-side data + outgoing migration packets to host, host swaps and uploads to peer's ghost / incoming buffers, GPU1 same.

Why two and not one:

| Option | Sync after | Mismatches per step | Notes |
|---|---|---|---|
| A | force only | 4 (correction, density read stale x/v/ρ; force reads stale x) | original V1.0 sketch |
| B | update_voxel only | 1 (force reads stale ρ — ~1e-4 error) | density still uses stale neighbour ρ |
| C | density only | 4 (correction iteration uses 1-step-old positions) | worse than B for KCG |
| **D** | **update_voxel + density** | **0** | each kernel sees same-step ghost state |

Cost of going from B → D: one extra round-trip per step. At ~3 MB ghost ρ/P (the second sync only carries 8 B/particle for ρ+P, much smaller than sync 1's full SoA) over 14 GB/s ≈ 0.2 ms — small relative to a 3-5 ms step at 1M particles. Mismatch elimination is worth it.

V1 explicitly does not bit-match across vendors; "0 mismatches" here means "each kernel reads ghost data computed in the **same step** as its own data, not a step behind". The actual ghost values still differ at floating-point ULP level cross-vendor, but the algorithmic structure is identical to single-GPU.

## Pipeline: 9 Phases per Step

Pure single-threaded blocking submission. `wait` = `vkQueueWaitIdle` on the relevant queue (no fence-and-poll optimisation in V1.0). Each phase finishes everywhere before the next begins.

```
Phase 1  GPU0 [predict + update_voxel] → wait
         GPU1 [predict + update_voxel] → wait
                                                    update_voxel atomic-appends:
                                                      - own crossings stay local
                                                      - cross-partition crossings → outgoing_to_peer

Phase 2  GPU0 [readback boundary SoA + outgoing_to_peer → host] → wait
         GPU1 [same] → wait

Phase 3  CPU memcpy: swap host buffers (GPU0.boundary → GPU1.ghost_in_host,
                                         GPU0.outgoing → GPU1.incoming_in_host,
                                         and vice versa)

Phase 4  GPU0 [upload ghost_in_host → ghost SoA, incoming → set 3 binding 9] → wait
         GPU1 [same] → wait
                                                    ── sync 1 complete ──

Phase 5  GPU0 [correction + density (scratch+copy)] → wait
         GPU1 [same] → wait
                                                    density.comp writes scratch;
                                                    same-cmd vkCmdCopyBuffer scratch→primary
                                                    (same as V0)

Phase 6  GPU0 [readback boundary ρ/P → host] → wait
         GPU1 [same] → wait
                                                    sync 2 carries only set 0 binding 1
                                                    (ρ, P), 8 B/boundary particle

Phase 7  CPU memcpy: swap host ρP buffers

Phase 8  GPU0 [upload host ρP → ghost_density_pressure (set 2 binding 1)] → wait
         GPU1 [same] → wait
                                                    ── sync 2 complete ──

Phase 9  GPU0 [force] → wait
         GPU1 [same] → wait

Phase 10 (next step's Phase 1)
```

Each `wait` in V1.0 = full queue-idle. V1.1 will replace with timeline semaphores + interleaved submission so GPU1 can be working on phase N while GPU0 waits on phase N+1's host-side prep.

## Boundary Identification (which particles to read back)

`update_voxel` already knows each particle's voxel. A particle is "boundary on the peer side" if `voxel.x_index ∈ [K_split_x - 1, K_split_x]` (own-side boundary, peer needs as ghost) OR `voxel.x_index == K_split_x + 1` for the peer-receiving side (informational only, peer's data we hold).

V1 simple approach: `update_voxel` writes a `boundary_pid_list` (set 3 new binding 10, atomic-append) of own particles in the 1-voxel ghost-source band. Phase 2 readback uses this list as a gather index; readback size = O(boundary_count) not O(N). At ~1M particles, boundary_count ≈ 5-10% → 50-100k particles → 5-10 MB SoA per direction.

Alternative considered: readback the full set 0 every step. Costs ~150 MB/step at 1M particles → blows the BW budget. Rejected.

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
| **V1.0** | Pure 9-phase pipeline runs the lid case across both GPUs without crash. Tier-2 reproducibility passes. Phase timing CSV instrumented. |
| **V1.0a** | Tier-1 smoke validation passes (energy + count drift bounded). |
| **V1.1** | Replace `vkQueueWaitIdle` with timeline semaphores + interleaved phase submission. Same numerical output. |
| **V1.2** | Per-kernel timestamp queries collected; per-kernel weight model replaces single global weight if kernel-mix imbalance is significant. (Whole-pipeline calibration is already done — see `KNOWN_GPU_SPH_WEIGHT` measurements above.) |
| **V1.3** (deferred) | NV+NV P2P backend swapped in; ghost via `vkCmdCopyBuffer` instead of CPU staging. Probe `probe_interop.py` first. |

V1.0 is the only required milestone for the paper's portability section. V1.1+ are perf optimisations.

## Out-of-Scope for V1

- Async compute queue (V2)
- Bit-exact ghost Kernel-A (V2 / same-vendor)
- Dynamic re-partitioning (V3+)
- 3+ GPU configurations (V3+)
- Y/Z split axes (V2 if benchmarks demand)
- Multi-resolution / micropolar / thermal extensions

## Cross-References

- Multi-GPU rationale + V0-V3 arc: `docs/sph_design.md`
- V0 buffer layout (set 0/1/3, unchanged in V1): `docs/sph_v0_design.md`
- Per-kernel invariants + spec constant table: `shaders/sph/common.glsl`, `shaders/sph/README.md`
- Phase 1 CPU-staged migration reference (transport pattern reuse): `main_multigpu_particles.py`
- Cross-vendor probe results (negative): `probe_interop.py` output, memory `phase2_findings`
