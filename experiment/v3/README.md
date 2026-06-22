# experiment/v3 — ABANDONED (negative-result record, do NOT build on this)

**Status: dead end. The forward branch is `experiment/v4/` (see `docs/sph_v4_summary.md`).**

## What this was

V3.3 explored a different cross-GPU **transport architecture**: a single host buffer imported by both
`VkDevice`s via `VK_EXT_external_memory_host` (shared host RAM — no CPU worker-thread memcpy), with
**compute-queue cross-device binary semaphores** (`VK_KHR_external_semaphore_win32` OPAQUE_WIN32) for
sync, eliminating the dedicated transfer queue and the host-signal worker. See
`utils/shared_host_transport_v3.py` and `utils/orchestrator_v3.py`.

## Why it was abandoned

It was **projected to be faster** (memory `project_cross_vendor_shared_host_breakthrough.md`: 274→300+
fps at 1M, by removing the ~236µs worker memcpy and the transfer-queue NV cache invalidation). But when
measured, it was **~10% SLOWER** than the V2/V4 CPU-worker + transfer-queue architecture at matched
settings:

| cavity 1M, 2.9:1.0, pool_safety=1.2, depth-1 | fps |
|---|---:|
| V3.3 shared-host (this dir) | **260.7** |
| V4 = V2 Path A+ architecture (`experiment/v4/`) | **292.9** |

The likely cause (not fully root-caused — would be paper follow-up): putting the cross-device semaphore
on the **compute queue** serializes the two GPUs' compute more than the V2/V4 design, which hid the
entire transfer chain behind `phase_b` on a *separate* transfer queue. The shared host RAM + compute-Q
sync trades away that overlap.

## Why it's kept (not deleted)

This is a **reproducible negative result** for the paper's cross-vendor portability section — and a
non-obvious one (projection contradicted by measurement). Text source only; `.spv`/`__pycache__`/`logs`
are gitignored, so the byte cost is negligible.

## Note

The V4 improvements (set-3 cleanup, `PoolHealthBuffer` watermark, per-slab pool sizing, migration
logging) were applied here too **before** V3 was abandoned — but the *architecture* is the deal-breaker,
not those mods. Do not port anything *from* here; port forward into `experiment/v4/` only.
