# Band-dispatch discrimination experiment — 2026-09-17 (branch `exp/band-compact`, not merged)

Question: is the 1.2–1.7× per-particle cost of the Phase C band kernels caused by their (band voxel, slot)
thread mapping, i.e. would the interior's mapping (one thread per particle, consecutive threads =
consecutive pids) bring it down to ≤ 1.1× interior?

## Re-check of the per-particle numbers (each band kernel over its own band: 2 / 3 / 4 columns)

Raw = kernel time / particles in that band; the interior rate = phase-B kernel time / (own − band).
Border-4 stretched 3-D case (K=2, 4M/GPU): band columns hold 111,556 particles each → bands 223,112 /
334,668 / 446,224; 2-D 4M: 10,225 per column → 20,450 / 30,675 / 40,900.

| kernel | 2-D 4M K=2 raw ns/p (band / interior) | 2-D fit over 4M/16M/32M: fixed µs + marginal ns/p | 3-D K=2 raw ns/p (band / interior) |
|---|---|---|---|
| correction (band 2) | 3.96 / 0.50 = 7.9× | 77 + 0.18 (marginal 0.37× interior) | 4.25 / 2.57 = 1.65× |
| density (band 3) | 2.87 / 0.52 = 5.5× | 39 + 1.26 (2.5×) | 3.87 / 2.80 = 1.38× |
| force (band 4) | 2.23 / 0.69 = 3.2× | 42 + 1.44 (2.1×) | 4.38 / 3.52 = 1.24× |

So "correction is cheaper than interior" holds only for the *marginal* per-particle cost in 2-D (the slope of
time vs band size); its raw per-particle cost is 3–8× because a ≈ 77 µs launch/tail term dominates a kernel
of 80–90 µs. In 3-D, where the bands hold 0.2–0.8M particles, the fixed term is < 5% and the raw ratios
(1.65 / 1.38 / 1.24×) are the per-particle cost.

## Experiment

`V5_BAND_COMPACT_DISPATCH=1` (branch `exp/band-compact`, commit 0e18e49; default off, not on the main path):
in Phase C, after install_migrations, a single-workgroup exclusive scan over the band-4 voxels (in
band-voxel order = column-major, which after a defrag is pid order) and a scatter kernel write the band pids
into a contiguous list (stored in the dormant `extension_fields` SoA; column starts and the indirect dispatch
sizes for band 2 / 3 / 4 in a 128 B meta buffer). The three band kernels then run their unchanged
`process_particle` bodies with one thread per list entry through `vkCmdDispatchIndirect`. Seam check, drift
0, stamps 0 on both runs.

| case | kernel | band mode (V3.4) µs | compact µs | compact ns/p | ratio to interior: band mode → compact |
|---|---|---|---|---|---|
| 2-D 4M K=2 | correction | 80–82 | 87–89 | 4.3 | 7.9× → 8.6× (fixed-cost dominated, +21 µs list build) |
| 2-D 4M K=2 | density | 86–88 | 94–96 | 3.1 | 5.5× → 5.9× |
| 2-D 4M K=2 | force | 86–99 | 88–90 | 2.2 | 3.2× → 3.2× |
| 3-D K=2 4M/GPU | correction | 944–952 | 749–771 | 3.41 | **1.65× → 1.31×** |
| 3-D K=2 4M/GPU | density | 1288–1300 | 1258–1280 | 3.79 | **1.38× → 1.35×** |
| 3-D K=2 4M/GPU | force | 1954–1956 | 1720–1758 | 3.90 | **1.24× → 1.10×** |
| 3-D | list build | — | 68 | | phase C 4.36–4.38 → 3.96–4.05 ms (−8%); fps 24.0 → 23.5 (one run each, within noise) |

## Verdict

The interior-style mapping removes part of the excess (correction −20%, force −11%) but density does not
move, and only force reaches the ≤ 1.1× criterion; correction (1.31×) and density (1.35×) do not. So the
thread mapping is not the main cause; per the plan no further A/B is run and the next step is a
**Nsight Graphics GPU Trace** of the density and force band kernels (SM occupancy, warp divergence,
L1/L2 hit rates). **Nsight Graphics is not installed on this machine** (only Nsight Compute 2025.4 and
Nsight Systems 2025.5 are present, neither profiles Vulkan compute dispatches at that level); installing
it needs the user (NVIDIA developer download). The band-mode and compact-mode runs are reproducible from
the branch once it is available.

Files: branch `exp/band-compact` (band_compact.comp, helpers `compact_thread_particle`, simulator recording,
`band_compact_us` in the anatomy); logs in the scratchpad `compact/`.

## Interior vs band: what the kernel actually executes (specialized SPIR-V diff)

The interior and band pipelines are the same shader module with different specialization constants
(`*_MODE` 1 vs 2, `BAND_VOXEL_DISPATCH` 0 vs 1). To see the code each variant runs, the frozen `.spv` was
specialized with `spirv-opt --set-spec-const-default-value … --freeze-spec-const
--fold-spec-const-op-composite --eliminate-dead-branches -O` for both variants and disassembled; the opcode
sequences (ids stripped) were diffed.

| kernel | interior: instructions / loads / access chains / branches / loops | band: same | band − interior |
|---|---|---|---|
| correction | 541 / 16 / 19 / 62 / 6 | 584 / 18 / 21 / 69 / 6 | +43 / +2 / +2 / +7 / 0 |
| density | 476 / 18 / 19 / 53 / 4 | 519 / 20 / 21 / 60 / 4 | +43 / +2 / +2 / +7 / 0 |
| force | 618 / 17 / 19 / 66 / 4 | 661 / 19 / 21 / 73 / 4 | +43 / +2 / +2 / +7 / 0 |

The 48 added / 5 removed opcodes are identical for the three kernels and all sit in `main()` before
`process_particle`: the band variant replaces "pid = thread + own_first_pid; if pid > own_last_pid return"
(5 opcodes) with `band_thread_particle`: thread → (band voxel, slot) (3 `OpUDiv`, 2 `OpUMod`, 2 `OpIMul`,
5 `OpIAdd`), side selection (3 `OpSelectionMerge` + `OpBranchConditional`, 3 `OpPhi`), then **two dependent
global loads** — `inside_particle_count[vid]` (slot ≥ count → exit) and `inside_particle_index[vid·MPV + slot]`
— before the body can issue its first load of `position_voxel_id[pid]`. **Per neighbour the two variants
execute exactly the same instructions** (the loop bodies are opcode-identical; same loop count, same loads
per neighbour). So the extra cost is not instruction count in the neighbour loop; what differs per particle is:

1. two serialized dependent loads (count → index → self position) instead of an add — ~2 extra DRAM
   latencies at the start of every thread, partially hidden by occupancy;
2. the self pid arrives through a gather (`inside_particle_index`, voxel-list order = atomic arrival order)
   instead of `thread_id + const`, so a warp's 32 self loads are no longer one coalesced 512 B line, and its
   neighbour loops start from 32 different voxels in a less regular order than the pid-sorted interior warp;
3. 75% (2-D, 24 of 96 slots) / 50% (3-D, 64 of 128) of the launched threads exit at the count check —
   warps that are only partly full run the loop body with idle lanes.

The compact-list experiment removes (1) and (3) and most of (2), and recovers −20% (correction) / −11%
(force) / −2% (density): consistent with the mapping being a secondary factor. What remains (density
1.35×, correction 1.31×) has to come from the memory side of the *same* instruction stream — the band
particles' neighbour lists and neighbour data are only 2–4 columns wide, i.e. a working set that is
streamed once per kernel rather than reused across a wide pid range, so the per-neighbour loads see a lower
L2 hit rate than in the interior; that is exactly what the GPU Trace metrics (L2 hit rate, warp stall
reasons) would show.

## Nsight Graphics

2026.3.1 downloaded (public link, 569 MB) and installed on the local machine (see the session log for the
exact outcome); GPU Trace of the density / force band kernels is the next step on the `exp/band-compact`
branch.
