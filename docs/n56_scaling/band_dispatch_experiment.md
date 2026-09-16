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
