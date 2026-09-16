# Bootstrap defrag · fake-band diagnosis · wall border 4 vs 9 — 2026-09-16/17 (local 2× RTX 5090)

## 1. Bootstrap defrag (committed, no switch)

`ChainOrchestratorV5.bootstrap_all` and `DualGpuOrchestratorV5.bootstrap_all` now run one defrag
(`submit_defrag_and_wait`, the periodic-defrag path) after the step command buffers are recorded and
before the first frame. The generator's initial particle order is not voxel-sorted; until the first
periodic defrag the interior kernels ran on that order.

Validation, 2-D 4M K=2 (`--anatomy`, chain bench 3000 steps, warmup 1000):

| | phase B at f500 | f1000 | f2000 | f3000 | first 1000 frames | 3000 steps total | steady fps | seam / drift / stamps |
|---|---|---|---|---|---|---|---|---|
| before (2 runs) | — | 23.67 ms | 3.51 | 3.54 | 22.2 / 21.2 s | 30.7 / 29.7 s | 233.5 / 235.6 | OK / 0 / 0 |
| after, cadence 500 | 3.51 ms | 3.47 | 3.50 | 3.54 | 4.4 s | 13.2 s | 226.2 | OK (overshoot −0.04 / −0.02 dx, dup 0) / 0 / 0 |
| after, cadence 1000 | — | 3.5 | 3.5 | 3.5 | 4.3 s | 13.1 s | 227.8 | OK / 0 / 0 |
| after, third run (band-cost control) | — | 3.52 | 3.52 | 3.57 | — | — | 230.9 | OK / 0 / 0 |

Phase B is now flat from the first defrag cadence on (the f1000 value used to be 6.7× the steady one:
correction_interior 9.45 → 1.03 ms, density 7.77 → 1.08, force 6.44 → 1.41). The first 1000 frames take
4.4 s instead of 22 s. The steady fps of the post-change runs (226–231) sat 1–3% under the two pre-change
runs (233–236) measured earlier that evening, so an interleaved A/B was run against the pre-change commit
(3d78b62, scratch worktree, same case, same switches): pre-change 226.8 / 226.7 fps, post-change 225.0 / 227.1
fps, and a later window (`--warmup 3000`) 225.9 — **the bootstrap defrag does not change the steady
throughput**; the 233–236 values belong to the earlier GPU state (the phase trace had already shown a
228–236 run-to-run band for identical configurations).

3-D 8M K=1 (`cavity3d_8m`, 10.5M total): first 1000 frames 188.5 s → 83 s, total 361 → 256 s, steady
11.6 → 11.6 fps, drift 0.

Expected saving on the cluster reference runs (3000 steps, steady = last 2000): a run costs
t_first1000 + 2000/fps; the first 1000 frames were 2.2× the steady rate at 64M on N56 (465 s per 64M K=1
reference = 243 s + 222 s). With the sort at bootstrap the first 1000 frames should cost ≈ 111 s, i.e.
≈ 333 s per reference (−28%); Task A's eleven 64M reference sets (86 min) become ≈ 61 min and the 3.24 h
job ≈ 2.6 h. For the 3-D matrix (3d_prep.md) the same ratio applies to every run, not only references.

## 2. Fake-band diagnosis (`V5_FAKE_BAND_TEST=<column>`, default off)

Switch: spec constant 59 `FAKE_BAND_COLUMN` (helpers `in_boundary_band`, `band_voxel_count`,
`band_thread_voxel`, `band_thread_particle`), set only on sims without peers. In a K=1 run the boundary band
is placed at own columns [c, c + range) for the correction / density / force ranges 2 / 3 / 4; the interior
pipelines skip it and the boundary pipelines (band-voxel dispatch, the production Phase C path) process it —
all neighbours in the local pool, no ghosts. Measurement instrument: a new phase-C tick separates the
density band kernel from the scratch→primary copy (`density_boundary_us`, `density_copy_us`). Correctness:
per-particle verifier, 2-D 4M K=1 plain vs fake band, 300 steps: 0 unmatched, every field at the
run-to-run noise level (acceleration max 1.9–2.3 in both noise and test pairs), worst column ratio 1.38 —
PASS (the diagnosis path computes the same physics).

Per-particle cost = kernel time / particles in the band (particles per x column from the initial lattice:
2-D 4M 10,225; 16M 20,115; 32M 28,395; 3-D 8M cube 191,844; stretched 4M/GPU 125,316); interior = phase-B
kernel time / (own − band). Same host stack and switches as the curve jobs (cascade + band dispatch on).

| case | kernel | interior ns/p | band particles | fake band µs | fake ns/p | ratio | real seam band µs (K=2) | real ns/p | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2-D 4M | correction (2 cols) | 0.50 | 20,450 | 82 | 4.01 | 8.0 | 81 | 3.96 | 7.9 |
| 2-D 4M | density (3 cols) | 0.53 | 30,675 | 88 | 2.87 | 5.5 | 88 | 2.87 | 5.5 |
| 2-D 4M | force (4 cols) | 0.70 | 40,900 | 92 | 2.26 | 3.2 | 91 | 2.23 | 3.2 |
| 2-D 16M | correction | 0.48 | 40,230 | 84 | 2.09 | 4.3 | 84 | 2.10 | 4.4 |
| 2-D 16M | density | 0.55 | 60,345 | 93 | 1.54 | 2.8 | 92 | 1.53 | 3.1 |
| 2-D 16M | force | 0.72 | 80,460 | 183 | 2.27 | 3.2 | 180 | 2.24 | 3.3 |
| 2-D 32M | correction | 0.46 | 56,790 | 87 | 1.53 | 3.3 | 88 | 1.54 | 3.4 |
| 2-D 32M | density | 0.49 | 85,185 | 166 | 1.95 | 3.9 | 158 | 1.86 | 3.8 |
| 2-D 32M | force | 0.71 | 113,580 | 196 | 1.73 | 2.4 | 194 | 1.71 | 2.6 |
| 3-D 8M cube / stretched K=2 | correction | 2.55 / 2.51 | 383,688 / 250,632 | 1510 | 3.94 | 1.55 | 968 | 3.86 | 1.54 |
| 3-D | density | 2.58 / 2.62 | 575,532 / 375,948 | 1989 | 3.46 | 1.34 | 1764 | 4.69 | 1.79 |
| 3-D | force | 3.03 / 3.09 | 767,376 / 501,264 | 2894 | 3.77 | 1.24 | 1986 | 3.96 | 1.28 |

Linear fit over the three 2-D sizes, band time = fixed + slope × particles:

| kernel | fake: fixed µs + ns/p | real: fixed µs + ns/p | per-particle ratio to interior (fake / real) |
|---|---|---|---|
| correction | 79 + 0.14 | 77 + 0.18 | 0.28 / 0.37 |
| density | 34 + 1.39 | 39 + 1.26 | 2.7 / 2.5 |
| force | 43 + 1.45 | 42 + 1.44 | 2.0 / 2.1 |

Reading:

- **Fake band = real band.** At equal band sizes the kernel times agree to a few percent in 2-D (all nine
  size × kernel pairs) and to ≤ 10% in 3-D for correction and force; the only kernel where the real seam
  costs measurably more than the fake band is the 3-D density band (4.69 vs 3.46 ns/p, +35%) — the one
  kernel that reads the uploaded ghost density. **The 2× (2-D) / 1.4× (3-D) per-particle cost is not ghost-
  pool locality; it is the band path itself.** Verdict by the task's criterion: fake band ≫ 1.1× interior
  (density 2.5–2.7×, force 2.0×, 3-D 1.24–1.55×) → the cause is in the band code path / dispatch pattern
  (the (voxel, slot) thread mapping with 75% (2-D, 24 of 96 slots) / 50% (3-D, 64 of 128) empty threads, the
  gather through `inside_particle_index`, or the kernel structure). Lane packing (V3.8) already showed that
  simply reducing empty threads does not help; the next candidate is a pid-contiguous band dispatch (band
  particles are contiguous after every defrag) — to be discussed, not changed here.
- **The 2-D correction band's "2×" is entirely fixed cost**: 77–79 µs per launch + 0.14–0.18 ns/particle,
  i.e. per particle it is *cheaper* than the interior kernel. Density and force bands are genuinely 2–2.7×
  per particle plus 35–45 µs fixed.
- The scratch→primary copy (`density_copy`) is ≈ 34 µs per million pool slots (2-D 4M 144 µs, 16M 544,
  32M 1082; 3-D 8M 354): ≈ 0.27 GB/ms effective — far below the 5090's bandwidth, so it is a candidate for
  a compute-shader copy or descriptor ping-pong (≈ 2% of the 64M K=8 frame).

Files: `scripts/band_cost_table.py` (+ spec with the log paths and band counts), logs in the scratchpad
`fakeband/`; verifier report `logs/verify_fakeband_4m_k1/`.

## 3. Wall border 4 vs 9 (3-D)

Code audit — what the wall particles do in the shaders: `predict.comp` returns for `MATERIAL_BOUNDARY`
(static walls); `force.comp` returns before computing anything for a wall particle; `density.comp` computes
a wall particle's density from its *fluid* neighbours only and stores the rest density (ρ₀), the pressure
from the EOS; a fluid particle's density and force sums run over all neighbours within h, walls included,
with V_wall = m/ρ₀. There is no normal computation, no pressure extrapolation, no multi-layer read: **nothing
reads beyond one support radius h = 4 dx**, so wall layers deeper than 4 dx from the fluid never enter any
sum. Border 4 is the minimum that gives every near-wall fluid particle a full wall support.

Cases: `cavity3d_8m_b4` (fluid 8,120,601 + wall 847,124 + lid 161,604 = 9,129,329; border 9: 10,503,459,
−13%) and `cavity3d_weak4_k2_8m_b4` (9,119,703 vs 10,557,873). 3000-step chain benches, warmup 1000,
`--anatomy`, `V5_GHOST_POOL_FACTOR=1.0`:

| run | total particles | steady fps | phase A / B / C (ms) | correction / density / force interior (ms) | device + defrag-scratch VRAM per GPU | ρ range | seam / drift / stamps |
|---|---|---|---|---|---|---|---|
| 8M K=1 border 9 | 10,503,459 | 11.6 | 0.90 / 88.3 / 0.36 | 27.9 / 27.7 / 32.7 | 1814 + 1613 MB | — | — / 0 / 0 |
| 8M K=1 border 4 | 9,129,329 | 12.4 (+7%) | 0.84 / 82.9 / 0.31 | 24.5 / 26.0 / 32.5 | 1574 + 1402 MB (−13%) | — | — / 0 / 0 |
| K=2 stretched border 9 | 10,557,873 | 22.0 | 0.70 / 39.6–41.7 / 4.6–4.7 | 12.4–13.2 / 12.6–13.3 / 14.6–15.3 | 981 + 876 / 1009 + 902 MB | [999.0, 1001.2] | OK / 0 / 0 |
| K=2 stretched border 4 | 9,119,703 | 24.0 (+9%) | 0.66 / 37.0–38.0 / 4.4 | 11.0–11.3 / 11.7–12.0 / 14.3–14.7 | 855 + 763 / 871 + 777 MB | [999.0, 1001.2] | OK / 0 / 0 |

Equivalence (per-particle verifier with `--b-case`, fluid particles only, 2000 steps, two runs per side;
criterion: test-pair differences within the K=1 run-to-run envelope):

| comparison | verifier worst column ratio | centreline u_x(y) profile: noise A–A / B–B vs test A–B (max, m/s) | near-wall fluid (within 2h): |Δu| rms noise vs test | ρ range A / B |
|---|---|---|---|---|
| 8M K=1, border 9 vs 4 | 1.94 → PASS | 0.0097 / 0.0074 vs 0.0026 / 0.0049 | 7.4e-3 / 7.3e-3 vs 7.2e-3 / 7.3e-3 | [998.8, 1001.3] / [998.8, 1001.3] |
| K=2 stretched, border 9 vs 4 | 1.35 → PASS | 0.0045 / 0.0066 vs 0.0014 / 0.0035 | 6.8e-3 / 6.7e-3 vs 7.1e-3 / 6.9e-3 | [998.9, 1001.3] / [999.0, 1001.3] |

The border-4 solution is inside the border-9 run-to-run envelope on every measure (velocity field, wall-adjacent
velocities and densities, density range, centreline profile identical to 4 digits). **Decision: 3-D default
border = 4** (`_demo_cavity_case_3d.py` default changed from 2·hdx+1 to hdx; `gen_3d_families.sh` builds
border 4 by default, `BORDER=9` gives the `_b9` variants). The four existing local cases (1m/2m/4m/8m) keep
border 9.

64M stretched domain at border 4 (`cavity3d_weak8_k8_64m_b4`, generated locally: fluid 65,005,209 + wall
4,333,332 + lid 1,293,636 = **70,632,177**, pool 81.2M slots at safety 1.15, 84.8M at the cluster wrapper's
1.2). Measured on one local 5090 (K=1, `--pool-safety 1.2`, 300 steps): device-local 12,172 MB + defrag scratch
10,845 MB = **23.0 GB** (271 B per pool slot), no host staging, drift 0, **1.7 fps** steady. So the 64M
stretched **K=1 reference fits a 32 GB card with ≈ 9 GB to spare at border 4**; border 9 (76.0M total,
91.2M slots) extrapolates to ≈ 24.7 GB and would fit too. The cost, not the memory, is the constraint: at
1.7 fps a 3000-step K=1 reference takes ≈ 29 min, so the 3-D 64M strong block should measure the K=1
reference on all 8 GPUs at once per trial and reuse the subsets for K=2 / 4 / 8 (≈ 1.5 h of references +
≈ 1.3 h of K runs ≈ 3 h, not the 1 h estimated in 3d_prep.md before the fps was known).

## 4. Commit

(see git log)
