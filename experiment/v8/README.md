# experiment/v8 — the v7 kernels with a periodic domain, and the Taylor–Green vortex

v8 = experiment/v7_single's host side (`SphSimulatorSingle`, one GPU, one compute queue, the fence ring) on a
copy of experiment/v7's kernels that knows a doubly periodic box. experiment/v7 and experiment/v7_single are
not modified. The cavity cases keep their results: with `physics.periodic` absent the four new specialization
constants take their defaults, the periodic branches fold away, and `_test_v8_equivalence.py` shows the
per-particle state bit-identical to v7_single's after 999 steps.

## What changed in the kernels (`shaders/`, compiled by `compile_shaders_v8.py` into `shaders/spv` + MANIFEST)

| file | change |
|---|---|
| `common.glsl` | spec constants 116 `PERIODIC_X`, 117 `PERIODIC_Y` (bool, default off), 118 / 119 `DOMAIN_LENGTH_X/Y` |
| `helpers.glsl` | `wrap_neighbor_coord` (the 3^d sweep's neighbour voxel wrapped on a periodic axis), `minimum_image` (separation `r_i - r_j` folded to the nearest image; exactly antisymmetric), `wrap_position` (a drifted position folded into `[origin, origin + L)`), `clamp_periodic_coord` |
| `correction_density.comp`, `correction.comp`, `density.comp`, `force.comp`, `wall_extrapolate.comp` | the neighbour loop takes `wrap_neighbor_coord` instead of `in_own_grid` and `minimum_image(self - neighbour)` instead of the raw difference |
| `predict.comp` | the drifted position is wrapped before its voxel is computed (a periodic axis never kills a particle) |
| `initialize_voxelization.comp` | the voxel coordinate is clamped on periodic axes |

Requirements the loader asserts: on a periodic axis the frame.obj span is an integer number (>= 3) of voxels
(`span = GRID_DIMENSION * h`), so the 3 neighbour columns of a sweep are distinct and no pair is within h both
directly and through an image. `deep_wall_marker.comp` / `deep_wall_skip.glsl` do not wrap (walls only; the
simulator refuses a periodic case with V7_DEEP_WALL_SKIP on).

## Host side (`utils/`)

| file | role |
|---|---|
| `case_loader_v8.py` | v7's loader, then `physics.periodic`, the periodic grid, and `initial_field: taylor_green` (per-particle velocity, density from the analytic pressure through the inverted Tait EOS) |
| `simulator_v8.py` | `SphSimulatorV8(SphSimulatorSingle)`: v8's SPIR-V, the four spec constants, the per-particle initial density |
| `taylor_green.py` | the analytic solution (velocity, pressure, vorticity, kinetic energy, decay factor) |

## Scripts

| script | purpose |
|---|---|
| `_make_tgv_case.py` | writes a TGV case (lattice, box, material of the Reynolds number, case.yaml) |
| `_run_tgv.py` | one run sampled against the analytic solution (samples.jsonl, snapshots, result.json) |
| `_run_tgv_campaign.py` | the campaign (RUNS table): resolution series, Re 1000, ablations, Mach, h/dx, repeats |
| `_analyze_tgv.py` | tables, convergence orders, effective viscosity, figures |
| `_test_v8_periodic.py` | GPU kernel sum = CPU minimum-image sum, no edge deficit, crossings, drift 0 |
| `_test_v8_equivalence.py` | cavity: v7_single vs v8 bit for bit (canonical lists, 999 steps) |

Report: `docs/tgv/v8_tgv.md`.
