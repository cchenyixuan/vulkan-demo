| 提交 | 标题 |
|---|---|
| `052cfb9` | wall_boundary crossing (stage 1 implementation): crossing-fraction walls of an axis-aligned box, one GPU, fused pass |
| `029313a` | wall_boundary crossing: run tools (cavity launcher, kernel timing, pipeline statistics, logged runs) |
| `5ee25b1` | wall_boundary crossing: validation tools (tables, hydrostatic box, bit-identity, pipeline recipes) + hydrostatic cases |
| `a2149ed` | wall_boundary crossing: slabs with a peer (stage 3 enablement), pipelined drivers, 3-D cavity cases |
| `465d328` | wall_boundary crossing: K = 2 comparison tools (bit comparison + voxel-list probe, seam audit driver), table diagnostics |
| `fac95ac` | wall_boundary crossing: tilted 3-D hydrostatic cases get 64 incoming slots (a lattice in uniform free fall enters a voxel through three faces in one step: 48 arrivals overflowed 32 slots and predict killed the excess); hydrostatic.py names the overflow in a failure |
| `dba0f27` | wall_boundary crossing: single_step judges acceleration and shift against their summed pair terms |
| `ae5f41e` | wall_boundary crossing: report tools (figures, tables with run records, report assembly), outcome classes for longer bit windows |
| `96c9da1` | wall_boundary crossing: K = 2 throughput tool, run records in the cavity tables, a stale file name in a docstring |
| `0e38558` | wall_boundary crossing: report tables (3-D cavity comparison, combined seam audit table, commits, copies of tool summaries and run records) |
| `6a1bada` | wall_boundary crossing: cavity tables list the repeat run and name the convergence quantities without pipe characters |
| `81ae0d9` | wall_boundary crossing: diagnostic switch numerics.wall_crossing.delta_wall (extension / rest_density) |
| `d10e144` | wall_boundary crossing: delta_wall mechanism test (tables, figure) and the report draft with its section 6 |
| `d62e005` | wall_boundary crossing: diagnostic delta_wall gains body_force (wall pair term rho_i g . (r_j - r_i) / c^2) |
| `21b700a` | wall_boundary crossing: hydrostatic.py records the mean density, the pressure level and the level-free pressure error at every sample |
| `e24d16e` | wall_boundary crossing: delta_wall_experiment.py gains the body_force run and the hydrostatic comparison (extension against body_force) |
| `e643008` | wall_boundary crossing: second delta_wall test (body_force) - tables, figures and section 7 of the report |
| `1932d2b` | wall_boundary crossing: density budget diagnostic (V7_DENSITY_BUDGET=1) and diagnostic continuity_wall_component |
| `7b443b7` | wall_boundary crossing: the density budget also records what the stored float32 density actually changed by |
| `0dd6338` | wall_boundary crossing: the density budget in a neighbour traversal of its own for both builds, twelve regions |
| `f25c696` | wall_boundary crossing: density budget tables (repeats, outcome classes), density_rounding.py, the two controls in the drift table |
| `2ed2d09` | wall_boundary crossing: regenerated drift table with the two controls (interim, before the report section) |
| `2351a4d` | wall_boundary crossing: density budget of the 250^2 cavity (tables, figures, section 8 of the report) |
| `44e4e6f` | wall_boundary crossing: continuity_wall_component defaults to normal (was full) |
| `6c8a55a` | wall_crossing reference: simple / adami keep the whole wall velocity difference in the budget comparison |
| `bb7e173` | wall_boundary crossing: analysis of the default change (normal_default.py), earlier tables relabelled |
| `048be28` | wall_boundary crossing: section 9 of the report, first half (default change, regression checks, short runs) |
| `19a0b90` | wall_boundary crossing: section 9 of the report complete (cavity statistics at 250^2 and 500^2, long-run histories) |
| `398f70a` | wall_boundary crossing: report rebuilt (commit table) |
| `b10c5ea` | wall_crossing tools: cavity_run --initial-density-offset (a run can start from a uniformly shifted fluid density; control experiment for the pressure branch) |
| `db3d609` | wall_boundary crossing: second-order continuation of the velocity (switches, both linear by default) |
| `263381b` | wall_crossing tools: numpy exactness study of the second-order continuation (quadratic_accuracy.py); the cavity runner hook reads the second-order states every 20,000 steps (a readback costs 20 / 75 ms at 250^2 / 500^2); single_step: floored criterion for the budget build's diagnostics |
| `a352556` | wall_crossing tools: section-10 tables (quadratic_extension.py tables), cavity table modes of the second-order runs and the control, normal_default.cavity parameterised |
| `449bd10` | wall_crossing tools: single_step --determinant-threshold (a test that puts half of the particles on the fallback of the second-order continuation); wall-layer distance statistics, figure fixes |
| `6296d80` | wall_crossing tools: cavity table label of the second pressure-branch control |
| `a70b805` | wall_boundary crossing: section 10 of the report (second-order continuation of the velocity: regression, numpy exactness, single step, screening, hydrostatic boxes, cavity at 250^2 and 500^2, pressure-branch controls, wall-layer distances, cost, K = 2) |
| `b18597b` | wall_boundary crossing: report rebuilt (commit table) |
| `1edf4ee` | wall_boundary crossing: section 11 of the report (old default minus first order: time-mean velocity and pressure fields of the stored 250^2 snapshots, shares by region, the dropped tangential term; no new run) |
| `282f598` | wall_boundary crossing: report rebuilt (commit table) |
| `bd9fec1` | wall_boundary crossing: pairwise geometry and reflection of pressure and density at the wall (switches, defaults unchanged) |
| `8748bcd` | wall_crossing tools: the three cases that locate a failed screening of the reflection configuration (pairwise geometry alone, with the pressure reflection only, with the density reflection only); numpy check of the pairwise geometry |
| `89441e9` | wall_boundary crossing: section 12 of the report (pairwise geometry and reflection of pressure and density at the wall: regression, numpy check, single step, hydrostatic boxes, the cavity screening and the three locating runs; the screening fails on the interior continuity term with the pairwise geometry alone, no run to t = 100) |
| `8075b88` | wall_boundary crossing: report rebuilt (commit table) |
| `a414a93` | wall_crossing tools: the case of the pairwise geometry with the whole vector in the continuity equation (pairwise_full); cavity_watch.py (stops a cavity run when the fluid leaves the box or exceeds a speed limit) |
| `2b2e3e2` | wall_boundary crossing: section 13 of the report (the reflection configuration and pairwise geometry + whole vector run to t = 100 at 250^2: cavity metrics beside the existing rows, histories of density, pressure and pressure noise, the interior continuity term, kinetic energy and the centre-line estimate of psi_min at t = 18, first-layer distances; section 12's sample-interval note corrected) |
| `8f2902d` | wall_boundary crossing: report rebuilt (commit table) |
| `3176d57` | wall_crossing tools: the case of the complete scheme (pairwise geometry, whole vector in the continuity equation, pressure and density reflection: reflection_full) and its table modes |
| `c9b17ff` | wall_boundary crossing: section 14 of the report (the complete scheme, pairwise geometry + whole vector + both reflections, run to t = 100 at 250^2 from rho0 and from rho0 + 0.3: cavity metrics beside the existing rows, mean density at fixed instants and its rate, mean pressure, negative share, fluctuation; the old default's pressure fluctuation from its snapshots) |
| `2186032` | wall_boundary crossing: report rebuilt (commit table) |
| `b0a2791` | volume constraint for closed domains (numerics.volume_constraint, off by default, wall_boundary crossing only) |
| `5f27286` | volume constraint: the log path's environment variable is SPH_VOLUME_CONSTRAINT_LOG (the cavity runner refuses any V7_* variable outside the release set of switches) |
| `e09cd4e` | wall_boundary crossing: section 15 of the report, first version (the volume constraint: regression, single step, short run, the 250^2 cavity to t = 100, hydrostatic boxes; the 500^2 run is still going) |
| `501a56e` | wall_boundary crossing: section 15 of the report completed (the 500^2 cavity with the complete scheme and the volume constraint: metrics beside the old default and adami, two-point order, histories, time per step) |
| `9435f56` | wall_boundary crossing: report rebuilt (commit table) |
| `cfb8cbf` | wall_boundary crossing: section 15, the range of the 500^2 mean density corrected (rho0 + 0.001 to + 0.004) |
| `e4e86af` | wall_boundary crossing: section 16 of the report (the post-processing behind psi_min, the velocity extrema and the vortex centre: how it works, a test on an analytic divergence-free field at 250^2 and 500^2, psi_min of eight runs by four integration paths, the vortex centres; stored snapshots only) and its tool stream_function_check.py |
| `088e967` | wall_boundary crossing: report rebuilt (commit table) |
| `faebf7a` | wall_crossing tools: snapshots in extra time windows (cavity_run.py --extra-snapshot-windows, cavity_runner_entry.py item 5: written from the runner's own sample reads to <run>/snapshots_extra), the record names t_end; table mode of the 250^2 complete scheme + constraint continued to t = 200 |

（加入本报告的那个提交排在这些之后。）
