# data/micropolar

Results behind `docs/micropolar_design.md` (sections 8 to 10), copied from the untracked `logs/micropolar/` on
2026-11-11. Produced by `experiment/validation/micropolar_tests.py` on one RTX 5090 (device 0) and one RX 7900 XTX
(device 1). Top-level `data/` did not exist on this branch; the earlier result sets live under
`docs/cluster_v6/data/` (e.g. `e9`).

| path | content |
|---|---|
| `stage1_2/` | T3 operators (json + printed lines), T4 relaxation, the K = 2 per-particle seam check, advection across the cut, restart; `seam_layout_base.log` = the 858-failure reference of `_test_seam_layout.py` on `micropolar-base` |
| `t1/` | T1 / T2 comparisons on sample points: `k1_compare.json` (10 base repeats), `k2_compare.json` / `.log` (40 base repeats) |
| `t5/` | the four cavity runs to t = 40.2 (centreline samples every 5,000 steps + run.json) and their pairwise comparisons |
| `checks/` | checks A (label), B (K = 3 seam), C (factors), E (3-D operators): json + printed lines |
| `checks/d/` | check D: `k1_source` (K = 1 from rest to t = 40.2), `k1_t40.npz` (its final step-boundary state, eleven restart fields, 74,529 particles), the fifteen continuations, `paired.json` / `paired.log` |

Not copied (large, still in `logs/micropolar/` of the worktree `vulkan-demo-v9-micropolar`): the particle dumps of
T1 / T2 (`logs/micropolar/t1/*.npz`, 323 MB) and the full run logs.

Added 2026-11-11 (night):

| path | content |
|---|---|
| `viscous/operators_host.md` / `.json` | the host comparison of the viscous Laplacian discretizations (`micropolar_tests.py operators-host`) |
| `viscous/cross_check.json`, `seam_laplacian_k{2,3}.json`, `t1_k{1,2}_compare.json` | shader against host for both viscous forms; seam checks with `laplacian`; T1 re-run |
| `viscous/cavity_n{250,500}.json`, `viscous/cavity_runs/` | the Re = 1000 cavity with both forms (summaries; per run samples.jsonl, meta.json, steady.json) |
| `checks/a_walls_differ.txt` | check A: simple against adami near the walls |
| `checks/d/k1_{6..10}`, `k2s_{6..10}`, `k1lap_1`, `paired10.*` | check D extended to ten runs per group; the micropolar continuation with `laplacian` |

Not copied: the cavity runs' profile samples, snapshots and checkpoints (`logs/micropolar/cavity_forms/`, 6 runs).

Added 2026-11-11 (Stage 3, Section 12 of the design note):

| path | content |
|---|---|
| `stage3/step_limit.json`, `lattice_spectral_radius.txt` | the measured explicit step limits (bisection probes) and the forward-Euler bound of the lattice symbols |
| `stage3/t1_k{1,2}_compare.json`, `seam_annulus_k{2,3}.json` | T1 after the rotating wall; the per-particle seam check on the annulus |
| `stage3/couette_pilot.json` / `.txt` | the cylinder pilot: every fitted number, profile and per-sector constant (`micropolar_couette.py analyse`) |
| `stage3/couette/<run>/` | per run `run.json` and `windows.npz` = the samples summed over the eight windows of an eighth of the run (`micropolar_couette.py reduce`); `analyse --runs data/micropolar/stage3/couette` reproduces `couette_pilot.txt` |
| `stage3/boundary_layer_operators.txt` | discrete gradient and Laplacian on an exponential field of length l (host) |

Not copied: the raw samples of the cylinder runs (`logs/micropolar/couette/`, 134 to 267 samples per run, 130 MB).

Added 2026-11-11 (Stage 3, Sections 12.8 and 12.9):

| path | content |
|---|---|
| `stage3/cavity_n1000.json` | the n = 1000 baseline: thesis configuration against the Monaghan-Gingold run of the validation series (`viscous_form_campaign.py analyse --resolution 1000 --monaghan-gingold-run ...`) |
| `stage3/cavity_runs/n1000_laplacian_eps1em6/`, `n1000_monaghan_gingold_v7rc3/` | per run samples.jsonl, meta.json, steady.json, windows.jsonl (the second copied from the worktree `vulkan-demo-v7-rc3`) |
| `stage3/cost.json` / `cost.txt` | the cost table: per-trial steps/s, memory, per-kernel intervals of the traced runs, the K = 2 point (`micropolar_cost.py summarise`) |
| `stage3/cost_level_test.txt` | nine further 1M runs of the paper row (idle start / host load / alone) |
| `stage3/cost_telemetry_session2.csv` | RTX 5090 power, SM clock, temperature every 5 s during trials 4 to 6 and the level test |

Not copied: the profile samples of the n = 1000 runs (24 MB each), the bench logs and step traces
(`logs/micropolar/cost/`).

Added 2026-11-12 (Stage 4, Section 13): `stage4/`, with its own `README.md` (concentric resolution study and
sweep, wall sums, the eccentric pilot and the cases on the porous walls, draft figures).

Added 2026-11-12 (Stage 5, Section 14): `stage5/`, with its own `README.md` (the wall rule in its slip form and
the plane film, the exact Stokes benchmark of the eccentric flow, the candidates for the wall and the
density-diffusion term between the cylinders and in the cavity).
