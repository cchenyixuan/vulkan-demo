# Stage 6 data (docs/micropolar_design.md, Section 15)

Runs on the tag `thesis-ch5-rc2` (cylinder cases with `numerics.delta_fluid_only: true`, cavity cases with the
default walls), the host calculations of the stage, and the reduced files the chapter-draft pack
(`data/micropolar/ch5_draft/`) reads. Raw runs are under `logs/micropolar/stage6*` and are not in the
repository. Commands are run from the repository root with `python -m experiment.validation.<module> ...`.

| file | what | made by |
|---|---|---|
| `concentric.json`, `.txt` | every finished concentric run: fits, D against the exact solution, noise floors, orders, the rule in its slip form, levels of density and pressure and their drift | `micropolar_stage6 concentric` |
| `couette/<run>/` | the samples of each concentric run summed over the eight eighths of the run, with its `run.json` | the same command (`micropolar_couette reduce`) |
| `eccentric.json`, `.txt` | every finished eccentric run: load, direction, torque by both readings against the exact Stokes values, wall pressure by angle, levels; each micropolar run against its control with the two predictions of the host; noise floors; the same flow at two resolutions | `micropolar_stage6 eccentric --json data/micropolar/stage6/eccentric.json` |
| `eccentric_stage5_runs.json`, `.txt` | the same analysis of the runs of Stages 4 and 5 without the density diffusion term (what the provisional pack items were made from; the Newtonian benchmark rows at Reynolds number 1 exist only here) | `micropolar_stage6 eccentric --set stage5 --json ...` |
| `fields/<run>.npz`, `.json` | the whole fluid of one eccentric run at its last step with the curl of the velocity | `micropolar_stage6 fields <run>` (`--set stage5` for a run of Stage 4) |
| `fields/cavity_<run>.npz`, `.json` | the fluid of a cavity run at its newest checkpoint with the curl of the velocity | `micropolar_stage6_cavity fields <run>` |
| `cavity.json`, `.txt` | the micropolar cavity and its two Newtonian bounds: centre lines, extrema, levels, comparisons | `micropolar_stage6_cavity analyse` |
| `cavity_reference.json`, `.txt`, `cavity_reference_<case>.npz` | the cavity cases by finite differences on the host (stream function, vorticity, microrotation; grids of 128, 192, 256 intervals): extrema per grid and extrapolated, place between the two Newtonian bounds, the particle runs against it; fields of the finest grid | `micropolar_cavity_reference run --cases newtonian_far newtonian_wall micropolar micropolar_l40 micropolar_no_inertia micropolar_free_spin micropolar_l10` |
| `film_check.txt`, `.json` | the plane film: closed forms against the four-constant solution; the sharper form of the rule | `micropolar_film check --out ...` |
| `film_flux.txt`, `.json` | the wedge term dQ/dh of the exact concentric solution against the plane film with and without the slip term | `micropolar_film flux --out ...` |
| `film_bearing.txt`, `.json` | thin-film bearing with the film of the model: load and torque changes by variant, and the rule on the exact Stokes solution | `micropolar_film bearing --out ...` |
| `rotor3d.json`, `.txt` | the 3-D runs: settling, levels, mid-plane against the 2-D solution, reach of the ends, wall layers on the end face, torque by both readings, pairs, chain against one slab, speed and memory | `micropolar_rotor3d_torque sums <runs>` (writes `torque.json` beside each raw run), then `micropolar_rotor3d_analysis analyse` |
| `rotor3d/<run>.npz`, `.json` | azimuthal means on the (r, z) grid over the last quarter and the fluid within 1.5 dx of the plane y = 0 at the last step with w - curl v / 2 | `micropolar_rotor3d_analysis maps <run>` |
| `rotor3d/g40_k3_m2_k2.npz`, `.json` | the run at g = 40 dx on two slabs was stopped at step 38,000 of 51,000 before any particle state was saved: azimuthal means on the (r, z) grid of one spacing over its samples of steps 26,000 to 35,000, with w - curl v / 2 from finite differences of the means; `rotor3d.json` carries its analysis under `partial_runs`; what the run left (`samples_partial.npz`, `heartbeat.json`, its log) is kept in `rotor3d_stopped/g40_k3_m2_k2/` | `micropolar_rotor3d_analysis analyse` and `maps g40_k3_m2_k2` (they read `rotor3d_stopped/` as long as the run has no `run.json`) |
| `cavity_reference_l160_incomplete.log` | log of the finite-difference case l = L / 160, stopped on its finest grid (grids of 256 and 384 intervals steady, 512 at t = 25) | `micropolar_cavity_reference run --cases micropolar_l160 --resolutions 256 384 512 --verbose` |
| `seam3d_k3.json`, `seam3d_k2.json` | per-particle seam check of the 3-D micropolar case with the key: three slabs on one card, two slabs on XTX + 5090 | `micropolar_tests seam --case cases/rotor3d/seam_g20_k3_m1/case.yaml --walls adami --steps 0 20 --device-map 0,0,0 (1,0) --pressure --json ...` |

The run table of the cylinder cases: `micropolar_stage6 list`. The queue: `micropolar_stage6 runner` with the plan
`logs/micropolar/stage6/plan.json` (`micropolar_stage6 status` prints it with the state of every job).

## To take the queue up again

The queue ended at 02:50 on 2026-11-13, when the machine ran out of memory and every background job of the
session was stopped (design note 15.6). Nothing was started again. `logs/micropolar/stage6/plan.json` (not in the
repository) still lists, in this order: the micropolar 3-D run at g = 40 dx on two slabs (from its beginning),
its Newtonian control on two slabs, the micropolar run on one slab, the cavity run with l = L / 40, and the
second run of `g200_k1_m5`; `plan_held_g400.json` beside it holds two of the g = 400 dx jobs.

```
set OPENBLAS_NUM_THREADS=1 & set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
set VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
python -u -m experiment.validation.micropolar_stage6 runner --disk-reserve 2
python -m experiment.validation.micropolar_stage6 status
```

A file `STOP` beside the plan ends the runner after the running jobs, `KILL` at once. After the runs, per run
of the 3-D case: `micropolar_rotor3d_torque sums <run>`, then `micropolar_rotor3d_analysis analyse`,
`maps <run>`, `sizing`, and `data/micropolar/ch5_draft/make_all.py 15` (a finished run of the same name takes the place of the stopped one in every table); after the cavity run
`micropolar_stage6_cavity analyse`, `micropolar_cavity_reference run --cases newtonian_far` (the comparison
table) and `make_all.py 11`; after `g200_k1_m5`: `micropolar_stage6 concentric`, then
`micropolar_stage6 eccentric --json data/micropolar/stage6/eccentric.json` and `make_all.py 03 04 08 09`.
