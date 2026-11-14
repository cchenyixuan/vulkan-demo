# Stage 8b: periodic boundaries, Taylor-Green and decaying turbulence

The periodic domain of `origin/v8-tgv` (`experiment/v8`) ported into `experiment/v7` (one card, one slab, 2-D),
checked with the switch off (T0, T1), against v8's Newtonian Taylor-Green numbers (port check), then the
micropolar Taylor-Green vortex against its exact two-mode solution and decaying 2-D turbulence, Newtonian against
micropolar. Design note: `docs/micropolar_design.md`, Section 18. Branch `v10-periodic`; the port is commit
78fb290 (thesis-ch5-rc2-50-g78fb290). Raw runs: `logs/micropolar/stage8b/` (not in git).

| file | what | made by |
|---|---|---|
| `t0_seam_layout.txt` | `experiment/v7/_test_seam_layout.py` on the base tree (thesis-ch5-rc2 solver) and on the new tree: the failures only in one of them | scratch script, Section 18.0 |
| `t1/compare_newtonian.*`, `t1/compare_micropolar.txt`, `t1/compare_micropolar_15.*` | T1: dumps after 3000 steps with the periodic switch off, new tree against the spread of the base tree's own repeats (Newtonian cavity 5 against 5; micropolar cavity 5 against 5, then 15 against 15) | `micropolar_tests dump` / `compare` |
| `taylor_green.txt`, `.json` | the port check (v8's set, nu_eff / nu against Section 12.1) and every micropolar Taylor-Green run against the exact two-mode solution and against its control, with the noise floors | `micropolar_periodic analyse` |
| `turbulence.txt`, `.json` | the turbulence runs: E(t), Z(t), spectra at t = 1, 2, 3, 4 (and 0.5), the spectrum ratio micropolar / Newtonian by band with the shifted-lattice floor, the 1025^2 resolution check and the linear estimates | `micropolar_turbulence analyse` |
| `fields/<run>_t<time>.npz` | vorticity of the two 2050^2 runs at t = 0, 0.5, 1, 2, 3, 4 on a 512^2 grid (2 x 2 means of the 1024^2 deposit), float16 | `micropolar_turbulence analyse` |

## Runs

| run | particles | what |
|---|---|---|
| `check_n400_ma0p1`, `_ma0p05`, `_ma0p02`, `_ma0p01`, `check_n800_ma0p1` | 160,000 / 640,000 | port check: v8's Newtonian Taylor-Green (box [0, 1)^2, Re 100, eps^2 1e-4 h^2) |
| `tgv_control_n400`, `tgv_control_n800` | 160,000 / 640,000 | Newtonian control, nu_inf = 0.05 (Re 20, k = 1, box [0, 2 pi)^2), micropolar path with c = 0 |
| `tgv_kl025_n400`, `tgv_kl05_n400`, `tgv_kl1_n400`, `tgv_kl2_n400`, `tgv_kl05_n800` | | micropolar, K = 3, \|k\| l = 0.25, 0.5, 1, 2, j = l^2, W(0) = k U(0) |
| `tgv_control_n400_repeat`, `tgv_kl1_n400_repeat` | | repeats (floor) |
| `tgv_control_n400_ma0p01`, `tgv_kl1_n400_ma0p01` | | Ma 0.01 (floor) |
| `tgv_control_n400_stored_density` | | the control without the delta density storage |
| `tgv_kl1_inertia_n400` | | \|k\| l = 1, j = 0.7549 l^2 (fast rate 5 x slow), W(0) = 0 |
| `tgv_kl1_j150_n400` | | \|k\| l = 1, j = 1.5 l^2: w = omega / 2 exact, ratio 1 |
| `tgv_kl05_j001_n400`, `tgv_kl1_j001_n400`, `tgv_control_n400_small_dt` | | j = l^2 / 100 (quasi-steady limit) with their control, at their dt (2.75e-5) |
| `turbulence_newtonian_n2050`, `turbulence_micropolar_n2050` | 4,202,500 | decaying turbulence, nu_inf = 1e-3, k0 = 10, u' = 1; micropolar K = 3, l = 1/7, j = l^2 |
| `turbulence_newtonian_n2050_shifted` | 4,202,500 | the lattice shifted by half a spacing (floor) |
| `turbulence_newtonian_n1025` | 1,050,625 | resolution |

## Commands

```
set OPENBLAS_NUM_THREADS=1 & set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
set VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
set V7_DELTA_DENSITY=1
python -m experiment.validation.micropolar_periodic cases
python -u -m experiment.validation.micropolar_periodic run tgv_kl1_n400 --device 1
python -m experiment.validation.micropolar_periodic analyse
python -m experiment.validation.micropolar_turbulence cases
python -u -m experiment.validation.micropolar_turbulence run turbulence_micropolar_n2050 --device 0
python -m experiment.validation.micropolar_turbulence analyse
python data/micropolar/ch5_draft/make_all.py 23 24 25
```

Every run except `tgv_control_n400_stored_density` ran with `V7_DELTA_DENSITY=1` (the density stored as an offset
from rho0, as v8-tgv's set). The runs were queued with `micropolar_stage6 runner` on
`logs/micropolar/stage8b/queue_check`, `queue_tgv` (XTX) and `queue_turbulence0` (5090).

## Readings taken in this stage

- Box sizes multiples of 5 particles across (the voxel of h = 5 dx tiles the periodic box): 2050 and 1025 for the
  task's 2048 and 1024.
- Taylor-Green at Re = 20 on nu_inf (50 on mu), Ma 0.02; fit window 5 <= t <= 20 for the run and for the exact
  solution sampled at the same times.
- The micro-inertia run needs j = 0.7549 l^2 for fast / slow = 5 (j = l^2 gives 4.06).
- Four runs beyond the task, to test the exact analysis: j = 1.5 l^2 (neutral), j = l^2 / 100 at |k| l = 0.5
  and 1, and their control at their dt.
- The density samples of the run tools carry rho0 twice under the delta storage (a diagnostic; the turbulence
  analysis subtracts it, the Taylor-Green analysis does not use it).
