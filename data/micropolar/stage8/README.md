# Stage 8: the dipole-wall collision, Newtonian against micropolar

Normal collision of a dipole with a no-slip wall (Clercx & Bruneau, Computers & Fluids 35 (2006) 245-279), Re = 625
on nu_inf, box [-1, 1]^2, adami walls at rest with w = 0. Design note: `docs/micropolar_design.md`, Section 17.
Branch `v10-showcase`; the solver directories `experiment/v7/shaders` and `experiment/v7/utils` are those of
`thesis-ch5-rc2` (every `run.json` under `logs/micropolar/stage8/` records `git describe` and their tree hashes).

| file | what | made by |
|---|---|---|
| `host_checks.txt`, `.json` | the set-up and the estimators on the host: omega_e for E(0) = 2 (299.528385; the paper's "about 320" gives E = 2.28), Omega(0) = 800, P(0), the largest initial speed, the Poisson pressure against one monopole, the grid estimator on the initial lattice and on a boundary layer of thickness 0.01 W | `micropolar_dipole_analysis checks` |
| `dipole.txt`, `.json` | every finished run: peaks of the enstrophy and the palinstrophy, E / Omega / P at t = 0.25, 0.5, 0.75, the tracks of the primary, secondary and lower primary cores, the wall x = W at the first collision, the leakage coefficient; the runs against the benchmark tables (validation), the micropolar runs and the two floor runs against the control, the shear length | `micropolar_dipole_analysis analyse` |
| `fields/<run>_t<time>.npz` | vorticity (and w - omega / 2) in the window 0.4 <= x <= 1, 0 <= y <= 0.6 at the times of pack item 20, float16 | `micropolar_dipole_analysis fields --runs ... --times ...` |

## Runs (`micropolar_dipole.py`, table RUNS)

| run | dx | fluid | what |
|---|---|---|---|
| `control_n250`, `control_n500`, `control_n1000` | W / 250, W / 500, W / 1000 | 0.25 M, 1 M, 4 M | the Newtonian control (viscosity nu_inf, micropolar path with nu_t = -nu, xi = 0) at three resolutions |
| `control_n500_zero_pressure` | W / 500 | 1 M | the same from the uniform rest density instead of the incompressible pressure |
| `control_n1000_shifted` | W / 1000 | 4 M | the dipole moved by dx / 2 along x (lattice shifted against the flow): noise floor |
| `control_n1000_half_mach` | W / 1000 | 4 M | speed of sound 1200 m/s instead of 600 (Ma 0.009 on the largest initial speed) |
| `mp_l100_n1000`, `mp_l50_n1000` | W / 1000 | 4 M | micropolar, K = 3, l = W / 100 (10 dx) and W / 50 (20 dx), Theta = l^2 |

## Commands

```
set OPENBLAS_NUM_THREADS=1 & set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
set VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
python -m experiment.validation.micropolar_dipole cases
python -u -m experiment.validation.micropolar_dipole run control_n1000 --device-map 0 --particle-states 0.375 1.0
python -m experiment.validation.micropolar_dipole_analysis checks
python -m experiment.validation.micropolar_dipole_analysis analyse
python -m experiment.validation.micropolar_dipole_analysis fields --runs control_n1000 mp_l100_n1000 mp_l50_n1000 --times 0.35 0.45 0.6 0.7
python data/micropolar/ch5_draft/make_all.py 20 21 22
```

The runs were queued with `micropolar_stage6 runner` on `logs/micropolar/stage8/queue0/plan.json` (5090) and
`queue1/plan.json` (XTX). A run writes `start.json`, `heartbeat.json`, `samples.npz` (every sample: energy,
enstrophy and palinstrophy on a grid of one spacing, vorticity extrema of the upper half, the wall x = W from the
particle fit and from the grid, a band of 48 grid nodes next to x = W, leakage sums), `fields/fields_<step>.npz`
(the vorticity on the grid every 0.025, float16) and `run.json`.

## Readings taken in this stage

- omega_e = 299.528385375226 from E(0) = 2 (the reference values of the paper at t = 0 are then met exactly).
- Speed of sound 600 m/s: Ma = 0.0184 on the largest initial speed 11.02 m/s (dt = 1.25e-6 s, acoustic limit;
  the diffusive limits are 3.2e-4 and 3.5e-4 s). While the dipole forms the largest speed reaches 12.2 m/s.
- The control on the micropolar path with nu_t = -nu (c = 0 exactly), xi = 0, Theta = (W / 50)^2: its w is a
  passive constant; the micropolar runs start from w = omega / 2.
- The initial density from the pressure of the incompressible flow (`control_n500_zero_pressure` shows that it
  does not matter: the enstrophy peaks move by 0.03 % and 0.16 %).
- The noise floor from the dipole moved by dx / 2 along x (a lattice shifted in both directions would move the
  wall planes); and from the control at half the Mach number.
- Palinstrophy on the grid of one spacing carries the grid-scale noise of the particle velocities (at t = 0.25
  0.8e5 at n = 250 to 1.8e5 at n = 1000 above the benchmark's 3.99e5, growing with resolution); quoted, not used
  for the verdict.
