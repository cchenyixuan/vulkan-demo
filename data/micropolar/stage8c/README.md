# Stage 8c: plane flows in the channel periodic in x, and the start-up of Couette flow

Steady plane Couette and Poiseuille flows between two no-spin walls in a channel periodic in x (length 100 dx),
against their exact solutions (eqs. 5.5, 2.24 - 2.26), and the start-up of plane Couette flow (gap 20 l) against a
host solution, as a test of the saturation of the wall-stress excess. Design note: `docs/micropolar_design.md`,
Section 19. Branch `v10-periodic`; no solver change (the periodic domain of Stage 8b with walls on y). Raw runs:
`logs/micropolar/stage8c/` (not in git).

| file | what | made by |
|---|---|---|
| `startup_host.json` | the host solution of the start-up (K = 3, H = 20 l, j = l^2, l^2 / 100, l^2 / 10^4, and the Newtonian fluid mu + kappa / 2): E(Lambda / l) on a grid of Lambda / l from 0.01 to 20, the stress series, the profiles at Lambda = 0.25 l, l, 4 l, both resolutions and their difference, the two limits | `micropolar_startup solve` |
| `channel.txt`, `.json` | every finished channel run: the steady runs against the exact solution (three readings of the wall stress, Q, rms of u and w, micropolar over control, floors) and the start-up runs against the host (E(t) against Lambda / l, floors, the smallest resolved Lambda / l, the profiles) | `micropolar_channel analyse` |

## Runs (`micropolar_channel.py`, table RUNS)

| run | what |
|---|---|
| `couette_m{1,2,5,10}_l10`, `couette_m{2,5}_l20` (and `_control`) | steady plane Couette, m = H / (2 l), l in dx; Re = U H / nu_inf = 10 |
| `poiseuille_m{1,2,5,10}_l10`, `poiseuille_m{2,5}_l20` (and `_control`) | steady plane Poiseuille under a body force; Re = 10 on the Newtonian centreline speed |
| every steady run `_repeat`, `_ma0p01` | the floors: a repeat, half the Mach number |
| `startup_l20`, `startup_l40`, `startup_l80` (and `_control`) | the start-up, H = 20 l, impulsive; Re = U H / nu_inf = 100; l = 20 dx to the steady state, the others to Lambda = H / 2 |
| `startup_l40_j001` (and `_control`) | j = l^2 / 100 at the dt of its angular diffusion (6.96e-7 s, 9.15 million steps) |
| `startup_l20_repeat`, `startup_l20_ma0p01` (and `_control_`) | the floors of the start-up, to Lambda = H / 2 |

## Commands

```
set OPENBLAS_NUM_THREADS=1 & set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
set VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
set V7_DELTA_DENSITY=1
python -m experiment.validation.micropolar_channel cases
python -u -m experiment.validation.micropolar_channel run couette_m2_l20 --device 1
python -m experiment.validation.micropolar_startup solve
python -m experiment.validation.micropolar_channel analyse
python data/micropolar/ch5_draft/make_all.py 27 28
```

The runs were queued with `micropolar_stage6 runner` on `logs/micropolar/stage8c/queue_a/plan.json` (XTX). A run
writes `samples.npz` (the binned profiles u(y), w(y) in rows of dx, the wall fits on both walls, Q, at 60 steps of a
steady run or about 40 per decade of t of a start-up), `states/state_<step>.npz` (the fluid within 2 h + dx of a
wall and the wall particles within h + dx: positions, velocities, densities, pressures, accelerations, the wall's
dummy values, the solver's M^-1; the last five samples of a steady run, every sample of a start-up) and `run.json`.

## Readings taken in this stage

- The start is impulsive (r = 1 for t > 0), in the runs and the host solution: the run loop cannot change a wall's
  velocity during a run without a solver change.
- The wall stress in three readings (fit within 4 dx, wall form, profile; design note 19.1). States saved at a
  defrag step are not used for the wall form (the step returns after the defrag has reordered the particles).
- Re = 10 for the steady flows (it only sets the number of steps), 100 for the start-up; Ma 0.02; the delta
  density storage, as Stage 8b.
