# Captions of the chapter-draft pack

Written by the item scripts (`style.record`); regenerate with `make_all.py`. Numbers are quoted with the digits the noise floor of each quantity supports.

## 01. Concentric cylinders: profiles against the exact solution

Status: provisional. Files: `fig_01_concentric_profiles.pdf`, `fig_01_concentric_profiles.png`.

Circular Couette flow of the micropolar fluid between concentric cylinders (R1 = 4 g, R2 = 5 g, rotor at Omega, w = 0 on both walls): bin means of the azimuthal velocity (a, b) and of the micro-rotation (c, d) over the last quarter of the run (bins of one dx, about forty shown; every bin in the insets, which cover the 0.08 g next to the rotor and to the stator) against the exact solution (line). (a, c) K = 3, l = 25 dx, m = 2, g = 100 dx, Ma = 0.02, Re = 10 on mu_inf (25 on mu), run `g100_k3_m2`; (b, d) K = 3, l = 10 dx, m = 10, g = 200 dx, Ma = 0.02, Re = 10 on mu_inf (25 on mu), run `g200_k3_m10`. Tag `thesis-ch5-rc1`. Rms deviation from the exact profile, relative to the rms of the profile: m = 2: v 6.2e-05 in the core and 7.5e-04 within h of a wall, w 2.9e-03 and 5.6e-03; m = 10: v 1.7e-04 in the core and 3.4e-04 within h of a wall, w 2.0e-03 and 9.3e-03.

## 02. Concentric cylinders: convergence of D and of the profiles

Status: provisional. Files: `fig_02_concentric_convergence.pdf`, `fig_02_concentric_convergence.png`.

Convergence between concentric cylinders, K = 3, w = 0 on both walls, in the micropolar length resolved (l / dx = 5, 10, 20; h = 5 dx; Ma = 0.02, Re = 10 on mu_inf, 25 on mu). (a) Deviation of the relative torque change D (velocity fit, against the run's Newtonian control) from the exact value; horizontal bars: the noise floor of D (spread of the control's beta over three lattices); dotted: power law through the points above their floor, order 2.2. (b), (c) Rms deviation of the bin means of v_theta and of w from the exact profiles, relative to the rms of the profile, further than h from both walls (circles) and within h of a wall (squares). Filled symbols, solid lines: m = 5 (runs `g050_k3_m5`, `g100_k3_m5`, `g200_k3_m5`); open symbols, dashed lines: m = 10 (`g100_k3_m10`, `g200_k3_m10`). Tag `thesis-ch5-rc1`. Deviation of D: m = 5: +2.98 %, +0.65 %, +0.04 %; m = 10: +3.03 %, +0.62 %. Not in the figure yet: `g400_k3_m10`.

## 03. Concentric cylinders: D against m for three K, with the slip-form rule

Status: provisional. Files: `fig_03_concentric_sweep.pdf`, `fig_03_concentric_sweep.png`.

Relative change D of the torque between concentric cylinders (R1 = 4 g, R2 = 5 g, w = 0 on both walls) against m = g / (2 l) for K = 3, 1 and 0.5 (Re = 10 on mu_inf; 25, 15 and 12.5 on mu; Ma = 0.02). (a) Exact solution (solid), the wall rule in its slip form (dashed; every wall displaced by delta = K l / (2 (1 + K)) into the fluid, the rotor's apparent speed reduced by 2 delta Omega) and the runs (points; bars: noise floor of D from three lattices, smaller than the symbols). (b) The same as ratios to the exact value; the rule leaves the frame below m of about 1.9 (1.36 to 1.49 at m = 1). Resolution of the rows (m, l / dx, g / dx): (1, 50, 100), (2, 25, 100), (5, 10, 100), (10, 10, 200), (20, 5, 200). Runs: the sweep rows of tab_04; tag `thesis-ch5-rc1`, `thesis-ch5-rc1-3-g645dc15`, `thesis-ch5-rc1-1-ga07f93d`, `thesis-ch5-rc1-2-gad23a67`. Ratio to exact: m = 1: runs 0.997 to 1.000, rule 1.365 to 1.487; m = 2: runs 0.999 to 1.001, rule 1.035 to 1.039; m = 5: runs 1.004 to 1.006, rule 0.997 to 0.997; m = 10: runs 1.004 to 1.006, rule 0.999 to 0.999; m = 20: runs 1.021 to 1.030, rule 0.999 to 0.999.

## 04. Table of every concentric row (and of the controls)

Status: provisional. Files: `tab_04_concentric_rows.csv`, `tab_04_concentric_rows.tex`, `tab_04b_concentric_controls.csv`, `tab_04b_concentric_controls.tex`.

Every micropolar row between concentric cylinders (R1 = 4 g, R2 = 5 g, U = Omega R1, w = 0 on both walls, Ma = 0.02, end time 2 g^2 / nu_inf, last quarter averaged). D_SPH: the potential-vortex constant of the velocity fit over that of the run's Newtonian control, minus one; D_exact and D_rule (the rule in its slip form) with the hydrodynamic radii of the control; noise floor: spread of the control's beta over three lattices, over D_exact; steady: relative change of the fitted constant between the last two eighths of the run; drifts: change of the fluid's mean density (over rho0) and mean pressure (over rho U^2) between the last two eighths ('-' where the run did not record them). The second table lists the controls. 18 rows; D_SPH / D_exact by l / dx: l = 5 dx: 1.0214 to 1.0303; l = 10 dx: 1.0038 to 1.0065; l = 20 dx: 1.0004 to 1.0004; l = 25 dx: 0.9991 to 1.0009; l = 50 dx: 0.9970 to 1.0000. Largest departure: `g100_k3_m10`, 1.0303. Tag `thesis-ch5-rc1`, `thesis-ch5-rc1-1-ga07f93d`, `thesis-ch5-rc1-3-g645dc15`, `thesis-ch5-rc1-2-gad23a67`.
