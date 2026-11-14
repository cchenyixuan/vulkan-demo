# Chapter 5: what the solver has shown

Written 2026-11-14 17:30 at commit `36d6f76` (branch `v10-periodic`; solver tag `thesis-ch5-rc2`, nothing under `experiment/v7/shaders` or `experiment/v7/utils` changed since). Every number below is read from the data files by `report.py`; the captions with all parameters and run names are in `captions.md`, the full account in `docs/micropolar_design.md`, Sections 15 (Stage 6), 16 (Stage 7) and 17 (Stage 8, the dipole-wall collision).

**State.** Every item is final except the one row at g = 400 dx (m = 20, K = 3, l = 10 dx): its Newtonian control is done, the micropolar run is going; items 02 to 04 gain that row when it ends. Nothing else is pending and there is no further queue.

Marks: **[run]** measured in a particle run on the tag; **[host]** computed on the host without the particle solver; **[inferred]** my reading, not measured.

## 1. Index

| no. | item | files | status |
|---|---|---|---|
| 01 | Concentric cylinders: profiles against the exact solution | `fig_01_concentric_profiles.pdf` | final (rc2 runs) for g <= 200 dx; the rows at g = 400 dx are not reached |
| 02 | Concentric cylinders: convergence of D and of the profiles | `fig_02_concentric_convergence.pdf` | final (rc2 runs) for g <= 200 dx; the rows at g = 400 dx are not reached |
| 03 | Concentric cylinders: D against m for three K, with the slip-form rule | `fig_03_concentric_sweep.pdf` | final (rc2 runs) for g <= 200 dx; the rows at g = 400 dx are not reached |
| 04 | Table of every concentric row (and of the controls) | `tab_04_concentric_rows.csv`, `tab_04_concentric_rows.tex`, `tab_04b_concentric_controls.csv`, `tab_04b_concentric_controls.tex` | final (rc2 runs) for g <= 200 dx; the rows at g = 400 dx are not reached |
| 05 | Table: Newtonian eccentric flow against the exact Stokes values | `tab_05_eccentric_newtonian.csv`, `tab_05_eccentric_newtonian.tex` | final |
| 06 | Eccentric cylinders: wall pressure around the rotor | `fig_06_eccentric_wall_pressure.pdf` | final |
| 07 | Eccentric fields: pressure, speed, micro-rotation, w - curl v / 2 | `fig_07_eccentric_fields.pdf` | final |
| 08 | Eccentric cylinders: measured over predicted change against narrowest gap / l | `fig_08_eccentric_prediction.pdf` | final |
| 09 | Table of every eccentric micropolar row | `tab_09_eccentric_rows.csv`, `tab_09_eccentric_rows.tex` | final |
| 10 | The film function over h^3 against h / l at K = 3 | `fig_10_film_function.pdf` | final (host only) |
| 11 | Micropolar cavity: centre lines against the finite-difference solution, fields, the slab cut | `fig_11_micropolar_cavity.pdf`, `tab_11_micropolar_cavity.csv`, `tab_11_micropolar_cavity.tex`, `tab_11b_cavity_finite_differences.csv`, `tab_11b_cavity_finite_differences.tex`, `tab_11c_lid_traction.csv`, `tab_11c_lid_traction.tex` | final |
| 12 | Newtonian cavity baseline at n = 1000 against Marchi et al. | `fig_12_cavity_baseline.pdf` | final (existing data) |
| 13 | Seepage of the thesis walls: velocity into the wall against wall pressure; the wall candidates | `fig_13_seepage.pdf`, `tab_13_candidates.csv`, `tab_13_candidates.tex` | final (existing data) |
| 14 | Solver checks: operator checks, passive label, chain against one slab, step limits, cost | `tab_14a_operator_checks.csv`, `tab_14a_operator_checks.tex`, `tab_14b_passive_label.csv`, `tab_14b_passive_label.tex`, `tab_14c_chain_against_one_slab.csv`, `tab_14c_chain_against_one_slab.tex`, `tab_14d_step_limits.csv`, `tab_14d_step_limits.tex`, `tab_14e_cost.csv`, `tab_14e_cost.tex` | final (existing data) |
| 15 | 3-D: fields in a plane through the axis and the mid-plane profiles | `fig_15_three_dimensional.pdf`, `tab_15b_three_dimensional.csv`, `tab_15b_three_dimensional.tex` | final |
| 16 | The map of the effect: contours of D in the plane (N, m) for the torque and the load | `fig_16_effect_map.pdf` | final (host only) |
| 20 | Dipole-wall collision: vorticity of the control and of the two micropolar runs at four times | `fig_20_dipole_showcase.pdf` | final |
| 21 | Dipole-wall collision: core trajectories and enstrophy against the benchmark | `fig_21_dipole_tracks.pdf` | final |
| 22 | Dipole-wall collision: numbers against the control with the noise floor | `tab_22_dipole_numbers.csv`, `tab_22_dipole_numbers.tex` | final |
| 23 | Micropolar Taylor-Green vortex: mu_eff against |k| l, and the two-mode solution | `fig_23_taylor_green.pdf` | final |
| 24 | Decaying 2-D turbulence: vorticity, E(t) and the enstrophy Z(t), spectra and their ratio | `fig_24_turbulence.pdf` | final |
| 25 | Table of the Taylor-Green rates and ratios with their noise floors | `tab_25_taylor_green.csv`, `tab_25_taylor_green.tex` | final |
| 26 | Dipole-wall collision: the checks of micro-inertia, saturation and resolution | `tab_26_dipole_checks.csv`, `tab_26_dipole_checks.tex` | final |

## 2. Concentric cylinders (items 01 to 04) [run]

Thesis configuration with `numerics.delta_fluid_only: true`, Re = 10 on mu_inf, Ma = 0.02, w = 0 on both walls; D = relative change of the torque against the Newtonian control on the same particles (velocity fit).

| K | m = g / (2 l) | g / dx | l / dx | D (run) | D (exact) | run / exact | noise floor / D | slip-form rule / exact |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 100 | 50 | 0.32250 | 0.32212 | 1.0012 | 0.0008 | 1.4873 |
| 3 | 2 | 100 | 25 | 0.17769 | 0.17746 | 1.0013 | 0.0012 | 1.0392 |
| 3 | 5 | 50 | 5 | 0.06696 | 0.06503 | 1.0297 | 0.0069 | 0.9972 |
| 3 | 5 | 100 | 10 | 0.06550 | 0.06503 | 1.0073 | 0.0030 | 0.9972 |
| 3 | 5 | 200 | 20 | 0.06511 | 0.06502 | 1.0015 | 0.0002 | 0.9972 |
| 3 | 10 | 100 | 5 | 0.03219 | 0.03121 | 1.0313 | 0.0061 | 0.9985 |
| 3 | 10 | 200 | 10 | 0.03144 | 0.03121 | 1.0075 | 0.0004 | 0.9985 |
| 3 | 20 | 200 | 5 | 0.01579 | 0.01530 | 1.0322 | 0.0007 | 0.9992 |
| 3 | 20 | 400 | 10 | 0.01553 | 0.01530 | 1.0152 | 0.0007 | 0.9992 |
| 1 | 1 | 100 | 50 | 0.18957 | 0.18950 | 1.0004 | 0.0012 | 1.4053 |
| 1 | 2 | 100 | 25 | 0.11030 | 0.11020 | 1.0009 | 0.0019 | 1.0363 |
| 1 | 5 | 100 | 10 | 0.04247 | 0.04221 | 1.0061 | 0.0046 | 0.9973 |
| 1 | 10 | 200 | 10 | 0.02067 | 0.02054 | 1.0066 | 0.0006 | 0.9986 |
| 1 | 20 | 200 | 5 | 0.01040 | 0.01013 | 1.0265 | 0.0011 | 0.9993 |
| 0.5 | 1 | 100 | 50 | 0.11722 | 0.11715 | 1.0006 | 0.0018 | 1.3645 |
| 0.5 | 2 | 100 | 25 | 0.07037 | 0.07026 | 1.0015 | 0.0028 | 1.0347 |
| 0.5 | 5 | 100 | 10 | 0.02780 | 0.02765 | 1.0052 | 0.0069 | 0.9974 |
| 0.5 | 10 | 200 | 10 | 0.01366 | 0.01358 | 1.0061 | 0.0008 | 0.9987 |
| 0.5 | 20 | 200 | 5 | 0.00689 | 0.00673 | 1.0241 | 0.0017 | 0.9993 |

- **Resolution of the layer decides the error:** run / exact is 1.0241 to 1.0322 at l = 5 dx; 1.0052 to 1.0152 at l = 10 dx; 1.0015 at l = 20 dx; 1.0009 to 1.0015 at l = 25 dx; 1.0004 to 1.0012 at l = 50 dx.
- **Order in h / l** between successive resolutions of one row: m = 5: 2.03, 2.30; m = 10: 2.06; m = 20: 1.09.
- **The pair at g = 400 dx** (m = 20, K = 3, l = 10 dx; 625,000 steps each, every counter zero): run / exact 1.0152, against 1.0322 for the same m at l = 5 dx. The excess of the m = 20 row halves with the resolution of the layer. **It does not land on the other rows at l = 10 dx** (1.0052 to 1.0075, where one doubling took a factor of four): D is 1.2e-04 above them, about 10 times the lattice floor at g = 200 dx. What is known of the pair: its step is the only one set by the viscous limit; it has one lattice, so no floor of its own (the table carries that of g = 200 dx); and its control is in another state than the others: constant over nominal 1.000006 (0.999791 at g = 200 dx), hydrodynamic radii +0.005 and +0.007 dx from the nominal ones (-0.064 and -0.066), mean density +1.2e-06 rho0 (-5.4e-06), rms of the fit 9.4e-05 U (1.5e-05). The fitted constants are steady. **Not explained**, and not followed: nothing else was to run at g = 400 dx. I would quote the row as 'the excess at m = 20 halves', not as a point of the l / dx law.
- **Noise floor** (spread of the control's constant over three lattices): 4.2e-04 at g = 50 dx, 1.8e-04 at g = 100 dx, 1.1e-05 at g = 200 dx.
- **The wall rule in its slip form** over exact: m = 1: 1.3645 to 1.4873; m = 2: 1.0347 to 1.0392; m = 5: 0.9972 to 0.9974; m = 10: 0.9985 to 0.9987; m = 20: 0.9992 to 0.9993. Exact up to exponentially small terms for shear-driven flow once m is about 5 or more; it fails below m of about 2.

## 3. Eccentric cylinders (items 05 to 09)

**Newtonian controls against the exact Stokes solution** [run, host] (R1 = 4 g, R2 = 5 g; item 05):

| run | eps | g / dx | lattice | load / exact - 1 (%) | torque, flux form, rotor (%) |
|---|---|---|---|---|---|
| `e30_newtonian` | 0.3 | 100 | as generated | -0.29 | -0.11 |
| `e60_newtonian` | 0.6 | 100 | as generated | -0.61 | +0.03 |
| `e30_g200_newtonian` | 0.3 | 200 | as generated | -0.12 | -0.10 |
| `e60_g200_newtonian` | 0.6 | 200 | as generated | -0.34 | -0.00 |
| `e30_newtonian_shifted` | 0.3 | 100 | shifted | +0.04 | -0.13 |
| `e30_newtonian_rotated` | 0.3 | 100 | rotated | +0.15 | -0.11 |
| `e60_newtonian_shifted` | 0.6 | 100 | shifted | -0.26 | -0.04 |
| `e60_newtonian_rotated` | 0.6 | 100 | rotated | -0.05 | +0.06 |
| `spread of e30 over three lattices` | 0.3 | 100 | max - min | 0.43 | 0.03 |
| `spread of e60 over three lattices` | 0.6 | 100 | max - min | 0.56 | 0.10 |

Noise floor from three lattices (max - min): eps = 0.3: load 0.43 %, D_T 0.0007, load change 0.04 points; eps = 0.6: load 0.56 %, D_T 0.0003, load change 0.04 points.

**Micropolar runs against the two predictions** [run, host] (item 09; D_T with the concentric reading errors taken off; film = the film function of micropolar lubrication theory in the thin-film calculation, which is not a result of ours; rule = the wall rule on the exact Stokes solution of its slip-form geometry):

| run | K | eps | g / dx | l / dx | narrowest gap / l | D_T (run) | D_T film | D_T rule | load change (%) | film (%) | rule (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `e30_k3_m2` | 3 | 0.3 | 100 | 25 | 2.8 | 0.2182 | 0.2230 | 0.2726 | +41.08 | +42.13 | +87.43 |
| `e30_k3_m5` | 3 | 0.3 | 100 | 10 | 7.0 | 0.0870 | 0.0901 | 0.0902 | +21.71 | +21.66 | +26.58 |
| `e30_g200_k3_m5` | 3 | 0.3 | 200 | 20 | 7.0 | 0.0863 | 0.0901 | 0.0902 | +21.54 | +21.66 | +26.58 |
| `e30_g200_k3_m10` | 3 | 0.3 | 200 | 10 | 14.0 | 0.0420 | 0.0440 | 0.0427 | +11.37 | +11.24 | +12.25 |
| `e60_k3_m2` | 3 | 0.6 | 100 | 25 | 1.6 | 0.3093 | 0.3130 | 0.5746 | +39.41 | +40.60 | +105.30 |
| `e60_k3_m5` | 3 | 0.6 | 100 | 10 | 4.0 | 0.1444 | 0.1481 | 0.1649 | +21.68 | +21.81 | +29.56 |
| `e60_g200_k3_m5` | 3 | 0.6 | 200 | 20 | 4.0 | 0.1434 | 0.1481 | 0.1649 | +21.53 | +21.81 | +29.56 |
| `e60_g200_k3_m10` | 3 | 0.6 | 200 | 10 | 8.0 | 0.0715 | 0.0746 | 0.0754 | +11.95 | +11.84 | +13.41 |
| `e30_k1_m2` | 1 | 0.3 | 100 | 25 | 2.8 | 0.1331 | 0.1356 | 0.1629 | +23.99 | +24.61 | +49.75 |
| `e30_k1_m5` | 1 | 0.3 | 100 | 10 | 7.0 | 0.0556 | 0.0576 | 0.0579 | +13.44 | +13.45 | +16.77 |
| `e30_g200_k1_m5` | 1 | 0.3 | 200 | 20 | 7.0 | 0.0553 | 0.0576 | 0.0579 | +13.34 | +13.45 | +16.77 |
| `e30_g200_k1_m10` | 1 | 0.3 | 200 | 10 | 14.0 | 0.0275 | 0.0287 | 0.0279 | +7.28 | +7.22 | +7.95 |
| `e60_k1_m2` | 1 | 0.6 | 100 | 25 | 1.6 | 0.1834 | 0.1851 | 0.3140 | +23.10 | +23.77 | +56.88 |
| `e60_k1_m5` | 1 | 0.6 | 100 | 10 | 4.0 | 0.0903 | 0.0926 | 0.1035 | +13.34 | +13.46 | +18.45 |
| `e60_g200_k1_m5` | 1 | 0.6 | 200 | 20 | 4.0 | 0.0897 | 0.0926 | 0.1035 | +13.27 | +13.46 | +18.45 |
| `e60_g200_k1_m10` | 1 | 0.6 | 200 | 10 | 8.0 | 0.0461 | 0.0481 | 0.0489 | +7.59 | +7.55 | +8.67 |

- **The runs reproduce the film function:** load change over film 0.971 to 1.011, torque change over film 0.953 to 0.991.
- **The rule over-predicts where the film is not thick against l:** load change over rule 0.374 to 0.928, torque change over rule 0.538 to 0.984. First order for pressure-driven flow.
- **Resolution pair** (g = 100 dx, l = 10 dx against g = 200 dx, l = 20 dx; eps = 0.3, K = 3): D_T 0.0870 and 0.0863, load change 21.71 and 21.54 %.
- The load does not see the slip term of the rule, the torque does (design note 15.3.3) [host].

## 4. The film function (item 10) [host]

f = h^3 - 6 delta h^2 coth(h / (2 l)) + 12 delta l h, delta = K l / (2 (1 + K)); the film function of micropolar lubrication theory (not ours; its identification with the form of that theory is algebra on our side, not compared with the original). The model's two ordinary differential equations between the walls agree with it to 1e-15. The sharper form of the rule, h^3 - 6 delta h^2 + 12 delta l h, holds: it differs by 6 delta h^2 (coth(h / (2 l)) - 1).

From the caption: Ratios to the film function: h / l = 4: rule 0.769, first order 0.627, sharper form 1.0301; h / l = 7: rule 0.925, first order 0.881, sharper form 1.0008; h / l = 10: rule 0.965, first order 0.945, sharper form 1.0000; h / l = 14: rule 0.983, first order 0.973, sharper form 1.0000; h / l = 20: rule 0.992, first order 0.987, sharper form 1.0000.

## 5. Micropolar cavity (item 11, tables 11b and 11c)

K = 3, Re = 400 on mu_inf (1,000 on mu), l = L / 20, Theta = l^2, w = 0 on the four walls, Ma = 0.01, default walls. **The check of the case is the second method**: the same equations by finite differences on the host (stream function, vorticity, micro-rotation; grids of 128, 192, 256 intervals, extrapolated).

| fluid | u_min: particles (n = 500) / finite differences | v_max | v_min | kinetic energy over fluid mass | particles stronger by (%) | centre lines, rms difference u, v (U) |
|---|---|---|---|---|---|---|
| micropolar | -0.3580 / -0.3567 | 0.3228 / 0.3213 | -0.4627 / -0.4614 | 0.0465 / 0.0463 | +0.3, +0.5, +0.3 | 2.1e-03, 1.6e-03 |
| Newtonian, mu_inf (Re 400) | -0.3311 / -0.3287 | 0.3066 / 0.3038 | -0.4573 / -0.4541 | 0.0412 / 0.0408 | +0.7, +0.9, +0.7 | 2.8e-03, 2.5e-03 |
| Newtonian, mu + kappa (Re 250) | -0.2890 / -0.2877 | 0.2548 / 0.2534 | -0.3931 / -0.3916 | 0.0386 / 0.0385 | +0.4, +0.6, +0.4 | 2.1e-03, 1.5e-03 |

- [run] Two slabs (5090 + XTX) against one slab: centre lines differ by 2.1e-05 (u) and 1.6e-05 (v) U rms. n = 250 against n = 500: 3.0e-03 and 3.5e-03 U rms, extrema 1.3 to 1.5 % stronger at n = 250.
- [host] The micropolar extrema are stronger than those of the Newtonian fluid of viscosity mu_inf (finite differences: +8.5, +5.8, +1.6 %). With walls that do not hold w the finite-difference solution is weaker than that Newtonian fluid; with w held on the lid alone it is strongest (table 11b). [inferred] The condition w = 0 on the driving wall makes the lid take hold of the fluid as of the more viscous one.
- [run, host] **Against Marchi et al. at Reynolds number 400** (transcribed table; its coordinates, printed to four decimals, taken as multiples of 1 / 128): the Newtonian particle run of viscosity mu_inf differs by 3.0e-03 (u) and 2.1e-03 (v) U rms (largest 8.8e-03 at y = 0.977, next to the lid); the finite-difference solution, extrapolated to zero spacing at the points, by 3.0e-05 and 1.9e-05 (largest 6.9e-05).
- [host] **Table 11c**, the traction of the lid over that on the Newtonian fluid of viscosity mu_inf, minus one, against delta / lambda (lambda = mu_inf U / traction = 0.063 L, the shear length of the lid):

| l / L | delta / L | delta / lambda | excess traction | excess / (delta / lambda) |
|---|---|---|---|---|
| 1/10 | 0.0375 | 0.597 | 0.296 | 0.49 |
| 1/20 | 0.0188 | 0.299 | 0.232 | 0.78 |
| 1/40 | 0.0094 | 0.149 | 0.144 | 0.97 |
| 1/80 | 0.0047 | 0.075 | 0.077 | 1.03 |

  The first-order rule with the shear length in the place of the gap, within 4 % from l = L / 40 down (ratios on like grids).

## 6. Newtonian cavity baseline, seepage, solver checks (items 12 to 14) [run, existing data]

- **Item 12**, Re = 1000, n = 1000 against Marchi et al.: L2 error of u 1.42e-03 U (largest 5.31e-03), of v 9.49e-04 U (largest 2.28e-03); steady at t = 69.5; rms error away from the walls 4.4e-04 (u) and 7.9e-04 (v).
- **Item 13**, the seepage of the thesis walls: velocity into the wall per wall pressure 0.0093 U per rho U^2 (thesis_e60_re10, rotor, correlation 0.97), 0.0090 U per rho U^2 (thesis_e60_re10, stator, correlation 0.97) in the bearing at Re = 10, Ma = 0.02; the cavity keeps its walls and the chapter states the law and its size (caption of item 13 and `tab_13_candidates`).
- **Item 14**, five tables from existing data (operator checks, passive label, chain against one slab, step limits, cost): `tab_14a` to `tab_14e`, read with the caption of item 14.

## 7. The 3-D case (item 15, table 15b) [run]

Rotor of radius 3 g and length 3 g in a housing of radius 4 g with end gaps g, K = 3, l = 10 dx, w = 0 on every wall, Re = 10 on mu_inf, Ma = 0.02, h = 4 dx; each micropolar run with its Newtonian control on the same particles and the same dt. The case of the chapter is g = 40 dx on two cards (5090 + XTX). The two columns are two cases (m = 1 and m = 2), not one case at two resolutions.

| quantity | g = 20 dx (g20_k3_m1) | g = 40 dx (g40_k3_m2_k2) |
|---|---|---|
| g / dx; l / dx; m = g / (2 l) | 20; 10; 1 | 40; 10; 2 |
| particles; slabs | 2,587,028; 1 | 15,604,476; 2 |
| Newtonian control | g20_newtonian | g40_newtonian_k2 |
| torque, Newtonian: flux form (N m) | 41.540 | 41.790 |
| torque, Newtonian: on the housing | 41.484 | 41.767 |
| torque, micropolar: flux form | 55.265 | 49.514 |
| torque, micropolar: on the housing | 55.674 | 49.997 |
| torque change D_T, flux form | 0.3304 | 0.1848 |
| D_T on the housing | 0.3421 | 0.1970 |
| D_T, wall form | 0.3297 | 0.1924 |
| D_T, wall form, lateral surface | 0.3166 | 0.1863 |
| D_T, wall form, end faces | 0.3610 | 0.2069 |
| D_T on the housing, lateral face | 0.3153 | 0.1815 |
| D_T on the housing, end faces | 0.4124 | 0.2378 |
| exact 2-D D for the radii (lateral gap) | 0.3034 | 0.1669 |
| the wall rule (Newtonian run in the slip-form geometry over the control): D_T, flux form | - | 0.2429 |
| the wall rule: D_T on the housing | - | 0.2429 |
| the wall rule: D_T, wall form | - | 0.2410 |
| the wall rule: D_T, wall form, lateral surface | - | 0.1593 |
| the wall rule: D_T, wall form, end faces | - | 0.4380 |
| the wall rule: D_T on the housing, lateral face | - | 0.2106 |
| the wall rule: D_T on the housing, end faces | - | 0.3277 |
| the wall rule: change of the mid-plane constant | - | 0.1724 |
| exact plane film between two discs (end faces) | 0.3998 | 0.2206 |
| mid-plane constant over the 2-D one, Newtonian | 0.9945 | 0.9979 |
| mid-plane constant over the 2-D one, micropolar | 0.9929 | 0.9990 |
| micropolar effect in the mid-plane (change of the constant) | 0.3012 | 0.1682 |
| the same over the exact 2-D D | 0.9929 | 1.0075 |
| reach of the ends, 1 % (g) | 0.78 | 0.46 |
| reach of the ends, 0.1 % (g) | 1.38 | 1.36 |
| wall layer on the end face, normal component: length over l sqrt(nu_inf / nu) | 0.983 | 0.978 |
| wall layer on the end face, tangential component: length over l | 1.217 | 0.995 |
| mean density - rho0, over rho0 (Newtonian; micropolar) | -2.0e-04; -9.9e-05 | -5.6e-05; -3.3e-05 |
| mean pressure in rho U^2 (Newtonian; micropolar) | -0.502; -0.247 | -0.141; -0.083 |
| its drift per eighth of the run (Newtonian; micropolar) | -1.3e-02; -5.7e-03 | -1.7e-04; -1.7e-05 |
| steps per second (Newtonian; micropolar) | 45.9; 36.8 | 7.4; 6.6 |

- **Three readings of the torque change at g = 40 dx:** 0.1848 (flux form), 0.1970 (on the housing), 0.1924 (wall form). In the Newtonian run the flux form and the housing agree to 5.5e-04; in the micropolar run the flux form is 1.0 % below the housing and still rising at the end of the run (it rises through the 13 states, the housing does not move). I would quote the housing and give the flux form as the lower reading.
- **Mid-plane:** the micropolar effect is 0.1682 against the exact 2-D D of 0.1669 (1.0075 of it).
- **Lateral surface and end faces:** D_T 0.1863 against the exact 2-D 0.1669; 0.2069 against the plane film between two discs, 0.2206. The rims belong to both.
- **The rule in 3-D** [host, run]: the argument for the normal component is right; that component has its own layer length l sqrt(nu_inf / nu), and the run shows both lengths (table).
- **The rule in 3-D, run** [run] (the Newtonian fluid in the slip-form geometry at g = 40 dx: lateral surface of the rotor at R1 + delta turning at Omega (R1 - delta) / (R1 + delta), its end faces out by delta at Omega, the housing in by delta; rows 'the wall rule' of the table). **Mid-plane:** change of the constant 0.1724 against 0.1682 in the micropolar run (+2.5 %). **Torque:** 0.2429 (flux form), 0.2429 (housing), 0.2410 (wall form) against 0.1848, 0.1970, 0.1924: the rule in this plainest reading overstates the change of the torque by 23 to 31 %. The excess arrives on both kinds of faces of the housing (end faces 0.328 against 0.238, lateral face 0.211 against 0.182); the mid-plane being right, [inferred] it comes from the end regions: the displaced rotor is a larger body (end faces to R1 + delta at the full rate, end gap g - 2 delta); the wall layer at a convex rim does not act as one. At the rim the two rates meet without blending: each wall particle of the rotor takes the rate of the surface it is nearer to (jump 2 delta Omega = 0.0625 U).
- **Chain:** per-particle seam checks pass on three slabs (one card) and two slabs (XTX + 5090); the cavity on two slabs equals the one-slab run to 2e-5 U.

## 8. The map of the effect (item 16) [host]

Contours D = 1 %, 5 %, 20 % in the plane of N = sqrt(K / (2 + K)) and m = g / (2 l): (a) the torque between the concentric cylinders from the exact solution, with the slip-form rule; (b) the load of the bearing at an eccentricity ratio of 0.6 from the film function in the thin-film calculation. m at which D equals 1 %, 5 %, 20 % ('-': outside 1 <= m <= 100):

| K | torque, exact | torque, slip-form rule | load, film function |
|---|---|---|---|
| 3 | 30.4 / 6.4 / 1.8 | 30.4 / 6.4 / 1.9 | - / 24.3 / 5.6 |
| 1 | 20.3 / 4.3 / - | 20.2 / 4.2 / 1.2 | 80.9 / 15.6 / 2.8 |
| 0.5 | 13.5 / 2.8 / - | 13.5 / 2.8 / - | 53.4 / 9.8 / - |

Noise floors shaded in the figure: 1.1e-05 (torque, g = 200 dx) and 3.6e-04 (change of the load, measured at g = 100 dx). The load reaches a given D at three to four times the m at which the torque does.

## 9. The dipole-wall collision (items 20 to 22) [run]

Normal collision of a dipole with a no-slip wall (Clercx & Bruneau 2006), box [-W, W]^2, Re = 625 on nu_inf, dx = W / 1000 (4 million fluid particles, h = 5 dx); the Newtonian control (viscosity mu_inf) and the micropolar fluid with K = 3, l = W / 100 and W / 50, w = 0 on every wall, w = omega / 2 at t = 0.

- **The control validates** against the benchmark: enstrophy peaks 918.8 at t = 0.3724 and 301.2 at 0.6492 (spectral 933.6 at 0.3711 and 305.2 at 0.6479: -1.6 % and -1.3 %); the primary core within 0.0054 W of the benchmark at t = 0.6, 0.625, 1.0; E and Omega at t = 0.25, 0.5, 0.75 within 1 %. The first peak converges from below (-9.4, -3.7, -1.6 % at n = 250, 500, 1000); halving the Mach number moves it by 0.01 %.

| against the control | l = W / 100 | l = W / 50 | noise floor |
|---|---|---|---|
| first enstrophy peak | -13.5 % | -15.7 % | 0.04 % |
| enstrophy at the control's second peak | -5.2 % | -11.1 % | 0.08 % |
| kinetic energy at t = 0.75 | -2.2 % | -3.8 % | 0.10 % |
| wall vorticity at the first collision | -26.1 % | -25.7 % | 0.44 % |
| wall shear traction (mu + kappa) omega_w | +18.2 % | +18.9 % | 0.44 % |
| vorticity flux into the fluid at the wall | -52.5 % | -37.5 % | 1.13 % |
| pressure difference along the wall | +1.4 % | -0.4 % | 3.89 % |
| rebound of the primary core (W) | -0.0064 | -0.0090 | 0.0013 |

- **Outside the first-order rule.** The shear length of the wall at the first collision is Lambda = 0.0053 W, so delta* / Lambda = 0.71 and 1.42 (Lambda / l = 0.53 and 0.26); the measured excess of the wall traction is +18 to +23 %, nearly independent of l: it saturates once the layer is thicker than the boundary layer. [inferred] Pinned at w = 0 across the boundary layer, the fluid has the viscosity mu + kappa there: sqrt((mu + kappa) / (mu + kappa / 2)) - 1 = 0.26 for a layer growing as sqrt(nu t).
- **The mechanism.** The pressure along the wall is the same in all runs (set by the outer flow). At a no-spin wall dp/ds = (mu + kappa) d omega / dn - kappa d w / dn: with the measured d w / dn the balance closes to 1.056 and 1.088 of the control, the couple term taking 0.39 (l = W / 100) and 0.09 (W / 50) of the vorticity term. The no-spin wall passes part of the vorticity it would produce to the micro-rotation, and less vorticity leaves the wall; the secondary vortex is weaker and **neither micropolar run has a second enstrophy maximum** (the control has one at t = 0.649) ([inferred] the second because of the first).
- **Figures:** item 20 (vorticity of the three runs at t = 0.35, 0.45, 0.6, 0.7, and w - omega / 2 of l = W / 50), item 21 (core trajectories, enstrophy with the benchmark points), table 22.

### 9.1 Three checks (table 26) [run]

Each run against its control with the definitions and floors of table 22 (the core positions from the strongest extremum within reach, for the reference too).

| against the control | l = W/50, j = l^2/100 | the same against j = l^2 | l = W/25, j = l^2/100 | l = W/50 at 10 dx | reference: l = W/50, 20 dx, j = l^2 |
|---|---|---|---|---|---|
| first enstrophy peak | -20.8 % | -6.1 % | -29.6 % | -15.3 % | -15.7 % |
| enstrophy at the control's second peak | -15.6 % | -5.1 % | -28.7 % | -10.7 % | -11.1 % |
| kinetic energy at t = 0.25 (before the collision) | -1.8 % | -1.3 % | -4.7 % | -0.4 % | -0.4 % |
| enstrophy at t = 0.25 | -5.1 % | -3.7 % | -11.1 % | -1.4 % | -1.4 % |
| primary core, vorticity maximum at t = 0.25 | -3.1 % | -2.4 % | -7.9 % | -1.6 % | -0.7 % |
| wall vorticity at the first collision | -28.1 % | -3.3 % | -33.0 % | -24.2 % | -25.7 % |
| wall shear traction, peak | +15.0 % | -3.3 % | +7.2 % | +21.3 % | +18.9 % |
| wall shear traction, integral | +18.7 % | -3.6 % | +16.0 % | +21.8 % | +23.2 % |
| vorticity flux into the fluid at the wall | -42.6 % | -8.2 % | -38.9 % | -44.2 % | -37.5 % |

- **The micro-inertia matters** (j = l^2 / 100 against j = l^2, l = W / 50): the first enstrophy peak moves by -6.1 %, 0.39 of the micropolar effect (-15.7 %), more than the quarter set in the task; the wall vorticity by -3.3 % (0.13 of the effect). Most of the difference is there before the collision: at t = 0.25 the kinetic energy is -1.76 % and the enstrophy -5.1 % off the control (j = l^2: -0.44 % and -1.4 %; floor 0.01 % and 0.04 %). With j = l^2 the micro-rotation lags the vorticity of a decaying flow and the bulk effect is held back (here to a quarter to a third of that with j = l^2 / 100; the exact Taylor-Green solution of Stage 8b gives a third of the quasi-steady mu_eff excess at small |k| l, and j = l^2 / 100 recovers it). The dt did not change (the angular-diffusion limit 3.5e-6 lies above 1.25e-6).
- **Saturation: the reading does not hold here** (l = W / 25 = 40 dx, j = l^2 / 100, Lambda / l = 0.13). The wall traction excess is +7.2 % (peak) and +16.0 % (integral), not the +26.5 % (sqrt(1.6) - 1) of a layer pinned at w = 0 across the boundary layer of the control's flow: the dipole itself (core radius r0 = 2.5 l) is in the micropolar range and arrives weaker. **The vorticity maximum of the primary core at t = 0.25 is 221.5 against 240.5 of the control (-7.9 %; floor -0.13 % shifted lattice, +2.3 % half Mach)**, E and the enstrophy at t = 0.25 -4.7 % and -11.1 %; the wall vorticity of the first collision -33.0 %, the first enstrophy peak -29.6 %. At the wall (mu + kappa) d omega / dn - kappa d w / dn is +1.0 % off the control's mu_inf d omega / dn and the pressure difference along the wall -1.0 %: the balance closes.
- **Resolution of l** (l = W / 50 at 10 dx, W / dx = 500, against `control_n500`): the first enstrophy peak -15.3 % and the wall vorticity -24.2 % (20 dx: -15.7 % and -25.7 %); differing by more than a tenth of the effect: wall shear traction, peak +21.3 % against +18.9 % (0.13 of the effect); vorticity flux into the fluid at the wall -44.2 % against -37.5 % (0.18 of the effect). [inferred] The quantities read at the wall (traction, flux) are those that resolve the layer w = 0 to omega / 2; the global ones agree.

## 10. The periodic box: Taylor-Green and decaying turbulence (items 23 to 25) [run]

The periodic domain of v8-tgv ported into the solver of the chapter (one card, one slab, 2-D; nothing changes with the switch off). Design note Section 18.

- **Port check.** v8's Newtonian Taylor-Green (Re 100, Ma 0.1 to 0.01, 400^2 and 800^2): nu_eff / nu 0.9948 to 1.0006, within 6e-04 of v8's.

Micropolar Taylor-Green vortex (k = 1, Re = 20 on nu_inf, K = 3, Ma 0.02, 400^2 unless named; j = l^2 unless named), rates fitted over 5 <= t <= 20:

| \|k\| l | rate / exact | (W/U) / exact | rate over control | exact | quasi-steady mu_eff |
|---|---|---|---|---|---|
| 0.25 | 0.9997 | 0.9989 | 1.0083 | 1.0076 | 1.0226 |
| 0.5 | 0.9996 | 0.9990 | 1.0287 | 1.0281 | 1.0811 |
| 1 | 0.9993 | 0.9994 | 1.0859 | 1.0856 | 1.2308 |
| 2 | 0.9990 | 1.0001 | 1.1612 | 1.1612 | 1.4286 |
| 0.5 (800^2) | 0.9996 | 0.9994 | 1.0285 | 1.0281 | 1.0811 |
| 0.5, j = l^2/100 | 1.0010 | 0.9996 | 1.0807 | 1.0806 | 1.0811 |
| 1, j = l^2/100 | 1.0001 | 1.0001 | 1.2285 | 1.2296 | 1.2308 |
| 1, j = 1.5 l^2 | 0.9994 | 0.9995 | 1.0004 | 1.0000 | 1.2308 |
| 1, j = 0.755 l^2, W(0) = 0 | 0.9993 | 0.9993 | 1.1331 | 1.1327 | 1.2308 |

- **Every run follows the exact two-mode solution** to 0.1 % (the Newtonian control misses its exact rate by -0.1 % as well). **With j = l^2 the runs reach only a third of the quasi-steady mu_eff excess**, as the exact solution predicts: a decaying flow keeps spin from its earlier vorticity, and with j scaling as l^2 that memory stays of the size of the effect at every |k| l. With j = l^2 / 100 the runs reach mu_eff (1.0807 and 1.2285 against 1.0811 and 1.2308); with j = 1.5 l^2 (rho j (mu + kappa / 2) = gamma) w = omega / 2 is exact and the micropolar vortex decays exactly as the Newtonian fluid mu + kappa / 2 (1.0004).
- **Floors:** repeat of the control +1.2e-06; repeat of |k|l = 1 +3.6e-06; control at Ma 0.01 +1.6e-04; |k|l = 1 at Ma 0.01 +9.0e-05; |k|l = 0.5 at 800^2 +2.6e-05; control at dt / 8.6 +1.9e-03 (relative change of the rate).
- **Decaying turbulence** (box 2 pi, k0 = 10, u' = 1, Re = 100 on nu_inf and 250 on mu; micropolar K = 3, l = 1/7; Z is the enstrophy): runs `turbulence_newtonian_n2050` (4,202,500 particles, 388,000 steps, 113 min), `turbulence_micropolar_n2050` (4,202,500 particles, 388,000 steps, 66 min), `turbulence_newtonian_n2050_shifted` (4,202,500 particles, 388,000 steps, 113 min), `turbulence_newtonian_n1025` (1,050,625 particles, 196,000 steps, 15 min), `turbulence_micropolar_j001_n2050` (4,202,500 particles, 388,000 steps, 113 min), `turbulence_micropolar_n2050_repeat` (4,202,500 particles, 388,000 steps, 66 min), `turbulence_newtonian_n2050_repeat` (4,202,500 particles, 388,000 steps, 66 min).
- **Micropolar (j = l^2) against Newtonian:** t = 1.0: E -0.4%, Z -1.7%; t = 2.0: E -0.8%, Z -2.6%; t = 4.0: E -1.3%, Z -3.9%.
- **Micropolar (j = l^2 / 100 (beyond the task)) against Newtonian:** t = 1.0: E -5.8%, Z -13.3%; t = 2.0: E -7.7%, Z -15.8%; t = 4.0: E -9.1%, Z -19.6%.
- **Floors:** the shifted lattice moves E by 2e-05 and Z by 4e-04 at t = 4 (a repeat and the other card alike); the spectra by 1 to 2 % per band at t = 2 and up to 10 % at t = 4 (chaotic divergence of the small scales). **With j = l^2 the effect is small and sits in the dissipation range** (spectrum ratio 0.87 at k 30 - 60 and 0.65 at 60 - 120 at t = 2): the spin relaxes on Theta / (2 c) = 8.5, longer than the run, and advection keeps w = omega / 2. **With j = l^2 / 100 it is seven times larger in E** and the energy-containing band follows the linear quasi-steady estimate; at higher k the cascade keeps both far above the single-mode estimates. Item 24, design note 18.5.
- **Figures:** item 23 (rate over the control against |k| l, the micro-inertia run), item 24 (turbulence), table 25.

## 11. To be careful with in the text

- **The film function** is that of micropolar lubrication theory; ours is that the runs reproduce it at a clearance ratio of 0.25. Its match with the published form has not been checked against the original.
- **The cavity is not 'between two Newtonian fluids'**, and the pack no longer says so; its check is the finite-difference solution, which in turn agrees with the reference of Marchi et al. at Reynolds number 400 (your transcription from the appendix of paper 1, `reference/marchi_re400.csv`, 28 + 28 points) to 3.0e-05 (u) and 1.9e-05 (v) U rms once extrapolated to zero spacing.
- **3-D torque change:** three readings, 0.185 to 0.197 at g = 40 dx; there is no exact 3-D value. Say which reading is quoted.
- **The dipole collision** uses omega_e = 299.528 (E(0) = 2), not the 'about 320' of the paper's text; the palinstrophy of the runs is dominated by grid-scale particle noise and is not a validation number; the wall vorticity flux is read from the grid with a bias common to all runs (only its ratios are quoted).
- **The slip-form rule in 3-D** is good for the mid-plane and a quarter too large for the torque of this short rotor (one run, one reading of the rule at the rims). Do not quote the 2-D accuracy of the rule for the 3-D torque.
- **The cylinder cases carry `numerics.delta_fluid_only: true`** (the key), the cavity cases the default walls. With the key the level of the pressure wanders (recorded for every run; no reading follows it).
- **Kinetic energy of the cavity** is quoted over the fluid mass (the particle masses sum to 1.134 rho0 L^2 at n = 500).
- **Within one support of a slab cut** the card's acceleration differs from a host sum over the saved particles by 2e-3 to 4e-3 of its rms; 1e-6 elsewhere. Tested on the saved two-slab state (`rotor3d_cut_check.txt`): taking the neighbours' pressure from the host's equation of state of their single-precision density does not bring it to 1e-6, because the host's evaluation is neither card's (0.21 Pa rms from the saved pressure). It is the difference the two cards show against each other in every particle in the seam check (4e-3 m/s^2, there 4e-5 of a prescribed field of 109 m/s^2), seen here only where a card uses pressures of particles it does not own. **[run] The cause is the two cards' own evaluation of the equation of state:** the same two slabs with both on the 5090 (2,000 steps at g = 20 dx) show 9.7e-07 and 1.0e-06 within one support of the cut, the value of the interior; on 5090 + XTX 2.1e-03 and 1.9e-03 (`rotor3d_cut_check_one_card.txt`, `rotor3d_cut_check_two_cards.txt`). No measured quantity of the pack moves by it (two slabs against one: 2e-5 U in the cavity).
- **Not run:** g = 400 dx rows other than the one pair; g = 80 dx; displaced (rule) runs in 3-D other than the one above; a particle cavity run at l = L / 40.

