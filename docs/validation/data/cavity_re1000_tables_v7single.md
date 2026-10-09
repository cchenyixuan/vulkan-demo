# n1000_k1_float32_xi0p001_eps0p0025_adami (logs/validation/cavity_re1000_v7_single)

```latex
% v7_single, Re = 1000, 1000^2, adami walls: extrema and errors against Marchi et al. 2021 Tc (wall-row frame)
% columns: quantity & SPH & Marchi 2021 & deviation (%) & std. error & position SPH & position Marchi & dx \\
$u_{\min}/U$ & -0.39042 & -0.38857 & +0.48 & 1.1e-05 & 0.1692 & 0.1717 & -0.0025 \\
$v_{\max}/U$ & 0.37928 & 0.37694 & +0.62 & 9.2e-06 & 0.1563 & 0.1578 & -0.0015 \\
$v_{\min}/U$ & -0.52946 & -0.52708 & +0.45 & 1.2e-05 & 0.9102 & 0.9092 & +0.0009 \\
$\psi_{\min}/(UL)$ & -0.11910 & -0.11894 & +0.14 & -- & (0.5304, 0.5645) & (0.5308, 0.5652) & (-0.0004, -0.0007) \\
% errors: line & rel. L2 (%) & L2 & Linf & at \\
$u(0.5, y)$ & 0.68 & 0.00215 & 0.00538 & $y = 0.1016$ \\
$v(x, 0.5)$ & 0.70 & 0.00222 & 0.00513 & $x = 0.9375$ \\
% steady state t_s = 79.38 (sliding 71.685), average window (79.99, 99.83], 99 samples, 20 snapshots; invariants drift 0 overflow 0 far_migration 0 stamp 0/0; 1 segment(s), 6.24 h
```

## Captions

**cavity_re1000_profiles.** Lid-driven cavity at Re = 1000, 1000 x 1000 fluid particles (h = 5 dx), Adami wall boundary, single GPU. Time-averaged centre-line velocity profiles over t U/L in (80.0, 99.8] (99 samples): (a) u/U along the vertical centre line x/L = 0.5, (b) v/U along the horizontal centre line y/L = 0.5. Solid line: SPH, linear MLS interpolation on 1001 points; hollow circles: the converged values T_c of Marchi et al. (2021); hollow squares: Ghia et al. (1982), for orientation only. The insets enlarge the neighbourhoods of u_min and v_min. The relative L2 error at the 28 + 28 reference points is 0.68 % (u) and 0.70 % (v).

**cavity_re1000_field.** The same run, time-averaged fields on a 513 x 513 grid (linear MLS of the particle velocities over 20 snapshots inside the averaging window). (a) Stream function psi/(UL) integrated upward from the bottom wall (contour levels of Ghia et al. 1982); the filled circle marks the SPH primary-vortex centre at (0.5304, 0.5645), the hollow star Marchi et al.'s (0.5308, 0.5652); psi_min deviates by +0.14 %. (b) Vorticity omega_z L/U (central differences of the averaged field, clipped to +-5) with contours at 0, +-0.5, +-1, +-2, +-3.

**cavity_re1000_steady (supplementary).** (a) Fluid kinetic energy against time; the shaded band is the averaging window. (b) The steady-state criterion of the run: relative change, between consecutive windows of about 10 time units, of the window-mean velocity at the 56 reference points (filled) and of the window-mean kinetic energy (hollow); the dotted line is the threshold 10^-3, the dashed line the first window end satisfying it (t_s = 79.38).

**cavity_re1000_errors (supplementary).** Pointwise error of the time-averaged SPH velocity against T_c at the 28 reference points of each centre line, (a) u and (b) v; error bars are +-2 standard errors of the time average (integrated autocorrelation time included).
