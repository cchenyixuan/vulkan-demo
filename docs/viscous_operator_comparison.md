# Discretizations of the viscous Laplacian: comparison and the `laplacian` option

2026-11-11, branch `v9-micropolar`. Host harness: `experiment/validation/viscous_operators.py` (numpy, float64),
command `micropolar_tests.py operators-host`; tables: `data/micropolar/viscous/operators_host.md` / `.json`. Solver
option: `numerics.viscous_form` (Section 6). No default was changed and no compensation was added.

Marks used below: **[run]** measured here; **[read]** inferred from the code or by algebra; citations are marked
**(checked)** when I compared the formula with the source text and **(not checked)** when it is written from
memory or from a secondary source.

## 1. Summary

1. On the **regular lattice** the Monaghan-Gingold (MG) form has the direction dependence of check C,
   `f = 0.9808 - 0.0221 cos(4 theta)` in 2-D at h/dx = 5 (0.957 to 1.030 in 3-D at h/dx = 4). Every form built on
   `v_ij (r_ij . G_ij) / r^2` (BM, BM-G, SW, FM) is isotropic there and returns exactly the EPS_H_SQUARED factor:
   0.98080 (2-D), 0.98617 (3-D), 0.99999 with `epsilon_squared_factor = 1e-6`.
2. On the **developed arrangement** (the K = 1 micropolar cavity state at t = 40.2) the anisotropy of MG is gone:
   bulk factor 0.9808 +- 0.0004 for all eight directions (1.0000 +- 0.0004 at 1e-6), the same mean as BM. In the
   wall bands (h to 3 h from a wall) MG is 0.3 % lower, 0.9777 +- 0.0011 (0.9964 +- 0.0013 at 1e-6); BM and the
   corrected forms stay at 0.9808 / 1.0000 there to four digits.
3. What differs on the developed arrangement is the **pointwise error**. For a transverse wave with 2 / 4 / 8
   waves across the cavity the rms residual relative to the signal is 33 % / 16 % / 8 % for MG, 1.3 % / 0.66 % /
   0.33 % for BM, 0.2 % / 0.1 % / 0.06 % for SW and 0.02 % for BM-G, FM and DS. The MG and BM residuals scale
   like `|grad v| / h` (they do not vanish for a linear velocity field); BM-G, FM and DS are exact for linear
   fields.
4. DS returns the continuous Laplacian for long waves without any EPS factor (0.99999) but its response falls to
   zero towards the lattice Nyquist wavenumber (0.03 at k dx = pi / 2, 0 at pi): it cannot damp particle-scale
   velocity noise. KK is 13 % off on the developed arrangement (no gradient correction).
5. MG returns 2.92 to 3.02 times the continuous value for a longitudinal wave, the others 0.98 to 1.00: MG adds
   a bulk viscosity of 2 nu to compressive motion, which the other forms do not have.
6. Smallest pair distance in the t = 40.2 state: **0.876 dx**. `r^2 >= 0.77 dx^2` against `eps^2 = 0.0625 dx^2`
   (default) or `2.5e-5 dx^2` (factor 1e-6): a much smaller EPS_H_SQUARED is safe for this state.
7. By the criterion of the task (noise lower by more than a factor of two), **BM-G is clearly better than BM** on
   the developed arrangement (factor 60 to 80 in the residual; the mean factor is the same). It was not built;
   Section 7 is the costed proposal. It needs no extra neighbour sweep, no storage and nothing from a neighbour.
8. `numerics.viscous_form: laplacian` (BM) is in the solver, default unchanged (Section 6).

For the thesis pairing argument: with either form the mean factor on a developed arrangement is the EPS factor,
0.9808 at the default, so the pairing error `(1 - f) N_c^2` is 1.2 % at K = 3 and goes to zero with a small
`epsilon_squared_factor`; with MG it stays at -0.3 % x N_c^2 in the wall bands, and wherever particles are still
on the initial lattice it follows the cos(4 theta) law.

## 2. Notation and the forms as implemented

`r_ij = r_i - r_j`, `v_ij = v_i - v_j`, `gW_ij = grad_i W_ij` (Wendland C4, support radius h),
`V_j = m_j / rho_j`, `M_i = sum_j V_j (r_j - r_i) (x) gW_ij`, `L_i = (M_i + xi_reg I)^-1` (the KGC matrix of
`correction.comp`), `G_ij = L_i gW_ij`, `eps2 = EPS_H_SQUARED`, d = dimension, `e_ij = r_ij / |r_ij|`,
`(grad v)_i = sum_j V_j (v_j - v_i) (x) G_ij` (component `[a, b] = d v_a / d x_b`). Each line is the
approximation of `lap v`; the viscous acceleration is nu times it. These are the expressions in
`viscous_operators.py`, line for line.

| tag | implemented formula | source |
|---|---|---|
| MG | `2 (d + 2) sum_j V_j (v_ij . r_ij) / (r_ij^2 + eps2) G_ij` | The code's form (`force.comp`). Monaghan & Gingold, J. Comput. Phys. 52 (1983) 374 **(not checked)**; the coefficient 2 (d + 2) and the delta-SPH usage as in Marrone et al. 2011 and the solver's paper **(not checked against the originals)**. |
| BM | `2 sum_j V_j v_ij (r_ij . G_ij) / (r_ij^2 + eps2)` | Brookshaw, Proc. Astron. Soc. Aust. 6 (1985) 207 **(not checked)**; Morris, Fox & Zhu, J. Comput. Phys. 136 (1997) 214 **(not checked)**; Cleary & Monaghan, J. Comput. Phys. 148 (1999) 227, in the form `sum_j 2 omega_j (u_i - u_j) (grad W_ij . x_ij) / r_ij^2` quoted as Eq. (51) of Negi & Ramachandran, arXiv:2107.11859v4 **(checked against that review, ar5iv HTML; not against Cleary & Monaghan)**. Differences here: the corrected gradient G_ij and eps2. |
| BM-G | BM with `v_ij -> v_ij - (grad v)_i r_ij` | Eq. (56) of Negi & Ramachandran (their "Fatehi" form before the tensor) **(checked against the review)**, here with G_ij in place of the raw gradient and with eps2. |
| DS | `sum_j V_j ((grad v)_j - (grad v)_i) G_ij` | The "coupled" form of Negi & Ramachandran, Table 8: `((grad u)_j - (grad u)_i) . grad W_ij omega_j` **(checked against the review)**. Flebbe et al., Astrophys. J. 431 (1994) 754; Watkins et al., Astron. Astrophys. Suppl. 119 (1996) 177; Bonet & Lok, Comput. Methods Appl. Mech. Engrg. 180 (1999) 97 **(none checked)**. |
| SW | `(tr(Gamma_i^-1) / d) [ 2 sum_j V_j v_ij (r_ij . gW_ij) / (r_ij^2 + eps2) - 2 (grad v)_i sum_j V_j gW_ij ]`, `Gamma_i = - sum_j V_j (r_ij . gW_ij) e_ij (x) e_ij` | Schwaiger, Int. J. Numer. Meth. Engng 73 (2008) 647, constant-coefficient case. **Not checked: I could not reach the paper; written from memory.** The review above cites it without the formula. What I can vouch for by algebra: with eps2 = 0 the bracket vanishes for a linear field on any arrangement, and Gamma -> I on a full isotropic support. The raw kernel gradient gW (not G) is used in the bracket and in Gamma, and the corrected gradient in (grad v)_i. Treat the SW columns as "a Schwaiger-type operator" until the formula is checked. |
| FM | `2 sum_j V_j (v_ij - (grad v)_i r_ij) (r_ij . Bhat_i gW_ij) / (r_ij^2 + eps2)`, Bhat_i symmetric from `[A_i^{bgt} L_i^{ta} C_i^{aem} + D_i^{bgem}] Bhat_i^{em} = -delta^{bg}`, `A^{bgt} = sum V r^b r^g gW^t`, `C^{aem} = sum V e^a e^e gW^m`, `D^{bgem} = sum V e^b e^g r^e gW^m` | Fatehi & Manzari, Comput. Math. Appl. 61 (2011) 482 **(original not checked)**; implemented from Eqs. (58) and (60) of Negi & Ramachandran **(checked against the review)**, the linear system solved per particle in the minimum-norm sense (it is symmetric in both index pairs). |
| KK | `sum_j V_j (v_j - v_i) (W'' + (d - 1) W' / r)_ij` | Second derivative of the kernel. Takeda, Miyama & Sekiya, Prog. Theor. Phys. 92 (1994) 939; Chen & Beraun, Comput. Methods Appl. Mech. Engrg. 190 (2000) 225 **(neither checked)**. Plain form, without Chen & Beraun's normalisation. |

Overviews: Negi & Ramachandran, arXiv:2107.11859 **(read: v4 on ar5iv; equations (51), (56), (58), (60), (61) and
Table 8 used)**; Basa, Quinlan & Lastiwka, Int. J. Numer. Meth. Fluids 60 (2009) 1127 and Colagrossi et al., Phys.
Rev. E 84 (2011) 026705 **(not reached, not used)**. The Español-Revenga family `alpha BM + (1 - alpha) MG` is
linear in the two.

## 3. Harness validation **[run]**

The host symbol of MG and of BM at one lattice particle against the shader's check C values (least squares over
the interior particles of a float32 block), 2-D h/dx = 5 with the wave vector along x and 3-D h/dx = 4 with the
tilted wave vector, at k h = 0.628 / 0.314 / 0.157, with the case EPS_H_SQUARED and with 0: sixteen triples,
largest difference **6.3e-6** (2-D: 1.2e-6). Example, 2-D, case eps: shader 0.955284 0.957864 0.958510, host
0.955284 0.957864 0.958511 (MG); shader 0.975644 0.979507 0.980475, host 0.975644 0.979507 0.980476 (BM = the
operator of `xi lap w`). Full table: `operators_host.md`.

## 4. Lattice **[run]** (exact symbols; k h = 0.01 for the long-wave rows)

Long transverse wave, 2-D, h/dx = 5:

| angle | MG, eps 0.0025 | MG, eps 1e-6 | BM = BM-G = SW = FM, eps 0.0025 | same, eps 1e-6 | DS | KK |
|---|---|---|---|---|---|---|
| 0 | 0.95873 | 0.97472 | 0.98080 | 0.99999 | 0.99999 | 1.00056 |
| 11.25 | 0.96519 | 0.98212 | 0.98080 | 0.99999 | 0.99999 | 1.00056 |
| 22.5 | 0.98080 | 0.99999 | 0.98080 | 0.99999 | 0.99999 | 1.00056 |
| 33.75 | 0.99641 | 1.01786 | 0.98080 | 0.99999 | 0.99999 | 1.00056 |
| 45 | 1.00287 | 1.02526 | 0.98080 | 0.99999 | 0.99999 | 1.00056 |

MG follows `0.98080 - 0.02207 cos(4 theta)` (eps 0.0025) and `0.99999 - 0.02527 cos(4 theta)` (eps 1e-6) to five
digits; its angle mean is the BM value.

Long transverse wave, 3-D, h/dx = 4:

| wave vector, polarisation | MG, eps 0.0025 | MG, eps 1e-6 | BM = BM-G = SW = FM, eps 0.0025 | same, eps 1e-6 | DS | KK |
|---|---|---|---|---|---|---|
| [100], v along [010] | 0.95706 | 0.96861 | 0.98617 | 0.99999 | 0.99999 | 0.99935 |
| [110], v along [1-10] | 1.02983 | 1.04707 | 0.98617 | 0.99999 | 0.99999 | 0.99935 |
| [110], v along [001] | 0.95706 | 0.96861 | 0.98617 | 0.99999 | 0.99999 | 0.99935 |
| [111], v along [1-10] | 1.00558 | 1.02091 | 0.98617 | 0.99999 | 0.99999 | 0.99935 |

In 3-D the MG factor depends on the polarisation as well as on the wave vector (the two transverse waves along
[110] differ by 7 %).

Long longitudinal wave (v parallel to k), eps 0.0025 (eps 1e-6 in brackets):

| case | MG | BM = BM-G = SW = FM | DS | KK |
|---|---|---|---|---|
| 2-D, axis | 2.96447 (3.02524) | 0.98080 (0.99999) | 0.99999 | 1.00056 |
| 2-D, diagonal | 2.92032 (2.97470) | 0.98080 (0.99999) | 0.99999 | 1.00056 |
| 3-D, [100] | 3.01672 (3.06274) | 0.98617 (0.99999) | 0.99999 | 0.99935 |
| 3-D, [111] | 2.91969 (2.95814) | 0.98617 (0.99999) | 0.99999 | 0.99935 |

Whole spectrum, transverse wave, 2-D, eps 0.0025 (ratio to `-k^2`; k dx / pi = 1 is the Nyquist wavenumber along
an axis):

| k dx / pi | MG axis | MG diagonal | BM (= BM-G, SW, FM) axis | diagonal | DS axis | KK axis |
|---|---|---|---|---|---|---|
| 0.05 | 0.9534 | 0.9975 | 0.9728 | 0.9728 | 0.9678 | 0.9843 |
| 0.1 | 0.9375 | 0.9817 | 0.9491 | 0.9491 | 0.8771 | 0.9370 |
| 0.2 | 0.8775 | 0.9217 | 0.8615 | 0.8614 | 0.5897 | 0.7683 |
| 0.3 | 0.7888 | 0.8331 | 0.7370 | 0.7370 | 0.3006 | 0.5488 |
| 0.4 | 0.6846 | 0.7290 | 0.5997 | 0.5997 | 0.1140 | 0.3385 |
| 0.5 | 0.5779 | 0.6224 | 0.4703 | 0.4703 | 0.0313 | 0.1781 |
| 0.6 | 0.4785 | 0.5230 | 0.3617 | 0.3616 | 0.0060 | 0.0790 |
| 0.8 | 0.3199 | 0.3646 | 0.2160 | 0.2159 | 0.0001 | 0.0114 |
| 1.0 | 0.2138 | 0.2592 | 0.1391 | 0.1392 | 0.0000 | 0.0046 |

- All forms lose response towards short waves (a kernel-based second difference at h/dx = 5); in absolute terms
  the damping rate `ratio x k^2` still grows with k for MG and BM up to the zone boundary (BM: 0.139 x pi^2 / dx^2
  at Nyquist against 0.470 x (pi / 2)^2 / dx^2 at half of it, i.e. 1.18 times larger).
- DS is effectively zero from k dx / pi = 0.6 on, as expected of a repeated first derivative.
- BM damps the shortest transverse waves at 0.65 (axis) to 0.54 (diagonal) of MG's rate.
- On the lattice BM-G, SW and FM coincide with BM (the gradient-subtraction terms vanish by symmetry and
  Bhat = I, Gamma = I up to eps).

## 5. Developed arrangement **[run]**

The K = 1 micropolar cavity state at t = 40.2 (`data/micropolar/checks/d/k1_t40.npz`): positions and densities of
all 74,529 particles, velocity replaced by `e exp(i k . r)` with e perpendicular to k on every particle. Evaluated
on 48,391 bulk fluid particles (more than 3 h from every wall) and on 2,373 to 2,389 particles per wall band
(h to 3 h from one wall, more than h from the others, so no wall particle is in a support). KGC regularisation 0
here (the solver's 0.001 multiplies every form that uses G by 0.9991, measured in check C). Factor = least-squares
projection onto the expected mode; intercept at k -> 0 of a line in k^2 through 2, 4 and 8 waves across the
cavity; uncertainty = standard error over the four quadrants (bulk) or the four walls (bands).

Mean factor at k -> 0:

| region, wave vector | MG, eps 0.0025 | MG, eps 1e-6 | BM, eps 0.0025 | BM, eps 1e-6 | BM-G / SW / FM | DS | KK |
|---|---|---|---|---|---|---|---|
| bulk, 0 deg | 0.9811 +- 0.0004 | 1.0003 +- 0.0005 | 0.9808 +- 0.0000 | 1.0000 +- 0.0000 | as BM | 0.9998 | 1.1300 +- 0.0008 |
| bulk, 22.5 deg | 0.9812 +- 0.0008 | 1.0004 +- 0.0009 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1301 |
| bulk, 45 deg | 0.9805 +- 0.0004 | 0.9996 +- 0.0005 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1301 |
| bulk, 67.5 deg | 0.9804 +- 0.0008 | 0.9995 +- 0.0009 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1302 |
| bulk, 90 deg | 0.9811 +- 0.0004 | 1.0003 +- 0.0005 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1301 |
| bulk, 112.5 deg | 0.9812 +- 0.0008 | 1.0004 +- 0.0009 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1301 |
| bulk, 135 deg | 0.9805 +- 0.0004 | 0.9996 +- 0.0005 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1300 |
| bulk, 157.5 deg | 0.9804 +- 0.0008 | 0.9996 +- 0.0009 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1300 |
| bulk, mean of 8 (std over directions) | 0.9808 (0.0004) | 1.0000 (0.0004) | 0.9808 (0.0000) | 1.0000 (0.0000) | as BM | 0.9998 | 1.1301 |
| wall bands, normal | 0.9777 +- 0.0011 | 0.9964 +- 0.0013 | 0.9808 +- 0.0000 | 1.0000 +- 0.0000 | as BM | 0.9998 | 1.1301 +- 0.0010 |
| wall bands, parallel | 0.9777 +- 0.0011 | 0.9964 +- 0.0013 | 0.9808 | 1.0000 | as BM | 0.9998 | 1.1300 +- 0.0011 |

- **MG has no measurable direction dependence in the bulk** of this state: a cos(4 theta) amplitude of at most
  0.0004, against 0.0221 on the lattice. The 0 / 90 and 45 / 135 pairs are equal by construction of a transverse
  wave (the fourth moment that enters is symmetric in the wave direction and the polarisation), so the eight
  directions are four independent values; for the same reason "normal" and "parallel" in the wall bands are the
  same number.
- **MG in the wall bands is 0.31 % (eps 0.0025) to 0.36 % (eps 1e-6) below the bulk value**, 2.7 to 2.8 standard
  errors. With four walls as the only blocks this is an indication, not a tight number.
- BM, BM-G, SW and FM have the mean factor of the lattice to four digits everywhere: the second-moment argument
  of the task (the KGC condition fixes the trace on any arrangement) holds on the developed state.
- The fit in k^2 is needed: at 8 waves (k h = 1.0) the raw factors are 0.972 (MG) and 0.968 (BM), at 2 waves
  0.980 for both.

Noise: rms of the residual after the projection, relative to the rms of the signal, at 2 / 4 / 8 waves across the
cavity, eps 0.0025 (bulk, mean of the eight directions; wall bands in brackets are 20 to 25 % higher throughout):

| form | 2 waves | 4 waves | 8 waves | at eps 1e-6 (2 / 4 / 8) |
|---|---|---|---|---|
| MG | 3.3e-1 | 1.6e-1 | 8.2e-2 | 3.7e-1 / 1.9e-1 / 9.3e-2 |
| BM | 1.3e-2 | 6.6e-3 | 3.3e-3 | 1.2e-2 / 6.2e-3 / 3.1e-3 |
| SW | 2.0e-3 | 1.1e-3 | 6.5e-4 | 4.6e-4 / 4.7e-4 / 4.8e-4 |
| BM-G | 1.7e-4 | 1.8e-4 | 2.1e-4 | 2.3e-4 / 2.4e-4 / 2.7e-4 |
| FM | 1.4e-4 | 1.5e-4 | 1.8e-4 | 3.2e-5 / 6.4e-5 / 1.3e-4 |
| DS | 1.2e-4 | 1.2e-4 | 1.5e-4 | (no denominator) |
| KK | 2.4e-2 | 1.2e-2 | 6.2e-3 | (no denominator) |

The MG, BM and KK residuals halve when the wavelength halves: an error proportional to `|grad v| / h`, i.e. a
first-order inconsistency on disordered particles, which the exactness test shows directly:

Exactness, rms `|lap_h v|` in units of `|grad v| / h`, eps 0.0025, bulk (wall bands):

| field | MG | BM | SW | BM-G | FM | DS | KK |
|---|---|---|---|---|---|---|---|
| linear, `v = A r` | 3.1e-2 (3.9e-2) | 3.3e-3 (4.2e-3) | 5.0e-4 (6.2e-4) | 2e-14 | 2e-14 | 2e-15 | 6.1e-3 (8.0e-3) |
| rigid rotation | 0 (0) | 3.3e-3 (4.2e-3) | 5.0e-4 (6.2e-4) | 5e-17 | 5e-17 | 3e-16 | 6.1e-3 (7.9e-3) |

- MG is exactly zero for the rigid rotation (each pair term is) but has the largest error of all for a general
  linear field: for a shear `|grad v| = U / L` the pointwise MG error equals the continuous viscous term
  `nu U / L^2` already at `L = h / 0.031 = 32 h`. It is a zero-mean error (it does not move the factor above).
- BM is ten times smaller, BM-G / FM / DS are exact.

**Which gradient the exactness row used, and why 3.1e-2 does not give the 33 % of the noise table.** The row's
linear field was `v = A r` with `A = [[0.3, 0.7], [-0.4, 0.5]]` and the unit was `|A|_F / h` with the Frobenius
norm (0.995). For BM the error is `2 A b_i` with a geometric vector b_i, so it depends on `|A|` only: 3.33e-3 for
every field below, which is why the BM numbers agree both ways (3.33e-3 / (k h) = 1.3 % at two waves,
k h = 0.251). For MG the pair term sees only the strain part of A (`v_ij . r_ij = r_ij^T A r_ij`), and the error
differs by the kind of strain. Measured on the same state (rms `|lap_h v| h / |A|_F`, bulk,
`data/micropolar/viscous/exactness_by_field.json`):

| linear field | MG | BM |
|---|---|---|
| rigid rotation | 0 | 3.33e-3 |
| isotropic expansion, A = I | 1.33e-2 | 3.33e-3 |
| pure strain, [[1, 0], [0, -1]] or [[0, 1], [1, 0]] | 1.16e-1 | 3.33e-3 |
| simple shear, [[0, 1], [0, 0]] (the local gradient of a transverse wave) | 8.23e-2 | 3.33e-3 |
| the row above, [[0.3, 0.7], [-0.4, 0.5]] (mostly rotation and expansion) | 3.08e-2 | 3.33e-3 |

A transverse wave is locally a simple shear, so its MG residual is 8.23e-2 / (k h) = 32.7 % at two waves, the
measured 33 %. The 3.1e-2 of the exactness row is the value for a field whose gradient is 78 % rotation and 57 %
expansion by norm and only 26 % deviatoric strain; it understates MG for shear flows by a factor 2.7. For a shear
`|grad v| = U / L` the pointwise MG error equals the continuous viscous term at `L = h / 0.082 = 12 h` (not the
32 h stated above for the mixed field).

Smallest pair distance in the state: 0.8761 dx.

## 6. The `laplacian` option in the solver

`numerics.viscous_form` in `case.yaml`: `monaghan_gingold` (default; today's code) or `laplacian` (BM with the
same neighbour velocity as before: the dummy velocity for an adami wall neighbour). Works with micropolar on or
off.

- Implementation: `#ifdef VISCOUS_LAPLACIAN ... #else <the old statement> #endif` in `force.comp`; two more
  variants, `force_laplacian` and `force_micropolar_laplacian`. No spec constant, no buffer, no change of a
  recording: the simulator only picks another force module (`_shader_module_file_name`). Refused with
  `V7_BAND_OVERLAP` (no FORCE_SEGMENT variant).
- **T0 [run]:** all 33 earlier SPIR-V files byte-identical (the 23 of `micropolar-base` among them), 2 new,
  `MANIFEST.txt` regenerated (35); `_test_seam_layout.py` 858 failures, the base's set; `check_micropolar` extended
  (laplacian = another force module only; buffers, spec entries, segments unchanged).
- **T1 [run]:** default form against `micropolar-base`, n = 250 adami cavity, 3,000 steps: K = 1, five runs
  against ten base repeats, ratios 0.84 to 1.30 (u), 0.85 to 1.23 (v), 0.93 to 1.00 (rho); K = 2 (5090 + XTX),
  five runs against forty, 0.61 to 0.71, 0.67 to 0.80, 0.62 to 0.83.
- **Shader against host [run]** (`viscous-cross-check`; one force evaluation with uniform rho_0, v = e sin(k . r),
  4 waves at 22.5 degrees; host with the float32 inputs, the case's KGC regularisation and EPS): rms
  `|shader - host| / rms(host)` on the lattice block 1.4e-6 (`monaghan_gingold` against MG) and 7.6e-7
  (`laplacian` against BM), 7,569 particles; on the positions of the t = 40.2 state 1.5e-6 and 9.1e-7, 56,605
  particles; maxima 4e-6 to 7e-6.
- **Seam [run]** with `laplacian`, the per-particle check on the prescribed state, both wall types: K = 2 on
  5090 + XTX and on one card, K = 3 on one card (0,0,0). After the bootstrap force the acceleration differs from
  K = 1 by 0.8e-7 to 1.3e-7 rms near the cuts (K = 1 repeat 0.8e-7); after 20 steps, one card: 0.9e-5 to 1.5e-5
  against 1.0e-5 to 1.6e-5 for the K = 1 repeat (K = 2), 1.7e-5 to 1.8e-5 against 1.2e-5 to 1.9e-5 (K = 3). No
  change against the default form, as expected: the force sweep reads the same fields. **[read]** Band widths and
  transport are untouched.

What the `laplacian` form gives up against the default:

- **No `grad div v` part.** MG approximates `nu (lap v + 2 grad div v)`: a bulk viscosity that damps compressive
  (acoustic) motion three times as strongly (Section 4, longitudinal row). With `laplacian` acoustic noise is
  damped by the shear viscosity and the delta term only.
- **Not identically zero for a rigid rotation on disordered particles**: 3.3e-3 `|grad v| / h` rms (Section 5);
  MG is exactly zero.
- **No pairwise angular-momentum conservation.** The BM pair force is along `v_ij`, not along `r_ij`. (Neither
  form is exactly pairwise symmetric in this solver, because `G_ij = L_i gW_ij` uses the correction matrix of
  particle i only.)
- Weaker damping at the particle scale: 0.54 to 0.65 of MG's rate for the shortest transverse waves.

## 7. Proposal, not built: gradient subtraction (BM-G)

By the task's criterion a corrected scheme is "clearly better" than BM on the developed arrangement: same mean
factor, residual 60 to 80 times smaller (BM-G, FM), exact for linear fields. BM-G is the cheapest of them and FM
adds nothing measurable at eps 0.0025 (its Bhat is the identity to within the noise here; at eps 1e-6 FM is up
to 7 times quieter than BM-G at long waves, both below 3e-4).

Cost of each candidate in this solver **[read]**:

| form | extra neighbour sweeps | extra bytes per particle | reads a derived quantity of the neighbour | seam |
|---|---|---|---|---|
| BM (`laplacian`) | 0 | 0 | no | none (done) |
| BM-G | 0 | 0 | no | none |
| SW | 0 | 0 | no | none |
| FM | 0 | 0 | no (but 27 + 81 moment components in 3-D and a 9 x 9 solve per particle inside the force sweep) | none |
| DS | 1, or the gradient accumulated in the correction / density sweep and stored | 36 B (2-D: 16 B) for (grad v)_i, defrag-free but transient | **yes, (grad v)_j** | the gradient of G1 has to exist on the receiver: computed there as self (needs v of G2, present) or transported; the force band grows by one column because column c reads the gradient of column c - 1, which reads the velocity of c - 2 after migration |
| KK | 0 | 0 | no | none; not recommended (13 % off without a normalisation) |

Why BM-G needs no extra sweep: with `b_i = sum_j V_j r_ij (r_ij . G_ij) / (r_ij^2 + eps2)` (geometry only) and
`S_i = sum_j V_j (v~_j - v_i) (x) G_ij` (the velocity gradient with the same neighbour velocity as the viscous
term),

```
BM-G_i = BM_i - 2 S_i b_i
```

so the force sweep accumulates the BM sum, the vector b_i (3 floats) and the tensor S_i (9 floats; 4 in 2-D) in
the same pass and combines them after the loop. Nothing is stored, nothing of a neighbour's derived state is
read, the bands and the transport do not change, and the adami dummy velocity enters S_i exactly as it enters
the viscous term. The micropolar force variant already accumulates `curl v` from the same differences.

- Shader: a third value of the macro in `force.comp` (`force_laplacian_corrected`, `force_micropolar_...`),
  about 15 lines; host: one more `numerics.viscous_form` value. Tests: T0, the host cross-check (the host form
  exists), the seam check, the exactness rows of Section 5 on the GPU.
- Expected cost: 12 more accumulators and one 3 x 3 product per particle in the force sweep; no measurable
  memory traffic. I did not measure it.
- What it gives up, beyond the list of Section 6: the correction term is not a pair force at all, so even the
  approximate pairwise momentum balance of BM is gone (FM has the same property; the review calls it "accurate
  and convergent but non-conservative"). Near a wall or a free surface (incomplete support) `b_i` is not small
  and the result depends on the quality of S_i there; the measurements above exclude supports that contain wall
  particles, so the near-wall behaviour is untested.
- SW would be the alternative that keeps a symmetric main term with a normalisation; its formula has to be
  checked against the paper first (Section 2).

## 8. The cavity with both forms **[run]**

Newtonian adami cavity, Re = 1000 (`cases/lid_driven_cavity_2d_n{250,500}_xi0p001_eps0p0025_adami` and their
variants `cases/viscous_cavity_n{250,500}_laplacian[_eps1em6]`), K = 1 on the 5090, `cavity_runner.py --solver v7
--expect release` from rest; driver and analysis `experiment/validation/viscous_form_campaign.py`. Errors against
Marchi et al. 2021 at the 28 + 28 reference points (the runner's MLS samples, wall frame), mean over the last 20
time units; all runs passed the runner's steady test (at t = 80.6 / 79.6; the n = 500 EPS 1e-6 run at 69.6) and its
invariant checks (drift, overflow, stamps) at every sample. In units of U; E_k in the runner's units; the density
range is max - min over the fluid in kg/m^3 (rho_0 = 1000).

| n | form | t_end | L2 u | Linf u | L2 v | Linf v | E_k | std E_k | density range, mean (max) | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 250 | monaghan_gingold | 100.6 | 1.10e-2 | 2.30e-2 | 1.22e-2 | 1.93e-2 | 53.772 | 5.3e-3 | 0.686 (0.843) | 3,253 |
| 250 | laplacian | 100.6 | 9.58e-3 | 1.91e-2 | 1.13e-2 | 1.78e-2 | 53.431 | 1.1e-2 | 0.572 (0.739) | 3,244 |
| 250 | laplacian, eps factor 1e-6 | 100.6 | 9.34e-3 | 2.13e-2 | 1.05e-2 | 1.69e-2 | 53.300 | 6.0e-3 | 0.584 (0.757) | 3,241 |
| 500 | monaghan_gingold | 99.5 | 5.04e-3 | 1.08e-2 | 5.56e-3 | 1.05e-2 | 51.705 | 4.8e-3 | 0.929 (1.117) | 1,736 |
| 500 | laplacian | 99.5 | 4.68e-3 | 9.93e-3 | 5.36e-3 | 9.46e-3 | 51.659 | 5.2e-3 | 0.800 (0.969) | 1,726 |
| 500 | laplacian, eps factor 1e-6 | 89.5 | 4.18e-3 | 1.09e-2 | 4.29e-3 | 7.26e-3 | 51.533 | 6.9e-3 | 0.834 (0.996) | 1,729 |

Difference to the monaghan_gingold run at the 56 reference points, and the kinetic energy ratio:

| n | form | rms u | max u | rms v | max v | E_k ratio |
|---|---|---|---|---|---|---|
| 250 | laplacian | 1.65e-3 | 3.89e-3 | 1.02e-3 | 2.17e-3 | 0.99365 |
| 250 | laplacian, eps factor 1e-6 | 2.55e-3 | 6.24e-3 | 3.06e-3 | 6.71e-3 | 0.99122 |
| 500 | laplacian | 5.2e-4 | 1.01e-3 | 5.0e-4 | 1.07e-3 | 0.99912 |
| 500 | laplacian, eps factor 1e-6 | 2.09e-3 | 5.03e-3 | 2.57e-3 | 5.75e-3 | 0.99669 |

Kinetic energy history (mean over +-2.5 time units around t):

| n | form | t = 5 | 10 | 20 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|---|
| 250 | monaghan_gingold | 23.073 | 34.742 | 49.033 | 53.391 | 53.705 | 53.763 |
| 250 | laplacian | 22.897 | 34.497 | 48.715 | 53.067 | 53.362 | 53.418 |
| 250 | laplacian, 1e-6 | 23.046 | 34.675 | 48.754 | 52.965 | 53.237 | 53.296 |
| 500 | monaghan_gingold | 22.472 | 33.684 | 47.127 | 51.345 | 51.630 | 51.694 |
| 500 | laplacian | 22.459 | 33.680 | 47.110 | 51.307 | 51.590 | 51.650 |
| 500 | laplacian, 1e-6 | 22.605 | 33.856 | 47.161 | 51.219 | 51.483 | 51.535 |

- **Stable in every run**, both resolutions, including the 1e-6 denominator (smallest pair distance 0.88 dx).
- **No extra acoustic noise with `laplacian`.** The fluid density range is 14 to 17 % smaller than with MG over
  the averaging window and during the start-up (first five time units: 0.60 against 0.67 at n = 250, 0.83 against
  0.93 at n = 500). The missing `grad div v` damping does not show in this measure; the kinetic energy scatter of
  the n = 250 laplacian run is twice MG's (1.1e-2 against 5.3e-3), at n = 500 the two are equal.
- **The forms agree better with resolution**: the profiles differ by 1.0e-3 to 1.7e-3 rms at n = 250 and 5e-4 at
  n = 500, E_k by 0.63 % and 0.09 %. `laplacian` is slightly closer to Marchi et al. at both resolutions (L2 u
  -13 % / -7 %, L2 v -7 % / -4 %).
- **The EPS factor is a viscosity change of the size measured in Section 5.** Removing it (0.9808 -> 1.0000 on
  the operator, +2 % effective viscosity) lowers E_k by a further 0.24 % at both resolutions and moves the
  profiles by about 2e-3 to 3e-3 rms, more than the change of form at n = 500. It reduces the error against
  Marchi et al. (n = 500: L2 u 4.68e-3 -> 4.18e-3, L2 v 5.36e-3 -> 4.29e-3). Note that
  `epsilon_squared_factor` also enters the delta-term denominator of the continuity equation, so this row
  changes the density diffusion as well.
- I did not use `effective_viscosity.py` / `reynolds_sensitivity.py`: they are written around the run names and
  the tracked output files of `docs/validation/`, and running them would have rewritten those tables. No effective
  Reynolds number is given; the factor measured on the operator (Section 5) is the direct number.

### The micropolar T5 configuration with `laplacian`

One K = 1 continuation of the t = 40.2 state with `numerics.viscous_form: laplacian`, 200,000 steps (6 time
units), first unit discarded, next five averaged, against the ten `monaghan_gingold` K = 1 continuations of check D:
stable, counters zero.

| quantity | MG, mean of ten (std) | laplacian | difference |
|---|---|---|---|
| min u on x = 0 | -0.403512 (5.3e-5) | -0.403258 | +2.5e-4 |
| min v on y = 0 | -0.525930 (3.2e-5) | -0.525622 | +3.1e-4 |
| max v on y = 0 | 0.383181 (3.0e-5) | 0.382872 | -3.1e-4 |
| min w on x = 0 | -2.107924 (1.4e-4) | -2.100409 | +7.5e-3 (0.36 %) |
| max w on x = 0 | 0.816694 (6.9e-4) | 0.813809 | -2.9e-3 |
| min w on y = 0 | -1.646905 (1.4e-4) | -1.646637 | +2.7e-4 |
| max w on y = 0 | 1.767781 (3.5e-4) | 1.765439 | -2.3e-3 |
| L2 of the profile difference | (4.4e-5 to 3.2e-4 among the MG runs) | u 6.5e-4, v 3.5e-4, w 2.5e-3 and 2.8e-3 | |

This is the drift of the first six time units after the switch, not a new steady state: the profiles were still
moving. It shows the direction and that the change is small (below 0.4 % in every extremum).
