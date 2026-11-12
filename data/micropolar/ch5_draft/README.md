# Evidence pack for the first draft of the chapter (Stage 6)

What the solver has shown, in the form that goes on the page: figures, tables, numbers with their sources.
Design note: `docs/micropolar_design.md`, Section 15. Run data of Stage 6: `data/micropolar/stage6/`.

## How to read the index

- **final**: made from runs on the tag `thesis-ch5-rc2` (Stage 6). An item that is final but still lacks rows
  says so in its caption.
- **provisional**: made from older runs; the column "data" and the caption say which. The item is regenerated
  when its rc2 runs exist; the swap is one line in `sources.py`.
- **not reached**: no file yet.

## Files

- `fig_<nn>_<name>.pdf` (vector; particle fields rasterised at 300 dpi inside it) and `.png`.
- `tab_<nn>_<name>.csv` and `.tex` (a `booktabs` tabular without float and caption).
- `fig_<nn>_<name>.py`, `tab_<nn>_<name>.py`: one script per item, run from the repository root.
- `captions.md`: the caption of every item with its parameters, run names, tag and key numbers.
- `items/<nn>.json`: what each script recorded (caption, status, files, sources, key numbers).
- `style.py`: the one style module (sizes, fonts, line conventions, the symbols of the axis labels).
- `sources.py`: where each item reads its runs.
- `make_all.py`: runs every item script and writes `captions.md` and the index below.

## Conventions

- matplotlib only. Width 150 mm or 72 mm, 9 pt, serif with Computer Modern mathtext, no titles, panel letters.
- Exact solutions: black solid lines. Runs: markers. The wall rule: dashed. The film function of micropolar
  lubrication theory: dash-dotted. Fields: `viridis` for magnitudes, `RdBu_r` centred on zero for signed
  quantities, equal aspect, labelled colour bar. Line plots are readable in grey (marker shape and line style
  carry the series).
- Units: lengths in g, velocities in U = Omega R1, micro-rotation in Omega, pressure in rho U^2; cavity: L, U,
  U / L. The symbol of the gap is `style.GAP`.
- Reynolds numbers: on `mu_inf` (the far viscosity `mu + kappa / 2`, the one the cases are set by) and on `mu`
  (`Re_g (1 + K / 2)`, the one the thesis quotes).

## Index (written 2026-11-12 18:00)

| no. | item | files | status | data |
|---|---|---|---|---|
| 01 | Concentric cylinders: profiles against the exact solution | `fig_01_concentric_profiles.pdf`, `fig_01_concentric_profiles.png` | provisional | thesis-ch5-rc1 |
| 02 | Concentric cylinders: convergence of D and of the profiles | `fig_02_concentric_convergence.pdf`, `fig_02_concentric_convergence.png` | provisional | thesis-ch5-rc1 |
| 03 | Concentric cylinders: D against m for three K, with the slip-form rule | `fig_03_concentric_sweep.pdf`, `fig_03_concentric_sweep.png` | provisional | thesis-ch5-rc1 |
| 04 | Table of every concentric row (and of the controls) | `tab_04_concentric_rows.csv`, `tab_04_concentric_rows.tex`, `tab_04b_concentric_controls.csv`, `tab_04b_concentric_controls.tex` | provisional | thesis-ch5-rc1 |
| 05 | Table: Newtonian eccentric flow against the exact Stokes values | `tab_05_eccentric_newtonian.csv`, `tab_05_eccentric_newtonian.tex` | provisional | thesis-ch5-rc1 and the commits of Stage 5 |
| 06 | Eccentric cylinders: wall pressure around the rotor | `fig_06_eccentric_wall_pressure.pdf`, `fig_06_eccentric_wall_pressure.png` | provisional | thesis-ch5-rc1 and the commits of Stage 5 |
| 07 | Eccentric fields: pressure, speed, micro-rotation, w - curl v / 2 | `fig_07_eccentric_fields.pdf`, `fig_07_eccentric_fields.png` | provisional | thesis-ch5-rc1 and the commits of Stage 4 |
| 08 | Eccentric cylinders: measured over predicted change against narrowest gap / l | `fig_08_eccentric_prediction.pdf`, `fig_08_eccentric_prediction.png` | provisional | thesis-ch5-rc1 and the commits of Stage 5; concentric: thesis-ch5-rc1 |
| 09 | Table of every eccentric micropolar row | `tab_09_eccentric_rows.csv`, `tab_09_eccentric_rows.tex` | provisional | thesis-ch5-rc1 and the commits of Stage 5 |
| 10 | The film function over h^3 against h / l at K = 3 | `fig_10_film_function.pdf`, `fig_10_film_function.png` | final (host only) | host |
| 11 | Micropolar cavity: centre lines with the two Newtonian bounds, fields, the slab cut | - | not reached | - |
| 12 | Newtonian cavity baseline at n = 1000 against Marchi et al. | `fig_12_cavity_baseline.pdf`, `fig_12_cavity_baseline.png` | final (existing data) | Stage 3 (design note 12.8), `micropolar-base-13-g7e76dcb` |
| 13 | Seepage of the thesis walls: velocity into the wall against wall pressure; the wall candidates | `fig_13_seepage.pdf`, `fig_13_seepage.png`, `tab_13_candidates.csv`, `tab_13_candidates.tex` | final (existing data) | Stage 5, `thesis-ch5-rc1-6-g5d07112`, `thesis-ch5-rc1-9-gbfcc024`; cavity `thesis-ch5-rc1-10-g5bc8e0d`, `thesis-ch5-rc1-9-gbfcc024`, `micropolar-base-12-gc08cb97` |
| 14 | Solver checks: operator checks, passive label, chain against one slab, step limits, cost | `tab_14a_operator_checks.csv`, `tab_14a_operator_checks.tex`, `tab_14b_passive_label.csv`, `tab_14b_passive_label.tex`, `tab_14c_chain_against_one_slab.csv`, `tab_14c_chain_against_one_slab.tex`, `tab_14d_step_limits.csv`, `tab_14d_step_limits.tex`, `tab_14e_cost.csv`, `tab_14e_cost.tex` | final (existing data) | Stages 1 to 3, `micropolar-base-9-g278a557` to `thesis-ch5-rc1` |
| 15 | 3-D: fields in a plane through the axis and the mid-plane profiles | - | not reached | - |
