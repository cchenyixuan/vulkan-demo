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

## Index (written 2026-11-12 17:23)

| no. | item | files | status | data |
|---|---|---|---|---|
| 01 | Concentric cylinders: profiles against the exact solution | `fig_01_concentric_profiles.pdf`, `fig_01_concentric_profiles.png` | provisional | thesis-ch5-rc1 |
| 02 | Concentric cylinders: convergence of D and of the profiles | `fig_02_concentric_convergence.pdf`, `fig_02_concentric_convergence.png` | provisional | thesis-ch5-rc1 |
| 03 | Concentric cylinders: D against m for three K, with the slip-form rule | `fig_03_concentric_sweep.pdf`, `fig_03_concentric_sweep.png` | provisional | thesis-ch5-rc1 |
| 04 | Table of every concentric row (and of the controls) | `tab_04_concentric_rows.csv`, `tab_04_concentric_rows.tex`, `tab_04b_concentric_controls.csv`, `tab_04b_concentric_controls.tex` | provisional | thesis-ch5-rc1 |
| 05 | Table: Newtonian flow between eccentric cylinders against the exact Stokes values | - | not reached | - |
| 06 | Eccentric: wall pressure around the rotor against the Stokes solution, with the micropolar runs | - | not reached | - |
| 07 | Eccentric fields, eps = 0.6, K = 3, l = 10 dx: pressure, speed, micro-rotation, w - curl v / 2 | - | not reached | - |
| 08 | Eccentric: measured over predicted change against narrowest gap / l | - | not reached | - |
| 09 | Table of every eccentric micropolar row | - | not reached | - |
| 10 | The film function over h^3 against h / l at K = 3 | - | not reached | - |
| 11 | Micropolar cavity: centre lines with the two Newtonian bounds, fields, the slab cut | - | not reached | - |
| 12 | Newtonian cavity baseline at n = 1000 against Marchi et al. | - | not reached | - |
| 13 | Seepage: velocity into the wall against wall pressure; candidates table | - | not reached | - |
| 14 | Tables of Sections 8 to 12: operator checks, passive label, K = 2 against K = 1, step limits, cost | - | not reached | - |
| 15 | 3-D: fields in a plane through the axis and the mid-plane profiles | - | not reached | - |
