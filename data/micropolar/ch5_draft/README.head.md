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
