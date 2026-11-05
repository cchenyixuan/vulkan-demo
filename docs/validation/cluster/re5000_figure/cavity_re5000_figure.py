"""cavity_re5000_figure.py (2026-11-06) - paper fig:cavity5000: the Re = 5000 lid-driven cavity on the cross-vendor pair,
drawn by the code of fig:cavity / fig:cavity3200 (experiment/validation/cavity_re1000_figure.py, the module that
cavity_re3200_figure.py also wraps): same width (elsarticle single column, 384 pt), panel geometry, fonts, line styles,
colours, legend, zoom-box positions, colour scale and stream-function levels. Uses existing data only: nothing is run.

Run: n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000 (RTX 5090 + RX 7900 XTX, 2500^2 fluid particles, one cut at
column 341, unequal slabs), continued from the 2x5090 run at t = 100.5. Window = its own samples after the start,
t = 100.701 .. 103.314 (14 samples, steps 33,567,000 .. 34,438,000), the window of tab:cavity5000 and of
re5000_mixed/report.md section 4. As in the Re 1000 / 3200 figures:
  (a) u(0.5, y), (b) v(x, 0.5): base.k1_lines_on_window = the mean of the 14 samples' dense MLS centre lines (1001
      points, wall-row frame) against Marchi 2021 (Table 27 run's profiles, Tables 28 / 29) and Ghia 1982 (Re 5000);
      zooms near u_min and v_min in the boxes of the Re 1000 figure, limits around the Re 5000 extrema;
  (c) base.mean_field + field_quantities: the time-averaged speed of ALL snapshots in the window (Re 1000: 11,
      Re 3200: 5; here the 3 at t 100.899 / 101.898 / 102.897), linear MLS on 1025^2 nodes spanning the innermost wall
      rows, streamlines from the Poisson stream function; the seam is base.run_facts' conversion of the cut column:
      X = (grid_origin_x + 341 * support + 0.5 + dx) / (1 + 2 dx), drawn as the white dashed line of the K = 8 figures.
Only differences from the Re 3200 call: the data, the single seam, the zoom limits and the legend label of the line
(line_label, a defaulted keyword added to the base module for this figure; the Re 1000 / 3200 figures keep K = 1).

Self-check: relative L2 of the drawn centre lines at the 28 + 28 Marchi points (cubic spline along the dense line, as
cavity_analysis does for its dense-line errors) and of the samples' own Marchi-point MLS values on the same steps;
must reproduce tab:cavity5000's 0.281 / 0.370 %.

Outputs: cavity_re5000.pdf / .png next to this script and copied to vulkan-demo/manuscripts/fig/; the centre lines
(dense_<run>_common_window.csv) and check.json next to this script; the mean-field cache in
logs/validation/cavity_re5000_v7/fig_cavity5000/ (git-ignored).

    PYTHONUTF8=1 .venv/Scripts/python.exe docs/validation/cluster/re5000_figure/cavity_re5000_figure.py [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.validation import cavity_re1000_figure as base, cavity_reference  # noqa: E402
from experiment.validation.cavity_analysis import load_run  # noqa: E402

RUNS_ROOT = _ROOT / "logs" / "validation" / "cavity_re5000_v7"
RUN_ID = "n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000"
FIRST_STEP, LAST_STEP, SAMPLE_STEPS = 33_567_000, 34_438_000, 67_000
REYNOLDS = 5000
LINE_LABEL = r"SPH, $K = 2$"          # the 5090 + XTX pair, named as the K = 2 column of Table 4 (also 5090 + XTX)
OUT_DIR = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = RUNS_ROOT / "fig_cavity5000"
PAPER_FIGURES = _ROOT.parent / "vulkan-demo" / "manuscripts" / "fig"
TABLE_L2_PCT = (0.281, 0.370)         # tab:cavity5000
# the zoom boxes of the Re 1000 figure (position and size); limits around the Re 5000 extrema (Marchi u_min -0.447 at
# y 0.074, v_min -0.576 at x 0.958; Ghia -0.436 at 0.070, -0.554 at 0.953), margins as in the Re 3200 figure; the (a)
# box keeps the Re 3200 y range 0.02 - 0.27 with ONE y label (0.1): a "0.2" label touched the main curve at 0.25 / 0.27 / 0.30 (Re 1000 (b) zoom: one label, same reason);
# zoom x ticks -0.45 / -0.35 as Re 3200 (-0.45 / -0.40 ran into each other). u_main_xlim: the lowest Marchi point is
# u = -0.4462, 0.8 pt from the base's left limit -0.46, so its circle was clipped; -0.485 leaves 2.1 pt (Re 3200: 2.1 pt)
ZOOM_RE5000 = {"inset_u": base.ZOOM_RE1000["inset_u"], "inset_v": base.ZOOM_RE1000["inset_v"], "u_main_xlim": (-0.485, 1.04),
               "u_xlim": (-0.47, -0.34), "u_ylim": (0.02, 0.27), "u_xticks": [-0.45, -0.35], "u_yticks": [0.1],
               "v_xlim": (0.895, 0.995), "v_ylim": (-0.598, -0.398), "v_xticks": [0.95], "v_xminor": [0.91, 0.99],
               "v_yticks": 0.1}


def relative_l2_pct(values: np.ndarray, reference: np.ndarray) -> float:
    return float(100.0 * np.sqrt(np.sum((values - reference) ** 2) / np.sum(reference ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-points", type=int, default=1025, help="nodes per side of the averaged field (wall rows)")
    parser.add_argument("--workers", type=int, default=3, help="snapshots interpolated in parallel")
    parser.add_argument("--width", type=float, default=base.ELSARTICLE_SINGLE_COLUMN_WIDTH, help="figure width, inches")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached mean field")
    parser.add_argument("--no-copy", action="store_true", help="do not copy the figure to manuscripts/fig")
    arguments = parser.parse_args()

    base.setup_style()
    facts = base.run_facts(RUN_ID, RUNS_ROOT)
    start_step = int(facts["meta"]["start_from"]["source_step"])
    keep = [index for index, step in enumerate(facts["sample_steps"]) if FIRST_STEP <= step <= LAST_STEP]
    steps = [facts["sample_steps"][index] for index in keep]
    if steps != list(range(FIRST_STEP, LAST_STEP + 1, SAMPLE_STEPS)) or min(steps) <= start_step:
        raise RuntimeError(f"{RUN_ID}: window samples {steps} are not the 14 steps after the start {start_step}")
    facts["sample_steps"] = steps
    facts["sample_times"] = [facts["sample_times"][index] for index in keep]
    window = base.common_window(facts)
    facts["window"] = window
    print(f"[figure] window t = {window[0]:.5f} .. {window[1]:.5f} ({len(steps)} samples, steps {steps[0]:,} .. {steps[-1]:,})",
          flush=True)

    # (a) (b): the base module writes its centre-line CSV to base.DATA -> point it at this directory
    base.DATA = OUT_DIR
    line = base.k1_lines_on_window(facts, facts, RUNS_ROOT, comparison="5090 + XTX (Re 5000)",
                                   script="docs/validation/cluster/re5000_figure/cavity_re5000_figure.py")
    csv_path = OUT_DIR / f"dense_{RUN_ID}_common_window.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace(
        "the K = 8 continuation's sample steps", "the mixed run's own sample steps after its t = 100.5 start"), encoding="utf-8")

    # (c): the window snapshots' time-averaged field; cache in a new git-ignored directory under the runs root
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    field = base.mean_field(facts, arguments.grid_points, arguments.workers, arguments.refresh, CACHE_ROOT)
    quantities = base.field_quantities(field)
    print(f"[figure] field: {len(field['snapshots'])} snapshots t = {', '.join(f'{t:.3f}' for t in field['times'])}, "
          f"MLS fallbacks {int(field['mls_fallbacks'])}, max |v| {quantities['speed_max']:.4f}, psi min / max "
          f"{quantities['psi_min']:.5f} / {quantities['psi_max']:.3e}", flush=True)
    seams = facts["seams_wall"]
    if len(seams) != 1 or facts["meta"].get("cuts") != [341]:
        raise RuntimeError(f"{RUN_ID}: cuts {facts['meta'].get('cuts')} -> seams {seams}, expected one cut at column 341")
    print(f"[figure] seam (cut column 341, wall frame) X = {seams[0]:.6f}", flush=True)

    out = OUT_DIR / "cavity_re5000.pdf"
    size = base.composite_figure(line, quantities, seams, arguments.width, out, reynolds=REYNOLDS, zoom=ZOOM_RE5000,
                                 seams_in_b=False, line_label=LINE_LABEL)

    # self-check against tab:cavity5000
    from scipy.interpolate import CubicSpline
    reference = cavity_reference.marchi2021(REYNOLDS)
    u_line = CubicSpline(line["coordinate"], line["u"])(reference["u"][0])
    v_line = CubicSpline(line["coordinate"], line["v"])(reference["v"][0])
    run = load_run(RUN_ID, RUNS_ROOT)
    mask = np.array([row["step"] in set(steps) for row in run["rows"]])
    u_points = run["arrays"]["u_marchi_wall_mls"][mask].mean(axis=0)
    v_points = run["arrays"]["v_marchi_wall_mls"][mask].mean(axis=0)
    check = {"run": RUN_ID, "window": list(window), "samples": len(steps), "sample_steps": [steps[0], steps[-1], SAMPLE_STEPS],
             "snapshots": [str(name) for name in field["snapshots"]], "snapshot_times": [float(t) for t in field["times"]],
             "grid_points": arguments.grid_points, "mls_fallbacks": int(field["mls_fallbacks"]),
             "cut_column": 341, "grid_origin_x": float(base.read_case_information(facts["meta"])["grid_origin_x"]),
             "support_radius": facts["support"], "spacing": facts["spacing"], "seam_wall_x": seams[0],
             "L2_pct_drawn_lines_spline": [relative_l2_pct(u_line, reference["u"][1]), relative_l2_pct(v_line, reference["v"][1])],
             "L2_pct_marchi_point_samples": [relative_l2_pct(u_points, reference["u"][1]), relative_l2_pct(v_points, reference["v"][1])],
             "L2_pct_table": list(TABLE_L2_PCT), "line_label": LINE_LABEL, "zoom": {k: v for k, v in ZOOM_RE5000.items()},
             "figure": size, "field_speed_max": quantities["speed_max"], "field_psi_min_poisson": quantities["psi_min"]}
    for key in ("L2_pct_drawn_lines_spline", "L2_pct_marchi_point_samples"):
        if any(abs(round(value, 3) - table) > 1e-9 for value, table in zip(check[key], TABLE_L2_PCT)):
            raise RuntimeError(f"self-check {key} {check[key]} does not round to {TABLE_L2_PCT}")
    (OUT_DIR / "check.json").write_text(json.dumps(check, indent=1, default=float), encoding="utf-8")
    print(json.dumps({key: check[key] for key in ("seam_wall_x", "L2_pct_drawn_lines_spline", "L2_pct_marchi_point_samples",
                                                  "snapshot_times", "figure")}, indent=1), flush=True)
    if not arguments.no_copy:
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            shutil.copy2(out.with_suffix(suffix), PAPER_FIGURES / f"cavity_re5000{suffix}")
        print(f"[figure] copied to {PAPER_FIGURES / 'cavity_re5000.pdf'} (+ .png)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
