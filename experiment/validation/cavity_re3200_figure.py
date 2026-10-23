"""cavity_re3200_figure.py - paper section 6.2: the Re = 3200 row of the lid-driven cavity figure, in the style and the
panel geometry of cavity_re1000.pdf (cavity_re1000_figure.py; same width, so that the two rows stack in one figure)
from the finished runs (v7-rc3, adami walls, 2000^2): K = 1 (one RTX 5090, warm-started from the 1000^2 Re 3200 run)
and K = 8 (one whole N56 node, SLURM job 1724841, continued from the K = 1 run's t = 79.5 checkpoint with
--average-span 15). Uses existing data only: nothing is run. Every time average is taken over the common window of
the two runs, t = 79.69875 .. 84.46875: the K = 8 continuation's 25 samples (every 0.2 time units); K = 1 restricted
to the same steps.

Output (docs/validation/figures/cavity_re3200.pdf + .png, width = elsarticle single-column text width, 384 pt):
  (d) u(0.5, y) and (e) v(x, 0.5): the time-averaged MLS centre lines of K = 1 (solid) against Marchi et al. 2021
      (open circles, Tables 24 / 25) and Ghia et al. 1982 (open squares, 129^2), zooms near u_min and v_min, the legend
      in (d); the 7 seams of the K = 8 run as thin dashed vertical lines in (e) (wall-row frame);
  (f) rms |delta v| of K = 8 - K = 1 (time-averaged MLS velocity of the window's 5 snapshots) on strips of 3 h on either
      side of every seam against the distance to the nearest seam in support radii h (both sides folded), one thin
      line per seam, the min - max range of the 6 control strips (half way between neighbouring seams) as a grey band
      - the (b) panel of cavity_seams.pdf for Re 3200, in the place of the field panel (c) of the Re 1000 row.
  The panel labels continue the Re 1000 row: (d), (e), (f).

Data: the K = 1 run's samples (dense MLS centre lines of 1001 points, wall-row frame) restricted to the K = 8 run's
sample steps -> docs/validation/data/dense_<K1 run>_common_window.csv; docs/validation/data/adami_k8_re3200_strip_bins.csv
(experiment/validation/adami_k8_report.py --reynolds 3200, docs/validation/adami_k8_re3200.md); references
cavity_reference.marchi2021(3200) and ghia1982(3200).

    .venv/Scripts/python.exe -m experiment.validation.cavity_re3200_figure [--width 5.333]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_re1000_figure as base, cavity_reference  # noqa: E402

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re3200_v7"
RUN_IDS = {"K1": "n2000_k1_float32_xi0p001_eps0p0025_adami_re3200_warm",
           "K8": "n2000_k8_float32_xi0p001_eps0p0025_adami_re3200"}
STRIP_BINS_CSV = base.DATA / "adami_k8_re3200_strip_bins.csv"
REYNOLDS = 3200
# zoom boxes as in the Re 1000 row; limits around the Re 3200 extrema (u_min -0.436 at y 0.093, v_min -0.568 at x 0.948)
ZOOM_RE3200 = {"inset_u": [0.63, 0.50, 0.33, 0.27], "inset_v": [0.27, 0.15, 0.21, 0.34],
               "u_xlim": (-0.46, -0.34), "u_ylim": (0.04, 0.25), "u_xticks": [-0.45, -0.35], "u_yticks": [0.1, 0.2],
               "v_xlim": (0.885, 0.985), "v_ylim": (-0.595, -0.395), "v_xticks": [0.94], "v_xminor": [0.90, 0.98],
               "v_yticks": 0.1}
STRIPS_Y_MAJOR = 4.0          # 1e-5 U; the Re 3200 strip values are 4 - 10e-5 (Re 1000: 3.4 - 7.3e-5)
STRIPS_LEGEND_COLUMNS = 2     # the panel is narrower than cavity_seams.pdf (b)
STRIPS_HEADROOM = 2.2         # four legend rows + title above the data
RIGHT_MARGIN = 0.10


def check_strips_clearances(figure, axis_f, minimum_pt: float = 0.5) -> None:
    """The legend of (f) against the seam lines and the control band (the (b) checks of cavity_seams.pdf)."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    points = base._drawn_points(axis_f, renderer)
    legend = axis_f.get_legend().get_window_extent(renderer)
    band_top = max(collection.get_window_extent(renderer).y1 for collection in axis_f.collections)
    base._report_clearances([("(f) legend", {"seam lines": base._clearance_points(legend, points, dpi),
                                             "control band": (legend.y0 - band_top) * 72.0 / dpi})], minimum_pt)


def row_figure(line: dict, folded: dict, seams_wall: list[float], width: float, path: pathlib.Path) -> dict:
    import matplotlib.pyplot as plt
    layout = base.LAYOUT
    reference = cavity_reference.marchi2021(REYNOLDS)
    ghia = cavity_reference.ghia1982(REYNOLDS)
    right_reserve = 0.33                       # the Re 1000 row's colourbar tick labels and label
    field_size = (width - layout["left"] - 2 * layout["profile_width"] - layout["gap_ab"] - layout["gap_bc"]
                  - layout["colorbar_pad"] - layout["colorbar_width"] - right_reserve)
    height = layout["bottom"] + field_size + layout["top_margin"]
    figure = plt.figure(figsize=(width, height))
    x = layout["left"]
    axis_u = base.axes_inches(figure, x, layout["bottom"], layout["profile_width"], field_size)
    x += layout["profile_width"] + layout["gap_ab"]
    axis_v = base.axes_inches(figure, x, layout["bottom"], layout["profile_width"], field_size)
    x += layout["profile_width"] + layout["gap_bc"]
    strips_width = width - x - RIGHT_MARGIN    # (f) starts where (c) starts and runs to the right margin
    axis_f = base.axes_inches(figure, x, layout["bottom"], strips_width, field_size)
    insets = base.draw_profiles(axis_u, axis_v, line, reference, ghia, legend=True, zoom=ZOOM_RE3200,
                                seams_wall=seams_wall)
    strips = base.draw_strips(axis_f, folded, y_major=STRIPS_Y_MAJOR, legend_columns=STRIPS_LEGEND_COLUMNS,
                              headroom=STRIPS_HEADROOM)
    for axis, text in zip((axis_u, axis_v, axis_f), ("(d)", "(e)", "(f)")):
        base.panel_label(axis, text)
    base.check_clearances(figure, axis_u, axis_v, axis_f, insets)     # its "(c)" lines are the (f) panel here
    check_strips_clearances(figure, axis_f)
    base.save(figure, path)
    plt.close(figure)
    return {"width_in": width, "height_in": height, "profile_height_in": field_size, "strips_width_in": strips_width,
            "peak_rms_1e-5": strips["peak"], "distances_h": [float(value) for value in strips["distances"]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", type=float, default=base.ELSARTICLE_SINGLE_COLUMN_WIDTH, help="figure width, inches")
    parser.add_argument("--out", default=str(base.FIGURES / "cavity_re3200.pdf"))
    arguments = parser.parse_args()

    base.setup_style()
    runs = {key: base.run_facts(run_id, RUNS_ROOT) for key, run_id in RUN_IDS.items()}
    window = base.common_window(runs["K8"])
    for run in runs.values():
        run["window"] = window
    print(f"[figure] common window t = {window[0]:.5f} .. {window[1]:.5f} ({len(runs['K8']['sample_steps'])} samples of "
          f"the K = 8 run, steps {min(runs['K8']['sample_steps']):,} .. {max(runs['K8']['sample_steps']):,})", flush=True)
    line = base.k1_lines_on_window(runs["K1"], runs["K8"], RUNS_ROOT, comparison="K = 1 / K = 8 (Re 3200)",
                                   script="cavity_re3200_figure.py")
    seams = runs["K8"]["seams_wall"]
    if len(seams) != base.SEAM_COUNT:
        raise RuntimeError(f"K8: {len(seams)} seams in meta.json, expected {base.SEAM_COUNT}")
    print("[figure] seams (K = 8, wall frame) X = " + " ".join(f"{seam:.4f}" for seam in seams), flush=True)
    folded = base.folded_strip_bins(STRIP_BINS_CSV)
    print(json.dumps(row_figure(line, folded, seams, arguments.width, pathlib.Path(arguments.out)), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
