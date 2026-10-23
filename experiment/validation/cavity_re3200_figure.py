"""cavity_re3200_figure.py - paper section 6.2: the Re = 3200 lid-driven cavity figures, laid out exactly like the
Re = 1000 ones of cavity_re1000_figure.py (same width 384 pt, panel geometry, fonts, line styles and colours) from the
finished runs (v7-rc3, adami walls, 2000^2): K = 1 (one RTX 5090, warm-started from the 1000^2 Re 3200 run) and K = 8
(one whole N56 node, 8 x RTX 5090, SLURM job 1724841, continued from the K = 1 run's t = 79.5 checkpoint with
--average-span 15). Uses existing data only: nothing is run. Every time average is taken over the common window of
the two runs, t = 79.69875 .. 84.46875: the K = 8 continuation's 25 samples (every 0.2 time units) and 5 snapshots
(every ~1 time unit); K = 1 restricted to the same steps.

Outputs (docs/validation/figures/, vector PDF + PNG preview):
  cavity_re3200.pdf   three panels in a row, (a) (b) (c):
    (a) u(0.5, y) and (b) v(x, 0.5): the time-averaged MLS centre lines of K = 1 (solid) against Marchi et al. 2021
        (open circles, Tables 24 / 25) and Ghia et al. 1982 (open squares, Re 3200, 129^2), zooms near u_min and v_min
        in the boxes of the Re 1000 figure (their limits follow the Re 3200 extrema); the 7 seams of the K = 8 run as
        thin dashed vertical lines in (b); the legend in (a) only;
    (c) K = 8: time-averaged speed |v|/U (colour 0 - 1) with streamlines and the 7 seams (thin white dashed lines),
        the stream-function levels of the Re 1000 figure.
  cavity_re3200_seams.pdf   the cavity_seams.pdf pair for Re 3200: (a) (K = 8 - K = 1) / sigma at the 56 Marchi 2021
    points (no K = 2 run at this Re), (b) rms |delta v| of K = 8 - K = 1 on the seam strips against the distance to the
    nearest seam, the 6 control strips as a grey band.

Data: the K = 1 run's samples (dense MLS centre lines of 1001 points, wall-row frame) restricted to the K = 8 run's
sample steps -> docs/validation/data/dense_<K1 run>_common_window.csv; the field of (c) from the 5 window snapshots
of the K = 8 run (linear MLS onto the node grid, cached as mean_field_<run>_<points>.npz next to the runs);
docs/validation/data/adami_k8_re3200_point_differences.csv and adami_k8_re3200_strip_bins.csv
(experiment/validation/adami_k8_report.py --reynolds 3200, docs/validation/adami_k8_re3200.md); references
cavity_reference.marchi2021(3200) and ghia1982(3200).

    PYTHONUTF8=1 .venv/Scripts/python.exe -m experiment.validation.cavity_re3200_figure [--grid-points 1025] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_re1000_figure as base  # noqa: E402

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re3200_v7"
RUN_IDS = {"K1": "n2000_k1_float32_xi0p001_eps0p0025_adami_re3200_warm",
           "K8": "n2000_k8_float32_xi0p001_eps0p0025_adami_re3200"}
POINT_DIFFERENCES_CSV = base.DATA / "adami_k8_re3200_point_differences.csv"
STRIP_BINS_CSV = base.DATA / "adami_k8_re3200_strip_bins.csv"
REYNOLDS = 3200
# the zoom boxes of the Re 1000 figure (position and size); limits around the Re 3200 extrema (u_min -0.436 at
# y 0.093, v_min -0.568 at x 0.948); the (a) box's y range reaches 0.27 so that its "0.2" label clears the main curve
ZOOM_RE3200 = {"inset_u": base.ZOOM_RE1000["inset_u"], "inset_v": base.ZOOM_RE1000["inset_v"],
               "u_xlim": (-0.46, -0.34), "u_ylim": (0.04, 0.27), "u_xticks": [-0.45, -0.35], "u_yticks": [0.1, 0.2],
               "v_xlim": (0.885, 0.985), "v_ylim": (-0.595, -0.395), "v_xticks": [0.94], "v_xminor": [0.90, 0.98],
               "v_yticks": 0.1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-points", type=int, default=1025, help="nodes per side of the averaged field (wall rows)")
    parser.add_argument("--workers", type=int, default=4, help="snapshots interpolated in parallel")
    parser.add_argument("--width", type=float, default=base.ELSARTICLE_SINGLE_COLUMN_WIDTH, help="figure width, inches")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached mean field")
    parser.add_argument("--out", default=str(base.FIGURES / "cavity_re3200.pdf"))
    parser.add_argument("--out-seams", default=str(base.FIGURES / "cavity_re3200_seams.pdf"))
    parser.add_argument("--seams-only", action="store_true", help="only cavity_re3200_seams.pdf (from the committed CSVs)")
    arguments = parser.parse_args()

    base.setup_style()
    seams_arguments = dict(point_csv=POINT_DIFFERENCES_CSV, strip_csv=STRIP_BINS_CSV, pairs_to_draw=("K8-K1",))
    if arguments.seams_only:
        print(json.dumps(base.seams_figure(arguments.width, pathlib.Path(arguments.out_seams), **seams_arguments),
                         indent=1), flush=True)
        return 0

    runs = {key: base.run_facts(run_id, RUNS_ROOT) for key, run_id in RUN_IDS.items()}
    window = base.common_window(runs["K8"])
    for run in runs.values():
        run["window"] = window
    print(f"[figure] common window t = {window[0]:.5f} .. {window[1]:.5f} ({len(runs['K8']['sample_steps'])} samples of "
          f"the K = 8 run, steps {min(runs['K8']['sample_steps']):,} .. {max(runs['K8']['sample_steps']):,})", flush=True)
    line = base.k1_lines_on_window(runs["K1"], runs["K8"], RUNS_ROOT, comparison="K = 1 / K = 8 (Re 3200)",
                                   script="cavity_re3200_figure.py")
    field = base.mean_field(runs["K8"], arguments.grid_points, arguments.workers, arguments.refresh, RUNS_ROOT)
    quantities = base.field_quantities(field)
    print(f"[figure] K8: {len(field['snapshots'])} snapshots t = {float(field['times'][0]):.2f} .. "
          f"{float(field['times'][-1]):.2f}, MLS fallbacks {int(field['mls_fallbacks'])}, max |v| "
          f"{quantities['speed_max']:.4f}, psi min / max {quantities['psi_min']:.5f} / {quantities['psi_max']:.3e}",
          flush=True)
    seams = runs["K8"]["seams_wall"]
    if len(seams) != base.SEAM_COUNT:
        raise RuntimeError(f"K8: {len(seams)} seams in meta.json, expected {base.SEAM_COUNT}")
    print("[figure] seams (K = 8, wall frame) X = " + " ".join(f"{seam:.4f}" for seam in seams), flush=True)
    size = base.composite_figure(line, quantities, seams, arguments.width, pathlib.Path(arguments.out),
                                 reynolds=REYNOLDS, zoom=ZOOM_RE3200, seams_in_b=True)
    print(json.dumps(size, indent=1), flush=True)
    print(json.dumps(base.seams_figure(arguments.width, pathlib.Path(arguments.out_seams), **seams_arguments), indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
