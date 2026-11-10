"""adami_chain_report.py - E40 (v7-rc3), paper section 6.2: the 1000^2 Re = 1000 cavity with adami walls at K = 1 and
K = 2, figures, tables and CSVs from two finished cavity_runner runs (CPU only).

Deliverables (docs/validation/figures/adami_*.pdf, docs/validation/data/adami_*.csv, docs/validation/adami_k1_k2.md):
  A  figure 1 (K = 1): u(0.5, y) and v(x, 0.5) against Marchi et al. 2021 (Ghia et al. 1982 drawn too), and the
     acceleration magnitude |a| of the fluid at the end of the run (Shepard-interpolated from the final checkpoint);
  B  figure 2 (K = 2): the same layout, the seam marked on the velocity field and on the v(x, 0.5) profile;
  C  table: L2 / Linf of both profiles, the three extrema and their positions, the primary vortex centre and psi_min,
     per run, with the reference values;
  D  K = 2 minus K = 1: the time-averaged velocity difference at the 56 reference points against both standard errors;
     and the time-averaged velocity difference within 3 support radii on either side of the seam (snapshots of the
     averaging window, MLS on a strip grid), with a control strip away from the seam;
  E  invariants of both runs (particle count, every overflow counter, stamp errors, drift) and the run facts.
Method, steady-state rule, sampling and references as in docs/validation/cavity_re1000.md (cavity_analysis.analyse_run).

    .venv/Scripts/python.exe -m experiment.validation.adami_chain_report [--no-stream-function] [--grid 1024]
    # a dry run on other runs (e.g. the 500^2 pair of a v7 campaign root):
    .venv/Scripts/python.exe -m experiment.validation.adami_chain_report --runs-root <root> --k1 <id> --k2 <id> \
        --out-prefix test --no-stream-function
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.cavity_analysis import (DATA, FIGURES, analyse_run, load_run, profile_csv,  # noqa: E402
                                                   setup_style, write_csv)
from experiment.validation.cavity_fields import UniformGrid, read_case_information, scatter_particles_to_grid  # noqa: E402

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re1000_v7"
RUN_IDS = {"K1": "n1000_k1_float32_xi0p001_eps0p0025_adami", "K2": "n1000_k2_float32_xi0p001_eps0p0025_adami"}
SEAM_STRIP_HALF_WIDTH = 3.0          # support radii on either side of the seam
SEAM_STRIP_BINS_PER_SUPPORT = 2      # x bins of h / 2
CONTROL_STRIP_OFFSET = -0.25         # the control strip's centre, metres left of the seam
EXTREMA_REFERENCE = {"u_min": ("u_min", "y_at_u_min"), "v_max": ("v_max", "x_at_v_max"), "v_min": ("v_min", "x_at_v_min")}


# ----------------------------------------------------------------------------------------------- helpers
def wall_frame(run: dict) -> sampling.Frame:
    return sampling.Frame.make("wall", run["spacing"])


def seam_positions(run: dict) -> list[dict]:
    """The K - 1 cuts of a run in metres (global voxel column x h + grid origin) and in the wall frame."""
    information = read_case_information(run["meta"])
    frame = wall_frame(run)
    support = float(run["meta"]["support_radius"])
    cuts = []
    for column in run["meta"].get("cuts") or []:
        x = information["grid_origin_x"] + int(column) * support
        cuts.append({"column": int(column), "x": x, "x_wall": float(frame.to_reference(np.array([x]))[0])})
    return cuts


def snapshots_in_window(run: dict, window: tuple[float, float]) -> list[pathlib.Path]:
    dt = float(run["meta"]["dt"])
    files = sorted((run["dir"] / "snapshots").glob("t*.npz"))
    return [path for path in files if window[0] - 1e-9 <= int(path.stem[1:]) * dt <= window[1] + 1e-9]


def strip_points(x_centre: float, support: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points of a strip |x - x_centre| <= 3 h: columns at the centres of h / 2 bins, rows every h / 2 between the
    innermost wall rows (fluid box +- 0.5, inset by h). Returns (points (M, 2), x offsets of the columns)."""
    half_bins = int(SEAM_STRIP_HALF_WIDTH * SEAM_STRIP_BINS_PER_SUPPORT)
    bin_width = support / SEAM_STRIP_BINS_PER_SUPPORT
    offsets = (np.arange(-half_bins, half_bins) + 0.5) * bin_width
    rows = np.arange(-0.5 + support, 0.5 - support + 1e-12, support / 2)
    xx, yy = np.meshgrid(x_centre + offsets, rows)
    return np.stack([xx.ravel(), yy.ravel()], axis=1), offsets


def strip_time_average(run: dict, window: tuple[float, float], points: np.ndarray, support: float) -> dict:
    """Mean and standard error over the window's snapshots of the MLS velocity at ``points``."""
    total = np.zeros((points.shape[0], 2))
    total_square = np.zeros((points.shape[0], 2))
    files = snapshots_in_window(run, window)
    x_low, x_high = points[:, 0].min() - 1.05 * support, points[:, 0].max() + 1.05 * support
    for path in files:
        with np.load(path) as archive:
            positions = archive["positions"].astype(np.float64)
            keep = (positions[:, 0] >= x_low) & (positions[:, 0] <= x_high)
            particles = sampling.ParticleSet(positions[keep], archive["velocities"][keep].astype(np.float64),
                                             archive["volumes"][keep].astype(np.float64))
        velocity = sampling.interpolate(particles, points, support)["mls"]
        total += velocity
        total_square += velocity ** 2
    count = len(files)
    mean = total / count
    variance = np.maximum(total_square / count - mean ** 2, 0.0) * count / max(count - 1, 1)
    return {"mean": mean, "std": np.sqrt(variance), "sem": np.sqrt(variance / count), "snapshots": count}


def acceleration_field(run: dict, grid_cells: int) -> dict:
    """|a| of the fluid particles of the newest checkpoint, Shepard-interpolated onto a grid over the fluid box
    (wall-frame cavity coordinates for the plot); walls excluded from the interpolation (their a is 0)."""
    checkpoint = sorted((run["dir"] / "checkpoints").glob("c*.npz"))[-1]
    with np.load(checkpoint) as archive:
        position = archive["position_voxel_id"][:, :2].astype(np.float64)
        acceleration = archive["acceleration"][:, :2].astype(np.float64)
        mass = archive["velocity_mass"][:, 3].astype(np.float64)
        density = archive["density_pressure"][:, 0].astype(np.float64) + float(archive["_stored_density_offset"])
        material = archive["material"].astype(np.int64)
    fluid = np.isin(material, np.asarray(run["meta"]["fluid_groups"], dtype=np.int64))
    spacing = run["spacing"]
    support = float(run["meta"]["support_radius"])
    half = 0.5 + spacing / 2            # the fluid box plus half a spacing (the effective wall position)
    grid = UniformGrid(-half, -half, 2 * half / grid_cells, grid_cells, grid_cells)
    magnitude = np.hypot(acceleration[fluid, 0], acceleration[fluid, 1])
    field, weight = scatter_particles_to_grid(position[fluid], mass[fluid] / density[fluid], magnitude[:, None], grid,
                                              support)
    values = field[:, :, 0]
    values[weight <= 0] = np.nan
    frame = wall_frame(run)
    extent = [float(frame.to_reference(np.array([-half]))[0]), float(frame.to_reference(np.array([half]))[0])] * 2
    return {"field": values, "extent": extent, "time": int(checkpoint.stem[1:]) * float(run["meta"]["dt"]),
            "checkpoint": checkpoint.name, "fluid_count": int(fluid.sum()),
            "percentiles": {p: float(np.nanpercentile(values, p)) for p in (50, 90, 99, 99.9)}}


# ----------------------------------------------------------------------------------------------- figure
def profile_field_figure(analysis: dict, field: dict, path: pathlib.Path, seam_wall: float | None, label: str,
                         field_vmax: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    setup_style()
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()
    dense = analysis["dense_reference"]
    u_mean = analysis["statistics"]["u_dense_mls"]["mean"]
    v_mean = analysis["statistics"]["v_dense_mls"]["mean"]
    figure = plt.figure(figsize=(7.2, 2.55))
    grid_spec = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.18), wspace=0.42, left=0.07, right=0.985,
                                    bottom=0.17, top=0.93)
    axes = [figure.add_subplot(grid_spec[0, index]) for index in range(3)]
    color = "#08519c"
    axes[0].plot(u_mean, dense, "-", color=color, lw=0.9, label=f"SPH {label}")
    axes[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec="k", ms=3.4, mew=0.7, label="Marchi et al. (2021)")
    axes[0].plot(ghia["u"][1], ghia["u"][0], "s", mfc="none", mec="#b2182b", ms=2.8, mew=0.6, label="Ghia et al. (1982)")
    axes[1].plot(dense, v_mean, "-", color=color, lw=0.9, label=f"SPH {label}")
    axes[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec="k", ms=3.4, mew=0.7, label="Marchi et al. (2021)")
    axes[1].plot(ghia["v"][0], ghia["v"][1], "s", mfc="none", mec="#b2182b", ms=2.8, mew=0.6, label="Ghia et al. (1982)")
    insets = [axes[0].inset_axes([0.55, 0.09, 0.40, 0.42]), axes[1].inset_axes([0.15, 0.06, 0.34, 0.40])]
    insets[0].plot(u_mean, dense, "-", color=color, lw=0.8)
    insets[0].plot(reference["u"][1], reference["u"][0], "o", mfc="none", mec="k", ms=3.0, mew=0.6)
    insets[1].plot(dense, v_mean, "-", color=color, lw=0.8)
    insets[1].plot(reference["v"][0], reference["v"][1], "o", mfc="none", mec="k", ms=3.0, mew=0.6)
    insets[0].set_xlim(-0.40, -0.30)
    insets[0].set_ylim(0.08, 0.30)
    insets[1].set_xlim(0.85, 0.96)
    insets[1].set_ylim(-0.54, -0.40)
    for inset in insets:
        inset.tick_params(labelsize=6.0, length=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
    axes[0].set_xlabel(r"$u(0.5, y)$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_ylim(0, 1)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$v(x, 0.5)$")
    axes[1].set_xlim(0, 1)
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.5)
    if seam_wall is not None:
        axes[1].axvline(seam_wall, color="#555555", lw=0.7, ls=(0, (3, 2)))
        low, high = axes[1].get_ylim()
        axes[1].text(seam_wall + 0.012, low + 0.03 * (high - low), "seam", ha="left", va="bottom", fontsize=6.5,
                     color="#555555")
    image = axes[2].imshow(field["field"], origin="lower", extent=field["extent"], cmap="viridis",
                           norm=Normalize(0.0, field_vmax), interpolation="nearest", aspect="equal")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$y$")
    if seam_wall is not None:
        axes[2].axvline(seam_wall, color="white", lw=0.8, ls=(0, (3, 2)))
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03)
    colorbar.set_label(r"$|\mathbf{a}|\,L/U^2$ at $t = %.0f$" % field["time"], fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)
    for axis, tag in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left",
                  color="white" if tag == "(c)" else "black")
        for spine in ("top", "right"):
            if tag != "(c)":
                axis.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


# ----------------------------------------------------------------------------------------------- tables
def extrema_rows(analysis: dict, reference: dict) -> dict:
    rows = {}
    for name, (value_key, position_key) in EXTREMA_REFERENCE.items():
        item = analysis["extrema"][name]
        rows[name] = {"value": item["value"], "position": item["position_wall"], "sem": item["sem"],
                      "reference_value": reference["extrema"][value_key][0],
                      "reference_position": reference["extrema"][position_key][0]}
    return rows


def summary_table(analyses: dict, reference: dict) -> tuple[list[str], list[list]]:
    header = ["run", "L2_u", "Linf_u", "L2_v", "Linf_v", "relL2_u", "relL2_v", "u_min", "y_at_u_min", "v_max", "x_at_v_max",
              "v_min", "x_at_v_min", "psi_min", "x_at_psi_min", "y_at_psi_min", "window_start", "window_end", "samples"]
    rows = []
    reference_row = ["Marchi2021", "", "", "", "", "", "", reference["extrema"]["u_min"][0], reference["extrema"]["y_at_u_min"][0],
                     reference["extrema"]["v_max"][0], reference["extrema"]["x_at_v_max"][0], reference["extrema"]["v_min"][0],
                     reference["extrema"]["x_at_v_min"][0], reference["extrema"]["psi_min"][0],
                     reference["extrema"]["x_at_psi_min"][0], reference["extrema"]["y_at_psi_min"][0], "", "", ""]
    rows.append(reference_row)
    for key, analysis in analyses.items():
        errors = analysis["errors"]
        extrema = extrema_rows(analysis, reference)
        stream = analysis.get("stream") or {}
        rows.append([key, errors[("wall", "mls", "u")]["L2"], errors[("wall", "mls", "u")]["Linf"],
                     errors[("wall", "mls", "v")]["L2"], errors[("wall", "mls", "v")]["Linf"],
                     errors[("wall", "mls", "u")]["L2_relative"], errors[("wall", "mls", "v")]["L2_relative"],
                     extrema["u_min"]["value"], extrema["u_min"]["position"], extrema["v_max"]["value"],
                     extrema["v_max"]["position"], extrema["v_min"]["value"], extrema["v_min"]["position"],
                     stream.get("psi_min_wall", ""), stream.get("x_wall", ""), stream.get("y_wall", ""),
                     analysis["window"][0], analysis["window"][1], analysis["samples_in_window"]])
    return header, rows


def point_difference_rows(analyses: dict, reference: dict) -> list[dict]:
    first, second = analyses["K1"], analyses["K2"]
    rows = []
    for line, coordinate_name in (("u", "y"), ("v", "x")):
        coordinates, values, _ = reference[line]
        one = first["statistics"][f"{line}_marchi_wall_mls"]
        two = second["statistics"][f"{line}_marchi_wall_mls"]
        for index, coordinate in enumerate(coordinates):
            difference = two["mean"][index] - one["mean"][index]
            sigma = math.sqrt(one["sem"][index] ** 2 + two["sem"][index] ** 2)
            rows.append({"line": line, "coordinate_name": coordinate_name, "coordinate": float(coordinate),
                         "marchi2021_Tc": float(values[index]), "mean_K1": float(one["mean"][index]),
                         "sem_K1": float(one["sem"][index]), "mean_K2": float(two["mean"][index]),
                         "sem_K2": float(two["sem"][index]), "difference_K2_minus_K1": float(difference),
                         "sigma": sigma, "difference_over_sigma": difference / sigma if sigma > 0 else float("nan"),
                         "samples_K1": int(one["n"]), "samples_K2": int(two["n"])})
    return rows


def seam_difference(runs: dict, analyses: dict, seam: dict, support: float) -> dict:
    """Time-averaged MLS velocity of both runs on the seam strip (and a control strip) and their difference."""
    spacing = runs["K1"]["spacing"]
    result = {}
    for name, x_centre in (("seam", seam["x"]), ("control", seam["x"] + CONTROL_STRIP_OFFSET)):
        points, offsets = strip_points(x_centre, support, spacing)
        averages = {key: strip_time_average(runs[key], analyses[key]["window"], points, support) for key in ("K1", "K2")}
        difference = averages["K2"]["mean"] - averages["K1"]["mean"]
        magnitude = np.hypot(difference[:, 0], difference[:, 1])
        sem_one = np.hypot(averages["K1"]["sem"][:, 0], averages["K1"]["sem"][:, 1])
        sem_two = np.hypot(averages["K2"]["sem"][:, 0], averages["K2"]["sem"][:, 1])
        column_of = np.round((points[:, 0] - x_centre) / (support / SEAM_STRIP_BINS_PER_SUPPORT) - 0.5).astype(int)
        bins = []
        for bin_index, offset in enumerate(offsets):
            selected = column_of == bin_index - len(offsets) // 2
            bins.append({"offset_h": float(offset / support), "points": int(selected.sum()),
                         "rms_difference": float(np.sqrt(np.mean(magnitude[selected] ** 2))),
                         "max_difference": float(magnitude[selected].max()),
                         "rms_sem_K1": float(np.sqrt(np.mean(sem_one[selected] ** 2))),
                         "rms_sem_K2": float(np.sqrt(np.mean(sem_two[selected] ** 2))),
                         "mean_speed": float(np.mean(np.hypot(averages["K1"]["mean"][selected, 0],
                                                              averages["K1"]["mean"][selected, 1])))})
        result[name] = {"x_centre": x_centre, "points": points, "offsets_h": offsets / support, "averages": averages,
                        "difference": difference, "magnitude": magnitude, "sem_K1": sem_one, "sem_K2": sem_two,
                        "bins": bins, "snapshots": averages["K1"]["snapshots"],
                        "rms_all": float(np.sqrt(np.mean(magnitude ** 2))), "max_all": float(magnitude.max()),
                        "rms_sem_K1_all": float(np.sqrt(np.mean(sem_one ** 2))),
                        "rms_sem_K2_all": float(np.sqrt(np.mean(sem_two ** 2)))}
    return result


def invariant_rows(runs: dict) -> list[dict]:
    rows = []
    for key, run in runs.items():
        totals = {"drift": 0, "overflow": 0, "far_migration": 0, "stamp_gpu": 0, "stamp_host": 0}
        for segment in run["segments"]:
            for name in totals:
                totals[name] += int((segment.get("totals") or {}).get(name, 0))
        alive = [int(row["alive"]) for row in run["rows"]]
        rows.append({"run": key, "id": run["id"], "expected_particles": int(run["meta"]["expected_total"]),
                     "alive_min": min(alive), "alive_max": max(alive), "samples": len(alive),
                     "segments": len(run["segments"]),
                     "segments_complete": sum(1 for segment in run["segments"] if segment.get("status") == "complete"),
                     "invariant_violations": sum(1 for segment in run["segments"] if segment.get("invariant_violation")),
                     "wall_hours": sum(float(segment.get("wall_s", 0.0)) for segment in run["segments"]) / 3600.0,
                     "mls_fallbacks_total": sum(int(row.get("mls_fallbacks", 0)) for row in run["rows"]),
                     "min_neighbours": min(int(row.get("min_neighbours", 0)) for row in run["rows"]),
                     "status": run["result"].get("status", ""), "git_head": run["meta"].get("git", {}).get("head"),
                     "git_dirty": run["meta"].get("git", {}).get("dirty"), "devices": run["meta"].get("device_map"),
                     **totals})
    return rows


def md_table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(rows_item for rows_item in row) + " |" for row in rows])


def number(value, digits: int = 4) -> str:
    if value == "" or value is None:
        return "–"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--k1", default=RUN_IDS["K1"])
    parser.add_argument("--k2", default=RUN_IDS["K2"])
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--no-stream-function", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="recompute a cached analysis")
    parser.add_argument("--only", choices=("K1", "K2"), default=None,
                        help="analyse (and cache) one run only, no figures / tables (while the other run is still going)")
    parser.add_argument("--grid", type=int, default=1024, help="cells per side of the |a| field")
    parser.add_argument("--out-prefix", default="adami", help="file name prefix of the figures / CSVs / document")
    parser.add_argument("--doc", default=None, help="markdown output (default docs/validation/<prefix>_k1_k2.md)")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.runs_root).resolve()
    prefix = arguments.out_prefix
    doc_path = pathlib.Path(arguments.doc) if arguments.doc else _REPO_ROOT / "docs" / "validation" / f"{prefix}_k1_k2.md"
    reference = cavity_reference.marchi2021()

    runs, analyses = {}, {}
    for key, run_id in (("K1", arguments.k1), ("K2", arguments.k2)):
        if arguments.only and key != arguments.only:
            continue
        run = load_run(run_id, root)
        if run is None:
            sys.exit(f"{root / run_id}: no samples.jsonl")
        print(f"[report] {key} = {run_id}: {len(run['rows'])} samples, t_last {run['times'].max():.2f}", flush=True)
        # the stream function of a 1000^2 run takes ~10 min: cache the analysis per run (next to the run, keyed
        # by the sample count) unless --refresh
        cache_path = root / f"analysis_{run_id}_{len(run['rows'])}.npy"
        if cache_path.exists() and not arguments.refresh and not arguments.no_stream_function:
            analysis = np.load(cache_path, allow_pickle=True).item()
            print(f"[report] {key}: analysis from {cache_path.name}", flush=True)
        else:
            analysis = analyse_run(run, arguments.average_span, arguments.steady_tol, arguments.window,
                                   not arguments.no_stream_function)
            if not arguments.no_stream_function:
                np.save(cache_path, np.array(analysis, dtype=object), allow_pickle=True)
        analysis["dense_reference"] = run["points"]["dense_reference"]
        analysis["segments"] = run["segments"]
        analysis["rows"] = run["rows"]
        profile_csv(analysis)
        runs[key], analyses[key] = run, analysis
        print(f"[report] {key}: window {analysis['window']}, u L2 {analysis['errors'][('wall', 'mls', 'u')]['L2']:.4f}, "
              f"v L2 {analysis['errors'][('wall', 'mls', 'v')]['L2']:.4f}, stream {analysis.get('stream')}", flush=True)
    if arguments.only:
        return 0
    support = float(runs["K1"]["meta"]["support_radius"])
    cuts = seam_positions(runs["K2"])
    if not cuts:
        sys.exit("the K = 2 run records no cut (meta.json 'cuts')")
    seam = cuts[0]
    print(f"[report] seam: column {seam['column']}, x = {seam['x']:.5f} m, X_wall = {seam['x_wall']:.5f}", flush=True)

    # figures A / B (one colour scale for both fields)
    fields = {key: acceleration_field(runs[key], arguments.grid) for key in ("K1", "K2")}
    vmax = max(field["percentiles"][99] for field in fields.values())
    profile_field_figure(analyses["K1"], fields["K1"], FIGURES / f"{prefix}_fig1_k1.pdf", None, "$K = 1$", vmax)
    profile_field_figure(analyses["K2"], fields["K2"], FIGURES / f"{prefix}_fig2_k2.pdf", seam["x_wall"], "$K = 2$", vmax)

    # table C
    header, rows = summary_table(analyses, reference)
    write_csv(DATA / f"{prefix}_k1_k2_summary.csv", header, rows,
              "E40: adami walls at K = 1 and K = 2 (v7-rc3), errors at the Marchi 2021 points (wall frame, MLS, absolute in\n"
              "units of U; relL2 = L2 / rms(Tc)), extrema from the time-averaged dense lines, primary vortex from the\n"
              "time-averaged MLS grid of the snapshots (psi integrated from the bottom wall); positions in the wall frame.")

    # D: the 56 points
    point_rows = point_difference_rows(analyses, reference)
    write_csv(DATA / f"{prefix}_k2_minus_k1_points.csv", list(point_rows[0].keys()),
              [list(row.values()) for row in point_rows],
              "E40: K = 2 minus K = 1 time-averaged MLS velocity at the 28 + 28 Marchi 2021 points (wall frame); sem = std *\n"
              "sqrt(tau_int / n) of each run's sample series; sigma = sqrt(sem_K1^2 + sem_K2^2).")
    # D: the seam strip
    strips = seam_difference(runs, analyses, seam, support)
    strip_rows = []
    for name, strip in strips.items():
        for index in range(strip["points"].shape[0]):
            strip_rows.append([name, f"{strip['points'][index, 0]:.6f}", f"{strip['points'][index, 1]:.6f}",
                               f"{(strip['points'][index, 0] - strip['x_centre']) / support:+.3f}",
                               f"{strip['averages']['K1']['mean'][index, 0]:.7f}", f"{strip['averages']['K1']['mean'][index, 1]:.7f}",
                               f"{strip['averages']['K2']['mean'][index, 0]:.7f}", f"{strip['averages']['K2']['mean'][index, 1]:.7f}",
                               f"{strip['difference'][index, 0]:+.3e}", f"{strip['difference'][index, 1]:+.3e}",
                               f"{strip['magnitude'][index]:.3e}", f"{strip['sem_K1'][index]:.3e}", f"{strip['sem_K2'][index]:.3e}"])
    write_csv(DATA / f"{prefix}_k2_minus_k1_seam.csv",
              ["strip", "x", "y", "offset_h", "u_K1", "v_K1", "u_K2", "v_K2", "du", "dv", "abs_du", "sem_K1", "sem_K2"],
              strip_rows,
              f"E40: time-averaged MLS velocity (snapshots of the averaging window, {strips['seam']['snapshots']} per run) on a\n"
              f"strip of 3 support radii on either side of the seam (x = {seam['x']:.5f} m, column {seam['column']}) and on a\n"
              f"control strip centred {CONTROL_STRIP_OFFSET:+.2f} m from it; columns every h/2, rows every h/2; sem over the snapshots.")
    bin_header = ["strip", "offset_h", "points", "rms_abs_du", "max_abs_du", "rms_sem_K1", "rms_sem_K2", "mean_speed_K1"]
    bin_rows = [[name, f"{item['offset_h']:+.2f}", item["points"], f"{item['rms_difference']:.3e}", f"{item['max_difference']:.3e}",
                 f"{item['rms_sem_K1']:.3e}", f"{item['rms_sem_K2']:.3e}", f"{item['mean_speed']:.4f}"]
                for name, strip in strips.items() for item in strip["bins"]]
    write_csv(DATA / f"{prefix}_k2_minus_k1_seam_bins.csv", bin_header, bin_rows,
              "E40: the seam strip difference per h/2 column (offset from the seam in support radii; negative = slab 0's side).")

    # E (+ the campaign's wall-clock estimates recorded before each launch, campaign.jsonl of the runs root)
    invariants = invariant_rows(runs)
    estimates = {}
    campaign_journal = root / "campaign.jsonl"
    if campaign_journal.exists():
        for line in campaign_journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if entry.get("run") in (runs["K1"]["id"], runs["K2"]["id"]) and entry.get("start_step") == 0:
                    estimates[entry["run"]] = entry          # the last estimate from step 0 wins
    for row in invariants:
        entry = estimates.get(row["id"], {})
        row["estimate_hours"] = entry.get("hours")
        row["estimate_fps"] = entry.get("fps")
        row["estimate_logged"] = entry.get("logged")
        fps_samples = sorted(float(sample["fps"]) for sample in runs[row["run"]]["rows"][1:] if sample.get("fps"))
        row["median_fps"] = fps_samples[len(fps_samples) // 2] if fps_samples else None
    write_csv(DATA / f"{prefix}_k1_k2_invariants.csv", list(invariants[0].keys()), [list(row.values()) for row in invariants],
              "E40: invariants and run facts of the two runs (segments.jsonl totals, samples.jsonl).")

    # the document
    code_hashes_equal = runs['K1']['meta'].get('code_hashes') == runs['K2']['meta'].get('code_hashes')
    lines = [f"# Adami 壁面 1000² 方腔:K = 1 与 K = 2(v7-rc3,E40)", "",
             f"算例 `{runs['K1']['case']}`(Re = 1000,ξ = 0.001,ε² = 0.0025 h²,`wall_boundary: adami`),求解器 experiment/v7 "
             f"在 tag v7-rc3 = 64eecf0(运行记录的提交:K = 1 {runs['K1']['meta'].get('git', {}).get('head')}、"
             f"K = 2 {runs['K2']['meta'].get('git', {}).get('head')},树均干净;1f3fb48 = tag + 两个只加本报告脚本的提交,"
             f"求解器未变;两次运行记录的求解器代码哈希{'相同' if code_hashes_equal else '**不同**'}),"
             f"默认开关,`cavity_runner --solver v7 --expect release`。K = 1:无头的 GPU 1;K = 2:GPU 0 + 1,等权重,"
             f"切口在全局第 {seam['column']} 列(x = {seam['x']:.4f} m,壁面行框架 X = {seam['x_wall']:.4f})。方法、稳态判据、取样与参考解沿用 "
             "`docs/validation/cavity_re1000.md`(Marchi et al. 2021 为误差参考,Ghia et al. 1982 只画在图里);时间平均取每个 run 最后 20 个时间单位。"
             "只报结果。", ""]
    lines += [f"## 图 1(K = 1)与图 2(K = 2)", "",
              f"![fig1](figures/{prefix}_fig1_k1.pdf)", "", f"![fig2](figures/{prefix}_fig2_k2.pdf)", "",
              f"(a) u(0.5, y) 与 (b) v(x, 0.5):时间平均的 MLS 中线剖面(壁面行框架)对 Marchi 2021(圆圈)与 Ghia 1982(方块);"
              f"(c) 流体的加速度大小 |a| L/U²,末尾检查点(t = {fields['K1']['time']:.1f} / {fields['K2']['time']:.1f})的 Shepard 插值,"
              f"{arguments.grid}² 网格,两图同一色标(上限 = 两个场 99 百分位的较大者 {vmax:.3g});图 2 的竖虚线 = 切口。"
              f"矢量 PDF:`docs/validation/figures/{prefix}_fig1_k1.pdf`、`{prefix}_fig2_k2.pdf`。", ""]
    lines += ["## 表:误差、极值、主涡", ""]
    table_rows = []
    for row in rows:
        table_rows.append([str(row[0])] + [number(value) for value in row[1:16]] + [number(row[16], 2), number(row[17], 2), str(row[18])])
    lines.append(md_table(["run", "L2 u", "L∞ u", "L2 v", "L∞ v", "rel L2 u", "rel L2 v", "u_min", "y(u_min)", "v_max", "x(v_max)",
                           "v_min", "x(v_min)", "ψ_min", "x(ψ_min)", "y(ψ_min)", "窗口起", "窗口止", "样本数"], table_rows))
    lines += ["", f"L2 = √(Σ e_i² / 28)、L∞ = max |e_i|(绝对值,U 为单位),rel L2 = L2 / rms(T_c);极值位置为壁面行框架;"
              f"ψ_min 与主涡中心来自平均窗口内每个时间单位一个快照的 MLS 网格平均(513²)。数据:`data/{prefix}_k1_k2_summary.csv`、"
              f"`data/profile_<run>.csv`、`data/dense_<run>.csv`。", ""]
    steady = {key: (analyses[key]["t_steady_online"], analyses[key]["t_steady_sliding"]) for key in analyses}
    lines += ["稳态时刻(窗口判据 / 滑动重算):" + ";".join(f"{key} {number(value[0], 2)} / {number(value[1], 2)}" for key, value in steady.items()), ""]
    # D
    lines += ["## K = 2 减 K = 1:56 个参考点", ""]
    sigma_ratios = [abs(row["difference_over_sigma"]) for row in point_rows if math.isfinite(row["difference_over_sigma"])]
    differences = [row["difference_K2_minus_K1"] for row in point_rows]
    lines.append(f"Δ = ū_K2 − ū_K1(时间平均的 MLS 速度),各自的标准误差 s.e. = std·√(τ_int/n)(样本间隔 0.2 时间单位),σ = √(s.e._K1² + s.e._K2²)。"
                 f"56 点汇总:max |Δ| = {max(abs(d) for d in differences):.2e},rms Δ = {math.sqrt(sum(d * d for d in differences) / len(differences)):.2e},"
                 f"max |Δ|/σ = {max(sigma_ratios):.2f},|Δ|/σ > 2 的点 {sum(1 for value in sigma_ratios if value > 2)} 个,> 3 的点 "
                 f"{sum(1 for value in sigma_ratios if value > 3)} 个(共 {len(sigma_ratios)})。")
    lines.append("")
    lines.append(md_table(["线", "坐标", "T_c", "ū_K1 ± s.e.", "ū_K2 ± s.e.", "Δ", "Δ/σ"],
                          [[row["line"], f"{row['coordinate']:.5f}", f"{row['marchi2021_Tc']:.5f}",
                            f"{row['mean_K1']:.5f} ± {row['sem_K1']:.1e}", f"{row['mean_K2']:.5f} ± {row['sem_K2']:.1e}",
                            f"{row['difference_K2_minus_K1']:+.2e}", f"{row['difference_over_sigma']:+.2f}"] for row in point_rows]))
    lines += ["", f"数据:`data/{prefix}_k2_minus_k1_points.csv`。", ""]
    lines += ["## K = 2 减 K = 1:接缝两侧各 3 个支撑半径", ""]
    lines.append(f"平均窗口内的快照(每个时间单位一个,每个 run {strips['seam']['snapshots']} 个)在条带 |x − x_seam| ≤ 3h 上的 MLS 速度时间平均,"
                 f"点列每 h/2 一列(列中心离切口 ±0.25h … ±2.75h,负号 = slab 0 一侧),每行 h/2;|Δu| = 两个 run 时间平均速度之差的模;"
                 f"s.e. = 各 run 在该点的快照标准差 / √n(未作自相关修正)。对照条带:同样的点列,中心在切口左侧 {-CONTROL_STRIP_OFFSET:.2f} m。")
    lines.append("")
    lines.append(md_table(["条带", "离切口 (h)", "点数", "rms |Δu|", "max |Δu|", "rms s.e. K1", "rms s.e. K2", "平均速度 K1"],
                          [[row[0], row[1], str(row[2]), row[3], row[4], row[5], row[6], row[7]] for row in bin_rows]))
    lines.append("")
    lines.append("整条带:" + ";".join(f"{name}:rms |Δu| {strip['rms_all']:.2e},max {strip['max_all']:.2e},rms s.e. K1 / K2 "
                                       f"{strip['rms_sem_K1_all']:.2e} / {strip['rms_sem_K2_all']:.2e}" for name, strip in strips.items()))
    lines += ["", f"数据:`data/{prefix}_k2_minus_k1_seam.csv`(逐点)、`data/{prefix}_k2_minus_k1_seam_bins.csv`(按列)。", ""]
    # E
    lines += ["## 不变量", ""]
    lines.append(md_table(["run", "粒子数(期望)", "存活(每个样本 min–max)", "overflow 合计", "far_migration", "帧戳错误 GPU / 主机", "drift",
                           "段数(完成 / 违反)", "墙钟 h", "样本数", "MLS 回退合计", "最少邻居数", "状态"],
                          [[row["run"], str(row["expected_particles"]), f"{row['alive_min']}–{row['alive_max']}", str(row["overflow"]),
                            str(row["far_migration"]), f"{row['stamp_gpu']} / {row['stamp_host']}", str(row["drift"]),
                            f"{row['segments']}({row['segments_complete']} / {row['invariant_violations']})", f"{row['wall_hours']:.2f}",
                            str(row["samples"]), str(row["mls_fallbacks_total"]), str(row["min_neighbours"]), row["status"]]
                           for row in invariants]))
    lines += ["", "每个样本(每 0.2 时间单位)与每个 defrag 边界都检查:存活粒子总数 = 初始总数、每张卡的全部 `overflow_*` 计数、far_migration、"
              "GPU 与主机帧戳错误;任一违反即作废(退出码 3)。overflow 合计 = 两段结束时所有 slab 所有 `overflow_*` 之和。"
              f"数据:`data/{prefix}_k1_k2_invariants.csv`。", ""]
    lines += ["墙钟(提交前的校准估算 → 实际):" + ";".join(
        f"{row['run']} {number(row['estimate_hours'], 2)} h(校准 {number(row['estimate_fps'], 1)} 步/s,{row['estimate_logged']})"
        f" → {row['wall_hours']:.2f} h(运行中样本间吞吐中位数 {number(row['median_fps'], 1)} 步/s)" for row in invariants), ""]
    # field figures of experiment/validation/cavity_fields.py (newest checkpoint of each run), when they were rendered
    fields_dir = _REPO_ROOT / "docs" / "validation" / "fields"
    k1_id, k2_id = runs["K1"]["id"], runs["K2"]["id"]
    if (fields_dir / f"{k1_id}_vorticity.png").exists() and (fields_dir / f"{k2_id}_vorticity.png").exists():
        lines += ["## 场图(每个 run 最新的检查点)", "",
                  "`experiment/validation/cavity_fields.py`(2026-10-02 批准的场图版式):全部粒子(流体 + 壁 + 盖)的 Shepard 插值到 "
                  "2048² 网格,边界带按真实厚度,四角放大图逐粒子画出(各自色标),ω_z 为平滑场的中心差分,流线来自 Poisson ψ;"
                  "主涡参考 = Marchi 2021 表 19。图 K = 2 里的点线 = 切口。", "",
                  f"![vorticity K1](fields/{k1_id}_vorticity.png)", "", f"![vorticity K2](fields/{k2_id}_vorticity.png)", "",
                  f"![K2 minus K1](fields/{k2_id}_minus_{k1_id}.png)", "",
                  "| run | 检查点 | 主涡 (X, Y) | ψ | ω_z | 到参考的距离 |", "|---|---|---|---|---|---|"]
        for key, run_id in (("K1", k1_id), ("K2", k2_id)):
            check_path = fields_dir / f"{run_id}_check.json"
            if not check_path.exists():
                continue
            check = json.loads(check_path.read_text(encoding="utf-8"))
            primary = check["vortex_centres_by_stream_function"][check.get("plotted_stream_function", "poisson")]["primary"]
            lines.append(f"| {key} | c{int(check['step']):09d}(t = {float(check['time']):.2f}) | "
                         f"({primary['position'][0]:.4f}, {primary['position'][1]:.4f}) | {primary['stream_function']:.5f} | "
                         f"{primary['vorticity_z']:.4f} | {primary['distance_to_reference']:.4f} L |")
        lines.append(f"| Marchi 2021 | – | ({primary['reference_position'][0]:.4f}, {primary['reference_position'][1]:.4f}) | "
                     f"{primary['reference_stream_function']:.5f} | – | – |")
        difference_path = fields_dir / f"difference_{k2_id}_minus_{k1_id}.json"
        if difference_path.exists():
            difference = json.loads(difference_path.read_text(encoding="utf-8"))
            interior = difference["statistics_interior"]
            lines += ["", f"K2 − K1(同一检查点步 {int(difference['minuend']['step']):,},两个 run 都是流体的格点,离壁 > 2h 的内部):"
                      f"max |Δu| {interior['max_abs_delta_u']:.2e}、rms Δu {interior['rms_delta_u']:.2e}、"
                      f"max |Δv| {interior['max_abs_delta_v']:.2e}、rms Δv {interior['rms_delta_v']:.2e}(U);"
                      f"max |Δω_z| {interior['max_abs_delta_vorticity']:.2f}、rms Δω_z {interior['rms_delta_vorticity']:.3f}(U/L)。"]
            for seam_entry in difference.get("seam", []):
                lines.append(f"切口 X = {seam_entry['cut_x']:.4f}:差值场的列间跳变在切口处 {seam_entry['difference_column_jump_at_cut']:.1e},"
                             f"内部中位数 {seam_entry['difference_column_jump_interior_median']:.1e}"
                             f"(p99 {seam_entry['difference_column_jump_interior_percentile_99']:.1e}),"
                             f"{100 * seam_entry['difference_column_jump_fraction_larger_than_at_cut']:.0f} % 的列对跳变更大;"
                             f"K = 2 自身速度在切口附近的二阶 x 差分最大 {seam_entry['dual_second_difference_near_cut_maximum']:.1e},"
                             f"内部中位数 {seam_entry['dual_second_difference_interior_median']:.1e}"
                             f"(p99 {seam_entry['dual_second_difference_interior_percentile_99']:.1e});"
                             f"切口处的 |Δu| 对邻域中位数 {seam_entry['magnitude_at_cut_over_neighbourhood_median']:.2f},"
                             f"Δω_z 的 {seam_entry['vorticity_at_cut_over_neighbourhood_median']:.2f}。")
            lines += ["", f"数字:`docs/validation/fields/fields_check.md`、`difference_{k2_id}_minus_{k1_id}.json`;"
                      f"速度场图 `<run>_velocity.png`,逐像素图 `<run>_{{velocity,vorticity}}_full.png`。"]
        lines.append("")
    lines += ["## 文件", "", f"- 图:`docs/validation/figures/{prefix}_fig1_k1.pdf`(.png)、`{prefix}_fig2_k2.pdf`(.png)",
              f"- 数据:`docs/validation/data/{prefix}_k1_k2_summary.csv`、`{prefix}_k2_minus_k1_points.csv`、`{prefix}_k2_minus_k1_seam.csv`、"
              f"`{prefix}_k2_minus_k1_seam_bins.csv`、`{prefix}_k1_k2_invariants.csv`、`profile_{runs['K1']['id']}.csv`、"
              f"`profile_{runs['K2']['id']}.csv`、`dense_<run>.csv`",
              f"- 运行目录(不入库):`{root / runs['K1']['id']}`、`{root / runs['K2']['id']}`",
              "- 生成:`.venv/Scripts/python.exe -m experiment.validation.adami_chain_report`", ""]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    json.dump({"seam": seam, "fields": {key: {name: value for name, value in field.items() if name != "field"}
                                          for key, field in fields.items()},
               "point_summary": {"max_abs_difference": max(abs(d) for d in differences),
                                 "max_abs_sigma_ratio": max(sigma_ratios)},
               "strips": {name: {"bins": strip["bins"], "rms_all": strip["rms_all"], "max_all": strip["max_all"]}
                          for name, strip in strips.items()},
               "invariants": invariants},
              open(DATA / f"{prefix}_k1_k2_report.json", "w", encoding="utf-8"), indent=1, default=str)
    print(f"[report] wrote {doc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
