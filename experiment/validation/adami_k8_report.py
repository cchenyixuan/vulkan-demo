"""adami_k8_report.py - paper section 6.2 supplement (2026-10-21): the 1000^2 Re = 1000 adami cavity continued at K = 8
on one whole N56 node (8 x RTX 5090, SLURM job 1719093, rc3 deployment) from the K = 1 run's t = 89.3025 checkpoint
(cavity_runner --start-from --slabs 8), compared with the K = 1 and K = 2 runs of docs/validation/adami_k1_k2.md on the
SAME sample steps (t >= 89.5: 52 samples every 0.2 time units, 11 snapshots every ~1 time unit).

Deliverables (docs/validation/adami_k8.md, docs/validation/data/adami_k8_*.csv + adami_k8_report.json,
docs/validation/figures/adami_k8_fig*.pdf/.png):
  A  summary table: L2 / Linf / relative L2 of u(0.5, y) and v(x, 0.5) at the 28 + 28 Marchi 2021 points (wall frame,
     linear MLS), the three extrema with positions (time-averaged dense lines), psi_min and the primary vortex centre
     (time-averaged MLS grid of the snapshots), for K = 1 / K = 2 / K = 8 on the common window, plus the full-window
     K = 1 / K = 2 rows of the E40 report for context;
  B  K = 8 minus K = 1 (and K = 2 minus K = 1, K = 8 minus K = 2) at the 56 points against the standard errors;
  C  the 7 seams: positions (global column, metres, wall frame) and the time-averaged velocity difference K = 8 minus
     K = 1 on a strip of 3 support radii on either side of every seam (snapshots of the window, MLS), with 6 control
     strips half way between the seams;
  D  invariants, throughput and wall clock of the three runs;
  E  figure 1: K = 8 profiles against Marchi 2021 / Ghia 1982 with the seams marked, and |a| of the fluid at the end
     (final checkpoint); figure 2: (a) difference over sigma at the 56 points, (b) seam strips per h/2 column.
Method, steady-state rule, sampling and references as in docs/validation/cavity_re1000.md and adami_k1_k2.md.

    .venv/Scripts/python.exe -m experiment.validation.adami_k8_report [--no-stream-function] [--refresh]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.adami_chain_report import (SEAM_STRIP_BINS_PER_SUPPORT, SEAM_STRIP_HALF_WIDTH,  # noqa: E402
                                                      acceleration_field, extrema_rows, md_table, number,
                                                      seam_positions, snapshots_in_window, strip_points)
from experiment.validation.cavity_analysis import (DATA, FIGURES, analyse_run, load_run, profile_csv,  # noqa: E402
                                                   setup_style, write_csv)

RUNS_ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re1000_v7"
RUN_IDS = {"K1": "n1000_k1_float32_xi0p001_eps0p0025_adami", "K2": "n1000_k2_float32_xi0p001_eps0p0025_adami",
           "K8": "n1000_k8_float32_xi0p001_eps0p0025_adami"}
E40_SUMMARY = DATA / "adami_k1_k2_summary.csv"
E40_REPORT = DATA / "adami_k1_k2_report.json"


# ----------------------------------------------------------------------------------------------- the common window
def restrict_to_steps(run: dict, steps: set[int]) -> dict:
    """A copy of ``run`` with only the samples whose step is in ``steps`` (the K = 8 run's sample steps)."""
    keep = np.array([row["step"] in steps for row in run["rows"]], dtype=bool)
    restricted = dict(run)
    restricted["rows"] = [row for row, flag in zip(run["rows"], keep) if flag]
    restricted["times"] = run["times"][keep]
    restricted["kinetic_energy"] = run["kinetic_energy"][keep]
    restricted["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    return restricted


def analyse_window(run: dict, tolerance: float, window: float, with_stream_function: bool) -> dict:
    """analyse_run over ALL of the run's (restricted) samples: the averaging span covers the first sample."""
    times = run["times"]
    span = float(times.max() - times.min()) + 1e-3
    analysis = analyse_run(run, span, tolerance, window, with_stream_function)
    assert analysis["samples_in_window"] == len(run["rows"]), (analysis["samples_in_window"], len(run["rows"]))
    analysis["dense_reference"] = run["points"]["dense_reference"]
    return analysis


# ----------------------------------------------------------------------------------------------- strips
def strip_averages(run: dict, window: tuple[float, float], strips: dict[str, np.ndarray], support: float) -> dict:
    """Mean and standard error over the window's snapshots of the MLS velocity on every strip (one KD-tree per
    snapshot for all strips together)."""
    names = list(strips)
    points = np.concatenate([strips[name] for name in names], axis=0)
    sizes = [strips[name].shape[0] for name in names]
    files = snapshots_in_window(run, window)
    total = np.zeros((points.shape[0], 2))
    total_square = np.zeros((points.shape[0], 2))
    fallbacks = 0
    for path in files:
        with np.load(path) as archive:
            particles = sampling.ParticleSet(archive["positions"].astype(np.float64),
                                             archive["velocities"].astype(np.float64),
                                             archive["volumes"].astype(np.float64))
        tree = sampling.cKDTree(particles.positions)
        result = sampling.interpolate(particles, points, support, tree=tree)
        velocity = result["mls"]
        fallbacks += int(np.asarray(result.get("mls_fallback", np.zeros(points.shape[0], dtype=bool))).sum())
        total += velocity
        total_square += velocity ** 2
    count = len(files)
    mean = total / count
    variance = np.maximum(total_square / count - mean ** 2, 0.0) * count / max(count - 1, 1)
    sem = np.sqrt(variance / count)
    averages, start = {}, 0
    for name, size in zip(names, sizes):
        averages[name] = {"mean": mean[start:start + size], "sem": sem[start:start + size], "snapshots": count}
        start += size
    averages["_fallbacks"] = fallbacks
    averages["_snapshots"] = [path.name for path in files]
    return averages


def strip_table(name: str, x_centre: float, points: np.ndarray, offsets: np.ndarray, first: dict, second: dict,
                support: float) -> dict:
    difference = second["mean"] - first["mean"]
    magnitude = np.hypot(difference[:, 0], difference[:, 1])
    sem_first = np.hypot(first["sem"][:, 0], first["sem"][:, 1])
    sem_second = np.hypot(second["sem"][:, 0], second["sem"][:, 1])
    speed = np.hypot(first["mean"][:, 0], first["mean"][:, 1])
    column_of = np.round((points[:, 0] - x_centre) / (support / SEAM_STRIP_BINS_PER_SUPPORT) - 0.5).astype(int)
    bins = []
    for bin_index, offset in enumerate(offsets):
        selected = column_of == bin_index - len(offsets) // 2
        bins.append({"offset_h": float(offset / support), "points": int(selected.sum()),
                     "rms_difference": float(np.sqrt(np.mean(magnitude[selected] ** 2))),
                     "max_difference": float(magnitude[selected].max()),
                     "rms_sem_first": float(np.sqrt(np.mean(sem_first[selected] ** 2))),
                     "rms_sem_second": float(np.sqrt(np.mean(sem_second[selected] ** 2))),
                     "mean_speed": float(speed[selected].mean())})
    inner = np.abs(points[:, 0] - x_centre) <= 0.5 * support
    return {"strip": name, "x_centre": float(x_centre), "points": int(points.shape[0]),
            "rms_all": float(np.sqrt(np.mean(magnitude ** 2))), "max_all": float(magnitude.max()),
            "rms_inner_half_h": float(np.sqrt(np.mean(magnitude[inner] ** 2))),
            "max_inner_half_h": float(magnitude[inner].max()),
            "rms_sem_first": float(np.sqrt(np.mean(sem_first ** 2))), "rms_sem_second": float(np.sqrt(np.mean(sem_second ** 2))),
            "mean_speed_first": float(speed.mean()), "bins": bins}


# ----------------------------------------------------------------------------------------------- tables
def summary_row(key: str, analysis: dict, reference: dict) -> list:
    errors = analysis["errors"]
    extrema = extrema_rows(analysis, reference)
    stream = analysis.get("stream") or {}
    return [key, errors[("wall", "mls", "u")]["L2"], errors[("wall", "mls", "u")]["Linf"],
            errors[("wall", "mls", "v")]["L2"], errors[("wall", "mls", "v")]["Linf"],
            errors[("wall", "mls", "u")]["L2_relative"], errors[("wall", "mls", "v")]["L2_relative"],
            extrema["u_min"]["value"], extrema["u_min"]["position"], extrema["v_max"]["value"],
            extrema["v_max"]["position"], extrema["v_min"]["value"], extrema["v_min"]["position"],
            stream.get("psi_min_wall", ""), stream.get("x_wall", ""), stream.get("y_wall", ""),
            analysis["window"][0], analysis["window"][1], analysis["samples_in_window"]]


def e40_rows() -> list[list]:
    """The K = 1 / K = 2 rows of the E40 summary (their full 20-time-unit windows), labelled."""
    rows = []
    if not E40_SUMMARY.exists():
        return rows
    with open(E40_SUMMARY, encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if row and row[0] in ("K1", "K2"):
                rows.append([row[0] + " (E40 full window)"] + [float(value) if value else "" for value in row[1:18]]
                            + [int(row[18]) if row[18] else ""])
    return rows


def point_rows(analyses: dict, reference: dict, first: str, second: str) -> list[dict]:
    rows = []
    for line, coordinate_name in (("u", "y"), ("v", "x")):
        coordinates, values, _ = reference[line]
        one = analyses[first]["statistics"][f"{line}_marchi_wall_mls"]
        two = analyses[second]["statistics"][f"{line}_marchi_wall_mls"]
        for index, coordinate in enumerate(coordinates):
            difference = float(two["mean"][index] - one["mean"][index])
            sigma = math.sqrt(float(one["sem"][index]) ** 2 + float(two["sem"][index]) ** 2)
            rows.append({"pair": f"{second}-{first}", "line": line, "coordinate_name": coordinate_name,
                         "coordinate": float(coordinate), "marchi2021_Tc": float(values[index]),
                         f"mean_{first}": float(one["mean"][index]), f"sem_{first}": float(one["sem"][index]),
                         f"mean_{second}": float(two["mean"][index]), f"sem_{second}": float(two["sem"][index]),
                         "difference": difference, "sigma": sigma,
                         "difference_over_sigma": difference / sigma if sigma > 0 else float("nan")})
    return rows


def point_summary(rows: list[dict]) -> dict:
    differences = np.array([row["difference"] for row in rows])
    ratios = np.array([abs(row["difference_over_sigma"]) for row in rows if math.isfinite(row["difference_over_sigma"])])
    return {"max_abs_difference": float(np.abs(differences).max()), "rms_difference": float(np.sqrt(np.mean(differences ** 2))),
            "max_abs_sigma_ratio": float(ratios.max()), "points_over_2_sigma": int((ratios > 2).sum()),
            "points_over_3_sigma": int((ratios > 3).sum()), "points": int(ratios.size)}


def invariants(run: dict, key: str, restricted: dict | None = None) -> dict:
    """Totals of the segments (whole run) and the sample-wise checks (over the given rows)."""
    totals = {"drift": 0, "overflow": 0, "far_migration": 0, "stamp_gpu": 0, "stamp_host": 0}
    for segment in run["segments"]:
        for name in totals:
            totals[name] += int((segment.get("totals") or {}).get(name, 0))
    rows = (restricted or run)["rows"]
    alive = [int(row["alive"]) for row in rows]
    fps = sorted(float(row["fps"]) for row in rows[1:] if row.get("fps"))
    return {"run": key, "id": run["id"], "expected_particles": int(run["meta"]["expected_total"]),
            "alive_min": min(alive), "alive_max": max(alive), "samples": len(alive),
            "segments": len(run["segments"]),
            "segments_complete": sum(1 for segment in run["segments"] if segment.get("status") == "complete"),
            "invariant_violations": sum(1 for segment in run["segments"] if segment.get("invariant_violation")),
            "wall_hours_run": sum(float(segment.get("wall_s", 0.0)) for segment in run["segments"]) / 3600.0,
            # wall clock between the first and the last sample of the window (the rows' wall_s counts from the
            # segment start; one segment in all three runs)
            "wall_s_window": float(rows[-1].get("wall_s", 0.0)) - float(rows[0].get("wall_s", 0.0)),
            "steps_window": int(rows[-1]["step"] - rows[0]["step"]),
            "median_fps_window": fps[len(fps) // 2] if fps else None,
            "mls_fallbacks_total": sum(int(row.get("mls_fallbacks", 0)) for row in rows),
            "min_neighbours": min(int(row.get("min_neighbours", 0)) for row in rows),
            "status": run["result"].get("status", ""), "git_head": run["meta"].get("git", {}).get("head"),
            "devices": run["meta"].get("device_map"), "slabs": int(run["meta"]["slabs"]), **totals}


# ----------------------------------------------------------------------------------------------- figures
def profile_field_figure(analysis: dict, field: dict | None, path: pathlib.Path, seams_wall: list[float], label: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    setup_style()
    reference = cavity_reference.marchi2021()
    ghia = cavity_reference.ghia1982()
    dense = analysis["dense_reference"]
    u_mean = analysis["statistics"]["u_dense_mls"]["mean"]
    v_mean = analysis["statistics"]["v_dense_mls"]["mean"]
    panels = 3 if field is not None else 2
    figure = plt.figure(figsize=(7.2, 2.55) if panels == 3 else (4.9, 2.55))
    grid_spec = figure.add_gridspec(1, panels, width_ratios=(1.0, 1.0, 1.18)[:panels], wspace=0.42, left=0.07,
                                    right=0.985, bottom=0.17, top=0.93)
    axes = [figure.add_subplot(grid_spec[0, index]) for index in range(panels)]
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
    low, high = axes[1].get_ylim()
    for index, seam in enumerate(seams_wall):
        axes[1].axvline(seam, color="#555555", lw=0.6, ls=(0, (3, 2)))
    # the label next to the middle seam, at the bottom (right of the inset, below the curve)
    middle = seams_wall[len(seams_wall) // 2]
    axes[1].text(middle + 0.012, low + 0.03 * (high - low), "seams", ha="left", va="bottom", fontsize=6.5, color="#555555")
    if field is not None:
        image = axes[2].imshow(field["field"], origin="lower", extent=field["extent"], cmap="viridis",
                               norm=Normalize(0.0, field["percentiles"][99]), interpolation="nearest", aspect="equal")
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel(r"$x$")
        axes[2].set_ylabel(r"$y$")
        for seam in seams_wall:
            axes[2].axvline(seam, color="white", lw=0.6, ls=(0, (3, 2)))
        colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03)
        colorbar.set_label(r"$|\mathbf{a}|\,L/U^2$ at $t = %.0f$" % field["time"], fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5)
    for axis, tag in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left", color="white" if tag == "(c)" else "black")
        if tag != "(c)":
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def difference_figure(pairs: dict[str, list[dict]], strips: list[dict], seam_names: list[str], control_names: list[str],
                      support: float, path: pathlib.Path) -> None:
    """(a) difference over sigma at the 56 points per pair; (b) rms |delta u| per h/2 column of every seam strip, the
    control strips as a grey band (min-max over the 6 controls)."""
    import matplotlib.pyplot as plt
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.32, "left": 0.08, "right": 0.985,
                                                                        "bottom": 0.18, "top": 0.9})
    markers = {"K8-K1": ("o", "#08519c"), "K2-K1": ("s", "#b2182b"), "K8-K2": ("^", "#4d9221")}
    for pair, rows in pairs.items():
        marker, color = markers.get(pair, ("x", "k"))
        x = [index for index, _ in enumerate(rows)]
        y = [row["difference_over_sigma"] for row in rows]
        axes[0].plot(x, y, marker, mfc="none", mec=color, ms=3.2, mew=0.7, label=pair.replace("-", " − ").replace("K", "K = "))
    axes[0].axhline(0, color="k", lw=0.5)
    for level in (-2, 2):
        axes[0].axhline(level, color="#888888", lw=0.5, ls=(0, (3, 2)))
    axes[0].axvline(27.5, color="#888888", lw=0.5)
    largest = max(abs(row["difference_over_sigma"]) for rows in pairs.values() for row in rows)
    top = max(2.6, largest + 0.4)
    axes[0].set_ylim(-top - 0.9, top)              # room for the legend below the data and the labels above
    axes[0].text(13.5, top - 0.12, r"$u(0.5, y)$", ha="center", va="top", fontsize=7)
    axes[0].text(41.5, top - 0.12, r"$v(x, 0.5)$", ha="center", va="top", fontsize=7)
    axes[0].set_xlabel("reference point (28 + 28, Marchi et al. 2021)")
    axes[0].set_ylabel(r"$\Delta / \sigma$")
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower center", ncol=3, columnspacing=0.9, handletextpad=0.3)
    by_name = {strip["strip"]: strip for strip in strips}
    offsets = [item["offset_h"] for item in by_name[seam_names[0]]["bins"]]
    controls = np.array([[item["rms_difference"] for item in by_name[name]["bins"]] for name in control_names])
    axes[1].fill_between(offsets, controls.min(axis=0), controls.max(axis=0), color="#bbbbbb", alpha=0.6, lw=0,
                         label="control strips (6)")
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(seam_names)))
    for name, color in zip(seam_names, colors):
        axes[1].plot(offsets, [item["rms_difference"] for item in by_name[name]["bins"]], "-", color=color, lw=0.8,
                     label=name.replace("seam", "seam "))
    axes[1].set_yscale("log")
    all_values = np.array([[item["rms_difference"] for item in strip["bins"]] for strip in strips])
    axes[1].set_ylim(all_values.min() * 0.85, all_values.max() * 2.2)     # the legend sits above the data
    axes[1].set_xlabel(r"offset from the seam ($h$)")
    axes[1].set_ylabel(r"rms $|\Delta \mathbf{u}|$, K = 8 − K = 1 ($U$)")
    axes[1].legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.4)
    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--k1", default=RUN_IDS["K1"])
    parser.add_argument("--k2", default=RUN_IDS["K2"])
    parser.add_argument("--k8", default=RUN_IDS["K8"])
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--no-stream-function", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="recompute a cached analysis")
    parser.add_argument("--grid", type=int, default=1024, help="cells per side of the |a| field")
    parser.add_argument("--out-prefix", default="adami_k8")
    parser.add_argument("--doc", default=None, help="markdown output (default docs/validation/<prefix>.md)")
    parser.add_argument("--job-json", default=None, help="facts of the SLURM job (default <k8 run dir>/job.json)")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.runs_root).resolve()
    prefix = arguments.out_prefix
    doc_path = pathlib.Path(arguments.doc) if arguments.doc else _REPO_ROOT / "docs" / "validation" / f"{prefix}.md"
    reference = cavity_reference.marchi2021()

    runs = {}
    for key, run_id in (("K1", arguments.k1), ("K2", arguments.k2), ("K8", arguments.k8)):
        run = load_run(run_id, root)
        if run is None:
            sys.exit(f"{root / run_id}: no samples.jsonl")
        runs[key] = run
        print(f"[k8] {key} = {run_id}: {len(run['rows'])} samples, t {run['times'].min():.3f}..{run['times'].max():.4f}", flush=True)
    k8_steps = {int(row["step"]) for row in runs["K8"]["rows"]}
    windows = {"K8": runs["K8"]}
    for key in ("K1", "K2"):
        windows[key] = restrict_to_steps(runs[key], k8_steps)
        steps = {int(row["step"]) for row in windows[key]["rows"]}
        if steps != k8_steps:
            sys.exit(f"{key}: sample steps differ from K8 (missing {sorted(k8_steps - steps)[:5]} ...)")
    start_from = runs["K8"]["meta"].get("start_from") or {}
    print(f"[k8] common window: {len(k8_steps)} samples, steps {min(k8_steps):,}..{max(k8_steps):,}; K8 started from "
          f"{start_from.get('source_run')} step {start_from.get('source_step')} (t = {start_from.get('source_time')})", flush=True)

    analyses = {}
    for key in ("K1", "K2", "K8"):
        run = windows[key]
        cache_path = root / f"analysis_{prefix}_{run['id']}_{len(run['rows'])}.npy"
        if cache_path.exists() and not arguments.refresh and not arguments.no_stream_function:
            analysis = np.load(cache_path, allow_pickle=True).item()
            print(f"[k8] {key}: analysis from {cache_path.name}", flush=True)
        else:
            analysis = analyse_window(run, arguments.steady_tol, arguments.window, not arguments.no_stream_function)
            if not arguments.no_stream_function:
                np.save(cache_path, np.array(analysis, dtype=object), allow_pickle=True)
        analyses[key] = analysis
        errors = analysis["errors"]
        print(f"[k8] {key}: window {analysis['window']}, rel L2 u {errors[('wall', 'mls', 'u')]['L2_relative']:.5f}, "
              f"v {errors[('wall', 'mls', 'v')]['L2_relative']:.5f}, stream {analysis.get('stream')}", flush=True)
    # the K = 8 run's own profile CSV (its whole run is the window); K1 / K2 keep their E40 profile CSVs
    profile_csv(analyses["K8"])

    # A: summary
    header = ["run", "L2_u", "Linf_u", "L2_v", "Linf_v", "relL2_u", "relL2_v", "u_min", "y_at_u_min", "v_max", "x_at_v_max",
              "v_min", "x_at_v_min", "psi_min", "x_at_psi_min", "y_at_psi_min", "window_start", "window_end", "samples"]
    rows = [["Marchi2021", "", "", "", "", "", "", reference["extrema"]["u_min"][0], reference["extrema"]["y_at_u_min"][0],
             reference["extrema"]["v_max"][0], reference["extrema"]["x_at_v_max"][0], reference["extrema"]["v_min"][0],
             reference["extrema"]["x_at_v_min"][0], reference["extrema"]["psi_min"][0],
             reference["extrema"]["x_at_psi_min"][0], reference["extrema"]["y_at_psi_min"][0], "", "", ""]]
    rows += [summary_row(key, analyses[key], reference) for key in ("K1", "K2", "K8")]
    rows += e40_rows()
    write_csv(DATA / f"{prefix}_summary.csv", header, rows,
              "6.2 supplement: K = 1 / K = 2 / K = 8 of the 1000^2 adami cavity on the common window t >= 89.5 (the K = 8\n"
              "continuation's samples), errors at the Marchi 2021 points (wall frame, MLS, absolute in units of U;\n"
              "relL2 = L2 / rms(Tc)), extrema from the time-averaged dense lines, primary vortex from the time-averaged\n"
              "MLS grid of the window's snapshots; the '(E40 full window)' rows repeat adami_k1_k2_summary.csv.")

    # B: points
    pairs = {pair: point_rows(analyses, reference, first, second)
             for pair, (first, second) in (("K8-K1", ("K1", "K8")), ("K2-K1", ("K1", "K2")), ("K8-K2", ("K2", "K8")))}
    point_summaries = {pair: point_summary(rows_) for pair, rows_ in pairs.items()}
    all_point_rows = [row for rows_ in pairs.values() for row in rows_]
    keys = ["pair", "line", "coordinate_name", "coordinate", "marchi2021_Tc", "difference", "sigma", "difference_over_sigma"]
    write_csv(DATA / f"{prefix}_point_differences.csv", keys + ["means_and_sems"],
              [[row[key] for key in keys] + [json.dumps({k: v for k, v in row.items() if k.startswith(("mean_", "sem_"))})]
               for row in all_point_rows],
              "6.2 supplement: time-averaged MLS velocity differences at the 28 + 28 Marchi 2021 points (wall frame) on\n"
              "the common window; sem = std * sqrt(tau_int / n) per run; sigma = sqrt(sem_first^2 + sem_second^2).")

    # C: seams and strips
    support = float(runs["K8"]["meta"]["support_radius"])
    spacing = runs["K8"]["spacing"]
    seams = seam_positions(runs["K8"])
    if len(seams) != int(runs["K8"]["meta"]["slabs"]) - 1:
        sys.exit(f"K8 records {len(seams)} cuts for {runs['K8']['meta']['slabs']} slabs")
    strips_points, strip_offsets = {}, None
    for index, seam in enumerate(seams, start=1):
        strips_points[f"seam{index}"], strip_offsets = strip_points(seam["x"], support, spacing)
    controls = []
    for index in range(len(seams) - 1):
        x_centre = 0.5 * (seams[index]["x"] + seams[index + 1]["x"])
        controls.append(x_centre)
        strips_points[f"control{index + 1}"], _ = strip_points(x_centre, support, spacing)
    window = analyses["K8"]["window"]
    print(f"[k8] strips: {len(strips_points)} x {next(iter(strips_points.values())).shape[0]} points, window {window}", flush=True)
    averages = {key: strip_averages(windows[key], window, strips_points, support) for key in ("K1", "K8")}
    if averages["K1"]["_snapshots"] != averages["K8"]["_snapshots"]:
        sys.exit(f"snapshot sets differ: {averages['K1']['_snapshots']} vs {averages['K8']['_snapshots']}")
    strips = []
    for name, points in strips_points.items():
        x_centre = seams[int(name[4:]) - 1]["x"] if name.startswith("seam") else controls[int(name[7:]) - 1]
        strips.append(strip_table(name, x_centre, points, strip_offsets, averages["K1"][name], averages["K8"][name], support))
    seam_names = [name for name in strips_points if name.startswith("seam")]
    control_names = [name for name in strips_points if name.startswith("control")]
    write_csv(DATA / f"{prefix}_seams.csv",
              ["seam", "column", "x_m", "x_wall", "slab_left_particles", "slab_right_particles"],
              [[index + 1, seam["column"], f"{seam['x']:.6f}", f"{seam['x_wall']:.6f}",
                runs["K8"]["meta"]["slab_particles"][index], runs["K8"]["meta"]["slab_particles"][index + 1]]
               for index, seam in enumerate(seams)],
              "6.2 supplement: the 7 seams of the K = 8 continuation (global voxel column of the cut, x in metres of the\n"
              "case frame, X in the wall-row frame) and the particles of the slabs on either side at the start.")
    write_csv(DATA / f"{prefix}_strips.csv",
              ["strip", "x_centre_m", "points", "rms_abs_du", "max_abs_du", "rms_abs_du_inner_half_h", "max_abs_du_inner_half_h",
               "rms_sem_K1", "rms_sem_K8", "mean_speed_K1"],
              [[s["strip"], f"{s['x_centre']:.6f}", s["points"], f"{s['rms_all']:.3e}", f"{s['max_all']:.3e}",
                f"{s['rms_inner_half_h']:.3e}", f"{s['max_inner_half_h']:.3e}", f"{s['rms_sem_first']:.3e}",
                f"{s['rms_sem_second']:.3e}", f"{s['mean_speed_first']:.4f}"] for s in strips],
              f"6.2 supplement: time-averaged MLS velocity (the window's {averages['K8']['_snapshots'].__len__()} snapshots per run)\n"
              "on strips of 3 support radii on either side of every seam (columns every h/2, rows every h/2), K = 8 minus\n"
              "K = 1; control strips half way between neighbouring seams; sem over the snapshots (no autocorrelation\n"
              "correction); 'inner_half_h' = the two columns nearest the seam.")
    write_csv(DATA / f"{prefix}_strip_bins.csv",
              ["strip", "offset_h", "points", "rms_abs_du", "max_abs_du", "rms_sem_K1", "rms_sem_K8", "mean_speed_K1"],
              [[s["strip"], f"{item['offset_h']:+.2f}", item["points"], f"{item['rms_difference']:.3e}", f"{item['max_difference']:.3e}",
                f"{item['rms_sem_first']:.3e}", f"{item['rms_sem_second']:.3e}", f"{item['mean_speed']:.4f}"]
               for s in strips for item in s["bins"]],
              "6.2 supplement: the strip difference per h/2 column (offset from the strip centre in support radii).")
    e40 = json.loads(E40_REPORT.read_text(encoding="utf-8")) if E40_REPORT.exists() else {}

    # D: invariants, throughput, wall clock
    facts = [invariants(runs["K8"], "K8"), invariants(runs["K1"], "K1", windows["K1"]), invariants(runs["K2"], "K2", windows["K2"])]
    write_csv(DATA / f"{prefix}_invariants.csv", list(facts[0].keys()), [list(row.values()) for row in facts],
              "6.2 supplement: invariants (segment totals of the whole run; alive / MLS fallbacks / neighbours over the\n"
              "window's samples), throughput (median of the sample-wise steps/s in the window) and wall clock.")
    job_path = pathlib.Path(arguments.job_json) if arguments.job_json else runs["K8"]["dir"] / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}

    # E: figures
    seams_wall = [seam["x_wall"] for seam in seams]
    field = None
    if sorted((runs["K8"]["dir"] / "checkpoints").glob("c*.npz")):
        field = acceleration_field(runs["K8"], arguments.grid)
    profile_field_figure(analyses["K8"], field, FIGURES / f"{prefix}_fig1_k8.pdf", seams_wall, "$K = 8$")
    difference_figure(pairs, strips, seam_names, control_names, support, FIGURES / f"{prefix}_fig2_differences.pdf")

    # the document
    k8 = runs["K8"]
    k8_facts = facts[0]
    hashes_note = ("物理代码哈希:K = 8 的 meta.json 记的是 Linux 部署树(LF 换行)的哈希,与 K = 1 / K = 2(Windows 检出,CRLF)"
                   "不同;逐文件比较(41 个文件:experiment/v7/utils/*.py、SPIR-V、算例、材料库,CRLF→LF 归一化后)完全一致,"
                   "部署树的 git tree hash = tag v7-rc3 的 experiment/v7。")
    lines = [f"# Adami 壁面 1000² 方腔:K = 8 续算(6.2 节补充,2026-10-21)", "",
             f"算例 `{k8['case']}`(Re = 1000,ξ = 0.001,ε² = 0.0025 h²,`wall_boundary: adami`),求解器 experiment/v7 = tag v7-rc3。"
             f"K = 8 运行在 N56 集群一整台节点(8 × RTX 5090,SLURM 作业 {job.get('job_id', '1719093')},节点 {job.get('node', '?')},"
             f"部署 `~/run/vulkan-demo-v7rc3` = 提交 {job.get('deploy_commit', '?')}),`cavity_runner --solver v7 --slabs 8 --start-from` "
             f"从 K = 1 运行的检查点 `{pathlib.Path(start_from.get('file', '')).name}`(步 {start_from.get('source_step', 0):,},"
             f"t = {start_from.get('source_time', 0):.4f},sha256 {start_from.get('sha256', '')[:12]}…)启动,全局步号延续,"
             f"等权重按状态的流体直方图切成 8 个 slab,推进到步 {k8['result'].get('stop_step', 0):,}"
             f"(t = {k8['result'].get('stop_step', 0) * float(k8['meta']['dt']):.3f},与 K = 1 / K = 2 的停止步相同)。K = 1、K = 2 为 docs/validation/adami_k1_k2.md 的两次运行(本地 5090 / 5090 + 7900 XTX)。"
             f"三者在**同一组取样步**上比较:t ≥ {window[0]:.3f} 的 {len(k8_steps)} 个样本(每 0.2 时间单位)与 "
             f"{len(averages['K8']['_snapshots'])} 个快照(每 ~1 时间单位,步号相同)。方法、稳态判据、取样与参考解沿用 "
             "`docs/validation/cavity_re1000.md`。只报结果。", "", hashes_note, ""]
    lines += ["## 图", "", f"![fig1](figures/{prefix}_fig1_k8.png)", "",
              f"图 1(K = 8):(a) u(0.5, y) 与 (b) v(x, 0.5) 的时间平均 MLS 中线剖面(壁面行框架)对 Marchi 2021(圆圈)与 Ghia 1982(方块),"
              f"(b) 中的 7 条竖虚线 = 切口;(c) 流体加速度大小 |a| L/U²,末尾检查点"
              + (f"(t = {field['time']:.1f})的 Shepard 插值,{arguments.grid}² 网格,色标上限 = 99 百分位 {field['percentiles'][99]:.3g},白色虚线 = 切口。"
                 if field else "(无检查点时省略)。"), "",
              f"![fig2](figures/{prefix}_fig2_differences.png)", "",
              "图 2:(a) 56 个参考点上两次运行时间平均速度之差除以 σ(各自标准误差的平方和开方);(b) 每条接缝条带上 K = 8 − K = 1 的 "
              "rms |Δu| 按离切口的 h/2 列,灰带 = 6 条对照条带(相邻切口中点)的最小–最大范围。", ""]
    lines += ["## 表 A:误差、极值、主涡(公共窗口)", ""]
    table_rows = []
    for row in rows:
        table_rows.append([str(row[0])] + [number(value) for value in row[1:16]] + [number(row[16], 2), number(row[17], 2), str(row[18])])
    lines.append(md_table(["run", "L2 u", "L∞ u", "L2 v", "L∞ v", "rel L2 u", "rel L2 v", "u_min", "y(u_min)", "v_max", "x(v_max)",
                           "v_min", "x(v_min)", "ψ_min", "x(ψ_min)", "y(ψ_min)", "窗口起", "窗口止", "样本数"], table_rows))
    lines += ["", "L2 = √(Σ e_i² / 28)、L∞ = max |e_i|(绝对值,U 为单位),rel L2 = L2 / rms(T_c);极值位置为壁面行框架;"
              f"ψ_min 与主涡中心来自窗口内快照的 MLS 网格平均(513²)。“E40 full window” 两行是 adami_k1_k2.md 的原值(各自最后 20 个时间单位)。"
              f"数据:`data/{prefix}_summary.csv`、`data/profile_{k8['id']}.csv`。", ""]
    lines += ["## 表 B:56 个参考点上的差", ""]
    lines.append(md_table(["对", "max |Δ|", "rms Δ", "max |Δ|/σ", "|Δ|/σ > 2", "|Δ|/σ > 3"],
                          [[pair.replace("-", " − "), f"{s['max_abs_difference']:.2e}", f"{s['rms_difference']:.2e}",
                            f"{s['max_abs_sigma_ratio']:.2f}", str(s["points_over_2_sigma"]), str(s["points_over_3_sigma"])]
                           for pair, s in point_summaries.items()]))
    lines += ["", f"Δ = 时间平均 MLS 速度之差,s.e. = std·√(τ_int/n)(n = {len(k8_steps)},样本间隔 0.2 时间单位),σ = √(s.e.₁² + s.e.₂²)。"
              f"数据:`data/{prefix}_point_differences.csv`。", ""]
    lines += ["## 表 C:7 条接缝与条带差", ""]
    lines.append(md_table(["接缝", "全局列", "x (m)", "X 壁面框架", "左 slab 粒子", "右 slab 粒子"],
                          [[str(index + 1), str(seam["column"]), f"{seam['x']:+.4f}", f"{seam['x_wall']:.4f}",
                            f"{k8['meta']['slab_particles'][index]:,}", f"{k8['meta']['slab_particles'][index + 1]:,}"]
                           for index, seam in enumerate(seams)]))
    lines += ["", f"切口 = 全局体素列边界(列宽 h = {support} m),按启动状态的流体直方图等权切分(切口列 {', '.join(str(s['column']) for s in seams)};"
              f"K = 2 的切口为第 {e40.get('seam', {}).get('column', '?')} 列 = 这里的第 4 条接缝)。", ""]
    lines.append(md_table(["条带", "中心 x (m)", "点数", "rms |Δu|", "max |Δu|", "rms |Δu| (±0.5h)", "max |Δu| (±0.5h)", "rms s.e. K1", "rms s.e. K8", "平均速度 K1"],
                          [[s["strip"], f"{s['x_centre']:+.4f}", str(s["points"]), f"{s['rms_all']:.2e}", f"{s['max_all']:.2e}",
                            f"{s['rms_inner_half_h']:.2e}", f"{s['max_inner_half_h']:.2e}", f"{s['rms_sem_first']:.2e}",
                            f"{s['rms_sem_second']:.2e}", f"{s['mean_speed_first']:.3f}"] for s in strips]))
    seam_rms = [s["rms_all"] for s in strips if s["strip"].startswith("seam")]
    control_rms = [s["rms_all"] for s in strips if s["strip"].startswith("control")]
    e40_strip = e40.get("strips", {})
    lines += ["", f"K = 8 − K = 1 的时间平均 MLS 速度(窗口内 {len(averages['K8']['_snapshots'])} 个快照)在条带 |x − x_c| ≤ 3h 上,"
              f"点列每 h/2 一列、每行 h/2;接缝条带 rms |Δu| {min(seam_rms):.2e}–{max(seam_rms):.2e},对照条带 {min(control_rms):.2e}–{max(control_rms):.2e}"
              + (f";E40 的 K = 2 − K = 1 在其切口条带 rms {e40_strip.get('seam', {}).get('rms_all', float('nan')):.2e}、"
                 f"对照 {e40_strip.get('control', {}).get('rms_all', float('nan')):.2e}。" if e40_strip else "。")
              + f" 快照 MLS 回退:K1 {averages['K1']['_fallbacks']}、K8 {averages['K8']['_fallbacks']}。"
              f"数据:`data/{prefix}_seams.csv`、`{prefix}_strips.csv`、`{prefix}_strip_bins.csv`。", ""]
    lines += ["## 表 D:不变量、吞吐、墙钟", ""]
    lines.append(md_table(["run", "K", "设备", "粒子数(期望)", "存活(窗口样本 min–max)", "overflow", "far_migration", "帧戳错误 GPU / 主机", "drift",
                           "段(完成 / 违反)", "窗口步数", "窗口墙钟 s", "步/s(窗口中位数)", "MLS 回退", "最少邻居", "状态"],
                          [[row["run"], str(row["slabs"]), str(row["devices"]), f"{row['expected_particles']:,}",
                            f"{row['alive_min']:,}–{row['alive_max']:,}", str(row["overflow"]), str(row["far_migration"]),
                            f"{row['stamp_gpu']} / {row['stamp_host']}", str(row["drift"]),
                            f"{row['segments']}({row['segments_complete']} / {row['invariant_violations']})",
                            f"{row['steps_window']:,}", f"{row['wall_s_window']:.0f}", number(row["median_fps_window"], 1),
                            str(row["mls_fallbacks_total"]), str(row["min_neighbours"]), row["status"]] for row in facts]))
    smoke = job.get("smoke", {})
    segment_steps = int(k8["result"].get("stop_step", 0)) - int(start_from.get("source_step", 0))
    segment_wall = float(k8["segments"][0].get("wall_s", float("nan")))
    lines += ["", "每个样本(每 0.2 时间单位)与每个 defrag 边界都检查:存活粒子总数 = 初始总数、每张卡的全部 `overflow_*` 计数、far_migration、"
              "GPU 与主机帧戳错误;任一违反即作废。K = 1 / K = 2 的段合计为整次运行(E40),存活 / 回退 / 邻居为公共窗口的样本;"
              "窗口墙钟 = 窗口首末样本之间的墙钟(不含启动),步/s = 窗口内逐样本吞吐的中位数。", "",
              f"K = 8 作业:{job.get('submitted', '?')} 提交,排队 {job.get('queued', '?')},{job.get('start', '?')} 在 {job.get('node', '?')} 开始,"
              f"{job.get('end', '?')} 结束,占用 {job.get('elapsed', '?')}(8 卡,时限 {job.get('time_limit', '?')}):"
              f"{smoke.get('steps', 27000):,} 步的 K = 8 冒烟段 {smoke.get('wall_s', float('nan')):.0f} s(含启动;不变量全 0),"
              f"正式段 {segment_wall:.0f} s 完成 {segment_steps:,} 步(≈ {segment_steps / segment_wall:.0f} 步/s 含约 1 min 启动;"
              f"提交前估算 {job.get('estimate_fps', '?')} 步/s → {job.get('estimate_hours', '?')} h),其余为环境解包、checkpoint 校验与分区检查。", ""]
    lines += ["## 文件", "", f"- 图:`docs/validation/figures/{prefix}_fig1_k8.pdf`(.png)、`{prefix}_fig2_differences.pdf`(.png)",
              f"- 数据:`docs/validation/data/{prefix}_summary.csv`、`{prefix}_point_differences.csv`、`{prefix}_seams.csv`、"
              f"`{prefix}_strips.csv`、`{prefix}_strip_bins.csv`、`{prefix}_invariants.csv`、`profile_{k8['id']}.csv`、`{prefix}_report.json`",
              f"- 运行目录(不入库):`{root / k8['id']}`(集群归档 `~/run/logs/cavity_k8_{job.get('job_id', '1719093')}`,"
              f"本地镜像 `logs/n56/2026-10-20/02_cav_k8_1719093`)",
              "- 生成:`.venv/Scripts/python.exe -m experiment.validation.adami_k8_report`", ""]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")

    def analysis_summary(key: str) -> dict:
        analysis = analyses[key]
        errors = analysis["errors"]
        return {"window": list(analysis["window"]), "samples": analysis["samples_in_window"],
                "errors": {f"{frame}_{method}_{line}": {k: v for k, v in item.items() if k != "pointwise" and k != "values"}
                           for (frame, method, line), item in errors.items() if method == "mls" and frame == "wall"},
                "extrema": analysis["extrema"], "stream": analysis.get("stream"),
                "kinetic_energy": {k: (float(v[0]) if hasattr(v, "__len__") else float(v))
                                   for k, v in analysis["kinetic_energy_window"].items()}}

    report = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "common_window": list(window),
              "samples": len(k8_steps), "snapshots": averages["K8"]["_snapshots"], "start_from": start_from,
              "job": job, "seams": seams, "slab_particles": k8["meta"]["slab_particles"], "cuts": k8["meta"]["cuts"],
              "summary_header": header, "summary_rows": rows, "analyses": {key: analysis_summary(key) for key in analyses},
              "points": {pair: rows_ for pair, rows_ in pairs.items()}, "point_summaries": point_summaries,
              "strips": strips, "strip_fallbacks": {key: averages[key]["_fallbacks"] for key in averages},
              "e40_strips": {name: {k: v for k, v in strip.items() if k != "bins"} for name, strip in e40_strip.items()},
              "invariants": facts, "field": {k: v for k, v in (field or {}).items() if k != "field"},
              "reference": {"u_min": reference["extrema"]["u_min"][0], "y_at_u_min": reference["extrema"]["y_at_u_min"][0],
                            "v_max": reference["extrema"]["v_max"][0], "x_at_v_max": reference["extrema"]["x_at_v_max"][0],
                            "v_min": reference["extrema"]["v_min"][0], "x_at_v_min": reference["extrema"]["x_at_v_min"][0],
                            "psi_min": reference["extrema"]["psi_min"][0], "x_at_psi_min": reference["extrema"]["x_at_psi_min"][0],
                            "y_at_psi_min": reference["extrema"]["y_at_psi_min"][0]}}
    json.dump(report, open(DATA / f"{prefix}_report.json", "w", encoding="utf-8"), indent=1, default=str)
    print(f"[k8] wrote {doc_path} and {DATA / f'{prefix}_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
