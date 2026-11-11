"""
Draft figures of Stage 4 (docs/micropolar_design.md, section 13) from couette_stage4.json in this directory.

  figure_a_profiles.png / .pdf   v_theta(r) and w_z(r) of the SPH runs (bin means of the last quarter) against the
                                 exact solution, K = 3 at m = 2 and m = 10 (w = 0 on both walls)
  figure_b_torque.png / .pdf     D against m for K = 0.5, 1, 3: exact solution (lines), SPH (points, the noise
                                 floor of D as error bars), the wall-displacement rule (dashed)

Run from the repository root:  python data/micropolar/stage4/make_figures.py
Plain matplotlib; one quantity per axis; the three K keep one colour and one marker each in both figures.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiment.validation import micropolar_couette_reference as reference          # noqa: E402
from experiment.validation.micropolar_couette import design_parameters               # noqa: E402
from experiment.validation.micropolar_stage4 import rule_torque_change                # noqa: E402

# fixed order: K = 0.5, 1, 3 (categorical slots 1 to 3 of the palette in use; marker as second encoding)
SERIES = {0.5: ("#2a78d6", "o"), 1.0: ("#eb6834", "s"), 3.0: ("#1baf7a", "^")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
INNER_RADIUS, OUTER_RADIUS, GAP, WALL_SPEED, FAR_VISCOSITY, DENSITY = 0.4, 0.5, 0.1, 1.0, 0.01, 1000.0

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def exact_change(coupling_ratio: float, gap_over_two_lengths: float) -> float:
    design = design_parameters(FAR_VISCOSITY, coupling_ratio, GAP / (2.0 * gap_over_two_lengths))
    eringen = reference.eringen_parameters_from_solver_parameters(
        DENSITY, design["viscosity"], design["coupling"], design["viscosity_rotation"])
    return reference.relative_torque_change(INNER_RADIUS, OUTER_RADIUS, eringen.shear_viscosity,
                                            eringen.coupling_viscosity, eringen.spin_viscosity, WALL_SPEED, 0.0,
                                            0.0, 0.0)


def rule_change(coupling_ratio: float, gap_over_two_lengths: float) -> float:
    design = design_parameters(FAR_VISCOSITY, coupling_ratio, GAP / (2.0 * gap_over_two_lengths))
    displacement = design["coupling"] * GAP / (2.0 * gap_over_two_lengths) / (2.0 * design["viscosity"])
    return rule_torque_change(INNER_RADIUS, OUTER_RADIUS, displacement)


def figure_profiles(runs: dict) -> None:
    cases = [name for name in ("g100_k3_m2", "g200_k3_m10") if name in runs and "exact_profile" in runs[name]]
    if not cases:
        print("figure A: runs missing")
        return
    colour = SERIES[3.0][0]
    figure, axes = plt.subplots(2, len(cases), figsize=(max(3.9 * len(cases), 4.6), 5.4), sharex=True,
                                squeeze=False)
    rotation_rate = WALL_SPEED / INNER_RADIUS
    for column, name in enumerate(cases):
        record = runs[name]
        profile, exact = record["profile"], record["exact_profile"]
        radius = (np.asarray(profile["radius"]) - INNER_RADIUS) / GAP
        exact_radius = (np.asarray(exact["radius"]) - INNER_RADIUS) / GAP
        step = max(1, radius.size // 50)                       # about fifty markers across the gap
        for row, (key, scale, label) in enumerate((("velocity", WALL_SPEED, r"$v_\theta\,/\,U$"),
                                                   ("spin", rotation_rate, r"$w_z\,/\,\Omega$"))):
            axis = axes[row][column]
            axis.plot(exact_radius, np.asarray(exact[key]) / scale, color=INK, linewidth=1.2, label="exact")
            axis.plot(radius[::step], np.asarray(profile[key])[::step] / scale, linestyle="none", marker="o",
                      markersize=3.5, markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.0,
                      label="SPH (bins of one dx, every %d shown)" % step)
            if column == 0:
                axis.set_ylabel(label)
            if row == 1:
                axis.set_xlabel(r"$(r - R_1)\,/\,g$")
        axes[0][column].set_title(
            f"K = 3, m = {record['gap_over_two_lengths']:g} (l = {record['length_in_spacings']:g} dx, "
            f"g = {record['gap_in_spacings']} dx)", fontsize=9, color=INK)
    axes[1][0].legend(loc="upper center")
    figure.suptitle("SPH against the exact solution (w = 0 on both walls)", fontsize=10, color=INK)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(HERE / f"figure_a_profiles.{suffix}", dpi=200)
    plt.close(figure)


def figure_torque(runs: dict) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    values = np.geomspace(0.8, 50.0, 200)
    for coupling_ratio, (colour, marker) in SERIES.items():
        exact = np.array([exact_change(coupling_ratio, value) for value in values])
        rule = np.array([rule_change(coupling_ratio, value) for value in values])
        axis.plot(values, exact, color=colour, linewidth=1.6)
        axis.plot(values, rule, color=colour, linewidth=1.2, linestyle=(0, (4, 3)))
        members = [record for record in runs.values()
                   if record.get("coupling_ratio") == coupling_ratio and "sweep" in record.get("sections", [])]
        if members:
            axis.errorbar([record["gap_over_two_lengths"] for record in members],
                          [record["relative_torque_change"] for record in members],
                          yerr=[record["noise_floor_of_change"] or 0.0 for record in members], linestyle="none",
                          marker=marker, markersize=6, markerfacecolor="white", markeredgecolor=colour,
                          markeredgewidth=1.4, ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
        axis.annotate(f"K = {coupling_ratio:g}", (values[-1], exact[-1]), xytext=(4, 0),
                      textcoords="offset points", color=INK, va="center")
    handles = [plt.Line2D([], [], color=MUTED, linewidth=1.6, label="exact solution"),
               plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="displacement rule"),
               plt.Line2D([], [], color=MUTED, linestyle="none", marker="o", markerfacecolor="white",
                          markeredgewidth=1.4, label="SPH (bars: noise floor of D)")]
    axis.legend(handles=handles, loc="lower left")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("m = g / (2 l)")
    axis.set_ylabel(r"D = T / T$_{\mathrm{Newtonian}}$ $-$ 1")
    axis.set_xlim(0.8, 70.0)
    axis.set_title("Relative torque change, w = 0 on both walls", fontsize=10, color=INK)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(HERE / f"figure_b_torque.{suffix}", dpi=200)
    plt.close(figure)


def main() -> int:
    analysis = json.loads((HERE / "couette_stage4.json").read_text(encoding="utf-8"))
    runs = analysis["runs"]
    figure_profiles(runs)
    figure_torque(runs)
    print("figures written to", HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
