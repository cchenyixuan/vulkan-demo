"""
Item 13 - the seepage of the thesis walls (walls that keep rho0, with the density diffusion term acting on
fluid-wall pairs): the velocity into the wall against the wall pressure, (a) around the rotor and the stator of
the eccentric bearing, (b) to (d) at the four walls of the lid-driven cavity; and the table of the four wall
candidates on the three tests (tab_13_candidates). Existing data of Stage 5 (design note 14.4.3, 14.4.4); not
rerun.

Run from the repository root:  python data/micropolar/ch5_draft/fig_13_seepage.py
"""
import re

import matplotlib.pyplot as plt
import numpy as np

import style

SOURCE = {
    "benchmark": "data/micropolar/stage5/benchmark.json",       # micropolar_stage5.py benchmark
    "concentric": "data/micropolar/stage5/concentric.json",     # micropolar_stage5.py concentric
    "cavity": "data/micropolar/stage5/cavity.json",             # micropolar_stage5_cavity.py analyse
    "status": "final (existing data)", "stage": "Stage 5 (design note 14.4.3 and 14.4.4)",
    # the cavity runs record a short commit only; its git describe, for the caption
    "describe": {"c08cb97": "micropolar-base-12-gc08cb97", "5bc8e0d": "thesis-ch5-rc1-10-g5bc8e0d",
                 "bfcc024": "thesis-ch5-rc1-9-gbfcc024"},
    "cavity_mach": 0.01,                                        # lid speed 1, sound speed 100 (the tool's constants)
}
# the four candidates of design note 14.4: the runs of each on the three tests
RUNS = [
    {"candidate": "(a) thesis", "key": "none (thesis configuration)",
     "bearing": {10: "thesis_e60_re10", 1: "thesis_e60_re1"}, "control": "thesis_g100_control",
     "micropolar": "g100_k3_m5", "cavity": {250: "n250_thesis", 500: "n500_thesis"}},
    {"candidate": "(b) diffusion off", "key": "numerics.use_density_diffusion: false",
     "bearing": {10: "nd_e60_re10", 1: "nd_e60_re1"}, "control": "nd_g100_control",
     "micropolar": "nd_g100_k3_m5", "cavity": {250: "n250_nd", 500: "n500_nd"}},
    {"candidate": "(c) fluid pairs only", "key": "numerics.delta_fluid_only: true",
     "bearing": {10: "fd_e60_re10", 1: "fd_e60_re1"}, "control": "fd_g100_newtonian",
     "micropolar": "fd_g100_k3_m5", "cavity": {250: "n250_fd", 500: "n500_fd"}},
    {"candidate": "(d') eos walls", "key": "numerics.wall_density: eos",
     "bearing": {10: "ew_e60_re10", 1: "ew_e60_re1"}, "control": "ew_g100_newtonian",
     "micropolar": "ew_g100_k3_m5", "cavity": {250: "n250_ew", 500: "n500_ew"}},
]
THESIS, TIGHT = RUNS[0], RUNS[2]            # the walls of the figure, and the tight candidate it is compared with
WALLS = ("bottom", "left", "right", "lid")
STEM, TABLE = "fig_13_seepage", "tab_13_candidates"
SCALE = 1.0e3                               # velocities in units of 1e-3 U
FIT = {"color": style.MUTED, "linestyle": "-", "linewidth": 0.8}
SMALL = 3.0
VELOCITY_LABEL = r"$v_n\,/\,U \times 10^{3}$"                 # velocity into the wall at its surface
DIFFERENCE_LABEL = r"$\Delta v_n\,/\,U \times 10^{3}$"        # thesis walls minus tight walls


def cloud(index: int) -> dict:
    """Open markers (every run of this item is Newtonian), small: the panels hold 72 to 200 points per series."""
    return {**style.run_markers(index, filled=False), "markersize": SMALL, "markeredgewidth": 0.6}


def bearing_panel(axis, benchmark: dict) -> dict:
    numbers = {}
    series = [(THESIS["bearing"][10], "rotor", r"$\mathit{Re} = 10$, rotor"),
              (THESIS["bearing"][10], "stator", r"$\mathit{Re} = 10$, stator"),
              (THESIS["bearing"][1], "rotor", r"$\mathit{Re} = 1$, rotor"),
              (THESIS["bearing"][1], "stator", r"$\mathit{Re} = 1$, stator")]
    for index, (name, wall, label) in enumerate(series):
        entry = benchmark[name]["walls"][wall]
        pressure = np.asarray(entry["wall_pressure_over_dynamic_pressure"])
        velocity = np.asarray(entry["velocity_into_wall_over_wall_speed"])
        axis.plot(pressure, SCALE * velocity, label=label, zorder=2, **cloud(index))
        ends = np.array([pressure.min(), pressure.max()])
        axis.plot(ends, SCALE * (entry["intercept"] + entry["slope"] * ends), zorder=3, **FIT)
        numbers[f"{name}, {wall}"] = {
            "slope": entry["slope"], "intercept": entry["intercept"], "correlation": entry["correlation"],
            "slope_over_mach": entry["slope_over_mach"], "velocity_min": entry["velocity_min"],
            "velocity_max": entry["velocity_max"], "pressure_min_over_dynamic_pressure": float(pressure.min()),
            "pressure_max_over_dynamic_pressure": float(pressure.max()), "bins": int(pressure.size),
            "states": benchmark[name]["walls"]["states"]}
    name = TIGHT["bearing"][10]
    pressure = np.concatenate([benchmark[name]["walls"][wall]["wall_pressure_over_dynamic_pressure"]
                               for wall in ("rotor", "stator")])
    velocity = np.concatenate([benchmark[name]["walls"][wall]["velocity_into_wall_over_wall_speed"]
                               for wall in ("rotor", "stator")])
    axis.plot(pressure, SCALE * velocity, label=r"tight (c), $\mathit{Re} = 10$", zorder=1, **cloud(4))
    for wall in ("rotor", "stator"):
        entry = benchmark[name]["walls"][wall]
        numbers[f"{name}, {wall}"] = {"slope": entry["slope"], "correlation": entry["correlation"],
                                      "velocity_min": entry["velocity_min"], "velocity_max": entry["velocity_max"]}
    axis.set_xlabel(style.math(style.PRESSURE))
    axis.set_ylabel(VELOCITY_LABEL)
    return numbers


def cavity_points(cavity: dict, resolution: int) -> dict:
    """Per wall: the wall pressure of the thesis run, its velocity into the wall, and that minus the tight run's."""
    thesis = cavity[THESIS["cavity"][resolution]]
    tight = cavity[TIGHT["cavity"][resolution]]
    points = {}
    for wall in WALLS:
        pressure = np.asarray(thesis["walls"][wall]["wall_pressure"])
        velocity = np.asarray(thesis["walls"][wall]["velocity_into_wall"])
        points[wall] = {"pressure": pressure, "velocity": velocity,
                        "difference": velocity - np.asarray(tight["walls"][wall]["velocity_into_wall"])}
    pressure = np.concatenate([points[wall]["pressure"] for wall in WALLS])
    difference = np.concatenate([points[wall]["difference"] for wall in WALLS])
    slope, intercept = np.polyfit(pressure, difference, 1)
    stored = tight["difference_to_thesis"]["all_walls"]
    if abs(slope - stored["slope"]) > 1e-9:
        raise RuntimeError(f"n = {resolution}: slope of the difference {slope} is not the stored {stored['slope']}")
    points["fit"] = {"slope": float(slope), "intercept": float(intercept), "correlation": stored["correlation"],
                     "rms": stored["rms"], "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
                     "difference_min": float(difference.min()), "difference_max": float(difference.max())}
    return points


def figure_and_numbers(benchmark: dict, cavity: dict) -> dict:
    style.setup()
    figure, axes = plt.subplots(2, 2, figsize=(style.WIDE, style.WIDE * 0.76))
    numbers = {"bearing": bearing_panel(axes[0, 0], benchmark), "cavity": {}}
    axes[0, 0].set_xlim(-3.2, 2.7)
    axes[0, 0].set_ylim(-10.0, 19.0)
    axes[0, 0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3)

    raw = axes[0, 1]
    for index, resolution in enumerate((250, 500)):
        points = cavity_points(cavity, resolution)
        thesis = cavity[THESIS["cavity"][resolution]]
        tight = cavity[TIGHT["cavity"][resolution]]
        pressure = np.concatenate([points[wall]["pressure"] for wall in WALLS])
        velocity = np.concatenate([points[wall]["velocity"] for wall in WALLS])
        raw.plot(pressure, SCALE * velocity, label=rf"$n = {resolution}$", zorder=2 + index, **cloud(index))
        axis = axes[1, index]
        fit = points["fit"]
        ends = np.array([fit["pressure_min"], fit["pressure_max"]])
        axis.plot(ends, SCALE * (fit["intercept"] + fit["slope"] * ends), zorder=1, **FIT)
        for wall_index, wall in enumerate(WALLS):       # the short clusters of the bottom and left walls on top
            axis.plot(points[wall]["pressure"], SCALE * points[wall]["difference"], label=wall,
                      zorder=6 - wall_index, **cloud(wall_index))
        axis.text(0.96, 0.06, rf"$n = {resolution}$" + "\n" + rf"slope ${fit['slope']:.4f}$", transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=style.FONT_SIZE - 1.0)
        axis.set_xlim(-0.12, 0.38)
        axis.set_ylim(-2.2, 4.8)
        axis.set_xlabel(style.math(style.PRESSURE))
        axis.set_ylabel(DIFFERENCE_LABEL)
        numbers["cavity"][f"n = {resolution}"] = {
            "thesis_run": thesis["run"], "tight_run": tight["run"],
            "thesis_commit": thesis["git"]["head"], "tight_commit": tight["git"]["head"],
            "thesis_snapshots": thesis["walls"]["snapshots"], "thesis_snapshot_times": thesis["snapshot_times"],
            "tight_snapshots": tight["walls"]["snapshots"], "tight_snapshot_times": tight["snapshot_times"],
            "difference_all_walls": fit,
            "slope_over_mach": fit["slope"] / SOURCE["cavity_mach"],
            "difference_by_wall": {wall: tight["difference_to_thesis"][wall] for wall in WALLS},
            "thesis_by_wall": {wall: {key: thesis["walls"][wall][key] for key in
                                      ("slope", "correlation", "velocity_rms", "velocity_min", "velocity_max",
                                       "pressure_min", "pressure_max")} for wall in WALLS},
            "tight_by_wall": {wall: {key: tight["walls"][wall][key] for key in ("slope", "correlation", "velocity_rms")}
                              for wall in WALLS}}
    axes[1, 0].legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.3,
                      labelspacing=0.3)
    raw.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3)
    small = {"fontsize": style.FONT_SIZE - 1.0, "color": style.MUTED}
    raw.text(0.20, 15.0, "right wall,\nbelow the lid", ha="left", va="center", **small)
    raw.text(-0.02, -7.6, "lid, left end", ha="left", va="center", **small)
    raw.set_xlim(-0.12, 0.38)
    raw.set_ylim(-10.5, 25.0)
    raw.set_xlabel(style.math(style.PRESSURE))
    raw.set_ylabel(VELOCITY_LABEL)
    for axis, letter in zip(axes.ravel(), "abcd"):
        style.panel_letter(axis, letter)
    figure.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.2)
    style.save(figure, STEM)
    return numbers


def short(cell: str) -> str:
    """1.05e-02 -> 1.05e-2, as the other tables of the pack write exponents in their .tex."""
    return cell.replace("e-0", "e-").replace("e+0", "e+")


def run_names(entry: dict, cavity: dict) -> list:
    """The runs of one candidate, by the names of their run directories (cavity.json keys the thesis runs
    n<n>_thesis; their directories are n<n>_laplacian_eps1em6)."""
    names = [entry["bearing"][10], entry["bearing"][1], entry["control"], entry["micropolar"]]
    return names + [cavity.get(entry["cavity"][resolution], {}).get("run", entry["cavity"][resolution])
                    for resolution in (250, 500)]


def lost_particles(record: dict, timestep: float) -> dict:
    """What the runner recorded when it stopped a cavity run: the step, and how many particles had left."""
    match = re.search(r"step (\d+): alive (\d+) != expected (\d+)", record["failure"])
    step, alive, expected = (int(value) for value in match.groups())
    return {"step": step, "time": step * timestep, "lost": expected - alive,
            "last_sample_time": record["last_sample_time"], "max_speed": record["max_speed_max"]}


def candidates_table(benchmark: dict, concentric: dict, cavity: dict) -> dict:
    header = ["candidate", "case key",
              "bearing eps = 0.6, Re = 10: load / exact - 1 (%)", "bearing, Re = 1: load / exact - 1 (%)",
              "bearing, Re = 10: fluid behind the wall surfaces at the end (% of the fluid)",
              "bearing, Re = 10: velocity into the rotor per wall pressure (U per rho U^2)", "correlation (rotor, Re = 10)",
              "bearing, Re = 10: velocity into the stator per wall pressure", "correlation (stator, Re = 10)",
              "bearing, Re = 1: velocity into the rotor per wall pressure", "correlation (rotor, Re = 1)",
              "bearing, Re = 1: velocity into the stator per wall pressure", "correlation (stator, Re = 1)",
              "concentric control g = 100 dx: beta / nominal", "concentric control: mean radial velocity / U",
              "concentric control: fit rms / U", "concentric, K = 3, l = 10 dx: D_SPH / D_exact",
              "cavity n = 250: L2 u", "cavity n = 250: L2 v", "cavity n = 250: mean density drift (kg/m^3 per L / U)",
              "cavity n = 250: thesis run minus this run, slope against the wall pressure", "correlation (n = 250)",
              "cavity n = 500: L2 u", "cavity n = 500: L2 v", "cavity n = 500: mean density drift (kg/m^3 per L / U)",
              "cavity n = 500: thesis run minus this run, slope against the wall pressure", "correlation (n = 500)",
              "cavity: how the runs ended", "runs"]
    tex_header = ["candidate", r"$F/F_{\mathrm{exact}}-1$ (\%), $\mathit{Re} = 10$", r"$\mathit{Re} = 1$",
                  r"behind the walls (\%)", r"$v_n/U$ per $p/(\rho U^2)$, rotor", r"$\bar v_r/U$, concentric",
                  r"$D_{\mathrm{SPH}}/D_{\mathrm{exact}}$", r"$L_2\,u$, $n = 250$", r"$L_2\,v$", r"$L_2\,u$, $n = 500$",
                  r"$L_2\,v$", r"$\mathrm{d}\bar\rho/\mathrm{d}t$, $n = 250$", r"$n = 500$"]
    plain, tex, numbers = [], [], {}
    for entry in RUNS:
        reynolds_ten, reynolds_one = benchmark[entry["bearing"][10]], benchmark[entry["bearing"][1]]
        control, micropolar = concentric[entry["control"]], concentric[entry["micropolar"]]
        cells = [entry["candidate"], entry["key"],
                 f"{100.0 * (reynolds_ten['load_over_exact'] - 1.0):+.2f}", f"{100.0 * (reynolds_one['load_over_exact'] - 1.0):+.2f}",
                 f"{100.0 * reynolds_ten['behind_walls']['share']:.2f}"]
        for run in (reynolds_ten, reynolds_one):
            for wall in ("rotor", "stator"):
                cells += [f"{run['walls'][wall]['slope']:+.5f}", f"{run['walls'][wall]['correlation']:+.2f}"]
        cells += [f"{control['beta_over_nominal']:.6f}", f"{control['radial_velocity_mean_over_wall_speed']:+.1e}",
                  f"{control['fit_rms_over_wall_speed']:.1e}", f"{micropolar['sph_over_exact']:.4f}"]
        record = {"bearing_load_re10": reynolds_ten["load_over_exact"] - 1.0, "bearing_load_re1": reynolds_one["load_over_exact"] - 1.0,
                  "behind_walls_share_re10": reynolds_ten["behind_walls"]["share"],
                  "rotor_slope_re10": reynolds_ten["walls"]["rotor"]["slope"], "rotor_slope_re1": reynolds_one["walls"]["rotor"]["slope"],
                  "beta_over_nominal": control["beta_over_nominal"],
                  "radial_velocity": control["radial_velocity_mean_over_wall_speed"],
                  "d_ratio": micropolar["sph_over_exact"], "cavity": {}}
        endings, cavity_tex = [], {}
        for resolution in (250, 500):
            name = entry["cavity"][resolution]
            thesis = cavity[THESIS["cavity"][resolution]]
            if name in cavity:
                run = cavity[name]
                first, last = run["snapshot_times"]
                drift = (run["density_mean_last_snapshot"] - run["density_mean_first_snapshot"]) / (last - first)
                difference = run.get("difference_to_thesis", {}).get("all_walls")
                drift_text = f"{drift:+.3f}" if abs(drift) >= 5.0e-4 else f"{drift:+.0e}"
                cells += [f"{run['u_l2']:.2e}", f"{run['v_l2']:.2e}", drift_text,
                          "-" if difference is None else f"{difference['slope']:+.4f}",
                          "-" if difference is None else f"{difference['correlation']:+.2f}"]
                endings.append(f"n = {resolution}: steady at t = {run['steady']['steady_time']:.1f}, mean density "
                               f"{run['density_mean_last_snapshot']:.2f} kg/m^3 at t = {last:.1f}")
                record["cavity"][resolution] = {"u_l2": run["u_l2"], "v_l2": run["v_l2"], "density_drift": drift,
                                                "density_mean_last": run["density_mean_last_snapshot"],
                                                "time_of_last_snapshot": last, "difference_to_thesis": difference}
                cavity_tex[resolution] = [short(f"{run['u_l2']:.2e}"), short(f"{run['v_l2']:.2e}"),
                                          short(drift_text if "e" in drift_text else f"{drift:+.2f}")]
            else:
                stopped = lost_particles(cavity["_stopped"][name], thesis["timestep"])
                cells += ["failed", "failed", "-", "-", "-"]
                endings.append(f"n = {resolution}: stopped, {stopped['lost']} particles had left the domain by t = "
                               f"{stopped['time']:.2f} (largest fluid speed {stopped['max_speed']:.2f} U)")
                record["cavity"][resolution] = stopped
                cavity_tex[resolution] = ["failed", "failed", "--"]
        cells += ["; ".join(endings), ", ".join(run_names(entry, cavity))]
        plain.append(cells)
        slope = reynolds_ten["walls"]["rotor"]
        tex.append([style.tex_escape(entry["candidate"]), cells[2], cells[3], cells[4],
                    f"{slope['slope']:+.5f} ({slope['correlation']:+.2f})", short(cells[14]), cells[16]]
                   + cavity_tex[250][:2] + cavity_tex[500][:2] + [cavity_tex[250][2], cavity_tex[500][2]])
        numbers[entry["candidate"]] = record
    style.write_table(TABLE, header, plain, tex_header, "l" + "r" * (len(tex_header) - 1), tex,
                      note="load: force on the rotor against the exact Stokes value; behind the walls: share of the "
                           "fluid behind a wall surface at the end, Re = 10; drift of the mean density in kg/m^3 "
                           "per L / U between the first and the last snapshot")
    return numbers


def main() -> int:
    benchmark = style.load_json(SOURCE["benchmark"])
    concentric = style.load_json(SOURCE["concentric"])
    cavity = style.load_json(SOURCE["cavity"])
    numbers = figure_and_numbers(benchmark, cavity)
    numbers["candidates"] = candidates_table(benchmark, concentric, cavity)

    reynolds_ten, reynolds_one = benchmark[THESIS["bearing"][10]], benchmark[THESIS["bearing"][1]]
    tight = benchmark[TIGHT["bearing"][10]]
    gap = reynolds_ten["outer_radius"] - reynolds_ten["inner_radius"]
    lost = re.search(r"step (\d+): (\d+) particles alive, (\d+) loaded", reynolds_one["failure"])

    def slopes(run):
        return (f"{run['walls']['rotor']['slope']:.4f} (rotor, correlation {run['walls']['rotor']['correlation']:.2f}) "
                f"and {run['walls']['stator']['slope']:.4f} (stator, {run['walls']['stator']['correlation']:.2f})")

    def describes(names):
        return ", ".join(f"`{tag}`" for tag in sorted({benchmark[name]["revision"]["describe"] for name in names}))

    def cavity_text(resolution):
        entry = numbers["cavity"][f"n = {resolution}"]
        fit, by_wall = entry["difference_all_walls"], entry["difference_by_wall"]
        return (f"n = {resolution}: {fit['slope']:.4f} for the four walls together (correlation "
                f"{fit['correlation']:.2f}; " + ", ".join(f"{wall} {by_wall[wall]['slope']:.4f}" for wall in WALLS)
                + f"), that is v_n / U = {entry['slope_over_mach']:.2f} Ma p / (rho U^2)")

    def snapshots(resolution):
        entry = numbers["cavity"][f"n = {resolution}"]
        return (f"n = {resolution}: {entry['thesis_snapshots']} snapshots from t = {entry['thesis_snapshot_times'][0]:.1f} "
                f"to {entry['thesis_snapshot_times'][1]:.1f} (thesis), {entry['tight_snapshots']} from t = "
                f"{entry['tight_snapshot_times'][0]:.1f} to {entry['tight_snapshot_times'][1]:.1f} (tight)")

    def raw_slopes(resolution):
        by_wall = numbers["cavity"][f"n = {resolution}"]["thesis_by_wall"]
        return f"n = {resolution}: " + ", ".join(f"{wall} {by_wall[wall]['slope']:.3f}" for wall in WALLS)

    def right_wall_rms(which):
        return " and ".join(f"{100.0 * numbers['cavity'][f'n = {resolution}'][which]['right']['velocity_rms']:.2f}"
                            for resolution in (250, 500))

    cavity_commits = sorted({numbers["cavity"][key][which] for key in numbers["cavity"]
                             for which in ("thesis_commit", "tight_commit")})
    candidates = numbers["candidates"]
    thesis_row, tight_row = candidates[THESIS["candidate"]], candidates[TIGHT["candidate"]]
    caption = (
        "Seepage of the thesis walls (adami walls that keep rho0, with the density diffusion term acting on "
        "fluid-wall pairs): velocity of the fluid into the wall at the wall surface, over U, against the wall "
        "pressure, over rho U^2; lines: least-squares fits. "
        f"(a) Newtonian flow between eccentric cylinders (eccentricity ratio {reynolds_ten['eccentricity_ratio']:g}, R1 = "
        f"{reynolds_ten['inner_radius'] / gap:g} g, R2 = {reynolds_ten['outer_radius'] / gap:g} g, g = {reynolds_ten['gap_in_spacings']:g} dx, "
        f"Ma = {reynolds_ten['mach']:g}) in the thesis configuration at Re = {reynolds_ten['reynolds']:g} and {reynolds_one['reynolds']:g}, "
        f"around the rotor and the stator: {len(reynolds_ten['walls']['angle'])} angle bins, velocity and pressure "
        f"extrapolated to the surface from the fluid between 0.5 and 12 dx from it, mean of the "
        f"{reynolds_ten['walls']['states']} states of the last quarter of the run ({reynolds_one['walls']['states']} at Re = "
        f"{reynolds_one['reynolds']:g}, where {int(lost.group(3)) - int(lost.group(2))} particles had left the domain by step "
        f"{int(lost.group(1)):,} of {reynolds_one['steps']:,}). Slopes: {slopes(reynolds_ten)} at Re = "
        f"{reynolds_ten['reynolds']:g}, that is v_n / U = {reynolds_ten['walls']['rotor']['slope_over_mach']:.2f} and "
        f"{reynolds_ten['walls']['stator']['slope_over_mach']:.2f} Ma p / (rho U^2); "
        f"{slopes(reynolds_one)} at Re = {reynolds_one['reynolds']:g}. For contrast the candidate (c), density diffusion between fluid "
        f"pairs only, at Re = {tight['reynolds']:g} (rotor and stator together): slope "
        f"{tight['walls']['rotor']['slope']:.5f} and {tight['walls']['stator']['slope']:.5f}. "
        f"(b) Lid-driven cavity, Re = 1000, Ma = {SOURCE['cavity_mach']:g}, thesis configuration, n = 250 and 500: "
        "the same quantities at the four walls (50 bins along each wall without the corner squares of side 12 dx; "
        "pressure from the particle volumes). The steep branch is the right wall below the lid, where the jet from "
        "the lid turns down and the extrapolation to the surface is poor: the rms velocity into the right wall is "
        f"{right_wall_rms('thesis_by_wall')} % of U at n = 250 and 500, and {right_wall_rms('tight_by_wall')} % with "
        "the tight walls (c), so it is not seepage; slopes wall by wall: " + raw_slopes(250) + "; " + raw_slopes(500)
        + ". "
        "(c), (d) The velocity into the wall of the thesis run minus that of the run with tight walls (c), against "
        "the wall pressure of the thesis run, for n = 250 and n = 500. Slopes: " + cavity_text(250) + "; "
        + cavity_text(500) + ". Snapshots averaged: " + snapshots(250) + "; " + snapshots(500) + ". "
        f"Runs: `{THESIS['bearing'][10]}`, `{THESIS['bearing'][1]}`, `{TIGHT['bearing'][10]}` "
        f"({describes([THESIS['bearing'][10], THESIS['bearing'][1], TIGHT['bearing'][10]])}); cavity "
        + ", ".join(f"`{cavity[entry['cavity'][resolution]]['run']}`" for entry in (THESIS, TIGHT)
                    for resolution in (250, 500))
        + " (commits " + ", ".join(f"{commit} = `{SOURCE['describe'][commit]}`" for commit in cavity_commits)
        + f"). {SOURCE['stage']}; existing data, not rerun on thesis-ch5-rc2. "
        f"Table {TABLE}: the four candidates on the three tests (bearing at eccentricity ratio 0.6 against the exact "
        "Stokes load, concentric control at g = 100 dx with the micropolar run at K = 3, l = 10 dx, cavity at n = 250 "
        "and 500 against Marchi et al.), every number from the three data files. Thesis walls: load "
        f"{100.0 * thesis_row['bearing_load_re10']:+.1f} % (Re = 10) and {100.0 * thesis_row['bearing_load_re1']:+.1f} % "
        f"(Re = 1), {100.0 * thesis_row['behind_walls_share_re10']:.1f} % of the fluid behind the wall surfaces; the "
        f"three tight candidates {100.0 * min(candidates[entry['candidate']]['bearing_load_re10'] for entry in RUNS[1:]):+.2f} "
        f"to {100.0 * max(candidates[entry['candidate']]['bearing_load_re10'] for entry in RUNS[1:]):+.2f} % and "
        f"{100.0 * min(candidates[entry['candidate']]['bearing_load_re1'] for entry in RUNS[1:]):+.2f} to "
        f"{100.0 * max(candidates[entry['candidate']]['bearing_load_re1'] for entry in RUNS[1:]):+.2f} %. In the cavity "
        f"(b) loses particles (by t = {candidates[RUNS[1]['candidate']]['cavity'][250]['time']:.1f} and "
        f"{candidates[RUNS[1]['candidate']]['cavity'][500]['time']:.1f}), and the mean density of (c) and (d') runs "
        f"away ((c): {tight_row['cavity'][250]['density_drift']:+.2f} and {tight_row['cavity'][500]['density_drift']:+.2f} "
        f"kg/m^3 per L / U at n = 250 and 500; thesis {thesis_row['cavity'][250]['density_drift']:+.0e} and "
        f"{thesis_row['cavity'][500]['density_drift']:+.0e}).")
    sources = [SOURCE["benchmark"], SOURCE["concentric"], SOURCE["cavity"]]
    for entry in RUNS:
        sources += run_names(entry, cavity)
    style.record("13", "Seepage of the thesis walls: velocity into the wall against wall pressure; the wall candidates",
                 SOURCE["status"], caption, [f"{STEM}.pdf", f"{STEM}.png", f"{TABLE}.csv", f"{TABLE}.tex"], sources,
                 numbers, "Stage 5, " + describes([name for entry in RUNS for name in entry["bearing"].values()])
                 + "; cavity " + ", ".join(f"`{SOURCE['describe'][commit]}`" for commit in cavity_commits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
