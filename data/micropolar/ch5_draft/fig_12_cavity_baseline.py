"""
Item 12 - the Newtonian lid-driven cavity at n = 1000, Re = 1000, in the thesis configuration against Marchi et
al. (2021): the centre-line profiles u(y) on x = L / 2 and v(x) on y = L / 2 at the 28 + 28 reference points, and
the errors (run minus reference) below them. Existing data of Stage 3 (design note 12.8); not rerun.

Run from the repository root:  python data/micropolar/ch5_draft/fig_12_cavity_baseline.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import style
from experiment.validation import cavity_reference

# the summary of viscous_form_campaign.py analyse (errors, means at the reference points) and the run's own record
SOURCE = {
    "analysis": "data/micropolar/stage3/cavity_n1000.json", "row": "laplacian_eps1em6",
    "run": "data/micropolar/stage3/cavity_runs/n1000_laplacian_eps1em6",
    "status": "final (existing data)", "stage": "Stage 3 (design note 12.8)",
    # git describe of the commit that the run's meta.json records (checked below against meta.json)
    "commit": "7e76dcb", "describe": "micropolar-base-13-g7e76dcb", "data_tag": "thesis-ch5-rc1",
    "reynolds": 1000, "wall_distance_of_interior_points": 0.05,
}
STEM = "fig_12_cavity_baseline"
ERROR_SCALE = 1.0e3


def main() -> int:
    style.setup()
    row = style.load_json(SOURCE["analysis"])[SOURCE["row"]]
    meta = style.load_json(SOURCE["run"] + "/meta.json")
    samples = [json.loads(line) for line in
               (style.REPO_ROOT / SOURCE["run"] / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if meta["git"]["head"] != SOURCE["commit"]:
        raise RuntimeError(f"meta.json records commit {meta['git']['head']}, SOURCE says {SOURCE['commit']}")
    reference = cavity_reference.marchi2021(SOURCE["reynolds"])
    resolution = round(1.0 / meta["spacing"])
    # the solver's step is dt = cfl h / c0, so the sound speed follows from the recorded step
    case_text = (style.REPO_ROOT / meta["case"].replace("\\", "/")).read_text(encoding="utf-8")
    sound_speed = float(next(line.split(":")[1].split("#")[0] for line in case_text.splitlines()
                             if line.strip().startswith("speed_of_sound:")))
    mach = 1.0 / sound_speed                                         # lid speed U = 1

    lines = {}
    for name in ("u", "v"):
        coordinate, expected = np.asarray(reference[name][0]), np.asarray(reference[name][1])
        mean = np.asarray(row[f"{name}_mean"])
        error = mean - expected
        interior = ((coordinate > SOURCE["wall_distance_of_interior_points"])
                    & (coordinate < 1.0 - SOURCE["wall_distance_of_interior_points"]))
        nearest = np.argsort(coordinate)[-4:]
        lines[name] = {"coordinate": coordinate, "expected": expected, "mean": mean, "error": error,
                       "l2": float(np.sqrt((error ** 2).mean())), "linf": float(np.abs(error).max()),
                       "interior_rms": float(np.sqrt((error[interior] ** 2).mean())),
                       "interior_points": int(interior.sum()),
                       "others_max": float(np.abs(np.delete(error, nearest)).max()),
                       "nearest_coordinates": coordinate[nearest], "nearest_errors": error[nearest]}
        # the script's errors are those of the analysis file
        if abs(lines[name]["l2"] - row[f"{name}_l2"]) > 1e-12 or abs(lines[name]["linf"] - row[f"{name}_linf"]) > 1e-12:
            raise RuntimeError(f"{name}: errors recomputed from the means differ from the analysis file")

    def extremum(name, kind):
        entry = lines[name]
        index = int(np.argmin(entry["expected"]) if kind == "min" else np.argmax(entry["expected"]))
        return {"coordinate": float(entry["coordinate"][index]), "reference": float(entry["expected"][index]),
                "run": float(entry["mean"][index]), "difference": float(entry["error"][index]),
                "relative": float(entry["error"][index] / entry["expected"][index])}
    extrema = {"min u on x = L / 2": extremum("u", "min"), "max v on y = L / 2": extremum("v", "max"),
               "min v on y = L / 2": extremum("v", "min")}

    figure, axes = plt.subplots(2, 2, figsize=(style.WIDE, style.WIDE * 0.60), sharex="col",
                                gridspec_kw={"height_ratios": [1.7, 1.0]})
    run_style = style.run_markers(0, filled=False)                   # open: a Newtonian run
    reference_line = {**style.EXACT, "linewidth": 0.6, "marker": ".", "markersize": 3.0}
    labels = {"u": (r"$y\,/\,L$", style.math(style.CAVITY_U), r"$(u - u_{\mathrm{ref}})\,/\,U \times 10^{3}$"),
              "v": (r"$x\,/\,L$", style.math(style.CAVITY_V), r"$(v - v_{\mathrm{ref}})\,/\,U \times 10^{3}$")}
    for column, name in enumerate(("u", "v")):
        entry = lines[name]
        upper, lower = axes[0, column], axes[1, column]
        upper.plot(entry["coordinate"], entry["expected"], zorder=3, label="Marchi et al. (2021)", **reference_line)
        upper.plot(entry["coordinate"], entry["mean"], zorder=2, label=rf"run, $n = {resolution}$", **run_style)
        lower.axhline(0.0, **{**style.EXACT, "linewidth": 0.6})
        lower.plot(entry["coordinate"], ERROR_SCALE * entry["error"], **run_style)
        upper.margins(y=0.07)
        upper.set_ylabel(labels[name][1])
        lower.set_ylabel(labels[name][2])
        lower.set_xlabel(labels[name][0])
        lower.set_xlim(0.0, 1.0)
        lower.set_ylim(-2.0, 6.0)
        lower.set_yticks([-2, 0, 2, 4, 6])
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.04, 0.98))
    for axis, letter in zip(axes.ravel(), "abcd"):
        style.panel_letter(axis, letter)
    figure.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.6)
    style.save(figure, STEM)

    def lid_side(name):
        entry = lines[name]
        return ", ".join(f"{error:+.1e}" for error in entry["nearest_errors"])
    line_u, line_v = lines["u"], lines["v"]
    density_spread = row["density_range_mean"] / 1000.0
    caption = (
        f"Newtonian lid-driven cavity in the thesis configuration (viscous form laplacian, EPS factor 1e-6, adami "
        f"walls) at Re = {SOURCE['reynolds']}, n = {resolution} ({meta['expected_total']:,} particles with the walls, "
        f"h = {meta['support_radius'] / meta['spacing']:g} dx, Ma = {mach:g}, dt = {meta['dt']:g} L / U, K = 1), from "
        f"rest, against the reference of Marchi et al. (2021). (a) u on x = L / 2 and (b) v on y = L / 2 at the "
        f"28 + 28 reference points: reference (black points joined by a thin line) and the run (circles; MLS "
        f"samples in the wall frame, mean of {row['samples']} samples over the last {meta['average_span']:g} time "
        f"units, to t = {row['end_time']:.1f} L / U; the run passed its steady test at t = "
        f"{row['steady']['steady_time']:.1f}). (c), (d) Run minus reference at the same points, in units of 1e-3 U. "
        f"L2 error of u {line_u['l2']:.2e} U (largest {line_u['linf']:.2e}), of v {line_v['l2']:.2e} U (largest {line_v['linf']:.2e}). "
        + "Extreme values at the reference points: "
        + "; ".join(f"{label} ({'y' if label.endswith('x = L / 2') else 'x'} = {item['coordinate']:.3f} L) "
                    f"{item['run']:.5f} against {item['reference']:.6f} ({item['difference']:+.1e} U, "
                    f"{100.0 * abs(item['relative']):.2f} % in magnitude)" for label, item in extrema.items())
        + f". The remaining error sits next to the lid and to the right wall: u at the four points nearest the lid "
        f"(y = {line_u['nearest_coordinates'][0]:.3f} to {line_u['nearest_coordinates'][-1]:.3f} L) {lid_side('u')} U, every "
        f"other u point within {line_u['others_max']:.1e} U; v at the four points nearest the right wall (x = "
        f"{line_v['nearest_coordinates'][0]:.3f} to {line_v['nearest_coordinates'][-1]:.3f} L) {lid_side('v')} U. Without the "
        f"points within {SOURCE['wall_distance_of_interior_points']:g} L of a wall the rms error is "
        f"{line_u['interior_rms']:.2e} U (u, {line_u['interior_points']} points) and {line_v['interior_rms']:.2e} U (v, "
        f"{line_v['interior_points']} points). Fluid density range {100.0 * density_spread:.2f} % of rho0 (mean over the "
        f"window). Run `{row['run']}`, {SOURCE['stage']}, commit {SOURCE['commit']} (`{SOURCE['describe']}`; the "
        f"data file is in the tree since `{SOURCE['data_tag']}`); existing data, not rerun on thesis-ch5-rc2. The "
        f"walls of this configuration are slightly porous (item 13).")
    numbers = {
        "run": row["run"], "resolution": resolution, "particles": meta["expected_total"], "mach": mach,
        "timestep": meta["dt"], "end_time": row["end_time"], "samples": row["samples"],
        "steady_time": row["steady"]["steady_time"], "last_sample_step": samples[-1]["step"],
        "u_l2": line_u["l2"], "u_linf": line_u["linf"], "v_l2": line_v["l2"], "v_linf": line_v["linf"],
        "u_interior_rms": line_u["interior_rms"], "v_interior_rms": line_v["interior_rms"],
        "u_interior_points": line_u["interior_points"], "v_interior_points": line_v["interior_points"],
        "u_errors_nearest_the_lid": dict(zip((f"{value:.6f}" for value in line_u["nearest_coordinates"]),
                                             line_u["nearest_errors"].tolist())),
        "v_errors_nearest_the_right_wall": dict(zip((f"{value:.6f}" for value in line_v["nearest_coordinates"]),
                                                    line_v["nearest_errors"].tolist())),
        "u_largest_error_of_the_other_points": line_u["others_max"], "extrema": extrema,
        "kinetic_energy_mean": row["kinetic_energy_mean"], "kinetic_energy_std": row["kinetic_energy_std"],
        "density_range_mean": row["density_range_mean"], "density_range_max": row["density_range_max"],
        "steps_per_second": row["steps_per_second"], "commit": SOURCE["commit"], "describe": SOURCE["describe"],
    }
    style.record("12", "Newtonian cavity baseline at n = 1000 against Marchi et al.", SOURCE["status"], caption,
                 [f"{STEM}.pdf", f"{STEM}.png"],
                 [SOURCE["analysis"], SOURCE["run"], "docs/validation/data/marchi2021_re1000.csv"], numbers,
                 f"{SOURCE['stage']}, `{SOURCE['describe']}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
