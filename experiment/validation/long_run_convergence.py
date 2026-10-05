"""long_run_convergence.py - time convergence of a cavity run continued past its stop time (a *_long run).

The 1000^2 main-series run (xi 0.001, epsilon_squared_factor 0.0025, K = 2, float32) stopped at t = 100 by the
steady rule (consecutive 10-unit windows changed by < 1e-3). n1000_k2_float32_xi0p001_eps0p0025_long is a copy of
its directory resumed from the t = 99.2 checkpoint with the same settings and run on to t = 300, so its sample
history is the original one up to t = 99.2 followed by the continuation. This script reports

  per window (default 10 time units, the runner's steady test):
      the relative change of the window-mean 56-point profile (wall frame, MLS) and of the window-mean kinetic
      energy against the previous window; the distance of the window-mean profile to the last window's;
      L2 / relative L2 of the profile against Marchi 2021 Tc; u_min, v_max, v_min from the window-mean dense lines;
  per block (default 20 time units, the report's averaging span; (80, 100] is the report's average):
      the same errors and extrema with standard errors (integrated autocorrelation time), and with --stream the
      stream-function minimum and vortex centre of the blocks that have particle snapshots.

Outputs: docs/validation/data/long_run_windows.csv, long_run_blocks.csv, tables_long_run.md (the report's tables),
figures/cavity_re1000_long_run.pdf.

    .venv/Scripts/python.exe -m experiment.validation.long_run_convergence [--run ...] [--stream]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.cavity_analysis import (DATA, EXTREMA_SEARCH, FIGURES, load_run, setup_style,  # noqa: E402
                                                   stream_function_minimum, window_statistics, write_csv)

REFERENCE_KEYS = {"u_min": ("u_min", "y_at_u_min"), "v_max": ("v_max", "x_at_v_max"), "v_min": ("v_min", "x_at_v_min")}


def summarise(run: dict, mask: np.ndarray, reference: dict, with_errors: bool = True) -> dict:
    """Errors against Marchi 2021 Tc and extrema of the mean over the samples in ``mask``."""
    arrays = run["arrays"]
    profile = window_statistics(np.concatenate([arrays["u_marchi_wall_mls"][mask], arrays["v_marchi_wall_mls"][mask]],
                                               axis=1))
    energy = window_statistics(run["kinetic_energy"][mask, None])
    result = {"samples": int(mask.sum()), "kinetic_energy": float(energy["mean"][0]),
              "kinetic_energy_sem": float(energy["sem"][0]), "profile": profile["mean"], "profile_sem": profile["sem"]}
    for line in ("u", "v"):
        error = arrays[f"{line}_marchi_wall_mls"][mask].mean(axis=0) - reference[line][1]
        result[f"{line}_L2"] = float(np.sqrt(np.mean(error ** 2)))
        result[f"{line}_L2_relative"] = result[f"{line}_L2"] / float(np.sqrt(np.mean(reference[line][1] ** 2)))
        result[f"{line}_Linf"] = float(np.abs(error).max())
    dense = run["points"]["dense_reference"]
    for name, (line, kind, search) in EXTREMA_SEARCH.items():
        if with_errors:
            statistics = window_statistics(arrays[f"{line}_dense_mls"][mask])
            mean = statistics["mean"]
        else:
            mean = arrays[f"{line}_dense_mls"][mask].mean(axis=0)
        position, value = sampling.refine_extremum(dense, mean, kind, search=search)
        result[name] = value
        result[f"{name}_position"] = position
        if with_errors:
            result[f"{name}_sem"] = float(statistics["sem"][int(np.argmin(np.abs(dense - position)))])
    return result


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def write_tables(reference: dict, approach_from: float = 40.0, split: float = 100.0) -> None:
    """tables_long_run.md (read by the report assembler): LONG_RUN = the 20-unit averages with Marchi 2021 Tc and
    the deviation of the last average; LONG_RUN_WINDOWS = the window test (change, noise floor) per window up to
    ``split`` and summarised per 100 units after it, plus the linear trend of the window-mean kinetic energy."""
    blocks = read_csv_rows(DATA / "long_run_blocks.csv")
    windows = read_csv_rows(DATA / "long_run_windows.csv")
    extrema_reference = reference["extrema"]
    rows = []
    for block in blocks:
        psi = (f"{float(block['psi_min_wall']):.5f} @ ({float(block['x_psi_min_wall']):.4f}, "
               f"{float(block['y_psi_min_wall']):.4f})" if block["psi_min_wall"] else "")
        rows.append([f"({float(block['block_start']):g}, {float(block['block_end']):g}]", block["samples"],
                     f"{100 * float(block['u_L2_relative']):.2f} %", f"{100 * float(block['v_L2_relative']):.2f} %"]
                    + [f"{float(block[name]):.5f} ± {float(block[name + '_sem']):.0e}" for name in EXTREMA_SEARCH]
                    + [psi])
    rows.append(["Marchi 2021 T_c(Table 19)", "", "", ""]
                + [f"{extrema_reference[value_key][0]:.5f}" for value_key, _ in REFERENCE_KEYS.values()]
                + [f"{extrema_reference['psi_min'][0]:.5f} @ ({extrema_reference['x_at_psi_min'][0]:.4f}, "
                   f"{extrema_reference['y_at_psi_min'][0]:.4f})"])
    last = blocks[-1]
    deviation = [f"{100 * (abs(float(last[name])) / abs(extrema_reference[value_key][0]) - 1):+.2f} %"
                 for name, (value_key, _) in REFERENCE_KEYS.items()]
    psi_deviation = (f"{100 * (abs(float(last['psi_min_wall'])) / abs(extrema_reference['psi_min'][0]) - 1):+.2f} %"
                     if last["psi_min_wall"] else "")
    rows.append([f"最后一个平均的幅值相对 T_c", "", "", ""] + deviation + [psi_deviation])
    long_run = md_table(["平均窗口 (t₀, t₁]", "样本", "u 相对 L2", "v 相对 L2", "u_min ± 标准误差", "v_max ± 标准误差",
                         "v_min ± 标准误差", "ψ_min @ (x, y)"], rows)

    window_rows = []
    for window in windows:
        start, end = float(window["window_start"]), float(window["window_end"])
        if approach_from < end <= split:
            window_rows.append([f"({start:g}, {end:g}]", f"{float(window['profile_change']):.1e}",
                                f"{float(window['profile_change_noise']):.1e}",
                                f"{float(window['kinetic_energy_change']):.1e}",
                                f"{float(window['kinetic_energy_change_noise']):.1e}"])
    late = [window for window in windows if float(window["window_start"]) >= split]
    for first in range(0, len(late), 10):
        group = late[first:first + 10]
        if not group:
            continue

        def span(key: str) -> str:
            values = np.array([float(window[key]) for window in group])
            return f"{values.min():.1e} – {values.max():.1e}(中位数 {np.median(values):.1e})"
        window_rows.append([f"({float(group[0]['window_start']):g}, {float(group[-1]['window_end']):g}],"
                            f"{len(group)} 个窗口", span("profile_change"), span("profile_change_noise"),
                            span("kinetic_energy_change"), span("kinetic_energy_change_noise")])
    windows_table = md_table(["窗口", "剖面变化", "剖面噪声底", "动能变化", "动能噪声底"], window_rows)

    centres = np.array([0.5 * (float(window["window_start"]) + float(window["window_end"])) for window in late])
    energy = np.array([float(window["kinetic_energy"]) for window in late])
    design = np.stack([np.ones_like(centres), centres - centres.mean()], axis=1)
    coefficients, residual, _, _ = np.linalg.lstsq(design, energy, rcond=None)
    sigma = np.sqrt(residual[0] / (energy.size - 2)) if residual.size else 0.0
    slope_error = sigma / np.sqrt(np.sum((centres - centres.mean()) ** 2))
    trend = (f"t ≥ {split:g} 的 {energy.size} 个窗口平均动能:线性趋势(相对值)每 100 个时间单位 "
             f"{100 * coefficients[1] / energy.mean():+.1e} ± {100 * slope_error / energy.mean():.1e}(按窗口独立估计标准误差),"
             f"范围 {energy.min():.5f} – {energy.max():.5f}(相对极差 {(energy.max() - energy.min()) / energy.mean():.1e})。")
    (DATA / "tables_long_run.md").write_text(
        f"<!-- LONG_RUN -->\n{long_run}\n\n<!-- LONG_RUN_WINDOWS -->\n{windows_table}\n\n<!-- LONG_RUN_TREND -->\n{trend}\n",
        encoding="utf-8")
    print(f"wrote {DATA / 'tables_long_run.md'}; {trend}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="n1000_k2_float32_xi0p001_eps0p0025_long")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--block", type=float, default=20.0)
    parser.add_argument("--first-block-end", type=float, default=100.0)
    parser.add_argument("--zoom-start", type=float, default=40.0, help="first window end shown in (c) and (d)")
    parser.add_argument("--stream", action="store_true", help="psi_min / vortex centre of blocks with snapshots (slow)")
    parser.add_argument("--no-output", action="store_true", help="print only (no CSV / figure)")
    parser.add_argument("--tables-only", action="store_true", help="only tables_long_run.md, from the two CSVs")
    arguments = parser.parse_args()
    if arguments.tables_only:
        write_tables(cavity_reference.marchi2021())
        return 0

    run = load_run(arguments.run)
    if run is None:
        print(f"no samples for {arguments.run}")
        return 1
    reference = cavity_reference.marchi2021()
    times = run["times"]
    t_last = float(times.max())
    # samples sit on defrag boundaries (~0.2 time units apart), so the last one falls up to one interval short of
    # the stop time: a window / block counts as complete if it ends within 1.5 intervals after the last sample
    slack = 1.5 * float(np.median(np.diff(times)))
    print(f"{arguments.run}: {times.size} samples, t = {times.min():.2f} .. {t_last:.2f}, variant {run['variant']}")

    # ---- consecutive windows
    window_rows, windows = [], []
    count = int(np.floor((t_last + slack) / arguments.window + 1e-9))
    for index in range(count):
        start, end = index * arguments.window, (index + 1) * arguments.window
        mask = (times > start + 1e-9) & (times <= end + 1e-9)
        if mask.sum() < 3:
            continue
        windows.append((start, end, summarise(run, mask, reference, with_errors=False)))
    last_profile = windows[-1][2]["profile"]
    for position, (start, end, entry) in enumerate(windows):
        if position > 0:
            previous = windows[position - 1][2]
            entry["profile_change"] = float(np.linalg.norm(entry["profile"] - previous["profile"])
                                            / np.linalg.norm(entry["profile"]))
            entry["kinetic_energy_change"] = abs(entry["kinetic_energy"] - previous["kinetic_energy"]) / entry["kinetic_energy"]
            # what two statistically independent windows would differ by (standard errors with the integrated
            # autocorrelation time): the floor the window test can reach
            entry["profile_change_noise"] = float(np.sqrt(np.sum(entry["profile_sem"] ** 2 + previous["profile_sem"] ** 2))
                                                  / np.linalg.norm(entry["profile"]))
            entry["kinetic_energy_change_noise"] = float(np.hypot(entry["kinetic_energy_sem"], previous["kinetic_energy_sem"])
                                                         / entry["kinetic_energy"])
        else:
            entry["profile_change"] = entry["kinetic_energy_change"] = float("nan")
            entry["profile_change_noise"] = entry["kinetic_energy_change_noise"] = float("nan")
        entry["distance_to_last"] = float(np.linalg.norm(entry["profile"] - last_profile) / np.linalg.norm(last_profile))
        window_rows.append([f"{start:g}", f"{end:g}", entry["samples"], f"{entry['kinetic_energy']:.6f}",
                            f"{entry['profile_change']:.3e}", f"{entry['profile_change_noise']:.3e}",
                            f"{entry['kinetic_energy_change']:.3e}", f"{entry['kinetic_energy_change_noise']:.3e}",
                            f"{entry['distance_to_last']:.3e}",
                            f"{entry['u_L2']:.6f}", f"{entry['u_L2_relative']:.5f}",
                            f"{entry['v_L2']:.6f}", f"{entry['v_L2_relative']:.5f}"]
                           + [item for name in EXTREMA_SEARCH for item in
                              (f"{entry[name]:.6f}", f"{entry[name + '_position']:.5f}")])

    # ---- blocks of the report's averaging span
    block_rows, blocks = [], []
    end = arguments.first_block_end
    while end <= t_last + slack:
        start = end - arguments.block
        mask = (times > start - 1e-9) & (times <= end + 1e-9)
        entry = summarise(run, mask, reference)
        if arguments.stream:
            entry["stream"] = stream_function_minimum(run, (start, end))
        blocks.append((start, end, entry))
        stream = entry.get("stream") or {}
        block_rows.append([f"{start:g}", f"{end:g}", entry["samples"], f"{entry['kinetic_energy']:.6f}",
                           f"{entry['u_L2']:.6f}", f"{entry['u_L2_relative']:.5f}", f"{entry['u_Linf']:.6f}",
                           f"{entry['v_L2']:.6f}", f"{entry['v_L2_relative']:.5f}", f"{entry['v_Linf']:.6f}"]
                          + [item for name in EXTREMA_SEARCH for item in
                             (f"{entry[name]:.6f}", f"{entry[name + '_sem']:.2e}", f"{entry[name + '_position']:.5f}")]
                          + [f"{stream['psi_min_wall']:.6f}" if stream else "",
                             f"{stream['x_wall']:.5f}" if stream else "", f"{stream['y_wall']:.5f}" if stream else "",
                             stream.get("snapshots", "")])
        end += arguments.block

    # ---- print
    print(f"\n{'window':>11s} {'n':>3s} {'d profile':>9s} {'(noise)':>9s} {'d KE':>9s} {'(noise)':>9s} {'to last':>9s}"
          f" {'u relL2':>8s} {'v relL2':>8s}"
          f" {'u_min':>8s} {'v_max':>8s} {'v_min':>8s}")
    for start, end, entry in windows:
        print(f"{start:5.0f}-{end:<5.0f} {entry['samples']:3d} {entry['profile_change']:9.2e} "
              f"{entry['profile_change_noise']:9.2e} {entry['kinetic_energy_change']:9.2e} "
              f"{entry['kinetic_energy_change_noise']:9.2e} {entry['distance_to_last']:9.2e} {entry['u_L2_relative']:8.4f} "
              f"{entry['v_L2_relative']:8.4f} {entry['u_min']:8.5f} {entry['v_max']:8.5f} {entry['v_min']:8.5f}")
    print(f"\n{'block':>11s} {'u relL2':>8s} {'v relL2':>8s}  extrema (value ± sem @ position), Marchi Tc in the last row")
    for start, end, entry in blocks:
        extrema = "  ".join(f"{name} {entry[name]:.5f}±{entry[name + '_sem']:.0e}@{entry[name + '_position']:.4f}"
                            for name in EXTREMA_SEARCH)
        stream = entry.get("stream")
        tail = (f"  psi_min {stream['psi_min_wall']:.5f} @ ({stream['x_wall']:.4f}, {stream['y_wall']:.4f})"
                if stream else "")
        print(f"{start:5.0f}-{end:<5.0f} {entry['u_L2_relative']:8.4f} {entry['v_L2_relative']:8.4f}  {extrema}{tail}")
    extrema_reference = reference["extrema"]
    print("Marchi Tc" + " " * 20 + "  ".join(
        f"{name} {extrema_reference[value_key][0]:.5f}@{extrema_reference[position_key][0]:.4f}"
        for name, (value_key, position_key) in REFERENCE_KEYS.items()))
    if arguments.no_output:
        return 0

    # ---- CSV
    extrema_header = [item for name in EXTREMA_SEARCH for item in (name, f"{name}_position_wall")]
    comment = (f"{arguments.run}: the 1000^2 main-series run (xi 0.001, epsilon_squared_factor 0.0025, K = 2) continued\n"
               f"from its t = 99.2 checkpoint; samples up to t = {t_last:.2f}. Wall frame, MLS; Marchi 2021 Tc reference.")
    write_csv(DATA / "long_run_windows.csv",
              ["window_start", "window_end", "samples", "kinetic_energy", "profile_change", "profile_change_noise",
               "kinetic_energy_change", "kinetic_energy_change_noise", "profile_distance_to_last_window", "u_L2", "u_L2_relative", "v_L2", "v_L2_relative"] + extrema_header,
              window_rows, comment + f"\nConsecutive {arguments.window:g}-unit windows; changes relative to the previous window;"
              "\n*_noise = the change two statistically independent windows would show (standard errors with the"
              "\nintegrated autocorrelation time).")
    write_csv(DATA / "long_run_blocks.csv",
              ["block_start", "block_end", "samples", "kinetic_energy", "u_L2", "u_L2_relative", "u_Linf", "v_L2",
               "v_L2_relative", "v_Linf"]
              + [item for name in EXTREMA_SEARCH for item in (name, f"{name}_sem", f"{name}_position_wall")]
              + ["psi_min_wall", "x_psi_min_wall", "y_psi_min_wall", "snapshots"],
              block_rows, comment + f"\n{arguments.block:g}-unit averages (the report's span; (80, 100] = the report's numbers).")

    # ---- figure
    import matplotlib.pyplot as plt
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    window_ends = np.array([end for _, end, _ in windows])
    changes = np.array([entry["profile_change"] for _, _, entry in windows])
    energy_changes = np.array([entry["kinetic_energy_change"] for _, _, entry in windows])
    distances = np.array([entry["distance_to_last"] for _, _, entry in windows])
    noise = np.array([entry["profile_change_noise"] for _, _, entry in windows])
    energy_noise = np.array([entry["kinetic_energy_change_noise"] for _, _, entry in windows])
    axis = axes[0, 0]
    axis.semilogy(window_ends, changes, "-o", ms=2.6, color="#08519c", lw=0.8, label="profile, window to window")
    axis.semilogy(window_ends, energy_changes, "-o", ms=2.6, mfc="none", color="#08519c", lw=0.5,
                  label="kinetic energy, window to window")
    axis.semilogy(window_ends[:-1], distances[:-1], "--", color="#e6550d", lw=0.9, label="profile, distance to last window")
    axis.semilogy(window_ends, noise, "-", color="#969696", lw=1.6, alpha=0.8, label="profile, statistical floor")
    axis.semilogy(window_ends, energy_noise, ":", color="#969696", lw=1.2, label="kinetic energy, statistical floor")
    axis.axhline(1.0e-3, color="#777777", lw=0.6, ls=":")
    axis.set_xlabel(r"window end $t$")
    axis.set_ylabel("relative change")
    axis.set_ylim(None, 5.0)                                        # room for the panel tag
    axis.legend(frameon=False, fontsize=6.5, loc="upper right")
    axis = axes[0, 1]
    times_all = run["times"]
    axis.plot(times_all, run["kinetic_energy"], color="#08519c", lw=0.6)
    axis.axvline(99.225, color="#777777", lw=0.6, ls=":")
    axis.text(101.0, 0.05, "continuation", transform=axis.get_xaxis_transform(), fontsize=7, color="#555555")
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\frac{1}{2}\sum_{\mathrm{fluid}} m\,|\mathbf{v}|^2$")
    late = times_all > 40
    axis.set_ylim(run["kinetic_energy"][late].min() * 0.995, run["kinetic_energy"][late].max() * 1.005)
    # (c), (d): from --zoom-start on (the start-up transient is in (a)); dotted = the report's (80, 100] average
    zoom = window_ends >= arguments.zoom_start
    report_block = next((entry for start, end, entry in blocks if abs(end - arguments.first_block_end) < 1e-6), None)
    axis = axes[1, 0]
    for line, marker, color in (("u", "o", "#08519c"), ("v", "s", "#6baed6")):
        axis.plot(window_ends[zoom], [100 * entry[f"{line}_L2_relative"] for (_, _, entry), keep in zip(windows, zoom)
                                      if keep], "-" + marker, ms=2.6, lw=0.8, color=color, label=f"${line}$, 10-unit window")
        axis.plot([end for _, end, _ in blocks], [100 * entry[f"{line}_L2_relative"] for _, _, entry in blocks], "D",
                  ms=4.0, mfc="none", color=color, label=f"${line}$, {arguments.block:g}-unit average")
        if report_block is not None:
            axis.axhline(100 * report_block[f"{line}_L2_relative"], color=color, lw=0.6, ls=":")
    axis.set_xlabel(r"window end $t$")
    axis.set_ylabel(r"relative $L_2$ vs Marchi 2021 (%)")
    lower, upper = axis.get_ylim()
    axis.set_ylim(lower, upper + 0.12 * (upper - lower))            # room for the panel tag
    axis.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    axis = axes[1, 1]
    labels = {"u_min": r"$|u_{\min}|$", "v_max": r"$v_{\max}$", "v_min": r"$|v_{\min}|$"}
    for (name, (value_key, _)), color in zip(REFERENCE_KEYS.items(), ("#08519c", "#e6550d", "#31a354")):
        reference_value = abs(extrema_reference[value_key][0])
        axis.plot(window_ends[zoom], [100 * (abs(entry[name]) / reference_value - 1.0)
                                      for (_, _, entry), keep in zip(windows, zoom) if keep],
                  "-o", ms=2.6, lw=0.8, color=color, label=labels[name])
        if report_block is not None:
            axis.axhline(100 * (abs(report_block[name]) / reference_value - 1.0), color=color, lw=0.6, ls=":")
    axis.set_xlabel(r"window end $t$")
    axis.set_ylabel("extremum magnitude vs Marchi 2021 (%)")
    axis.legend(frameon=False, fontsize=6.5, loc="lower right")
    for axis, tag in zip(axes.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        axis.set_xlim(0 if tag in ("(a)", "(b)") else arguments.zoom_start - 2, None)
        axis.text(0.02, 0.98, tag, transform=axis.transAxes, va="top", ha="left")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "cavity_re1000_long_run.pdf")
    plt.close(figure)
    print(f"wrote {DATA / 'long_run_windows.csv'}, {DATA / 'long_run_blocks.csv'}, {FIGURES / 'cavity_re1000_long_run.pdf'}")
    write_tables(reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
