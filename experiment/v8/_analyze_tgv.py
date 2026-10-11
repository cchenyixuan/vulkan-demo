"""
_analyze_tgv.py — tables, convergence orders, effective viscosity and figures of the Taylor–Green campaign
(experiment/v8/_run_tgv_campaign.py's runs under --runs).

    .venv/Scripts/python.exe experiment/v8/_analyze_tgv.py --runs logs/v8/tgv --out docs/tgv/data --figures manuscripts/fig

Per run (summary.json / runs.csv):
  nu_eff / nu      least-squares slope of ln(E(t) / E(0)) over t in [fit window] against -2 (2 nu k^2) t: the viscosity the
                   decay actually shows (E ~ F^2 = exp(-4 nu k^2 t)); also the instantaneous value at t = 1 and 2
  velocity errors  at t = 0.5, 1, 2: relative L2 against u(x, t; nu), L2 / U0, Linf / U0, and from the snapshots the
                   relative L2 against u(x, t; nu_eff) (the field's shape error once the decay-rate error is removed)
  pressure errors  centred L2 / (rho0 U0^2) at t = 0.5, 1 and relative to the instantaneous amplitude rho0 U0^2 F^2 / 2
  density          max |rho / rho0 - 1| at t = 1; kernel sum range at t = 1; drift / overflow; steps per second

Series tables (summary.md): the Re 100 and Re 1000 resolution series with the convergence orders (log-log slopes
against dx of the errors at t = 1), the ablations, the Mach series, h/dx, the initial density and the repeats.
Figures (PDF + 300 dpi PNG, single-column width, fonts >= 7 pt): tgv_energy (E/E0 and the velocity error against t,
Re 100), tgv_convergence (errors against L/dx), tgv_viscosity (nu_eff / nu), tgv_ablation, tgv_fields (200^2 at t = 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.v8.utils import taylor_green   # noqa: E402

SERIES = {
    "re100": ["re100_n50", "re100_n100", "re100_n200", "re100_n400", "re100_n800", "re100_n1600", "re100_n3200"],
    "re1000": ["re1000_n100", "re1000_n200", "re1000_n400", "re1000_n800", "re1000_n1600", "re1000_n3200"],
    "ablation": ["re100_n200", "re100_n200_nopst", "re100_n200_nodelta", "re100_n200_nokcg", "re100_n200_vanilla"],
    "mach": ["re100_n200", "re100_n200_ma0p05", "re100_n200_ma0p02", "re100_n200_ma0p01"],
    "support": ["re100_n200", "re100_n200_hdx4", "re100_n210_hdx3p5"],
    "initial": ["re100_n200", "re100_n200_uniform"],
    "repeats": ["re100_n200", "re100_n200_r2", "re100_n200_r3"],
    "diagnostics": ["re100_n200", "re100_n200_eps0", "re100_n200_eps0p01", "re100_n200_cfl0p015"],
    "epsilon": ["re100_n200_eps0", "re100_n200_eps1e-5", "re100_n200_eps1e-4", "re100_n200_eps4e-4", "re100_n200_eps1e-3",
                "re100_n200", "re100_n200_eps0p01"],
}
LABELS = {
    "re100_n200": "default (δ+, PST, KCG; Ma 0.1; h/dx 5; ρ(p₀))", "re100_n200_nopst": "no PST",
    "re100_n200_nodelta": "no δ diffusion", "re100_n200_nokcg": "no KCG", "re100_n200_vanilla": "none (vanilla WCSPH)",
    "re100_n200_ma0p05": "Ma 0.05 (c₀ = 20)", "re100_n200_ma0p02": "Ma 0.02 (c₀ = 50)", "re100_n200_ma0p01": "Ma 0.01 (c₀ = 100)",
    "re100_n200_hdx4": "h/dx = 4", "re100_n210_hdx3p5": "h/dx = 3.5 (210²)", "re100_n200_uniform": "ρ₀ everywhere at t = 0",
    "re100_n200_r2": "repeat 2", "re100_n200_r3": "repeat 3",
    "re100_n200_eps0": "ε² = 10⁻⁸ h² (→ 0)", "re100_n200_eps0p01": "ε² = 0.01 h² (the pre-2026-10-05 value)",
    "re100_n200_cfl0p015": "CFL 0.015 (dt / 10 at Ma 0.1)",
    "re100_n200_eps1e-5": "ε² = 10⁻⁵ h²", "re100_n200_eps1e-4": "ε² = 10⁻⁴ h²", "re100_n200_eps4e-4": "ε² = 4·10⁻⁴ h²",
    "re100_n200_eps1e-3": "ε² = 10⁻³ h²",
}
ERROR_TIMES = (0.5, 1.0, 2.0)
FIT_WINDOW = (0.1, 2.0)


def load_run(run_dir: pathlib.Path) -> dict | None:
    result_path, samples_path = run_dir / "result.json", run_dir / "samples.jsonl"
    if not result_path.exists() or not samples_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = {}
    for path in sorted((run_dir / "snapshots").glob("*.npz")):
        snapshots[path.stem] = path
    return {"name": run_dir.name, "result": result, "rows": rows, "snapshots": snapshots, "dir": run_dir}


def row_at(rows: list[dict], time_value: float) -> dict | None:
    """The sample nearest to the time, or None when the run has none within 0.02 of it (a run that ended earlier)."""
    row = min(rows, key=lambda row: abs(row["time"] - time_value))
    return row if abs(row["time"] - time_value) <= 0.02 else None


def fit_viscosity(rows: list[dict], field: dict, window: tuple[float, float]) -> dict:
    """ln(E / E0) = -2 (2 nu k^2) (nu_eff / nu) t + b over the window, least squares; nu_eff / nu and the fit's rms."""
    e0 = rows[0]["kinetic_energy"]
    decay_rate = float(field["decay_rate"])           # 2 nu k^2
    points = [(row["time"], math.log(row["kinetic_energy"] / e0)) for row in rows
              if window[0] <= row["time"] <= window[1] and row["kinetic_energy"] > 0]
    if len(points) < 3:
        return {"ratio": float("nan"), "intercept": float("nan"), "rms": float("nan"), "points": len(points)}
    t = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    slope, intercept = np.polyfit(t, y, 1)
    residual = y - (slope * t + intercept)
    return {"ratio": float(-slope / (2.0 * decay_rate)), "intercept": float(intercept),
            "rms": float(math.sqrt((residual ** 2).mean())), "points": len(points), "window": list(window)}


def instantaneous_viscosity(row: dict, e0: float) -> float:
    if row["time"] <= 0 or row["kinetic_energy"] <= 0:
        return float("nan")
    return math.log(row["kinetic_energy"] / e0) / math.log(row["decay_factor_exact"] ** 2)


def instantaneous_or_nan(row: dict | None, e0: float) -> float:
    return instantaneous_viscosity(row, e0) if row is not None else float("nan")


def snapshot_errors(path: pathlib.Path, field: dict, viscosity_ratio: float) -> dict:
    """Relative L2 velocity error against the analytic field with nu (raw) and with nu_eff (shape), and the
    centred pressure error relative to the instantaneous amplitude."""
    data = np.load(path)
    time_value = float(data["time"])
    position = data["position"].astype(np.float64)
    velocity = data["velocity"].astype(np.float64)
    keywords = dict(amplitude=float(field["amplitude"]), k=float(field["wavenumber"]),
                    x0=float(field["origin"][0]), y0=float(field["origin"][1]))
    exact = taylor_green.velocity(position[:, 0], position[:, 1], time_value, viscosity=float(field["viscosity"]), **keywords)
    effective = taylor_green.velocity(position[:, 0], position[:, 1], time_value,
                                      viscosity=float(field["viscosity"]) * viscosity_ratio, **keywords)
    raw = math.sqrt(((velocity - exact) ** 2).sum() / (exact ** 2).sum())
    shape = math.sqrt(((velocity - effective) ** 2).sum() / (effective ** 2).sum())
    # best-fit amplitude: min over alpha of |v - alpha u|: alpha = <v, u> / <u, u>
    alpha = float((velocity * exact).sum() / (exact ** 2).sum())
    fitted = math.sqrt(((velocity - alpha * exact) ** 2).sum() / (exact ** 2).sum())
    pressure_value = data["pressure"].astype(np.float64)
    pressure_exact = data["exact_pressure"].astype(np.float64)
    error = pressure_value - pressure_exact
    amplitude = 0.5 * float(field["rest_density"]) * float(field["amplitude"]) ** 2 \
        * taylor_green.decay_factor(time_value, float(field["viscosity"]), float(field["wavenumber"])) ** 2
    return {"time": time_value, "velocity_relative": raw, "velocity_relative_effective": shape,
            "velocity_relative_fitted_amplitude": fitted, "amplitude_ratio": alpha,
            "pressure_centered_relative": float(math.sqrt(((error - error.mean()) ** 2).mean()) / amplitude)}


def summarize_run(run: dict) -> dict:
    result, rows = run["result"], run["rows"]
    field = result["field"]
    e0 = rows[0]["kinetic_energy"]
    fit = fit_viscosity(rows, field, (FIT_WINDOW[0], min(FIT_WINDOW[1], rows[-1]["time"])))
    summary = {
        "name": run["name"], "case": result["case"], "particles": result["particles"], "side": int(round(math.sqrt(result["particles"]))),
        "reynolds": field["reynolds"], "mach": field["mach"], "speed_of_sound": field["speed_of_sound"],
        "smoothing_length": None, "timestep": result["timestep"], "steps": result["steps"],
        "steps_per_second": result["steps_per_second"], "wall_seconds": result["wall_seconds"],
        "drift": result["drift"], "overflow": result["overflow"], "correction_fallback": result["correction_fallback"],
        "density_from_pressure": field["density_from_pressure"], "commit": result["git"]["short"],
        "viscosity_fit": fit, "viscosity_ratio": fit["ratio"],
        "viscosity_ratio_t1": instantaneous_or_nan(row_at(rows, 1.0), e0),
        "viscosity_ratio_t2": instantaneous_or_nan(row_at(rows, 2.0), e0),
        "t_end": rows[-1]["time"],
        "kinetic_energy_0": e0, "kinetic_energy_lattice_0": rows[0]["kinetic_energy_lattice"],
        "kinetic_energy_exact_0": rows[0]["kinetic_energy_exact"],
        "errors": {}, "snapshot_errors": {},
    }
    for time_value in ERROR_TIMES:
        row = row_at(rows, time_value)
        if row is None:
            summary["errors"][f"{time_value:g}"] = None
            continue
        summary["errors"][f"{time_value:g}"] = {
            "time": row["time"], "energy_ratio": row["kinetic_energy"] / e0, "energy_ratio_exact": row["decay_factor_exact"] ** 2,
            "velocity_relative": row["velocity_l2_relative"], "velocity_l2": row["velocity_l2"], "velocity_linf": row["velocity_linf"],
            "max_speed_ratio": row["max_speed"] / row["max_speed_exact"],
            "pressure_l2": row["pressure_l2"], "pressure_l2_centered": row["pressure_l2_centered"],
            "pressure_centered_relative": row["pressure_l2_centered"] / (0.5 * row["decay_factor_exact"] ** 2),
            "pressure_offset": row["pressure_offset"], "density_max_deviation": row["density_max_deviation"],
            "density_mean_deviation": row["density_mean_deviation"],
            "kernel_sum_min": row["kernel_sum_min"], "kernel_sum_max": row["kernel_sum_max"]}
    for stem, path in run["snapshots"].items():
        if stem == "t0":
            continue
        summary["snapshot_errors"][stem] = snapshot_errors(path, field, fit["ratio"])
    # the case's h from the case.yaml next to the run's case path
    case_path = _REPOSITORY_ROOT / result["case"]
    if case_path.exists():
        for line in case_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("h:"):
                summary["smoothing_length"] = float(line.split(":")[1])
                break
    if summary["smoothing_length"]:
        summary["support_over_spacing"] = round(summary["smoothing_length"] * summary["side"] / float(field["length"]), 3)
    return summary


def log_slope(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a > 0 and b is not None and not (isinstance(b, float) and math.isnan(b)) and b > 0]
    if len(pairs) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs]), 1)
    return float(slope)


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        if abs(value) >= 100 or value == 0:
            return f"{value:.{digits}g}"
        return f"{value:.{digits}g}"
    return str(value)


def series_table(summaries: dict[str, dict], names: list[str], label_column: str = "run") -> list[str]:
    use_labels = label_column != "run"
    lines = [f"| {label_column} | N | nu_eff/nu (fit) | at t=1 | at t=2 | E/E0 t=1 (exact) | v rel L2 t=1 | v rel L2 t=2 | "
             f"shape err t=1 | shape err t=2 | p err t=1 (rel ampl) | max rho dev t=1 | steps/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        s = summaries.get(name)
        if not s:
            continue
        e1, e2 = s["errors"]["1"] or {}, s["errors"]["2"] or {}
        shape1 = s["snapshot_errors"].get("t1", {}).get("velocity_relative_effective")
        shape2 = s["snapshot_errors"].get("t2", {}).get("velocity_relative_effective")
        lines.append(f"| {LABELS.get(name, name) if use_labels else name} | {s['side']}² | {fmt(s['viscosity_ratio'], 4)} | {fmt(s['viscosity_ratio_t1'], 4)} | "
                     f"{fmt(s['viscosity_ratio_t2'], 4)} | {fmt(e1.get('energy_ratio'), 4)} ({fmt(e1.get('energy_ratio_exact'), 4)}) | "
                     f"{fmt(e1.get('velocity_relative'))} | {fmt(e2.get('velocity_relative'))} | {fmt(shape1)} | {fmt(shape2)} | "
                     f"{fmt(e1.get('pressure_centered_relative'))} | {fmt(e1.get('density_max_deviation'), 2)} | {fmt(s['steps_per_second'], 4)} |")
    return lines


def convergence_rows(summaries: dict[str, dict], names: list[str]) -> dict:
    present = [summaries[n] for n in names if n in summaries]
    if len(present) < 2:
        return {}
    dx = [1.0 / s["side"] for s in present]
    out = {"runs": [s["name"] for s in present], "dx": dx}
    nan = float("nan")
    for key, getter in (("velocity_relative_t1", lambda s: (s["errors"]["1"] or {}).get("velocity_relative", nan)),
                        ("velocity_relative_t2", lambda s: (s["errors"]["2"] or {}).get("velocity_relative", nan)),
                        ("shape_t1", lambda s: s["snapshot_errors"].get("t1", {}).get("velocity_relative_effective", float("nan"))),
                        ("shape_t2", lambda s: s["snapshot_errors"].get("t2", {}).get("velocity_relative_effective", float("nan"))),
                        ("fitted_amplitude_t1", lambda s: s["snapshot_errors"].get("t1", {}).get("velocity_relative_fitted_amplitude", float("nan"))),
                        ("pressure_t1", lambda s: (s["errors"]["1"] or {}).get("pressure_centered_relative", nan)),
                        ("viscosity_deficit", lambda s: abs(1.0 - s["viscosity_ratio"]))):
        values = [getter(s) for s in present]
        out[key] = {"values": values, "order": log_slope(dx, values)}
    return out


# ------------------------------------------------------------------------------------------------------- figures

def setup_figures(font_size: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from _plot_single_baseline_paper import setup_style
    setup_style(font_size)


def save(fig, out: pathlib.Path) -> None:
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")


PALETTE = ["#0b2e5c", "#1b4f9c", "#2a78d6", "#4ea1e8", "#7fc1f0", "#b3dcf7", "#d9ecfa", "#c0392b"]


def figure_energy(runs: dict, summaries: dict, out: pathlib.Path, width: float, height: float) -> None:
    import matplotlib.pyplot as plt
    from _plot_single_baseline_paper import tidy
    names = [n for n in SERIES["re100"] if n in runs]
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    colors = PALETTE[:7][7 - len(names):][::-1]
    for color, name in zip(colors, names):
        rows = runs[name]["rows"]
        e0 = rows[0]["kinetic_energy"]
        t = [r["time"] for r in rows]
        axes[0].plot(t, [r["kinetic_energy"] / e0 for r in rows], color=color, linewidth=0.9, label=f"{summaries[name]['side']}²")
        axes[1].plot(t, [100.0 * r["velocity_l2_relative"] for r in rows], color=color, linewidth=0.9)
    rows = runs[names[-1]]["rows"]
    axes[0].plot([r["time"] for r in rows], [r["decay_factor_exact"] ** 2 for r in rows], color="black", linestyle=(0, (3, 2)),
                 linewidth=0.8, label="exact")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$E_k(t)/E_k(0)$")
    axes[0].legend(frameon=False, ncol=2, handlelength=1.8, loc="lower left")
    axes[1].set_ylabel("rel. $L_2$ velocity error (%)")
    axes[1].set_xlabel("$t\\,U_0/L$")
    for ax in axes:
        ax.grid(True, axis="y")
        tidy(ax)
    axes[0].text(0.98, 0.95, "(a)", transform=axes[0].transAxes, ha="right", va="top")
    axes[1].text(0.98, 0.95, "(b)", transform=axes[1].transAxes, ha="right", va="top")
    fig.subplots_adjust(hspace=0.12)
    save(fig, out)
    plt.close(fig)


def figure_convergence(summaries: dict, convergence: dict, out: pathlib.Path, width: float, height: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from _plot_single_baseline_paper import tidy
    fig, ax = plt.subplots(figsize=(width, height))
    styles = {"re100": dict(color=PALETTE[2]), "re1000": dict(color=PALETTE[7])}
    for series, style in styles.items():
        conv = convergence.get(series)
        if not conv:
            continue
        sides = [summaries[n]["side"] for n in conv["runs"]]
        ax.plot(sides, [100 * v for v in conv["velocity_relative_t1"]["values"]], marker="o", markersize=3.5, linewidth=0.9,
                label=f"Re {series[2:]}: velocity, $t=1$", **style)
        ax.plot(sides, [100 * v for v in conv["shape_t1"]["values"]], marker="s", markersize=3.2, linewidth=0.9,
                linestyle=(0, (4, 2)), markerfacecolor="white", label=f"Re {series[2:]}: velocity vs $\\nu_\\mathrm{{eff}}$", **style)
        ax.plot(sides, [100 * v for v in conv["pressure_t1"]["values"]], marker="^", markersize=3.5, linewidth=0.9,
                linestyle=(0, (1, 1.5)), label=f"Re {series[2:]}: pressure", **style)
    # slope guides anchored below the Re 100 shape error at 100^2
    x_guide = np.array([100.0, 400.0])
    conv = convergence.get("re100")
    base = 100 * conv["shape_t1"]["values"][conv["runs"].index("re100_n100")] * 0.6 if conv and "re100_n100" in conv["runs"] else 0.2
    for order, text in ((1, "1st"), (2, "2nd")):
        y_guide = base * (x_guide / x_guide[0]) ** (-order)
        ax.plot(x_guide, y_guide, color="#888888", linewidth=0.6)
        ax.annotate(text, (x_guide[1], y_guide[1]), fontsize=7, color="#555555", xytext=(3, -2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$L/\\Delta x$")
    ax.set_ylabel("rel. $L_2$ error at $t=1$ (%)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(sorted({summaries[n]["side"] for s in ("re100", "re1000") for n in SERIES[s] if n in summaries}))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=6.5, handlelength=2.4, loc="best")
    tidy(ax)
    save(fig, out)
    plt.close(fig)


def figure_viscosity(summaries: dict, out: pathlib.Path, width: float, height: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from _plot_single_baseline_paper import tidy
    fig, ax = plt.subplots(figsize=(width, height))
    for series, color, label in (("re100", PALETTE[2], "Re 100"), ("re1000", PALETTE[7], "Re 1000")):
        names = [n for n in SERIES[series] if n in summaries]
        ax.plot([summaries[n]["side"] for n in names], [summaries[n]["viscosity_ratio"] for n in names], marker="o",
                markersize=3.5, linewidth=0.9, color=color, label=f"{label}, $h/\\Delta x = 5$")
    extras = [("re100_n200_hdx4", "s", "$h/\\Delta x = 4$"), ("re100_n210_hdx3p5", "D", "$h/\\Delta x = 3.5$"),
              ("re100_n200_ma0p01", "^", "Ma 0.01")]
    for name, marker, label in extras:
        if name in summaries:
            ax.plot([summaries[name]["side"]], [summaries[name]["viscosity_ratio"]], marker=marker, markersize=4,
                    linestyle="none", color="#333333", markerfacecolor="white", label=f"Re 100, {label}")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks(sorted({summaries[n]["side"] for s in ("re100", "re1000") for n in SERIES[s] if n in summaries}))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("$L/\\Delta x$")
    ax.set_ylabel("$\\nu_\\mathrm{eff}/\\nu$ from the energy decay")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    tidy(ax)
    save(fig, out)
    plt.close(fig)


def figure_ablation(runs: dict, summaries: dict, out: pathlib.Path, width: float, height: float) -> None:
    import matplotlib.pyplot as plt
    from _plot_single_baseline_paper import tidy
    names = [n for n in SERIES["ablation"] if n in runs]
    short = {"re100_n200": "δ+ (default)", "re100_n200_nopst": "no PST", "re100_n200_nodelta": "no δ",
             "re100_n200_nokcg": "no KCG", "re100_n200_vanilla": "vanilla"}
    colors = {"re100_n200": PALETTE[2], "re100_n200_nopst": "#c0392b", "re100_n200_nodelta": "#2e8b57",
              "re100_n200_nokcg": "#8e44ad", "re100_n200_vanilla": "#777777"}
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    for name in names:
        rows = runs[name]["rows"]
        e0 = rows[0]["kinetic_energy"]
        t = [r["time"] for r in rows]
        axes[0].plot(t, [r["kinetic_energy"] / e0 for r in rows], color=colors[name], linewidth=0.9, label=short[name])
        axes[1].plot(t, [100.0 * r["velocity_l2_relative"] for r in rows], color=colors[name], linewidth=0.9)
    rows = runs[names[0]]["rows"]
    axes[0].plot([r["time"] for r in rows], [r["decay_factor_exact"] ** 2 for r in rows], color="black",
                 linestyle=(0, (3, 2)), linewidth=0.8, label="exact")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$E_k(t)/E_k(0)$")
    axes[0].legend(frameon=False, ncol=2, handlelength=1.8, loc="lower left", fontsize=6.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.03, 80)
    axes[1].set_ylabel("rel. $L_2$ velocity error (%)")
    axes[1].set_xlabel("$t\\,U_0/L$")
    for ax in axes:
        ax.grid(True, axis="y")
        tidy(ax)
    axes[0].text(0.98, 0.95, "(a)", transform=axes[0].transAxes, ha="right", va="top")
    axes[1].text(0.98, 0.95, "(b)", transform=axes[1].transAxes, ha="right", va="top")
    fig.subplots_adjust(hspace=0.12)
    save(fig, out)
    plt.close(fig)


def figure_fields(run: dict, out: pathlib.Path, width: float) -> None:
    """Particle scatter at t = 1 (the 200^2 default run): speed, pressure, velocity error, and the particles near a
    periodic edge coloured by speed (the streams cross the box edges)."""
    import matplotlib.pyplot as plt
    path = run["snapshots"].get("t1")
    if path is None:
        return
    data = np.load(path)
    position = data["position"]
    velocity = data["velocity"]
    speed = np.sqrt((velocity ** 2).sum(axis=1))
    error = np.sqrt(((velocity - data["exact_velocity"]) ** 2).sum(axis=1))
    pressure_value = data["pressure"] - data["pressure"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 1.02))
    panels = [(axes[0, 0], speed, "speed $|v|/U_0$", "viridis", None),
              (axes[0, 1], pressure_value, "pressure $p/(\\rho_0 U_0^2)$ (mean removed)", "RdBu_r", "sym"),
              (axes[1, 0], error, "velocity error $|v - u|/U_0$", "magma", None)]
    for ax, values, title, cmap, kind in panels:
        limit = float(np.abs(values).max()) if kind == "sym" else None
        sc = ax.scatter(position[:, 0], position[:, 1], c=values, s=0.25, cmap=cmap, linewidths=0,
                        vmin=-limit if limit else None, vmax=limit if limit else None, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6, length=2)
        bar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=6, length=2)
    ax = axes[1, 1]
    window = 0.06
    near = (position[:, 0] > 1.0 - window) | (position[:, 0] < window)
    near &= (position[:, 1] > 0.2) & (position[:, 1] < 0.2 + 2 * window)
    shifted_x = np.where(position[near, 0] < 0.5, position[near, 0] + 1.0, position[near, 0])
    sc = ax.scatter(shifted_x, position[near, 1], c=speed[near], s=6, cmap="viridis", linewidths=0)
    ax.axvline(1.0, color="black", linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_aspect("equal")
    ax.set_title("particles across the periodic edge $x = L$", fontsize=7, pad=3)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel("$x/L$ (right part shown at $x+L$)", fontsize=6.5)
    bar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=6, length=2)
    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- report

def write_outputs(summaries: dict, convergence: dict, out: pathlib.Path, title: str = "Taylor–Green campaign summary") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"runs": summaries, "convergence": convergence,
                                                   "series": SERIES, "fit_window": FIT_WINDOW}, indent=1, default=float),
                                      encoding="utf-8")
    lines = [f"# {title} (generated by experiment/v8/_analyze_tgv.py)", ""]
    lines += ["## Re = 100, h/dx = 5, Ma = 0.1: resolution series", ""] + series_table(summaries, SERIES["re100"]) + [""]
    conv = convergence.get("re100", {})
    if conv:
        lines += ["orders (log-log slope against dx): " + ", ".join(
            f"{key} {conv[key]['order']:.2f}" for key in ("velocity_relative_t1", "shape_t1", "fitted_amplitude_t1", "pressure_t1", "viscosity_deficit")), ""]
    lines += ["## Re = 1000", ""] + series_table(summaries, SERIES["re1000"]) + [""]
    conv = convergence.get("re1000", {})
    if conv:
        lines += ["orders: " + ", ".join(
            f"{key} {conv[key]['order']:.2f}" for key in ("velocity_relative_t1", "shape_t1", "pressure_t1")), ""]
    for key, title in (("ablation", "Ablations (Re 100, 200²)"), ("mach", "Mach number (Re 100, 200²)"),
                       ("support", "Kernel support h/dx (Re 100)"), ("initial", "Initial density (Re 100, 200²)"),
                       ("repeats", "Repeats (Re 100, 200²)"), ("diagnostics", "Diagnostics: ε² and dt (Re 100, 200²)"),
                       ("epsilon", "ε² sweep (Re 100, 200²)")):
        lines += [f"## {title}", ""] + series_table(summaries, SERIES[key], "variant") + [""]
    lines += ["## Every run", "", "| run | particles | dt | steps | steps/s | wall s | drift | overflow | KCG fallback | commit |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['particles']:,} | {s['timestep']:.4g} | {s['steps']:,} | {s['steps_per_second']:.0f} | "
                     f"{s['wall_seconds']:.0f} | {s['drift']} | {s['overflow'] or 0} | {s['correction_fallback']} | {s['commit']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "side", "particles", "reynolds", "mach", "support_over_spacing", "timestep", "steps", "steps_per_second",
                         "viscosity_ratio_fit", "viscosity_ratio_t1", "viscosity_ratio_t2",
                         "energy_ratio_t1", "energy_ratio_exact_t1", "velocity_relative_t05", "velocity_relative_t1", "velocity_relative_t2",
                         "velocity_l2_t1", "velocity_linf_t1", "shape_t1", "shape_t2", "fitted_amplitude_t1", "amplitude_ratio_t1",
                         "pressure_centered_relative_t1", "pressure_l2_centered_t1", "pressure_offset_t1",
                         "density_max_deviation_t1", "kernel_sum_min_t1", "drift", "overflow", "correction_fallback", "commit"])
        for name, s in summaries.items():
            e05, e1, e2 = s["errors"]["0.5"] or {}, s["errors"]["1"] or {}, s["errors"]["2"] or {}
            s1 = s["snapshot_errors"].get("t1", {})
            s2 = s["snapshot_errors"].get("t2", {})
            writer.writerow([name, s["side"], s["particles"], s["reynolds"], s["mach"], s.get("support_over_spacing"), s["timestep"],
                             s["steps"], s["steps_per_second"], s["viscosity_ratio"], s["viscosity_ratio_t1"], s["viscosity_ratio_t2"],
                             e1.get("energy_ratio"), e1.get("energy_ratio_exact"), e05.get("velocity_relative"), e1.get("velocity_relative"),
                             e2.get("velocity_relative"), e1.get("velocity_l2"), e1.get("velocity_linf"), s1.get("velocity_relative_effective"),
                             s2.get("velocity_relative_effective"), s1.get("velocity_relative_fitted_amplitude"), s1.get("amplitude_ratio"),
                             e1.get("pressure_centered_relative"), e1.get("pressure_l2_centered"), e1.get("pressure_offset"),
                             e1.get("density_max_deviation"), e1.get("kernel_sum_min"), s["drift"], json.dumps(s["overflow"]),
                             s["correction_fallback"], s["commit"]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="logs/v8/tgv")
    parser.add_argument("--out", default="docs/tgv/data")
    parser.add_argument("--figures", default="manuscripts/fig")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--figure-suffix", default="", help="appended to every figure name (a second run set)")
    parser.add_argument("--title", default="Taylor–Green campaign summary", help="summary.md title")
    arguments = parser.parse_args()
    runs_root = _REPOSITORY_ROOT / arguments.runs
    runs = {}
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir():
            loaded = load_run(run_dir)
            if loaded and loaded["result"].get("status") == "ok":
                runs[run_dir.name] = loaded
    summaries = {name: summarize_run(run) for name, run in runs.items()}
    convergence = {series: convergence_rows(summaries, SERIES[series]) for series in ("re100", "re1000")}
    out = _REPOSITORY_ROOT / arguments.out
    write_outputs(summaries, convergence, out, arguments.title)
    figures = _REPOSITORY_ROOT / arguments.figures
    figures.mkdir(parents=True, exist_ok=True)
    setup_figures(arguments.font_size)
    width = arguments.width
    suffix = arguments.figure_suffix
    if any(n in runs for n in SERIES["re100"]):
        figure_energy(runs, summaries, figures / f"tgv_energy{suffix}", width, width * 1.25)
    if convergence.get("re100"):
        figure_convergence(summaries, convergence, figures / f"tgv_convergence{suffix}", width, width * 0.8)
        figure_viscosity(summaries, figures / f"tgv_viscosity{suffix}", width, width * 0.75)
    if len([n for n in SERIES["ablation"] if n in runs]) >= 2:
        figure_ablation(runs, summaries, figures / f"tgv_ablation{suffix}", width, width * 1.25)
    if "re100_n200" in runs:
        figure_fields(runs["re100_n200"], figures / f"tgv_fields{suffix}", width * 2.0)
    print(f"[analyze] {len(runs)} runs -> {out} ; figures in {figures}")
    for series in ("re100", "re1000"):
        conv = convergence.get(series)
        if conv:
            print(f"[analyze] {series}: " + ", ".join(f"{k} order {conv[k]['order']:.2f}" for k in
                                                   ("velocity_relative_t1", "shape_t1", "pressure_t1", "viscosity_deficit")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
