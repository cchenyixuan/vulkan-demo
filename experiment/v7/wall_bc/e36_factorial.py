"""e36_factorial.py - the 2 x 2 of the E36 wall diagnostics (CPU only): no-slip (the Adami dummy velocity in the
viscous term) x wall pressure (the Adami p_w; the walls store rho0 as with the v6 walls), from one e36_compare.py
summary that holds the four cells at one resolution:

    neither   = WALL_BC 0 (v6 walls, or v6 itself)  no-slip off, pressure off
    no-slip   = WALL_BC 2 (no-slip only)            no-slip on,  pressure off
    pressure  = WALL_BC 4 (pressure only)           no-slip off, pressure on
    both      = WALL_BC 3 (adami_rho0)              no-slip on,  pressure on

Quantity d = the deviation of psi_min and of the three centre-line extrema (u_min, v_max, v_min) from Marchi 2021, in
percent of the magnitude (e36_compare.deficit: 100 (|value| - |reference|) / |reference|; positive = stronger than the
reference), wall-row frame, time average over the run's window. Effects in percentage points:

    simple effects   no-slip | pressure off = d(no-slip) - d(neither),   no-slip | pressure on = d(both) - d(pressure)
                     pressure | no-slip off = d(pressure) - d(neither),  pressure | no-slip on = d(both) - d(no-slip)
    main effects     the mean of the factor's two simple effects
    interaction      no-slip | pressure on - no-slip | pressure off  (= pressure | no-slip on - pressure | no-slip off)
                     = d(both) - d(pressure) - d(no-slip) + d(neither), i.e. d(both) = d(neither) + the two simple
                     effects at the other factor's off level + the interaction. (This is the difference of the simple
                     effects; the "AB effect" of the +-1 coded convention is half of it.)

Noise scale (one run per cell; standard errors assumed independent between cells): the extrema's sem of
cavity_analysis (std * sqrt(tau_int / n) of the time series at the extremum position) and, for psi_min, half the
difference of the two half-window minima (e36_compare psi_halves); a main effect's standard error is
0.5 sqrt(sum sigma^2), the interaction's sqrt(sum sigma^2).

With --post (an e36_post.py summary holding the same four labels under "transport"), psi_min is also given from the
checkpoint states (the mean over the states in the window) for the stored velocity and for the transport velocity
u + shift / dt: the Adami wall pressure gives the stored velocity a net flux that PST cancels, which biases the
stored-velocity psi_min of the pressure-on cells (E36 section 7.4).

Gates (refused): a cell that is not K = 1, a different case or dt, a different averaging window.

Writes <out>/factorial.json and <out>/factorial_tables.md and prints the tables.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.e36_factorial --summary docs/wall_bc/e36_n250_factorial/e36_summary.json \\
        --neither "v6 K=1" --no-slip "诊断 2:只换无滑移" --pressure "诊断 4:只换压力" --both "adami_rho0" \\
        --post docs/wall_bc/e36_n250_factorial/post/post_summary.json --out docs/wall_bc/e36_n250_factorial
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference  # noqa: E402
from experiment.v7.wall_bc.e36_compare import deficit  # noqa: E402

QUANTITIES = (("psi_min", "ψ_min"), ("u_min", "u_min"), ("v_max", "v_max"), ("v_min", "v_min"))
CELLS = ("neither", "no_slip", "pressure", "both")
CELL_NAMES = {"neither": "v6 墙(WALL_BC 0):两者皆无", "no_slip": "只换无滑移(WALL_BC 2)",
              "pressure": "只换压力(WALL_BC 4)", "both": "adami_rho0(WALL_BC 3):两者皆有"}
EFFECT_ROWS = (("no_slip_at_pressure_off", "无滑移的简单效应,压力关(2 − 0)"),
               ("no_slip_at_pressure_on", "无滑移的简单效应,压力开(3 − 4)"),
               ("pressure_at_no_slip_off", "压力的简单效应,无滑移关(4 − 0)"),
               ("pressure_at_no_slip_on", "压力的简单效应,无滑移开(3 − 2)"),
               ("main_no_slip", "**无滑移的主效应**"),
               ("main_pressure", "**压力边界的主效应**"),
               ("interaction", "**交互项**(3 − 4 − 2 + 0)"))


def reference_values() -> dict:
    reference = cavity_reference.marchi2021()["extrema"]
    return {name: reference[name][0] for name, _ in QUANTITIES}


def deviations(item: dict) -> tuple[dict, dict]:
    """(deviation in %, its noise scale in percentage points) of the four quantities of one e36_compare run."""
    reference = reference_values()
    primary = item["stream"]["vortices"]["primary"]
    value = {"psi_min": deficit(primary["psi"], reference["psi_min"])}
    halves = primary.get("psi_halves") or [None, None]
    noise = {"psi_min": (50.0 * abs(halves[1] - halves[0]) / abs(reference["psi_min"])
                         if None not in halves else math.nan)}
    for name in ("u_min", "v_max", "v_min"):
        value[name] = deficit(item["extrema"][name]["value"], reference[name])
        sem = item["extrema"][name].get("sem")
        noise[name] = 100.0 * sem / abs(reference[name]) if sem is not None else math.nan
    return value, noise


def effects(cell: dict) -> dict:
    no_slip_at_pressure_off = cell["no_slip"] - cell["neither"]
    no_slip_at_pressure_on = cell["both"] - cell["pressure"]
    pressure_at_no_slip_off = cell["pressure"] - cell["neither"]
    pressure_at_no_slip_on = cell["both"] - cell["no_slip"]
    return {"no_slip_at_pressure_off": no_slip_at_pressure_off, "no_slip_at_pressure_on": no_slip_at_pressure_on,
            "pressure_at_no_slip_off": pressure_at_no_slip_off, "pressure_at_no_slip_on": pressure_at_no_slip_on,
            "main_no_slip": 0.5 * (no_slip_at_pressure_off + no_slip_at_pressure_on),
            "main_pressure": 0.5 * (pressure_at_no_slip_off + pressure_at_no_slip_on),
            "interaction": no_slip_at_pressure_on - no_slip_at_pressure_off}


def effect_noise(sigma: dict) -> dict:
    total = math.sqrt(sum(sigma[cell] ** 2 for cell in CELLS))
    pair = {"no_slip_at_pressure_off": ("no_slip", "neither"), "no_slip_at_pressure_on": ("both", "pressure"),
            "pressure_at_no_slip_off": ("pressure", "neither"), "pressure_at_no_slip_on": ("both", "no_slip")}
    result = {key: math.hypot(sigma[a], sigma[b]) for key, (a, b) in pair.items()}
    result.update(main_no_slip=0.5 * total, main_pressure=0.5 * total, interaction=total)
    return result


def table(title: str, symbols: list, cell_values: dict, cell_noise: dict | None, effect_values: dict,
          effect_errors: dict | None) -> str:
    def cell_text(value, noise):
        return f"{value:+.2f}" + (f" ± {noise:.2f}" if noise is not None and not math.isnan(noise) else "")
    lines = [f"| {title} | " + " | ".join(symbols) + " |", "|---|" + "---|" * len(symbols)]
    for cell in CELLS:
        lines.append(f"| {CELL_NAMES[cell]} | " + " | ".join(
            cell_text(cell_values[cell][name], cell_noise[cell][name] if cell_noise else None)
            for name in cell_values[cell]) + " |")
    lines.append("| *效应(百分点)* |" + " |" * len(symbols))
    for key, name in EFFECT_ROWS:
        lines.append(f"| {name} | " + " | ".join(
            cell_text(effect_values[quantity][key], effect_errors[quantity][key] if effect_errors else None)
            for quantity in effect_values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="e36_compare.py e36_summary.json with the four cells")
    parser.add_argument("--neither", required=True, help="label of the WALL_BC 0 (or v6) run")
    parser.add_argument("--no-slip", required=True, help="label of the WALL_BC 2 run")
    parser.add_argument("--pressure", required=True, help="label of the WALL_BC 4 run")
    parser.add_argument("--both", required=True, help="label of the WALL_BC 3 run")
    parser.add_argument("--post", default=None, help="e36_post.py post_summary.json with the same four labels")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    runs = {item["label"]: item for item in json.loads(pathlib.Path(arguments.summary).read_text(encoding="utf-8"))["runs"]}
    labels = {"neither": arguments.neither, "no_slip": arguments.no_slip, "pressure": arguments.pressure,
              "both": arguments.both}
    expected_wall_bc = {"neither": (0, None), "no_slip": (2,), "pressure": (4,), "both": (3,)}
    for cell, label in labels.items():
        if label not in runs:
            sys.exit(f"--{cell.replace('_', '-')} {label!r}: not in {sorted(runs)}")
        run = runs[label]
        if run.get("wall_bc") not in expected_wall_bc[cell]:
            sys.exit(f"--{cell.replace('_', '-')} {label!r} has wall_bc {run.get('wall_bc')}, expected {expected_wall_bc[cell]}")
        if run.get("slabs") != 1:
            sys.exit(f"--{cell.replace('_', '-')} {label!r}: K = {run.get('slabs')}, the cells must be K = 1 runs")
    first = runs[labels["neither"]]
    for cell, label in labels.items():
        run = runs[label]
        for key in ("case", "dt"):
            if run.get(key) is None or run.get(key) != first.get(key):
                sys.exit(f"{label!r}: {key} {run.get(key)!r} differs from {first.get(key)!r} (or is missing: re-run "
                         "e36_compare.py, which records case and dt since the WALL_BC 4 commit)")
        if any(abs(a - b) > 1e-6 for a, b in zip(run["window"], first["window"])):
            sys.exit(f"{label!r}: averaging window {run['window']} differs from {first['window']}")

    cell_values, cell_noise = {}, {}
    for cell, label in labels.items():
        cell_values[cell], cell_noise[cell] = deviations(runs[label])
    effect_values = {name: effects({cell: cell_values[cell][name] for cell in CELLS}) for name, _ in QUANTITIES}
    effect_errors = {name: effect_noise({cell: cell_noise[cell][name] for cell in CELLS}) for name, _ in QUANTITIES}
    symbols = [symbol for _, symbol in QUANTITIES]
    markdown = table("量(相对 Marchi 的偏差,%;± 噪声尺度)", symbols, cell_values, cell_noise, effect_values, effect_errors)
    report = {"labels": labels, "window": first["window"], "case": first["case"],
              "deviation_percent": cell_values, "noise_percentage_points": cell_noise,
              "effects_percentage_points": effect_values, "effect_noise_percentage_points": effect_errors,
              "cells": {cell: {"git": runs[label].get("git"), "fps": runs[label].get("fps", {}).get("median"),
                               "mean_velocity": runs[label].get("mean_velocity"),
                               "fluid_density_range_window": runs[label].get("fluid_density_range_window"),
                               "t_steady_online": runs[label].get("t_steady_online")}
                        for cell, label in labels.items()}}

    if arguments.post:
        transport = json.loads(pathlib.Path(arguments.post).read_text(encoding="utf-8"))["transport"]
        reference = reference_values()
        post_values = {}
        for cell, label in labels.items():
            if label not in transport:
                sys.exit(f"--post: {label!r} not in {sorted(transport)}")
            states = transport[label]["states"]
            post_values[cell] = {
                kind: deficit(sum(state[kind]["psi_min"]["psi"] for state in states) / len(states), reference["psi_min"])
                for kind in ("stored", "transport")}
            post_values[cell]["states"] = [state["time"] for state in states]
        post_effects = {kind: effects({cell: post_values[cell][kind] for cell in CELLS}) for kind in ("stored", "transport")}
        lines = ["| ψ_min 在检查点状态上(相对 Marchi 的偏差,%) | 存储速度 | 输运速度 u + δr/dt |", "|---|---|---|"]
        for cell in CELLS:
            lines.append(f"| {CELL_NAMES[cell]}(t = {', '.join(f'{time:.2f}' for time in post_values[cell]['states'])}) | "
                         f"{post_values[cell]['stored']:+.2f} | {post_values[cell]['transport']:+.2f} |")
        lines.append("| *效应(百分点)* | | |")
        for key, name in EFFECT_ROWS:
            lines.append(f"| {name} | {post_effects['stored'][key]:+.2f} | {post_effects['transport'][key]:+.2f} |")
        markdown += "\n" + "\n".join(lines) + "\n"
        report["checkpoint_psi_min"] = {"deviation_percent": post_values, "effects_percentage_points": post_effects}

    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "factorial.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "factorial_tables.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
