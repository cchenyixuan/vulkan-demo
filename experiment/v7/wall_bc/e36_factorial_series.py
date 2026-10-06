"""e36_factorial_series.py - the E36 2 x 2 at two resolutions (CPU only): per cell the deviation at both resolutions and
the first-order two-point limit f_inf = 2 f_fine - f_coarse; per effect (simple, main, interaction) both values, the
ratio fine / coarse and the split "constant + first order" c + s h / h_fine at an assumed order p
(c = f_fine - (f_coarse - f_fine) / (2^p - 1)) for p = 0.8, 1, 1.2 and the order of a pure power law without constant
(log2(f_coarse / f_fine)). Two resolutions cannot fix the order: the p columns show how much the constant depends on it.

Inputs: the factorial.json of e36_factorial.py at the coarse and at the fine resolution (refinement ratio 2).
Writes <out>/factorial_series.md and <out>/factorial_series.json and prints the tables.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.e36_factorial_series \\
        --coarse docs/wall_bc/e36_n250_factorial/factorial.json --fine docs/wall_bc/e36_n500_factorial/factorial.json \\
        --out docs/wall_bc/e36_factorial_series
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

QUANTITIES = (("psi_min", "ψ_min"), ("u_min", "u_min"), ("v_max", "v_max"), ("v_min", "v_min"))
CELLS = (("neither", "v6 墙(0)"), ("no_slip", "只换无滑移(2)"), ("pressure", "只换压力(4)"), ("both", "adami_rho0(3)"))
EFFECTS = (("no_slip_at_pressure_off", "无滑移,压力关(2 − 0)"), ("no_slip_at_pressure_on", "无滑移,压力开(3 − 4)"),
           ("pressure_at_no_slip_off", "压力,无滑移关(4 − 0)"), ("pressure_at_no_slip_on", "压力,无滑移开(3 − 2)"),
           ("main_no_slip", "**无滑移主效应**"), ("main_pressure", "**压力主效应**"), ("interaction", "**交互项**"))
ORDERS = (0.8, 1.0, 1.2)


def constant(coarse: float, fine: float, order: float) -> float:
    return fine - (coarse - fine) / (2.0 ** order - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse", required=True)
    parser.add_argument("--fine", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    coarse = json.loads(pathlib.Path(arguments.coarse).read_text(encoding="utf-8"))
    fine = json.loads(pathlib.Path(arguments.fine).read_text(encoding="utf-8"))
    if coarse["case"] == fine["case"]:
        sys.exit("--coarse and --fine are the same case")
    report = {"coarse": arguments.coarse, "fine": arguments.fine, "cells": {}, "effects": {}}
    lines = ["| 格子(相对 Marchi 的偏差,%) | 量 | 粗 | 细 | 一阶两点极限 |", "|---|---|---|---|---|"]
    for cell, name in CELLS:
        report["cells"][cell] = {}
        for quantity, symbol in QUANTITIES:
            f_coarse = coarse["deviation_percent"][cell][quantity]
            f_fine = fine["deviation_percent"][cell][quantity]
            limit = 2.0 * f_fine - f_coarse
            report["cells"][cell][quantity] = {"coarse": f_coarse, "fine": f_fine, "limit_first_order": limit}
            lines.append(f"| {name} | {symbol} | {f_coarse:+.2f} | {f_fine:+.2f} | {limit:+.2f} |")
    lines += ["", "| 效应(百分点) | 量 | 粗 | 细 | 细 / 粗 | 常数 p = 0.8 | 常数 p = 1 | 常数 p = 1.2 | 纯幂律的阶 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for effect, name in EFFECTS:
        report["effects"][effect] = {}
        for quantity, symbol in QUANTITIES:
            e_coarse = coarse["effects_percentage_points"][quantity][effect]
            e_fine = fine["effects_percentage_points"][quantity][effect]
            ratio = e_fine / e_coarse if e_coarse else math.nan
            constants = {order: constant(e_coarse, e_fine, order) for order in ORDERS}
            power = math.log2(e_coarse / e_fine) if e_coarse * e_fine > 0 else math.nan
            report["effects"][effect][quantity] = {"coarse": e_coarse, "fine": e_fine, "ratio": ratio,
                                                   "constant": {str(order): value for order, value in constants.items()},
                                                   "pure_power_order": power}
            if effect == "interaction":
                lines.append(f"| {name} | {symbol} | {e_coarse:+.2f} | {e_fine:+.2f} | – | – | – | – | – |")
            else:
                lines.append(f"| {name} | {symbol} | {e_coarse:+.2f} | {e_fine:+.2f} | {ratio:.2f} | "
                             + " | ".join(f"{constants[order]:+.2f}" for order in ORDERS)
                             + f" | {power:.2f} |")
    markdown = "\n".join(lines) + "\n"
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "factorial_series.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "factorial_series.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
