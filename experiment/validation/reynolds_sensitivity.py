"""reynolds_sensitivity.py - does the measured effective viscosity explain the extremum shifts? (discussion only)

The released viscous operator delivers nu_eff = F nu with F ~ 0.85 (effective_viscosity.py), so a run solves the
cavity at Re_eff = 1000 / F. Marchi 2021 covers Re = 1000 only, so the sensitivity of the extrema to Re is taken from
the tabulated points of Ghia et al. (1982) at Re = 400, 1000 and 3200 (docs/validation/data/ghia1982_extrema.csv):
for each extremum the tabulated point with the largest |value| gives a position (quantised to the 17 tabulated points
per line) and a value. The derivative with respect to ln Re at Re = 1000 is bracketed by the 1000-3200 secant and by
the slope at 1000 of the parabola through the three Reynolds numbers.

Two checks, both written as CSV and as markdown sections (tables_reynolds.md):
  1. the change for Re 1000 -> 1000 / F_release, against the release runs' 1000^2 offsets from Marchi 2021 Tc;
  2. the change between numerics settings at the same resolution, predicted from ln(F_release / F_setting) with the F
     measured on that resolution, against the observed change of the K = 2 float32 runs.
This is an order-of-magnitude check, not a reference: Ghia's positions are only the tabulated points.

    .venv/Scripts/python.exe -m experiment.validation.reynolds_sensitivity
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation.cavity_analysis import DATA, write_csv  # noqa: E402

QUANTITIES = (("u_min", "y(u_min)"), ("v_max", "x(v_max)"), ("v_min", "x(v_min)"))
# effective_viscosity.csv variant -> run-id suffix of the K = 2 float32 runs with that numerics setting
SETTINGS = {"released": ("", r"ξ = 0.1, ε² = 0.01h²(发布)"), "xi = 0.001": ("_xi0p001", "ξ = 0.001, ε² = 0.01h²"),
            "xi = 0.001, eps^2 / 4": ("_xi0p001_eps0p0025", "ξ = 0.001, ε² = 0.0025h²")}


def read_rows(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def bracket(values: dict[int, float]) -> tuple[float, float]:
    """(secant 1000-3200, parabola slope at 1000) of a quantity against ln Re."""
    low, middle, high = (math.log(reynolds) for reynolds in (400, 1000, 3200))
    first = (values[1000] - values[400]) / (middle - low)
    second = (values[3200] - values[1000]) / (high - middle)
    local = (first * (high - middle) + second * (middle - low)) / (high - low)
    return second, local


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def main() -> int:
    ghia = read_rows(DATA / "ghia1982_extrema.csv")
    viscosity = read_rows(DATA / "effective_viscosity.csv")
    factors = {}                                   # (resolution, setting) -> median over the two test fields
    for row in viscosity:
        factors.setdefault((int(row["resolution"]), row["variant"]), []).append(float(row["nu_eff_over_nu_median"]))
    factors = {key: float(np.mean(values)) for key, values in factors.items()}
    release_factor = float(np.mean([value for (resolution, setting), value in factors.items() if setting == "released"]))
    extrema = {row["run"]: row for row in read_rows(DATA / "extrema.csv")}
    marchi = {row["quantity"]: float(row["Tc"]) for row in read_rows(DATA / "marchi2021_re1000.csv")
              if row["group"] == "extrema"}
    slopes = {}
    for value_name, position_name in QUANTITIES:
        positions = {int(row["reynolds"]): float(row["tabulated_position"]) for row in ghia if row["extremum"] == value_name}
        values = {int(row["reynolds"]): float(row["tabulated_value"]) for row in ghia if row["extremum"] == value_name}
        slopes[position_name] = (positions, bracket(positions), "position", value_name)
        slopes[value_name] = (values, bracket(values), "value", value_name)
    reference_key = {"y(u_min)": "y_at_u_min", "x(v_max)": "x_at_v_max", "x(v_min)": "x_at_v_min",
                     "u_min": "u_min", "v_max": "v_max", "v_min": "v_min"}

    # ---- 1. Re 1000 -> 1000 / F_release against the release runs' offsets ------------------------------------
    shift = math.log(1.0 / release_factor)
    rows, md_rows = [], []
    print(f"F_release = {release_factor:.4f} -> Re_eff = {1000.0 / release_factor:.0f}, ln(1/F) = {shift:.4f}")
    observed = extrema.get("n1000_k2_float32")
    for label, (series, (secant, local), kind, value_name) in slopes.items():
        predicted = sorted((secant * shift, local * shift), key=abs)
        reference = marchi[reference_key[label]]
        seen = (float(observed[f"{value_name}_position_error"]) if kind == "position"
                else float(observed[f"{value_name}_error"])) if observed else float("nan")
        relative = (f"{100 * predicted[0] / abs(reference) * math.copysign(1, reference):+.1f} … "
                    f"{100 * predicted[1] / abs(reference) * math.copysign(1, reference):+.1f} %") if kind == "value" else ""
        rows.append([label, kind, *(f"{series[reynolds]:.5f}" for reynolds in (400, 1000, 3200)), f"{secant:+.5f}",
                     f"{local:+.5f}", f"{release_factor:.4f}", f"{predicted[0]:+.5f}", f"{predicted[1]:+.5f}", relative,
                     f"{seen:+.5f}"])
        md_rows.append([label, f"{predicted[0]:+.4f} … {predicted[1]:+.4f}" + (f"({relative})" if relative else ""),
                        f"{seen:+.4f}" + (f"({100 * seen / abs(reference) * math.copysign(1, reference):+.1f} %)"
                                          if kind == "value" else "")])
    write_csv(DATA / "reynolds_sensitivity.csv",
              ["quantity", "kind", "ghia_re400", "ghia_re1000", "ghia_re3200", "d_dlnRe_secant_1000_3200",
               "d_dlnRe_parabola_at_1000", "nu_eff_over_nu", "predicted_change_low", "predicted_change_high",
               "predicted_change_relative", "observed_1000_minus_Tc"], rows,
              "Rough size of the change of the extrema for Re 1000 -> 1000 / F (F = nu_eff / nu of the released viscous\n"
              "operator, mean of the three resolutions in effective_viscosity.csv), from Ghia 1982's tabulated points at\n"
              "Re 400 / 1000 / 3200 (ghia1982_extrema.csv); derivative against ln Re bracketed by the 1000-3200 secant and\n"
              "the slope at 1000 of the parabola through the three. Observed: K = 2 float32 1000^2 (release) minus Marchi 2021 Tc.")
    sections = {"REYNOLDS_RELEASE": md_table(["量", f"Re 1000 → {1000.0 / release_factor:.0f} 的预期变化",
                                              "1000²(发布配置)的 SPH − T_c"], md_rows)}

    # ---- 2. changes between numerics settings, same resolution ----------------------------------------------------
    rows, md_rows = [], []
    for resolution in (250, 500, 1000):
        base = extrema.get(f"n{resolution}_k2_float32")
        release = factors.get((resolution, "released"))
        if base is None or release is None:
            continue
        for setting, (suffix, setting_label) in SETTINGS.items():
            if setting == "released":
                continue
            run = extrema.get(f"n{resolution}_k2_float32{suffix}")
            factor = factors.get((resolution, setting))
            if run is None or factor is None:
                continue
            change = math.log(release / factor)                     # ln Re_eff(setting) - ln Re_eff(release)
            cells = [f"{resolution}²", setting_label, f"{release:.3f} → {factor:.3f}"]
            for _, position_name in QUANTITIES:
                series, (secant, local), _, value_name = slopes[position_name]
                predicted = sorted((secant * change, local * change))
                seen = float(run[f"{value_name}_position_wall"]) - float(base[f"{value_name}_position_wall"])
                inside = predicted[0] - 1e-12 <= seen <= predicted[1] + 1e-12
                rows.append([resolution, setting, f"{release:.4f}", f"{factor:.4f}", f"{change:+.4f}", position_name,
                             f"{predicted[0]:+.5f}", f"{predicted[1]:+.5f}", f"{seen:+.5f}", int(inside)])
                cells.append(f"{seen:+.4f}({predicted[0]:+.4f} … {predicted[1]:+.4f})" + ("" if inside else " ✗"))
            md_rows.append(cells)
            print(" | ".join(cells))
    if rows:
        write_csv(DATA / "setting_changes.csv",
                  ["resolution", "setting", "F_release", "F_setting", "dlnRe_eff", "quantity", "predicted_low",
                   "predicted_high", "observed", "observed_within_prediction"], rows,
                  "Change of the extremum positions (wall-row frame) from the release numerics to another setting at the\n"
                  "same resolution (K = 2 float32 runs), against the change predicted from the measured effective viscosity:\n"
                  "Ghia 1982 d(position)/d(ln Re) bracket (reynolds_sensitivity.csv) times ln(F_release / F_setting).")
        sections["REYNOLDS_SETTINGS"] = md_table(
            ["分辨率", "设置", "ν_eff/ν:发布 → 该设置", "Δy(u_min) 实测(预测范围)", "Δx(v_max) 实测(预测范围)",
             "Δx(v_min) 实测(预测范围)"], md_rows)
    text = "\n\n".join(f"<!-- {name} -->\n{table}" for name, table in sections.items()) + "\n"
    (DATA / "tables_reynolds.md").write_text(text, encoding="utf-8")
    print("wrote", DATA / "reynolds_sensitivity.csv", DATA / "setting_changes.csv", DATA / "tables_reynolds.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
