"""cavity_reference.py - the Re = 1000 reference data of the cavity validation.

Marchi et al. 2021 (docs/validation/data/marchi2021_re1000.csv): the convergent Richardson-extrapolated
values Tc of Tables 19 (extrema, vortex centre), 20 (u at x = 1/2, 28 points) and 21 (v at y = 1/2,
28 points). Ghia et al. 1982 (docs/validation/data/ghia1982_re1000.csv): shown as a familiar reference
only; errors are taken against Marchi 2021.
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

_DATA = pathlib.Path(__file__).resolve().parents[2] / "docs" / "validation" / "data"
MARCHI_CSV = _DATA / "marchi2021_re1000.csv"
GHIA_CSV = _DATA / "ghia1982_re1000.csv"


def _rows(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def marchi2021() -> dict:
    """{'u': (y (28,), u (28,), Uc (28,)), 'v': (x (28,), v (28,), Uc (28,)),
        'extrema': {name: (value, Uc)}, 'rows': raw rows}. Coordinates are exact dyadic values."""
    rows = _rows(MARCHI_CSV)
    result = {"rows": rows, "extrema": {}}
    for line, group in (("u", "u_at_x_0.5"), ("v", "v_at_y_0.5")):
        selected = sorted((row for row in rows if row["group"] == group), key=lambda row: float(row["coordinate"]))
        result[line] = (np.array([float(row["coordinate"]) for row in selected]),
                        np.array([float(row["Tc"]) for row in selected]),
                        np.array([float(row["Uc"]) for row in selected]))
    for row in rows:
        if row["group"] == "extrema":
            result["extrema"][row["quantity"]] = (float(row["Tc"]), float(row["Uc"]))
    if result["u"][0].size != 28 or result["v"][0].size != 28 or len(result["extrema"]) != 9:
        raise ValueError(f"{MARCHI_CSV}: expected 28 + 28 profile points and 9 extrema")
    return result


def ghia1982() -> dict:
    """{'u': (y, u), 'v': (x, v)} including the wall points, Re = 1000."""
    rows = _rows(GHIA_CSV)
    result = {}
    for line, group in (("u", "u_at_x_0.5"), ("v", "v_at_y_0.5")):
        selected = sorted((row for row in rows if row["line"] == group), key=lambda row: float(row["coordinate"]))
        result[line] = (np.array([float(row["coordinate"]) for row in selected]),
                        np.array([float(row["value"]) for row in selected]))
    return result
