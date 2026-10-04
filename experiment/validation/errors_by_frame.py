"""errors_by_frame.py - L2 errors and primary-vortex deviations against Marchi et al. 2021 in two unit-square
definitions: the innermost boundary rows (the analysis' wall-row frame: the bottom-left innermost boundary particle is
(0, 0), the top-right one (1, 1), positions scaled by 1 / (1 + 2 dx)) and the fluid lattice box (X = x + 1/2). Reads the
analysis cache of cavity_analysis.py and writes docs/validation/fields/errors_by_frame.md.

    .venv/Scripts/python.exe -m experiment.validation.errors_by_frame
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference  # noqa: E402

RUNS = ["n250_k2_float32", "n500_k2_float32", "n1000_k2_float32",
        "n250_k2_float32_xi0p001", "n500_k2_float32_xi0p001", "n1000_k2_float32_xi0p001",
        "n250_k2_float32_xi0p001_eps0p0025", "n500_k2_float32_xi0p001_eps0p0025", "n1000_k2_float32_xi0p001_eps0p0025"]
SETTING_LABELS = {"": "发布 ξ=0.1", "_xi0p001": "ξ=0.001", "_xi0p001_eps0p0025": "ξ+ε 主系列"}


def main() -> int:
    reference = cavity_reference.marchi2021()
    extrema = reference["extrema"]
    rms = {line: float(np.sqrt(np.mean(reference[line][1] ** 2))) for line in "uv"}
    x0, y0, p0 = extrema["x_at_psi_min"][0], extrema["y_at_psi_min"][0], extrema["psi_min"][0]
    cache = {item["id"]: item for item in
             np.load(_REPO_ROOT / "logs" / "validation" / "cavity_re1000" / "analysis_cache.npy", allow_pickle=True)}
    lines = ["# Errors against Marchi et al. 2021 in two unit-square definitions", "",
             "Time averages t* 80-100, K = 2 float32. Innermost boundary rows: the bottom-left innermost boundary particle "
             "is (0, 0), the top-right one (1, 1), positions scaled by 1 / (1 + 2 dx) (the analysis' wall-row frame). "
             "Fluid box: X = x + 1/2. L2 = rms error at the 28 + 28 Table 20 / 21 points divided by rms(Tc); vortex = "
             "minimum of psi integrated from the bottom wall of the frame; positions in units of the frame's L. Generated "
             "by `experiment/validation/errors_by_frame.py` from the analysis cache.", "",
             "| 设置 | 分辨率 | 框架 | L2 u | L2 v | 主涡中心偏差 (Δx, Δy) | 距离 | ψ_min 偏差 | y(u_min) 偏差 | "
             "x(v_max) 偏差 | x(v_min) 偏差 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run_id in RUNS:
        item = cache.get(run_id)
        if item is None:
            continue
        resolution = run_id.split("_")[0][1:]
        suffix = run_id[len(f"n{resolution}_k2_float32"):]
        stream = item.get("stream") or {}
        for frame, label in (("wall", "最内侧边界行"), ("fluid", "流体框")):
            error_u = item["errors"][(frame, "mls", "u")]["L2"] / rms["u"]
            error_v = item["errors"][(frame, "mls", "v")]["L2"] / rms["v"]
            x, y, psi = stream.get(f"x_{frame}"), stream.get(f"y_{frame}"), stream.get(f"psi_min_{frame}")
            positions = [item["extrema"][name][f"position_{frame}"] for name in ("u_min", "v_max", "v_min")]
            offsets = [positions[0] - extrema["y_at_u_min"][0], positions[1] - extrema["x_at_v_max"][0],
                       positions[2] - extrema["x_at_v_min"][0]]
            vortex = (f"({x - x0:+.4f}, {y - y0:+.4f}) | {np.hypot(x - x0, y - y0):.4f} | "
                      f"{100 * (abs(psi) - abs(p0)) / abs(p0):+.1f} %") if x is not None else "– | – | –"
            lines.append(f"| {SETTING_LABELS[suffix]} | {resolution}² | {label} | {100 * error_u:.1f} % | "
                         f"{100 * error_v:.1f} % | {vortex} | " + " | ".join(f"{offset:+.4f}" for offset in offsets) + " |")
    path = _REPO_ROOT / "docs" / "validation" / "fields" / "errors_by_frame.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
