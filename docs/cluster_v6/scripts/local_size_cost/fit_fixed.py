"""Fixed per-step cost check (user 2026-10-26, late): T = a N + b fitted separately to the local cubes (4M..76M) and to the
three stretched boxes, for the step and for each sweep kernel (correction_density_interior, force_deep_interior); the table
N, T, a N, T - a N and the two kernels' residuals. Reads runs_table.json (analyze_runs.py); writes fit_fixed.md."""
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d")
rows = json.load((ROOT / "runs_table.json").open(encoding="utf-8"))
CUBES = [("cube4m", "cube 4M (160^3)"), ("8m_base", "cube 8M (200^3)"), ("cube16m", "cube 16M (252^3)"), ("cube32m", "cube 32M (320^3)"),
         ("64m_base", "cube 64M (400^3)"), ("cube76m", "cube 76M (424^3)")]
BOXES = [("box_k2", "box 2 x 8M (192^2 x 384)"), ("box_k4", "box 4 x 8M (192^2 x 768)"), ("box_k8", "box 8 x 8M (192^2 x 1536)")]
KERNELS = (("correction_density_interior", "corr+dens"), ("force_deep_interior", "force"))


def mean(item, key):
    vals = [r[key] for r in rows if r["item"] == item and r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def kernel_us(item, kernel):
    vals = [r["kernels_us"].get(kernel) for r in rows if r["item"] == item and r.get("kernels_us")]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def collect(items):
    out = []
    for key, label in items:
        if mean(key, "step_ms") is None:
            continue
        out.append({"key": key, "label": label, "n": mean(key, "n_total"), "T": mean(key, "step_ms"),
                    "runs": sum(1 for r in rows if r["item"] == key),
                    **{short: kernel_us(key, kernel) / 1e3 for kernel, short in KERNELS}})
    return out


def fit(points, field):
    x = np.array([p["n"] for p in points], dtype=float)
    y = np.array([p[field] for p in points], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b


def section(name, points):
    lines = [f"### {name}: {len(points)} points, T = a N + b (ms; a in ns per particle)"]
    if len(points) < 2:
        return lines + ["(not enough points)"]
    fits = {field: fit(points, field) for field in ("T", "corr+dens", "force")}
    for field, (a, b) in fits.items():
        res = [p[field] - (a * p["n"] + b) for p in points]
        lines.append(f"- {field}: a = {a * 1e6:.4f} ns, b = {b:.3f} ms; residuals (ms) " + ", ".join(f"{r:+.3f}" for r in res) +
                     f"; rms {np.sqrt(np.mean(np.square(res))):.3f} ms")
    a_T, b_T = fits["T"]
    lines += ["", "| case | runs | N | T (ms) | a N (ms) | T - a N (ms) | corr+dens: T_k - a_k N (ms) | force: T_k - a_k N (ms) | "
              "sum of the two kernels (ms) | other (ms) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        rk = {short: p[short] - fits[short][0] * p["n"] for _, short in KERNELS}
        lines.append(f"| {p['label']} | {p['runs']} | {p['n']:,.0f} | {p['T']:.3f} | {a_T * p['n']:.3f} | {p['T'] - a_T * p['n']:+.3f} | "
                     f"{rk['corr+dens']:+.3f} | {rk['force']:+.3f} | {p['corr+dens'] + p['force']:.3f} | {p['T'] - p['corr+dens'] - p['force']:.3f} |")
    # the residual T - a N against N: is it constant? spread and trend
    res = [p["T"] - a_T * p["n"] for p in points]
    lines.append(f"\nT - a N: mean {statistics.mean(res):.3f} ms, min {min(res):.3f}, max {max(res):.3f}, spread {max(res) - min(res):.3f} ms "
                 f"({(max(res) - min(res)) / statistics.mean(res) * 100 if statistics.mean(res) else float('nan'):.0f} % of the mean)")
    # the same without the smallest / largest point, to see the leverage
    if len(points) >= 4:
        for drop in (0, len(points) - 1):
            sub = [p for i, p in enumerate(points) if i != drop]
            a2, b2 = fit(sub, "T")
            lines.append(f"- without {points[drop]['label']}: a = {a2 * 1e6:.4f} ns, b = {b2:.3f} ms")
    return lines


cubes, boxes = collect(CUBES), collect(BOXES)
lines = section("cubes", cubes) + [""] + section("boxes", boxes)
if len(cubes) >= 3:
    # alternative: T = a N + c N^(2/3) (a surface term) for the cubes, for comparison with the constant
    x = np.array([p["n"] for p in cubes], dtype=float)
    y = np.array([p["T"] for p in cubes], dtype=float)
    A = np.vstack([x, x ** (2.0 / 3.0)]).T
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [a, c]
    lines += ["", f"cubes, alternative T = a N + c N^(2/3): a = {a * 1e6:.4f} ns, c = {c:.4e} ms; residuals (ms) " +
              ", ".join(f"{r:+.3f}" for r in res) + f"; rms {np.sqrt(np.mean(res ** 2)):.3f} ms"]
    A3 = np.vstack([x, x ** (2.0 / 3.0), np.ones_like(x)]).T
    (a3, c3, b3), *_ = np.linalg.lstsq(A3, y, rcond=None)
    res3 = y - A3 @ [a3, c3, b3]
    lines += [f"cubes, T = a N + c N^(2/3) + b: a = {a3 * 1e6:.4f} ns, c = {c3:.4e} ms, b = {b3:.3f} ms; residuals (ms) " +
              ", ".join(f"{r:+.3f}" for r in res3) + f"; rms {np.sqrt(np.mean(res3 ** 2)):.3f} ms"]
text = "\n".join(lines)
(ROOT / "fit_fixed.md").write_text(text, encoding="utf-8")
print(text)
