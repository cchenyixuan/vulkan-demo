"""Fixed-term table (user 2026-10-27, item 2/3): T - a N with the box-fit slopes (a from the three stretched boxes, 7.14) for the
8M / 64M cubes, the 2 x 8M box and the flat box cavity3d_n384_x96 (96 x 384 x 384 fluid, 384^2 cross-section); the same for the
two sweep kernels with their own box-fit slopes. Reads runs_table.json (analyze_runs.py); writes fixed_terms.md / .json."""
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d")
rows = json.load((ROOT / "runs_table.json").open(encoding="utf-8"))
KERNELS = (("correction_density_interior", "corr+dens"), ("force_deep_interior", "force"))
# item -> (label, x length in fluid cells, cross-section side in fluid cells)
ITEMS = [("8m_base", "cube 8M (200^3)", 200, 200), ("64m_base", "cube 64M (400^3)", 400, 400),
         ("box_k2", "box 2 x 8M (384 x 192^2)", 384, 192), ("box_k4", "box 4 x 8M (768 x 192^2)", 768, 192),
         ("box_k8", "box 8 x 8M (1536 x 192^2)", 1536, 192), ("flatbox", "flat box (96 x 384^2)", 96, 384)]
BOX_FIT = ("box_k2", "box_k4", "box_k8")


def runs(item):
    return [r for r in rows if r["item"] == item]


def point(item):
    rr = runs(item)
    p = {"item": item, "runs": len(rr), "n": rr[0]["n_total"], "T": statistics.mean(r["step_ms"] for r in rr),
         "T_runs": [r["step_ms"] for r in rr]}
    for kernel, short in KERNELS:
        p[short] = statistics.mean(r["kernels_us"][kernel] for r in rr) / 1e3
    return p


def fit(points, field):
    x = np.array([p["n"] for p in points], dtype=float)
    y = np.array([p[field] for p in points], dtype=float)
    (a, b), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    return float(a), float(b)


boxes = [point(k) for k in BOX_FIT]
slopes = {field: fit(boxes, field) for field in ("T", "corr+dens", "force")}
out = {"slopes_box_fit": {f: {"a_ns": a * 1e6, "b_ms": b} for f, (a, b) in slopes.items()}, "rows": []}
lines = ["### fixed term T - a N with the box-fit slopes (three stretched boxes, 7.14)", "",
         "slopes: " + "; ".join(f"{f}: a = {a * 1e6:.3f} ns, b = {b:.3f} ms" for f, (a, b) in slopes.items()), "",
         "| case | runs | N | x cells | section | T (ms) | a N (ms) | T - a N (ms) | corr+dens: T_k - a_k N (ms) | force: T_k - a_k N (ms) | "
         "two kernels (ms) | other: (T - a N) - two kernels (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for item, label, x_cells, side in ITEMS:
    if not runs(item):
        continue
    p = point(item)
    aT = slopes["T"][0] * p["n"]
    fixed = p["T"] - aT
    k = {short: p[short] - slopes[short][0] * p["n"] for _, short in KERNELS}
    rec = {"item": item, "label": label, "runs": p["runs"], "N": p["n"], "x_cells": x_cells, "section_cells": side, "T_ms": p["T"],
           "T_runs_ms": p["T_runs"], "aN_ms": aT, "fixed_ms": fixed, "fixed_corr_dens_ms": k["corr+dens"], "fixed_force_ms": k["force"],
           "fixed_other_ms": fixed - k["corr+dens"] - k["force"], "corr_dens_ms": p["corr+dens"], "force_ms": p["force"]}
    out["rows"].append(rec)
    lines.append(f"| {label} | {p['runs']} | {p['n']:,} | {x_cells} | {side}^2 | {p['T']:.3f} | {aT:.3f} | **{fixed:+.3f}** | "
                 f"{k['corr+dens']:+.3f} | {k['force']:+.3f} | {k['corr+dens'] + k['force']:+.3f} | {fixed - k['corr+dens'] - k['force']:+.3f} |")
# the flat box against the 2 x 8M box: same fluid count (14,155,776), cross-section x 4, length / 4
fb, b2 = next(r for r in out["rows"] if r["item"] == "flatbox"), next(r for r in out["rows"] if r["item"] == "box_k2")
lines += ["", f"flat box / 2 x 8M box: fixed term {fb['fixed_ms']:.3f} / {b2['fixed_ms']:.3f} ms = {fb['fixed_ms'] / b2['fixed_ms']:.3f} "
          f"(cross-section ratio 4, x-length ratio 1/4, fluid count equal 14,155,776; total N {fb['N']:,} vs {b2['N']:,}); "
          f"per x cell: {fb['fixed_ms'] / fb['x_cells'] * 1e3:.1f} vs {b2['fixed_ms'] / b2['x_cells'] * 1e3:.1f} us; "
          f"per section cell^2: {fb['fixed_ms'] / fb['section_cells'] ** 2 * 1e6:.1f} vs {b2['fixed_ms'] / b2['section_cells'] ** 2 * 1e6:.1f} ns"]
text = "\n".join(lines)
(ROOT / "fixed_terms.md").write_text(text, encoding="utf-8")
json.dump(out, (ROOT / "fixed_terms.json").open("w", encoding="utf-8"), indent=1)
print(text)
