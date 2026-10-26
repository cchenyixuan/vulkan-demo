"""Discriminating table (user 2026-10-26 evening): the three stretched boxes beside the cubes measured locally, with the two
predictions: the interface model c = 5.61 + 8.64 f_w (the cluster fit, ns per particle of the total; f_w = walls incl. lid /
total) and the equal-particle-count interpolation along the local cube curve (linear in log N between the measured cubes).
Reads runs_table.json (analyze_runs.py); writes box_table.md next to it."""
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d")
rows = json.load((ROOT / "runs_table.json").open(encoding="utf-8"))
ITEMS = [("8m_base", "cube 8M (200^3)"), ("cube16m", "cube 16M (252^3)"), ("cube32m", "cube 32M (320^3)"), ("64m_base", "cube 64M (400^3)"),
         ("box_k2", "box 2 x 8M (192^2 x 384)"), ("box_k4", "box 4 x 8M (192^2 x 768)"), ("box_k8", "box 8 x 8M (192^2 x 1536)")]
C_F, C_W = 5.61, 14.25          # the cluster fit (report 7.12, item 1, model A)


def mean(item, key):
    vals = [r[key] for r in rows if r["item"] == item and r.get(key) is not None]
    return statistics.mean(vals) if vals else float("nan")


def runs(item):
    return sum(1 for r in rows if r["item"] == item)


data = {}
for item, label in ITEMS:
    if runs(item) == 0:
        continue
    r0 = next(r for r in rows if r["item"] == item)
    data[item] = {"label": label, "runs": runs(item), "n_total": r0["n_total"], "n_fluid": r0["n_fluid"], "n_wall": r0["n_wall"],
                  "f_w": r0["n_wall"] / r0["n_total"], "ns_total": mean(item, "ns_total"), "ns_fluid": mean(item, "ns_fluid"),
                  "cd": mean(item, "correction_density_ns"), "force": mean(item, "force_ns"), "sm": mean(item, "sm_p50_MHz"),
                  "watt": mean(item, "power_p50_W"), "step_ms": mean(item, "step_ms"),
                  "spread": (max(r["ns_total"] for r in rows if r["item"] == item) - min(r["ns_total"] for r in rows if r["item"] == item))}
cubes = sorted((d for k, d in data.items() if k.startswith("cube") or k in ("8m_base", "64m_base")), key=lambda d: d["n_total"])


def cube_interp(n_total: float) -> float:
    xs = [math.log(d["n_total"]) for d in cubes]
    ys = [d["ns_total"] for d in cubes]
    x = math.log(n_total)
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


# local-anchored interface model: least squares a + b f_w on the local cubes (for the text)
if len(cubes) >= 2:
    fw = [d["f_w"] for d in cubes]
    y = [d["ns_total"] for d in cubes]
    mx, my = statistics.mean(fw), statistics.mean(y)
    b_local = sum((a - mx) * (c - my) for a, c in zip(fw, y)) / sum((a - mx) ** 2 for a in fw)
    a_local = my - b_local * mx
else:
    a_local = b_local = float("nan")

scale = statistics.mean(d["ns_total"] / (C_F + (C_W - C_F) * d["f_w"]) for d in cubes) if cubes else float("nan")
lines = [f"interface model (cluster fit): c = {C_F} + {C_W - C_F:.2f} f_w ns; local cubes sit at {scale:.4f} x that model on average "
         f"(platform offset); local-anchored interface model from the local cubes: c = {a_local:.3f} + {b_local:.2f} f_w ns",
         "",
         "| case | runs | N_total | f_w (walls incl. lid) | ns/particle (total) | ns/particle (fluid) | corr+dens (ns/total) | force (ns/total) | "
         "SM p50 (MHz) | W | pred. interface 5.61 + 8.64 f_w | x local scale | pred. cube interp. (same N) | measured - interface | measured - cube interp. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for item, label in ITEMS:
    if item not in data:
        continue
    d = data[item]
    p_int = C_F + (C_W - C_F) * d["f_w"]
    p_cube = cube_interp(d["n_total"])
    is_cube = item.startswith("cube") or item in ("8m_base", "64m_base")
    lines.append(f"| {label} | {d['runs']} | {d['n_total']:,} | {d['f_w']:.4f} | {d['ns_total']:.4f} (±{d['spread'] / 2:.4f}) | {d['ns_fluid']:.4f} | {d['cd']:.4f} | "
                 f"{d['force']:.4f} | {d['sm']:.0f} | {d['watt']:.1f} | {p_int:.4f} | {p_int * scale:.4f} | {p_cube:.4f}{' (anchor)' if is_cube else ''} | "
                 f"{(d['ns_total'] - p_int * scale) / d['ns_total'] * 100:+.2f} % (scaled) | {(d['ns_total'] - p_cube) / d['ns_total'] * 100:+.2f} % |")
lines += ["", "box vs cube of the same N (local): " + "; ".join(
    f"{data[b]['label']}: measured {data[b]['ns_total']:.4f}, cube interp. {cube_interp(data[b]['n_total']):.4f}, interface (scaled) "
    f"{(C_F + (C_W - C_F) * data[b]['f_w']) * scale:.4f}" for b in ("box_k2", "box_k4", "box_k8") if b in data)]
text = "\n".join(lines)
(ROOT / "box_table.md").write_text(text, encoding="utf-8")
print(text)
