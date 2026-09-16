"""Per-particle cost of the band (phase C) kernels vs the interior (phase B) kernels from chain-bench
--anatomy logs, for fake bands (V5_FAKE_BAND_TEST, K=1) and real seam bands (K=2).

usage: band_cost_table.py <spec.json>
spec: {"runs": [{"label", "log", "frames": [2000, 3000], "sims": [0], "n_total": <own particles per sim>,
                "band": [n2, n3, n4]}, ...]}
Per run (mean over the listed frames and sims): kernel time / particle count for
  correction_interior vs correction_boundary (band 2), density_deep_interior vs density_boundary
  (band 3, copy excluded), force_deep_interior vs force_boundary (band 4).
Then a linear fit time = fixed + slope * particles over the runs that share a tag ("2d-fake",
"2d-real") to separate the per-launch fixed cost from the per-particle cost."""
import json
import re
import sys

import numpy as np

spec = json.load(open(sys.argv[1]))
KERNELS = (("correction", "correction_interior", "correction_boundary", 0),
           ("density", "density_deep_interior", "density_boundary", 1),
           ("force", "force_deep_interior", "force_boundary", 2))


def anatomy(path, frames, sims):
    rows = []
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    for line in handle:
        m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
        if m and int(m.group(1)) in frames and int(m.group(2)) in sims:
            rows.append({k: float(v) for k, v in (p.split("=") for p in m.group(3).split())})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in rows[0]} if rows else {}


print(f"{'run':28s} {'kernel':11s} {'interior us':>11s} {'ns/p':>6s} {'band us':>8s} {'band n':>9s} {'ns/p':>6s} {'ratio':>6s}")
fits = {}
for run in spec["runs"]:
    a = anatomy(run["log"], run["frames"], run.get("sims", [0]))
    if not a:
        print(f"{run['label']:28s} (no anatomy lines)"); continue
    for name, ikey, bkey, bidx in KERNELS:
        if bkey not in a and name == "force" and "force" in a:
            bkey = "force"          # phase-C logs: force_us = force_boundary (+ barrier)
        if ikey not in a or bkey not in a or a[bkey] <= 0:
            continue
        n_band = run["band"][bidx]
        n_int = run["n_total"] - n_band
        i_ns = 1e3 * a[ikey] / n_int
        b_ns = 1e3 * a[bkey] / n_band
        print(f"{run['label']:28s} {name:11s} {a[ikey]:11.0f} {i_ns:6.3f} {a[bkey]:8.0f} {n_band:9,d} {b_ns:6.3f} {b_ns / i_ns:6.2f}")
        fits.setdefault((run.get("tag", run["label"]), name), []).append((n_band, a[bkey], i_ns))
    if "density_copy" in a:
        print(f"{'':28s} {'(copy)':11s} {'':11s} {'':6s} {a['density_copy']:8.0f}")
print("\nlinear fit of band-kernel time vs band particles (per tag, >= 2 sizes):")
for (tag, name), pts in sorted(fits.items()):
    if len(pts) < 2:
        continue
    n = np.array([p[0] for p in pts], dtype=float); t = np.array([p[1] for p in pts])
    slope, fixed = np.polyfit(n, t, 1)
    interior = float(np.mean([p[2] for p in pts]))
    print(f"  {tag:10s} {name:11s} fixed {fixed:6.0f} us + {1e3 * slope:6.3f} ns/particle   (interior {interior:.3f} ns/p -> per-particle ratio {1e3 * slope / interior:.2f})")
