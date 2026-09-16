"""Compare two wall-shell variants of the same cavity from verifier dumps.

usage: wall_profile_compare.py <verify_out_dir> [--dx 0.005]

Expects the four dumps written by experiment/v5/_verify_cascade_force.py with --b-case:
legacy_1/2 = reference case (e.g. border 9), cascade_1/2 = candidate case (e.g. border 4).
Fluid particles only (material 0). Reports, for the noise pair (legacy_1 vs legacy_2) and
the test pairs (legacy_i vs cascade_i):
  - fluid density range and mean,
  - the lid-driven velocity profile u_x(y) on the vertical centreline (|x-xc|, |z-zc| < 2 dx),
    40 bins, and its max / rms difference,
  - near-wall velocity statistics (fluid within 2 h of any wall): mean |u| and rms difference
    between matched particles (KD-tree match by position, tolerance 0.5 dx).
"""
import argparse
import pathlib

import numpy as np
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--dx", type=float, default=0.005)
parser.add_argument("--h", type=float, default=0.02)
args = parser.parse_args()
out = pathlib.Path(args.out_dir)


def load(name):
    d = dict(np.load(out / f"{name}.npz"))
    slabs = int(d["slabs"])
    pos = np.concatenate([d[f"s{i}_position"] for i in range(slabs)]).astype(np.float64)
    vel = np.concatenate([d[f"s{i}_velocity"] for i in range(slabs)]).astype(np.float64)
    rho = np.concatenate([d[f"s{i}_density"] for i in range(slabs)]).astype(np.float64)
    mat = np.concatenate([d[f"s{i}_material"] for i in range(slabs)])
    fluid = mat == 0
    return pos[fluid], vel[fluid], rho[fluid]


runs = {name: load(name) for name in ("legacy_1", "legacy_2", "cascade_1", "cascade_2")}
pos0 = runs["legacy_1"][0]
lo, hi = pos0.min(axis=0), pos0.max(axis=0)
centre = 0.5 * (lo + hi)
print(f"fluid box x[{lo[0]:.4f},{hi[0]:.4f}] y[{lo[1]:.4f},{hi[1]:.4f}] z[{lo[2]:.4f},{hi[2]:.4f}]")
for name, (pos, vel, rho) in runs.items():
    print(f"{name:10s} fluid n={pos.shape[0]:,} rho [{rho.min():.2f}, {rho.max():.2f}] mean {rho.mean():.3f}  "
          f"|u| max {np.linalg.norm(vel, axis=1).max():.4f}")


def profile(pos, vel):
    sel = (np.abs(pos[:, 0] - centre[0]) < 2 * args.dx) & (np.abs(pos[:, 2] - centre[2]) < 2 * args.dx)
    y = pos[sel, 1]; ux = vel[sel, 0]
    edges = np.linspace(lo[1], hi[1], 41)
    idx = np.clip(np.digitize(y, edges) - 1, 0, 39)
    prof = np.array([ux[idx == k].mean() if (idx == k).any() else np.nan for k in range(40)])
    return 0.5 * (edges[:-1] + edges[1:]), prof


def near_wall(pos):
    d = np.minimum.reduce([pos[:, 0] - lo[0], hi[0] - pos[:, 0], pos[:, 1] - lo[1], hi[1] - pos[:, 1],
                           pos[:, 2] - lo[2], hi[2] - pos[:, 2]])
    return d < 2 * args.h


profiles = {name: profile(r[0], r[1]) for name, r in runs.items()}
print("\ncentreline u_x(y) profile differences (40 bins):")
for a, b, label in (("legacy_1", "legacy_2", "noise A vs A"), ("cascade_1", "cascade_2", "noise B vs B"),
                    ("legacy_1", "cascade_1", "TEST A vs B"), ("legacy_2", "cascade_2", "TEST A vs B (2nd)")):
    pa, pb = profiles[a][1], profiles[b][1]
    diff = np.abs(pa - pb); ok = ~np.isnan(diff)
    print(f"  {label:20s} max {diff[ok].max():.5f} m/s  rms {np.sqrt(np.mean(diff[ok] ** 2)):.5f}  (profile |u_x| max {np.nanmax(np.abs(pa)):.4f})")

print("\nnear-wall fluid (within 2h of a wall), matched by position (tol 0.5 dx):")
for a, b, label in (("legacy_1", "legacy_2", "noise A vs A"), ("cascade_1", "cascade_2", "noise B vs B"),
                    ("legacy_1", "cascade_1", "TEST A vs B"), ("legacy_2", "cascade_2", "TEST A vs B (2nd)")):
    pa, va, ra = runs[a]; pb, vb, rb = runs[b]
    sel = near_wall(pa)
    tree = cKDTree(pb)
    dist, idx = tree.query(pa[sel], k=1)
    ok = dist <= 0.5 * args.dx
    dv = np.linalg.norm(va[sel][ok] - vb[idx[ok]], axis=1)
    dr = np.abs(ra[sel][ok] - rb[idx[ok]])
    print(f"  {label:20s} n={sel.sum():,} matched {100 * ok.mean():.2f}%  mean|u| {np.linalg.norm(va[sel], axis=1).mean():.5f}  "
          f"|du| rms {np.sqrt(np.mean(dv ** 2)):.2e} max {dv.max():.2e}   |drho| rms {np.sqrt(np.mean(dr ** 2)):.3f} max {dr.max():.3f}")
y, p = profiles["legacy_1"]
print("\ncentreline u_x(y), legacy_1 vs cascade_1 (y, uA, uB):")
for k in range(0, 40, 4):
    print(f"  y={y[k]:.3f}  {p[k]:+.4f}  {profiles['cascade_1'][1][k]:+.4f}")
