"""Paper figures that combine 2-D and 3-D on one axis:
  eta_fixed_n_ksweep_2d_3d.png : fixed-N K sweep, 2-D 64M/32M (probe36) + 3-D 64M/32M stretched (probe37)
  eta_weak_2d_3d.png           : weak scaling, 2-D five loads per GPU (probe35) + 3-D 8M and 4M per GPU (probe37)
usage: plot_paper_figs.py <docs/n56_scaling> <out_dir>"""
import json, pathlib, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

docs, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
ks2d = json.load(open(docs / "ksweep_table.json"))
m3d = json.load(open(docs / "matrix_3d_table.json"))["table"]

# ---------------- fixed-N K sweep, 2-D + 3-D
fig, ax = plt.subplots(figsize=(7.2, 4.8))
series = [("2-D 64M (8.05M/GPU at K=8)", "#2a78d6", "o-", [(K, ks2d[f"64m_k{K}"]) for K in (2, 4, 8) if f"64m_k{K}" in ks2d]),
          ("2-D 32M (4.06M/GPU at K=8)", "#eb6834", "s-", [(K, ks2d[f"32m_k{K}"]) for K in (2, 4, 8) if f"32m_k{K}" in ks2d]),
          ("3-D 64M stretched (8.8M/GPU at K=8)", "#2f9e5c", "^-", [(K, m3d[f"strong64m_k{K}"]) for K in (2, 4, 8) if f"strong64m_k{K}" in m3d]),
          ("3-D 32M stretched (4.5M/GPU at K=8)", "#8e44ad", "v-", [(K, m3d[f"strong32m_k{K}"]) for K in (2, 4, 8) if f"strong32m_k{K}" in m3d])]
for label, color, fmt, rows in series:
    ks = [1] + [K for K, _ in rows]; etas = [100.0] + [100 * r["eta"] for _, r in rows]; errs = [0] + [100 * r["std"] for _, r in rows]
    ax.errorbar(ks, etas, yerr=errs, fmt=fmt, color=color, capsize=4, linewidth=1.5, markersize=6, label=label)
    for K, r in rows:
        ax.annotate(f"{100 * r['eta']:.1f}", (K, 100 * r["eta"]), textcoords="offset points", xytext=(0, 6 if "64M" in label else -13), ha="center", fontsize=7.5, color=color)
if "64m_k2x" in ks2d:
    r = ks2d["64m_k2x"]
    ax.errorbar([2.12], [100 * r["eta"]], yerr=[100 * r["std"]], fmt="D", color="#2a78d6", mfc="white", capsize=3, markersize=6, label="2-D 64M K=2, cross-NUMA pair")
ax.axhline(100, color="#888", linestyle=":", linewidth=1)
ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_ylim(62, 101.5)
ax.annotate("2-D 32M K=8: 4.06M/GPU, transport\nno longer hidden (t_transport ≈ 0.75 T_B)", (8, 67.4), textcoords="offset points", xytext=(-150, 6), fontsize=7, color="#eb6834")
ax.set_xlabel("GPUs K (total N fixed)"); ax.set_ylabel("strong-scaling efficiency η (%)"); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("Fixed-N strong scaling on one node: 2-D and 3-D, 3 trials, references on the participating GPUs", fontsize=8.5)
fig.tight_layout(); fig.savefig(out_dir / "eta_fixed_n_ksweep_2d_3d.png", dpi=150); print("saved eta_fixed_n_ksweep_2d_3d.png")

# ---------------- weak scaling, 2-D + 3-D
# 2-D families: probe35 (docs/n56_scaling/eta_weak_vs_k.png / memory of 2026-09-16), eta_mean +- std at K = 2, 4, 8
weak2d = {"2-D 32M/GPU": ([98.3, 97.3, 95.9], [0.3, 0.0, 0.4], "#4a3aa7"), "2-D 16M/GPU": ([98.1, 97.2, 96.1], [0.2, 0.1, 0.3], "#2a78d6"),
          "2-D 8M/GPU": ([95.7, 94.1, 92.2], [0.3, 0.4, 0.4], "#2f9e5c"), "2-D 4M/GPU": ([88.3, 79.1, 81.6], [8.8, 9.1, 5.4], "#eb6834"),
          "2-D 2M/GPU": ([63.5, 62.4, 53.3], [22.2, 10.8, 4.9], "#c0392b")}
weak3d = {"3-D 8M/GPU": ([100 * m3d[f"weak8_k{K}"]["eta"] for K in (2, 4, 8)], [100 * m3d[f"weak8_k{K}"]["std"] for K in (2, 4, 8)], "#1a6b3a"),
          "3-D 4M/GPU": ([100 * m3d[f"weak4_k{K}"]["eta"] for K in (2, 4, 8)], [100 * m3d[f"weak4_k{K}"]["std"] for K in (2, 4, 8)], "#b35a12")}
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for label, (etas, errs, color) in weak2d.items():
    ax.errorbar([1, 2, 4, 8], [100.0] + etas, yerr=[0] + errs, fmt="o--", color=color, capsize=3, linewidth=1.2, markersize=5, label=label, alpha=0.9)
for label, (etas, errs, color) in weak3d.items():
    ax.errorbar([1, 2, 4, 8], [100.0] + etas, yerr=[0] + errs, fmt="^-", color=color, capsize=3, linewidth=1.8, markersize=7, label=label)
ax.axhline(100, color="#888", linestyle=":", linewidth=1)
ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_ylim(40, 102)
ax.set_xlabel("GPUs K (particles per GPU fixed)"); ax.set_ylabel("weak-scaling efficiency η (%)"); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="lower left", ncol=2); ax.set_title("Weak scaling on one node: 2-D (five loads) and 3-D (two loads), 3 trials, 8-GPU simultaneous K=1 references", fontsize=8.5)
fig.tight_layout(); fig.savefig(out_dir / "eta_weak_2d_3d.png", dpi=150); print("saved eta_weak_2d_3d.png")
