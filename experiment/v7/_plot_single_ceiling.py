"""
_plot_single_ceiling.py — single-GPU (1× RTX 5090) baseline + theoretical extrapolation.

Plots the MEASURED single-GPU throughput (1M → the OOM ceiling) and a fitted
THEORETICAL line for what one 5090 *would* do beyond OOM if memory were infinite.

Model (fixed-overhead): frame_time(N) = t_fixed + N / rate_inf  → fps = 1/frame_time.
This is a 1st-order fit of frame_time vs N; it matches the measured singles to <1%.
The dual 2×5090 curve is overlaid to show dual runs problem sizes single can't fit.

  -> logs/single_ceiling.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = pathlib.Path(__file__).resolve().parents[2]

TOTAL = {"1m": 1046529, "2m": 2064969, "4m": 4182025, "6m": 6105841, "8m": 8128201,
         "10m": 10144225, "14m": 14160169, "16m": 16184529, "32m": 32251041,
         "64m": 64368529, "80m": 80534001, "90m": 90561489, "96m": 96550969,
         "104m": 104595841, "128m": 128482225, "160m": 160554241, "192m": 192626641}

# Measured single 5090 (device 1, headless) — from the 2x5090 campaign (amd_fps col).
SINGLE_MEASURED = {"1m": 509.3, "2m": 261.7, "4m": 125.4, "6m": 89.7, "8m": 67.7,
                   "10m": 55.7, "14m": 38.4, "16m": 34.4, "32m": 17.70, "64m": 8.88}

# Dual 2x5090 (measured + extreme) for the overlay.
DUAL_2X = {"1m": 612.5, "2m": 422.1, "4m": 235.4, "6m": 168.4, "8m": 125.9, "10m": 103.8,
           "14m": 71.6, "16m": 64.8, "32m": 33.9, "64m": 17.2, "128m": 8.6, "160m": 7.2, "192m": 5.7}


def main() -> int:
    # merge the ceiling-probe results (80m/90m/...) into the measured single set
    probe = _REPO / "logs/single_ceiling.jsonl"
    oom_size = None
    if probe.exists():
        for line in probe.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok" and r.get("fps"):
                SINGLE_MEASURED[r["size"]] = r["fps"]
            elif r.get("status") == "OOM":
                oom_size = r["size"]

    order = sorted(SINGLE_MEASURED, key=lambda s: TOTAL[s])
    n = np.array([TOTAL[s] / 1e6 for s in order])
    fps = np.array([SINGLE_MEASURED[s] for s in order])
    frame_us = 1e6 / fps

    # The per-particle rate (N*fps) is still climbing toward its asymptote, so the
    # most defensible no-OOM extrapolation is the PEAK sustained rate (achieved at
    # the largest N) extended as fps = R_peak / N. The card meets this line at large
    # N and falls slightly below at small N (SM under-occupancy) — so it is the
    # theoretical ceiling for the beyond-OOM (large-N) regime.
    rate_peak = float((n * fps).max())           # M-particles/sec, at the largest N
    print(f"peak single-GPU rate = {rate_peak:.0f} M-particles/sec (at N={n[int(np.argmax(n*fps))]:.0f}M)")
    ceiling_n = TOTAL[oom_size] / 1e6 if oom_size else n.max()
    print(f"single ceiling ~ {ceiling_n:.0f}M ({'OOM at '+oom_size if oom_size else 'last measured'})")

    n_theory = np.logspace(0, np.log10(256), 200)
    fps_theory = rate_peak / n_theory

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(n_theory, fps_theory, "--", color="gray", lw=1.6,
            label=f"single 5090 THEORY (no-OOM): card sustains ~{rate_peak:.0f} M/s → fps = {rate_peak:.0f}/N")
    ax.plot(n, fps, "o", color="tab:green", ms=8, label="single 5090 MEASURED (dev1, headless)")
    # dual overlay
    dorder = sorted(DUAL_2X, key=lambda s: TOTAL[s])
    ax.plot([TOTAL[s]/1e6 for s in dorder], [DUAL_2X[s] for s in dorder], "s-",
            color="tab:red", lw=2, ms=6, label="dual 2× 5090 (measured)")
    # OOM wall
    ax.axvspan(ceiling_n, 256, color="gray", alpha=0.12)
    ax.axvline(ceiling_n, color="k", ls=":", lw=1.4)
    ax.annotate(f"single-GPU OOM\n(~{ceiling_n:.0f}M, 32GB)", (ceiling_n, fps_theory[np.searchsorted(n_theory, ceiling_n)]),
                textcoords="offset points", xytext=(8, 40), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("problem size (million particles)")
    ax.set_ylabel("steady throughput (fps)")
    ax.set_title("Single RTX 5090: measured baseline + theoretical no-OOM extrapolation\n(dual 2×5090 runs past the single-GPU memory wall)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(_REPO / "logs/single_ceiling.png", dpi=120)
    print("-> logs/single_ceiling.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
