"""
_plot_compare.py — overlay cross-vendor (5090+AMD) vs same-vendor (2× 5090).

Cross-vendor 7-point curve is hardcoded (measured 2026-06-18, frozen). The 2×5090
curve is read live from logs/scaling_2x5090.jsonl (the V5 campaign). Emits:
  logs/compare_eta.png  — η_strong vs N, both rigs
  logs/compare_fps.png  — dual fps vs N (log-log), both rigs + single-5090 reference

Robust to partial 2×5090 data (plots whatever summaries exist).
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = pathlib.Path(__file__).resolve().parents[2]
_2X = _REPO / "logs/scaling_2x5090.jsonl"

TOTAL = {"1m": 1046529, "2m": 2064969, "4m": 4182025, "6m": 6105841,
         "8m": 8128201, "10m": 10144225, "14m": 14160169, "16m": 16184529,
         "32m": 32251041, "64m": 64368529, "128m": 128482225, "160m": 160554241,
         "192m": 192626641, "256m": 256736529}

# Cross-vendor (RTX 5090 + RX 7900 XTX), measured 2026-06-18.
CROSS_VENDOR = {
    "1m":  {"nv": 494.0, "other": 254.0, "dual": 531.0, "w": 2.6, "eta": 71.0},
    "4m":  {"nv": 126.0, "other": 70.0,  "dual": 180.0, "w": 2.0, "eta": 91.9},
    "6m":  {"nv": 89.4,  "other": 47.6,  "dual": 129.5, "w": 1.8, "eta": 94.5},
    "8m":  {"nv": 68.1,  "other": 33.9,  "dual": 99.2,  "w": 1.8, "eta": 97.3},
    "10m": {"nv": 55.9,  "other": 27.3,  "dual": 81.9,  "w": 1.8, "eta": 98.5},
    "14m": {"nv": 38.7,  "other": 19.2,  "dual": 56.8,  "w": 1.9, "eta": 98.1},
    "16m": {"nv": 34.6,  "other": 16.9,  "dual": 50.9,  "w": 1.9, "eta": 98.9},
}

# 2x5090 EXTREME points — dual-only (single-GPU OOMs beyond ~90M on 32GB, so no
# measured η; per-particle single rate has plateaued ~573 M/s → est η ≈ 96–100%).
EXTREME_2X_FPS = {"128m": 8.6, "160m": 7.2, "192m": 5.7}   # 192M = the ceiling (~30GB/card, pool 95%)


def load_2x5090() -> dict:
    points = {}
    if not _2X.exists():
        return points
    for line in _2X.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "summary" and "best_fps" in r:
            points[r["size"]] = {
                "nv": r["nv_fps"], "other": r["amd_fps"], "dual": r["best_fps"],
                "w": r["best_weight"], "eta": 100.0 * r["eta_strong"]}
    return points


def series(points: dict, key: str):
    order = sorted(points, key=lambda s: TOTAL[s])
    return ([TOTAL[s] / 1e6 for s in order], [points[s][key] for s in order], order)


def main() -> int:
    same = load_2x5090()
    print(f"cross-vendor points: {len(CROSS_VENDOR)}   2x5090 points: {len(same)}")
    for s in sorted(same, key=lambda s: TOTAL[s]):
        p = same[s]
        print(f"  2x5090 {s:>4}: dual={p['dual']:.1f} fps  w={p['w']}  eta={p['eta']:.1f}%")

    # --- eta vs N ---------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(9.5, 6))
    nx, ey, ox = series(CROSS_VENDOR, "eta")
    ax.plot(nx, ey, "o-", color="tab:blue", lw=2, ms=8, label="cross-vendor (5090 + 7900 XTX)")
    for x, y, s in zip(nx, ey, ox):
        ax.annotate(f"w{CROSS_VENDOR[s]['w']}", (x, y), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7, color="tab:blue")
    if same:
        sx, sy, sox = series(same, "eta")
        ax.plot(sx, sy, "s-", color="tab:red", lw=2, ms=8, label="same-vendor (2× 5090)")
        for x, y, s in zip(sx, sy, sox):
            ax.annotate(f"w{same[s]['w']}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="tab:red")
    ax.axhline(100, color="gray", ls="--", lw=1, label="ideal (100%)")
    ax.set_xlabel("problem size (million particles)")
    ax.set_ylabel("strong-scaling efficiency η  [%]")
    ax.set_title("Strong-scaling efficiency: cross-vendor vs same-vendor (host-staged, no P2P)")
    ax.set_ylim(60, 105); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig1.tight_layout(); fig1.savefig(_REPO / "logs/compare_eta.png", dpi=120)
    print("-> logs/compare_eta.png")

    # --- fps vs N (log-log) ----------------------------------------------
    fig2, ax = plt.subplots(figsize=(9.5, 6))
    nx, dy, _ = series(CROSS_VENDOR, "dual")
    ax.plot(nx, dy, "o-", color="tab:blue", lw=2, ms=7, label="dual cross-vendor")
    _, nvy, _ = series(CROSS_VENDOR, "nv")
    ax.plot(nx, nvy, ":", color="gray", lw=1.5, label="single RTX 5090 (ref)")
    if same:
        sx, sdy, _ = series(same, "dual")
        extreme = sorted(EXTREME_2X_FPS, key=lambda s: TOTAL[s])
        ex_x = [TOTAL[s] / 1e6 for s in extreme]
        ex_y = [EXTREME_2X_FPS[s] for s in extreme]
        ax.plot(sx + ex_x, sdy + ex_y, "s-", color="tab:red", lw=2.2, ms=7, label="dual 2× 5090")
        ax.plot(ex_x, ex_y, "*", color="darkred", ms=15,
                label="2× 5090 extreme (dual-only; single OOMs)")
        ax.annotate("192M @ 5.7 fps\n(ceiling, ~30GB/card)", (ex_x[-1], ex_y[-1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="darkred")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("problem size (million particles)")
    ax.set_ylabel("steady throughput (fps)")
    ax.set_title("Throughput: cross-vendor vs same-vendor (log-log)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig2.tight_layout(); fig2.savefig(_REPO / "logs/compare_fps.png", dpi=120)
    print("-> logs/compare_fps.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
