"""
_plot_scaling.py — draw the scaling curves from the campaign results.

Combines the earlier-measured 1M/4M/6M points with the campaign's 8M..16M
summaries (logs/scaling_results.jsonl) and draws:
  logs/scaling_eta.png  — strong-scaling efficiency η vs problem size
  logs/scaling_fps.png  — fps vs problem size (single 5090 / single AMD / dual best / ideal ceiling)

Robust to partial data: plots whatever summaries are present.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = pathlib.Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "logs/scaling_results.jsonl"

# Total particle count per case (x-axis), in millions.
TOTAL = {"1m": 1046529, "4m": 4182025, "6m": 6105841, "8m": 8128201,
         "10m": 10144225, "14m": 14160169, "16m": 16184529}

# Earlier-measured points (this session). best_fps is the achievable optimum
# (depth-3 at 1M/4M where it adds ≤4%; depth-2 = depth-3 from 6M up).
HISTORIC = {
    "1m": {"nv_fps": 494.1, "amd_fps": 254.2, "best_fps": 531.0, "best_weight": 2.6},
    "4m": {"nv_fps": 125.7, "amd_fps": 70.1, "best_fps": 180.0, "best_weight": 2.0},
    "6m": {"nv_fps": 89.4, "amd_fps": 47.6, "best_fps": 129.5, "best_weight": 1.8},
}


def load_points() -> dict:
    points = {k: dict(v) for k, v in HISTORIC.items()}
    if _RESULTS.exists():
        for line in _RESULTS.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "summary" and "best_fps" in record:
                size = record["size"]
                points[size] = {"nv_fps": record["nv_fps"], "amd_fps": record["amd_fps"],
                                "best_fps": record["best_fps"], "best_weight": record["best_weight"]}
    # finalize derived fields
    for size, p in points.items():
        p["n_millions"] = TOTAL[size] / 1e6
        p["ceiling"] = p["nv_fps"] + p["amd_fps"]
        p["eta"] = 100.0 * p["best_fps"] / p["ceiling"]
    return points


def main() -> int:
    points = load_points()
    order = sorted(points, key=lambda s: TOTAL[s])
    n = [points[s]["n_millions"] for s in order]
    eta = [points[s]["eta"] for s in order]
    nv = [points[s]["nv_fps"] for s in order]
    amd = [points[s]["amd_fps"] for s in order]
    dual = [points[s]["best_fps"] for s in order]
    ceiling = [points[s]["ceiling"] for s in order]
    weights = [points[s]["best_weight"] for s in order]
    print("size  N(M)   NV    AMD   ceil  dual  eta%   w")
    for s in order:
        p = points[s]
        print(f"{s:>4} {p['n_millions']:5.1f} {p['nv_fps']:6.1f} {p['amd_fps']:6.1f} "
              f"{p['ceiling']:6.1f} {p['best_fps']:6.1f} {p['eta']:5.1f}  {p['best_weight']}")

    # --- Figure 1: eta vs N -------------------------------------------------
    fig1, ax = plt.subplots(figsize=(9, 6))
    ax.plot(n, eta, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="ideal (100%)")
    for xi, yi, wi in zip(n, eta, weights):
        ax.annotate(f"{yi:.1f}%\n(w={wi})", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("problem size (million particles)")
    ax.set_ylabel("strong-scaling efficiency  η = dual / (single_NV + single_AMD)   [%]")
    ax.set_title("V4 cross-vendor strong-scaling efficiency vs problem size\n(RTX 5090 + RX 7900 XTX)")
    ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig1.tight_layout(); fig1.savefig(_REPO / "logs/scaling_eta.png", dpi=120)
    print("-> logs/scaling_eta.png")

    # --- Figure 2: fps vs N -------------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 6))
    ax.plot(n, dual, "o-", color="tab:purple", linewidth=2.2, markersize=8, label="dual (best weight)")
    ax.plot(n, ceiling, "k--", linewidth=1.2, label="ideal ceiling (NV+AMD)")
    ax.plot(n, nv, "s-", color="tab:green", linewidth=1.6, markersize=6, label="single RTX 5090")
    ax.plot(n, amd, "^-", color="tab:red", linewidth=1.6, markersize=6, label="single RX 7900 XTX")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("problem size (million particles)")
    ax.set_ylabel("steady throughput (fps)")
    ax.set_title("V4 throughput vs problem size (log-log)\n(RTX 5090 + RX 7900 XTX)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig2.tight_layout(); fig2.savefig(_REPO / "logs/scaling_fps.png", dpi=120)
    print("-> logs/scaling_fps.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
