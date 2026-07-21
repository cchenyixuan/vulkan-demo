"""
_run_v5_replay.py — M5b calibration + prediction driver for the replay model.

Stage 1  CALIBRATE on K=2 native (measured segments from M5a):
         fit host_overhead_us so replay matches the measured K=2 fps,
         report the residual of everything else (the model's honesty check:
         only ONE free parameter, everything else directly measured).
Stage 2  VALIDATE on K=4-on-2-GPUs WITHOUT refitting: per-slab segments are
         scaled from K=2 measurements (compute ~ own-particle share,
         transport ~ per-seam boundary — same seam size for any K), the
         device map introduces engine sharing, and the prediction is
         compared against the MEASURED K=4 fps (M4 campaign: 92.8 @ 8M;
         30-min soak: 90.6).
Stage 3  PREDICT: K=2..10, {PCIe-staged, NVLink-P2P, ideal-zero-transport},
         1M + 8M → docs/m5b_replay_predictions.png + printed tables.

Usage:
    .venv/Scripts/python.exe experiment/v5/_run_v5_replay.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v5.utils.replay_v5 import (
    LinkDurations,
    PerSimDurations,
    ReplayParams,
    make_chain,
    replay,
)

# ---------------------------------------------------------------------------
# Measured inputs (M5a medians, docs/m5a_frame_anatomy.md; depth-1,
# warmup 5000, per-direction scheme, pool_safety 1.2, real dual 2x5090).
# ---------------------------------------------------------------------------

MEASURED = {
    "1M": {
        "phase_a": 78.6, "phase_b": 564.5, "phase_c": 580.9,
        "readback_dma": 153.3, "upload_dma": 177.7, "memcpy": 411.8,
        # measured references — POST-REBOOT clean baselines (2026-07-21):
        # the dirty-driver -12% depth-2 anomaly resolved on reboot.
        "fps_dual_depth2": 596.7,
        "fps_chain_k2": 606.5,
    },
    "8M": {
        "phase_a": 461.0, "phase_b": 4025.0, "phase_c": 3282.6,
        "readback_dma": 472.4, "upload_dma": 534.5, "memcpy": 1157.5,
        "fps_dual_depth2": 126.2,     # post-reboot clean baseline
        "fps_k4_measured": 92.8,      # M4 campaign 50k
        "fps_k4_soak": 90.6,          # 30-min soak mean
    },
}

# Blind-validation set: measured chain fps on 2 physical GPUs (M4 campaign
# 50k steady, maps alternate 0,1,0,1,...). The context-switch parameter is
# fit ONLY on 8M K=4; every other row is a no-refit prediction target.
# Post-reboot re-measured sweep (2026-07-21, current code, clean driver).
VALIDATION_POINTS = [
    # (case, K, measured_fps, note)
    ("1M", 3, 393.2, ""),
    ("1M", 4, 298.4, ""),
    ("1M", 6, 215.2, ""),
    ("1M", 8, 160.9, ""),
    ("8M", 4, 92.8, "pre-reboot m4 campaign"),
    ("8M", 6, 79.1, "pre-reboot m4 campaign"),
    ("8M", 8, 58.5, "pre-reboot m4 campaign"),
]


def build_k2(case: dict):
    table = [
        {"phase_a": case["phase_a"], "phase_b": case["phase_b"],
         "phase_c": case["phase_c"], "readback_dma": case["readback_dma"],
         "upload_dma": case["upload_dma"]},
        {"phase_a": case["phase_a"], "phase_b": case["phase_b"],
         "phase_c": case["phase_c"], "readback_dma": case["readback_dma"],
         "upload_dma": case["upload_dma"]},
    ]
    memcpy = {(0, 1): case["memcpy"], (1, 0): case["memcpy"]}
    return make_chain(table, memcpy)


def build_k_chain(case: dict, slab_count: int, device_map=None,
                  coresidency_alpha: float = 0.0):
    """Scale K=2 measurements to a K-chain: compute segments scale with the
    own-particle share (each slab owns 2/K of a K=2 slab's particles);
    transport segments are per-seam and stay CONSTANT (seam area does not
    depend on K). Interior slabs pay phase A ghost_send twice — approximated
    inside phase_a's boundary-proportional part being small (<7%).

    ``coresidency_alpha`` models CACHE/BW ANTAGONISM between sims sharing a
    physical device: each compute segment is slowed by
    (1 + alpha * (coresidents - 1)). Motivated by the stage-2 finding that
    a scheduling-cost model fails with opposite-sign errors across cases —
    the loss is multiplicative on kernels, not additive between batches
    (two working sets thrashing one L2). alpha is per-case (working-set
    size dependent), fit on K=4, validated blind on K=3/6/8."""
    scale = 2.0 / slab_count
    if device_map is None:
        device_map = [index % 2 for index in range(slab_count)]
    occupancy = {}
    for device in device_map:
        occupancy[device] = occupancy.get(device, 0) + 1
    table = []
    for index in range(slab_count):
        slowdown = 1.0 + coresidency_alpha * (occupancy[device_map[index]] - 1)
        table.append({
            "phase_a": case["phase_a"] * scale * slowdown,
            "phase_b": case["phase_b"] * scale * slowdown,
            "phase_c": case["phase_c"] * scale * slowdown,
            "readback_dma": case["readback_dma"],
            "upload_dma": case["upload_dma"],
        })
    memcpy = {}
    for index in range(slab_count - 1):
        memcpy[(index, index + 1)] = case["memcpy"]
        memcpy[(index + 1, index)] = case["memcpy"]
    return make_chain(table, memcpy)


def fit_host_overhead(case: dict, target_fps: float) -> float:
    """One-parameter fit: bisect host_overhead_us so K=2 replay hits the
    measured fps."""
    sims, links = build_k2(case)
    low, high = 0.0, 800.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        result = replay(sims, links, [0, 1],
                        ReplayParams(host_overhead_us=mid))
        if result.steady_fps > target_fps:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def main() -> int:
    print("=" * 72)
    print("Stage 1 — K=2 calibration (one free parameter: host_overhead_us)")
    print("=" * 72)
    overhead = {}
    for tag, case in MEASURED.items():
        raw = replay(*build_k2(case), [0, 1], ReplayParams())
        target = case["fps_dual_depth2"]
        overhead[tag] = fit_host_overhead(case, target)
        fitted = replay(*build_k2(case), [0, 1],
                        ReplayParams(host_overhead_us=overhead[tag]))
        print(f"{tag}: raw model {raw.steady_fps:.1f} fps vs measured "
              f"{target:.1f} -> host_overhead fit {overhead[tag]:.0f} µs "
              f"(fitted {fitted.steady_fps:.1f} fps; model explains "
              f"{100 * target / raw.steady_fps:.1f}% before the fit)")

    print()
    print("=" * 72)
    print("Stage 2 — oversubscription validation: host memcpy serialization "
          "model, zero fitted parameters")
    print("=" * 72)

    # ZERO-parameter oversubscription model (2026-07-21 breakthrough):
    # the post-reboot K-sweep showed period ~linear in K — worker memcpys
    # SERIALIZE on the shared host beyond the 2-concurrent baseline that
    # the K=2-measured memcpy durations already embed. memcpy_channels=1
    # for K>=3 (marginal copies serialize); K=2 keeps parallel (inputs
    # were measured under exactly that 2-way concurrency). The former
    # coresidency-alpha model is retired (it needed per-case fits and
    # still missed K=3 by 37%).
    print(f"{'case':>4} {'K':>3} {'measured':>9} {'replay':>9} "
          f"{'error':>8}")
    worst = 0.0
    for tag, slab_count, measured_fps, _ in VALIDATION_POINTS:
        case_data = MEASURED[tag]
        device_map_chain = [index % 2 for index in range(slab_count)]
        sims, links = build_k_chain(case_data, slab_count,
                                    device_map_chain, 0.0)
        channels = 0 if slab_count <= 2 else 1
        res = replay(sims, links, device_map_chain,
                     ReplayParams(host_overhead_us=overhead[tag],
                                  memcpy_channels=channels))
        error = 100 * (res.steady_fps - measured_fps) / measured_fps
        worst = max(worst, abs(error))
        print(f"{tag:>4} {slab_count:>3} {measured_fps:>9.1f} "
              f"{res.steady_fps:>9.1f} {error:>+7.1f}%")
    print(f"\nworst error (all points, ZERO oversubscription-specific "
          f"fits): {worst:.1f}% ({'PASS' if worst <= 10.0 else 'NEEDS WORK'}"
          f" vs ±10% bar)")

    print()
    print("=" * 72)
    print("Stage 3 — predictions K=2..10 (1 sim/GPU, i.e. the 10x3090 shape)")
    print("=" * 72)
    transports = {
        "PCIe-staged (measured)": dict(),
        "NVLink-P2P (rb/up->25us, memcpy->0)": dict(
            readback_dma=25.0, upload_dma=25.0, memcpy=0.0),
        "ideal zero-transport": dict(
            readback_dma=0.0, upload_dma=0.0, memcpy=0.0, sched=0.0, sem=0.0),
    }
    predictions = {}
    for tag, case in MEASURED.items():
        print(f"\n--- {tag} (host_overhead {overhead[tag]:.0f} µs) ---")
        print(f"{'K':>3} " + " ".join(f"{name:>28}" for name in transports))
        rows = {}
        for slab_count in range(2, 11):
            row = []
            for name, override in transports.items():
                mod = dict(case)
                mod["readback_dma"] = override.get(
                    "readback_dma", case["readback_dma"])
                mod["upload_dma"] = override.get(
                    "upload_dma", case["upload_dma"])
                mod["memcpy"] = override.get("memcpy", case["memcpy"])
                sims, links = build_k_chain(mod, slab_count)
                params = ReplayParams(
                    host_overhead_us=overhead[tag],
                    sched_gap_us=override.get("sched", 44.8),
                    sem_latency_us=override.get("sem", 35.8))
                # 1 sim per GPU — the real-cluster shape. The 10x3090 is
                # a SINGLE HOST: staged memcpys serialize there too
                # (channels=1); P2P/ideal transports have no memcpy.
                if mod["memcpy"] > 0 and slab_count > 2:
                    params.memcpy_channels = 1
                res = replay(sims, links, list(range(slab_count)), params)
                row.append(res.steady_fps)
            rows[slab_count] = row
            print(f"{slab_count:>3} " + " ".join(f"{v:>28.1f}" for v in row))
        predictions[tag] = rows

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLORS = ["#2a78d6", "#008300", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for ax, (tag, rows) in zip(axes, predictions.items()):
        ks = sorted(rows)
        for index, name in enumerate(transports):
            ax.plot(ks, [rows[k][index] for k in ks], color=COLORS[index],
                    linewidth=1.8, marker="o", markersize=4,
                    label=name.split(" (")[0])
        ax.set_title(f"{tag} — predicted fps vs GPU count (1 sim/GPU)",
                     fontsize=10, loc="left", color="#333333")
        ax.set_xlabel("K (GPUs)", fontsize=9, color="#767676")
        ax.set_ylabel("fps", fontsize=9, color="#767676")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.legend(fontsize=8, frameon=False, labelcolor="#333333")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors="#767676", labelsize=8)
    fig.suptitle("M5b replay predictions (calibrated on K=2; validated on "
                 "K=4-on-2-GPUs)", fontsize=11, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("docs/m5b_replay_predictions.png", facecolor="white")
    print("\nwrote docs/m5b_replay_predictions.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
