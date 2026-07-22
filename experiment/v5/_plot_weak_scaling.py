"""
_plot_weak_scaling.py — analyze + plot the constant-work-per-slab campaign.

Reads logs/weak_scaling_20260722/summary.jsonl (latest row wins per point).

Headline metric per chain point:
    ideal_us    = slabs_per_gpu * T_solo_us      (perfect compute serialization,
                  zero coordination cost; dual points use the SLOWER device's
                  T_solo because the chain period binds on it)
    overhead_us = frame_mean_us - ideal_us       (exposed coordination cost)
    eta_weak    = ideal_us / frame_mean_us

Matched pairs (equal slabs/GPU, chain length doubles, links go intra->cross):
    chain1_k2_dX (1 link)  vs dual_k4 (3 links)
    chain1_k3_dX (2 links) vs dual_k6 (5 links)
    chain1_k4_dX (3 links) vs dual_k8 (7 links)

Outputs: logs/weak_scaling_20260722/weak_scaling.png + stdout tables.
"""

from __future__ import annotations

import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DIR = _REPO / "logs/weak_scaling_20260722"

LINK_COUNT = {"dual_k2": 1, "chain1_k2_d0": 1, "chain1_k2_d1": 1,
              "chain1_k3_d0": 2, "chain1_k3_d1": 2, "dual_k4": 3,
              "chain1_k4_d0": 3, "chain1_k4_d1": 3, "dual_k6": 5,
              "dual_k8": 7}
MATCHED_PAIRS = [("chain1_k2", "dual_k4"), ("chain1_k3", "dual_k6"),
                 ("chain1_k4", "dual_k8")]


def load_rows() -> dict:
    rows: dict = {}
    with open(_DIR / "summary.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["point"]] = row          # later rows overwrite = re-runs win
    return rows


def main() -> int:
    rows = load_rows()
    solo = {}
    for device in ("0", "1"):
        row = rows.get(f"solo_d{device}")
        if row is None or "frame_mean_us" not in row:
            print(f"missing solo_d{device}; cannot compute overhead")
            return 1
        solo[device] = row["frame_mean_us"]
    print(f"T_solo: dev0={solo['0']:.1f}us ({1e6/solo['0']:.1f} fps)  "
          f"dev1={solo['1']:.1f}us ({1e6/solo['1']:.1f} fps)")

    table = []
    for name, row in rows.items():
        if name.startswith("solo") or "frame_mean_us" not in row:
            continue
        if row.get("drift") not in (0, None) or row.get("validation_failed"):
            print(f"WARNING {name}: drift={row.get('drift')} "
                  f"validation_failed={row.get('validation_failed')}")
        slabs = row["slabs_per_gpu"]
        if row["devices"] == "0,1":
            reference = slabs * max(solo.values())
        else:
            reference = slabs * solo[row["devices"]]
        measured = row["frame_mean_us"]
        table.append({
            "point": name, "slabs_per_gpu": slabs,
            "links": LINK_COUNT.get(name), "fps": row.get("steady_fps"),
            "measured_us": measured, "ideal_us": round(reference, 1),
            "overhead_us": round(measured - reference, 1),
            "eta_weak": round(reference / measured, 4),
        })
    table.sort(key=lambda r: (r["slabs_per_gpu"], r["point"]))

    header = f"{'point':<14}{'slabs/GPU':>10}{'links':>7}{'fps':>8}" \
             f"{'measured_us':>13}{'ideal_us':>10}{'overhead_us':>13}{'eta_weak':>10}"
    print(header)
    print("-" * len(header))
    for row in table:
        print(f"{row['point']:<14}{row['slabs_per_gpu']:>10}{row['links']:>7}"
              f"{row['fps']:>8}{row['measured_us']:>13.1f}{row['ideal_us']:>10.1f}"
              f"{row['overhead_us']:>13.1f}{row['eta_weak']:>10.4f}")

    by_point = {row["point"]: row for row in table}
    print("\nMatched pairs (same slabs/GPU; chain doubles, links intra->cross):")
    for single_stem, dual_name in MATCHED_PAIRS:
        dual = by_point.get(dual_name)
        singles = [by_point.get(f"{single_stem}_d{d}") for d in ("0", "1")]
        singles = [s for s in singles if s]
        if not dual or not singles:
            continue
        single_avg = sum(s["overhead_us"] for s in singles) / len(singles)
        print(f"  {single_stem} (overhead avg {single_avg:.0f}us over "
              f"{len(singles)} device(s)) -> {dual_name} "
              f"(overhead {dual['overhead_us']:.0f}us): "
              f"delta {dual['overhead_us'] - single_avg:+.0f}us for "
              f"+{dual['links'] - singles[0]['links']} links crossing GPUs")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (axis_period, axis_overhead) = plt.subplots(1, 2, figsize=(13, 5))
    groups = {
        "chain1 (dev0)": ([], [], "tab:orange", "o"),
        "chain1 (dev1)": ([], [], "tab:red", "s"),
        "dual ABAB":     ([], [], "tab:blue", "D"),
    }
    for row in table:
        if row["point"].endswith("_d0"):
            key = "chain1 (dev0)"
        elif row["point"].endswith("_d1"):
            key = "chain1 (dev1)"
        else:
            key = "dual ABAB"
        groups[key][0].append(row["slabs_per_gpu"])
        groups[key][1].append(row["measured_us"] / 1000.0)

    slabs_axis = [1, 2, 3, 4]
    axis_period.plot(slabs_axis, [s * max(solo.values()) / 1000.0 for s in slabs_axis],
                     "k--", label="ideal = slabs/GPU x T_solo(slow dev)", lw=1)
    axis_period.plot(slabs_axis, [s * min(solo.values()) / 1000.0 for s in slabs_axis],
                     "k:", label="ideal (fast dev)", lw=1)
    for label, (xs, ys, color, marker) in groups.items():
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        axis_period.plot([xs[i] for i in order], [ys[i] for i in order],
                         marker=marker, color=color, label=label)
    axis_period.set_xlabel("slabs per GPU")
    axis_period.set_ylabel("steady frame period (ms)")
    axis_period.set_title("Weak scaling: constant 2M per slab")
    axis_period.set_xticks(slabs_axis)
    axis_period.grid(alpha=0.3)
    axis_period.legend(fontsize=8)

    for label, (_, _, color, marker) in groups.items():
        xs = [row["links"] for row in table
              if (row["point"].endswith("_d0")) == (label == "chain1 (dev0)")
              and (row["point"].endswith("_d1")) == (label == "chain1 (dev1)")
              and (("dual" in row["point"])) == (label == "dual ABAB")]
        ys = [row["overhead_us"] for row in table
              if (row["point"].endswith("_d0")) == (label == "chain1 (dev0)")
              and (row["point"].endswith("_d1")) == (label == "chain1 (dev1)")
              and (("dual" in row["point"])) == (label == "dual ABAB")]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        axis_overhead.plot([xs[i] for i in order], [ys[i] for i in order],
                           marker=marker, color=color, label=label)
    axis_overhead.axhline(0.0, color="k", lw=0.8)
    axis_overhead.set_xlabel("ghost links in chain")
    axis_overhead.set_ylabel("overhead = measured - ideal (us)")
    axis_overhead.set_title("Exposed coordination cost vs chain length")
    axis_overhead.grid(alpha=0.3)
    axis_overhead.legend(fontsize=8)

    figure.tight_layout()
    out_path = _DIR / "weak_scaling.png"
    figure.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
