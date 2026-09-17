"""3-D anatomy vs K (probe37): stacked per-GPU frame anatomy for the 64M and 32M stretched blocks
(K=1 reference, K=2/4/8 runs) with the inter-phase gaps split into c->a (queue ran dry / global loop
gating) and b->c (waiting for the neighbour's upload), plus the 64M K=8 loss ledger, 2-D (probe36)
and 3-D (probe37) side by side.

usage: plot_3d_anatomy.py <logs/n56> <out_dir>"""
import collections, glob, json, pathlib, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)


def anatomy_rows(folder, pattern, min_frame=1000):
    rows = []
    for f in sorted(glob.glob(str(folder / pattern))):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
            if m and int(m.group(1)) >= min_frame:
                rows.append({k: float(v) for k, v in (p.split("=") for p in m.group(3).split())})
    return rows


def steady_fps(folder, pattern):
    vals = []
    for f in sorted(glob.glob(str(folder / pattern))):
        s = re.search(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s", open(f, encoding="utf-8", errors="replace").read())
        if s:
            vals.append(int(s.group(1)) / float(s.group(2)))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows, period_ms):
    """per-GPU means (ms): A, B, C, c->a gap, b->c gap, other idle = period - sum."""
    def m(k):
        v = [r[k] for r in rows if k in r]; return float(np.mean(v)) / 1000 if v else 0.0
    # anatomy lines print the keys without the _us suffix (phase_a=245 ...)
    a, b, c = m("phase_a"), m("phase_b"), m("phase_c")
    ca, bc, ab = m("c_to_a_gap"), m("b_to_c_gap"), m("a_to_b_gap")
    other = max(0.0, period_ms - (a + b + c + ca + bc + ab))
    slack = [r[k] / 1000 for r in rows for k in ("upload_leading_to_c_gap", "upload_trailing_to_c_gap") if k in r]
    return dict(A=a, B=b, C=c, c_to_a=ca, b_to_c=bc + ab, other=other, period=period_ms, n=len(rows),
                slack_p50=float(np.median(slack)) if slack else float("nan"), slack_min=float(np.min(slack)) if slack else float("nan"))


blocks = {}
for out in sorted(glob.glob(str(root / "*" / "*" / "probe37_3d_*.out"))):
    folder = pathlib.Path(out).parent
    for block in ("strong64m", "strong32m"):
        ref_rows = anatomy_rows(folder, f"{block}_ref_g*.log")
        ref_fps = steady_fps(folder, f"{block}_ref_g*.log")
        blocks[(block, 1)] = summarize(ref_rows, 1000 / ref_fps)
        for K in (2, 4, 8):
            rows = anatomy_rows(folder, f"{block}_k{K}_run_t*.log")
            fps = steady_fps(folder, f"{block}_k{K}_run_t*.log")
            if rows:
                blocks[(block, K)] = summarize(rows, 1000 / fps)
# 2-D 64M from probe36 for the ledger
two_d = {}
for out in sorted(glob.glob(str(root / "*" / "*" / "probe36_ksweep_*.out"))):
    folder = pathlib.Path(out).parent
    sub = folder / "probe36_1594545"
    src = sub if sub.exists() else folder
    for K, pat in ((1, "64m_k8_ref_t*_g*.log"), (8, "64m_k8_run_t*.log")):
        rows = anatomy_rows(src, pat, min_frame=2000)
        fps = steady_fps(src, pat)
        if rows:
            two_d[K] = summarize(rows, 1000 / fps)

print(f"{'block':10s} {'K':>2s} {'period':>8s} {'A':>6s} {'B':>8s} {'C':>6s} {'c->a':>6s} {'b->c':>6s} {'other':>6s} {'B*K':>7s} {'slack p50/min':>14s}")
for (block, K), s in sorted(blocks.items()):
    print(f"{block:10s} {K:2d} {s['period']:8.2f} {s['A']:6.2f} {s['B']:8.2f} {s['C']:6.2f} {s['c_to_a']:6.2f} {s['b_to_c']:6.2f} {s['other']:6.2f} {s['B'] * K:7.1f} {s['slack_p50']:7.1f}/{s['slack_min']:6.1f}")

# ---------------- figure: stacked anatomy vs K, 64M and 32M
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
parts = [("A", "phase A", "#9ecae1"), ("B", "phase B", "#3182bd"), ("C", "phase C", "#e6550d"),
         ("c_to_a", "gap c→a (queue dry)", "#969696"), ("b_to_c", "gap a→b + b→c (upload wait)", "#fdae6b"), ("other", "other idle", "#d9d9d9")]
for col, (block, title) in enumerate((("strong64m", "3-D 64M stretched"), ("strong32m", "3-D 32M stretched"))):
    ks = [K for (b, K) in sorted(blocks) if b == block]
    x = np.arange(len(ks))
    for row, (subset, sub_title) in enumerate(((parts, "per-GPU frame vs K"), ([p for p in parts if p[0] != "B"], "everything except phase B (the non-scaling part)"))):
        ax = axes[row][col]; bottom = np.zeros(len(ks))
        for key, label, color in subset:
            vals = np.array([blocks[(block, K)][key] for K in ks])
            ax.bar(x, vals, bottom=bottom, color=color, label=label, width=0.6)
            top = max(sum(blocks[(block, K)][p[0]] for p in subset) for K in ks)
            for xi, v, b0 in zip(x, vals, bottom):
                if v > 0.03 * top:
                    ax.text(xi, b0 + v / 2, f"{v:.2f}" if row else f"{v:.1f}", ha="center", va="center", fontsize=7.5)
            bottom += vals
        for xi, K in zip(x, ks):
            note = (f"period {blocks[(block, K)]['period']:.1f} ms\nideal {blocks[(block, 1)]['period'] / K:.1f}" if row == 0
                    else f"{bottom[xi]:.1f} ms\n{100 * bottom[xi] / blocks[(block, K)]['period']:.0f}% of frame")
            ax.text(xi, bottom[xi] * 1.01, note, ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([f"K={K}" + (" (ref)" if K == 1 else "") for K in ks])
        ax.set_ylabel("ms per frame per GPU (f1000, all GPUs, 3 trials)"); ax.set_title(f"{title}: {sub_title}", fontsize=9); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, bottom.max() * 1.22)
axes[0][0].legend(fontsize=7.5, loc="upper right"); axes[1][0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(out_dir / "anatomy_vs_k_3d.png", dpi=140); print("saved", out_dir / "anatomy_vs_k_3d.png")

# ---------------- ledger: 64M K=8, 2-D vs 3-D
def ledger(ref, run, K=8):
    ideal = ref["period"] / K
    ideal_ac = (ref["A"] + ref["C"]) / K
    ideal_gaps = (ref["c_to_a"] + ref["b_to_c"] + ref["other"]) / K
    excess = run["period"] - ideal
    return dict(ideal_period=ideal, period=run["period"], eta=ideal / run["period"], excess=excess,
                A=run["A"], C=run["C"], AC_ideal=ideal_ac, AC_excess=run["A"] + run["C"] - ideal_ac,
                B=run["B"], B_ideal=ref["B"] / K, B_excess=run["B"] - ref["B"] / K,
                c_to_a=run["c_to_a"], b_to_c=run["b_to_c"], other=run["other"], gaps_excess=run["c_to_a"] + run["b_to_c"] + run["other"] - ideal_gaps)
rows = {}
if 1 in two_d and 8 in two_d:
    rows["2-D 64M (probe36)"] = ledger(two_d[1], two_d[8])
if ("strong64m", 1) in blocks and ("strong64m", 8) in blocks:
    rows["3-D 64M stretched (probe37)"] = ledger(blocks[("strong64m", 1)], blocks[("strong64m", 8)])
print("\n64M K=8 loss ledger (ms per frame per GPU):")
keys = [("ideal period = T_ref / 8", "ideal_period"), ("measured period", "period"), ("eta = ideal / measured", "eta"), ("excess over ideal", "excess"),
        ("  phase A (measured)", "A"), ("  phase C (measured)", "C"), ("  A + C at ideal (ref / 8)", "AC_ideal"), ("  -> A + C excess (per-frame floor)", "AC_excess"),
        ("  phase B (measured)", "B"), ("  phase B ideal (ref / 8)", "B_ideal"), ("  -> B excess (negative = faster than ideal)", "B_excess"),
        ("  gap c->a", "c_to_a"), ("  gap a->b + b->c", "b_to_c"), ("  other idle", "other"), ("  -> gaps + idle excess", "gaps_excess")]
print(f"{'item':44s} " + " ".join(f"{name:>28s}" for name in rows))
for label, k in keys:
    print(f"{label:44s} " + " ".join(f"{(100 * v[k] if k == 'eta' else v[k]):28.2f}" for v in rows.values()) + ("  %" if k == "eta" else ""))
json.dump({"blocks": {f"{b}_k{K}": s for (b, K), s in blocks.items()}, "two_d_64m": two_d, "ledger": rows}, open(out_dir / "anatomy_3d.json", "w"), indent=1)
