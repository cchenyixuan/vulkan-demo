"""Standardized WEAK-scaling efficiency (probe35 protocol), two families (16M/GPU and 2M/GPU).
  eta_weak(K, t) = fps_K(t) / mean_{i<K} fps_1(t; GPU i)   (reference = K=1 case on all 8 GPUs simultaneously,
  restricted to the GPUs 0..K-1 that the K-GPU run used); error bar = std over the 3 trials; eta_min uses the
  slowest of those GPUs.
usage: plot_weak35.py <logs/n56 root> <out.png>"""
import csv, glob, pathlib, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

root = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
refs = collections.defaultdict(dict)   # (family, trial) -> {gpu: fps}
runs = {}                              # (family, K, trial) -> fps
nodes = {}; drifts = set()
for out in sorted(glob.glob(str(root / "*" / "*" / "probe35_weak_*.out"))):
    text = open(out, encoding="utf-8", errors="replace").read()
    m = re.search(r"=== node (\S+) job (\d+) family=(\w+)", text)
    if m:
        nodes[m.group(3)] = (m.group(1), m.group(2))
    for m in re.finditer(r"RESULT label=(weak\d*)_(ref|k(\d+))_t(\d+)(?:_g(\d+))? rc=(\d+) steady_fps=([0-9.na]+) drift=([-0-9na]+)", text):
        family, kind, k, trial, gpu, rc, fps, drift = m.groups()
        if rc != "0" or fps == "nan":
            continue
        drifts.add(drift)
        if kind == "ref":
            refs[(family, int(trial))][int(gpu)] = float(fps)
        else:
            runs[(family, int(k), int(trial))] = float(fps)

families = [("weak32", "32M per GPU (32M .. 256M)", "#4a3aa7"), ("weak16", "16M per GPU (16M .. 128M)", "#2a78d6"),
            ("weak8", "8M per GPU (8M .. 64M)", "#1baf7a"), ("weak4", "4M per GPU (4M .. 32M)", "#eda100"),
            ("weak", "2M per GPU (2M .. 16M)", "#eb6834")]
table = {}
print("nodes:", nodes, "| drift values:", sorted(drifts))
print(f"{'family':>7s} {'K':>2s} {'fps_K (trials)':>24s} {'ref mean_{i<K} (trials)':>26s} {'ref spread':>13s} {'eta':>7s} {'+-':>5s} {'eta_min':>8s}")
for family, label, color in families:
    ks = sorted({k for (f, k, t) in runs if f == family})
    rows = []
    for K in ks:
        trials = sorted(t for (f, k, t) in runs if f == family and k == K and (family, t) in refs)
        eta, eta_min, ref_means, spread = [], [], [], []
        for t in trials:
            sub = np.array([refs[(family, t)][g] for g in range(K) if g in refs[(family, t)]])
            if sub.size == 0:
                continue
            eta.append(runs[(family, K, t)] / sub.mean()); eta_min.append(runs[(family, K, t)] / sub.min())
            ref_means.append(sub.mean()); spread.append((sub.min(), sub.max()))
        if not eta:
            continue
        rows.append({"K": K, "fps": [runs[(family, K, t)] for t in trials], "eta": np.mean(eta),
                     "std": np.std(eta, ddof=1) if len(eta) > 1 else 0.0, "eta_min": np.mean(eta_min),
                     "ref_mean": ref_means, "spread": spread[0]})
        print(f"{family:>7s} {K:>2d} {str([round(v, 1) for v in rows[-1]['fps']]):>24s} {str([round(v, 2) for v in ref_means]):>26s} "
              f"{spread[0][0]:6.2f}-{spread[0][1]:<6.2f} {np.mean(eta) * 100:6.1f}% {rows[-1]['std'] * 100:4.1f}% {np.mean(eta_min) * 100:7.1f}%")
    # K=1 point: eta = 1 by definition (reference itself); show the all-8 spread for context
    all8 = [np.array(list(refs[(family, t)].values())) for t in sorted(t for (f, t) in refs if f == family)]
    table[family] = (rows, all8)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
for family, label, color in families:
    rows, all8 = table.get(family, ([], []))
    if not rows:
        continue
    ks = [1] + [r["K"] for r in rows]
    etas = [100.0] + [r["eta"] * 100 for r in rows]
    errs = [0.0] + [r["std"] * 100 for r in rows]
    k8 = [r for r in rows if r["K"] == 8]
    tag = f"  (K=8: {k8[0]['eta'] * 100:.1f} +- {k8[0]['std'] * 100:.1f}%)" if k8 else ""
    ax.errorbar(ks, etas, yerr=errs, fmt="o-", color=color, capsize=4, linewidth=1.6, markersize=6, label=label + tag)
ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 3, 4, 6, 8]); ax.set_xticklabels(["1", "2", "3", "4", "6", "8"])
ax.set_xlabel("GPUs (K), particles per GPU fixed"); ax.set_ylabel("weak-scaling efficiency (%)"); ax.set_ylim(0, 105)
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("8x RTX 5090, 3 trials (mean +- std); reference = each GPU's own K=1, all 8 measured simultaneously", fontsize=8.5)
ax = axes[1]
for family, label, color in families:
    rows, all8 = table.get(family, ([], []))
    if not rows:
        continue
    ks = [1] + [r["K"] for r in rows]
    ref1 = np.mean([a.mean() for a in all8]) if all8 else np.nan
    fps = [ref1] + [np.mean(r["fps"]) for r in rows]
    err = [np.std([a.mean() for a in all8], ddof=1) if len(all8) > 1 else 0] + [np.std(r["fps"], ddof=1) if len(r["fps"]) > 1 else 0 for r in rows]
    ax.errorbar(ks, fps, yerr=err, fmt="s-", color=color, capsize=4, label=f"{label}: steps/s")
    ax.axhline(ref1, color=color, linestyle=":", linewidth=1)
ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 3, 4, 6, 8]); ax.set_xticklabels(["1", "2", "3", "4", "6", "8"])
ax.set_yscale("log"); ax.set_xlabel("GPUs (K)"); ax.set_ylabel("steps per second (ideal weak scaling = flat dotted line)"); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8); ax.set_title("throughput vs K at fixed load per GPU", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print("saved", OUT)
