"""Task A: fixed-N K sweep (probe36) — eta vs K for 64M and 32M with error bars, K-run anatomy vs K,
transport slack samples, and the cross-NUMA K=2 control.
  reference(N, K, t) = mean over the K participating GPUs of fps_1 (measured simultaneously on those GPUs)
  eta_mean = fps_K / (K * mean_i fps_1,i),  eta_min = fps_K / (K * min_i fps_1,i),  error bar = std over trials.
RESULT label format: <size>_<tag>_ref_t<T>_g<G> | <size>_<tag>_run_t<T>, tag in {k2, k4, k8, k2x}.
usage: plot_ksweep36.py <logs/n56 root> <out_dir>"""
import collections, glob, json, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

root = pathlib.Path(sys.argv[1]); out_dir = pathlib.Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
refs = collections.defaultdict(dict)   # (size, tag, trial) -> {gpu: fps}
runs = {}                              # (size, tag, trial) -> fps
run_dirs = []
for out in sorted(glob.glob(str(root / "*" / "*" / "probe36_ksweep_*.out"))):
    text = open(out, encoding="utf-8", errors="replace").read()
    m = re.search(r"=== node (\S+) job (\d+) probe36_ksweep commit=(\S+)", text)
    if m:
        print("node", m.group(1), "job", m.group(2), "commit", m.group(3))
    run_dirs.append(pathlib.Path(out).parent)
    for m in re.finditer(r"RESULT label=(\w+?)_(k\d+x?)_(ref|run)_t(\d+)(?:_g(\d+))? rc=(\d+) steady_fps=([0-9.na]+) drift=([-0-9na]+) stamps=([0-9+na]+)", text):
        size, tag, kind, trial, gpu, rc, fps, drift, stamps = m.groups()
        if rc != "0" or fps == "nan":
            continue
        if drift != "0" or stamps not in ("0+0",):
            print(f"!! {size} {tag} t{trial} {kind}: drift={drift} stamps={stamps}")
        if kind == "ref":
            refs[(size, tag, int(trial))][int(gpu)] = float(fps)
        else:
            runs[(size, tag, int(trial))] = float(fps)

def anatomy(path, min_frame=2000):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
        if m and int(m.group(1)) >= min_frame:
            rows.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(3))})
    return rows

K_OF = {"k2": 2, "k4": 4, "k8": 8, "k2x": 2}
table = {}
print(f"{'size':>4s} {'tag':>4s} {'K':>2s} {'run fps (trials)':>24s} {'ref mean (trials)':>26s} {'ref spread t1':>13s} {'eta':>7s} {'+-':>5s} {'eta_min':>8s}")
for size in ("64m", "32m"):
    for tag in ("k2", "k4", "k8", "k2x"):
        trials = sorted(t for (s, g, t) in runs if s == size and g == tag and (s, g, t) in refs)
        if not trials:
            continue
        K = K_OF[tag]
        eta, eta_min, ref_means, spread = [], [], [], []
        for t in trials:
            r = np.array(list(refs[(size, tag, t)].values()))
            eta.append(runs[(size, tag, t)] / (K * r.mean())); eta_min.append(runs[(size, tag, t)] / (K * r.min()))
            ref_means.append(r.mean()); spread.append((r.min(), r.max()))
        table[(size, tag)] = {"K": K, "fps": [runs[(size, tag, t)] for t in trials], "eta": np.mean(eta),
                              "std": np.std(eta, ddof=1) if len(eta) > 1 else 0.0, "eta_min": np.mean(eta_min),
                              "ref_mean": ref_means, "spread": spread[0], "trials": trials}
        print(f"{size:>4s} {tag:>4s} {K:>2d} {str([round(v, 1) for v in table[(size, tag)]['fps']]):>24s} {str([round(float(v), 2) for v in ref_means]):>26s} "
              f"{spread[0][0]:6.2f}-{spread[0][1]:<6.2f} {np.mean(eta) * 100:6.1f}% {table[(size, tag)]['std'] * 100:4.1f}% {np.mean(eta_min) * 100:7.1f}%")

# ---------- figure 1: eta vs K ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
colors = {"64m": "#2a78d6", "32m": "#eb6834"}
for size, label in (("64m", "64M (8M per GPU at K=8)"), ("32m", "32M (4M per GPU at K=8)")):
    ks = [1] + [table[(size, g)]["K"] for g in ("k2", "k4", "k8") if (size, g) in table]
    etas = [100.0] + [table[(size, g)]["eta"] * 100 for g in ("k2", "k4", "k8") if (size, g) in table]
    errs = [0.0] + [table[(size, g)]["std"] * 100 for g in ("k2", "k4", "k8") if (size, g) in table]
    emin = [100.0] + [table[(size, g)]["eta_min"] * 100 for g in ("k2", "k4", "k8") if (size, g) in table]
    if len(ks) > 1:
        ax.errorbar(ks, etas, yerr=errs, fmt="o-", color=colors[size], capsize=4, linewidth=1.6, markersize=6, label=label + ", eta_mean")
        ax.plot(ks, emin, "s--", color=colors[size], linewidth=1, markersize=4, alpha=0.7, label=label + ", eta_min")
        for k, e in zip(ks[1:], etas[1:]):
            ax.annotate(f"{e:.1f}%", (k, e), textcoords="offset points", xytext=(0, 8 if size == "64m" else -14), ha="center", fontsize=8, color=colors[size])
    if (size, "k2x") in table:
        r = table[(size, "k2x")]
        ax.errorbar([2.15], [r["eta"] * 100], yerr=[r["std"] * 100], fmt="D", color=colors[size], capsize=3, markersize=6, mfc="white", label=f"{size} K=2 on cross-NUMA pair (3,4)")
ax.axhline(100, color="#888", linestyle=":", linewidth=1)
ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_ylim(60, 104)
ax.set_xlabel("GPUs (K), total N fixed"); ax.set_ylabel("strong-scaling efficiency (%)"); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("Fixed-N K sweep, one node, 3 trials, references on the participating GPUs", fontsize=9)
ax = axes[1]
for size in ("64m", "32m"):
    ks = [table[(size, g)]["K"] for g in ("k2", "k4", "k8") if (size, g) in table]
    if not ks:
        continue
    fps = [np.mean(table[(size, g)]["fps"]) for g in ("k2", "k4", "k8") if (size, g) in table]
    err = [np.std(table[(size, g)]["fps"], ddof=1) if len(table[(size, g)]["fps"]) > 1 else 0 for g in ("k2", "k4", "k8") if (size, g) in table]
    ref1 = np.mean([np.mean(table[(size, g)]["ref_mean"]) for g in ("k2", "k4", "k8") if (size, g) in table])
    ax.errorbar([1] + ks, [ref1] + fps, yerr=[0] + err, fmt="s-", color=colors[size], capsize=4, label=f"{size} measured")
    ax.plot([1] + ks, [ref1 * k for k in [1] + ks], ":", color=colors[size], linewidth=1, label=f"{size} ideal = K x mean single-GPU")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_xlabel("GPUs (K)"); ax.set_ylabel("steps per second")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5); ax.set_title("throughput vs K", fontsize=9)
fig.tight_layout(); fig.savefig(out_dir / "eta_strong_vs_k.png", dpi=140); print("saved", out_dir / "eta_strong_vs_k.png")

# ---------- figure 2: anatomy vs K (64M) + transport slack samples ----------
def collect_anatomy(size, tag, kind):
    rows = []
    for d in run_dirs:
        pattern = f"{size}_{tag}_{kind}_t*" + ("_g*.log" if kind == "ref" else ".log")
        for f in sorted(d.glob(pattern)):
            rows += anatomy(f)
    return rows
bars = []
for tag, K in (("ref", 1), ("k2", 2), ("k4", 4), ("k8", 8)):
    rows = collect_anatomy("64m", "k8" if tag == "ref" else tag, "ref" if tag == "ref" else "run")
    if not rows:
        continue
    g = lambda k: np.mean([r.get(k, 0) for r in rows]) / 1000
    fps = np.mean(table[("64m", "k8")]["ref_mean"]) if tag == "ref" else np.mean(table[("64m", tag)]["fps"])
    period = 1000.0 / fps
    vals = {"phase A": g("phase_a"), "phase B": g("phase_b"), "phase C": g("phase_c"),
            "gaps": g("a_to_b_gap") + g("b_to_c_gap") + g("c_to_a_gap")}
    vals["other idle"] = max(0.0, period - sum(vals.values()))
    slack = [r[k] / 1000 for r in rows for k in ("upload_leading_to_c_gap", "upload_trailing_to_c_gap") if k in r]
    bars.append((f"K={K}" + (" (ref)" if tag == "ref" else ""), vals, period, len(rows), slack))
    print(f"anatomy 64m K={K}: n={len(rows)} " + " ".join(f"{k}={v:.2f}" for k, v in vals.items()) + f" period={period:.2f} slack_p50={np.median(slack) if slack else float('nan'):.2f} ms")
if bars:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    x = np.arange(len(bars)); bottoms = np.zeros(len(bars))
    colors2 = {"phase A": "#9bb7d4", "phase B": "#2a78d6", "phase C": "#eb6834", "gaps": "#bbbbbb", "other idle": "#eeeeee"}
    for key in ("phase A", "phase B", "phase C", "gaps", "other idle"):
        v = np.array([b[1][key] for b in bars])
        ax.bar(x, v, bottom=bottoms, color=colors2[key], width=0.6, edgecolor="white", label=key)
        for xi, (b0, vv) in enumerate(zip(bottoms, v)):
            if vv > 0.5:
                ax.text(xi, b0 + vv / 2, f"{vv:.1f}", ha="center", va="center", fontsize=8)
        bottoms += v
    for xi, b in enumerate(bars):
        ax.text(xi, b[2] + 1.5, f"period {b[2]:.1f} ms", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars]); ax.set_ylabel("ms per frame per GPU (f2000 + f3000, all GPUs, 3 trials)")
    ax.set_title("64M: per-GPU frame anatomy vs K", fontsize=9); ax.legend(fontsize=7.5, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    data = [b[4] for b in bars[1:] if b[4]]
    labels = [b[0] for b in bars[1:] if b[4]]
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_ylabel("upload landed before Phase C start (ms)"); ax.set_title("64M: transport slack samples per K (2 frames x GPUs x links x trials)", fontsize=9)
        ax.axhline(0, color="#888", linestyle=":", linewidth=1); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "anatomy_vs_k_64m.png", dpi=140); print("saved", out_dir / "anatomy_vs_k_64m.png")
json.dump({f"{s}_{g}": {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in r.items() if k != "spread"} | {"spread": list(map(float, r["spread"]))}
           for (s, g), r in table.items()}, open(out_dir / "ksweep_table.json", "w"), indent=1, default=float)
