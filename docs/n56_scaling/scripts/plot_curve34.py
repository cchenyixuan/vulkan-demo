"""Standardized strong-scaling efficiency vs problem size (probe34 protocol).
  reference(N, trial t) = mean over the 8 GPUs of fps_1(N; GPU i) measured SIMULTANEOUSLY (128M: mean over
  the 4 GPU pairs of fps_2, factor 4 instead of 8)
  eta_mean(N, t) = fps_8(N, t) / (8 * mean_i fps_ref)      eta_min(N, t) = fps_8(N, t) / (8 * min_i fps_ref)
  error bar = std over the 3 trials; per-GPU reference spread and SM clocks (telemetry) reported in the table.
usage: plot_curve34.py <logs/n56 root> <out.png>"""
import csv, glob, pathlib, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
refs = collections.defaultdict(dict)   # (size, trial) -> {gpu_label: fps}
k8 = {}                                # (size, trial) -> fps
drifts = set(); nodes = {}
for out in sorted(glob.glob(str(root / "*" / "*" / "probe34_curve_*.out"))):
    text = open(out, encoding="utf-8", errors="replace").read()
    m = re.search(r"=== node (\S+) job (\d+) sizes=\[([^\]]*)\]", text)
    if m:
        nodes[m.group(3)] = (m.group(1), m.group(2))
    job = m.group(2) if m else out
    if not re.search(r"RESULT label=\w+_k8_t\d+ rc=0 steady_fps=[0-9.]+", text):
        continue   # timed-out / cancelled job: no K=8 result, skip its references
    for m in re.finditer(r"RESULT label=(\w+?)_(ref|k8)_t(\d+)(?:_g(\d+))? rc=(\d+) steady_fps=([0-9.na]+) drift=([-0-9na]+)", text):
        size, kind, trial, gpu, rc, fps, drift = m.groups()
        if rc != "0" or fps == "nan":
            continue
        drifts.add(drift)
        if kind == "ref":
            refs[(size, int(trial))][gpu] = float(fps)
        else:
            k8[(size, int(trial))] = float(fps)

# telemetry: mean SM clock per GPU while busy (util >= 80), per job
clocks = {}
for tele in sorted(glob.glob(str(root / "*" / "*probe34*" / "telemetry.csv"))):
    per = collections.defaultdict(list)
    for row in csv.reader(open(tele, encoding="utf-8", errors="replace")):
        if len(row) < 6 or row[0].startswith("timestamp"):
            continue
        try:
            idx = int(row[1]); util = float(row[2].strip().rstrip("%")); sm = float(row[5].strip().rstrip("MHz"))
        except ValueError:
            continue
        if util >= 80:
            per[idx].append(sm)
    clocks[pathlib.Path(tele).parent.name] = {i: float(np.mean(v)) for i, v in sorted(per.items()) if v}

sizes = ["4m", "16m", "32m", "64m", "128m"]
per_gpu = {"4m": 4.18e6 / 8, "16m": 16.4e6 / 8, "32m": 32.5e6 / 8, "64m": 64.4e6 / 8, "128m": 128.3e6 / 8}
rows = []
for size in sizes:
    trials = sorted(t for (s, t) in k8 if s == size and (s, t) in refs)
    if not trials:
        continue
    factor = 4 if size == "128m" else 8
    eta_mean, eta_min, ref_means, ref_spread = [], [], [], []
    for t in trials:
        r = np.array(list(refs[(size, t)].values()))
        eta_mean.append(k8[(size, t)] / (factor * r.mean()))
        eta_min.append(k8[(size, t)] / (factor * r.min()))
        ref_means.append(r.mean()); ref_spread.append((r.min(), r.max()))
    rows.append({"size": size, "x": per_gpu[size] / 1e6, "factor": factor, "trials": trials,
                 "k8": [k8[(size, t)] for t in trials], "ref_mean": ref_means, "ref_spread": ref_spread,
                 "eta_mean": np.mean(eta_mean), "eta_mean_std": np.std(eta_mean, ddof=1) if len(trials) > 1 else 0.0,
                 "eta_min": np.mean(eta_min), "eta_min_std": np.std(eta_min, ddof=1) if len(trials) > 1 else 0.0})
print("nodes:", nodes, "| drift values seen:", sorted(drifts))
print("busy SM clocks per GPU (MHz) per job:", {k: {i: round(v) for i, v in c.items()} for k, c in clocks.items()})
print(f"{'size':>5s} {'per GPU':>8s} {'K=8 fps (trials)':>26s} {'ref mean (trials)':>26s} {'ref min-max (trial 1)':>22s} {'eta_mean':>9s} {'+-':>5s} {'eta_min':>8s} {'+-':>5s}")
for r in rows:
    print(f"{r['size']:>5s} {r['x']:6.2f}M {str([round(v, 1) for v in r['k8']]):>26s} {str([round(v, 2) for v in r['ref_mean']]):>26s} "
          f"{r['ref_spread'][0][0]:8.2f}-{r['ref_spread'][0][1]:<8.2f}     {r['eta_mean'] * 100:6.1f}% {r['eta_mean_std'] * 100:4.1f}% {r['eta_min'] * 100:7.1f}% {r['eta_min_std'] * 100:4.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
x = [r["x"] for r in rows]
ax.errorbar(x, [r["eta_mean"] * 100 for r in rows], yerr=[r["eta_mean_std"] * 100 for r in rows], fmt="o-", color="#2a78d6",
            capsize=4, linewidth=1.6, markersize=6, label="eta_mean: vs mean of the 8 GPUs' own K=1")
ax.errorbar(x, [r["eta_min"] * 100 for r in rows], yerr=[r["eta_min_std"] * 100 for r in rows], fmt="s--", color="#eb6834",
            capsize=4, linewidth=1.2, markersize=5, label="eta_min: vs the slowest GPU's K=1")
for r in rows:
    ax.annotate(f"{r['eta_mean'] * 100:.1f}%", (r["x"], r["eta_mean"] * 100), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#2a78d6")
ax.set_xscale("log"); ax.set_xlabel("particles per GPU (M); total N = 8x; 128M point uses 4 x K=2 reference")
ax.set_ylabel("strong-scaling efficiency at K=8 (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3, which="both")
from matplotlib.ticker import NullFormatter, NullLocator
ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(x); ax.set_xticklabels([f"{r['size']}\n{r['x']:.2f}M" for r in rows], fontsize=8)
ax.legend(fontsize=8, loc="lower right"); ax.set_title("8x RTX 5090 (one node), 3 trials, references measured on all 8 GPUs simultaneously", fontsize=9)
ax = axes[1]
ax.errorbar(x, [np.mean(r["k8"]) for r in rows], yerr=[np.std(r["k8"], ddof=1) if len(r["k8"]) > 1 else 0 for r in rows], fmt="s-", color="#eb6834", capsize=4, label="K=8 measured")
ax.errorbar(x, [np.mean(r["ref_mean"]) * r["factor"] for r in rows], yerr=[np.std(r["ref_mean"], ddof=1) * r["factor"] if len(r["ref_mean"]) > 1 else 0 for r in rows],
            fmt="o--", color="#666", capsize=4, label="ideal = 8 x mean single-GPU rate")
for r in rows:
    lo, hi = r["ref_spread"][0]
    ax.plot([r["x"], r["x"]], [lo * r["factor"], hi * r["factor"]], color="#bbb", linewidth=6, alpha=0.6, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("particles per GPU (M)"); ax.set_ylabel("steps per second")
ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(x); ax.set_xticklabels([r["size"] for r in rows], fontsize=8); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8); ax.set_title("throughput; grey band = 8 x (slowest .. fastest GPU) reference spread", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print("saved", OUT)
