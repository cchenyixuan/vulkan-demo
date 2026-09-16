"""probe24: 64M K=8 cascading-force A/B — fps vs time (3 soaks) + per-GPU frame anatomy (cascade off vs on)."""
import json, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pathlib.Path(sys.argv[1])
OUT = D / "cascade_k8.png"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

runs = [("legacy (force in C)", "soak_k8_legacy__intervals.jsonl", S[0]),
        ("cascade (force in B) #1", "soak_k8_cascade__intervals.jsonl", S[1]),
        ("cascade (force in B) #2", "soak_k8_cascade2__intervals.jsonl", S[2])]

def anatomy(path):
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
        if not m:
            continue
        kv = dict(re.findall(r"(\w+)=(-?\d+)", m.group(3)))
        if int(m.group(1)) < 2000:  # f1000 = first defrag boundary, clocks not settled (B +4 ms in both runs)
            continue
        rows.setdefault(int(m.group(2)), []).append({k: int(v) for k, v in kv.items()})
    return rows

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
summary = []
for label, fn, color in runs:
    R = [json.loads(l) for l in open(D / fn, encoding="utf-8") if l.strip()]
    t = np.array([r["elapsed_s"] for r in R]); f = np.array([r["interval_fps"] for r in R])
    m = f[1:].mean()
    summary.append((label, m, f[1:].min(), f[1:].max(), int(R[-1]["drift"])))
    ax.plot(t, f, color=color, linewidth=1.6, marker="o", markersize=3, label=f"{label}: mean {m:.1f} fps")
ax.axhline(8 * 9.29, color="#666", linestyle="--", linewidth=1, label="8 x single-GPU 64M (74.3 fps)")
ax.set_xlabel("elapsed (s)"); ax.set_ylabel("interval fps (5 s samples)")
ax.set_title("64M K=8 on hp_5090 node wqd10nbj04g2, depth 2, full host stack", fontsize=10)
ax.set_ylim(40, 80); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

# --- anatomy panel: mean over samples (f1000/f2000/f3000) and sims ---
ax = axes[1]
sets = [("cascade off\n(probe22)", D / "probe22__anatomy_k8.log", 1 / 64.1), ("cascade on\n(probe24)", D / "anatomy_k8_cascade.log", 1 / 65.1)]
names = ["phase_a", "phase_b", "phase_c", "b_to_c_gap"]
colors = {"phase_a": "#9bb7d4", "phase_b": "#2a78d6", "phase_c": "#eb6834", "b_to_c_gap": "#bbbbbb", "other idle": "#eeeeee"}
x = np.arange(len(sets))
bottoms = np.zeros(len(sets))
means = []
for i, (lab, path, period_s) in enumerate(sets):
    rows = anatomy(path)
    allrows = [r for sim in rows.values() for r in sim]
    mean = {k: np.mean([r[k] for r in allrows if k in r]) / 1000 for k in names}
    period = period_s * 1e3
    mean["other idle"] = max(0.0, period - sum(mean.values()))
    means.append((lab, mean, period, len(allrows)))
for key in names + ["other idle"]:
    vals = np.array([m[1][key] for m in means])
    ax.bar(x, vals, bottom=bottoms, color=colors[key], width=0.55, edgecolor="white", label=key)
    for xi, (b, v) in enumerate(zip(bottoms, vals)):
        if v > 0.35:
            ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8)
    bottoms += vals
for xi, (lab, mean, period, n) in enumerate(means):
    ax.text(xi, period + 0.3, f"period {period:.1f} ms", ha="center", fontsize=8)
ax.axhline(107.08 / 8, color="#666", linestyle="--", linewidth=1)
ax.text(-0.55, 107.08 / 8 - 0.2, "1/8 of single-GPU 64M frame: 13.4 ms", fontsize=7.5, ha="left", va="top", color="#444")
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in means], fontsize=9)
ax.set_ylabel("ms per frame (GPU timestamps, mean of f2000+f3000 x 8 GPUs)")
ax.set_title("Per-GPU frame anatomy, 64M K=8", fontsize=10)
ax.set_ylim(0, 21); ax.set_xlim(-0.6, 1.6); ax.legend(fontsize=7.5, loc="upper center", ncol=5, columnspacing=0.8, handlelength=1.0, frameon=False)
fig.tight_layout(); fig.savefig(OUT, dpi=140)
print("saved", OUT)
for s in summary:
    print(f"{s[0]:28s} mean {s[1]:5.1f}  min {s[2]:5.1f}  max {s[3]:5.1f}  drift {s[4]}")
for lab, mean, period, n in means:
    busy = sum(mean[k] for k in ("phase_a", "phase_b", "phase_c"))
    print(f"{lab.replace(chr(10), ' '):24s} n={n:2d} A={mean['phase_a']:.2f} B={mean['phase_b']:.2f} C={mean['phase_c']:.2f} "
          f"b_to_c={mean['b_to_c_gap']:.2f} busy={busy:.2f} period={period:.2f} idle={period-busy:.2f} ms  busy/period={busy/period*100:.1f}%")
