"""probe28: 64M K=8 on wqd10nbj04g2 — legacy vs cascade vs cascade+band-dispatch.
Panel 1: interval fps vs time (probe24 legacy, probe28 cascade / band / band2) + same-node 8xK=1 line.
Panel 2: per-GPU frame anatomy (A/B/C + gaps + idle) for cascade off (probe22), cascade (probe26 d2), cascade+band (probe28).
Panel 3: same-node strong-scaling efficiency.
usage: plot_band.py <logs/n56 root> <out.png>"""
import glob, json, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
def find(pattern):
    hits = sorted(glob.glob(str(root / "*" / pattern)))
    if not hits:
        raise SystemExit(f"missing {pattern}")
    return pathlib.Path(hits[-1])
p22, p24, p26, p28 = find("*probe22_*"), find("*probe24_*"), find("*probe26_*"), find("*probe28_*")

def intervals(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return np.array([r["elapsed_s"] for r in rows]), np.array([r["interval_fps"] for r in rows]), rows

def anatomy_means(path, min_frame=2000):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
        if m and int(m.group(1)) >= min_frame:
            rows.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(3))})
    g = lambda k: np.mean([r.get(k, 0) for r in rows]) / 1000 if rows else 0.0
    return {k: g(k) for k in ("phase_a", "phase_b", "phase_c", "a_to_b_gap", "b_to_c_gap", "c_to_a_gap",
                              "correction_boundary", "density", "force", "install_leading", "install_trailing")}, len(rows)

S = {"legacy": "#2a78d6", "cascade": "#eb6834", "band": "#1baf7a", "band2": "#eda100"}
runs = [("legacy (force in C)", p24 / "soak_k8_legacy" / "intervals.jsonl", S["legacy"]),
        ("cascade (force in B)", p28 / "soak_k8_cascade" / "intervals.jsonl", S["cascade"]),
        ("cascade + band dispatch #1", p28 / "soak_k8_band" / "intervals.jsonl", S["band"]),
        ("cascade + band dispatch #2", p28 / "soak_k8_band2" / "intervals.jsonl", S["band2"])]
k1_rows = intervals(p28 / "soak_k1_alone" / "intervals.jsonl")[2]
k1_alone = k1_rows[-1]["interval_fps"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1.1, 0.8]})
ax = axes[0]
means = {}
for label, path, color in runs:
    t, f, rows = intervals(path)
    m = f[1:].mean(); means[label] = (m, f[1:].min(), f[1:].max(), int(rows[-1]["drift"]))
    ax.plot(t, f, color=color, linewidth=1.6, marker="o", markersize=3, label=f"{label}: mean {m:.1f} fps")
ax.axhline(8 * k1_alone, color="#666", linestyle="--", linewidth=1, label=f"8 x same-node K=1 ({8 * k1_alone:.1f} fps)")
ax.set_xlabel("elapsed (s)"); ax.set_ylabel("interval fps (5 s samples)")
ax.set_title("64M K=8, node wqd10nbj04g2, depth 2, full host stack", fontsize=10)
ax.set_ylim(40, 80); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")

ax = axes[1]
sets = [("cascade off\n(probe22)", p22 / "anatomy_k8.log", means["legacy (force in C)"][0]),
        ("cascade\n(probe26)", p26 / "anatomy_k8_d2.log", means["cascade (force in B)"][0]),
        ("cascade + band\n(probe28)", p28 / "anatomy_k8_band.log", (means["cascade + band dispatch #1"][0] + means["cascade + band dispatch #2"][0]) / 2)]
keys = ["phase_a", "phase_b", "phase_c", "gaps"]
colors = {"phase_a": "#9bb7d4", "phase_b": "#2a78d6", "phase_c": "#eb6834", "gaps": "#bbbbbb", "other idle": "#eeeeee"}
x = np.arange(len(sets)); bottoms = np.zeros(len(sets)); bars = []
for lab, path, fps in sets:
    m, n = anatomy_means(path)
    period = 1000.0 / fps
    vals = {"phase_a": m["phase_a"], "phase_b": m["phase_b"], "phase_c": m["phase_c"],
            "gaps": m["a_to_b_gap"] + m["b_to_c_gap"] + m["c_to_a_gap"]}
    vals["other idle"] = max(0.0, period - sum(vals.values()))
    bars.append((lab, vals, period, n, m))
for key in keys + ["other idle"]:
    v = np.array([b[1][key] for b in bars])
    ax.bar(x, v, bottom=bottoms, color=colors[key], width=0.55, edgecolor="white", label=key)
    for xi, (b0, vv) in enumerate(zip(bottoms, v)):
        if vv > 0.4:
            ax.text(xi, b0 + vv / 2, f"{vv:.2f}", ha="center", va="center", fontsize=8)
    bottoms += v
for xi, (lab, vals, period, n, m) in enumerate(bars):
    ax.text(xi, period + 0.3, f"period {period:.1f} ms", ha="center", fontsize=8)
ideal = 1000.0 / k1_alone / 8
ax.axhline(ideal, color="#666", linestyle="--", linewidth=1)
ax.text(-0.42, ideal - 0.25, f"same-node K=1 frame / 8 = {ideal:.1f} ms", fontsize=7.5, ha="left", va="top", color="#444")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_ylabel("ms per frame (GPU timestamps, f2000+f3000 x 8 GPUs)"); ax.set_ylim(0, 21); ax.set_xlim(-0.6, 2.6)
ax.set_title("Per-GPU frame anatomy, 64M K=8", fontsize=10)
ax.legend(fontsize=7, loc="upper center", ncol=5, columnspacing=0.7, handlelength=1.0, frameon=False)

ax = axes[2]
labels = ["legacy", "cascade", "cascade\n+ band"]
fps_vals = [means["legacy (force in C)"][0], means["cascade (force in B)"][0],
            (means["cascade + band dispatch #1"][0] + means["cascade + band dispatch #2"][0]) / 2]
eta = [v / (8 * k1_alone) * 100 for v in fps_vals]
ax.bar(np.arange(3), eta, color=[S["legacy"], S["cascade"], S["band"]], width=0.6)
for i, (e, v) in enumerate(zip(eta, fps_vals)):
    ax.text(i, e + 0.5, f"{e:.1f}%\n{v:.1f} fps", ha="center", fontsize=8)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(70, 102)
ax.set_ylabel("strong-scaling efficiency vs same-node K=1 (%)"); ax.set_title(f"eta_strong, 64M K=8 (K=1 alone {k1_alone:.2f} fps)", fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print("saved", OUT)
for k, (m, lo, hi, d) in means.items():
    print(f"{k:28s} mean {m:5.1f}  min {lo:5.1f}  max {hi:5.1f}  drift {d}")
for lab, vals, period, n, m in bars:
    busy = vals["phase_a"] + vals["phase_b"] + vals["phase_c"]
    print(f"{lab.replace(chr(10), ' '):26s} n={n:2d} A={vals['phase_a']:.2f} B={vals['phase_b']:.2f} C={vals['phase_c']:.2f} gaps={vals['gaps']:.2f} busy={busy:.2f} period={period:.2f} idle={period-busy:.2f} ms"
          + (f" | C: corr_b={m['correction_boundary']:.3f} dens={m['density']:.3f} force={m['force']:.3f}" if m["force"] else ""))
print(f"K=1 alone same node: {k1_alone:.2f} fps; eta legacy/cascade/band = " + " / ".join(f"{e:.1f}%" for e in eta))
