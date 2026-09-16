"""Item 2 (CPU submit path) campaign figure: 64M K=8 on wqd10nbj04g2.
Panel 1: mean interval fps per configuration (chronological), same-node 8xK=1 line.
Panel 2: per-GPU frame gap anatomy (a_to_b / b_to_c / c_to_a means, f2000+f3000 x 8 GPUs) per configuration.
Panel 3: CPU loop: submit ms/frame and cpu_share per configuration (from intervals.jsonl loop stats).
usage: plot_item2.py <logs/n56 root> <out.png>"""
import glob, json, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
def find(pattern):
    hits = sorted(glob.glob(str(root / "*" / pattern)))
    return pathlib.Path(hits[-1]) if hits else None
P = {k: find(f"*{k}_*") for k in ("probe24", "probe26", "probe28", "probe29", "probe30", "probe31")}

def soak_mean(path):
    if path is None or not path.exists():
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    f = np.array([r["interval_fps"] for r in rows])
    loops = [r["loop"] for r in rows[1:] if r.get("loop")]
    sub = np.mean([l["submit_ms_mean"] for l in loops]) if loops else None
    share = np.mean([l["cpu_share"] for l in loops]) if loops else None
    return f[1:].mean(), f[1:].min(), f[1:].max(), int(rows[-1]["drift"]), sub, share

def anatomy_gaps(path, min_frame=2000):
    if path is None or not path.exists():
        return None
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
        if m and int(m.group(1)) >= min_frame:
            rows.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(3))})
    if not rows:
        return None
    g = lambda k: np.mean([r.get(k, 0) for r in rows]) / 1000
    return {"a_to_b": g("a_to_b_gap"), "b_to_c": g("b_to_c_gap"), "c_to_a": g("c_to_a_gap"),
            "busy": g("phase_a") + g("phase_b") + g("phase_c"), "n": len(rows)}

configs = [  # label, soak path(s), anatomy path
    ("legacy", [P["probe24"] / "soak_k8_legacy"], None),
    ("cascade", [P["probe28"] / "soak_k8_cascade"], P["probe26"] / "anatomy_k8_d2.log" if P["probe26"] else None),
    ("band", [P["probe28"] / "soak_k8_band", P["probe28"] / "soak_k8_band2", P["probe29"] / "soak_k8_band_d2"], P["probe28"] / "anatomy_k8_band.log" if P["probe28"] else None),
    ("fast", [P["probe29"] / "soak_k8_fast_d2", P["probe29"] / "soak_k8_fast_d2b", P["probe30"] / "soak_k8_fast", P["probe30"] / "soak_k8_fast2"], P["probe29"] / "anatomy_k8_fast.log" if P["probe29"] else None),
    ("fast d3", [P["probe29"] / "soak_k8_fast_d3"], None),
    ("no-wait", [P["probe30"] / "soak_k8_nowait", P["probe30"] / "soak_k8_nowait2"], P["probe30"] / "anatomy_k8_nowait.log" if P["probe30"] else None),
    ("ready", [P["probe31"] / "soak_k8_ready", P["probe31"] / "soak_k8_ready2"] if P["probe31"] else [], P["probe31"] / "anatomy_k8_ready.log" if P["probe31"] else None),
]
k1_rows = [json.loads(l) for l in open(P["probe29"] / "soak_k1_alone" / "intervals.jsonl", encoding="utf-8") if l.strip()]
k1 = k1_rows[-1]["interval_fps"]

labels, means, spans, subs, shares, gaps = [], [], [], [], [], []
for label, soaks, anat in configs:
    stats = [s for s in (soak_mean(p / "intervals.jsonl") for p in soaks if p is not None) if s]
    if not stats:
        continue
    labels.append(label)
    m = np.mean([s[0] for s in stats]); means.append(m)
    spans.append((min(s[1] for s in stats), max(s[2] for s in stats)))
    subs.append(np.mean([s[4] for s in stats if s[4] is not None]) if any(s[4] is not None for s in stats) else np.nan)
    shares.append(np.mean([s[5] for s in stats if s[5] is not None]) if any(s[5] is not None for s in stats) else np.nan)
    gaps.append(anatomy_gaps(anat))
    print(f"{label.replace(chr(10), ' '):22s} runs={len(stats)} mean fps {m:5.1f} (min {spans[-1][0]:.1f} max {spans[-1][1]:.1f}) drift={[s[3] for s in stats]} submit={subs[-1]} share={shares[-1]} gaps={gaps[-1]}")

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0, 0.8, 1.0]})
ax = axes[0]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300"][:len(labels)]
ax.bar(x, means, color=colors, width=0.62)
for i, (m, (lo, hi)) in enumerate(zip(means, spans)):
    ax.plot([i, i], [lo, hi], color="#333", linewidth=1)
    ax.text(i, hi + 0.25, f"{m:.1f}\n{m / (8 * k1) * 100:.1f}%", ha="center", fontsize=8)
ax.axhline(8 * k1, color="#666", linestyle="--", linewidth=1)
ax.text(len(labels) - 0.5, 8 * k1 + 0.3, f"8 x same-node K=1 = {8 * k1:.1f} fps", ha="right", fontsize=8, color="#444")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(55, 76)
ax.set_ylabel("mean interval fps (bars: min/max over intervals)"); ax.set_title("64M K=8, wqd10nbj04g2: throughput per step", fontsize=10)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
keys = [("a_to_b", "#9bb7d4"), ("b_to_c", "#eb6834"), ("c_to_a", "#2a78d6")]
bottoms = np.zeros(len(labels))
for key, color in keys:
    vals = np.array([g[key] if g else 0.0 for g in gaps])
    ax.bar(x, vals, bottom=bottoms, color=color, width=0.62, label=f"{key} gap")
    bottoms += vals
for i, g in enumerate(gaps):
    ax.text(i, bottoms[i] + 0.02, f"{bottoms[i]:.2f}" if g else "n/a", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("GPU idle at phase boundaries (ms / frame)")
ax.set_title("Per-GPU inter-phase gaps (anatomy, f2000+f3000 x 8)", fontsize=10); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

ax = axes[2]
ax.bar(x, np.nan_to_num(subs), color="#888", width=0.62)
for i, (s, sh) in enumerate(zip(subs, shares)):
    if not np.isnan(s):
        ax.text(i, s + 0.05, f"{s:.2f} ms\n{sh * 100:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("CPU submit time per frame (ms) / CPU share")
ax.set_title("CPU loop cost (loop trace in soaks)", fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 4.5)

# --- light load: 4M K=4 (1M per GPU) chain-bench STEADY fps per configuration ---
def steady(path):
    if path is None or not path.exists():
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"STEADY .*= ([0-9.]+) fps", line)
        if m:
            return float(m.group(1))
    return None


light = [("band", steady(P["probe28"] / "seam_band_k4.log" if P["probe28"] else None)),
         ("fast", steady(P["probe29"] / "seam_fast_k4.log" if P["probe29"] else None)),
         ("no-wait", steady(P["probe30"] / "seam_nowait_k4.log" if P["probe30"] else None)),
         ("ready", steady(P["probe31"] / "seam_ready_k4.log" if P["probe31"] else None))]
p32 = find("*probe32_*")
if p32 is not None:
    for path in sorted(p32.glob("bench_ps*_*.log")):
        tag = "ready\n(repro)" if "bench_ps2_" in path.name else "no-wait\n(repro)"
        light.append((tag, steady(path)))
light = [(l, v) for l, v in light if v]
ax = axes[3]
xl = np.arange(len(light))
ax.bar(xl, [v for _, v in light], color=["#1baf7a" if "ready" in l else "#888" for l, _ in light], width=0.62)
for i, (l, v) in enumerate(light):
    ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(xl); ax.set_xticklabels([l for l, _ in light], fontsize=7.5)
ax.set_ylabel("steady fps (chain bench, 5000 steps)"); ax.set_title("Light load: 4M K=4 (1M per GPU)", fontsize=10)
ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(v for _, v in light) * 1.15)
print("light load:", light)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print("saved", OUT)
