"""Unified hiding-margin figure: every strong-scaling point on one axis.

x = (T_B - t_transport) / T_B, with t_transport taken from the measured slack: each link's upload lands
slack = c_start - upload_end before phase C, so t_transport = T_B - slack and x = slack_p50 / T_B
(the p50 over all links, GPUs and sampled frames of that point; negative = transport exposed).
y = standardized eta (mean over trials, error bar = std). Points:
  2-D K=8 vs size (probe34 eta; anatomy from probe36 for 64M/32M and probe38 for 4M/16M/128M),
  2-D fixed-N K sweep 64M/32M K=2/4/8 (probe36), 3-D 64M/32M K=2/4/8 (probe37).
usage: plot_hiding_margin.py <logs/n56> <out_dir>"""
import glob, json, pathlib, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
DOCS = pathlib.Path(__file__).resolve().parent.parent      # docs/n56_scaling (ksweep_table.json, matrix_3d_table.json)


def find(pattern):
    return [pathlib.Path(p) for p in sorted(glob.glob(str(root / "*" / "*" / pattern)))]


def slack_and_tb(files, min_frame):
    """Per sampled frame and GPU: the binding link's margin. The recorded slack
    (c_start - upload_end) is clamped at 0 because phase C waits for the upload, so
    an exposed transport shows up as slack 0 plus a b->c gap. Margin per frame =
    (min over links of slack - b_to_c_gap) / T_B; t_transport = T_B - margin*T_B."""
    slack, tb, bc, margin = [], [], [], []
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            m = re.match(r"\[anatomy\] f(\d+) s(\d+): (.*)", line)
            if not m or int(m.group(1)) < min_frame:
                continue
            d = {k: float(v) for k, v in (p.split("=") for p in m.group(3).split())}
            links = [d[k] / 1000 for k in ("upload_leading_to_c_gap", "upload_trailing_to_c_gap") if k in d]
            if not links or "phase_b" not in d:
                continue
            slack += links
            tb.append(d["phase_b"] / 1000)
            gap = d.get("b_to_c_gap", 0.0) / 1000
            bc.append(gap)
            margin.append((min(links) - gap) / (d["phase_b"] / 1000))
    if not slack or not tb:
        return None
    return dict(slack_p50=float(np.median(slack)), slack_p10=float(np.percentile(slack, 10)), slack_min=float(np.min(slack)),
                T_B=float(np.mean(tb)), b_to_c=float(np.mean(bc)) if bc else 0.0, n=len(margin),
                margin_p50=float(np.median(margin)), margin_p10=float(np.percentile(margin, 10)), margin_min=float(np.min(margin)))


points = []
# ---- 2-D K=8 vs size: eta from the standardized curve (probe34 v2, 3 trials), anatomy from probe36 / probe38
eta_2d_size = {"4m": (0.108, 0.014, 0.52), "16m": (0.441, 0.021, 2.05), "32m": (0.674, 0.023, 4.06), "64m": (0.914, 0.003, 8.05), "128m": (0.954, 0.002, 16.0)}
for size, (eta, std, per_gpu) in eta_2d_size.items():
    files = []
    for folder in find("probe36_ksweep_*.out"):
        sub = folder.parent / "probe36_1594545"; src = sub if sub.exists() else folder.parent
        files += sorted(glob.glob(str(src / f"{size}_k8_run_t*.log")))
    for folder in find("probe38_anat_*.out"):
        files += sorted(glob.glob(str(folder.parent / f"2d{size}_k8_anat.log")))
    s = slack_and_tb(files, 1000 if any("anat" in f for f in files) else 2000)
    if s:
        points.append(dict(group="2-D K=8 vs size", label=f"2-D {size.upper()} K=8", eta=eta, std=std, per_gpu=per_gpu, **s))
# ---- 2-D fixed-N K sweep (probe36) ----
for folder in find("probe36_ksweep_*.out"):
    sub = folder.parent / "probe36_1594545"; src = sub if sub.exists() else folder.parent
    table = json.load(open(DOCS / "ksweep_table.json")) if (DOCS / "ksweep_table.json").exists() else {}
    for size in ("64m", "32m"):
        for tag in ("k2", "k4"):
            key = f"{size}_{tag}"
            if key not in table:
                continue
            s = slack_and_tb(sorted(glob.glob(str(src / f"{size}_{tag}_run_t*.log"))), 2000)
            if s:
                r = table[key]
                points.append(dict(group="2-D fixed-N K sweep", label=f"2-D {size.upper()} K={r['K']}", eta=r["eta"], std=r["std"], per_gpu={"64m": 64.4, "32m": 32.5}[size] / r["K"], **s))
# ---- 3-D (probe37) ----
for folder in find("probe37_3d_*.out"):
    table = json.load(open(DOCS / "matrix_3d_table.json"))["table"]
    for block, total in (("strong64m", 70.6), ("strong32m", 35.7)):
        for K in (2, 4, 8):
            key = f"{block}_k{K}"
            if key not in table:
                continue
            s = slack_and_tb(sorted(glob.glob(str(folder.parent / f"{block}_k{K}_run_t*.log"))), 1000)
            if s:
                r = table[key]
                points.append(dict(group="3-D stretched", label=f"3-D {block[6:].upper()} K={K}", eta=r["eta"], std=r["std"], per_gpu=total / K, **s))

print(f"{'point':18s} {'eta':>6s} {'T_B ms':>8s} {'slack p50':>9s} {'b->c':>6s} {'margin p50':>10s} {'p10':>7s} {'min':>7s} {'n':>4s}")
for p in points:
    p["x"] = p["margin_p50"]
    print(f"{p['label']:18s} {100 * p['eta']:5.1f}% {p['T_B']:8.2f} {p['slack_p50']:9.2f} {p['b_to_c']:6.2f} {p['margin_p50']:10.3f} {p['margin_p10']:7.3f} {p['margin_min']:7.3f} {p['n']:4d}")

# ---- x on a log axis: r = t_transport / T_B = 1 - margin (r < 1: hidden, r > 1: exposed) ----
for p in points:
    p["r"] = 1.0 - p["margin_p50"]
    p["K"] = int(re.search(r"K=(\d+)", p["label"]).group(1))
    p["dim"] = "3-D" if "3-D" in p["label"] else "2-D"
# ideal period per point (ms): T_ref / K from the references
T_ref = {"2-D 64M": 110.8, "2-D 32M": 1000 / 18.2, "2-D 128M": 4 / 8.95 * 1000, "2-D 16M": 1000 / 36.0, "2-D 4M": 1000 / 140.8,
         "3-D 64M": 1000 / 1.617, "3-D 32M": 1000 / 3.20}
for p in points:
    key = " ".join(p["label"].split()[:2])
    p["T_ideal"] = T_ref[key] / 8 if "128M" in key else T_ref[key] / p["K"]
    p["excess_ms"] = p["T_ideal"] * (1 / p["eta"] - 1)

# ---- hidden-regime model: excess ms per frame = alpha_dim + beta_dim * (K - 1); eta = T_ideal / (T_ideal + excess)
fit = {}
hidden = [p for p in points if p["r"] < 0.85]          # 2-D 32M K=8 (r = 0.75 but p10 exposed) is kept out below
hidden = [p for p in hidden if p["label"] != "2-D 32M K=8"]
# model A: excess = alpha + beta (K-1)                      (constant floor + chain length)
# model B: excess = alpha + beta (K-1) + gamma T_ideal     (+ a share of the frame: band kernels grow with the slab face, gaps with the period)
for dim in ("2-D", "3-D"):
    ps = [p for p in hidden if p["dim"] == dim]
    y = np.array([p["excess_ms"] for p in ps])
    A1 = np.array([[1.0, p["K"] - 1] for p in ps]); c1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    A2 = np.array([[1.0, p["K"] - 1, p["T_ideal"]] for p in ps]); c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    fit[dim] = dict(A=dict(alpha_ms=float(c1[0]), beta_ms=float(c1[1])), B=dict(alpha_ms=float(c2[0]), beta_ms=float(c2[1]), gamma=float(c2[2])), n=len(ps))
    for p in ps:
        exA = c1[0] + c1[1] * (p["K"] - 1); exB = c2[0] + c2[1] * (p["K"] - 1) + c2[2] * p["T_ideal"]
        p["eta_fitA"] = p["T_ideal"] / (p["T_ideal"] + exA); p["residA"] = p["eta"] - p["eta_fitA"]
        p["eta_fit"] = p["T_ideal"] / (p["T_ideal"] + exB); p["resid"] = p["eta"] - p["eta_fit"]
print("\nhidden-regime fits (eta = T_ideal / (T_ideal + excess), excess in ms per frame):")
for dim, f in fit.items():
    print(f"  {dim} model A: excess = {f['A']['alpha_ms']:.2f} + {f['A']['beta_ms']:.2f} (K-1)                         n = {f['n']}")
    print(f"  {dim} model B: excess = {f['B']['alpha_ms']:.2f} + {f['B']['beta_ms']:.2f} (K-1) + {100 * f['B']['gamma']:.2f}% * T_ideal")
print(f"{'point':14s} {'r=t/T_B':>8s} {'T_ideal':>8s} {'excess':>7s} {'eta':>6s} {'fitA':>6s} {'resA':>6s} {'fitB':>6s} {'resB':>6s} {'std':>6s} {'flag(B)':>7s}")
outliers, outliersA = [], []
for p in points:
    if "eta_fit" not in p:
        print(f"{p['label']:14s} {p['r']:8.3f} {p['T_ideal']:8.2f} {p['excess_ms']:7.2f} {100 * p['eta']:5.1f}%  (not fitted: transport exposed)")
        continue
    flag = "OUT" if abs(p["resid"]) > max(p["std"], 0.002) else ""
    if flag:
        outliers.append(p["label"])
    if abs(p["residA"]) > max(p["std"], 0.002):
        outliersA.append(p["label"])
    print(f"{p['label']:14s} {p['r']:8.3f} {p['T_ideal']:8.2f} {p['excess_ms']:7.2f} {100 * p['eta']:5.1f}% {100 * p['eta_fitA']:5.1f}% {100 * p['residA']:+5.2f} {100 * p['eta_fit']:5.1f}% {100 * p['resid']:+5.2f} {100 * p['std']:5.1f}% {flag:>7s}")
print("model A residual beyond the error bar:", outliersA or "none")
print("model B residual beyond the error bar:", outliers or "none")

fig, (ax, axz) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
markers = {"2-D K=8 vs size": ("o", "#2a78d6"), "2-D fixed-N K sweep": ("s", "#eb6834"), "3-D stretched": ("^", "#2f9e5c")}
for group, (mk, color) in markers.items():
    ps = [p for p in points if p["group"] == group]
    ax.errorbar([p["r"] for p in ps], [100 * p["eta"] for p in ps], yerr=[100 * p["std"] for p in ps],
                fmt=mk, color=color, capsize=3, markersize=7, linestyle="none", label=group)
    for p in ps:
        if p["r"] > 0.7:
            ax.annotate(p["label"], (p["r"], 100 * p["eta"]), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=color)
    pz = [p for p in ps if "eta_fit" in p]
    if pz:
        axz.errorbar([p["r"] for p in pz], [100 * p["eta"] for p in pz], yerr=[100 * p["std"] for p in pz],
                     fmt=mk, color=color, capsize=3, markersize=7, linestyle="none", label=group + " (measured)")
        axz.plot([p["r"] for p in pz], [100 * p["eta_fit"] for p in pz], mk, color=color, mfc="white", markersize=7, linestyle="none",
                 label=group + " (model B)")
        for p in pz:
            axz.annotate(p["label"].replace("2-D ", "").replace("3-D ", "") + (" 3-D" if p["dim"] == "3-D" else ""),
                         (p["r"], 100 * p["eta"]), textcoords="offset points", xytext=(6, 3 if p["resid"] >= 0 else -9), fontsize=7, color=color)
ax.axvline(1, color="#888", linestyle=":", linewidth=1); ax.text(1.04, 50, "t_transport = T_B", rotation=90, fontsize=8, color="#666", va="center")
ax.set_xscale("log"); ax.set_xlim(0.09, 12); ax.set_ylim(0, 102)
ax.set_xlabel("t_transport / T_B of the binding link (log; p50 over sampled frames)")
ax.set_ylabel("standardized strong-scaling efficiency η (%)"); ax.grid(alpha=0.3, which="both")
ax.set_title("All 15 strong-scaling points: hidden (r < 1) vs exposed (r > 1) transport", fontsize=9)
ax.legend(fontsize=8, loc="lower left")
axz.set_xscale("log"); axz.set_xlim(0.1, 0.8); axz.set_ylim(90, 99.5); axz.grid(alpha=0.3, which="both")
axz.set_xlabel("t_transport / T_B (hidden regime, log)")
fb2, fb3 = fit["2-D"]["B"], fit["3-D"]["B"]
axz.set_title(f"Hidden regime: measured vs model B  excess = α + β(K−1) + γ·T_ideal\n2-D: α {fb2['alpha_ms']:.2f} ms, β {fb2['beta_ms']:.2f} ms, γ {100 * fb2['gamma']:.2f}%   3-D: α {fb3['alpha_ms']:.2f}, β {fb3['beta_ms']:.2f}, γ {100 * fb3['gamma']:.2f}%", fontsize=8)
axz.legend(fontsize=6.5, loc="lower left", ncol=2)
json.dump({"fit": fit, "outliers": outliers}, open(out_dir / "hiding_margin_fit.json", "w"), indent=1)
fig.tight_layout(); fig.savefig(out_dir / "eta_vs_hiding_margin.png", dpi=140); print("saved", out_dir / "eta_vs_hiding_margin.png")
json.dump(points, open(out_dir / "hiding_margin_points.json", "w"), indent=1)
