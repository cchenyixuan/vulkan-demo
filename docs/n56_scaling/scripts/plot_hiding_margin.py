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

fig, ax = plt.subplots(figsize=(9, 5.5))
markers = {"2-D K=8 vs size": ("o", "#2a78d6"), "2-D fixed-N K sweep": ("s", "#eb6834"), "3-D stretched": ("^", "#2f9e5c")}
for group, (mk, color) in markers.items():
    ps = [p for p in points if p["group"] == group]
    if not ps:
        continue
    ax.errorbar([p["x"] for p in ps], [100 * p["eta"] for p in ps], yerr=[100 * p["std"] for p in ps],
                xerr=[[p["x"] - p["margin_p10"] for p in ps], [0 for p in ps]],
                fmt=mk, color=color, capsize=3, markersize=7, linestyle="none", label=group)
    for p in ps:
        ax.annotate(p["label"].replace("2-D ", "").replace("3-D ", ""), (p["x"], 100 * p["eta"]), textcoords="offset points", xytext=(6, -3 if "3-D" in p["label"] else 4), fontsize=7, color=color)
ax.axvline(0, color="#888", linestyle=":", linewidth=1)
ax.set_xscale("symlog", linthresh=1.0, linscale=2.0); ax.set_xlim(-8, 1.05)
ax.set_xticks([-6, -3, -1, -0.5, 0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["−6", "−3", "−1", "−0.5", "0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel("hiding margin (T_B − t_transport) / T_B of the binding link, p50 over frames (x-bar to p10); < 0: phase C waited")
ax.set_ylabel("standardized strong-scaling efficiency η (%)"); ax.grid(alpha=0.3)
ax.set_title("All strong-scaling points vs their transport hiding margin (one node, 8× RTX 5090)", fontsize=9)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(out_dir / "eta_vs_hiding_margin.png", dpi=140); print("saved", out_dir / "eta_vs_hiding_margin.png")
json.dump(points, open(out_dir / "hiding_margin_points.json", "w"), indent=1)
