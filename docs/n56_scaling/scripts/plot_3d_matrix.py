"""3-D matrix (probe37): eta_strong vs K (64M / 32M stretched), eta_weak vs K (8M / 4M per GPU), cube control.

usage: plot_3d_matrix.py <logs/n56> <out_dir>
Reads logs/n56/<day>/<seq>_probe37_3d_<job>/probe37_3d_<job>.out (RESULT lines) and the per-run logs in
probe37_<job>/ for a precise steady fps (steps / seconds; the RESULT line's fps has one decimal, which at
1.6 fps is a 3% quantum) and the f1000 anatomy. Reference: one 8-GPU simultaneous K=1 set per N;
eta_strong(N, K) = fps_K / (K * mean ref over GPUs 0..K-1); eta_weak(K) = fps(weak_kK) / mean ref(GPUs 0..K-1);
the weak families' K=8 points are the strong blocks' K=8 runs (same case, same K)."""
import collections, glob, json, pathlib, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

root, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
runs = {}      # label -> dict(fps, drift, stamps, wall, log)
anatomy = {}   # label -> {sim: {key: us}}
for out in sorted(glob.glob(str(root / "*" / "*" / "probe37_3d_*.out"))):
    folder = pathlib.Path(out).parent
    job = re.search(r"probe37_3d_(\d+)\.out", out).group(1)
    print("job", job, folder.name)
    for line in open(out, encoding="utf-8", errors="replace"):
        m = re.match(r"RESULT label=(\S+) rc=(\d+) steady_fps=(\S+) drift=(\S+) stamps=(\S+) particles=(\S+) wall=(\d+)s", line)
        if not m or m.group(2) != "0":
            continue
        label = m.group(1)
        log = folder / f"{label}.log"                      # organize_n56_logs flattens the per-run dir
        if not log.exists():
            log = folder / f"probe37_{job}" / f"{label}.log"
        fps = float(m.group(3))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            s = re.search(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s", text)
            if s:
                fps = int(s.group(1)) / float(s.group(2))
            a = collections.defaultdict(dict)
            for am in re.finditer(r"\[anatomy\] f(\d+) s(\d+): (.*)", text):
                if int(am.group(1)) >= 1000:
                    a[int(am.group(2))].update({k: float(v) for k, v in (p.split("=") for p in am.group(3).split())})
            anatomy[label] = dict(a)
        runs[label] = dict(fps=fps, drift=int(m.group(4)), stamps=m.group(5), wall=int(m.group(7)), particles=m.group(6))

def ref_mean(block, K):
    vals = [runs[f"{block}_ref_g{g}"]["fps"] for g in range(K) if f"{block}_ref_g{g}" in runs]
    return (np.mean(vals), vals) if len(vals) == K else (None, vals)

def kruns(block, K):
    return [runs[l]["fps"] for l in sorted(runs) if re.fullmatch(rf"{block}_k{K}_run_t\d+", l)]

table = {}
print(f"\n{'block':10s} {'K':>2s} {'per GPU':>8s} {'K-run fps (trials)':>26s} {'ref mean (range)':>22s} {'eta':>7s} {'+-':>5s} {'eta_min':>8s}")
for block, total in (("strong64m", 70.63), ("strong32m", 35.2)):
    for K in (2, 4, 8):
        f = kruns(block, K); r, rv = ref_mean(block, K)
        if not f or r is None:
            continue
        eta = [x / (K * r) for x in f]; eta_min = [x / (K * min(rv)) for x in f]
        table[f"{block}_k{K}"] = dict(K=K, fps=f, ref=rv, eta=float(np.mean(eta)), std=float(np.std(eta, ddof=1)) if len(eta) > 1 else 0.0, eta_min=float(np.mean(eta_min)))
        print(f"{block:10s} {K:2d} {total / K:7.1f}M {str([round(x, 2) for x in f]):>26s} {r:8.3f} ({min(rv):.3f}-{max(rv):.3f}) {100 * np.mean(eta):6.1f}% {100 * table[f'{block}_k{K}']['std']:4.1f}% {100 * np.mean(eta_min):7.1f}%")
for block, strong in (("weak8", "strong64m"), ("weak4", "strong32m")):
    for K in (2, 4, 8):
        f = kruns(strong, 8) if K == 8 else kruns(block, K); r, rv = ref_mean(block, K)
        if not f or r is None:
            continue
        eta = [x / r for x in f]; eta_min = [x / min(rv) for x in f]
        table[f"{block}_k{K}"] = dict(K=K, fps=f, ref=rv, eta=float(np.mean(eta)), std=float(np.std(eta, ddof=1)) if len(eta) > 1 else 0.0, eta_min=float(np.mean(eta_min)))
        print(f"{block:10s} {K:2d} {'':>8s} {str([round(x, 2) for x in f]):>26s} {r:8.3f} ({min(rv):.3f}-{max(rv):.3f}) {100 * np.mean(eta):6.1f}% {100 * table[f'{block}_k{K}']['std']:4.1f}% {100 * np.mean(eta_min):7.1f}%")
cube = kruns("cube64m", 8)
if cube:
    print(f"cube64m K=8 fps {[round(x, 2) for x in cube]} vs strong64m K=8 {[round(x, 2) for x in kruns('strong64m', 8)]}")

# anatomy per block/K (mean over sims and trials, f1000)
print("\nanatomy (ms, mean over GPUs and trials, f1000):")
anat = {}
for key in sorted(set(re.sub(r"_t\d+$", "", l) for l in anatomy)):
    rows = [v for l, sims in anatomy.items() if re.sub(r"_t\d+$", "", l) == key for v in sims.values()]
    if not rows:
        continue
    def mean(k):
        vals = [r[k] for r in rows if k in r]; return np.mean(vals) / 1000 if vals else float("nan")
    # the anatomy line prints keys without the _us suffix (phase_a=245 ...)
    anat[key] = {k: mean(k) for k in ("phase_a", "phase_b", "phase_c", "b_to_c_gap", "c_to_a_gap", "correction_boundary", "density", "force", "upload_leading_to_c_gap", "upload_trailing_to_c_gap")}
    a = anat[key]
    print(f"  {key:22s} A {a['phase_a']:6.2f} B {a['phase_b']:7.2f} C {a['phase_c']:6.2f}  C/B {a['phase_c'] / a['phase_b'] if a['phase_b'] else float('nan'):.3f}  b->c gap {a['b_to_c_gap']:5.2f} c->a {a['c_to_a_gap']:5.2f}  slack lead/trail {a['upload_leading_to_c_gap']:6.1f}/{a['upload_trailing_to_c_gap']:6.1f}  n={len(rows)}")

# figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
colors = {"strong64m": "#2a78d6", "strong32m": "#eb6834", "weak8": "#2a78d6", "weak4": "#eb6834"}
for ax, blocks, title, ylabel in ((axes[0], (("strong64m", "64M stretched (8M/GPU at K=8)"), ("strong32m", "32M stretched (4M/GPU at K=8)")), "3-D strong scaling, fixed N, one node", "η_strong (%)"),
                                  (axes[1], (("weak8", "8M per GPU"), ("weak4", "4M per GPU")), "3-D weak scaling", "η_weak (%)")):
    for block, label in blocks:
        ks = [1] + [K for K in (2, 4, 8) if f"{block}_k{K}" in table]
        if len(ks) == 1:
            continue
        etas = [100.0] + [100 * table[f"{block}_k{K}"]["eta"] for K in ks[1:]]
        errs = [0.0] + [100 * table[f"{block}_k{K}"]["std"] for K in ks[1:]]
        emin = [100.0] + [100 * table[f"{block}_k{K}"]["eta_min"] for K in ks[1:]]
        ax.errorbar(ks, etas, yerr=errs, fmt="o-", color=colors[block], capsize=4, linewidth=1.6, markersize=6, label=label + ", η_mean")
        ax.plot(ks, emin, "s--", color=colors[block], linewidth=1, markersize=4, alpha=0.7, label=label + ", η_min")
        for k, e in zip(ks[1:], etas[1:]):
            ax.annotate(f"{e:.1f}%", (k, e), textcoords="offset points", xytext=(0, 8 if "64" in block or "8" in block[-1] else -14), ha="center", fontsize=8, color=colors[block])
    ax.axhline(100, color="#888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_ylim(85, 101.5)
    ax.set_xlabel("GPUs (K)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="lower left"); ax.set_title(title, fontsize=9)
if cube and kruns("strong64m", 8):
    axes[0].annotate(f"cube 401³ control, K=8: {np.mean(cube):.2f} fps vs stretched 64M {np.mean(kruns('strong64m', 8)):.2f} fps",
                     (0.98, 0.04), xycoords="axes fraction", fontsize=8, ha="right")
fig.tight_layout(); fig.savefig(out_dir / "eta_3d_matrix.png", dpi=140); print("saved", out_dir / "eta_3d_matrix.png")
json.dump({"table": table, "cube64m_k8_fps": cube, "runs": runs, "anatomy_ms": anat}, open(out_dir / "matrix_3d_table.json", "w"), indent=1, default=float)
