"""Items 1-2: linear fits of the K = 1 step time against the particle classes.
3-D: the cluster's eight-way K = 1 references of the five cubes (phase-2 index.csv), 76M as an out-of-sample check.
2-D: the E7 batch-2 reference members of the single-cavity K = 8 points (mirrors' results.jsonl).
Model A: T = c_f N_fluid + c_w N_wall (walls incl. lid).  Model B: T = c_f N_fluid + c_n N_near (fluid within one h of a wall)."""
import csv, json, pathlib, statistics
import numpy as np

E8 = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
LOGS = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56")

# ---- 3-D: counts from generate.txt (fluid n^3, wall, lid); near-wall = fluid within h = 4 dx of any of the 6 faces
CUBES = {"cavity3d_n152": (152, 3511808, 491776, 92416), "cavity3d_n192": (192, 7077888, 774656, 147456),
         "cavity3d_n244": (244, 14526784, 1238080, 238144), "cavity3d_n312": (312, 30371328, 2007296, 389376),
         "cavity3d_n392": (392, 60236288, 3149056, 614656), "cavity3d_n416": (416, 71991296, 3541504, 692224)}
HDX3 = 4
rows = list(csv.DictReader(open(E8 / "docs/cluster_v6/data/e8/phase2/index.csv", encoding="utf-8")))
T3 = {}
for case in CUBES:
    point = "3d_n%s_K8" % case.split("_n")[1]
    fps = [float(r["fps_nodefrag"]) for r in rows if r["point"] == point and r["role"] == "reference" and r["status"] == "pass" and r["case"] == case]
    if fps:
        T3[case] = (statistics.mean(1000.0 / f for f in fps), len(fps), min(fps), max(fps))

# ---- 2-D: E7 batch-2 mirrors, reference members of the single-cavity strong K = 8 points
CAV = {"cavity2d_n520": (520, 270400, 17644, 5720), "cavity2d_n1000": (1000, 1000000, 33484, 11000),
       "cavity2d_n1440": (1440, 2073600, 48004, 15840), "cavity2d_n2000": (2000, 4000000, 66484, 22000),
       "cavity2d_n2840": (2840, 8065600, 94204, 31240), "cavity2d_n4000": (4000, 16000000, 132484, 44000),
       "cavity2d_n5640": (5640, 31809600, 186604, 62040), "cavity2d_n8000": (8000, 64000000, 264484, 88000)}
HDX2 = 5
T2, src2 = {}, {}
for mirror in sorted(LOGS.glob("2026-10-*/*_e7_b2*")):
    res = mirror / "results.jsonl"
    if not res.exists():
        continue
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        label = r.get("label", "")
        if "_R1_" not in label or r.get("status") != "pass" or r.get("steady_fps_nodefrag") is None:
            continue
        point = label.split("_t")[0]            # 2d_n520_K8 / x_2d_n8000_K8
        if "_k" in point:                        # weak boxes
            continue
        case = "cavity2d_n" + point.split("_n")[1].split("_")[0]
        if case not in CAV:
            continue
        T2.setdefault(case, []).append(1000.0 / float(r["steady_fps_nodefrag"]))
        src2.setdefault(case, set()).add(mirror.name)

def fit(name, data, counts, hdx, dim, check=()):
    cases = [c for c in counts if c in data and c not in check]
    def near(n):
        k = hdx  # layers within one h: cell centres at (i + 1/2) dx, i < h/dx
        return n ** dim - (n - 2 * k) ** dim
    def design(c, model):
        n, nf, nw, nl = counts[c]
        return [nf, (nw + nl) if model == "A" else near(n)]
    out = [f"### {name}: {len(cases)} points" + (f" (+ check {', '.join(check)})" if check else "")]
    for model, label in (("A", "T = c_f N_fluid + c_w N_wall"), ("B", "T = c_f N_fluid + c_n N_near(1 h)")):
        X = np.array([design(c, model) for c in cases], dtype=float)
        y = np.array([data[c][0] if isinstance(data[c], tuple) else statistics.mean(data[c]) for c in cases])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        rel = (pred - y) / y
        out.append(f"\n**Model {model}: {label}** -> c_f = {coef[0] * 1e6:.4f} ns, {'c_w' if model == 'A' else 'c_n'} = {coef[1] * 1e6:.4f} ns"
                   f" (ratio {coef[1] / coef[0]:.3f}); rms rel. residual {np.sqrt(np.mean(rel ** 2)) * 100:.2f} %, max {np.max(np.abs(rel)) * 100:.2f} %")
        out.append("| case | N_fluid | N_second | T measured (ms) | n | T fit (ms) | residual (ms) | residual % |")
        out.append("|---|---|---|---|---|---|---|---|")
        for c, p, yy in zip(cases, pred, y):
            n_runs = data[c][1] if isinstance(data[c], tuple) else len(data[c])
            out.append(f"| {c} | {design(c, model)[0]:,} | {design(c, model)[1]:,} | {yy:.3f} | {n_runs} | {p:.3f} | {p - yy:+.3f} | {(p - yy) / yy * 100:+.2f} % |")
        for c in check:
            if c in data:
                d = np.array(design(c, model)); yy = data[c][0] if isinstance(data[c], tuple) else statistics.mean(data[c]); p = float(d @ coef)
                out.append(f"| {c} (check) | {d[0]:,.0f} | {d[1]:,.0f} | {yy:.3f} | {data[c][1] if isinstance(data[c], tuple) else len(data[c])} | {p:.3f} | {p - yy:+.3f} | {(p - yy) / yy * 100:+.2f} % |")
    return "\n".join(out)

print(fit("3-D cubes, cluster K = 1 references (ms per step, mean of 1000/fps_nodefrag over members and trials)", T3, CUBES, HDX3, 3, check=("cavity3d_n416",)))
print()
print(fit("2-D cavities, E7 batch-2 K = 1 references", T2, CAV, HDX2, 2))
print("\n2-D sources:", {c: sorted(v) for c, v in src2.items()})
print("2-D runs per case:", {c: len(v) for c, v in T2.items()})
