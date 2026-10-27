"""Prefix-sum defrag vs default defrag (user 2026-10-27): per-particle cost of the 8M / 64M cubes at K = 1 (300 steps,
window 100-299, GPU 1 at 575 W) and the m = 8 piece profiles of the two sweep kernels, prefix defrag (host-scan
variant of prefix_defrag_run.py) against the default atomicAdd defrag (7.17's m1_plain + cd8t / fo8t runs and this
batch's controls), plus the layout audit (pid order vs voxel order after the bootstrap defrag, exact column span and
material composition of the 8 pid pieces). Reads prefix/runs/<cube>_<mode>_<cfg>_<r>/ and the 7.17 outputs in
split/; writes prefix_defrag.md / .json."""
import csv
import datetime
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/prefix")
SPLIT = ROOT.parent / "split"
RUNS = ROOT / "runs"
CUBES = {"8m": ("cube 8M (200^3)", 8_000_000), "64m": ("cube 64M (400^3)", 64_000_000)}
KERNELS = {"cd": ("correction_density_interior", "b_start", "b_correction_density_interior_end"),
           "fo": ("force_deep_interior_scratch", "b_correction_density_interior_end", "b_force_deep_interior_end")}
MATERIALS = ("fluid", "wall", "lid")          # material index order of the aligned cavity cases (audit material totals)
FIXED = json.load(open("C:/Users/cchen/PycharmProjects/vulkan-demo-e8/docs/cluster_v6/data/e8/local_size_cost/fixed_terms.json", encoding="utf-8"))
FIXED_OF = {"8m": next(r for r in FIXED["rows"] if r["item"] == "8m_base"), "64m": next(r for r in FIXED["rows"] if r["item"] == "64m_base")}
M = 8
WARMUP = 100


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s", text)
    if m:
        out["step_ms"] = 1000.0 * float(m.group(3)) / int(m.group(2))
    for stage in ("loop_start", "loop_end"):
        m = re.search(rf"stage {stage}: t=([\d.]+)s .*?epoch=([\d.]+)", text)
        if m:
            out[stage] = float(m.group(2))
    m = re.search(r"drift=(-?\d+)", text)
    out["drift"] = int(m.group(1)) if m else None
    m = re.search(r"\[prefix\] mode=(\w+) numerics.use_prefix_sum_defrag=(\w+) spec constant 46 entry=\[\(46, 'B', (\d)\)\]", text)
    out["mode"], out["flag"], out["spec46"] = (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)
    m = re.search(r"\[prefix\] defrag #0 done in ([\d.]+) ms: alive after the cmd ([\d,]+)", text)
    out["defrag0_ms"], out["alive_after_cmd"] = (float(m.group(1)), int(m.group(2).replace(",", ""))) if m else (None, None)
    m = re.search(r"readback \+ scan \+ upload ([\d.]+) ms", text)
    out["scan_ms"] = float(m.group(1)) if m else None
    m = re.search(r"\[split\] pipelines: SPLIT_CD=(\d+) SPLIT_FO=(\d+)", text)
    out["split"] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return out


def telemetry(path, t0, t1):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((stamp, float(parts[3].split()[0]), float(parts[4].split()[0])))
        except ValueError:
            continue
    window = [r for r in rows if t0 <= r[0] <= t1]
    if not window:
        return {}
    return {"power_p50_W": statistics.median(r[1] for r in window), "sm_p50_MHz": statistics.median(r[2] for r in window)}


def trace_kernels(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {"cd": [], "fo": [], "period": []}
    prev_end = None
    for r in rows:
        try:
            bs, cde, fe, ce = (int(r["b_start"]), int(r["b_correction_density_interior_end"]), int(r["b_force_deep_interior_end"]), int(r["c_end"]))
        except ValueError:
            prev_end = None
            continue
        if int(r["step"]) >= WARMUP:
            out["cd"].append((cde - bs) / 1e6)
            out["fo"].append((fe - cde) / 1e6)
            if prev_end is not None:
                out["period"].append((ce - prev_end) / 1e6)
        prev_end = ce
    return {k: statistics.median(v) for k, v in out.items() if v} | {"steps": len(out["cd"])}


def trace_pieces(path, short):
    kname, start_label, end_label = KERNELS[short]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labels = [start_label] + [f"b_split_{short}_p{i}_end" for i in range(M - 1)] + [end_label]
    missing = [l for l in labels if l not in rows[0]]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    per_piece = [[] for _ in range(M)]
    for r in rows:
        if int(r["step"]) < WARMUP:
            continue
        try:
            ticks = [int(r[l]) for l in labels]
        except ValueError:
            continue
        for i in range(M):
            per_piece[i].append((ticks[i + 1] - ticks[i]) / 1e6)
    return [statistics.median(v) for v in per_piece]


def audit_of(path):
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])      # the bootstrap defrag
    return rec


runs = {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    m = re.fullmatch(r"(8m|64m)_(host|off|raw)_(m1|cd8t|fo8t)_(r\d|ctl|check)", d.name)
    if not m or not (d / "run.log").exists():
        continue
    cube, mode, cfg, r = m.groups()
    log = parse_log(d / "run.log")
    if "step_ms" not in log:
        print(f"{d.name}: no STEADY line (skipped)")
        continue
    rec = {"cube": cube, "mode": mode, "cfg": cfg, "r": r, **log}
    if (d / "trace" / "steps_device.csv").exists():
        rec.update(trace_kernels(d / "trace" / "steps_device.csv"))
        if cfg in ("cd8t", "fo8t"):
            rec["pieces_ms"] = trace_pieces(d / "trace" / "steps_device.csv", cfg[:2])
    if "step_ms" in log and "loop_start" in log:
        rec.update(telemetry(d / "telemetry.csv", log["loop_start"] + WARMUP * log["step_ms"] / 1e3, log.get("loop_end", 1e18)))
    rec["audit"] = audit_of(d / "layout_audit.jsonl")
    runs[d.name] = rec

# 7.17 references (default defrag, same protocol, 2026-10-27 03:00-04:10)
tail = json.load(open(SPLIT / "split_tail.json", encoding="utf-8"))
ref_runs = {k: v for k, v in tail["runs"].items() if v["cfg"] == "m1_plain"}
ref_pieces = json.load(open(SPLIT / "piece_times.json", encoding="utf-8"))

lines = ["### prefix-sum defrag series: per run (kernel times = step-trace medians over steps 100-299, ms)", "",
         "| run | defrag | spec 46 | split cd/fo | step (ms) | corr+dens (ms) | force (ms) | period (ms) | SM MHz | W | drift | bootstrap defrag (ms) | host scan (ms) | pid order after defrag |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, r in runs.items():
    a = r["audit"]
    order = ("-" if not a else ("exact voxel order" if a["sorted_exactly"] else
             f"NOT voxel order: {a['descents']:,} descents, {a['pids_behind_running_max']:,} pids behind, {a['zero_voxel_id']:,} voxel-id-0 pids"))
    lines.append(f"| {name} | {r['mode']} | {r['spec46']} | {r['split'][0]}/{r['split'][1]} | {r.get('step_ms', float('nan')):.3f} | {r.get('cd', float('nan')):.3f} | "
                 f"{r.get('fo', float('nan')):.3f} | {r.get('period', float('nan')):.3f} | {r.get('sm_p50_MHz', float('nan')):.0f} | {r.get('power_p50_W', float('nan')):.0f} | "
                 f"{r['drift']} | {r['defrag0_ms']} | {r['scan_ms'] if r['scan_ms'] is not None else '-'} | {order} |")

result = {"runs": {k: {kk: vv for kk, vv in v.items() if kk != "audit"} | {"audit_summary": None if not v["audit"] else {kk: v["audit"][kk] for kk in ("descents", "pids_behind_running_max", "voxel_blocks", "nonempty_voxels", "sorted_exactly", "zero_voxel_id")}}
                    for k, v in runs.items()}, "cost": {}, "pieces": {}}

lines += ["", "### per-particle cost: prefix defrag vs default defrag (K = 1, 300 steps window 100-299, 575 W)", "",
          "| cube | defrag | runs | step (ms) | ns / particle | corr+dens (ms) | ns / particle | force (ms) | ns / particle | SM MHz |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for cube, (label, n) in CUBES.items():
    groups = {"default (7.17 m1_plain)": [r for r in ref_runs.values() if r["cube"] == cube],
              "default (this batch, control)": [r for r in runs.values() if r["cube"] == cube and r["mode"] == "off" and r["cfg"] == "m1"],
              "prefix (host scan)": [r for r in runs.values() if r["cube"] == cube and r["mode"] == "host" and r["cfg"] == "m1"]}
    means = {}
    for gname, sel in groups.items():
        if not sel:
            continue
        mean = {k: statistics.mean(r[k] for r in sel) for k in ("step_ms", "cd", "fo")}
        mhz = statistics.mean(r.get("sm_p50_MHz", float("nan")) for r in sel)
        means[gname] = mean | {"n_runs": len(sel), "sm_MHz": mhz, "runs": [f"{r['step_ms']:.3f}" for r in sel]}
        lines.append(f"| {label} | {gname} | {len(sel)} ({', '.join(means[gname]['runs'])}) | {mean['step_ms']:.3f} | {mean['step_ms'] / n * 1e6:.3f} | "
                     f"{mean['cd']:.3f} | {mean['cd'] / n * 1e6:.3f} | {mean['fo']:.3f} | {mean['fo'] / n * 1e6:.3f} | {mhz:.0f} |")
    default = [g for g in ("default (7.17 m1_plain)", "default (this batch, control)") if g in means]
    if default and "prefix (host scan)" in means:
        d_all = [r for g in default for r in groups[g]]
        dmean = {k: statistics.mean(r[k] for r in d_all) for k in ("step_ms", "cd", "fo")}
        p = means["prefix (host scan)"]
        delta = {k: p[k] - dmean[k] for k in ("step_ms", "cd", "fo")}
        fixed = FIXED_OF[cube]
        lines.append(f"| {label} | **prefix - default (all default runs)** | {len(d_all)} default | **{delta['step_ms']:+.3f}** ({delta['step_ms'] / dmean['step_ms'] * 100:+.1f} %) | "
                     f"{delta['step_ms'] / n * 1e6:+.3f} | **{delta['cd']:+.3f}** ({delta['cd'] / dmean['cd'] * 100:+.1f} %; 7.16 fixed term {fixed['fixed_corr_dens_ms']:+.2f}) | "
                     f"{delta['cd'] / n * 1e6:+.3f} | **{delta['fo']:+.3f}** ({delta['fo'] / dmean['fo'] * 100:+.1f} %; 7.16 fixed term {fixed['fixed_force_ms']:+.2f}) | {delta['fo'] / n * 1e6:+.3f} | |")
        result["cost"][cube] = {"n": n, "groups": means, "default_mean": dmean, "delta": delta,
                                "delta_pct": {k: delta[k] / dmean[k] * 100 for k in delta},
                                "fixed_terms_716": {"cd": fixed["fixed_corr_dens_ms"], "fo": fixed["fixed_force_ms"]}}

lines += ["", "### layout audit after the bootstrap defrag (position_voxel_id read back; own pids in order; pieces = 8 equal pid ranges)", ""]
for cube, (label, n) in CUBES.items():
    for mode, mname in (("off", "default defrag (atomicAdd order)"), ("host", "prefix defrag (voxel-id order)")):
        sel = [r for r in runs.values() if r["cube"] == cube and r["mode"] == mode and r["audit"]]
        if not sel:
            continue
        a = sel[0]["audit"]
        lines.append(f"**{label}, {mname}** ({sel[0]['cube']}_{mode}, {len(sel)} audited runs; first run shown): {a['live']:,} live pids, "
                     f"{a['voxel_blocks']:,} contiguous voxel blocks for {a['nonempty_voxels']:,} non-empty voxels, {a['descents']:,} block transitions go to a lower voxel id, "
                     f"{a['pids_behind_running_max']:,} pids ({a['pids_behind_running_max'] / a['live'] * 100:.1f} %) sit behind an earlier pid's voxel -> "
                     f"**{'exact voxel order' if a['sorted_exactly'] else 'not voxel order'}**; material totals fluid {a['material_totals'][0]:,} wall {a['material_totals'][1]:,} lid {a['material_totals'][2]:,}")
        lines += ["", "| piece | pids | x columns covered (first..last in pid order; min..max) | distinct columns | fluid | wall | lid | wall share |", "|---|---|---|---|---|---|---|---|"]
        for p in a["pieces"]:
            f_, w_, l_ = p["material_counts"][:3]
            lines.append(f"| {p['piece']} | {p['range'][0]:,}..{p['range'][1]:,} | {p['x_first']}..{p['x_last']} ({p['x_min']}..{p['x_max']}) | {len(p['columns'])} | "
                         f"{f_:,} | {w_:,} | {l_:,} | {w_ / (p['range'][1] - p['range'][0]) * 100:.1f} % |")
        lines.append("")
        result["pieces"].setdefault(cube, {})[f"audit_{mode}"] = {"descents": a["descents"], "behind": a["pids_behind_running_max"], "blocks": a["voxel_blocks"],
                                                                 "nonempty": a["nonempty_voxels"], "sorted_exactly": a["sorted_exactly"], "pieces": a["pieces"], "run": sel[0]["cube"] + "_" + mode}

lines += ["### m = 8 piece durations: prefix defrag vs default defrag (ms; mean over two runs of the per-step medians over steps 100-299; piece 0 = lowest pids)", ""]
for cube, (label, n) in CUBES.items():
    for short, (kname, _, _) in KERNELS.items():
        sel = [r for r in runs.values() if r["cube"] == cube and r["mode"] == "host" and r["cfg"] == f"{short}8t" and "pieces_ms" in r]
        if not sel:
            continue
        prefix = [statistics.mean(r["pieces_ms"][i] for r in sel) for i in range(M)]
        default = ref_pieces[cube]["pieces"].get(short, {}).get("mean_ms")
        audit = next((r["audit"] for r in runs.values() if r["cube"] == cube and r["mode"] == "host" and r["audit"]), None)
        lines += [f"**{label}, {kname}** (prefix: {', '.join(r['cube'] + '_host_' + r['cfg'] + '_' + r['r'] for r in sel)}; default: 7.17 {', '.join(ref_pieces[cube]['pieces'][short]['runs'])})", "",
                  "| piece | x columns (prefix, exact) | pids | prefix (ms) | prefix per 1M (ms) | default (ms) | default per 1M (ms) | prefix - default |", "|---|---|---|---|---|---|---|---|"]
        for i in range(M):
            p = audit["pieces"][i] if audit else None
            cols = f"{p['x_first']}..{p['x_last']}" if p else "-"
            pids = (p["range"][1] - p["range"][0]) if p else n // M
            dflt = default[i] if default else float("nan")
            lines.append(f"| {i} | {cols} | {pids:,} | {prefix[i]:.3f} | {prefix[i] / pids * 1e6:.3f} | {dflt:.3f} | {dflt / pids * 1e6:.3f} | {prefix[i] - dflt:+.3f} |")
        med = statistics.median(prefix)
        inner = sorted(prefix[1:7])
        lines.append(f"| sum / max-median | | | {sum(prefix):.3f} / {max(prefix) / med:.3f} | | {sum(default):.3f} / {max(default) / statistics.median(default):.3f} | | {sum(prefix) - sum(default):+.3f} |")
        lines.append(f"- prefix: max piece {prefix.index(max(prefix))} ({max(prefix):.3f} ms), min piece {prefix.index(min(prefix))} ({min(prefix):.3f} ms), middle-six median {(inner[2] + inner[3]) / 2:.3f}; "
                     f"per-run medians agree within {max(abs(a - b) for a, b in zip(sel[0]['pieces_ms'], sel[1]['pieces_ms'])) if len(sel) > 1 else 0:.3f} ms; "
                     f"the unsplit kernel in these runs: {statistics.mean(r['fo' if short == 'cd' else 'cd'] for r in sel):.3f} ms")
        lines.append("")
        result["pieces"].setdefault(cube, {})[short] = {"prefix_ms": prefix, "prefix_runs": [r["pieces_ms"] for r in sel], "default_ms": default,
                                                        "columns": [(p["x_first"], p["x_last"]) for p in audit["pieces"]] if audit else None}

text = "\n".join(lines)
(ROOT / "prefix_defrag.md").write_text(text, encoding="utf-8")
json.dump(result, open(ROOT / "prefix_defrag.json", "w", encoding="utf-8"), indent=1, default=str)
print(text)
