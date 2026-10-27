"""E9 local validation summary (items 4-6): reads scratchpad e9/runs/<label>/{run.log,trace/steps_device.csv}, the
dumps' comparison and writes local_validation.md / .json. Reference values: E8 report 7.18 (prefix monkeypatch runs,
same protocol, same card at 575 W) and the default-defrag runs of 7.17 / 7.18."""
import csv
import json
import pathlib
import re
import statistics

SPE = pathlib.Path("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9")
RUNS = SPE / "runs"
PREFIX_718 = json.load(open(SPE.parent / "e8" / "prefix" / "prefix_defrag.json", encoding="utf-8"))["cost"]
N = {"8m": 8_000_000, "64m": 64_000_000, "2d1m": 1_044_484, "2d8m": 8_191_044}


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {"audits": [], "seam_ok": None}
    m = re.search(r"STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s", text)
    if m:
        out["warmup"], out["measured"], out["step_ms"] = int(m.group(1)), int(m.group(2)), 1000.0 * float(m.group(3)) / int(m.group(2))
    m = re.search(r"steady_fps=([\d.]+) steady_fps_nodefrag=([\d.]+)", text)
    if m:
        out["fps"], out["fps_nodefrag"] = float(m.group(1)), float(m.group(2))
    out["defrag_events"] = re.findall(r"defrag_event call=\d+ f(\d+) drain_ms=([\d.]+) idle_ms=([\d.]+) wall_ms=([\d.]+)", text)
    m = re.search(r"drift=(-?\d+)", text)
    out["drift"] = int(m.group(1)) if m else None
    m = re.search(r"defrag_order=(\w+)", text)
    out["defrag_order"] = m.group(1) if m else None
    for a in re.finditer(r"audit: ([\d,]+) own pids, ([\d,]+) voxel blocks for ([\d,]+) non-empty voxels, ([\d,]+) block descents, ([\d,]+) pids without a voxel -> (EXACT|NOT)", text):
        out["audits"].append({"pids": int(a.group(1).replace(",", "")), "blocks": int(a.group(2).replace(",", "")),
                              "nonempty": int(a.group(3).replace(",", "")), "descents": int(a.group(4).replace(",", "")),
                              "zero": int(a.group(5).replace(",", "")), "exact": a.group(6) == "EXACT"})
    seams = re.findall(r"seam \d+ \(col \d+\): .*? dup=(\d+) (OK|FAIL)", text)
    if seams:
        out["seam_ok"] = all(s[1] == "OK" for s in seams)
        out["seam_lines"] = re.findall(r"\[chain_v7\] seam \d+ .*", text)
    out["scan_ms"] = [float(v) for v in re.findall(r"own particles, ([\d.]+) ms", text)]
    out["overflow_total"] = re.search(r"overflow_total=(\d+)", text).group(1) if "overflow_total=" in text else None
    return out


def trace_kernels(path, warmup):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cd, fo, period_by_step = [], [], {}
    prev = None
    for r in rows:
        if r.get("sim") not in (None, "", "0"):          # K > 1: the rows of every slab are interleaved; slab 0's periods
            continue
        step = int(r["step"])
        try:
            bs, cde, fe, ce = int(r["b_start"]), int(r["b_correction_density_interior_end"]), int(r["b_force_deep_interior_end"]), int(r["c_end"])
        except (ValueError, KeyError):
            prev = None
            continue
        if prev is not None:
            period_by_step[step] = (ce - prev) / 1e6
        prev = ce
        if step >= warmup:
            cd.append((cde - bs) / 1e6)
            fo.append((fe - cde) / 1e6)
    return {"cd": statistics.median(cd) if cd else None, "fo": statistics.median(fo) if fo else None, "periods": period_by_step}


runs = {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    if not (d / "run.log").exists():
        continue
    rec = parse_log(d / "run.log")
    if "step_ms" not in rec:
        continue
    trace = d / "trace" / "steps_device.csv"
    if trace.exists():
        rec.update(trace_kernels(trace, rec["warmup"]))
    rec["status"] = (d / "status.txt").read_text().strip() if (d / "status.txt").exists() else ""
    runs[d.name] = rec

lines = ["### E9 local validation (GPU 1, 575 W, the E9 tree of the e8 worktree: prefix defrag order as the production path)", "",
         "#### item 4: K = 1, 300 steps (window 100-299), pid-order audit after the bootstrap defrag", "",
         "| run | defrag order | step (ms) | ns / particle | corr+dens (ms) | force (ms) | 7.18 prefix step (ms) | diff | audit (bootstrap defrag) | drift |",
         "|---|---|---|---|---|---|---|---|---|---|"]
result = {"runs": {}, "item4": {}, "item5": {}, "item6": {}}
for label, cube in (("e9_8m_k1_audit", "8m"), ("e9_64m_k1_audit", "64m"), ("e9_8m_k1", "8m"), ("e9_64m_k1", "64m")):
    r = runs.get(label)
    if not r:
        continue
    ref = PREFIX_718[cube]["groups"]["prefix (host scan)"]["step_ms"]
    a = r["audits"][0] if r["audits"] else None
    audit = (f"{a['blocks']:,} blocks = {a['nonempty']:,} voxels, {a['descents']} descents, {a['zero']} without voxel -> {'EXACT' if a['exact'] else 'NOT'}" if a else "-")
    lines.append(f"| {label} | {r['defrag_order']} | {r['step_ms']:.3f} | {r['step_ms'] / N[cube] * 1e6:.3f} | {r.get('cd') or float('nan'):.3f} | {r.get('fo') or float('nan'):.3f} | "
                 f"{ref:.3f} | {(r['step_ms'] / ref - 1) * 100:+.2f} % | {audit} | {r['drift']} |")
    result["item4"][label] = {"step_ms": r["step_ms"], "ns_per_particle": r["step_ms"] / N[cube] * 1e6, "cd": r.get("cd"), "fo": r.get("fo"),
                              "ref_718_prefix_ms": ref, "diff_pct": (r["step_ms"] / ref - 1) * 100, "audit": a, "drift": r["drift"]}

lines += ["", "#### item 5: K = 2 on one card (8M cube, --device-map 1,1), 300 steps", ""]
r = runs.get("e9_8m_k2_trace")
if r:
    lines.append(f"- `e9_8m_k2_trace`: {r['step_ms']:.3f} ms/step ({1000 / r['step_ms']:.2f} fps, two slabs on one card), drift {r['drift']}, overflow_total {r['overflow_total']}, "
                 f"seam check {'OK' if r['seam_ok'] else 'FAIL / none'}; audits (bootstrap defrag, both slabs, then the end-of-run defrag): "
                 f"{', '.join(('EXACT' if a['exact'] else 'NOT') + f' ({a[chr(112) + chr(105) + chr(100) + chr(115)]:,} pids, {a[chr(100) + chr(101) + chr(115) + chr(99) + chr(101) + chr(110) + chr(116) + chr(115)]} descents)' for a in r['audits'])}")
    for s in r.get("seam_lines", []):
        lines.append(f"  - {s}")
    result["item5"]["k2_trace"] = {k: r[k] for k in ("step_ms", "drift", "seam_ok", "audits", "overflow_total")}
cmp_md = SPE / "compare" / "compare_8m.md"
if cmp_md.exists():
    lines += ["", "field comparison K = 2 vs K = 1 at step 300 (`e8_field_compare.py`, phase-1 format):", ""]
    lines += [line for line in cmp_md.read_text(encoding="utf-8").splitlines()]
    try:
        result["item5"]["compare"] = json.load(open(SPE / "compare" / "compare_8m.json", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
r = runs.get("e9_8m_k2_cad100")
if r and r.get("periods"):
    periods = r["periods"]
    def med(a, b):
        vals = [periods[s] for s in range(a, b) if s in periods]
        return statistics.median(vals) if vals else float("nan")
    windows = [(50, 100), (101, 150), (150, 200), (201, 250), (250, 300)]
    lines += ["", f"- `e9_8m_k2_cad100` (defrag cadence 100, periodic defrags at frames 100 and 200 inside the run): step {r['step_ms']:.3f} ms, drift {r['drift']}, "
              f"seam check {'OK' if r['seam_ok'] else 'FAIL / none'}, defrag events {[(f, w) for f, _, _, w in r['defrag_events']]} (frame, wall ms); audits "
              f"{['EXACT' if a['exact'] else 'NOT' for a in r['audits']]}",
              "", "| window (steps) | median step period (ms) |", "|---|---|"]
    for a, b in windows:
        lines.append(f"| {a}-{b - 1} | {med(a, b):.3f} |")
    first_after = {f: periods.get(int(f) + 1) for f, _, _, _ in r["defrag_events"]}
    lines.append(f"| first step after each defrag | {', '.join(f'f{f}+1: {v:.3f}' if v else f'f{f}+1: -' for f, v in first_after.items())} |")
    before = med(50, 100); after = med(201, 250)
    lines.append(f"- before the first periodic defrag (50-99) {before:.3f} ms, after the second (201-249) {after:.3f} ms: {(after / before - 1) * 100:+.2f} % -> "
                 f"{'no jump' if abs(after / before - 1) < 0.01 else 'CHECK'}")
    result["item5"]["cad100"] = {"step_ms": r["step_ms"], "windows": {f"{a}-{b - 1}": med(a, b) for a, b in windows}, "first_after": first_after,
                                 "defrag_events": r["defrag_events"], "audits": r["audits"], "drift": r["drift"]}

lines += ["", "#### item 6: 2-D sensitivity, K = 1, 1300 steps (window 300-1299, one periodic defrag at 1000 inside; fps = steady_fps_nodefrag)", "",
          "| case | particles | atomic (ms/step) | prefix (ms/step) | prefix - atomic | atomic audit | prefix audit |", "|---|---|---|---|---|---|---|"]
for tag, label in (("2d1m", "cavity2d_n1000 (1M)"), ("2d8m", "cavity2d_n2840 (8M)")):
    a, p = runs.get(f"e9_{tag}_atomic"), runs.get(f"e9_{tag}_prefix")
    if not (a and p):
        continue
    ms_a = 1000.0 / a["fps_nodefrag"] if a.get("fps_nodefrag") else a["step_ms"]
    ms_p = 1000.0 / p["fps_nodefrag"] if p.get("fps_nodefrag") else p["step_ms"]
    aud = lambda r: (f"{r['audits'][0]['descents']:,} descents" if r["audits"] else "-")
    lines.append(f"| {label} | {N[tag]:,} | {ms_a:.4f} | {ms_p:.4f} | **{(ms_p / ms_a - 1) * 100:+.2f} %** | {aud(a)} | {aud(p)} |")
    result["item6"][tag] = {"atomic_ms": ms_a, "prefix_ms": ms_p, "diff_pct": (ms_p / ms_a - 1) * 100, "atomic_audit": a["audits"][:1], "prefix_audit": p["audits"][:1],
                            "atomic_step_ms_raw": a["step_ms"], "prefix_step_ms_raw": p["step_ms"]}
result["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "periods"} for k, v in runs.items()}
text = "\n".join(lines)
(SPE / "local_validation.md").write_text(text, encoding="utf-8")
json.dump(result, open(SPE / "local_validation.json", "w", encoding="utf-8"), indent=1, default=str)
print(text)
