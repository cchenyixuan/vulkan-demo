"""Split-dispatch tail series (user 2026-10-27): per cube (8M, 64M) and per sweep kernel, T(m) - T(1) against m - 1 for
m = 1, 2, 4, 8 (two runs each; the other kernel unsplit), slope = the per-dispatch fill + drain cost, next to the fixed
terms of 7.16 (T - a N with the box slopes). Kernel times from the runs' own step traces (median over the steady steps
100-299), step times from the STEADY line. Reads split/runs/<cube>_<cfg>_r<r>/; writes split_tail.md / .json."""
import csv
import datetime
import json
import pathlib
import re
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/split")
RUNS = ROOT / "runs"
FIXED = json.load(open("C:/Users/cchen/PycharmProjects/vulkan-demo-e8/docs/cluster_v6/data/e8/local_size_cost/fixed_terms.json", encoding="utf-8"))
FIXED_OF = {"8m": next(r for r in FIXED["rows"] if r["item"] == "8m_base"), "64m": next(r for r in FIXED["rows"] if r["item"] == "64m_base")}
CUBES = {"8m": "cube 8M (200^3)", "64m": "cube 64M (400^3)"}
KERNELS = {"cd": ("correction_density_interior", "b_start", "b_correction_density_interior_end", "fixed_corr_dens_ms"),
           "fo": ("force_deep_interior_scratch", "b_correction_density_interior_end", "b_force_deep_interior_end", "fixed_force_ms")}
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
    m = re.search(r"drift=(\d+)", text)
    out["drift"] = int(m.group(1)) if m else None
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
        step = int(r["step"])
        try:
            bs, cde, fe, ce = (int(r["b_start"]), int(r["b_correction_density_interior_end"]), int(r["b_force_deep_interior_end"]), int(r["c_end"]))
        except ValueError:
            prev_end = None
            continue
        if step >= WARMUP:
            out["cd"].append((cde - bs) / 1e6)
            out["fo"].append((fe - cde) / 1e6)
            if prev_end is not None:
                out["period"].append((ce - prev_end) / 1e6)
        prev_end = ce
    return {k: statistics.median(v) for k, v in out.items() if v} | {"steps": len(out["cd"])}


runs = {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    m = re.fullmatch(r"(8m|64m)_(m1_plain|m1_seg|cd\d|fo\d)_r(\d)", d.name)
    if not m or not (d / "run.log").exists():
        continue
    cube, cfg, r = m.group(1), m.group(2), int(m.group(3))
    log = parse_log(d / "run.log")
    if "step_ms" not in log:
        print(f"{d.name}: no STEADY line")
        continue
    rec = {"cube": cube, "cfg": cfg, "r": r, **log}
    if (d / "trace" / "steps_device.csv").exists():
        rec.update(trace_kernels(d / "trace" / "steps_device.csv"))
    rec.update(telemetry(d / "telemetry.csv", log.get("loop_start", 0) + WARMUP * log["step_ms"] / 1e3, log.get("loop_end", 1e18)))
    runs[d.name] = rec

lines = ["### split-dispatch tail series: per run (kernel times = step-trace medians over steps 100-299, ms)", "",
         "| run | SPLIT_CD | SPLIT_FO | step (ms) | corr+dens (ms) | force (ms) | trace period (ms) | SM MHz | W | drift |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for name, r in runs.items():
    lines.append(f"| {name} | {r['split'][0]} | {r['split'][1]} | {r['step_ms']:.3f} | {r.get('cd', float('nan')):.3f} | {r.get('fo', float('nan')):.3f} | "
                 f"{r.get('period', float('nan')):.3f} | {r.get('sm_p50_MHz', float('nan')):.0f} | {r.get('power_p50_W', float('nan')):.0f} | {r['drift']} |")

result = {"runs": runs, "fits": {}}
lines += ["", "### T(m) - T(1) per kernel (the other kernel unsplit); T(1) = the m1_plain runs", ""]
for cube, label in CUBES.items():
    base = [r for r in runs.values() if r["cube"] == cube and r["cfg"] == "m1_plain"]
    seg1 = [r for r in runs.values() if r["cube"] == cube and r["cfg"] == "m1_seg"]
    if not base:
        continue
    t1 = {k: statistics.mean(r[k] for r in base) for k in ("cd", "fo", "step_ms")}
    lines.append(f"**{label}**: T(1) corr+dens {t1['cd']:.3f} ms, force {t1['fo']:.3f} ms, step {t1['step_ms']:.3f} ms "
                 f"(m1_plain x{len(base)}: steps {[round(r['step_ms'], 3) for r in base]})"
                 + (f"; m1_seg (SEGMENT pipelines, first thread 0, no split): corr+dens {statistics.mean(r['cd'] for r in seg1):.3f}, "
                    f"force {statistics.mean(r['fo'] for r in seg1):.3f}, step {statistics.mean(r['step_ms'] for r in seg1):.3f}" if seg1 else ""))
    lines += ["", "| kernel | m | runs | T_k(m) (ms) | T_k(m) - T_k(1) (ms) | per boundary (ms) | other kernel (ms) | step - step(1) (ms) |",
              "|---|---|---|---|---|---|---|---|"]
    for k, (kname, _, _, fixed_key) in KERNELS.items():
        other = "fo" if k == "cd" else "cd"
        xs, ys, ys_step = [], [], []
        for m_value in (2, 4, 8):
            sel = [r for r in runs.values() if r["cube"] == cube and r["cfg"] == f"{k}{m_value}"]
            if not sel:
                continue
            tk = statistics.mean(r[k] for r in sel)
            dk = tk - t1[k]
            dstep = statistics.mean(r["step_ms"] for r in sel) - t1["step_ms"]
            lines.append(f"| {kname} | {m_value} | {len(sel)} | {tk:.3f} ({', '.join(f'{r[k]:.3f}' for r in sel)}) | {dk:+.3f} | {dk / (m_value - 1):+.3f} | "
                         f"{statistics.mean(r[other] for r in sel):.3f} (T(1) {t1[other]:.3f}) | {dstep:+.3f} |")
            for r in sel:
                xs.append(m_value - 1); ys.append(r[k] - t1[k]); ys_step.append(r["step_ms"] - t1["step_ms"])
        if len(xs) >= 2:
            x, y, ystep = np.array(xs, float), np.array(ys, float), np.array(ys_step, float)
            slope0 = float((x * y).sum() / (x * x).sum())                       # through the origin
            (slope, intercept), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
            slope0_step = float((x * ystep).sum() / (x * x).sum())
            res = y - slope0 * x
            fixed = FIXED_OF[cube][fixed_key]
            result["fits"][f"{cube}_{k}"] = {"kernel": kname, "slope_origin_ms": slope0, "slope_free_ms": float(slope), "intercept_free_ms": float(intercept),
                                             "slope_origin_step_ms": slope0_step, "rms_ms": float(np.sqrt((res ** 2).mean())), "fixed_term_ms": fixed,
                                             "ratio": slope0 / fixed if fixed else None, "T1_ms": t1[k], "points": list(zip(xs, ys))}
            lines.append(f"| **{kname} slope** | | | | slope through origin **{slope0:+.3f} ms per boundary** (free fit {slope:+.3f} m + {intercept:+.3f}; "
                         f"rms {np.sqrt((res ** 2).mean()):.3f}; from step times {slope0_step:+.3f}) | | fixed term 7.16: **{fixed:+.3f} ms** | ratio slope / fixed = **{slope0 / fixed:.2f}** |")
    lines.append("")

lines += ["### summary", "", "| cube | kernel | T(1) (ms) | slope: T(m) - T(1) per (m - 1) (ms) | fixed term T_k - a_k N (ms) | slope / fixed |", "|---|---|---|---|---|---|"]
for key, f in result["fits"].items():
    cube, k = key.split("_")
    lines.append(f"| {CUBES[cube]} | {f['kernel']} | {f['T1_ms']:.3f} | {f['slope_origin_ms']:+.3f} (free {f['slope_free_ms']:+.3f}, rms {f['rms_ms']:.3f}) | "
                 f"{f['fixed_term_ms']:+.3f} | {f['ratio']:.2f} |")
text = "\n".join(lines)
(ROOT / "split_tail.md").write_text(text, encoding="utf-8")
json.dump(result, open(ROOT / "split_tail.json", "w", encoding="utf-8"), indent=1, default=str)
print(text)
