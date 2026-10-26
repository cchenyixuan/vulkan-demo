"""Local 3-D size-cost runs (items 3-6): one table row per run directory under runs/ (run.log, telemetry.csv, trace/).
Columns: steady fps (window 100-299), ns per particle (total and fluid), correction_density and force kernels (ns per
particle of the total), SM clock and power medians over the loop window, the deep-wall-skip setting, the trace's own
period. Also the item summaries: 8M vs 64M ratio, deep-wall-skip effect, c_w from border 8 - border 4, lid 0 vs 1."""
import csv
import datetime
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, "C:/Users/cchen/PycharmProjects/vulkan-demo-e8/docs/cluster_v6/scripts")
import e8_kernel_table as kt  # noqa: E402

DEFAULT_ROOT = ("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/"
                "fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d/runs")
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
# label prefix -> (base case for the trace reader, N_total, N_fluid, N_wall incl. lid)
CASES = {"8m_base": ("cavity3d_n192", 8000000, 7077888, 922112),
         "8m_dwskip_off": ("cavity3d_n192", 8000000, 7077888, 922112),
         "8m_lid0": ("cavity3d_n192", 8000000, 7077888, 922112),
         "8m_border8": ("cavity3d_n192", 8998912, 7077888, 1921024),
         "8m_border8_skip0": ("cavity3d_n192", 8998912, 7077888, 1921024),
         "64m_base": ("cavity3d_n392", 64000000, 60236288, 3763712),
         "64m_dwskip_off": ("cavity3d_n392", 64000000, 60236288, 3763712)}
WALLS_ADDED = 1921024 - 922112


def parse_log(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    if m:
        out.update(warmup=int(m.group(1)), steady_steps=int(m.group(2)), steady_s=float(m.group(3)), steady_fps=float(m.group(4)))
    m = re.search(r"TOTAL: (\d+) steps in ([\d.]+)s", text)
    if m:
        out.update(total_steps=int(m.group(1)), total_s=float(m.group(2)))
    for stage in ("loop_start", "loop_end"):
        m = re.search(rf"stage {stage}: t=([\d.]+)s .*?epoch=([\d.]+)", text)
        if m:
            out[stage] = float(m.group(2))
    m = re.search(r"drift=(\d+)", text)
    out["drift"] = int(m.group(1)) if m else None
    m = re.search(r"V7_DEEP_WALL_SKIP=(\w+)", text)
    out["deep_wall_skip"] = m.group(1) if m else "auto"
    return out


def telemetry(path: pathlib.Path, t0: float, t1: float) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((stamp, float(parts[1].split()[0]), float(parts[2].split()[0]), float(parts[3].split()[0]),
                         float(parts[4].split()[0]), float(parts[6].split()[0])))
        except ValueError:
            continue
    window = [r for r in rows if t0 <= r[0] <= t1] or [r for r in rows if r[1] >= 50]
    if not window:
        return {}
    return {"samples": len(window), "util_p50": statistics.median(r[1] for r in window), "mem_max_MiB": max(r[2] for r in window),
            "power_p50_W": statistics.median(r[3] for r in window), "power_max_W": max(r[3] for r in window),
            "sm_p50_MHz": statistics.median(r[4] for r in window), "sm_min_MHz": min(r[4] for r in window),
            "temp_max_C": max(r[5] for r in window)}


def kernels(trace_dir: pathlib.Path, base_case: str, n_total: int) -> dict:
    result = kt.analyze(trace_dir, base_case)
    per = {name: result["mean_over_sims"][name]["us_p50_mean"] for name in result["kernel_order"]}
    cd = sum(us for name, us in per.items()
             if name.startswith("correction_density") or name.startswith("correction_") or name.startswith("density"))
    force = sum(us for name, us in per.items() if name.startswith("force"))
    marker = sum(us for name, us in per.items() if "deep_wall" in name)
    period = result["period_mean_of_sims_ms"]
    return {"period_ms": period, "kernels_us": per, "correction_density_us": cd, "force_us": force,
            "deep_wall_marker_us": marker, "correction_density_ns": cd * 1e3 / n_total, "force_ns": force * 1e3 / n_total,
            "other_us": period * 1e3 - cd - force}


def main() -> int:
    rows = []
    for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        prefix = re.sub(r"_r\d+$", "", run_dir.name)
        if prefix not in CASES or not (run_dir / "run.log").exists():
            continue
        base, n_total, n_fluid, n_wall = CASES[prefix]
        log = parse_log(run_dir / "run.log")
        if "steady_fps" not in log:
            print(f"{run_dir.name}: no STEADY line")
            continue
        step_ms = 1000.0 * log["steady_s"] / log["steady_steps"]          # the STEADY line prints fps to 0.1 only
        log["steady_fps"] = 1000.0 / step_ms
        t0 = log.get("loop_start", 0) + log["warmup"] * step_ms / 1000.0
        tele = telemetry(run_dir / "telemetry.csv", t0, log.get("loop_end", 1e18))
        kern = kernels(run_dir / "trace", base, n_total) if (run_dir / "trace" / "run_meta.json").exists() else {}
        row = {"run": run_dir.name, "item": prefix, "n_total": n_total, "n_fluid": n_fluid, "n_wall": n_wall,
               "steady_fps": log["steady_fps"], "step_ms": step_ms, "ns_total": step_ms * 1e6 / n_total,
               "ns_fluid": step_ms * 1e6 / n_fluid, "drift": log["drift"], "deep_wall_skip": log["deep_wall_skip"]}
        row.update(tele)
        row.update({k: v for k, v in kern.items() if k != "kernels_us"})
        row["kernels_us"] = kern.get("kernels_us", {})
        rows.append(row)
    (ROOT.parent / "runs_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    nan = float("nan")
    lines = ["| run | N_total | fps (100-299) | ms/step | ns/particle (total) | ns/particle (fluid) | corr+dens (ns/total) | "
             "force (ns/total) | marker (us) | trace period (ms) | SM p50 (MHz) | power p50 (W) | T max | drift |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['n_total']:,} | {r['steady_fps']:.3f} | {r['step_ms']:.3f} | {r['ns_total']:.4f} | "
                     f"{r['ns_fluid']:.4f} | {r.get('correction_density_ns', nan):.4f} | {r.get('force_ns', nan):.4f} | "
                     f"{r.get('deep_wall_marker_us', nan):.0f} | {r.get('period_ms', nan):.3f} | {r.get('sm_p50_MHz', nan):.0f} | "
                     f"{r.get('power_p50_W', nan):.1f} | {r.get('temp_max_C', nan):.0f} | {r['drift']} |")
    items = {}
    for r in rows:
        items.setdefault(r["item"], []).append(r)

    def mean(item, key):
        vals = [r[key] for r in items.get(item, []) if r.get(key) is not None]
        return statistics.mean(vals) if vals else nan

    lines += ["", "### item means",
              "| item | runs | ns/particle (total) | ns/particle (fluid) | corr+dens ns | force ns | SM MHz | W |",
              "|---|---|---|---|---|---|---|---|"]
    for item in CASES:
        if item in items:
            lines.append(f"| {item} | {len(items[item])} | {mean(item, 'ns_total'):.4f} | {mean(item, 'ns_fluid'):.4f} | "
                         f"{mean(item, 'correction_density_ns'):.4f} | {mean(item, 'force_ns'):.4f} | {mean(item, 'sm_p50_MHz'):.0f} | "
                         f"{mean(item, 'power_p50_W'):.1f} |")
    lines += ["", "### derived"]
    if "8m_base" in items and "64m_base" in items:
        lines.append(f"- item 3: ns/particle 8M / 64M = {mean('8m_base', 'ns_total') / mean('64m_base', 'ns_total'):.4f} "
                     f"(cluster 6.637 / 6.116 = 1.085); corr+dens "
                     f"{mean('8m_base', 'correction_density_ns') / mean('64m_base', 'correction_density_ns'):.4f}, "
                     f"force {mean('8m_base', 'force_ns') / mean('64m_base', 'force_ns'):.4f}")
    for size in ("8m", "64m"):
        if f"{size}_base" in items and f"{size}_dwskip_off" in items:
            lines.append(f"- item 4 ({size}): deep-wall-skip off / auto = "
                         f"{mean(f'{size}_dwskip_off', 'ns_total') / mean(f'{size}_base', 'ns_total'):.4f} "
                         f"(step {mean(f'{size}_dwskip_off', 'step_ms'):.3f} vs {mean(f'{size}_base', 'step_ms'):.3f} ms)")
    if all(k in items for k in ("8m_base", "8m_dwskip_off", "64m_base", "64m_dwskip_off")):
        lines.append(f"- item 4: 8M / 64M ratio with skip off = {mean('8m_dwskip_off', 'ns_total') / mean('64m_dwskip_off', 'ns_total'):.4f}")
    if "8m_base" in items and "8m_border8" in items:
        d_ms = mean("8m_border8", "step_ms") - mean("8m_base", "step_ms")
        lines.append(f"- item 5: border 8 - border 4 = {d_ms:+.3f} ms per step for {WALLS_ADDED:,} more wall particles -> "
                     f"c_w = {d_ms * 1e6 / WALLS_ADDED:.3f} ns per wall particle (fit: 14.25 ns); corr+dens "
                     f"{mean('8m_border8', 'correction_density_us') - mean('8m_base', 'correction_density_us'):+.0f} us, "
                     f"force {mean('8m_border8', 'force_us') - mean('8m_base', 'force_us'):+.0f} us")
    if "8m_base" in items and "8m_lid0" in items:
        lines.append(f"- item 6: lid 0 / lid 1 = {mean('8m_lid0', 'ns_total') / mean('8m_base', 'ns_total'):.4f} "
                     f"(step {mean('8m_lid0', 'step_ms'):.3f} vs {mean('8m_base', 'step_ms'):.3f} ms)")
    text = "\n".join(lines)
    (ROOT.parent / "runs_table.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
