"""E9 item 7: tab:kernelcost (Section 3 Table 2) old (v7-single rc1 host, atomic defrag order, 2026-10-10) vs new
(v7-single-rc4 + the breakdown-driver fix, prefix order, 2026-10-27), same protocol (3 runs, frames 1001-3000, 1 in
flight, 600 W), plus the per-size GPU step of the single runs (fig:baseline's affected points). Reads the two
summary.json, writes section3_compare.md / .json."""
import json
import pathlib

V7S = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-v7-single")
OLD = json.load(open(V7S / "docs/seam_audit/v7_single/section3_summary.json", encoding="utf-8"))
NEW = json.load(open(V7S / "logs/v7_single/section3_rc4/summary.json", encoding="utf-8"))
OUT = pathlib.Path("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9")


def walk(node, depth=0, limit=3):
    if isinstance(node, dict) and depth < limit:
        for k, v in list(node.items())[:40]:
            print("  " * depth + str(k), type(v).__name__, (v if isinstance(v, (int, float, str)) else ""))
            walk(v, depth + 1, limit)


if "--explore" in __import__("sys").argv:
    print("OLD keys"); walk(OLD, 0, 2)
    print("NEW keys"); walk(NEW, 0, 2)
    raise SystemExit(0)

# summary.json layout (section3 summarize): "table2": {"8m": {kernel: {"mean_us":..., "std_us":..., "ns_per_particle":..., ...}}, "3d_8m": {...}},
# "sizes": {tag: {"gpu_step_us": ..., ...}}  (the exact keys are discovered below)
rows, records = [], {"table2": {}, "sizes": {}}


def pick(d, *names):
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return None


sizes_old, sizes_new = OLD["sizes"], NEW["sizes"]
lines = ["### Section 3 Table 2 (tab:kernelcost): old (atomic defrag order, 2026-10-10, commit f06af58) vs new (prefix order, v7-single-rc4 + ffd1a2f, 2026-10-27), 600 W, 3 runs each, frames 1001-3000, 1 in flight, defrag frames excluded", "",
         "| case | kernel | old us/step (std) | new us/step (std) | change | > 0.5 %? | old ns/particle | new ns/particle | old % | new % |", "|---|---|---|---|---|---|---|---|---|---|"]
for case, label in (("8m", "2-D 8M (8,128,201)"), ("3d_8m", "3-D 8M (9,129,329)")):
    o_case, n_case = sizes_old[case], sizes_new[case]
    for kernel in n_case["stage_names"] + ["gpu_frame"]:
        o, n = o_case["stages"][kernel], n_case["stages"][kernel]
        ou, nu = o["mean_us"], n["mean_us"]
        change = (nu / ou - 1) * 100
        name = "step total" if kernel == "gpu_frame" else kernel
        lines.append(f"| {label} | {name} | {ou:.0f} ({o['std_us']:.0f}) | {nu:.0f} ({n['std_us']:.0f}) | **{change:+.2f} %** | {'yes' if abs(change) > 0.5 else 'no'} | "
                     f"{o['ns_per_particle']:.2f} | {n['ns_per_particle']:.2f} | {o['share_pct']:.1f} | {n['share_pct']:.1f} |")
        records["table2"].setdefault(case, {})[name] = {"old_us": ou, "new_us": nu, "old_std_us": o["std_us"], "new_std_us": n["std_us"], "change_pct": change,
                                                       "updated": abs(change) > 0.5, "old_ns_per_particle": o["ns_per_particle"], "new_ns_per_particle": n["ns_per_particle"],
                                                       "old_share_pct": o["share_pct"], "new_share_pct": n["share_pct"]}
    records["table2"][case]["throughput_mps"] = {"old": o_case["gpu_throughput_mps"], "new": n_case["gpu_throughput_mps"]}
    records["table2"][case]["step_ps_per_candidate"] = {"old": o_case["step_ps_per_candidate"], "new": n_case["step_ps_per_candidate"]}
    records["table2"][case]["covariates_new"] = n_case["covariates"]
    records["table2"][case]["covariates_old"] = o_case.get("covariates")
    lines.append(f"| {label} | GPU throughput (M particle-steps/s) | {o_case['gpu_throughput_mps']:.1f} | {n_case['gpu_throughput_mps']:.1f} | "
                 f"**{(n_case['gpu_throughput_mps'] / o_case['gpu_throughput_mps'] - 1) * 100:+.2f} %** | | ps/candidate (step) {o_case['step_ps_per_candidate']:.2f} | {n_case['step_ps_per_candidate']:.2f} | | |")
lines += ["", "### per-size GPU step of the single runs (frames 1001-3000, defrag frames excluded): fig:baseline's affected points", "",
          "| size | particles | old GPU step (us) | new (us) | change | affected (> 0.5 %)? | new SM MHz / W / PCIe |", "|---|---|---|---|---|---|---|"]
for tag in sizes_new:
    o, n = sizes_old.get(tag, {}), sizes_new[tag]
    if not o:
        continue
    ou, nu = o["gpu_frame_us"], n["gpu_frame_us"]
    change = (nu / ou - 1) * 100
    cov = n["covariates"][0] if n.get("covariates") else {}
    lines.append(f"| {tag} | {n['particles']:,} | {ou:.0f} | {nu:.0f} | **{change:+.2f} %** | {'yes' if abs(change) > 0.5 else 'no'} | "
                 f"{cov.get('sm_mhz', '-')} / {cov.get('power_w', '-')} / {cov.get('pcie', '-')} |")
    records["sizes"][tag] = {"old_us": ou, "new_us": nu, "change_pct": change, "affected": abs(change) > 0.5, "particles": n["particles"], "covariate": cov}
text = "\n".join(lines)
(OUT / "section3_compare.md").write_text(text, encoding="utf-8")
json.dump(records, open(OUT / "section3_compare.json", "w", encoding="utf-8"), indent=1)
print(text)
