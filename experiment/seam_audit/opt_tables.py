"""
opt_tables.py — markdown tables for docs/seam_audit/v6_opt.md from the opt campaign / gate / pool /
side-measurement / delta-density result files under logs/seam_audit/opt (CPU only).

    .venv/Scripts/python.exe -m experiment.seam_audit.opt_tables --out logs/seam_audit/opt/tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

NEWLINE = chr(10)

ROOT = pathlib.Path(__file__).resolve().parents[2] / "logs" / "seam_audit" / "opt"
CASE_ORDER = ("2d_narrow", "2d_1m", "2d_4m", "2d_16m", "3d_8m", "3d_narrow")
CASE_LABEL = {"2d_narrow": "2-D 10k", "2d_1m": "2-D 1M", "2d_4m": "2-D 4M", "2d_16m": "2-D 16M",
              "3d_8m": "3-D 8M", "3d_narrow": "3-D narrow"}


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}"


def load_summary(campaign: str) -> dict:
    path = ROOT / campaign / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["case"], row["config"]): row for row in rows}


def trial_ratio(campaign: str, case: str, config: str, reference: str):
    """Trial-wise fps ratio config / reference (mean, std over trials)."""
    records = [json.loads(line) for line in (ROOT / campaign / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {}
    for record in records:
        if record.get("ok"):
            by[(record["case"], record["config"], record["trial"])] = record["result"]["fps"]
    ratios = [by[(case, config, trial)] / by[(case, reference, trial)]
              for (c, k, trial) in by if c == case and k == config and (case, reference, trial) in by]
    if not ratios:
        return None, None
    return statistics.mean(ratios), (statistics.stdev(ratios) if len(ratios) > 1 else 0.0)


def gate(name: str) -> str:
    path = ROOT / f"validate_{name}" / "verdict.json"
    if not path.exists():
        return "—"
    verdict = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for step in ("audit", "single", "k4"):
        entry = verdict.get(step)
        if not entry:
            continue
        worst = entry.get("worst")
        label = f"{step} {'✓' if entry.get('pass') else '✗'}" + (f" {worst:.2f}" if worst is not None else "")
        if step == "k4" and entry.get("attempts"):
            label += f" (attempts {entry['attempts']})"
        parts.append(label)
    return ", ".join(parts)


def step_table(campaign: str, before: str, after: str, title: str) -> str:
    summary = load_summary(campaign)
    lines = [f"**{title}** (campaign `{campaign}`, `{before}` → `{after}`, 3 interleaved trials; per link per frame)", "",
             "| case | fps before | fps after | after / before (trial-wise) | phase C µs | DMA KiB | host KiB | "
             "readback µs | host µs | upload µs | t_tr µs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in CASE_ORDER:
        b, a = summary.get((case, before)), summary.get((case, after))
        if not b or not a:
            continue
        ratio, ratio_std = trial_ratio(campaign, case, after, before)

        def pair(key, scale=1.0):
            return f"{fmt(b[key] / scale if b[key] is not None else None)} → {fmt(a[key] / scale if a[key] is not None else None)}"
        lines.append(
            f"| {CASE_LABEL[case]} | {fmt(b['fps_mean'])} ± {fmt(b['fps_std'], 1)} | {fmt(a['fps_mean'])} ± {fmt(a['fps_std'], 1)} | "
            f"{fmt(100 * ratio, 2)} ± {fmt(100 * ratio_std, 2)} % | {pair('phase_c_us')} | {pair('dma_bytes', 1024)} | "
            f"{pair('host_bytes', 1024)} | {pair('readback_us')} | {pair('host_copy_us')} | {pair('upload_us')} | {pair('t_tr_us')} |")
    return "\n".join(lines)


def phase_c_table(campaign: str, configs: list, reference: str) -> str:
    summary = load_summary(campaign)
    header = "| case | config | fps | vs " + reference + " (trial-wise) | phase C µs | correction band | density band | force band | list build |"
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for case in CASE_ORDER:
        for config in configs:
            row = summary.get((case, config))
            if not row:
                continue
            ratio, ratio_std = trial_ratio(campaign, case, config, reference)
            lines.append(f"| {CASE_LABEL[case]} | {config} | {fmt(row['fps_mean'])} ± {fmt(row['fps_std'])} | "
                         f"{fmt(100 * ratio, 2)} ± {fmt(100 * ratio_std, 2)} % | {fmt(row['phase_c_us'])} | "
                         f"{fmt(row['correction_boundary_us'])} | {fmt(row['density_boundary_us'])} | "
                         f"{fmt(row['force_band_us'])} | {fmt(row.get('band_compact_us'))} |")
    return "\n".join(lines)


def pool_results() -> dict:
    """pool_peaks results by job; a job re-run under pool_peaks_rerun (V6_INIT_SEAM_CLAMP=1 restart)
    replaces the first run."""
    latest = {}
    for directory in ("pool_peaks", "pool_peaks_rerun"):
        path = ROOT / directory / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("ok"):
                latest[record["job"]] = record["result"]
    return latest


def pool_table() -> str:
    latest = pool_results()
    lines = ["| run | frames | flow | replicas per column, max (mean) | slots per column at f = 1 | required f "
             "| migrants / frame / direction, max | per face voxel | departed peak / sim | drift |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for job, result in latest.items():
        replica_max = max(stats["max"] for link in result["links"].values()
                          for region, stats in link["regions"].items() if region in ("inner", "outer"))
        replica_mean = statistics.mean(stats["mean"] for link in result["links"].values()
                                       for region, stats in link["regions"].items() if region in ("inner", "outer"))
        unit = next(stats["capacity_factor_1"] for link in result["links"].values()
                    for region, stats in link["regions"].items() if region == "inner")
        migrant_max = max(stats["max"] for link in result["links"].values()
                          for region, stats in link["regions"].items() if region == "migrant")
        face = next(iter(result["links"].values()))["face_voxels"]
        departed = max(entry["peak"] for entry in result["departed"])
        flow = (f"developed (t = {result['checkpoint_time']:.1f} s)" if result.get("checkpoint") else "from rest")
        lines.append(f"| {job} | {result['frames']:,} | {flow} | {replica_max:,} ({replica_mean:,.0f}) | {unit:,} | "
                     f"{replica_max / unit:.4f} | {migrant_max} | {migrant_max / face:.4f} | {departed} | "
                     f"{result['invariants']['drift']} |")
    return "\n".join(lines)


# release pool factors per dimension: (V6_GHOST_POOL_FACTOR, V6_MIGRANT_POOL_FACTOR,
# V6_DEPARTED_FACE_FRACTION) and the per-voxel capacities of the cases (ppv, incoming)
RELEASE_POOLS = {2: (0.29, 0.05, 0.8), 3: (0.5, 0.02, 0.64)}
PRODUCTION_FACTOR = {2: 0.25, 3: 1.0}


def margin_table() -> str:
    """Peak demand vs capacity at the production factor and at the release factors."""
    latest = pool_results()
    lines = ["| run | replica peak | slots f=prod | headroom | slots f=release | headroom | migrant peak | "
             "migrant slots (release) | headroom | departed peak | departed slots (release) | headroom |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for job, result in latest.items():
        dimension = 3 if job.startswith("3d") else 2
        replica_factor, migrant_factor, departed_fraction = RELEASE_POOLS[dimension]
        link = next(iter(result["links"].values()))
        face = link["face_voxels"]
        unit_replica = link["regions"]["inner"]["capacity_factor_1"]
        unit_migrant = link["regions"]["migrant"]["capacity_factor_1"]
        replica_peak = max(stats["max"] for entry in result["links"].values()
                           for region, stats in entry["regions"].items() if region in ("inner", "outer"))
        migrant_peak = max(entry["regions"]["migrant"]["max"] for entry in result["links"].values())
        departed_peak = max(entry["peak"] for entry in result["departed"])
        sides = max(entry["peer_sides"] for entry in result["departed"])
        slots_production = math.ceil(unit_replica * PRODUCTION_FACTOR[dimension])
        slots_release = math.ceil(unit_replica * replica_factor)
        migrant_slots = max(64, math.ceil(unit_migrant * migrant_factor))
        departed_slots = max(64, math.ceil(departed_fraction * face * sides))

        def headroom(slots, peak):
            return f"{100 * (slots / peak - 1):.0f} %" if peak else "∞"
        lines.append(f"| {job} | {replica_peak:,} | {slots_production:,} | {headroom(slots_production, replica_peak)} | "
                     f"{slots_release:,} | {headroom(slots_release, replica_peak)} | {migrant_peak} | {migrant_slots:,} | "
                     f"{headroom(migrant_slots, migrant_peak)} | {departed_peak} | {departed_slots:,} | "
                     f"{headroom(departed_slots, departed_peak)} |")
    return "\n".join(lines)


def k1_table() -> str:
    path = ROOT / "k1_chain_vs_single" / "summary.md"
    return path.read_text(encoding="utf-8") if path.exists() else "(not run)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "tables.md"))
    args = parser.parse_args()
    parts = ["## (d)", step_table("perf_d_lean", "l2", "lean", "(d) lean transport"), "",
             "Gate: " + gate("lean"), "", "## pool demand", pool_table(), "", "## pool margins", margin_table(), ""]
    if (ROOT / "perf_e_a1" / "summary.json").exists():
        parts += ["## (e)", step_table("perf_e_a1", "lean", "e", "(e) pool sizing"), "", "Gate: " + gate("pool_e"), "",
                  "## (a1)", step_table("perf_e_a1", "e", "a1", "(a1) compact ghost lists"), "",
                  "Gate: " + gate("compact_a1"), ""]
    if (ROOT / "perf_packed_phase_c" / "summary.json").exists():
        parts += ["## (b)(c)", step_table("perf_packed_phase_c", "a1", "packed", "(b)(c) packed replicas"), "",
                  "Gate: " + gate("packed"), "", "## phase C",
                  phase_c_table("perf_packed_phase_c", ["packed", "lanes32", "lanes64", "compact"], "packed"), ""]
    parts += ["## byte model", byte_model_table(), "", "## (f) direct staging", direct_staging_table(), "",
              "## K = 1 chain vs single", k1_table(), "", "## delta density (first runs)", delta_density_table("delta_density"), "",
              "## delta density (dense noise sampling)", delta_density_table("delta_density_dense"), "",
              "## delta density noise by region (final states)", delta_density_noise_split("delta_density"), "",
              delta_density_noise_split("delta_density_dense"), ""]
    parts += ["## delta density perf", delta_density_perf_table(), ""]
    if (ROOT / E23_CAMPAIGN / "summary.json").exists():
        parts += ["## E23 gates", e23_gates_table(), "",
                  "## E23 audit diagnostics", e23_audit_diagnostics_table(), "",
                  "## E23 A/B 3-D detail", e23_ab_detail_table(), "",
                  "## E23 G1 without P", step_table(E23_CAMPAIGN, "base", "release",
                                                    "E23 G1 without P (1b52dd2 build → E23 build, --counterbalance)"), "",
                  "## E23 host-byte formula", e23_formula_table(), "",
                  "## E23 install kernels", e23_install_table(), ""]
    if (ROOT / "perf_final" / "summary.json").exists():
        parts += ["## final", final_table(), "",
                  "## final phase C", phase_c_table("perf_final", ["l1x", "packed", "release", "release_compact"], "packed"), ""]
    pathlib.Path(args.out).write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts))
    return 0



# ---------------------------------------------------------------- additional sections
CASE_GEOMETRY = {  # perf case -> (pool-peaks run used for the live counts, dimension, ppv, incoming)
    "2d_narrow": ("2d_narrow_init", 2, 96, 16), "2d_1m": ("2d_1m_re1000_t53", 2, 96, 16),
    "2d_4m": ("2d_4m_init", 2, 96, 16), "2d_16m": ("2d_16m_init", 2, 96, 16),
    "3d_8m": ("3d_8m_init", 3, 128, 32), "3d_narrow": ("3d_narrow_init", 3, 128, 32)}


def byte_model_table() -> str:
    """Per link per frame (KiB) of the (1,2) staging layout for each step, from the
    segment formulas and the measured live counts; packing on top of (d)(e)(a1)."""
    latest = pool_results()
    lines = ["| case | lean (d) DMA / host | + pools (e) | + compact lists (a1) | + packed (b)(c) | packing saves DMA / host | "
             "+ G1 without P (E23) | E23 saves DMA / host |",
             "|---|---|---|---|---|---|---|---|"]
    for case, (job, dimension, ppv, incoming) in CASE_GEOMETRY.items():
        link = latest[job]["links"]["s0_to_s1"]
        face = link["face_voxels"]
        live_inner, live_outer = link["regions"]["inner"]["mean"], link["regions"]["outer"]["mean"]
        live_migrant = link["regions"]["migrant"]["mean"]
        ghost_voxels = 2 * face

        def sizes(replica_factor, migrant_factor, compact, packed):
            replica_slots = math.ceil(face * (ppv + incoming) * replica_factor)
            migrant_slots = math.ceil(face * incoming * (migrant_factor if migrant_factor is not None else replica_factor))
            if migrant_factor is not None:
                migrant_slots = max(64, migrant_slots)
            lists = ghost_voxels * 4 + (ghost_voxels * 4 if compact else ghost_voxels * ppv * 4)
            # packed: the (b)(c) format G1 36 B + G2 32 B; packed == "e23": G1 and G2 32 B (G1 without P)
            pair = 64 if packed == "e23" else (68 if packed else 88)
            dma = replica_slots * pair + migrant_slots * 44 + lists + 16
            inner_bytes = 32 if packed == "e23" else 36
            host = ((live_inner * inner_bytes + live_outer * 32) if packed else (live_inner + live_outer) * 44) \
                + live_migrant * 44 + lists + 16
            return dma / 1024, host / 1024
        release = RELEASE_POOLS[dimension]
        lean = sizes(PRODUCTION_FACTOR[dimension], None, False, False)
        pools = sizes(release[0], release[1], False, False)
        compact = sizes(release[0], release[1], True, False)
        packed = sizes(release[0], release[1], True, True)
        e23 = sizes(release[0], release[1], True, "e23")
        lines.append(f"| {CASE_LABEL[case]} | {fmt(lean[0])} / {fmt(lean[1])} | {fmt(pools[0])} / {fmt(pools[1])} | "
                     f"{fmt(compact[0])} / {fmt(compact[1])} | {fmt(packed[0])} / {fmt(packed[1])} | "
                     f"{100 * (1 - packed[0] / compact[0]):.1f} % / {100 * (1 - packed[1] / compact[1]):.1f} % | "
                     f"{fmt(e23[0])} / {fmt(e23[1])} | "
                     f"{100 * (1 - e23[0] / packed[0]):.1f} % / {100 * (1 - e23[1] / packed[1]):.1f} % |")
    return "\n".join(lines)


# ---------------------------------------------------------------- E23: G1 drops P (G1 and G2 32 B)
E23_CAMPAIGN = "perf_g1_nop_v2"         # the final E23 build d2b5e98 (perf_g1_nop: first build d710fb8)
# opt_validate / ab_restart runs of the E23 work: (label, validate name prefix, ab_restart directory)
E23_GATE_RUNS = (("改动前构建 1b52dd2(今天的数值参数)", "e23_pre_1b52dd2", "ab_e23_pre_1b52dd2"),
                 ("E23 第一版 d710fb8", "e23_g1_32b", "ab_e23_g1_32b"),
                 ("E23 最终 d2b5e98", "e23_g1_32b_v2", "ab_e23_g1_32b_v2"))
DENSITY_ULP = 2.0 ** -14                 # float32 spacing of rho in [512, 1024)
E23_LINK_LABEL = {"s0_to_s1": "s0 → s1", "s1_to_s0": "s1 → s0"}
# at K = 2 each sim has one inbound direction: s0 receives from s1 into its trailing ghosts, s1 from s0 into its leading
E23_INBOUND = {0: "trailing", 1: "leading"}


def campaign_records(campaign: str) -> list:
    path = ROOT / campaign / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def e23_build_labels(campaign: str = E23_CAMPAIGN) -> dict:
    """config -> 'build <head>' from the build provenance the worker records (opt_campaign build_provenance)."""
    labels = {}
    for record in campaign_records(campaign):
        build = (record.get("result") or {}).get("build") or {}
        if build.get("head"):
            labels.setdefault(record["config"], f"{build['head']}" + ("+dirty" if build.get("experiment_v6_dirty") else ""))
    return labels


def e23_gate_values(name: str) -> dict:
    """audit / single worst and K = 4 of an opt_validate run (absent steps -> None)."""
    path = ROOT / f"validate_{name}" / "verdict.json"
    if not path.exists():
        return {}
    verdict = json.loads(path.read_text(encoding="utf-8"))
    return {step: (verdict[step].get("worst"), verdict[step].get("pass")) for step in ("audit", "single", "k4")
            if verdict.get(step)}


def e23_ab_values(directory: str) -> dict:
    path = ROOT / directory / "result.json"
    if not path.exists():
        return {}
    return {case["case"]: (case["worst_ratio"], case["pass"]) for case in json.loads(path.read_text(encoding="utf-8"))}


def e23_gates_table() -> str:
    """Worst ratios of the gates: the v6_opt.md release numbers (old numerics xi 0.1 / eps^2 0.01 h^2), the
    pre-change build with today's numerics (same day, same tools) and the E23 build; the audit three times each."""
    def audits(prefix: str) -> list:
        values = []
        for name in (prefix, prefix + "_audit2", prefix + "_audit3"):
            entry = e23_gate_values(name).get("audit")
            if entry:
                values.append(entry)
        return values

    def show(entries: list, threshold: float) -> str:
        return ", ".join(f"{value:.2f}{'' if passed else ' ✗'}" for value, passed in entries) + f"(门槛 {threshold:g})"
    release = e23_gate_values("release_final")
    rows = [("v6_opt.md 发布组合(旧数值参数)", [release.get("audit")] if release.get("audit") else [],
             release.get("single"), release.get("k4"), e23_ab_values("ab_release_final"))]
    rows += [(label, audits(prefix), e23_gate_values(prefix).get("single"), e23_gate_values(prefix).get("k4"),
              e23_ab_values(ab_directory)) for label, prefix, ab_directory in E23_GATE_RUNS]
    lines = ["| 构建 | 审计(每次) | 单步 | A/B 2-D 1M | A/B 3-D 1M | K = 4 |", "|---|---|---|---|---|---|"]
    for label, audit, single, k4, ab in rows:
        def ab_cell(case):
            entry = ab.get(case)
            return f"{entry[0]:.2f}{'' if entry[1] else ' ✗'}" if entry else "—"
        lines.append(f"| {label} | {show(audit, 2.0) if audit else '—'} | "
                     f"{f'{single[0]:.2f}' + ('' if single[1] else ' ✗') if single else '—'} | "
                     f"{ab_cell('cavity2d_1m')} | {ab_cell('cavity3d_1m')} | "
                     f"{('通过' if k4[1] else '未过') if k4 else '—'} |")
    return NEWLINE.join(lines)


def e23_audit_windows(prefix: str) -> list:
    """(repeat 1-3, worst, window_report) of the three audits of one build."""
    out = []
    for repeat, name in enumerate((prefix, prefix + "_audit2", prefix + "_audit3"), start=1):
        verdict_path = ROOT / f"validate_{name}" / "verdict.json"
        reports = sorted((ROOT / f"validate_{name}" / "audit" / "analysis" / "cavity2d_1m").glob("*/window_report.json"))
        if verdict_path.exists() and reports:
            out.append((repeat, json.loads(verdict_path.read_text(encoding="utf-8"))["audit"]["worst"],
                        json.loads(reports[0].read_text(encoding="utf-8"))))
    return out


def e23_audit_diagnostics_table() -> str:
    """The audit's worst value is a crossing-window density_rms ratio. Per audit and pair: the control group's
    density difference rms in float32 ULP of rho ~ 1000 (K = 2 vs K = 1, and the noise pair K = 1 vs K = 1) and its
    ratio, and each crossing group's density / acceleration ratio divided by the control group's (particles far from
    the seam): ~1 = the elevation is global, not a seam effect."""
    groups = ("flagged", "departed", "arrived")
    lines = ["| 构建 | 审计 | 最差 | 对 | control:K2 − K1 rms(ULP) | K1 − K1 噪声 rms(ULP) | 噪声中位数(ULP) | control 比值 | "
             "density:越界 / 迁出 / 迁入 ÷ control | acceleration:越界 / 迁出 / 迁入 ÷ control |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, prefix, _ab in (("1b52dd2", "e23_pre_1b52dd2", None), ("d710fb8", "e23_g1_32b", None),
                               ("d2b5e98", "e23_g1_32b_v2", None)):
        for repeat, worst, report in e23_audit_windows(prefix):
            for index, pair in enumerate(report["pairs"]):
                statistics = pair["statistics"]
                density = statistics["control"]["density"]
                control = density["ratio"]["rms"]
                control_acceleration = statistics["control"]["acceleration"]["ratio"]["rms"]
                lines.append(
                    f"| {label} | {repeat} | {worst:.2f} | B{index + 1} | "
                    f"{density['test']['rms'] / DENSITY_ULP:.2f} | {density['noise']['rms'] / DENSITY_ULP:.2f} | "
                    f"{density['noise']['p50'] / DENSITY_ULP:.0f} | {control:.2f} | "
                    + " / ".join(f"{statistics[g]['density']['ratio']['rms'] / control:.2f}" for g in groups) + " | "
                    + " / ".join(f"{statistics[g]['acceleration']['ratio']['rms'] / control_acceleration:.2f}"
                                 for g in groups) + " |")
    return NEWLINE.join(lines)


def e23_ab_detail_table(field_group=("near migrants", "5")) -> str:
    """ab_restart, 3-D 1M, k = 5, near-migrant group (p90): test medians (base x test pairs) and floors (base x
    base) of every A/B run of the E23 work; the base runs never use packed replicas, so the floor is the same
    configuration in every run."""
    group, step = field_group
    lines = ["| A/B 运行 | 场 | test 中位数 | 底(中位数) | 比值 |", "|---|---|---|---|---|"]
    for label, _prefix, directory in E23_GATE_RUNS:
        path = ROOT / directory / "result.json"
        if not path.exists():
            continue
        cases = {case["case"]: case for case in json.loads(path.read_text(encoding="utf-8"))}
        case = cases.get("cavity3d_1m")
        if not case:
            continue
        for field in ("velocity", "acceleration"):
            entry = case["steps"][step]["groups"][group]["fields"][field]
            lines.append(f"| {label} | {field} | {entry['test_median']:.3e} | {entry['floor_median']:.3e} | {entry['ratio']:.2f} |")
    return NEWLINE.join(lines)


def e23_formula_table(campaign: str = E23_CAMPAIGN) -> str:
    """Host bytes per link and frame on the depth-1 anatomy frames of every trial: B_host = 32 (n0 + n1) + 44 n_mig
    + 16 NyNz + 16 from the count words of the same frame, against the worker's measured count-aware copy (frame
    means pooled over the trials, weighted by frames). The pre-change build (base, G1 36 B) must come out 4 n0 above
    the formula; the segment model (min(size, count x stride) per segment) is exact in both builds."""
    totals: dict = {}
    for record in campaign_records(campaign):
        if not record.get("ok"):
            continue
        for link, values in record["result"].get("links", {}).items():
            formula = values.get("formula")
            if not formula or not formula.get("frames"):
                continue
            key = (record["case"], record["config"], link)
            entry = totals.setdefault(key, {"frames": 0, "n0": 0.0, "n1": 0.0, "n_mig": 0.0, "measured": 0.0,
                                            "predicted": 0.0, "segment": 0.0, "max": 0, "max_segment": 0,
                                            "voxels": formula["cross_section_voxels"]})
            frames = formula["frames"]
            entry["frames"] += frames
            for name, field in (("n0", "n0_mean"), ("n1", "n1_mean"), ("n_mig", "n_mig_mean"),
                                ("measured", "measured_mean"), ("predicted", "predicted_mean"),
                                ("segment", "segment_model_mean")):
                entry[name] += formula[field] * frames
            entry["max"] = max(entry["max"], formula["max_abs_difference"])
            entry["max_segment"] = max(entry["max_segment"], formula["max_abs_segment_difference"])
    lines = ["| case | build | link | frames | n0 | n1 | n_mig | NyNz | predicted B | measured B | measured − predicted | "
             "max per-frame difference | segment model − measured (max) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    labels = e23_build_labels(campaign)
    for case in CASE_ORDER:
        for config, label in (("release", labels.get("release", "E23")), ("base", labels.get("base", "base"))):
            for link in ("s0_to_s1", "s1_to_s0"):
                entry = totals.get((case, config, link))
                if not entry:
                    continue
                frames = entry["frames"]
                mean = {name: entry[name] / frames for name in ("n0", "n1", "n_mig", "measured", "predicted", "segment")}
                lines.append(f"| {CASE_LABEL[case]} | {label} | {E23_LINK_LABEL[link]} | {frames} | {fmt(mean['n0'])} | "
                             f"{fmt(mean['n1'])} | {fmt(mean['n_mig'], 2)} | {entry['voxels']:,} | {fmt(mean['predicted'])} | "
                             f"{fmt(mean['measured'])} | {fmt(mean['measured'] - mean['predicted'])} | {entry['max']:,} | "
                             f"{fmt(mean['segment'] - mean['measured'])} ({entry['max_segment']}) |")
    return NEWLINE.join(lines)


def e23_install_table(campaign: str = E23_CAMPAIGN, cases=("2d_16m", "3d_8m")) -> str:
    """GPU time per step of the three install kernels (bench timestamps, depth-1 anatomy frames pooled over the
    three trials): expand_ghost_lists, install_migrations_<dir>, append_departed and their sum (c_append_departed_end
    - c_start), median and p95 per sim = per inbound direction at K = 2."""
    def quantiles(values):
        ordered = sorted(values)
        return statistics.median(ordered), ordered[int(0.95 * (len(ordered) - 1))]
    pooled: dict = {}
    for record in campaign_records(campaign):
        if not record.get("ok") or record["case"] not in cases:
            continue
        for index, series in enumerate(record["result"].get("anatomy_series", [])):
            for key, values in series.items():
                pooled.setdefault((record["case"], record["config"], index, key), []).extend(values)
    lines = ["| case | build | sim (inbound direction) | frames | expand_ghost_lists µs median / p95 | "
             "install_migrations µs | append_departed µs | sum µs |", "|---|---|---|---|---|---|---|---|"]
    labels = e23_build_labels(campaign)
    for case in cases:
        for config, label in (("release", labels.get("release", "E23")), ("base", labels.get("base", "base"))):
            for index, direction in E23_INBOUND.items():
                expand = pooled.get((case, config, index, "expand_lists_us"))
                install = pooled.get((case, config, index, f"install_{direction}_us"))
                append = pooled.get((case, config, index, "append_departed_us"))
                chain = pooled.get((case, config, index, "install_chain_us"))
                if not expand or not chain:
                    continue
                cells = []
                for values in (expand, install, append, chain):
                    if values:
                        median, p95 = quantiles(values)
                        cells.append(f"{median:.2f} / {p95:.2f}")
                    else:
                        cells.append("—")
                lines.append(f"| {CASE_LABEL[case]} | {label} | s{index} ({direction}) | {len(chain)} | " + " | ".join(cells) + " |")
    return NEWLINE.join(lines)


def direct_staging_table() -> str:
    path = ROOT / "direct_staging_v2" / "results.json"
    if not path.exists():
        path = ROOT / "direct_staging" / "results.json"
    if not path.exists():
        return "(not run)"
    results = json.loads(path.read_text(encoding="utf-8"))
    lines = ["| workload | live KiB | DMA KiB (live / 0.8) | today: kernel + readback DMA µs | direct HOST_CACHED: kernel µs | "
             "direct HOST_COHERENT: kernel µs | CPU read of the live bytes: cached / coherent µs |",
             "|---|---|---|---|---|---|---|"]
    for name, entry in results["workloads"].items():
        modes = entry["modes"]
        lines.append(f"| {name} ({entry['threads']} threads × {entry['records']}) | {fmt(entry['live_bytes'] / 1024)} | "
                     f"{fmt(entry['dma_bytes'] / 1024)} | {fmt(modes['vram']['kernel_us'])} + {fmt(modes['vram']['dma_us'])} | "
                     f"{fmt(modes['cached']['kernel_us'])} | {fmt(modes['coherent']['kernel_us'])} | "
                     f"{fmt(modes['cached']['host_read_us'])} / {fmt(modes['coherent']['host_read_us'])} |")
    return "\n".join(lines)


def delta_density_table(directory_name: str = "delta_density") -> str:
    """Per scenario / variant: throughput (different cards: indicative only), Ghia errors, kinetic energy
    vs the baseline, and the noise / quantization series (median and max over the samples)."""
    directory = ROOT / directory_name
    if not directory.exists():
        return "(not run)"
    lines = ["| scenario | variant | samples | final t (s) | alive | KE rel. diff vs baseline (max over t) | Ghia u rms / v rms | "
             "P noise rms Pa, all (median / max) | interior (median / max) | within 4 h of a wall (median / max) | "
             "ρ stored value changed per step (median) | distinct P values near ρ₀ / particles (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scenario in ("developed", "rest"):
        results = {}
        for variant in ("baseline", "delta"):
            path = directory / f"{scenario}_{variant}.json"
            if path.exists():
                results[variant] = json.loads(path.read_text(encoding="utf-8"))
        for variant, result in results.items():
            noise = result.get("noise", [])
            noise_values = [entry["pressure_noise_rms"] for entry in noise]
            changed = [entry["fraction_density_changed"] for entry in noise if "fraction_density_changed" in entry]
            levels = [entry["pressure_levels_near_rest"] for entry in noise]
            particles = [entry["particles_near_rest"] for entry in noise]
            ke_difference = "—"
            if variant == "delta" and "baseline" in results:
                base = {sample["step"]: sample["kinetic_energy"] for sample in results["baseline"]["samples"]}
                differences = [abs(sample["kinetic_energy"] / base[sample["step"]] - 1)
                               for sample in result["samples"] if sample["step"] in base and base[sample["step"]] > 0
                               and sample["step"] > 0]
                ke_difference = f"{100 * max(differences):.4f} %" if differences else "—"
            ghia = result.get("ghia", {})

            def median_max(key, digits=2):
                values = [entry[key] for entry in noise if key in entry]
                if not values:
                    return "—"
                return f"{fmt(statistics.median(values), digits)} / {fmt(max(values), digits)}"
            lines.append(
                f"| {scenario} | {variant} | {len(noise)} | {fmt(result.get('final_time'), 3)} | {result.get('final_alive')} | "
                f"{ke_difference} | {fmt(ghia.get('u_rms'), 4)} / {fmt(ghia.get('v_rms'), 4)} | "
                f"{fmt(statistics.median(noise_values), 2) if noise_values else '—'} / {fmt(max(noise_values), 2) if noise_values else '—'} | "
                f"{median_max('pressure_noise_interior_rms', 3)} | {median_max('pressure_noise_wall_rms')} | "
                f"{fmt(100 * statistics.median(changed), 2) + ' %' if changed else '—'} | "
                f"{statistics.median(levels):,.0f} / {statistics.median(particles):,.0f} |")
    return "\n".join(lines)


def delta_density_noise_split(directory_name: str) -> str:
    """Spatial pressure noise of the final states split by region: rms of P minus its
    Shepard average over h, interior fluid vs fluid within 4 h of a wall vs within 4 h of
    the lid (every 25th fluid particle)."""
    import numpy as np
    from scipy.spatial import cKDTree
    from experiment.seam_audit.delta_density_eval import shepard
    directory = ROOT / directory_name
    smoothing_length = 0.005
    lines = ["| scenario | variant | P rms (Pa) | noise rms: all | interior | within 4 h of a wall | within 4 h of the lid | "
             "99.9 % quantile of the residual |", "|---|---|---|---|---|---|---|---|"]
    for scenario in ("developed", "rest"):
        for variant in ("baseline", "delta"):
            path = directory / f"{scenario}_{variant}_final_state.npz"
            if not path.exists():
                continue
            data = np.load(path)
            positions = data["position"].astype(np.float64)
            pressure = data["pressure"].astype(np.float64)
            density = data["density"]
            fluid = data["fluid"]
            volumes = 1.0e-3 / density            # m = rho0 dx^2 = 1e-3 kg per unit depth
            tree = cKDTree(positions)
            # the same subsample as delta_density_eval's time series (every fluid_count // 20000-th
            # fluid particle), so the final-state rows match the series' last samples; the all / wall /
            # lid columns are dominated by a few outliers and change with the subsample, interior does not
            sample = np.flatnonzero(fluid)[::max(1, int(fluid.sum()) // 20000)]
            residual = pressure[sample] - shepard(tree, positions, volumes, pressure[:, None], positions[sample],
                                                  smoothing_length)[:, 0]
            extent = np.maximum(np.abs(positions[sample, 0]), np.abs(positions[sample, 1]))
            near_wall = extent > 0.5 - 4 * smoothing_length
            near_lid = positions[sample, 1] > 0.5 - 4 * smoothing_length

            def rms(mask):
                return float(np.sqrt(np.nanmean(residual[mask] ** 2)))
            lines.append(f"| {scenario} | {variant} | {np.sqrt(np.mean(pressure[fluid] ** 2)):.1f} | "
                         f"{rms(np.ones_like(near_wall)):.2f} | {rms(~near_wall):.3f} | {rms(near_wall):.2f} | "
                         f"{rms(near_lid):.2f} | {np.nanquantile(np.abs(residual), 0.999):.1f} |")
    return "\n".join(lines)


def delta_density_perf_table() -> str:
    path = ROOT / "delta_density" / "perf.jsonl"
    if not path.exists():
        return "(not run)"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines = ["| case | baseline fps | delta fps | delta / baseline (trial-wise) | drift |", "|---|---|---|---|---|"]
    for case in sorted({record["case"] for record in records}, key=lambda name: int(name.rstrip("m"))):
        base = {record["trial"]: record["fps"] for record in records if record["case"] == case and record["variant"] == "baseline"}
        delta = {record["trial"]: record["fps"] for record in records if record["case"] == case and record["variant"] == "delta"}
        ratios = [delta[trial] / base[trial] for trial in base if trial in delta]
        drifts = sorted({record["drift"] for record in records if record["case"] == case})
        lines.append(f"| {case} | {statistics.mean(base.values()):,.1f} ± {statistics.stdev(base.values()):.1f} | "
                     f"{statistics.mean(delta.values()):,.1f} ± {statistics.stdev(delta.values()):.1f} | "
                     f"{100 * statistics.mean(ratios):.2f} ± {100 * statistics.stdev(ratios):.2f} % | {drifts} |")
    return NEWLINE.join(lines)


def final_table(campaign: str = "perf_final") -> str:
    summary = load_summary(campaign)
    configs = ["v5eq", "l2", "l1x", "packed", "release", "release_compact"]
    lines = ["| case | config | fps | vs (1,2) baseline | vs v5 | phase C µs | DMA KiB | host KiB | t_tr µs | valid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for case in CASE_ORDER:
        for config in configs:
            row = summary.get((case, config))
            if not row:
                continue
            versus_l2, versus_l2_std = trial_ratio(campaign, case, config, "l2")
            versus_v5, versus_v5_std = trial_ratio(campaign, case, config, "v5eq")
            lines.append(f"| {CASE_LABEL[case]} | {config} | {fmt(row['fps_mean'])} ± {fmt(row['fps_std'])} | "
                         f"{fmt(100 * versus_l2, 2)} ± {fmt(100 * versus_l2_std, 2)} % | "
                         f"{fmt(100 * versus_v5, 2)} ± {fmt(100 * versus_v5_std, 2)} % | {fmt(row['phase_c_us'])} | "
                         f"{fmt(row['dma_bytes'] / 1024)} | {fmt(row['host_bytes'] / 1024)} | {fmt(row['t_tr_us'])} | "
                         f"{'yes' if row['valid'] else '**NO**'} |")
    return NEWLINE.join(lines)


def ab_table(directory: str = "ab_release_final") -> str:
    """Repeated A/B equivalence test (ab_restart.py): per case and step, the acceleration and density
    statistics of the floor (identical pairs, shuffled-order pairs) and of the release-vs-base pairs."""
    path = ROOT / directory / "result.json"
    if not path.exists():
        return "(not run)"
    results = json.loads(path.read_text(encoding="utf-8"))
    lines = ["| case | k | crossed | group (statistic) | acceleration: identical / shuffled / release (medians) | "
             "ratio | ρ flips: floor / release (medians) |", "|---|---|---|---|---|---|---|"]
    for result in results:
        for step, step_result in result["steps"].items():
            for group_name, group in step_result["groups"].items():
                entry = group["fields"]["acceleration"]
                lines.append(f"| {result['case']} | {step} | {step_result['migrated_so_far']} | {group_name} "
                             f"({group['statistic']}, n = {group['n']:,}) | {entry['identical_median']:.2e} / "
                             f"{entry['shuffled_median']:.2e} / {entry['test_median']:.2e} | {entry['ratio']:.2f} | "
                             f"{group['density_flips_floor_median']:g} / {group['density_flips_test_median']:g} |")
    verdicts = "; ".join(f"{result['case']}: {'pass' if result['pass'] else 'FAIL'}, worst ratio "
                         f"{result['worst_ratio']:.2f}" for result in results)
    return "\n".join(lines) + "\n\nVerdict: " + verdicts


if __name__ == "__main__":
    raise SystemExit(main())
