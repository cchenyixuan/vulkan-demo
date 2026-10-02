"""
link_inventory_tables.py - markdown tables of docs/seam_audit/v6_design.md §2
and the byte estimates of §4 from link_inventory.py results.

Input: <root>/<case>_<config>.json for case in CASES and config in CONFIGS
(produced by link_inventory.py with the matching V6_KEEP_DEPARTED /
V6_GHOST_LAYERS). Per link per frame = mean of the two directions of the K=2
chain (s0 trailing = link s0->s1, s1 leading = link s1->s0).

Usage:
  .venv/Scripts/python.exe -m experiment.seam_audit.link_inventory_tables \\
      --root logs/seam_audit/link_inventory --out <markdown file>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

CASES = (("2d_1m", "2-D 1M"), ("2d_4m", "2-D 4M"), ("2d_16m", "2-D 16M"), ("3d_8m", "3-D 8M"))
CONFIGS = (("v5", "v5 = (0,1)"), ("k1_l1", "(1,1)"), ("k1_l2", "(1,2)"))
FIELD_ORDER = ("position_voxel_id", "velocity_mass", "density_pressure", "acceleration", "shift",
               "material", "correction_inverse", "density_gradient_kernel_sum", "extension_fields")
REGION_ORDER = ("replica+migrant", "G1 replica", "G2 replica", "migrant",
                "slot counts · G 列", "slot counts · G1 列", "slot counts · G2 列",
                "slot index · G 列", "slot index · G1 列", "slot index · G2 列",
                "count word", "frame stamp")
# Bytes per particle a field must carry (docs §3): position xyz + w, velocity xyz + mass,
# rho (the P half of the vec2 is dead for G2 / G1 / migrants), material.
NEEDED_FIELDS = {"position_voxel_id", "velocity_mass", "density_pressure", "material"}


def load(root: pathlib.Path, case: str, config: str):
    path = root / f"{case}_{config}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def per_link(document: dict) -> dict:
    """{(region, field): {dma, host, count_mean, count_max, slots, stride}} averaged over
    the two link directions."""
    keys = sorted(document["layouts"])
    merged: dict = {}
    two_layers = document.get("ghost_layers", 1) == 2
    for key in keys:
        status_seen = 0
        for entry in document["layouts"][key]:
            region = entry["region"]
            field = entry["buffer"] if region not in ("count word",) else f"word{status_seen}"
            if region == "count word":
                status_seen += 1
            # voxel lists: one segment covers every ghost column of the direction
            # (dense per voxel, equal halves for two columns) -> one row per column
            parts = [(region, 1.0)]
            if region in ("slot counts", "slot index"):
                parts = ([(region + " · G1 列", 0.5), (region + " · G2 列", 0.5)] if two_layers
                         else [(region + " · G 列", 1.0)])
            for part_region, share in parts:
                slot = merged.setdefault((part_region, field), {
                    "dma": 0.0, "host": 0.0, "count_mean": 0.0, "count_max": 0,
                    "slots": entry.get("slots"), "stride": entry.get("stride")})
                slot["dma"] += share * entry["size"] / len(keys)
                slot["host"] += share * entry["host_bytes_mean"] / len(keys)
                if entry.get("count_mean") is not None:
                    slot["count_mean"] += entry["count_mean"] / len(keys)
                    slot["count_max"] = max(slot["count_max"], entry.get("count_max") or 0)
    return merged


def kilobytes(value: float) -> str:
    if value != value:                      # nan
        return "–"
    if 0 < abs(value) < 102.4:
        return f"{value:,.0f} B"
    return f"{value / 1024:,.1f}"


def region_rows(merged: dict) -> list:
    rows = []
    for region in REGION_ORDER:
        fields = [key for key in merged if key[0] == region]
        fields.sort(key=lambda key: FIELD_ORDER.index(key[1]) if key[1] in FIELD_ORDER else 99)
        for key in fields:
            rows.append(key)
    return rows


def case_table(root: pathlib.Path, case: str, title: str) -> list:
    documents = {config: load(root, case, config) for config, _ in CONFIGS}
    if documents["v5"] is None:
        return [f"*{title}: no inventory*", ""]
    merged = {config: per_link(document) for config, document in documents.items() if document}
    base = documents["v5"]
    lines = [f"#### {title}",
             "",
             f"face = {base['face_voxels']:,} voxel,C = {base['max_particles_per_voxel']},"
             f"C_inc = {base['max_incoming_per_voxel']},f = {base['switches'].get('V6_GHOST_POOL_FACTOR')};"
             f"v5 池 {base['ghost_pool_per_direction']:,} 槽/方向"
             + (f";(1,2) 池 {documents['k1_l2']['ghost_pool_per_direction']:,} 槽"
                f"(R = {documents['k1_l2']['replica_region_size']:,})" if documents.get("k1_l2") else "")
             + f";采样 {base['samples']} 帧(预热 {base['warmup']} 帧后每 {base['sample_every']} 帧一次)。",
             "",
             "| 段 | 字段 | stride | v5 / (1,1) DMA | v5 host | (1,1) host | (1,2) DMA | (1,2) host |",
             "|---|---|---|---|---|---|---|---|"]
    keys = []
    for config in ("v5", "k1_l2"):
        if config in merged:
            for key in region_rows(merged[config]):
                if key not in keys:
                    keys.append(key)
    keys.sort(key=lambda key: (REGION_ORDER.index(key[0]),
                               FIELD_ORDER.index(key[1]) if key[1] in FIELD_ORDER else 99, key[1]))
    totals = {config: {"dma": 0.0, "host": 0.0} for config in merged}
    for key in keys:
        region, field = key
        cells = []
        for config in ("v5", "k1_l1", "k1_l2"):
            entry = merged.get(config, {}).get(key)
            if config == "v5":
                cells.append(kilobytes(entry["dma"]) if entry else "–")
                cells.append(kilobytes(entry["host"]) if entry else "–")
            elif config == "k1_l1":
                cells.append(kilobytes(entry["host"]) if entry else "–")
            else:
                cells.append(kilobytes(entry["dma"]) if entry else "–")
                cells.append(kilobytes(entry["host"]) if entry else "–")
        for config in merged:
            entry = merged[config].get(key)
            if entry:
                totals[config]["dma"] += entry["dma"]
                totals[config]["host"] += entry["host"]
        stride = next((merged[config][key]["stride"] for config in merged if key in merged[config]
                       and merged[config][key]["stride"]), None)
        field_label = field if not field.startswith("word") else "计数字"
        lines.append(f"| {region} | {field_label} | {stride or ''} | " + " | ".join(cells) + " |")
    lines.append("| **合计** | | | "
                 + " | ".join([kilobytes(totals['v5']['dma']), kilobytes(totals['v5']['host']),
                               kilobytes(totals.get('k1_l1', {}).get('host', float('nan'))),
                               kilobytes(totals.get('k1_l2', {}).get('dma', float('nan'))),
                               kilobytes(totals.get('k1_l2', {}).get('host', float('nan')))]) + " |")
    lines.append("")
    # live counts
    count_lines = []
    for config, label in CONFIGS:
        if config not in merged:
            continue
        parts = []
        for region in ("replica+migrant", "G1 replica", "G2 replica", "migrant"):
            entry = merged[config].get((region, "position_voxel_id"))
            if entry:
                parts.append(f"{region} {entry['count_mean']:,.0f}(采样最大 {entry['count_max']:,},"
                             f"容量 {entry['slots']:,})")
        departed = documents[config].get("departed_per_frame")
        if departed and departed[0].get("mean") is not None and documents[config]["keep_departed"]:
            parts.append("departed/帧 " + "/".join(f"{item['mean']:.2f}(采样最大 {item['max']})"
                                                   for item in departed))
        healths = documents[config].get("pool_health") or []
        if healths:
            parts.append("整段运行峰值(PoolHealth)departed " + "/".join(
                str(health.get("peak_departed_count", 0)) for health in healths)
                + "(单帧);install 计数峰值(两次 defrag 之间累计)" + "/".join(str(health.get("peak_migration_count", 0))
                                             for health in healths))
        worker = documents[config]["worker_bytes_per_frame"]
        parts.append("worker 计数器 " + "/".join(kilobytes(value) for value in worker.values() if value) + " KB")
        count_lines.append(f"- {label}:" + ";".join(parts))
    lines += count_lines + [""]
    return lines


def totals(root: pathlib.Path) -> dict:
    out = {}
    for case, _ in CASES:
        out[case] = {}
        for config, _ in CONFIGS:
            document = load(root, case, config)
            if document is None:
                continue
            merged = per_link(document)
            out[case][config] = {"merged": merged, "document": document,
                                 "dma": sum(entry["dma"] for entry in merged.values()),
                                 "host": sum(entry["host"] for entry in merged.values())}
    return out


def region_entry(merged: dict, region: str):
    return merged.get((region, "position_voxel_id"))


def option_estimates(entry: dict) -> dict:
    """Bytes per link per frame (DMA, host) saved by each §4 option, for one
    case and configuration (mean of both link directions)."""
    merged, document = entry["merged"], entry["document"]
    face = document["face_voxels"]
    capacity_c = document["max_particles_per_voxel"]
    index_dma = sum(value["dma"] for key, value in merged.items() if key[0].startswith("slot index"))
    index_host = sum(value["host"] for key, value in merged.items() if key[0].startswith("slot index"))
    columns = round(index_dma / (4 * face * capacity_c))
    replica_regions = [region for region in ("replica+migrant", "G1 replica", "G2 replica")
                       if region_entry(merged, region)]
    live_replicas = sum(region_entry(merged, region)["count_mean"] for region in replica_regions)
    replica_slots = sum(region_entry(merged, region)["slots"] for region in replica_regions)
    estimates = {}
    # (a1) index array -> one base word per voxel and column (counts stay): lists identical
    estimates["a_base_count"] = (index_dma - 4 * face * columns, index_host - 4 * face * columns)
    # (a2) live entries only (sender compacts, receiver expands)
    estimates["a_live_only"] = (index_dma - 4 * face * columns - 4 * replica_slots,
                                index_host - 4 * face * columns - 4 * live_replicas)
    # (a3) no index array at all: receiver rebuilds the lists from the replicas' .w
    estimates["a_rebuild"] = (index_dma, index_host)
    # (b) mass + voxel id of every replica (8 B, needs a packed format); for the
    # mixed v5 pool the id is still needed by install -> mass only
    # the v5 mixed pool also carries migrants, whose mass is state and the slot
    # sentinel, and install needs .w: no saving there unless install rebuilds the
    # migrant mass from the material table (see the doc)
    if "replica+migrant" in replica_regions:
        estimates["b_mass_id"] = (0.0, 0.0)
    else:
        estimates["b_mass_id"] = (8 * replica_slots, 8 * live_replicas)
    # (c) G2: drop .w and P (8 B per G2 replica, packed format)
    g2 = region_entry(merged, "G2 replica")
    estimates["c_g2"] = (8 * g2["slots"], 8 * g2["count_mean"]) if g2 else (0.0, 0.0)
    # (d) migrant / mixed region: transport only the 4 needed fields (44 B) of 140 B;
    # plain SoA segments, no packing (extension_fields for the audit: +16 B)
    dead_per_slot = 140 - 44
    mixed = region_entry(merged, "replica+migrant")
    migrant = region_entry(merged, "migrant")
    if mixed:
        estimates["d_dead_fields"] = (dead_per_slot * mixed["slots"], dead_per_slot * mixed["count_mean"])
    elif migrant:
        estimates["d_dead_fields"] = (dead_per_slot * migrant["slots"], dead_per_slot * migrant["count_mean"])
    else:
        estimates["d_dead_fields"] = (0.0, 0.0)
    # (e) ghost pool factor down to (peak live / capacity) x 1.25, DMA of the per-particle
    # segments after (d): replicas 44 B per slot, migrants 44 B
    factor = float(document["switches"].get("V6_GHOST_POOL_FACTOR", "1") or 1)
    peaks = [region_entry(merged, region)["count_max"] / region_entry(merged, region)["slots"]
             for region in replica_regions]
    rule_factor = math.ceil(factor * max(peaks) * 1.25 * 100) / 100
    safe_factor = min(factor, rule_factor)
    estimates["_rule_factor"] = rule_factor
    per_particle_dma_after_d = 44 * (replica_slots + (migrant["slots"] if migrant else 0))
    estimates["e_factor"] = (per_particle_dma_after_d * (1 - safe_factor / factor), 0.0)
    estimates["_safe_factor"] = safe_factor
    estimates["_factor"] = factor
    estimates["_peak_utilisation"] = max(peaks)
    estimates["_columns"] = columns
    return estimates


def savings_lines(root: pathlib.Path) -> list:
    """§4 estimates per case (KB per link per frame, mean of both directions, DMA / host)."""
    data = totals(root)
    lines = ["| 算例 | 配置 | 当前 DMA / host | (a1) index → (base,count) | (a2) 只发 live 项 | (a3) 不发 index、按 .w 重建 | (b) 质量 + vid(要打包) | (c) G2 的 .w + P(要打包,.w 与 (b) 重叠) | (d) migrant / 混合区只传 4 个字段 | (e) f → 安全下限 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    combined = ["| 算例 | 配置 | 当前 DMA / host (KB) | 精简后 DMA / host (KB):(a1)+(d)+(e),不改格式 | 再打包成必需字段(LAYERS=2 replica 32 B,migrant 40 B;混合池仍 44 B) |",
                "|---|---|---|---|---|"]
    for case, title in CASES:
        for config, label in (("v5", "v5 / (1,1)"), ("k1_l2", "(1,2)")):
            entry = data.get(case, {}).get(config)
            if not entry:
                continue
            estimates = option_estimates(entry)
            def signed(value):
                return f"−{kilobytes(value)}" if value >= 0 else f"+{kilobytes(-value)}"

            def cell(key):
                dma, host = estimates[key]
                return f"{signed(dma)} / {signed(host)}"
            lines.append(f"| {title} | {label} | {kilobytes(entry['dma'])} / {kilobytes(entry['host'])} | "
                         + " | ".join(cell(key) for key in ("a_base_count", "a_live_only", "a_rebuild",
                                                            "b_mass_id", "c_g2", "d_dead_fields"))
                         + (f" | f {estimates['_factor']:g} → {estimates['_safe_factor']:g}"
                            f"(采样峰值占用 {estimates['_peak_utilisation']:.2f}):−{kilobytes(estimates['e_factor'][0])} / 0 |"
                            if estimates['_rule_factor'] < estimates['_factor'] else
                            f" | 规则给 f = {estimates['_rule_factor']:g} > 当前 {estimates['_factor']:g}"
                            f"(采样峰值占用 {estimates['_peak_utilisation']:.2f}):不省 |"))
            lean_dma = (entry["dma"] - estimates["a_base_count"][0] - estimates["d_dead_fields"][0]
                        - estimates["e_factor"][0])
            lean_host = entry["host"] - estimates["a_base_count"][1] - estimates["d_dead_fields"][1]
            # packed records (no double counting): LAYERS=2 replicas 44 -> 32 B (mass, .w, P
            # dropped), v5 mixed pool 44 -> 40 B (mass dropped; .w for install, P for C5),
            # migrants 44 -> 40 B (P dropped)
            merged = entry["merged"]
            packed_dma, packed_host = lean_dma, lean_host
            for region, record in (("G1 replica", 32), ("G2 replica", 32),
                                   ("replica+migrant", 44), ("migrant", 40)):
                reference = region_entry(merged, region)
                if reference:
                    factor_scale = estimates["_safe_factor"] / estimates["_factor"]
                    packed_dma -= (44 - record) * reference["slots"] * factor_scale
                    packed_host -= (44 - record) * reference["count_mean"]
            combined.append(f"| {title} | {label} | {kilobytes(entry['dma'])} / {kilobytes(entry['host'])} | "
                            f"{kilobytes(lean_dma)} / {kilobytes(lean_host)}"
                            f"({100 * lean_dma / entry['dma']:.0f} % / {100 * lean_host / entry['host']:.0f} %) | "
                            f"{kilobytes(packed_dma)} / {kilobytes(packed_host)} |")
    return lines + [""] + combined


PERF_SUMMARIES = ("logs/seam_audit/perf_quiet/summary.json", "logs/seam_audit/perf_final/summary.json")
PERF_CONFIG = {"v5": "v5", "k1_l1": "v6_k1_l1", "k1_l2": "v6_k1_l2"}


def cross_check_lines(root: pathlib.Path) -> list:
    """Inventory totals vs the worker counter and vs the perf campaign's
    measured bytes per link per frame (same switches, longer runs)."""
    perf = {}
    for summary_path in PERF_SUMMARIES:
        path = pathlib.Path(summary_path)
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                perf.setdefault((row["case"], row["config"]), row)
    data = totals(root)
    lines = ["| 算例 | 配置 | 段表 DMA | perf DMA | 段表 host | worker 计数器 | perf host |",
             "|---|---|---|---|---|---|---|"]
    for case, title in CASES:
        for config, label in CONFIGS:
            entry = data.get(case, {}).get(config)
            if not entry:
                continue
            worker = [value for value in entry["document"]["worker_bytes_per_frame"].values() if value]
            row = perf.get((case, PERF_CONFIG[config]), {})
            lines.append(f"| {title} | {label} | {kilobytes(entry['dma'])} | "
                         f"{kilobytes(row['dma_bytes_per_link_frame']) if row else '–'} | "
                         f"{kilobytes(entry['host'])} | "
                         f"{kilobytes(sum(worker) / len(worker)) if worker else '–'} | "
                         f"{kilobytes(row['host_copy_bytes_per_link_frame']) if row else '–'} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="logs/seam_audit/link_inventory")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.root)
    lines = []
    for case, title in CASES:
        lines += case_table(root, case, title)
    lines += ["", "<!-- cross-check -->", ""] + cross_check_lines(root)
    lines += ["", "<!-- savings -->", ""] + savings_lines(root)
    pathlib.Path(arguments.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
