"""E9 paper addenda (user 2026-10-29, three items on top of paper_e9.md):

  1. 64M K8 three hops E8 vs E9 (readback, host incl. memcpy, upload; p50 pooled over links and steps, per-link
     p50 range, and the slowest-link decomposition that fig:loss / the ledger use) + the 41 MB memcpy of the 64M
     packet at K = 2 / 4 / 8: median time and bandwidth (E9, E8 beside it) + the node control (E8 code on an nbj04
     node = e8_D3).
  2. fig:loss(a), 3-D 2M-per-card point (16M K8): how the host part is taken (per step the slowest link by t_own,
     median over the steps = e7_loss_figure.worst_link_medians) and what the other readings give (median of the
     per-step max over links, max / min of the per-link medians, pooled median), E9 beside E8; 1M and 8M per card
     for context.
  3. 8M and 16M K8 (E9): per card the G1 replicas received and the boundary kernels (processed particles, time,
     ns per processed particle) against the deep-interior kernels and the K = 1 cube rates, the G1 redundant time,
     and a composition of the seam work (busy - whole-case ideal) per card.

Inputs: data/{e8,e9}/phase2/hops.json, data/{e8,e9}/loss_3d/ledger.json, data/e9/phase2/kernel_table.json (--all),
data/e8/phase2/kernel_table_all.json, and the step traces named in hops.json (item 2's recompute). Output:
data/e9/paper_e9_addenda.{md,json}; a pointer line is appended to data/e9/paper_e9.md once.

  python docs/cluster_v6/scripts/e9_paper_addenda.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # the e8-3d worktree
DATA = ROOT / "docs/cluster_v6/data"
E7_SCRIPTS_DEFAULT = ROOT.parent / "vulkan-demo-e7/docs/cluster_v6/scripts"

HOP_POINTS = ("3d_n192_K8", "3d_n244_K8", "3d_n392_K8")
NODES = {"e8": {"02_e8_e8B_1721839": "wqd10nbj11g2", "03_e8_e8A_1721840": "wqd10nbj11g2", "04_e8_e8C_1722072": "wqd10nbj11g2",
                "09_e8_e8D3_1725242": "wqd10nbj04g2"},
         "e9": {"01_e9_e9A_1741918": "wqd10nbj04g2", "02_e9_e9B_1741917": "wqd10nbj04g4", "03_e9_e9C_1741921": "wqd10nbj04g2",
                "04_e9_e9D_1741924": "wqd10nbj04g4"}}
LABEL = {"3d_n152_K8": "4M K8", "3d_n192_K8": "8M K8", "3d_n244_K8": "16M K8", "3d_n312_K8": "33M K8", "3d_n392_K8": "64M K8",
         "3d_n416_K8": "76M K8", "3d_n496_K8": "128M K8", "3d_n392_K2": "64M K2", "3d_n392_K4": "64M K4",
         "3d_n192_k2_K2": "weak 2 x 8M", "3d_n192_k4_K4": "weak 4 x 8M", "3d_n192_k8_K8": "weak 8 x 8M"}


def load(tag: str) -> dict:
    hops = json.load(open(DATA / tag / "phase2/hops.json", encoding="utf-8"))
    ledger = {r["point"]: r for r in json.load(open(DATA / tag / "loss_3d/ledger.json", encoding="utf-8"))}
    kt_path = DATA / tag / ("phase2/kernel_table.json" if tag == "e9" else "phase2/kernel_table_all.json")
    kernels = {(e["case"], e["K"]): e for e in json.load(open(kt_path, encoding="utf-8"))}
    return {"hops": hops, "ledger": ledger, "kernels": kernels, "kernel_table": kt_path.relative_to(ROOT).as_posix()}


# ----------------------------------------------------------------------------------------------- item 1
def hops_row(tag: str, point: str, campaign: dict) -> dict:
    p = campaign["hops"][point]
    po, rng = p["pooled"], p["link_p50_range"]
    bytes_host = statistics.median(l["host_copy_bytes"] for l in p["per_link"].values())
    bytes_dma = statistics.median(l["dma_bytes"] for l in p["per_link"].values())
    led = campaign["ledger"].get(point)
    return {"campaign": tag, "point": point, "job": p["job"], "node": NODES[tag].get(p["job"], "?"), "links": p["links"],
            "steps_per_link": p["steps_per_link"], "host_copy_bytes_p50": bytes_host, "dma_bytes_p50": bytes_dma,
            "pooled_p50_us": {k: po[k][0] for k in ("t_chain", "readback", "host", "memcpy", "host_rest", "upload", "readback_dma", "upload_dma")},
            "pooled_p95_us": {k: po[k][1] for k in ("t_chain", "readback", "host", "memcpy", "host_rest", "upload")},
            "link_p50_range_us": {k: rng[k] for k in ("readback", "host", "memcpy", "upload", "t_chain")},
            "memcpy_gbps_at_p50": bytes_host / po["memcpy"][0] / 1e3,
            "slowest_link_median_us": None if led is None else {"t_tr": led["t_tr_us"], "readback": led["t_readback_us"],
                                                                "host": led["t_host_us"], "upload": led["t_upload_us"]},
            "per_link": {name: {"host_copy_bytes": l["host_copy_bytes"], "readback": l["readback"][0], "host": l["host"][0],
                                "memcpy": l["memcpy"][0], "upload": l["upload"][0], "t_chain": l["t_chain"][0]}
                         for name, l in p["per_link"].items()}}


def item1(c8: dict, c9: dict) -> dict:
    out = {"hops_64m_k8": {"e8": hops_row("e8", "3d_n392_K8", c8), "e9": hops_row("e9", "3d_n392_K8", c9)},
           "memcpy_64m_packet": {}, "node_control": {}}
    for tag, c in (("e8", c8), ("e9", c9)):
        out["memcpy_64m_packet"][tag] = {pt: hops_row(tag, pt, c) for pt in ("3d_n392_K2", "3d_n392_K4", "3d_n392_K8")}
        out["node_control"][tag] = {pt: hops_row(tag, pt, c) for pt in ("3d_n416_K8", "3d_n496_K8", "3d_n192_k8_K8", "3d_n192_k2_K2")
                                    if pt in c["hops"]}
    return out


# ----------------------------------------------------------------------------------------------- item 2
def item2(c8: dict, c9: dict, e7_scripts: pathlib.Path) -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(e7_scripts))
    from experiment.v7.analysis import step_trace_model  # noqa: E402
    from e7_loss_figure import worst_link_medians  # noqa: E402
    out = {}
    for tag, c in (("e8", c8), ("e9", c9)):
        for point in HOP_POINTS:
            trace = pathlib.Path(c["hops"][point]["trace"])
            run = step_trace_model.load_run(trace)
            med = worst_link_medians(run)
            metrics = step_trace_model.link_metrics(run)
            index = {name: {int(s): i for i, s in enumerate(v["step"])} for name, v in metrics.items()}
            steps = sorted(set.intersection(*[set(ix) for ix in index.values()]))
            per_link_host_p50 = {name: float(np.nanmedian(v["chain_host"])) for name, v in metrics.items()}
            max_host_by_step, slow_by_host = [], []
            for s in steps:
                hs = {name: float(v["chain_host"][index[name][s]]) for name, v in metrics.items()}
                max_host_by_step.append(max(hs.values()))
                slow_by_host.append(max(hs, key=hs.get))
            pooled = float(np.nanmedian(np.concatenate([v["chain_host"] for v in metrics.values()])))
            led = c["ledger"][point]
            assert abs(led["t_host_us"] - med["t_host"]) < 0.5, (point, led["t_host_us"], med["t_host"])
            out[f"{tag}:{point}"] = {
                "campaign": tag, "point": point, "node": led["host"], "trace": str(trace), "links": med["links"], "steps": med["steps"],
                "per_card_M": led["per_card"] / 1e6,
                "fig_t_host_us": med["t_host"], "fig_t_tr_us": med["t_own"], "fig_t_readback_us": med["t_readback"],
                "fig_t_upload_us": med["t_upload"], "T_B_us": med["T_B"], "r_p50": med["r_own"],
                "slowest_link_share_by_t_tr": med["slowest_link_share"],
                "slowest_link_share_by_host": {str(l): round(slow_by_host.count(l) / len(slow_by_host), 3) for l in sorted(set(slow_by_host))},
                "alt_median_of_per_step_max_host_us": float(np.nanmedian(max_host_by_step)),
                "alt_max_per_link_host_p50_us": max(per_link_host_p50.values()),
                "alt_min_per_link_host_p50_us": min(per_link_host_p50.values()),
                "alt_pooled_host_p50_us": pooled,
                "per_link_host_p50_us": per_link_host_p50,
                "eta": led["eta"]}
    return out


# ----------------------------------------------------------------------------------------------- item 3
def card_rows(tag: str, case: str, c: dict) -> dict:
    k8 = c["kernels"][(case, 8)]
    k1 = c["kernels"][(case, 1)]
    k1_sim = k1["sims"][0]
    k1_rate = {"cd": k1_sim["kernels"]["b_correction_density_interior_end"]["ns_per_particle"],
               "f": k1_sim["kernels"]["b_force_deep_interior_end"]["ns_per_particle"]}
    k1_small_gap_ns = (k1_sim["period_p50_ms"] * 1e3 - k1_sim["kernels"]["b_correction_density_interior_end"]["us_p50"]
                       - k1_sim["kernels"]["b_force_deep_interior_end"]["us_p50"]) * 1e3 / k1_sim["own_particles"]
    point = k8["point"]
    led = c["ledger"][point]
    total_own = sum(s["own_particles"] for s in k8["sims"])
    cards = {card["sim"]: card for card in led["cards"]}
    rows = []
    for s in k8["sims"]:
        kern, g1 = s["kernels"], s["g1"]
        n0 = max(g1["n0_per_side"].values())
        sides = len(g1["n0_per_side"])
        own = s["own_particles"]
        columns = round(own / n0)
        g1_total = sum(g1["n0_per_side"].values())
        proc = {"cdb": 2 * sides * n0 + g1_total, "fb": 3 * sides * n0, "cdi": own - 2 * sides * n0, "fdi": own - 3 * sides * n0}
        proc = {k: (v if v > 0.02 * n0 else 0) for k, v in proc.items()}      # a few leftover particles = an empty dispatch
        us = {"cdb": kern["c_correction_density_boundary_end"]["us_p50"], "fb": kern["c_force_end"]["us_p50"],
              "cdi": kern["b_correction_density_interior_end"]["us_p50"], "fdi": kern["b_force_deep_interior_end"]["us_p50"]}
        rate = {k: (us[k] * 1e3 / proc[k] if proc[k] > 0 else None) for k in us}
        small_us = sum(v["us_p50"] for kk, v in kern.items() if kk.startswith("a_")) + \
            sum(v["us_p50"] for kk, v in kern.items() if kk.startswith("c_") and kk not in ("c_correction_density_boundary_end", "c_force_end"))
        all_kernels_us = sum(v["us_p50"] for v in kern.values())
        busy_us = cards[s["sim"]]["busy_mean"] * 1e3
        ideal_us = led["tau_ref_ms"] * 1e3 * own / (total_own / k8["K"])
        seam_us = busy_us - ideal_us
        g1_redundant_us = g1_total * rate["cdb"] / 1e3
        band_excess_us = ((rate["cdb"] - k1_rate["cd"]) * proc["cdb"] + (rate["fb"] - k1_rate["f"]) * proc["fb"]) / 1e3
        interior_excess_us = sum(((rate[k] - k1_rate[r]) * proc[k] / 1e3) for k, r in (("cdi", "cd"), ("fdi", "f")) if proc[k] > 0)
        gaps_us = busy_us - all_kernels_us
        allowance_us = k1_small_gap_ns * own / 1e3          # the K = 1 reference's own small kernels + gaps, scaled to this card
        explained = g1_redundant_us + band_excess_us + interior_excess_us + small_us + gaps_us - allowance_us
        rows.append({"sim": s["sim"], "columns": columns, "own": own, "sides": sides, "n0_per_side": g1["n0_per_side"], "g1_total": g1_total,
                     "g1_share_of_own": g1_total / own, "processed": proc, "us_p50": us, "ns_per_processed": rate,
                     "small_kernels_us": small_us, "all_kernels_us": all_kernels_us, "busy_us": busy_us, "ideal_us": ideal_us, "seam_us": seam_us,
                     "seam_share_of_period": seam_us / (led["period_ms"] * 1e3),
                     "g1_redundant_us": g1_redundant_us, "g1_redundant_share_of_period": g1_redundant_us / (led["period_ms"] * 1e3),
                     "band_rate_excess_us": band_excess_us, "interior_rate_excess_us": interior_excess_us, "gaps_us": gaps_us,
                     "k1_small_gap_allowance_us": allowance_us, "explained_us": explained, "residual_us": seam_us - explained})
    return {"campaign": tag, "case": case, "point": point, "node": led["host"], "K": 8, "total_own": total_own,
            "period_ms": led["period_ms"], "tau_ref_ms": led["tau_ref_ms"], "busy_mean_ms": led["busy_ms"], "seam_ledger": led["seam"],
            "g1_share_ledger": led["g1_share"], "eta": led["eta"],
            "k1_rate_ns": k1_rate, "k1_small_gap_ns_per_particle": k1_small_gap_ns, "k1_period_ms": k1_sim["period_p50_ms"],
            "cards": rows}


def item3(c8: dict, c9: dict) -> dict:
    return {f"{tag}:{case}": card_rows(tag, case, c) for tag, c in (("e9", c9), ("e8", c8)) for case in ("cavity3d_n192", "cavity3d_n244")}


# ----------------------------------------------------------------------------------------------- markdown
def f0(x) -> str:
    return f"{x:,.0f}"


def rng(pair) -> str:
    return f"{pair[0] / 1e3:.2f}–{pair[1] / 1e3:.2f}"


def md_item1(d: dict) -> list:
    e8, e9 = d["hops_64m_k8"]["e8"], d["hops_64m_k8"]["e9"]
    lines = ["## 1. 64M K8 三跳：E8 对 E9（readback、host 含 memcpy、upload），以及 41 MB memcpy 按 K", "",
             f"三跳 = `hops.json`（`e8_p2_summary.py` 的 hops 工具）：每条链路每个稳态步一次，14 条链路 × 1000 步合并取 p50 / p95（us）。"
             f"host = memcpy + host_rest（readback 完成到 worker 取走 + stamp 检查）。E8 = {e8['job']}（{e8['node']}），E9 = {e9['job']}（{e9['node']}）；"
             f"包 {f0(e8['host_copy_bytes_p50'])} B 主机拷贝 / {f0(e8['dma_bytes_p50'])} B DMA，两个 campaign 相同。", "",
             "| 跳 | E8 p50（p95） | E9 p50（p95） | E9 / E8 | E8 逐链路 p50 范围 (ms) | E9 逐链路 p50 范围 (ms) |", "|---|---|---|---|---|---|"]
    for key, label in (("readback", "readback（含 DMA 前等待）"), ("host", "host 含 memcpy"), ("memcpy", "　其中 memcpy"),
                       ("host_rest", "　其中 host_rest"), ("upload", "upload（含 DMA 前等待）"), ("t_chain", "t_chain 三跳合计")):
        a, b = e8["pooled_p50_us"][key], e9["pooled_p50_us"][key]
        pa, pb = e8["pooled_p95_us"][key], e9["pooled_p95_us"][key]
        ra = rng(e8["link_p50_range_us"][key]) if key in e8["link_p50_range_us"] else "—"
        rb = rng(e9["link_p50_range_us"][key]) if key in e9["link_p50_range_us"] else "—"
        lines.append(f"| {label} | {a / 1e3:.2f} ms（{pa / 1e3:.2f}） | **{b / 1e3:.2f} ms**（{pb / 1e3:.2f}） | {b / a:.2f} | {ra} | {rb} |")
    s8, s9 = e8["slowest_link_median_us"], e9["slowest_link_median_us"]
    lines += ["",
              f"最慢链路口径（fig:loss / 账本的 t_tr：每步取 t_tr 最大的链路，再对步取中位；ms）：E8 **{s8['t_tr'] / 1e3:.2f} = {s8['readback'] / 1e3:.2f} + {s8['host'] / 1e3:.2f} + {s8['upload'] / 1e3:.2f}**，"
              f"E9 **{s9['t_tr'] / 1e3:.2f} = {s9['readback'] / 1e3:.2f} + {s9['host'] / 1e3:.2f} + {s9['upload'] / 1e3:.2f}**（readback + host + upload）。"
              f"64M K8 两个 campaign 的 T_B 都远大于 t_tr（E8 32.4 ms、E9 28.9 ms），暴露 0，这些差别不进 η。", "",
              "逐链路 p50（ms；E8 → E9）：", "", "| 链路 | readback | host | 其中 memcpy | upload | t_chain |", "|---|---|---|---|---|---|"]
    for name in e8["per_link"]:
        a, b = e8["per_link"][name], e9["per_link"][name]
        lines.append(f"| {name} | {a['readback'] / 1e3:.2f} → {b['readback'] / 1e3:.2f} | {a['host'] / 1e3:.2f} → {b['host'] / 1e3:.2f} | "
                     f"{a['memcpy'] / 1e3:.2f} → {b['memcpy'] / 1e3:.2f} | {a['upload'] / 1e3:.2f} → {b['upload'] / 1e3:.2f} | {a['t_chain'] / 1e3:.2f} → {b['t_chain'] / 1e3:.2f} |")
    lines += ["", "### 1b. 64M 包（41,126,480 B 主机拷贝）的 memcpy 按 K：所有链路 × 稳态步合并的中位时间与该中位对应的带宽", "",
              "| K | campaign | 作业 / 节点 | 链路 × 步 | memcpy p50 (us) | p95 | GB/s @ p50 | 逐链路 p50 范围 (us) | host_rest p50 (us) |", "|---|---|---|---|---|---|---|---|---|"]
    for pt in ("3d_n392_K2", "3d_n392_K4", "3d_n392_K8"):
        for tag in ("e9", "e8"):
            r = d["memcpy_64m_packet"][tag][pt]
            bold = "**" if tag == "e9" else ""
            lines.append(f"| {pt[-1]} | {tag.upper()} | {r['job']} / {r['node']} | {r['links']} × {r['steps_per_link'][0]} | {bold}{f0(r['pooled_p50_us']['memcpy'])}{bold} | "
                         f"{f0(r['pooled_p95_us']['memcpy'])} | {bold}{r['memcpy_gbps_at_p50']:.2f}{bold} | {f0(r['link_p50_range_us']['memcpy'][0])}–{f0(r['link_p50_range_us']['memcpy'][1])} | "
                         f"{r['pooled_p50_us']['host_rest']:.0f} |")
    lines += ["", "### 1c. 节点对照（三跳的 E8 → E9 差别是节点，不是 defrag 序）", "",
              "ab2600f（e8-3d-p2）→ cb3c9f5（v7-rc4）之间只有 31bd493 触及 `experiment/v7/utils/`（defrag 的前缀和路径：case_loader_v7 / case_v7 / defrag_prefix / simulator_v7），传输层与 worker 没有改动。"
              "E8 的 e8_D3（76M K8，E8 代码）跑在 nbj04 组节点上，它的 host_rest 与 memcpy 带宽落在 E9 一侧：", "",
              "| 点 | campaign | 作业 / 节点 | 包 (B) | memcpy p50 (us) | GB/s | host_rest p50 (us) | readback p50 | upload p50 |", "|---|---|---|---|---|---|---|---|---|"]
    for pt in ("3d_n416_K8", "3d_n496_K8", "3d_n192_k8_K8", "3d_n192_k2_K2"):
        for tag in ("e8", "e9"):
            r = d["node_control"][tag].get(pt)
            if r is None:
                continue
            lines.append(f"| {LABEL[pt]} | {tag.upper()} | {r['job']} / {r['node']} | {f0(r['host_copy_bytes_p50'])} | {f0(r['pooled_p50_us']['memcpy'])} | {r['memcpy_gbps_at_p50']:.2f} | "
                         f"{r['pooled_p50_us']['host_rest']:.0f} | {f0(r['pooled_p50_us']['readback'])} | {f0(r['pooled_p50_us']['upload'])} |")
    lines += ["",
              "读法：host_rest 在 nbj11g2 上 150–260 us，在 nbj04g2 / g4 上 14–28 us，与代码版本无关（E8 代码的 e8_D3 在 nbj04g2 也是 28 us）；"
              "K = 8 下 14 个 worker 同时 memcpy 时 nbj11g2 6.4 GB/s（64M）/ 3.7（128M），nbj04 组 7.3–10.0 GB/s；K = 2 / 4 两边都是 15–17 GB/s（并发少，节点差别不显）。"
              "E9 的 K2 / K4（nbj04g2）与 K8（nbj04g4）也不在同一节点，K 序列里的 memcpy 退化（14.6 → 10.5 → 9.65 GB/s）比 E8 同节点序列（15.7 → 10.8 → 6.4）平缓，部分是节点。"]
    return lines


def md_item2(d: dict) -> list:
    lines = ["## 2. fig:loss(a) 3-D 每卡 2M 点（16M K8）的 host 部分：取法核对", "",
             "取法（`e7_loss_figure.worst_link_medians`，`e8_loss_3d.py` 第 201 行调用）：每个稳态步在 14 条链路里取 **t_tr 最大的那条**（t_tr = t_chain 减去两个传输队列的空等），"
             "记下它的 readback / host / upload，三条序列各自对步取中位 → 图上 t_tr、host 两条线。所以 host 线 = 最慢链路（按 t_tr 选）的 host 中位，**不是**所有链路合并的中位，也不是逐链路中位的最大值。"
             "下表把四种读法并列（ms），并给出最慢链路的归属份额：", "",
             "| 点 | campaign / 节点 | 图上 host（最慢链路，按 t_tr） | 图上 t_tr = rb + host + up | T_B | 每步各链路 host 最大值的中位 | 逐链路 host 中位 max / min | 全部链路合并 host 中位 | 最慢链路份额（按 t_tr，前三） | η |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for point in HOP_POINTS:
        for tag in ("e9", "e8"):
            r = d[f"{tag}:{point}"]
            top = sorted(r["slowest_link_share_by_t_tr"].items(), key=lambda kv: -kv[1])[:3]
            top_s = ", ".join(f"{l} {s * 100:.0f} %" for l, s in top)
            bold = "**" if (tag == "e9" and point == "3d_n244_K8") else ""
            lines.append(f"| {LABEL[point]}（{r['per_card_M']:.1f}M / 卡） | {tag.upper()} / {r['node']} | {bold}{r['fig_t_host_us'] / 1e3:.2f}{bold} | "
                         f"{r['fig_t_tr_us'] / 1e3:.2f} = {r['fig_t_readback_us'] / 1e3:.2f} + {r['fig_t_host_us'] / 1e3:.2f} + {r['fig_t_upload_us'] / 1e3:.2f} | {r['T_B_us'] / 1e3:.2f} | "
                         f"{r['alt_median_of_per_step_max_host_us'] / 1e3:.2f} | {r['alt_max_per_link_host_p50_us'] / 1e3:.2f} / {r['alt_min_per_link_host_p50_us'] / 1e3:.2f} | "
                         f"{r['alt_pooled_host_p50_us'] / 1e3:.2f} | {top_s} | {r['eta']:.3f} |")
    r9, r8 = d["e9:3d_n244_K8"], d["e8:3d_n244_K8"]
    lines += ["",
              f"结论：16M K8 的 host 点 **E9 = {r9['fig_t_host_us'] / 1e3:.2f} ms** 确实取自最慢链路——E9 里 3→4 这条链路在 {r9['slowest_link_share_by_t_tr'].get('3->4', 0) * 100:.0f} % 的步里最慢"
              f"（它自己的 host 中位 {r9['per_link_host_p50_us']['s3_to_s4'] / 1e3:.2f} ms），其余链路 host 中位 {r9['alt_min_per_link_host_p50_us'] / 1e3:.2f}–{r9['alt_max_per_link_host_p50_us'] / 1e3:.2f} ms，"
              f"合并中位 {r9['alt_pooled_host_p50_us'] / 1e3:.2f} ms；三种“最慢”读法（图值、每步 host 最大值的中位、逐链路中位的最大值）相差 ≤ {max(abs(r9['alt_median_of_per_step_max_host_us'] - r9['fig_t_host_us']), abs(r9['alt_max_per_link_host_p50_us'] - r9['fig_t_host_us'])) / 1e3:.2f} ms。"
              f"E8 同点图上 {r8['fig_t_host_us'] / 1e3:.2f} ms，逐链路 host 中位 {r8['alt_min_per_link_host_p50_us'] / 1e3:.2f}–{r8['alt_max_per_link_host_p50_us'] / 1e3:.2f} ms（合并 {r8['alt_pooled_host_p50_us'] / 1e3:.2f}）：图值高于任一条链路自己的中位，"
              f"因为最慢链路在 3→4 / 5→6 / 2→1 / 2→3 之间轮换（份额 {', '.join(f'{l} {s * 100:.0f} %' for l, s in sorted(r8['slowest_link_share_by_t_tr'].items(), key=lambda kv: -kv[1])[:4])}），每步取最慢再取中位自然落在各链路中位之上。"
              f"E9 的 host 从 {r8['fig_t_host_us'] / 1e3:.2f} 降到 {r9['fig_t_host_us'] / 1e3:.2f} ms 是节点（nbj11g2 → nbj04g4，见 1c），同一包 16.3 MB 两边相同；这点在 E9 已过 t_tr = T_B 交点（r = {r9['r_p50']:.2f}，E8 {r8['r_p50']:.2f}），暴露 0.4 %（E8 3.2 %）。"]
    return lines


def md_item3(d: dict) -> list:
    lines = ["## 3. 8M 与 16M K8（E9）：G1 复本数、boundary 核时长，seam 工作的构成", "",
             "每卡一行（trace 的 `--step-trace-detail full` 逐核 p50，us；`kernel_table.json`，--all）。列 = 每卡 x 列数（每列 = 4 层 × n²，8M 160,000、16M 254,016 粒子）；"
             "带宽 2 / 2 / 3 列（correction / density / force），所以 corr+dens boundary 核处理每侧 2 列自有 + 1 列 G1 复本，force boundary 核处理每侧 3 列自有；"
             "deep-interior 核处理其余列。ns / 处理粒子与 K = 1 立方体同核的每粒子成本对照。理想 = 整算例 K = 1 窗口周期 ÷ 8 按该卡粒子数折算；seam = busy − 理想。"
             "构成的四项：G1 冗余 = G1 复本数 × corr+dens boundary 的每处理粒子成本；带核超额 = (带核成本 − K1 成本) × 处理数；interior 超额同理；小核 + 空隙 = A/C 段小核 + (busy − 全部核之和) 减去 K = 1 自己的小核 + 空隙折算。", ""]
    for key in ("e9:cavity3d_n192", "e9:cavity3d_n244"):
        r = d[key]
        lines += [f"### {LABEL[r['point']]}（{r['node']}；周期 {r['period_ms']:.3f} ms，busy 均值 {r['busy_mean_ms']:.3f}，整算例 ÷ 8 理想 {r['tau_ref_ms']:.3f}，账本 seam {r['seam_ledger'] * 100:.1f} %，G1 冗余份额 {r['g1_share_ledger'] * 100:.1f} %，η {r['eta']:.3f}；"
                  f"K = 1 立方体 corr+dens {r['k1_rate_ns']['cd']:.2f} / force {r['k1_rate_ns']['f']:.2f} ns/粒子）", "",
                  "| 卡 | 列 | 自有粒子 | G1 复本（每侧 → 合计，% 自有） | corr+dens boundary：处理 / us / ns | force boundary：处理 / us / ns | corr+dens interior：处理 / us / ns | force deep interior：处理 / us / ns | busy | seam（% 周期） | G1 冗余 us（% 周期） | 带核超额 | interior 超额 | 小核 + 空隙 − K1 折算 | 残差 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in r["cards"]:
            sides = " + ".join(f"{int(v) / 1e3:.0f}k" for v in c["n0_per_side"].values())
            def cell(k):
                if c["processed"][k] <= 0:
                    return f"0 / {c['us_p50'][k]:.0f} / —"
                return f"{c['processed'][k] / 1e6:.3f}M / {c['us_p50'][k]:,.0f} / {c['ns_per_processed'][k]:.2f}"
            lines.append(f"| s{c['sim']} | {c['columns']} | {c['own'] / 1e6:.3f}M | {sides} → {c['g1_total'] / 1e3:.0f}k（{c['g1_share_of_own'] * 100:.0f} %） | {cell('cdb')} | {cell('fb')} | {cell('cdi')} | {cell('fdi')} | "
                         f"{c['busy_us'] / 1e3:.2f} | {c['seam_us'] / 1e3:.2f}（{c['seam_share_of_period'] * 100:.1f}） | {c['g1_redundant_us']:,.0f}（{c['g1_redundant_share_of_period'] * 100:.1f}） | "
                         f"{c['band_rate_excess_us']:,.0f} | {c['interior_rate_excess_us']:,.0f} | {c['small_kernels_us'] + c['gaps_us'] - c['k1_small_gap_allowance_us']:,.0f} | {c['residual_us']:,.0f} |")
        lines.append("")
    # summary over interior cards
    lines += ["### 小结（内部卡 = 两侧都有邻卡的 s1–s6；均值）", "",
              "| 点 | 列 / 卡 | 自有 | G1 复本 / 卡（% 自有） | force 带 = 3 列 × 2 侧占自有 | corr+dens boundary us（ns / 处理粒子，对 K1） | force boundary us（ns，对 K1） | deep interior 核 | G1 冗余 % 周期 | seam % 周期 | seam 构成：G1 冗余 / 带核超额 / interior 超额 / 小核 + 空隙 / 残差（% 周期） | E8 同核 us（corr+dens b / force b） |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in ("cavity3d_n192", "cavity3d_n244"):
        r, r8 = d[f"e9:{case}"], d[f"e8:{case}"]
        inner = [c for c in r["cards"] if c["sides"] == 2]
        inner8 = [c for c in r8["cards"] if c["sides"] == 2]
        P = r["period_ms"] * 1e3
        m = lambda key: statistics.mean(c[key] for c in inner)  # noqa: E731
        mu = lambda k: statistics.mean(c["us_p50"][k] for c in inner)  # noqa: E731
        mr = lambda k: statistics.mean(c["ns_per_processed"][k] for c in inner if c["processed"][k] > 0)  # noqa: E731
        cols = sorted(set(c["columns"] for c in inner))
        fdi_txt = (f"force deep interior 0 粒子（{mu('fdi'):.0f} us 空派发）" if all(c["processed"]["fdi"] <= 0 for c in inner)
                   else f"corr+dens interior {mu('cdi'):,.0f} us（{mr('cdi'):.2f} ns）、force deep interior {mu('fdi'):,.0f} us（{mr('fdi'):.2f} ns）")
        if all(c["processed"]["fdi"] <= 0 for c in inner):
            fdi_txt = f"corr+dens interior {mu('cdi'):,.0f} us（{mr('cdi'):.2f} ns，2 列）；" + fdi_txt
        comp = (f"{m('g1_redundant_us') / P * 100:.1f} / {m('band_rate_excess_us') / P * 100:.1f} / {m('interior_rate_excess_us') / P * 100:.1f} / "
                f"{(m('small_kernels_us') + m('gaps_us') - m('k1_small_gap_allowance_us')) / P * 100:.1f} / {m('residual_us') / P * 100:+.1f}")
        lines.append(f"| {LABEL[r['point']]} | {'/'.join(map(str, cols))} | {m('own') / 1e6:.2f}M | {m('g1_total') / 1e3:.0f}k（{m('g1_share_of_own') * 100:.0f} %） | "
                     f"{min(1.0, statistics.mean(c['processed']['fb'] / c['own'] for c in inner)) * 100:.0f} % | {mu('cdb'):,.0f}（{mr('cdb'):.2f}，K1 {r['k1_rate_ns']['cd']:.2f}，×{mr('cdb') / r['k1_rate_ns']['cd']:.2f}） | "
                     f"{mu('fb'):,.0f}（{mr('fb'):.2f}，K1 {r['k1_rate_ns']['f']:.2f}，×{mr('fb') / r['k1_rate_ns']['f']:.2f}） | {fdi_txt} | {m('g1_redundant_us') / P * 100:.1f} | {m('seam_us') / P * 100:.1f} | {comp} | "
                     f"{statistics.mean(c['us_p50']['cdb'] for c in inner8):,.0f} / {statistics.mean(c['us_p50']['fb'] for c in inner8):,.0f} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e7-scripts", default=str(E7_SCRIPTS_DEFAULT), help="e7 worktree scripts dir (e7_loss_figure, e7_full_summary)")
    parser.add_argument("--out", default=str(DATA / "e9/paper_e9_addenda"), help="output stem (.md + .json)")
    parser.add_argument("--paper", default=str(DATA / "e9/paper_e9.md"), help="paper_e9.md to append the pointer line to")
    a = parser.parse_args()
    c8, c9 = load("e8"), load("e9")
    d1 = item1(c8, c9)
    d2 = item2(c8, c9, pathlib.Path(a.e7_scripts))
    d3 = item3(c8, c9)
    out = pathlib.Path(a.out)
    head = ["# E9 论文补三项（2026-10-29）", "",
            f"输入：`{c8['kernel_table']}`、`{c9['kernel_table']}`、`data/{{e8,e9}}/phase2/hops.json`、`data/{{e8,e9}}/loss_3d/ledger.json`、hops.json 里记的 step trace（第 2 项重算）。"
            "生成：`docs/cluster_v6/scripts/e9_paper_addenda.py`；数值全在同名 `.json`。E8 = atomicAdd defrag 序（2026-10-22/23 的镜像），E9 = prefix 序（2026-10-28）。", ""]
    text = "\n".join(head + md_item1(d1) + [""] + md_item2(d2) + [""] + md_item3(d3)) + "\n"
    out.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    json.dump({"item1": d1, "item2": d2, "item3": d3}, open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    paper = pathlib.Path(a.paper)
    pointer = f"\n补三项（2026-10-29：64M K8 三跳 E8/E9 + memcpy 按 K、fig:loss(a) 2M/卡 host 取法、8M/16M K8 G1 与 boundary 核）见 `{out.with_suffix('.md').name}`。\n"
    if paper.exists() and out.with_suffix(".md").name not in paper.read_text(encoding="utf-8"):
        with open(paper, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(pointer)
    print(f"wrote {out.with_suffix('.md')} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
