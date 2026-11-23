"""
round4_tables.py - the timing tables of claude/wall_store_report.md section 10 from the round-4 campaigns
(logs/wall_store/round4/timing{,2,3}/<case>/summary.json: kernel_timing's whole-step GPU time, single GPU; dual_timing's
steady steps per second, two GPUs; three rounds interleaved, median and minimum-maximum over the rounds).

    .venv/Scripts/python.exe -m experiment.wall_crossing.round4_tables

Writes claude/wall_store/tables/round4_lanes.md (10.6), round4_kernel_range.md (10.7), round4_summary.md (10.8) and
logs/wall_store/round4/tables.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPOSITORY_ROOT / "logs" / "wall_store" / "round4"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
CASES = (("2-D 250²", "n250", False), ("2-D 500²", "n500", False), ("2-D 1000²", "n1000", False),
         ("2-D 1000²，双卡", "dual_n1000", True), ("3-D 1M", "c3d", False), ("3-D 1M，双卡", "dual_c3d", True))
# the slab's fluid particles in near-wall voxels at the start (the lanes rule's input; the simulator prints it)
NEAR_WALL = {"n250": (5401,), "n500": (10901,), "n1000": (21901,), "dual_n1000": (10450, 11451), "c3d": (251613,),
             "dual_c3d": (124180, 127433)}
WAVE = 48 * 32 * 170     # one wave of the RTX 5090: warps per SM x 32 x SMs


def load(campaign: str, case: str) -> dict:
    path = ROOT / campaign / case / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))["labels"] if path.exists() else {}


def step(labels: dict, label: str, dual: bool):
    """(time per step us, minimum, maximum) - dual: 1e6 / steady steps per second."""
    entry = labels.get(label)
    if not entry:
        return None
    if dual:
        rate = entry["steady_fps"]
        return 1e6 / rate["median"], 1e6 / rate["maximum"], 1e6 / rate["minimum"]
    value = entry["kernels"]["step_total_us"]
    return value["median_us"], value["minimum_us"], value["maximum_us"]


def cell(value) -> str:
    return "—" if value is None else f"{value[0]:,.1f}（{value[1]:,.1f}–{value[2]:,.1f}）"


def change(value, reference) -> str:
    return "—" if value is None or reference is None else f"{100 * (value[0] / reference[0] - 1):+.1f} %"


def lanes_table(result: dict) -> str:
    lines = ["| 算例 | 近壁流体粒子（每张卡） | 近壁线程束 / SM | 规则选的 k | k = 1 µs | k = 2 | k = 4 | k = 8 | 规则相对 k = 1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for text, case, dual in CASES:
        labels = load("timing2", case)
        if not labels:
            continue
        counts = NEAR_WALL[case]
        reference = step(labels, "default", dual)
        rows = {k: step(labels, f"lanes{k}", dual) for k in (2, 4, 8)}
        chosen = 4 if max(counts) * 4 <= WAVE else 1
        warps = " / ".join(f"{count / 32 / 170:.1f}" for count in counts)
        count_text = " / ".join(f"{count:,}" for count in counts)
        rule = reference if chosen == 1 else rows[4]
        lines.append(f"| {text} | {count_text} | {warps} | {chosen} | {cell(reference)} | "
                     + " | ".join(f"{cell(rows[k])} {change(rows[k], reference)}" for k in (2, 4, 8))
                     + f" | {change(rule, reference) if chosen > 1 else '0（就是 k = 1）'} |")
        result.setdefault("lanes", {})[case] = {"k1": reference, **{f"k{k}": rows[k] for k in rows}, "chosen": chosen}
    return "\n".join(lines) + "\n"


def kernel_range_table(result: dict) -> str:
    lines = ["| 算例 | 第二次计时：S 合进循环，k = 1 | 加 (e) | 变化 | 遍间计时：同左两项 | 变化 | 第三次计时：S 合进循环，k = 4 | k = 4 加 (e) | 变化 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for text, case, dual in CASES:
        second, third = load("timing2", case), load("timing3", case)
        if not second:
            continue
        base, with_e = step(second, "default", dual), step(second, "e", dual)
        base_pass, e_pass = step(second, "default_pass", dual), step(second, "e_pass", dual)
        lanes4, final = step(third, "lanes4", dual), step(third, "final", dual)
        pass_text = (f"{base_pass[0]:,.1f} → {e_pass[0]:,.1f}" if base_pass and e_pass else "—")
        lines.append(f"| {text} | {cell(base)} | {cell(with_e)} | {change(with_e, base)} | {pass_text} | "
                     f"{change(e_pass, base_pass)} | {cell(lanes4)} | {cell(final)} | {change(final, lanes4)} |")
        result.setdefault("kernel_range", {})[case] = {"combined": base, "e": with_e, "lanes4": lanes4, "final": final}
    return "\n".join(lines) + "\n"


def summary_table(result: dict) -> str:
    """Section 5 of the task: 262156d's default | round 4's (a)+(d) | + the section-4 changes; each against simple's
    default and simple with (a), from the same campaign (2-D: campaign 3; 3-D: campaign 2, where the final default is
    the one of the 'default' entry: S in the loop, k = 1 by the rule, no (e))."""
    lines = ["| 算例 | simple 缺省 | simple (a) | 262156d 缺省 | 新缺省 (a)+(d) | 加第 4 节的改动（最终缺省） |",
             "|---|---|---|---|---|---|"]
    for text, case, dual in CASES:
        # the final default: 2-D = combined + k = 4 by the rule, no (e) (campaign 3's lanes4 entry, every switch
        # pinned to it); 3-D = combined, k = 1 by the rule (campaign 2's default entry)
        campaign, final_label = ("timing2", "default") if case in ("c3d", "dual_c3d") else ("timing3", "lanes4")
        labels = load(campaign, case)
        if not labels:
            continue
        simple, simple_a = step(labels, "simple", dual), step(labels, "simple_a", dual)
        cells = []
        for label in ("base", "ad", final_label):
            value = step(labels, label, dual)
            ratio_simple = f"{value[0] / simple[0]:.3f}"
            ratio_a = f"{value[0] / simple_a[0]:.3f}" if simple_a else "—"
            cells.append(f"{value[0]:,.1f} µs · {ratio_simple} · {ratio_a}")
        lines.append(f"| {text} | {simple[0]:,.1f} | {'—' if not simple_a else f'{simple_a[0]:,.1f}'} | "
                     + " | ".join(cells) + " |")
        result.setdefault("summary", {})[case] = {"simple": simple, "simple_a": simple_a,
                                                   **{label: step(labels, label, dual)
                                                      for label in ("base", "ad", final_label)}}
    return "\n".join(lines) + "\n"


def main() -> int:
    result = {}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, function in (("round4_lanes.md", lanes_table), ("round4_kernel_range.md", kernel_range_table),
                           ("round4_summary.md", summary_table)):
        table = function(result)
        (TABLES / name).write_text(table, encoding="utf-8", newline="\n")
        print(name)
        print(table)
    (ROOT / "tables.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
