"""
round2_tables.py - the register and bit-check tables of the wall store's round 2 (claude/wall_store_report.md):

  registers: the driver's register count (VK_KHR_pipeline_executable_properties, pipeline_statistics.py) of the
    near-wall, interior and one-pipeline builds of both passes, 2-D (250^2) and 3-D (50 per side): 3aea634 (the
    round-1 probe's baseline), after items R + B, the final tree (49da2b9) and the final tree with one item taken back
    (its V7_WALL_CROSSING_* switch), from logs/wall_store/register_probe/<prefix>summary.json;
  bits: frozen_evaluate.py --sort-voxel-lists outputs of this tree against 351c27c (and one item taken back), rows
    whose float32 bits differ, from logs/wall_store/frozen/bits_*.npz.

  timing (sections 4-5): whole-step times of logs/wall_store/round2_timing (kernel_timing.py single, dual_timing.py
    steady and anatomy, the dispatch-order runs), the per-kernel 2-D 250^2 dispatches against round 1's.

    .venv/Scripts/python.exe -m experiment.wall_crossing.round2_tables

Writes claude/wall_store/tables/round2_registers.md, round2_bits.md, round2_bits_reverted.md, round2_timing.md,
round2_timing_dual_order.md, round2_a_versus_b.md, round2_n250_dispatches.md and round2_dual_anatomy.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.wall_store_check import markdown_table, theoretical_occupancy, write_table  # noqa: E402

PROBES = REPOSITORY_ROOT / "logs" / "wall_store" / "register_probe"
FROZEN = REPOSITORY_ROOT / "logs" / "wall_store" / "frozen"
COLUMNS = (("", "3aea634（第 2 节后相同）"), ("after_rb_", "加 R、B 后"), ("final_", "最终 49da2b9"),
           ("final_revert_rule_types_", "最终，退回 R"), ("final_revert_body_motion_", "最终，退回 B"),
           ("final_revert_mirror_", "最终，退回 M"), ("final_revert_wall_rate_loop_", "最终，退回 S"))
ROWS = (("fused", "融合遍 近壁 (B)"), ("force", "受力 近壁 (B)"), ("fused_interior", "融合遍 内部 (B)"),
        ("force_interior", "受力 内部 (B)"), ("fused_one_pipeline", "融合遍 (A)"), ("force_one_pipeline", "受力 (A)"))
BIT_FIELDS = ("density_rate", "acceleration", "shift", "correction_inverse", "density_gradient_kernel_sum")
BIT_LABELS = {"density_rate": "ρ 的变化（ρ_{n+1}）", "acceleration": "加速度", "shift": "位移修正",
              "correction_inverse": "L_i", "density_gradient_kernel_sum": "密度梯度 + 核函数和"}
STATES = (("cavity2d", "2-D 250² 方腔，3,000 步后（63,001 个流体粒子）"),
          ("cavity3d", "3-D 50³ 方腔，1,000 步后（132,651 个）"),
          ("g30", "倾斜 30° 静水盒，标签 t = 2.01 的状态（63,001 个）"))


def differing_rows(first: np.ndarray, second: np.ndarray) -> int:
    first, second = np.asarray(first), np.asarray(second)
    if first.dtype == np.float32:
        return int((first.view(np.uint32) != second.view(np.uint32)).reshape(first.shape[0], -1).any(axis=1).sum())
    return int((first != second).reshape(first.shape[0], -1).any(axis=1).sum())


TIMING = REPOSITORY_ROOT / "logs" / "wall_store" / "round2_timing"
ROUND1 = REPOSITORY_ROOT / "logs" / "wall_store"


def timing_tables() -> None:
    """Sections 4-5: whole-step times (single: kernel_timing.py step_total_us, the median over three rounds; dual:
    1e6 / the steady rate's median), the near-wall dispatches of 2-D 250^2 (per-kernel timestamps), the B -> C wait of
    the dual anatomy."""
    single = (("n250", "2-D 250² 单卡"), ("n500", "2-D 500² 单卡"), ("n1000", "2-D 1000² 单卡"), ("c3d", "3-D 1M 单卡"))
    dual = (("c3d", "3-D 1M 双卡"), ("n1000", "2-D 1000² 双卡"))

    def load(path: pathlib.Path):
        return json.loads(path.read_text(encoding="utf-8"))["labels"] if path.exists() else None

    def step(labels: dict, label: str, is_dual: bool):
        """(median, low, high) microseconds per step."""
        if is_dual:
            value = labels[label]["steady_fps"]
            return 1e6 / value["median"], 1e6 / value["maximum"], 1e6 / value["minimum"]
        value = labels[label]["kernels"]["step_total_us"]
        return value["median_us"], value["minimum_us"], value["maximum_us"]

    def cell(value: tuple, pool: float, spread: bool = True) -> str:
        median, low, high = value
        relative = f"{100 * (median / pool - 1):+.1f} %"
        return f"{median:.1f}（{low:.1f}–{high:.1f}，{relative}）" if spread else f"{median:.1f}（{relative}）"

    rows, decision = [], []
    for size, text in single:
        labels = load(TIMING / f"single_{size}" / "summary.json")
        if labels is None:
            continue
        pool = step(labels, "pool", False)
        values = {key: step(labels, key, False) for key in ("s2", "new_a", "new_b_near_wall_first",
                                                             "new_b_interior_first")}
        rows.append([text, f"{pool[0]:.1f}（{pool[1]:.1f}–{pool[2]:.1f}）"]
                    + [cell(values[key], pool[0]) for key in values])
        a, b = values["new_a"][0], values["new_b_near_wall_first"][0]
        decision.append([text, f"{a:.1f}", f"{b:.1f}", f"{100 * (a / b - 1):+.1f} %", "是" if a / b - 1 > 0.01 else "否"])
    for size, text in dual:
        labels = load(TIMING / f"dual_{size}_steady" / "summary.json")
        if labels is None:
            continue
        pool = step(labels, "pool", True)
        order = "内部先" if size == "c3d" else "近壁先"
        cells = [text, f"{pool[0]:.1f}（{pool[1]:.1f}–{pool[2]:.1f}）", cell(step(labels, "s2", True), pool[0]),
                 cell(step(labels, "new_a", True), pool[0])]
        b_cell = cell(step(labels, "new_b", True), pool[0]) + f"，第 1 轮的缺省顺序：{order}"
        cells += [b_cell, "见下表"] if order == "近壁先" else ["见下表", b_cell]
        rows.append(cells)
    write_table("round2_timing.md", markdown_table(
        ["算例", "标签 crossing-pool µs / 步（三轮范围）", "3aea634 = 第 2 节之后（351c27c，同一份代码）",
         "第 3 节之后 (A)", "第 3 节之后 (B)，近壁先", "第 3 节之后 (B)，内部先"], rows)
        + "\n\n单卡：整步 GPU 时间（`V7_BENCH_TICK_DETAIL=step`），三轮中位数；双卡：稳态步率中位数的倒数。百分数相对标签。"
          "351c27c 用它自己的缺省：(B)，2-D 近壁先，3-D 内部先。")

    rows = []
    for size, text in dual:
        labels = load(TIMING / f"dual_{size}_order_steady" / "summary.json")
        if labels is None:
            continue
        pool = step(labels, "pool", True)
        values = {key: step(labels, key, True) for key in ("new_a", "new_b_near_wall_first", "new_b_interior_first")}
        rows.append([text, f"{pool[0]:.1f}（{pool[1]:.1f}–{pool[2]:.1f}）"] + [cell(values[key], pool[0])
                                                                            for key in values])
        a, b = values["new_a"][0], values["new_b_near_wall_first"][0]
        decision.append([text, f"{a:.1f}", f"{b:.1f}", f"{100 * (a / b - 1):+.1f} %", "是" if a / b - 1 > 0.01 else "否"])
    if rows:
        write_table("round2_timing_dual_order.md", markdown_table(
            ["双卡，第二次（调度顺序）", "标签 crossing-pool µs / 步（三轮范围）", "第 3 节之后 (A)",
             "第 3 节之后 (B)，近壁先", "第 3 节之后 (B)，内部先"], rows))
    write_table("round2_a_versus_b.md", markdown_table(
        ["算例", "(A) µs / 步", "(B) 近壁先 µs / 步", "(A) 对 (B)", "(A) 慢 1 % 以上"], decision))

    # the near-wall dispatches of 2-D 250^2: round 1 (its per-kernel run), and now
    rows = []
    kernels_now = TIMING / "kernels_n250" / "summary.json"
    kernels_before = ROUND1 / "timing_kernels" / "n250" / "summary.json"
    sources = []
    if kernels_before.exists():
        before = json.loads(kernels_before.read_text(encoding="utf-8"))["labels"]
        sources += [("上一轮：标签", before["pool"]), ("上一轮：3aea634 (B)", before["store_b"]),
                    ("上一轮：3aea634 (A)", before["store_a"])]
    if kernels_now.exists():
        now = json.loads(kernels_now.read_text(encoding="utf-8"))["labels"]
        sources += [("本轮：标签", now["pool"]), ("本轮：351c27c (B)", now["s2"]),
                    ("本轮：第 3 节之后 (B)，近壁先", now["new_b_near_wall_first"]),
                    ("本轮：第 3 节之后 (B)，内部先", now["new_b_interior_first"]),
                    ("本轮：第 3 节之后 (A)", now["new_a"])]
    for label, entry in sources:
        kernels = entry["kernels"]

        def value(key):
            item = kernels.get(key)
            return item["median_us"] if item else None

        def text(key):
            number = value(key)
            return "—" if number is None else f"{number:.1f}"

        cells = [label, text("correction_density_split_interior_us"), text("correction_density_split_near_wall_us"),
                 text("force_split_interior_us"), text("force_split_near_wall_us"), text("correction_density_us"),
                 text("force_us"), text("step_total_us")]
        rows.append(cells)
    if rows:
        write_table("round2_n250_dispatches.md", markdown_table(
            ["2-D 250² 单卡，逐核时间戳（µs / 步）", "融合遍 内部调度", "融合遍 近壁调度", "受力 内部调度", "受力 近壁调度",
             "融合遍合计", "受力合计", "整步（带时间戳）"], rows)
            + "\n\n(A) 只有一次调度（合计列）。逐核时间戳把每次调度隔开，(B) 的两次调度不重叠，所以合计是两者之和；"
              "整步计时（不带时间戳）里两次调度之间没有屏障，可以重叠。")

    rows = []
    for size, text in dual:
        order = "内部先" if size == "c3d" else "近壁先"
        for campaign, entries in (
                ("anatomy", (("pool", "标签 crossing-pool"), ("s2", "3aea634 = 351c27c (B)，" + order),
                             ("new_a", "第 3 节之后 (A)"), ("new_b", "第 3 节之后 (B)，" + order))),
                ("order_anatomy", (("new_b_near_wall_first", "第 3 节之后 (B)，近壁先（第二次）"),
                                   ("new_b_interior_first", "第 3 节之后 (B)，内部先（第二次）")))):
            path = TIMING / f"dual_{size}_{campaign}" / "summary.json"
            if not path.exists():
                continue
            labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
            for label, label_text in entries:
                slabs = labels[label]["slabs"]
                rows.append([text, label_text]
                            + [f"{slabs['0'][key]['median_us']:.0f} / {slabs['1'][key]['median_us']:.0f}"
                               for key in ("phase_a", "phase_b", "phase_c", "b_to_c_gap")])
    if rows:
        write_table("round2_dual_anatomy.md", markdown_table(
            ["双卡分相计时（µs，0 号卡 / 1 号卡，两轮中位数）", "版本", "phase A", "phase B", "phase C", "B→C 等待"], rows))


def main() -> int:
    timing_tables()
    summaries = {prefix: json.loads((PROBES / f"{prefix}summary.json").read_text(encoding="utf-8"))["baseline"]
                 for prefix, _ in COLUMNS}
    rows = []
    for key, label in ROWS:
        cells = [label]
        for prefix, _ in COLUMNS:
            counts = [summaries[prefix][dimension][key] for dimension in ("2-D", "3-D")]
            cells.append(" / ".join(f"{count}（{100 * theoretical_occupancy(count):.1f} %）" for count in counts))
        rows.append(cells)
    write_table("round2_registers.md", markdown_table(["管线", *(label for _, label in COLUMNS)], rows)
                + "\n\n每格：2-D / 3-D 的寄存器数（理论占用率）。")

    if not (FROZEN / "bits_cavity3d_s2_1.npz").exists():
        # the evaluations were deleted after the tables were written (claude/wall_store_report.md 8.1): keep those
        print("[round2_tables] no frozen evaluations (deleted after use): round2_bits*.md left as they are")
        return 0
    rows = []
    for state, label in STATES:
        new = [np.load(FROZEN / f"bits_{state}_crossing_{run}.npz") for run in (1, 2)]
        old = [np.load(FROZEN / f"bits_{state}_s2_{run}.npz") for run in (1, 2)]
        for field in BIT_FIELDS:
            rows.append([label, BIT_LABELS[field], f"{differing_rows(new[0][field], new[1][field])}",
                         f"{differing_rows(old[0][field], old[1][field])}",
                         f"{differing_rows(new[0][field], old[0][field])} / {new[0][field].shape[0]:,}"])
    write_table("round2_bits.md", markdown_table(
        ["冻结状态", "输出", "本版两次之间不同的行", "351c27c 两次之间", "本版对 351c27c 不同的行"], rows))
    reverted = []
    old = np.load(FROZEN / "bits_cavity3d_s2_1.npz")
    for name, label in (("V7_WALL_CROSSING_RULE_TYPES", "退回 R"), ("V7_WALL_CROSSING_BODY_MOTION", "退回 B"),
                        ("V7_WALL_CROSSING_MIRROR", "退回 M"), ("V7_WALL_CROSSING_WALL_RATE_LOOP", "退回 S")):
        path = FROZEN / f"bits_cavity3d_revert_{name}.npz"
        if path.exists():
            evaluation = np.load(path)
            reverted.append([label, *(f"{differing_rows(evaluation[field], old[field])}" for field in BIT_FIELDS)])
    if reverted:
        write_table("round2_bits_reverted.md", markdown_table(
            ["3-D 方腔，本版退回一项", *(BIT_LABELS[field] for field in BIT_FIELDS)], reverted)
            + "\n\n每格：与 351c27c 的 float32 位不同的行数（共 132,651 行）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
