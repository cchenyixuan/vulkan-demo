"""
round3_tables.py - the timing tables of claude/wall_store_report.md section 9 (wall store round 3, part B), from
kernel_timing.py / dual_timing.py output under logs/wall_store/round3/timing and lean_registers.py's summary.

    .venv/Scripts/python.exe -m experiment.wall_crossing.round3_tables

Entries (labels of the timing runs; timing_round3.sh): <method>[_<build>]_<mode>, method simple / crossing, build one
of the lean items (zero_z, distance_squared, kernel_range, all), factored / all_factored (crossing's near-wall force
with M_i out of the sums), lanes2 / 4 / 8 (section 9.4), constraint_off (crossing without the volume constraint);
mode step (only the step's first and last timestamp: the whole step undistorted), pass (a timestamp between passes,
none between a split pass's two dispatches) or kernel (every dispatch isolated). Every number is the median over three
interleaved rounds of the per-run mean (2000 measured steps in 2-D, 1000 in 3-D, after the warmup). The defrag (one
every case.numerics.defrag_cadence steps, its own submit) is the mean of the defrag timestamps of the step-mode runs
per measured step.

Tables (claude/wall_store/tables/): round3_breakdown_<case>.md, round3_breakdown_versus_simple.md,
round3_variants_<case>.md, round3_ratios.md, round3_lanes.md; logs/wall_store/round3/timing/round3_tables.json.
"""
from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
TIMING = REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "timing"
REGISTERS = REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "registers" / "summary.json"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
CASES = {"n250": "2-D 250²", "n500": "2-D 500²", "n1000": "2-D 1000²", "c3d": "3-D 1M"}
# near-wall fluid particles of the wall store (the fluid particles of near-wall voxels; section 9.5's count)
BUILD_TEXT = {"base": "缺省", "zero_z": "(a) z 置 0", "distance_squared": "(b) 先比距离平方",
              "kernel_range": "(e) 去掉核函数里重复的范围判断", "all": "(a)+(b)+(e)", "factored": "(d) 近壁受力提出 M_i",
              "all_factored": "(a)+(b)+(e)+(d)", "zero_z_distance_squared": "(a)+(b)",
              "zero_z_kernel_range": "(a)+(e)", "distance_squared_kernel_range": "(b)+(e)"}


def load(case: str) -> dict:
    """The labels of the case's main timing run and of its supplement (timing_round3_supplement.sh; distinct labels,
    measured later in the same way)."""
    labels = {}
    for folder in (f"{case}_supplement", case):
        path = TIMING / folder / "summary.json"
        if path.exists():
            labels.update(json.loads(path.read_text(encoding="utf-8"))["labels"])
    return labels


def in_situ(case: str, labels: dict, result: dict) -> None:
    """Section 9.7: the crossing builds' passes with both dispatches together (a timestamp between passes only)
    against the same dispatches each alone (every dispatch isolated): how much of the near-wall dispatch the interior
    dispatch hides."""
    lines = ["| 版本 | 融合遍：内部单独 + 近壁单独 → 一起（重叠） | 受力：同左 | GPU 一步：首尾计时（近壁先） | 内部先 |",
             "|---|---|---|---|---|"]
    for build in ("base", "zero_z", "distance_squared", "kernel_range", "zero_z_distance_squared", "zero_z_kernel_range",
                  "distance_squared_kernel_range", "all", "factored", "all_factored"):
        prefix = "crossing" if build == "base" else f"crossing_{build}"
        together_label = "crossing_base_pass" if build == "base" else f"{prefix}_pass"
        if together_label not in labels:
            continue
        cells = []
        entry = {}
        for name, column in (("fused", "correction_density"), ("force", "force")):
            interior = kernel(labels, f"{prefix}_kernel", f"{column}_split_interior_us")
            near_wall = kernel(labels, f"{prefix}_kernel", f"{column}_split_near_wall_us")
            together = kernel(labels, together_label, f"{column}_us")
            overlap = None if None in (interior, near_wall, together) else interior + near_wall - together
            entry[name] = {"interior": interior, "near_wall": near_wall, "together": together, "overlap": overlap}
            cells.append(f"{cell(interior)} + {cell(near_wall)} → {cell(together)}（{cell(overlap)}）")
        step = kernel(labels, f"{prefix}_step", "step_total_us")
        interior_first = {"base": "crossing_interior_first_step", "all": "crossing_all_interior_first_step"}.get(build)
        entry["step"] = step
        entry["step_interior_first"] = None if interior_first is None else kernel(labels, interior_first,
                                                                                  "step_total_us")
        result.setdefault("in_situ", {}).setdefault(case, {})[build] = entry
        lines.append(f"| {BUILD_TEXT[build]} | {cells[0]} | {cells[1]} | {cell(step)} | {cell(entry['step_interior_first'])} |")
    (TABLES / f"round3_in_situ_{case}.md").write_text(
        f"{CASES[case]}（单卡，µs / 步）：每遍两次调度单独计时之和与一起运行（中间不打时间戳）的时间，括号里是重叠。\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def kernel(labels: dict, label: str, column: str) -> float | None:
    entry = labels.get(label, {}).get("kernels", {}).get(column)
    return None if entry is None else float(entry["median_us"])


def spread(labels: dict, label: str, column: str) -> tuple | None:
    entry = labels.get(label, {}).get("kernels", {}).get(column)
    return None if entry is None else (float(entry["minimum_us"]), float(entry["maximum_us"]))


def defrag_per_step(case: str, label: str) -> float | None:
    """Mean over the rounds of (sum of the defrag timestamps / measured steps) of a step-mode label's CSVs."""
    values = []
    for path in sorted((TIMING / case).glob(f"{label}_r*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        total = sum(float(row["defrag_us"]) for row in rows if row.get("defrag_us"))
        values.append(total / len(rows))
    return statistics.median(values) if values else None


def cell(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def breakdown(case: str, labels: dict, result: dict) -> list:
    """Section 9.5: per-dispatch breakdown of simple and crossing (default builds) that adds up to the whole step."""
    simple, crossing = "simple_kernel", "crossing_pass"
    rows = []
    entry = {}
    for name, column in (("predict", "predict_us"), ("update_voxel", "update_voxel_us"),
                         ("fused", "correction_density_us"), ("copy", "density_copy_us"), ("force", "force_us")):
        entry[name] = {"simple": kernel(labels, simple, column), "crossing": kernel(labels, crossing, column)}
    for name, column in (("fused", "correction_density"), ("force", "force")):
        interior = kernel(labels, "crossing_kernel", f"{column}_split_interior_us")
        near_wall = kernel(labels, "crossing_kernel", f"{column}_split_near_wall_us")
        entry[f"{name}_interior_alone"] = interior
        entry[f"{name}_near_wall_alone"] = near_wall
        together = entry[name]["crossing"]
        entry[f"{name}_overlap"] = (None if None in (interior, near_wall, together)
                                    else interior + near_wall - together)
    for method, label in (("simple", simple), ("crossing", crossing)):
        total = kernel(labels, label, "step_total_us")
        parts = sum(entry[name][method] or 0.0 for name in ("predict", "update_voxel", "fused", "copy", "force"))
        entry.setdefault("gaps", {})[method] = None if total is None else total - parts
        entry.setdefault("ticked_step", {})[method] = total
        entry.setdefault("clean_step", {})[method] = kernel(labels, f"{method}_step", "step_total_us")
        entry.setdefault("defrag", {})[method] = defrag_per_step(case, f"{method}_step")
        clean, defrag = entry["clean_step"][method], entry["defrag"][method]
        entry.setdefault("whole_step", {})[method] = None if None in (clean, defrag) else clean + defrag
        entry.setdefault("frame_wall_clock", {})[method] = kernel(labels, f"{method}_step", "frame_total_us")
    entry["constraint_fused"] = (None if kernel(labels, "crossing_constraint_off_pass", "correction_density_us") is None
                                 else entry["fused"]["crossing"]
                                 - kernel(labels, "crossing_constraint_off_pass", "correction_density_us"))
    entry["constraint_step"] = (None if kernel(labels, "crossing_constraint_off_pass", "step_total_us") is None
                                else entry["ticked_step"]["crossing"]
                                - kernel(labels, "crossing_constraint_off_pass", "step_total_us"))
    result.setdefault("breakdown", {})[case] = entry

    def line(text, simple_value, crossing_value, digits=1):
        difference = (None if simple_value is None or crossing_value is None else crossing_value - simple_value)
        rows.append(f"| {text} | {cell(simple_value, digits)} | {cell(crossing_value, digits)} | "
                    + ("—" if difference is None else f"{difference:+,.{digits}f}") + " |")

    line("预测（predict）", entry["predict"]["simple"], entry["predict"]["crossing"])
    line("体素更新（update_voxel）", entry["update_voxel"]["simple"], entry["update_voxel"]["crossing"])
    line("**融合遍**（crossing：两次调度一起，中间不打时间戳）", entry["fused"]["simple"], entry["fused"]["crossing"])
    line("　内部调度（单独计时）", None, entry["fused_interior_alone"])
    line("　近壁调度（单独计时）", None, entry["fused_near_wall_alone"])
    line("　两次调度的重叠（单独之和 − 一起）", None, entry["fused_overlap"])
    line("密度拷贝", entry["copy"]["simple"], entry["copy"]["crossing"])
    line("**受力**（同上）", entry["force"]["simple"], entry["force"]["crossing"])
    line("　内部调度（单独计时）", None, entry["force_interior_alone"])
    line("　近壁调度（单独计时）", None, entry["force_near_wall_alone"])
    line("　两次调度的重叠", None, entry["force_overlap"])
    line("遍与遍之间的空隙（同一次运行的一步 − 上面各遍）", entry["gaps"]["simple"], entry["gaps"]["crossing"])
    line("= GPU 一步（遍之间打时间戳的运行）", entry["ticked_step"]["simple"], entry["ticked_step"]["crossing"])
    line("GPU 一步（只在首尾打时间戳，单独的运行）", entry["clean_step"]["simple"], entry["clean_step"]["crossing"])
    line("defrag（按步平摊）", entry["defrag"]["simple"], entry["defrag"]["crossing"], 2)
    line("**整步**（首尾时间戳 + defrag 平摊）", entry["whole_step"]["simple"], entry["whole_step"]["crossing"])
    line("（参考）墙钟每步，同步提交", entry["frame_wall_clock"]["simple"], entry["frame_wall_clock"]["crossing"])
    header = (f"| {CASES[case]}，µs / 步 | simple | crossing（缺省 (B)） | crossing − simple |\n|---|---|---|---|")
    (TABLES / f"round3_breakdown_{case}.md").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8",
                                                        newline="\n")
    return rows


def versus_simple(result: dict) -> None:
    """Section 9.5: crossing minus simple, by item (µs per step)."""
    lines = ["| 项（µs / 步） | " + " | ".join(CASES[case] for case in result["breakdown"]) + " |",
             "|---|" + "---|" * len(result["breakdown"])]
    items = []
    for case, entry in result["breakdown"].items():
        values = {}
        for name in ("fused", "force"):
            together = entry[name]["crossing"]
            interior = entry[f"{name}_interior_alone"]
            values[f"{name}_near_wall_added"] = None if None in (together, interior) else together - interior
            values[f"{name}_interior_saving"] = (None if None in (interior, entry[name]["simple"])
                                                 else interior - entry[name]["simple"])
        values["constraint"] = entry["constraint_fused"]
        values["other"] = sum((entry[name]["crossing"] or 0.0) - (entry[name]["simple"] or 0.0)
                              for name in ("predict", "update_voxel", "copy"))
        values["gaps"] = (entry["gaps"]["crossing"] or 0.0) - (entry["gaps"]["simple"] or 0.0)
        values["defrag"] = (entry["defrag"]["crossing"] or 0.0) - (entry["defrag"]["simple"] or 0.0)
        values["ticks"] = ((entry["clean_step"]["crossing"] - entry["ticked_step"]["crossing"])
                           - (entry["clean_step"]["simple"] - entry["ticked_step"]["simple"]))
        values["total"] = entry["whole_step"]["crossing"] - entry["whole_step"]["simple"]
        values["ratio"] = entry["whole_step"]["crossing"] / entry["whole_step"]["simple"]
        items.append(values)
        result.setdefault("versus_simple", {})[case] = values
    for key, text in (("fused_near_wall_added", "多出：融合遍的近壁调度（两次一起 − 内部单独）"),
                      ("force_near_wall_added", "多出：受力的近壁调度（同上）"),
                      ("constraint", "多出：总体积修正（融合遍里的速率项：有 − 无）"),
                      ("fused_interior_saving", "省下：融合遍内部调度单独 − simple 整次调度"),
                      ("force_interior_saving", "省下：受力内部调度单独 − simple 整次调度"),
                      ("other", "其他遍（预测、体素更新、拷贝）"), ("defrag", "defrag（平摊）"),
                      ("gaps", "遍与遍之间的空隙"), ("ticks", "时间戳的影响（首尾计时 − 遍间计时，两者之差）"),
                      ("total", "**合计 = 整步之差**"), ("ratio", "整步之比 crossing ÷ simple")):
        digits = 3 if key == "ratio" else 1
        lines.append(f"| {text} | " + " | ".join(
            "—" if values[key] is None else (f"{values[key]:.{digits}f}" if key == "ratio"
                                              else f"{values[key]:+,.{digits}f}") for values in items) + " |")
    (TABLES / "round3_breakdown_versus_simple.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def variants(case: str, labels: dict, registers: dict, result: dict) -> None:
    """Section 9.7: per build, registers / occupancy / static instructions of the two passes, the dispatch times
    (each dispatch isolated) and the whole step (first and last timestamp only)."""
    dimension = "3-D" if case == "c3d" else "2-D"
    lines = ["| 方法 | 版本 | 融合遍：寄存器 · 占用率 · 静态指令数 | 受力：同左 | 融合遍调度 µs（单独） | 受力调度 µs（单独） | "
             "GPU 一步 µs（首尾计时） | 相对缺省 |", "|---|---|---|---|---|---|---|---|"]
    for method in ("simple", "crossing"):
        base = kernel(labels, f"{method}_step", "step_total_us")
        builds = ("base", "zero_z", "distance_squared", "kernel_range", "zero_z_distance_squared",
                  "zero_z_kernel_range", "distance_squared_kernel_range", "all") + (
            ("factored", "all_factored") if method == "crossing" else ())
        for build in builds:
            if case == "c3d" and build.startswith("zero_z"):
                continue
            if build in ("zero_z_distance_squared", "zero_z_kernel_range", "distance_squared_kernel_range")                     and f"{method}_{build}_step" not in labels:
                continue
            prefix = method if build == "base" else f"{method}_{build}"
            step = kernel(labels, f"{prefix}_step", "step_total_us")
            statistics_entry = registers.get(dimension, {}).get(method, {}).get(build, {})

            def described(names):
                parts = []
                for name in names:
                    value = statistics_entry.get(name)
                    if value:
                        parts.append(f"{value['registers']} · {100 * value['occupancy']:.1f} % · "
                                     f"{value['binary_size'] // 16:,}")
                return " / ".join(parts) if parts else "—"

            if method == "simple":
                fused_registers = described(["correction_density_all"])
                force_registers = described(["force_all"])
                fused_time = cell(kernel(labels, f"{prefix}_kernel", "correction_density_us"))
                force_time = cell(kernel(labels, f"{prefix}_kernel", "force_us"))
            else:
                fused_registers = described(["correction_density_all_split_interior",
                                             "correction_density_all_split_near_wall"])
                force_registers = described(["force_all_split_interior", "force_all_split_near_wall"])
                fused_time = (f"{cell(kernel(labels, f'{prefix}_kernel', 'correction_density_split_interior_us'))} / "
                              f"{cell(kernel(labels, f'{prefix}_kernel', 'correction_density_split_near_wall_us'))}")
                force_time = (f"{cell(kernel(labels, f'{prefix}_kernel', 'force_split_interior_us'))} / "
                              f"{cell(kernel(labels, f'{prefix}_kernel', 'force_split_near_wall_us'))}")
            ratio = None if None in (step, base) else step / base
            result.setdefault("variants", {}).setdefault(case, {}).setdefault(method, {})[build] = {
                "step": step, "spread": spread(labels, f"{prefix}_step", "step_total_us"), "ratio": ratio}
            low_high = spread(labels, f"{prefix}_step", "step_total_us")
            lines.append(f"| {method} | {BUILD_TEXT[build]} | {fused_registers} | {force_registers} | {fused_time} | "
                         f"{force_time} | {cell(step)}"
                         + ("" if low_high is None else f"（{low_high[0]:,.1f}–{low_high[1]:,.1f}）") + " | "
                         + ("—" if ratio is None else f"{100 * (ratio - 1):+.1f} %") + " |")
    (TABLES / f"round3_variants_{case}.md").write_text(
        f"{CASES[case]}（单卡）；寄存器 · 理论占用率 · 静态指令数（驱动给的二进制大小 / 16 字节）；crossing 两个数是内部 / 近壁管线。\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def dual(result: dict) -> dict:
    path = TIMING / "dual_c3d" / "summary.json"
    if not path.exists():
        return {}
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    steps = {}
    for label, entry in labels.items():
        rate = (entry.get("steady_fps") or {}).get("median")
        if rate:
            steps[label] = 1.0e6 / float(rate)
            result.setdefault("dual_fps", {})[label] = entry["steady_fps"]
    result["dual"] = steps
    return steps


def ratios(result: dict) -> None:
    """The four ratios of the task (single: whole-step GPU time; dual: the steady step time), for two choices of the
    copies: every item ((a)+(b)+(e) for simple, (a)+(b)+(e)+(d) for crossing, the task's copies with all items on) and
    each method's fastest measured combination (in the single runs; the dual run timed the all-items builds and
    (d))."""
    lines = ["| 算例 | 副本 | simple 副本 ÷ simple | crossing 副本 ÷ crossing | crossing 副本 ÷ simple 副本 | "
             "crossing 副本 ÷ simple | （参考）crossing ÷ simple |", "|---|---|---|---|---|---|---|"]
    rows = []
    for case in ("n500", "n1000", "c3d"):
        entry = result.get("variants", {}).get(case)
        if not entry:
            continue
        simple = entry["simple"]["base"]["step"]
        crossing = entry["crossing"]["base"]["step"]
        rows.append((CASES[case] + "，单卡", "全部各项", simple, entry["simple"]["all"]["step"], crossing,
                     entry["crossing"]["all_factored"]["step"]))
        best_simple = min((value["step"], build) for build, value in entry["simple"].items() if value["step"])
        best_crossing = min((value["step"], build) for build, value in entry["crossing"].items() if value["step"])
        rows.append((CASES[case] + "，单卡",
                     f"各自最好：simple {BUILD_TEXT[best_simple[1]]}，crossing {BUILD_TEXT[best_crossing[1]]}",
                     simple, best_simple[0], crossing, best_crossing[0]))
    steps = result.get("dual", {})
    if steps:
        rows.append(("3-D 1M，双卡", "全部各项", steps.get("simple"), steps.get("simple_all"), steps.get("crossing"),
                     steps.get("crossing_all_factored")))
        candidates = [(steps[label], label) for label in ("crossing_all_factored", "crossing_factored", "crossing_all")
                      if steps.get(label)]
        if candidates and steps.get("simple_all"):
            best = min(candidates)
            text = {"crossing_all_factored": BUILD_TEXT["all_factored"], "crossing_factored": BUILD_TEXT["factored"],
                    "crossing_all": BUILD_TEXT["all"]}[best[1]]
            rows.append(("3-D 1M，双卡", f"各自最好：simple (a)+(b)+(e)，crossing {text}", steps.get("simple"),
                         steps.get("simple_all"), steps.get("crossing"), best[0]))
    result["ratios"] = []
    for text, copies, simple, simple_copy, crossing, crossing_copy in rows:
        if None in (simple, simple_copy, crossing, crossing_copy):
            continue
        values = (simple_copy / simple, crossing_copy / crossing, crossing_copy / simple_copy, crossing_copy / simple,
                  crossing / simple)
        result["ratios"].append({"case": text, "copies": copies, "values": values})
        lines.append(f"| {text} | {copies} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    (TABLES / "round3_ratios.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8", newline=chr(10))


def lanes(result: dict) -> None:
    """Section 9.8: k invocations per near-wall particle: the near-wall dispatches (isolated) and the whole step."""
    lines = ["| 算例 | k | 近壁融合遍 µs（单独） | 近壁受力 µs（单独） | GPU 一步 µs（首尾计时） | 相对 k = 1 | 相对 simple |",
             "|---|---|---|---|---|---|---|"]
    for case in ("n250", "n500", "n1000", "c3d"):
        labels = load(case)
        if not labels:
            continue
        base = kernel(labels, "crossing_step", "step_total_us")
        simple = kernel(labels, "simple_step", "step_total_us")
        for k, prefix in ((1, "crossing"), (2, "crossing_lanes2"), (4, "crossing_lanes4"), (8, "crossing_lanes8")):
            step = kernel(labels, f"{prefix}_step", "step_total_us")
            fused = kernel(labels, f"{prefix}_kernel", "correction_density_split_near_wall_us")
            force = kernel(labels, f"{prefix}_kernel", "force_split_near_wall_us")
            result.setdefault("lanes", {}).setdefault(case, {})[k] = {"step": step, "fused_near_wall": fused,
                                                                      "force_near_wall": force}
            lines.append(f"| {CASES[case]} | {k} | {cell(fused)} | {cell(force)} | {cell(step)} | "
                         + ("—" if None in (step, base) else f"{100 * (step / base - 1):+.1f} %") + " | "
                         + ("—" if None in (step, simple) else f"{step / simple:.3f}") + " |")
    (TABLES / "round3_lanes.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    registers = json.loads(REGISTERS.read_text(encoding="utf-8")) if REGISTERS.exists() else {}
    result = {}
    for case in ("n500", "n1000", "c3d"):
        labels = load(case)
        if not labels:
            continue
        breakdown(case, labels, result)
        variants(case, labels, registers, result)
        in_situ(case, labels, result)
    if result.get("breakdown"):
        versus_simple(result)
    dual(result)
    ratios(result)
    lanes(result)
    (TIMING / "round3_tables.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    for path in sorted(TABLES.glob("round3_*.md")):
        if path.name.startswith(("round3_breakdown", "round3_variants", "round3_ratios", "round3_lanes.md", "round3_in_situ")):
            print(f"== {path.name}\n{path.read_text(encoding='utf-8')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
