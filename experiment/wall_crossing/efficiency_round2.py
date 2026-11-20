"""
efficiency_round2.py - report section 27: the timing tables of the second efficiency round.

    .venv/Scripts/python.exe -m experiment.wall_crossing.efficiency_round2

Reads logs/wall_crossing/section27/timing/<size>_step/summary.json (kernel_timing.py, V7_BENCH_TICK_DETAIL=step) and
c3d_k2_steady / c3d_k2_anatomy (dual_timing.py) and writes claude/wall_crossing/tables/efficiency_round2_*.md.
Single-GPU labels: simple; base_2f71864 (the commit before this round, from its own checkout); trim_none / trim_mirror
/ trim_delta / trim_both (this tree, V7_WALL_CROSSING_REGISTER_TRIM none / mirror / delta / mirror,delta; trim_mirror
= the default = +3.2); pairwise (+3.2 +3.3: pressure_wall_evaluation pairwise, default trim). Two-slab labels: simple; base_2f71864 (band mode 1, no trims);
mode1_trims (+3.2); mode1_trims_pairwise (+3.2 +3.3); mode2_trims (+3.1 +3.2, this tree's defaults);
mode2_trims_pairwise (+3.1 +3.2 +3.3); *_frame_shift_3 (2.3: the cut at 510,050 / 520,251 fluid particles).
"""
from __future__ import annotations

import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

TIMING = LOG_ROOT / "section27" / "timing_final"
SIZES = (("n250", "2-D 250²"), ("n500", "2-D 500²"), ("n1000", "2-D 1000²"), ("c3d", "3-D 1M"))


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    rows = []
    for size, text in SIZES:
        path = TIMING / f"{size}_step" / "summary.json"
        if not path.exists():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
        value = {label: entry["kernels"]["step_total_us"]["median_us"] for label, entry in labels.items()}
        spread = {label: (entry["kernels"]["step_total_us"]["minimum_us"], entry["kernels"]["step_total_us"]["maximum_us"])
                  for label, entry in labels.items()}
        simple = value["simple"]
        def ratio(label: str) -> str:
            return f"{value[label] / simple:.3f}"
        def change(first: str, second: str) -> str:
            return f"{100 * (value[second] / value[first] - 1):+.1f} %"
        rows.append([text, f"{simple:.1f}（{spread['simple'][0]:.1f}–{spread['simple'][1]:.1f}）", ratio("base_2f71864"),
                     ratio("trim_none"), ratio("trim_mirror"), ratio("pairwise"),
                     change("base_2f71864", "trim_none"), change("trim_none", "trim_mirror"),
                     change("trim_none", "trim_delta"), change("trim_none", "trim_both"),
                     change("trim_mirror", "pairwise"), change("base_2f71864", "pairwise")])
    write("efficiency_round2_single.md", markdown_table(
        ["算例", "simple 每步 µs（三轮范围）", "2f71864 / simple", "本轮代码、精简关 / simple", "+3.2（M_d 共享）/ simple",
         "+3.2 +3.3 / simple", "本轮代码精简关对 2f71864", "M_d 共享（= 3.2）", "只开 δ 项", "两项都开", "3.3（在 3.2 之上）",
         "+3.2 +3.3 对 2f71864"], rows))

    path = TIMING / "c3d_k2_steady" / "summary.json"
    if path.exists():
        labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
        fps = {label: entry["steady_fps"] for label, entry in labels.items()}
        simple = fps["simple"]["median"]
        names = {"simple": "simple", "base_2f71864": "2f71864（方式 1）", "mode1_trims": "+3.2（方式 1）",
                 "mode1_trims_pairwise": "+3.2 +3.3（方式 1）", "mode2_trims": "+3.1 +3.2（本轮缺省）",
                 "mode2_trims_pairwise": "+3.1 +3.2 +3.3", "mode2_trims_frame_shift_3": "+3.1 +3.2，frame 外移 3 层",
                 "simple_frame_shift_3": "simple，frame 外移 3 层"}
        dual_rows = []
        for label, entry in fps.items():
            dual_rows.append([names.get(label, label), f"{entry['median']:.1f}（{entry['minimum']:.1f}–{entry['maximum']:.1f}）",
                              f"{simple / entry['median']:.3f}",
                              f"{100 * (fps['base_2f71864']['median'] / entry['median'] - 1):+.1f} %"
                              if label != "base_2f71864" else "—"])
        write("efficiency_round2_dual_steady.md", markdown_table(
            ["运行（3-D 1M 双卡）", "稳态步速 fps（三轮范围）", "每步时间 / simple", "每步时间对 2f71864"], dual_rows))
    path = TIMING / "c3d_k2_anatomy" / "summary.json"
    if path.exists():
        labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
        keys = ("phase_a", "phase_b", "phase_c", "b_to_c_gap", "correction_density_interior", "force_deep_interior",
                "correction_density_boundary", "force", "readback_leading_dma", "readback_trailing_dma",
                "upload_leading_dma", "upload_trailing_dma")
        phase_rows = []
        for label, entry in labels.items():
            for slab, values in sorted(entry["slabs"].items()):
                phase_rows.append([label, slab] + [f"{values[key]['median_us']:.0f}" if key in values else "—"
                                                   for key in keys])
        write("efficiency_round2_dual_phases.md", markdown_table(["运行", "卡"] + list(keys), phase_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
