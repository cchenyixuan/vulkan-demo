"""
resume_integral_check.py - report section 25 D: the resume of a run with the volume constraint's integral term.

    .venv/Scripts/python.exe -m experiment.wall_crossing.resume_integral_check --continuous s25_resume_continuous \
        --segmented s25_resume_segmented --old s25_resume_old_checkpoint --restore-log LOG

(1) the checkpoint of the segmented run (its newest c*.npz before the resume) against the first segment's last
constraint update before it and against what the resumed process printed (--restore-log, the box_run output of the
resumed segment): I and v_ref, bit for bit (float64 repr);
(2) the resumed run against the continuous one after the resume: theta and I (each run's updates, the continuous
run's values interpolated linearly in time at the resumed run's update times) and the mean density (series.csv at the
same steps): the largest differences;
(3) the old-checkpoint run (the checkpoint without _volume_constraint_integral): I of its first update after the
resume and the warning in its output (--old-log).
Writes claude/wall_crossing/tables/resume_integral_*.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402


def records_with_segments(path: pathlib.Path) -> list:
    """The constraint log's update records with absolute steps (each segment's steps restart at 0 after a marker)."""
    records, offset = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "segment_start_step" in record:
            offset = int(record["segment_start_step"])
            continue
        record = dict(record, absolute_step=offset + int(record["step"]))
        records.append(record)
    return records


def read_series(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(float(row["step"])): float(row["density_mean"]) for row in rows}


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--segmented", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--restore-log", required=True)
    parser.add_argument("--old-log", required=True)
    parser.add_argument("--second-continuous", default=None,
                        help="a second continuous run of the same case: the same differences between the two "
                             "continuous runs, as the scale of the run-to-run divergence")
    arguments = parser.parse_args()
    runs = LOG_ROOT / "box_run"
    timestep = None

    # (1) checkpoint vs the first segment's last update vs the resumed process's print
    segmented_records = records_with_segments(runs / arguments.segmented / "volume_constraint.jsonl")
    checkpoints = sorted((runs / arguments.segmented / "checkpoints").glob("c*.npz"))
    checkpoint = checkpoints[0]
    checkpoint_step = int(checkpoint.stem[1:])
    with np.load(checkpoint) as archive:
        stored_integral = float(archive["_volume_constraint_integral"])
        stored_reference = float(archive["_volume_constraint_reference_mean_volume"])
    before = [record for record in segmented_records if record["absolute_step"] < checkpoint_step
              and record["absolute_step"] <= checkpoint_step][:0]
    segment_marks = [json.loads(line) for line in (runs / arguments.segmented / "volume_constraint.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip() and "segment_start_step" in line]
    resume_step = int(segment_marks[-1]["segment_start_step"]) if segment_marks else checkpoint_step
    first_segment = [record for record in segmented_records if record["absolute_step"] < resume_step
                     or (not segment_marks)]
    last_before = [record for record in first_segment if record["absolute_step"] <= checkpoint_step][-1]
    restore_text = pathlib.Path(arguments.restore_log).read_text(encoding="utf-8", errors="replace")
    printed_integral = float(re.search(r"volume constraint \(integral\): I (\S+) from the checkpoint", restore_text)[1])
    printed_reference = float(re.search(r"v_ref (\S+), rate", restore_text)[1])
    check_rows = [
        ["I（积分项）", repr(last_before["integral"]), repr(stored_integral), repr(printed_integral),
         "逐位相同" if last_before["integral"] == stored_integral == printed_integral else "不同"],
        ["v_ref", repr(last_before["reference_mean_volume"]), repr(stored_reference), repr(printed_reference),
         "逐位相同" if last_before["reference_mean_volume"] == stored_reference == printed_reference else "不同"]]
    write("resume_integral_restore.md", markdown_table(
        ["量", f"第一段最后一次更新（第 {last_before['absolute_step']:,} 步）", f"检查点 `{checkpoint.name}`",
         "续跑进程读回后打印的值", "结论"], check_rows))

    # (2) the resumed run against the continuous one after the resume
    continuous_records = records_with_segments(runs / arguments.continuous / "volume_constraint.jsonl")
    resumed = [record for record in segmented_records if record["absolute_step"] >= resume_step]
    continuous_steps = np.array([record["absolute_step"] for record in continuous_records], dtype=float)
    comparison_rows, theta_differences, integral_differences = [], [], []
    for record in resumed:
        theta_continuous = float(np.interp(record["absolute_step"], continuous_steps,
                                           [entry["theta"] for entry in continuous_records]))
        integral_continuous = float(np.interp(record["absolute_step"], continuous_steps,
                                              [entry["integral"] for entry in continuous_records]))
        theta_differences.append(record["theta"] - theta_continuous)
        integral_differences.append(record["integral"] - integral_continuous)
    density_segmented = read_series(runs / arguments.segmented / "series.csv")
    density_continuous = read_series(runs / arguments.continuous / "series.csv")
    common = sorted(step for step in density_segmented if step > resume_step and step in density_continuous)
    density_differences = [density_segmented[step] - density_continuous[step] for step in common]
    theta_scale = float(np.std([entry["theta"] for entry in continuous_records
                                if entry["absolute_step"] >= resume_step]))
    density_scale = float(np.std([density_continuous[step] for step in common]))
    second = {}
    if arguments.second_continuous:
        second_records = records_with_segments(runs / arguments.second_continuous / "volume_constraint.jsonl")
        later = [record for record in second_records if record["absolute_step"] >= resume_step]
        second["theta"] = max(abs(record["theta"] - float(np.interp(record["absolute_step"], continuous_steps,
                              [entry["theta"] for entry in continuous_records]))) for record in later)
        second["integral"] = max(abs(record["integral"] - float(np.interp(record["absolute_step"], continuous_steps,
                                 [entry["integral"] for entry in continuous_records]))) for record in later)
        density_second = read_series(runs / arguments.second_continuous / "series.csv")
        second["density"] = max(abs(density_second[step] - density_continuous[step]) for step in common
                                if step in density_second)
    comparison_rows = [
        ["θ", str(len(theta_differences)), f"{max(abs(value) for value in theta_differences):.2e}",
         f"{theta_scale:.2e}", f"{second['theta']:.2e}" if second else "—"],
        ["I", str(len(integral_differences)), f"{max(abs(value) for value in integral_differences):.2e}",
         f"{abs(np.mean([entry['integral'] for entry in continuous_records if entry['absolute_step'] >= resume_step])):.2e}（均值的大小）",
         f"{second['integral']:.2e}" if second else "—"],
        ["平均密度（kg/m³）", str(len(density_differences)), f"{max(abs(value) for value in density_differences):.2e}",
         f"{density_scale:.2e}", f"{second['density']:.2e}" if second else "—"]]
    write("resume_integral_continuity.md", markdown_table(
        ["量", f"续跑后比较的点数（第 {resume_step:,} 步以后）", "续跑与连续运行之差的最大绝对值",
         "连续运行在同一段里自身的标准差（作比较的尺度）", "两次连续运行之间同一段的最大差"], comparison_rows))

    # (3) the old checkpoint without I
    old_records = records_with_segments(runs / arguments.old / "volume_constraint.jsonl")
    old_marks = [json.loads(line) for line in (runs / arguments.old / "volume_constraint.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip() and "segment_start_step" in line]
    old_resume = int(old_marks[-1]["segment_start_step"])
    old_after = [record for record in old_records if record["absolute_step"] >= old_resume]
    old_text = pathlib.Path(arguments.old_log).read_text(encoding="utf-8", errors="replace")
    warning = re.search(r"WARNING volume constraint \(integral\)[^\n]*", old_text)
    first = old_after[0]
    write("resume_integral_old_checkpoint.md", markdown_table(
        ["续跑起点", "打印的警告", "第一次更新的 θ", "第一次更新后的 I（= θ Δ / (τ T_I)，从 0 起）", "之后最后一次更新的 I"],
        [[f"第 {old_resume:,} 步", f"`{warning[0]}`" if warning else "没有",
          f"{first['theta']:+.6e}", f"{first['integral']:+.6e}", f"{old_after[-1]['integral']:+.6e}"]]))
    print(json.dumps({"checkpoint": checkpoint.name, "resume_step": resume_step,
                      "integral_bitwise": last_before["integral"] == stored_integral == printed_integral,
                      "reference_bitwise": last_before["reference_mean_volume"] == stored_reference == printed_reference,
                      "theta_max_difference": max(abs(value) for value in theta_differences),
                      "integral_max_difference": max(abs(value) for value in integral_differences),
                      "density_max_difference": max(abs(value) for value in density_differences),
                      "old_warning": bool(warning)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
