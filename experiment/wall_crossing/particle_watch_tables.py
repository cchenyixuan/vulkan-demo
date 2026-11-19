"""
particle_watch_tables.py - the tables of report section 20 (the particle watch on 3-D K = 2), read from the files the
runs wrote (nothing typed in):

    .venv/Scripts/python.exe -m experiment.wall_crossing.particle_watch_tables

  particle_watch_switch_check.md   the watch switch against the commit before it: canonical_dump --compare of each
                                   pair (logs/wall_crossing/particle_watch/switch_check/*.npz) and, per pair, the
                                   identical rows and the largest |difference| of position, velocity and density
  particle_watch_runs.md           the watch runs (logs/wall_crossing/particle_watch/<name>/ and <name>.log): the
                                   first event (step, kind, particles, where) or none, and the run record
  particle_watch_steps_<name>.md   per step of an event run: own fluid particles, own particles in a ghost column /
                                   in no voxel list, the counter deltas, the largest displacement and shift
  particle_watch_k1_k2_3d.md       3-D 1M crossing, 700 steps: K = 2 with the 3-D ghost pool factor against K = 1,
                                   next to K = 1's own run-to-run spread (rms / largest difference)
  claude/wall_crossing/particle_watch/<name>_summary.json   the report tool's summary of each event bundle
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.report_tables import REPORT_ROOT, markdown_table, write     # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                                       # noqa: E402

WATCH_ROOT = LOG_ROOT / "particle_watch"
SWITCH_ROOT = WATCH_ROOT / "switch_check"
RUNS = (("run2_reproduction_crossing_k2", "复现（第一次，工具 2c49ec0）"),
        ("run2_reproduction_crossing_k2_b", "复现（正式，工具 1c0cbeb）"),
        ("run3_control_simple_k2", "对照一：simple，双卡"),
        ("run4_control_crossing_k1", "对照二：crossing，单卡"),
        ("extra_a_crossing_k2_ghost_pool_0p5", "我加的对照 A：crossing，双卡，副本区系数 0.5"),
        ("extra_b_small_crossing_k2_ghost_pool_0p5", "我加的对照 B：小算例 crossing，双卡，副本区系数 0.5"))
SMOKE = (("smoke/hydrostatic_3d_k2", "开发试跑 1（小算例，双卡）"),
         ("smoke/hydrostatic_3d_k2_b", "开发试跑 2（小算例，双卡）"),
         ("smoke/hydrostatic_3d_k2_c", "开发试跑 3（小算例，双卡）"))


def run_records() -> dict:
    records = {}
    path = LOG_ROOT / "runs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        output = str(record.get("output", ""))
        if "particle_watch" in output:
            records[pathlib.Path(output).stem] = record
            records[output.split("particle_watch/")[-1].rsplit(".", 1)[0]] = record
    return records


def switch_check_table() -> None:
    pairs = []
    for slabs in (1, 2):
        case = f"2-D 250² crossing，K = {slabs}，700 步"
        pairs += [(case, "参照对参照（d94c26c 跑两次）", f"n250_k{slabs}_reference_r1", f"n250_k{slabs}_reference_r2"),
                  (case, "开关关闭对参照", f"n250_k{slabs}_off_r1", f"n250_k{slabs}_reference_r1"),
                  (case, "开关打开对参照", f"n250_k{slabs}_on_r1", f"n250_k{slabs}_reference_r1")]
    for steps in (1, 50, 700):
        case = f"3-D 小算例，K = 1，{steps} 步"
        pairs += [(case, "参照对参照（d94c26c 跑两次）", f"reference_s{steps}_r1", f"reference_s{steps}_r2"),
                  (case, "开关关闭对开关关闭", f"off_s{steps}_r1", f"off_s{steps}_r2"),
                  (case, "开关打开对开关打开", f"on_s{steps}_r1", f"on_s{steps}_r2"),
                  (case, "开关关闭对参照", f"off_s{steps}_r1", f"reference_s{steps}_r1"),
                  (case, "开关打开对开关关闭", f"on_s{steps}_r1", f"off_s{steps}_r1")]
    records = run_records()
    rows = []
    for case, comparison, first, second in pairs:
        first_path, second_path = SWITCH_ROOT / f"{first}.npz", SWITCH_ROOT / f"{second}.npz"
        if not (first_path.exists() and second_path.exists()):
            continue
        result = subprocess.run([sys.executable, str(REPOSITORY_ROOT / "experiment/seam_audit/canonical_dump.py"),
                                 "--compare", str(first_path), str(second_path)], capture_output=True, text=True,
                                cwd=REPOSITORY_ROOT)
        verdict = result.stdout.strip().splitlines()[-1]
        verdict = "逐位相同（10 个字段）" if verdict.startswith("BIT-IDENTICAL") else (
            "不逐位相同" if verdict.startswith("NOT bit") else verdict)
        cells = []
        with np.load(first_path) as a, np.load(second_path) as b:
            total = int(a["global_id"].size)
            same = np.array_equal(a["global_id"], b["global_id"])
            for field, columns in (("position_voxel_id", slice(0, 3)), ("velocity_mass", slice(0, 3)),
                                   ("density_pressure", slice(0, 1))):
                if not same:
                    cells.append("编号集合不同")
                    continue
                x, y = a[field][:, columns], b[field][:, columns]
                equal = int((x.view(np.uint32) == y.view(np.uint32)).all(axis=1).sum())
                largest = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
                cells.append(f"{equal} / {total}，{largest:.1e}")
        ids = " / ".join(records[name]["run_id"] for name in (first, second) if name in records)
        rows.append([case, comparison, verdict, *cells, ids])
    write("particle_watch_switch_check.md", markdown_table(
        ["算例", "比较", "canonical_dump --compare", "位置：相同的行 / 总数，最大差", "速度：相同的行 / 总数，最大差",
         "密度：相同的行 / 总数，最大差", "记录"], rows))


def event_location(event: dict, summary: dict | None) -> str:
    if not summary or not summary.get("primary"):
        return "—"
    primary = summary["primary"]
    position = ", ".join(f"{value:.4f}" for value in primary["position"])
    cut = ", ".join(f"{value:+.4f}" for value in primary["distance_to_cut"])
    role = primary["stored_column_role"]
    role_text = ("本卡自有列" if role == "own column" else
                 ("本卡内层 ghost 列" if "inner" in role else "本卡外层 ghost 列") if "ghost" in role else role)
    listed = "" if primary["listed"] else "，不在任何体素表里"
    return (f"粒子 {primary['global_id']}（卡 {primary['slab']}）在 ({position})，越过 "
            f"{'/'.join(primary['beyond'])}（{ {'face': '面', 'edge': '棱', 'corner': '角'}.get(primary['beyond_kind'], '') }），"
            f"到分界面 {cut} m，在{role_text}{listed}")


def runs_table() -> None:
    records = run_records()
    rows = []
    for name, label in RUNS + SMOKE:
        directory = WATCH_ROOT / name
        log = WATCH_ROOT / f"{name}.log"
        record = records.get(name.split("/")[-1]) or records.get(name)
        if record is None and not log.exists():
            continue
        steps_path = directory / "watch_steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()] \
            if steps_path.exists() else []
        event_path = directory / "event.json"
        summary_path = directory / "report" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        if event_path.exists():
            event = json.loads(event_path.read_text(encoding="utf-8"))
            parts = []
            for kind, value in event["events"].items():
                if kind == "wall_layer":
                    depth = -min(item["box_margin"] for item in value["particles"])
                    parts.append(f"进入壁面层：{value['count']} 个流体粒子越过盒面，最深 {depth:.1e} m")
                elif kind == "count":
                    parts.append(f"粒子数变少：{value['fluid_total']}，初始 {value['initial']}")
                else:
                    parts.append(f"数值异常：{value['count']} 个粒子")
            what = "；".join(parts)
            first = f"第 {event['event_step']} 步（t = {event['event_time']:.5f}）"
            where = event_location(event, summary)
            case = pathlib.Path(event["case"]).parent.name
            slabs = len(event["device_map"])
            environment = {key: value for key, value in event["environment"].items() if key == "V7_GHOST_POOL_FACTOR"}
        else:
            what, where = "没有事件", "—"
            first = f"跑完 {steps[-1]['step']} 步" if steps else "—"
            case = pathlib.Path(record["case"]).parent.name if record else "—"
            slabs = len(record.get("gpu_uuids", [])) if record else "—"
            environment = {}
            dump = WATCH_ROOT / f"{name}.npz"
            if dump.exists():
                with np.load(dump) as archive:
                    meta = json.loads(str(archive["meta"]))
                environment = {key: value for key, value in meta["environment"].items()
                               if key == "V7_GHOST_POOL_FACTOR"}
                slabs = meta["slabs"]
                if meta["alive"] != meta["expected"]:
                    what = f"没有事件；转储时存活 {meta['alive']} / {meta['expected']}"
        last = steps[-1] if steps else None
        unlisted = " / ".join(str(slab["own_alive_unlisted"]) for slab in last["slabs"]) if last else "—"
        overflow = " / ".join(str(slab["delta"]["overflow_ghost_count"]) for slab in last["slabs"]) if last else "—"
        rows.append([label, case, str(slabs), ", ".join(f"{key}={value}" for key, value in environment.items()) or "工具的发布配置（0.29）",
                     first, what, where, unlisted, overflow,
                     record["run_id"] if record else "—", f"`logs/wall_crossing/particle_watch/{name}/`"])
    write("particle_watch_runs.md", markdown_table(
        ["运行", "算例", "K", "副本区系数", "停在", "事件", "出事粒子", "最后一步：不在任何体素表里的自有粒子（卡 0 / 卡 1）",
         "最后一步：overflow_ghost_count 增量（卡 0 / 卡 1）", "记录", "目录"], rows))


def steps_tables() -> None:
    for name, _ in RUNS:
        steps_path = WATCH_ROOT / name / "watch_steps.jsonl"
        if not steps_path.exists():
            continue
        steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
        if len(steps) > 40:          # a run without an event: every 50th step and the last
            steps = [record for record in steps if record["step"] % 50 == 0 or record is steps[-1]]
        rows = []
        for record in steps:
            slabs = record["slabs"]
            displacement = record.get("largest_fluid_displacement_since_previous_check") or {}
            shift = record.get("largest_fluid_shift") or {}
            rows.append([
                str(record["step"]), str(record["fluid_total"]), str(record["fluid_outside_box"]),
                f"{record['smallest_box_margin']:.2e}" if record["smallest_box_margin"] is not None else "—",
                f"{record['largest_fluid_speed']:.3f}",
                " / ".join(str(slab["own_in_ghost_column"]) for slab in slabs),
                " / ".join(str(slab["own_alive_unlisted"]) for slab in slabs),
                " / ".join(str(slab["delta"]["overflow_ghost_count"]) for slab in slabs),
                " / ".join(str(slab["delta"]["migration_install_count"]) for slab in slabs),
                " / ".join(str(slab["delta"]["overflow_departed_count"]) for slab in slabs),
                " / ".join(str(slab["delta"]["far_migration_count"]) for slab in slabs),
                (f"{displacement['value']:.2e}（{(displacement.get('x_minus_cut') or [float('nan')])[0]:+.3f}）"
                 if displacement.get("value") is not None else "—"),
                f"{displacement['median']:.1e}" if displacement.get("median") is not None else "—",
                (f"{shift['value']:.2e}（{(shift.get('x_minus_cut') or [float('nan')])[0]:+.3f}）"
                 if shift.get("value") else "—")])
        write(f"particle_watch_steps_{name}.md", markdown_table(
            ["步", "自有流体粒子", "越出盒", "离盒面最近（m）", "最大速度", "自有粒子在 ghost 列（卡 0 / 1）",
             "不在任何体素表里（卡 0 / 1）", "overflow_ghost_count 增量", "迁入", "overflow_departed 增量",
             "far_migration 增量", "最大单步位移 m（x − x_cut）", "单步位移中位数 m", "最大位移修正 m（x − x_cut）"], rows))


def k1_k2_table() -> None:
    """3-D 1M crossing, 700 canonical steps: K = 2 with the 3-D ghost pool factor (extra control A) against K = 1,
    measured against K = 1's own run-to-run spread (3-D runs do not repeat bit for bit)."""
    section19 = LOG_ROOT / "k2_equivalence" / "cavity3d_1m_crossing_default_volume_700"
    pairs = (("K = 1 对 K = 1（第 19 节的两次）", section19 / "single_s700.npz", section19 / "single_repeat2_s700.npz"),
             ("K = 1（对照二）对 K = 1（第 19 节）", WATCH_ROOT / "run4_control_crossing_k1.npz", section19 / "single_s700.npz"),
             ("K = 2、副本区系数 0.5（对照 A）对 K = 1（对照二）", WATCH_ROOT / "extra_a_crossing_k2_ghost_pool_0p5.npz",
              WATCH_ROOT / "run4_control_crossing_k1.npz"),
             ("K = 2、副本区系数 0.5（对照 A）对 K = 1（第 19 节）", WATCH_ROOT / "extra_a_crossing_k2_ghost_pool_0p5.npz",
              section19 / "single_s700.npz"))
    rows = []
    for label, first, second in pairs:
        if not (first.exists() and second.exists()):
            continue
        with np.load(first) as a, np.load(second) as b:
            meta_a, meta_b = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
            if not np.array_equal(a["global_id"], b["global_id"]):
                rows.append([label, "编号集合不同", "", "", "", ""])
                continue
            cells = []
            for field, columns in (("position_voxel_id", slice(0, 3)), ("velocity_mass", slice(0, 3)),
                                   ("density_pressure", slice(0, 1))):
                difference = np.abs(a[field][:, columns].astype(np.float64) - b[field][:, columns].astype(np.float64))
                cells.append(f"{np.sqrt((difference ** 2).mean()):.2e} / {difference.max():.2e}")
            rows.append([label, f"{meta_a['alive']} / {meta_b['alive']}", *cells,
                         f"{meta_a['repo_head']} / {meta_b['repo_head']}（SPIR-V {meta_a['code_sha256']['spv'][:8]} / "
                         f"{meta_b['code_sha256']['spv'][:8]}）"])
    write("particle_watch_k1_k2_3d.md", markdown_table(
        ["比较", "存活粒子", "位置差 均方根 / 最大（m）", "速度差 均方根 / 最大（m/s）", "密度差 均方根 / 最大（kg/m³）",
         "提交"], rows))


def summaries() -> None:
    target = REPORT_ROOT / "particle_watch"
    target.mkdir(parents=True, exist_ok=True)
    for name, _ in RUNS + SMOKE:
        path = WATCH_ROOT / name / "report" / "summary.json"
        if path.exists():
            (target / f"{name.replace('/', '_')}_summary.json").write_text(path.read_text(encoding="utf-8"),
                                                                           encoding="utf-8", newline="\n")
    path = SWITCH_ROOT / "comparison.json"
    if path.exists():
        (target / "switch_check_comparison.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8",
                                                             newline="\n")


def main() -> int:
    switch_check_table()
    runs_table()
    steps_tables()
    k1_k2_table()
    summaries()
    return 0


if __name__ == "__main__":
    sys.exit(main())
