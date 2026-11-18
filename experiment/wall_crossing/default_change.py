"""
default_change.py - the checks of the default change of 2026-11-19 (wall_boundary crossing defaults to the complete
scheme: geometry pairwise, continuity_wall_component full, pressure_wall reflection, delta_wall reflection; commit
3cba19f) and the tables of its report section.

    .venv/Scripts/python.exe -m experiment.wall_crossing.default_change combinations [--gpu-device 0]
    .venv/Scripts/python.exe -m experiment.wall_crossing.default_change tests
    .venv/Scripts/python.exe -m experiment.wall_crossing.default_change tables

combinations (check 5): every refusal that the default change brings, triggered once:
    - numerics.wall_crossing mappings through parse_wall_crossing_settings (the loader's parser): a quadratic
      extension alone (viscous, continuity, both), with geometry nearest_wall only, geometry nearest_wall alone,
      with one of the two reflections still at its default; the refusals that were there before (continuity
      extension quadratic with continuity wall_velocity) for comparison; and the combinations that must load;
    - the existing quadratic case files through the full loader (load_case_v7);
    - single_step.py's own refusal of a quadratic extension without the three nearest-wall flags (exit code 2,
      before any GPU work);
    - with --gpu-device: the volume constraint of the default 250^2 case under V7_DENSITY_BUDGET=1 (a canonical_dump
      of one step: the simulator refuses it at construction).
  Output: logs/wall_crossing/default_change/combinations.json.
tests (check 6): the v7 test scripts (experiment/v7/_test_*.py) of this checkout, each with its log in
  logs/wall_crossing/tests/default_complete_scheme_<script>.log and one runs.jsonl record; the failure list of
  _test_seam_layout.py in logs/wall_crossing/tests/failures_default_complete_scheme.txt (compared with the base's).
tables: claude/wall_crossing/tables/default_change_*.md from the outputs above, the outcome classes of checks 1 to 3
  (logs/wall_crossing/outcome_classes/), the single-step summaries of check 4
  (logs/wall_crossing/single_step_default_complete_scheme/), the SPIR-V manifests, and the crossing case files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

DEFAULT_COMMIT = "3cba19f"          # the commit that changes the defaults
BEFORE_COMMIT = "48d370f"           # the last commit with the defaults of 2026-11-18
REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
OUT = LOG_ROOT / "default_change"
SINGLE_STEP_OUT = "logs/wall_crossing/single_step_default_complete_scheme"
CAVITY = "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025"
TEST_SCRIPTS = ("_test_chain_cuts", "_test_partition_chain", "_test_transport_relay", "_test_weight_calibration",
                "_test_seam_layout")

# (label, numerics.wall_crossing mapping, expected: "refused" | "loads", what it checks)
MAPPINGS = (
    ("viscous_extension quadratic alone", {"viscous_extension": "quadratic"}, "refused",
     "二阶粘性开关单独打开（新缺省是逐对几何和两个反射）"),
    ("continuity_extension quadratic alone", {"continuity_extension": "quadratic"}, "refused",
     "连续性二阶开关单独打开（还缺粘性二阶）"),
    ("both extensions quadratic", {"viscous_extension": "quadratic", "continuity_extension": "quadratic"}, "refused",
     "两个二阶开关都打开（第 10 节的全二阶算例就是这样写的）"),
    ("quadratic + geometry nearest_wall only",
     {"viscous_extension": "quadratic", "geometry": "nearest_wall"}, "refused",
     "二阶 + 只写了 geometry: nearest_wall（两个反射仍是缺省）"),
    ("geometry nearest_wall alone", {"geometry": "nearest_wall"}, "refused",
     "只写 geometry: nearest_wall（两个反射仍是缺省）"),
    ("geometry nearest_wall + pressure_wall extension",
     {"geometry": "nearest_wall", "pressure_wall": "extension"}, "refused",
     "nearest_wall + 压力 extension，密度反射仍是缺省"),
    ("geometry nearest_wall + delta_wall extension",
     {"geometry": "nearest_wall", "delta_wall": "extension"}, "refused",
     "nearest_wall + 密度 extension，压力反射仍是缺省"),
    ("continuity quadratic + wall_velocity (refused before too)",
     {"viscous_extension": "quadratic", "continuity_extension": "quadratic", "continuity": "wall_velocity",
      "geometry": "nearest_wall", "pressure_wall": "extension", "delta_wall": "extension"}, "refused",
     "改动之前就有的拒绝（连续性二阶不能与 wall_velocity 同用），检查它仍在"),
    ("no switch (the defaults)", {}, "loads", "不写任何开关：完整方案"),
    ("the four defaults until 2026-11-19 written out",
     {"geometry": "nearest_wall", "continuity_wall_component": "normal", "pressure_wall": "extension",
      "delta_wall": "extension"}, "loads", "原来的四项缺省显式写出（一阶）"),
    ("quadratic with the three nearest-wall keys",
     {"viscous_extension": "quadratic", "continuity_extension": "quadratic", "geometry": "nearest_wall",
      "pressure_wall": "extension", "delta_wall": "extension"}, "loads",
     "二阶 + 报错信息要求的三项：可以载入"),
    ("geometry nearest_wall + delta_wall rest_density + pressure_wall extension",
     {"geometry": "nearest_wall", "pressure_wall": "extension", "delta_wall": "rest_density"}, "loads",
     "nearest_wall 配诊断值 rest_density（不是反射）：可以载入"),
)
QUADRATIC_CASES = (f"{CAVITY}_crossing_quadratic_viscous/case.yaml", f"{CAVITY}_crossing_quadratic_full/case.yaml")


# ================================================================================================= combinations
def combinations(gpu_device: int | None) -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.wall_crossing_v7 import parse_wall_crossing_settings

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with LoggedRun(kind="analysis", case="none") as run:
        run.extra["label"] = "default_change combinations: the refusals of the default change (check 5)"
        for label, mapping, expected, text in MAPPINGS:
            try:
                settings = parse_wall_crossing_settings({"wall_crossing": dict(mapping)})
                outcome, message = "loads", (f"{settings.geometry} / {settings.continuity_wall_component} / "
                                             f"{settings.pressure_wall} / {settings.delta_wall} / "
                                             f"{settings.viscous_extension} / {settings.continuity_extension}")
            except ValueError as error:
                outcome, message = "refused", str(error)
            results.append({"path": "parse_wall_crossing_settings", "label": label, "text": text,
                            "mapping": mapping, "expected": expected, "outcome": outcome, "message": message,
                            "passed": outcome == expected})
        for case in QUADRATIC_CASES:
            try:
                load_case_v7(str(REPOSITORY_ROOT / case))
                outcome, message = "loads", ""
            except ValueError as error:
                outcome, message = "refused", str(error)
            results.append({"path": "load_case_v7", "label": case, "text": "第 10 节的二阶算例文件（没有改）",
                            "mapping": None, "expected": "refused", "outcome": outcome, "message": message,
                            "passed": outcome == "refused"})
        # single_step.py's own refusal (argparse: exit code 2 before any device is touched)
        command = [sys.executable, "-m", "experiment.wall_crossing.single_step", "--dimension", "2", "--device", "0",
                   "--viscous-extension", "quadratic", "--out", "logs/wall_crossing/default_change/unused"]
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, encoding="utf-8",
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        message = (completed.stderr.strip().splitlines() or [""])[-1]
        results.append({"path": "single_step.py", "label": "--viscous-extension quadratic alone",
                        "text": "单步工具：只给二阶开关", "mapping": None, "expected": "refused",
                        "outcome": "refused" if completed.returncode == 2 else f"exit {completed.returncode}",
                        "message": message, "passed": completed.returncode == 2 and "nearest_wall" in message})
        if gpu_device is not None:
            # the volume constraint of the default 250^2 case with the density budget build: refused by the
            # simulator at construction (canonical_dump of one step)
            dump = OUT / "budget_with_constraint_unused.npz"
            command = [sys.executable, str(REPOSITORY_ROOT / "experiment" / "seam_audit" / "canonical_dump.py"),
                       "--solver", "v7", "--case", f"{CAVITY}_crossing/case.yaml", "--device-map", str(gpu_device),
                       "--steps", "1", "--canonical-lists", "--out", str(dump)]
            environment = dict(os.environ, V7_DENSITY_BUDGET="1", PYTHONIOENCODING="utf-8",
                               VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True,
                                       encoding="utf-8", env=environment)
            lines = (completed.stdout + completed.stderr).strip().splitlines()
            message = next((line for line in reversed(lines) if "ValueError" in line), lines[-1] if lines else "")
            results.append({"path": "SphSimulatorV7 (V7_DENSITY_BUDGET=1)", "label": f"{CAVITY}_crossing",
                            "text": "缺省方腔算例（现在带约束）用收支构建", "mapping": None, "expected": "refused",
                            "outcome": "refused" if completed.returncode != 0 and "ValueError" in message
                            else f"exit {completed.returncode}",
                            "message": message, "passed": completed.returncode != 0 and "constraint_off" in message})
        passed = all(result["passed"] for result in results)
        run.extra["status"] = "ok" if passed else "failed: an outcome differs from the expected one"
        run.extra["checks"] = len(results)
        (OUT / "combinations.json").write_text(json.dumps({"run_id": run.run_id, "results": results}, indent=1,
                                                          ensure_ascii=False), encoding="utf-8", newline="\n")
    for result in results:
        print(f"[default_change] {'ok  ' if result['passed'] else 'FAIL'} {result['path']}: {result['label']} -> "
              f"{result['outcome']}: {result['message'][:200]}")
    print(f"[default_change] {sum(result['passed'] for result in results)} of {len(results)} as expected")
    return 0 if passed else 1


# ======================================================================================================== tests
def failures_of(log_text: str) -> list:
    """The failure lines of _test_seam_layout.py's log (after '[seam_layout] N FAILURE(S):')."""
    lines, inside = [], False
    for line in log_text.replace("\r", "").splitlines():
        if "FAILURE(S):" in line:
            inside = True
            continue
        if inside and line.startswith("  - "):
            lines.append(line)
    return sorted(lines)


def tests() -> int:
    directory = LOG_ROOT / "tests"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    for script in TEST_SCRIPTS:
        log_path = directory / f"default_complete_scheme_{script}.log"
        command = [sys.executable, f"experiment/v7/{script}.py"]
        with LoggedRun(kind="seam_layout_test" if script == "_test_seam_layout" else "unit_test", case="none",
                       command=" ".join(command)) as run:
            with open(log_path, "w", encoding="utf-8", newline="\n") as log:
                exit_code = subprocess.call(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1"))
            text = log_path.read_text(encoding="utf-8", errors="replace")
            run.extra["label"] = f"experiment/v7/{script}.py, default change ({DEFAULT_COMMIT})"
            run.extra["output"] = str(log_path).replace("\\", "/")
            run.extra["exit_code"] = exit_code
            run.extra["status"] = "ok" if exit_code == 0 else f"failed: exit code {exit_code}"
            entry = {"exit_code": exit_code, "run_id": run.run_id, "log": str(log_path.relative_to(REPOSITORY_ROOT)),
                     "last_line": (text.strip().splitlines() or [""])[-1][:300]}
            if script == "_test_seam_layout":
                failures = failures_of(text)
                (directory / "failures_default_complete_scheme.txt").write_text("\n".join(failures) + "\n",
                                                                                 encoding="utf-8", newline="\n")
                base = sorted(line.rstrip("\r") for line in
                              (directory / "failures_base.txt").read_text(encoding="utf-8").splitlines() if line.strip())
                entry.update({"failures": len(failures), "failures_base": len(base), "same_as_base": failures == base,
                              "spirv_current": "check_spirv_current" not in "\n".join(failures)})
            if "FileNotFoundError" in text:
                entry["missing_input"] = next(line for line in text.splitlines() if "FileNotFoundError" in line)[:300]
            summary[script] = entry
            print(f"[default_change] {script}: exit {exit_code}; {entry['last_line'][:160]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tests.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    return 0


# ======================================================================================================= tables
def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell).replace("|", "/").replace("\n", " ") for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(name: str, text: str) -> None:
    path = REPORT_ROOT / "tables" / name
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[default_change] wrote {path.relative_to(REPOSITORY_ROOT)}")


def git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "show", f"{commit}:{path}"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def manifest_rows(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[0]) == 64 and parts[2].endswith(".spv"):
            rows[parts[2]] = parts[0]
    return rows


def classes_row(name: str, check_text: str, reference_text: str) -> list | None:
    path = LOG_ROOT / "outcome_classes" / name / "classes.json"
    if not path.exists():
        return None
    classes = json.loads(path.read_text(encoding="utf-8"))
    members = "；".join(f"类 {index}：参照 {group['reference_runs']} 次，本算例 {group['case_runs']} 次"
                       for index, group in enumerate(classes["classes"]))
    verdict = "逐位相同" if classes["case_within_reference_classes"] else "**不同**"
    if classes["case_within_reference_classes"] and not classes["every_reference_class_reached"]:
        verdict += "（参照有本算例没落到的类）"
    return [check_text, reference_text, f"{len(classes['classes'])} 个结果类。{members}", verdict, classes["run_id"]]


def tables() -> int:
    import yaml

    from experiment.v7.utils.wall_crossing_v7 import WALL_CROSSING_FIRST_ORDER_SETTINGS, parse_wall_crossing_settings

    summary = {}
    # ---------------------------------------------------------------- regression (checks 1 to 3) and the manifest
    rows = []
    for name, check_text, reference_text in (
            ("default_against_explicit_complete_scheme_s700",
             "1. 新缺省、约束关闭（`..._crossing_constraint_off`：不写开关，不写约束），250²，700 步",
             f"同一提交（{DEFAULT_COMMIT}）上显式写出完整方案的 `..._crossing_reflection_full`，交替运行"),
            ("default_volume_against_explicit_complete_scheme_volume_s700",
             "1. 新缺省、约束打开（`..._crossing`：不写开关，约束打开），250²，700 步",
             f"同一提交上显式写出完整方案并打开约束的 `..._crossing_reflection_full_volume`，交替运行"),
            ("first_order_explicit_against_48d370f_s700",
             f"2. 显式写出原来四项缺省（`..._crossing_first_order`），{DEFAULT_COMMIT}，250²，700 步",
             f"改动之前的提交（{BEFORE_COMMIT}，冻结检出）上缺省运行（`..._crossing`，当时不写开关、没有约束）的 6 个转储"),
            ("simple_against_base_at_3cba19f_s999", "3. simple，250²，999 步",
             "基点构建的 2 个转储（2026-11-17，第 2 节起一直用的那 2 个）"),
            ("adami_against_base_at_3cba19f_s200", "3. adami，250²，200 步", "基点构建的 2 个转储（同上）")):
        row = classes_row(name, check_text, reference_text)
        if row is not None:
            rows.append(row)
    manifest_now = manifest_rows((REPOSITORY_ROOT / "experiment/v7/shaders/spv/MANIFEST.txt").read_text(encoding="utf-8"))
    manifest_base = manifest_rows(git_show("eda5b90", "experiment/v7/shaders/spv/MANIFEST.txt"))
    manifest_before = manifest_rows(git_show(BEFORE_COMMIT, "experiment/v7/shaders/spv/MANIFEST.txt"))
    same_as_base = [name for name in manifest_base if manifest_now.get(name) == manifest_base[name]]
    changed = sorted(name for name in manifest_now if manifest_before.get(name) != manifest_now[name])
    rows.append(["3. simple、adami 用的 SPIR-V", f"基点（eda5b90）清单里的 {len(manifest_base)} 个文件",
                 f"{len(same_as_base)} 个逐字节相同", "通过" if len(same_as_base) == len(manifest_base) else "**不通过**",
                 "MANIFEST.txt"])
    rows.append([f"相对 {BEFORE_COMMIT} 变了的 SPIR-V", f"全部 {len(manifest_now)} 个文件",
                 "、".join(name.replace(".comp.spv", "") for name in changed) or "无", "—（只差缺省字面量，大小不变）",
                 "MANIFEST.txt"])
    summary["manifest"] = {"same_as_base": len(same_as_base), "base": len(manifest_base), "changed": changed}
    write_table("default_change_regression.md", markdown_table(["检查", "参照", "结果", "结论", "记录"], rows))

    # ---------------------------------------------------------------- single step (check 4)
    rows = []
    single_step_root = REPOSITORY_ROOT / SINGLE_STEP_OUT
    for path in sorted(single_step_root.glob("summary_*.json")):
        for outcome in json.loads(path.read_text(encoding="utf-8")):
            reference = outcome.get("reference")
            budget = outcome.get("budget")
            if outcome["configuration"] == "smooth":
                criterion = (reference or {}).get("criterion_value") if reference else None
                text = []
                if reference is not None:
                    text.append(f"判据值 {reference.get('worst_relative_error', reference.get('criterion', float('nan'))):.1e}"
                                if isinstance(reference.get('worst_relative_error', reference.get('criterion')), float)
                                else "参考解对照")
                if budget is not None:
                    text.append("收支四项" + ("通过" if budget["passed"] else "**不通过**"))
                rows.append([path.stem.replace("summary_", ""), outcome["label"], outcome["wall_boundary"],
                             "；".join(text), "通过" if outcome["passed"] else "**不通过**", outcome["run_id"]])
            else:
                rows.append([path.stem.replace("summary_", ""), outcome["label"], outcome["wall_boundary"],
                             "性质检验（残差，只报告）", "—", outcome["run_id"]])
    if rows:
        write_table("default_change_single_step.md",
                    markdown_table(["运行组", "标签", "壁面", "比较", "结论", "记录"], rows))

    # ---------------------------------------------------------------- combinations (check 5)
    path = OUT / "combinations.json"
    if path.exists():
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        rows = [[result["path"], result["text"], "拒绝" if result["expected"] == "refused" else "载入",
                 ("拒绝" if result["outcome"] == "refused" else "载入" if result["outcome"] == "loads"
                  else result["outcome"]),
                 "`" + result["message"].replace("`", "'")[:900] + "`" if result["message"] else "—",
                 "通过" if result["passed"] else "**不通过**"] for result in results]
        write_table("default_change_combinations.md",
                    markdown_table(["路径", "组合", "应当", "实际", "信息（原文）", "结论"], rows))

    # ---------------------------------------------------------------- tests (check 6)
    path = OUT / "tests.json"
    if path.exists():
        results = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for script, entry in results.items():
            if script == "_test_seam_layout":
                result = (f"{entry['failures']} 条失败，" + ("与基点的清单逐行相同" if entry["same_as_base"]
                                                       else "**与基点的清单不同**"))
                verdict = "通过（标准同前：失败清单与基点相同）" if entry["same_as_base"] else "**不通过**"
            elif entry.get("missing_input"):
                result = "没有跑起来：输入文件不在本检出里（" + entry["missing_input"].split("such file or directory:")[-1].strip()[:120] + "）"
                verdict = "没有运行（改动前同样）"
            else:
                result = entry["last_line"]
                verdict = "通过" if entry["exit_code"] == 0 else "**不通过**"
            rows.append([f"`{script}.py`", result, verdict, entry["run_id"]])
        write_table("default_change_tests.md", markdown_table(["测试", "结果", "结论", "记录"], rows))

    # ---------------------------------------------------------------- the crossing case files
    old_defaults = dict(WALL_CROSSING_FIRST_ORDER_SETTINGS)
    keys = ("geometry", "continuity_wall_component", "pressure_wall", "delta_wall")
    groups = {"constraint": [], "new": [], "same": [], "changed": [], "refused": []}
    changed_case_files = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "diff", "--name-only", BEFORE_COMMIT,
                                         DEFAULT_COMMIT, "--", "cases"], capture_output=True, text=True,
                                        check=True).stdout.split()
    for path in sorted((REPOSITORY_ROOT / "cases").rglob("case.yaml")):
        text = path.read_text(encoding="utf-8")
        if "wall_boundary: crossing" not in text:
            continue
        name = str(path.parent.relative_to(REPOSITORY_ROOT / "cases")).replace("\\", "/")
        relative = f"cases/{name}/case.yaml"
        numerics = yaml.safe_load(text)["numerics"]
        mapping = numerics.get("wall_crossing") or {}
        old = {key: mapping.get(key, old_defaults[key]) for key in keys}
        try:
            settings = parse_wall_crossing_settings(numerics)
            new = {key: getattr(settings, key) for key in keys}
        except ValueError:
            new = None
        if relative in changed_case_files:
            created = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-e",
                                      f"{BEFORE_COMMIT}:{relative}"], capture_output=True).returncode != 0
            groups["new" if created else "constraint"].append(name)
        elif new is None:
            groups["refused"].append(name)
        elif new == old:
            groups["same"].append(name)
        else:
            groups["changed"].append((name, ", ".join(f"{key} {old[key]} → {new[key]}" for key in keys
                                                      if old[key] != new[key])))
    rows = [["加上约束（缺省算例）", "<br>".join(f"`{name}`" for name in groups["constraint"]),
             "开关不写（取新缺省），约束打开"],
            ["新建（回归检查用）", "<br>".join(f"`{name}`" for name in groups["new"]),
             "`_constraint_off`：不写开关、没有约束；`_first_order`：原来四项缺省显式写出、没有约束"],
            ["没动，含义不变", "<br>".join(f"`{name}`" for name in groups["same"]), "四项都显式写着"],
            ["没动，含义变了（诊断算例）", "<br>".join(f"`{name}`：{change}" for name, change in groups["changed"]),
             "它们只写了一部分开关，其余取缺省"],
            ["没动，现在载入时报错", "<br>".join(f"`{name}`" for name in groups["refused"]),
             "二阶算例：报错信息要求同时写 geometry: nearest_wall、pressure_wall: extension、delta_wall: extension"]]
    summary["cases"] = {key: len(value) for key, value in groups.items()}
    write_table("default_change_cases.md", markdown_table(["类别", "算例（`cases/` 下）", "说明"], rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8",
                                     newline="\n")
    print(f"[default_change] cases: {summary['cases']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("combinations", "tests", "tables"))
    parser.add_argument("--gpu-device", type=int, default=None,
                        help="combinations: also trigger the simulator's refusal of the constraint with the budget build")
    arguments = parser.parse_args()
    if arguments.action == "combinations":
        return combinations(arguments.gpu_device)
    if arguments.action == "tests":
        return tests()
    return tables()


if __name__ == "__main__":
    sys.exit(main())
