"""
_run_acceptance.py — the acceptance battery of v7_single (docs/seam_audit/v7_single.md, 验收 1-4), one GPU.

  cpu        acceptance 1: _compare_recordings.py over the five cases and three switch settings;
  gpu        acceptance 2 + 3: canonical_dump of S (v7_single), V (v7 --v7-mode single) and C (v7 chain K = 1)
             on every (case, steps) of the battery, default switches and V7_FUSED_CORRECTION_DENSITY=0;
             --compare S vs V (must be bit-identical) and S vs C (reported); every dump's invariants;
  validation acceptance 4: v7_single under the Khronos validation layer for 2000 steps (2-D 1M, 3-D 8M);
  all        the three, then summary.json + summary.md in --out.

Every GPU run is its own subprocess (fresh driver state), on --device (the discrete-first index of the
headless card; --gpu-uuid checks it). Results are appended to <out>/runs.jsonl and finished runs are
skipped on a rerun.

    .venv/Scripts/python.exe experiment/v7_single/_run_acceptance.py all --out logs/v7_single/acceptance --device 1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
# (tag, case, steps) of the bitwise battery; adami K = 1 is not self-reproducible at 999 steps (v7_perf.md)
BATTERY = [
    ("n250_200", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml", 200),
    ("n250_999", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml", 999),
    ("cavity2d_1m_200", "cases/lid_driven_cavity_2d_gen/case.yaml", 200),
    ("cavity3d_8m_b4_200", "cases/cavity3d_8m_b4/case.yaml", 200),
    ("cavity3d_1m_200", "cases/cavity3d_1m/case.yaml", 200),
    ("n250_adami_200", "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml", 200),
]
CPU_CASES = ["cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
             "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml",
             "cases/lid_driven_cavity_2d_gen/case.yaml", "cases/cavity3d_8m_b4/case.yaml", "cases/cavity3d_1m/case.yaml"]
SETTINGS = {"default": [], "unfused": ["--env", "V7_FUSED_CORRECTION_DENSITY=0"]}
BUILDS = {"S": ["--solver", "v7_single"], "V": ["--solver", "v7", "--v7-mode", "single"], "C": ["--solver", "v7"]}
VALIDATION_RUNS = [("cavity2d_1m", "cases/lid_driven_cavity_2d_gen/case.yaml"),
                   ("cavity3d_8m_b4", "cases/cavity3d_8m_b4/case.yaml")]
VALIDATION_STEPS = 2000


def run(command: list[str], log_path: pathlib.Path, env: dict) -> tuple[int, str]:
    started = time.time()
    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    text = completed.stdout + "\n" + completed.stderr + f"\n[acceptance] wall {time.time() - started:.1f} s rc {completed.returncode}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return completed.returncode, text


def environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def append(out: pathlib.Path, record: dict) -> None:
    with open(out / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def done_runs(out: pathlib.Path) -> dict:
    records = {}
    path = out / "runs.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def parse_compare(text: str) -> dict:
    """The verdict and per-field rows of a canonical_dump --compare output."""
    fields = {}
    masked = None
    for line in text.splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) >= 3 and parts[1] in ("identical", "DIFFERENT") and "/" in stripped:
            rows = parts[2].split("/") if "/" in parts[2] else None
            try:
                equal, total = int(parts[2]), int(parts[4])
            except (ValueError, IndexError):
                equal = total = None
            largest = None
            if "max |diff|" in stripped:
                try:
                    largest = float(stripped.split("max |diff|")[1].split()[0])
                except ValueError:
                    largest = None
            fields[parts[0]] = {"identical": parts[1] == "identical", "equal_rows": equal, "rows": total,
                                "max_abs_diff": largest}
        if "deep-wall rows masked" in stripped:
            masked = stripped
    verdict = "BIT-IDENTICAL" if "BIT-IDENTICAL" in text else ("NOT bit-identical" if "NOT bit-identical" in text
                                                               else "NOT COMPARABLE")
    return {"verdict": verdict, "fields": fields, "masked": masked}


def ulp_report(first: pathlib.Path, second: pathlib.Path) -> dict:
    """Rows and the largest ULP distance per differing float field (S vs C report)."""
    import numpy as np
    out = {}
    with np.load(first) as a, np.load(second) as b:
        for name in a.files:
            if name == "meta" or name not in b.files or name.startswith("deep_wall_skip"):
                continue
            x, y = a[name], b[name]
            if x.shape != y.shape:
                out[name] = {"shape": [list(x.shape), list(y.shape)]}
                continue
            if x.dtype.kind != "f":
                differing = int((x != y).reshape(x.shape[0], -1).any(axis=1).sum())
                out[name] = {"differing_rows": differing}
                continue
            raw_x, raw_y = x.view(np.int32).astype(np.int64), y.view(np.int32).astype(np.int64)
            # ULP distance of same-sign floats; different signs are counted through zero
            ulp = np.where(np.signbit(x) == np.signbit(y), np.abs(raw_x - raw_y),
                           np.abs(raw_x & 0x7FFFFFFF) + np.abs(raw_y & 0x7FFFFFFF))
            unequal = (x.view(np.uint32) != y.view(np.uint32))
            rows = int(unequal.reshape(x.shape[0], -1).any(axis=1).sum())
            out[name] = {"differing_rows": rows, "max_ulp": int(ulp[unequal].max()) if rows else 0,
                         "max_abs_diff": float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if rows else 0.0}
    return out


def invariants_of(dump: pathlib.Path) -> dict:
    import numpy as np
    with np.load(dump) as archive:
        meta = json.loads(str(archive["meta"]))
    status = meta.get("status") or []
    overflow = {f"slab{index}.{key}": value for index, record in enumerate(status)
                for key, value in record.items() if key.startswith("overflow_") and value}
    return {"alive": meta.get("alive"), "expected": meta.get("expected"), "drift": (meta.get("alive") or 0) - (meta.get("expected") or 0),
            "overflow": overflow, "fps": meta.get("fps"), "head": meta.get("repo_head"), "dirty": meta.get("repo_dirty_v6"),
            "solver": meta.get("solver"), "mode": meta.get("mode"), "switches": meta.get("solver_switches"),
            "deep_wall": meta.get("deep_wall", {}) and meta["deep_wall"].get("resolution")}


def gpu_battery(out: pathlib.Path, device: int, only: set | None) -> dict:
    env = environment()
    results = {}
    records = done_runs(out)
    dumps = out / "gpu"
    dumps.mkdir(parents=True, exist_ok=True)
    for tag, case, steps in BATTERY:
        if only and tag not in only:
            continue
        for setting_name, setting_arguments in SETTINGS.items():
            entry = {"case": case, "steps": steps, "setting": setting_name, "dumps": {}, "invariants": {}}
            for build, build_arguments in BUILDS.items():
                run_id = f"{tag}/{setting_name}/{build}"
                dump = dumps / f"{tag}_{setting_name}_{build}.npz"
                if run_id in records and records[run_id].get("rc") == 0 and dump.exists():
                    entry["dumps"][build] = str(dump)
                    entry["invariants"][build] = records[run_id]["invariants"]
                    continue
                command = [PYTHON, "experiment/seam_audit/canonical_dump.py", *build_arguments, "--case", case,
                           "--device-map", str(device), "--steps", str(steps), "--canonical-lists",
                           *setting_arguments, "--out", str(dump)]
                print(f"[acceptance] {run_id} ...", end="", flush=True)
                rc, text = run(command, out / "logs" / f"{tag}_{setting_name}_{build}.log", env)
                record = {"id": run_id, "rc": rc, "command": command[1:], "when": time.strftime("%Y-%m-%d %H:%M:%S")}
                if rc == 0 and dump.exists():
                    record["invariants"] = invariants_of(dump)
                    entry["dumps"][build] = str(dump)
                    entry["invariants"][build] = record["invariants"]
                    print(f" ok fps {record['invariants']['fps']:.0f} drift {record['invariants']['drift']} "
                          f"overflow {record['invariants']['overflow'] or 0}", flush=True)
                else:
                    print(f" FAILED rc {rc} (see logs)", flush=True)
                append(out, record)
                records[run_id] = record
            comparisons = {}
            for pair in (("S", "V"), ("S", "C")):
                if all(build in entry["dumps"] for build in pair):
                    command = [PYTHON, "-m", "experiment.seam_audit.canonical_dump", "--compare",
                               entry["dumps"][pair[0]], entry["dumps"][pair[1]]]
                    rc, text = run(command, out / "logs" / f"{tag}_{setting_name}_{pair[0]}_vs_{pair[1]}.log", env)
                    parsed = parse_compare(text)
                    parsed["rc"] = rc
                    if pair == ("S", "C") and parsed["verdict"] != "BIT-IDENTICAL":
                        parsed["ulp"] = ulp_report(pathlib.Path(entry["dumps"]["S"]), pathlib.Path(entry["dumps"]["C"]))
                    if parsed["verdict"] == "BIT-IDENTICAL":
                        # the masked comparison also holds without the mask on K = 1 (skipped walls keep 0 on both)
                        rc_full, text_full = run(command + ["--no-mask"],
                                                 out / "logs" / f"{tag}_{setting_name}_{pair[0]}_vs_{pair[1]}_nomask.log", env)
                        parsed["no_mask_verdict"] = parse_compare(text_full)["verdict"]
                    comparisons["_vs_".join(pair)] = parsed
                    print(f"[acceptance] {tag}/{setting_name} {pair[0]} vs {pair[1]}: {parsed['verdict']}"
                          + (f" (no-mask {parsed.get('no_mask_verdict')})" if "no_mask_verdict" in parsed else ""), flush=True)
            entry["comparisons"] = comparisons
            results[f"{tag}/{setting_name}"] = entry
    return results


def validation_runs(out: pathlib.Path, device: int) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env.pop("VK_LOADER_LAYERS_DISABLE", None)         # the layer must load
    env["PYTHONIOENCODING"] = "utf-8"
    results = {}
    records = done_runs(out)
    for tag, case in VALIDATION_RUNS:
        run_id = f"validation/{tag}"
        if run_id in records and records[run_id].get("rc") == 0:
            results[tag] = records[run_id]
            continue
        command = [PYTHON, "experiment/v7_single/_run_single_bench.py", "--case", case, "--device", str(device),
                   "--warmup", "200", "--measure", str(VALIDATION_STEPS - 200), "--in-flight", "2", "--validation",
                   "--tag", "acceptance_validation"]
        print(f"[acceptance] {run_id} ({VALIDATION_STEPS} steps under the validation layer) ...", end="", flush=True)
        rc, text = run(command, out / "logs" / f"validation_{tag}.log", env)
        result = None
        for line in text.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
        messages = sum(1 for line in text.splitlines() if line.startswith("[Vulkan "))
        record = {"id": run_id, "rc": rc, "command": command[1:], "steps": VALIDATION_STEPS,
                  "validation_enabled": result.get("validation_enabled") if result else None,
                  "message_counts": result.get("validation") if result else None, "message_lines": messages,
                  "drift": result.get("drift") if result else None, "overflow": result.get("overflow") if result else None,
                  "fps": result.get("fps") if result else None, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f" rc {rc} layer {record['validation_enabled']} messages {record['message_counts']} drift {record['drift']}",
              flush=True)
        append(out, record)
        results[tag] = record
    return results


def cpu_compare(out: pathlib.Path) -> dict:
    env = environment()
    command = [PYTHON, "experiment/v7_single/_compare_recordings.py", "--inventory",
               "--out", str(out / "cpu_compare.json")]
    for case in CPU_CASES:
        command += ["--case", case]
    print("[acceptance] CPU command-stream comparison ...", flush=True)
    rc, text = run(command, out / "logs" / "cpu_compare.log", env)
    report = json.loads((out / "cpu_compare.json").read_text(encoding="utf-8")) if (out / "cpu_compare.json").exists() else {}
    summary = {"rc": rc, "all_identical": report.get("all_identical"), "cases": {}}
    for case, settings in report.get("cases", {}).items():
        summary["cases"][case] = {setting: {name: {"identical": item["identical"], "allowed": item["allowed_v7_only"],
                                                     "step_counts": item["counts"]}
                                            for name, item in payload["recordings"].items()}
                                  for setting, payload in settings.items()}
    print(f"[acceptance] CPU comparison rc {rc}: {'ALL IDENTICAL' if summary['all_identical'] else 'NOT IDENTICAL'}", flush=True)
    return summary


def write_summary(out: pathlib.Path, summary: dict) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = ["# v7_single acceptance summary", "", f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    cpu = summary.get("cpu")
    if cpu:
        lines += ["## 1. CPU command streams (v7 single mode vs v7_single)", "",
                  f"all identical: **{cpu.get('all_identical')}** (rc {cpu.get('rc')})", "",
                  "| case | setting | step | bootstrap_init | bootstrap_compute | defrag | restart |", "|---|---|---|---|---|---|---|"]
        for case, settings in cpu.get("cases", {}).items():
            for setting, recordings in settings.items():
                cells = []
                for name in ("step", "bootstrap_init", "bootstrap_compute", "defrag", "restart"):
                    item = recordings[name]
                    cells.append(("identical" if item["identical"] else "DIFFERENT") + (" (+allowed fill)" if item["allowed"] else ""))
                lines.append(f"| {case} | {setting} | " + " | ".join(cells) + " |")
        lines.append("")
    gpu = summary.get("gpu")
    if gpu:
        lines += ["## 2-3. GPU bitwise (canonical lists) and invariants", "",
                  "| battery | setting | S vs V | S vs C | drift S/V/C | overflow S/V/C | fps S/V/C |", "|---|---|---|---|---|---|---|"]
        for key, entry in gpu.items():
            comparisons = entry.get("comparisons", {})
            sv = comparisons.get("S_vs_V", {})
            sc = comparisons.get("S_vs_C", {})
            sc_text = sc.get("verdict", "n/a")
            if sc.get("ulp"):
                worst = max(((name, item) for name, item in sc["ulp"].items() if item.get("differing_rows")),
                            key=lambda pair: pair[1].get("max_ulp", 0), default=None)
                if worst:
                    sc_text += f" ({sum(item.get('differing_rows', 0) for item in sc['ulp'].values())} rows over the fields, max {worst[1].get('max_ulp')} ULP in {worst[0]})"
            invariants = entry.get("invariants", {})
            drift = "/".join(str(invariants.get(build, {}).get("drift", "?")) for build in "SVC")
            overflow = "/".join(str(len(invariants.get(build, {}).get("overflow", {}) or {})) for build in "SVC")
            fps = "/".join(f"{invariants.get(build, {}).get('fps') or 0:.0f}" for build in "SVC")
            lines.append(f"| {key.split('/')[0]} | {entry['setting']} | {sv.get('verdict', 'n/a')}"
                         + (f" (no-mask {sv['no_mask_verdict']})" if sv.get("no_mask_verdict") else "")
                         + f" | {sc_text} | {drift} | {overflow} | {fps} |")
        lines.append("")
    validation = summary.get("validation")
    if validation:
        lines += ["## 4. Validation layer (default settings, 2000 steps, v7_single)", "",
                  "| case | layer loaded | messages | drift | overflow | fps |", "|---|---|---|---|---|---|"]
        for tag, record in validation.items():
            lines.append(f"| {tag} | {record.get('validation_enabled')} | {record.get('message_counts')} "
                         f"(stderr lines {record.get('message_lines')}) | {record.get('drift')} | {record.get('overflow')} | {record.get('fps')} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cpu", "gpu", "validation", "all", "summary"))
    parser.add_argument("--out", default="logs/v7_single/acceptance")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--gpu-uuid", default=HEADLESS_5090_UUID)
    parser.add_argument("--only", default=None, help="comma-separated battery tags (gpu)")
    arguments = parser.parse_args()
    out = (_REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if arguments.command in ("gpu", "validation", "all") and arguments.gpu_uuid:
        from experiment.v7_single._run_single_bench import resolve_device_index
        index = resolve_device_index(arguments.gpu_uuid)
        if index != arguments.device:
            raise SystemExit(f"--device {arguments.device} is not the card {arguments.gpu_uuid} (index {index})")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    only = set(arguments.only.split(",")) if arguments.only else None
    if arguments.command in ("cpu", "all"):
        summary["cpu"] = cpu_compare(out)
    if arguments.command in ("gpu", "all"):
        summary.setdefault("gpu", {}).update(gpu_battery(out, arguments.device, only))
    if arguments.command in ("validation", "all"):
        summary["validation"] = validation_runs(out, arguments.device)
    write_summary(out, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
