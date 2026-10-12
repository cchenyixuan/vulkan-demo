"""
adami_chain_gates.py — E40 (v7-rc3): the GPU gates of wall_boundary adami on a chain of K > 1 slabs, run from
this checkout (the rc3 build) against an rc2 checkout, all logs under --out.

    .venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates bitexact --rc2 ../vulkan-demo-v7-wake
    .venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates seam
    .venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates cost

bitexact: canonical_dump (--solver v7 --canonical-lists) of the same case / K / steps with the rc2 checkout (--repo)
  and with this one, then --compare: the default walls (simple) at K = 1 and K = 2 (200 steps; K = 2 also 999 steps,
  the last before the first defrag) in 2-D and K = 2 in 3-D, and adami at K = 1 (200 and 999 steps) must be
  BIT-IDENTICAL to rc2; adami at K = 2 is run twice with this build (the self check: no rc2 reference exists).
seam: dump_state (--version v7) of the adami 1000^2 case at the horizons 300 and 900 (before the first defrag at
  1000): K = 1 on device 1 twice (the reference A1 / A2), K = 2 on devices 0 / 1 twice (the test B1 / B2), then
  analyze.py per horizon on the fluid particles and on all particles (the walls' density_pressure.y is the adami
  wall pressure): test / noise ratios by distance to the seam.
cost: _run_v7_chain_bench.py of the adami case and its simple sibling at K = 1 (device 1) and K = 2 (devices 0 / 1),
  steady fps after the warmup, three alternating trials; the step time ratio adami / simple per K.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SIMPLE_CASE = "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml"
ADAMI_CASE = "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml"
CASE_3D = "cases/cavity3d_1m/case.yaml"
K1_DEVICE = "1"             # the headless 5090 (device 1 in the discrete-first order)
K2_DEVICES = "0,1"

# name, case, device map, steps, transport extension (particles may cross the cut), against rc2 (else self check)
BITEXACT_RUNS = (
    ("simple_k1_200", SIMPLE_CASE, K1_DEVICE, 200, False, True),
    ("simple_k2_200", SIMPLE_CASE, K2_DEVICES, 200, True, True),
    ("simple_k2_999", SIMPLE_CASE, K2_DEVICES, 999, True, True),
    ("adami_k1_200", ADAMI_CASE, K1_DEVICE, 200, False, True),
    ("adami_k1_999", ADAMI_CASE, K1_DEVICE, 999, False, True),
    ("adami_k2_200_self", ADAMI_CASE, K2_DEVICES, 200, True, False),
    # the 999-step comparisons differ for ~98 % of the particles on every field, also adami at K = 1 (no seam):
    # past ~200 steps two arrivals in one voxel in one step swap (the harness's own caveat), so the deterministic
    # window is probed at 500 steps and the 999-step runs are repeated with this build alone (the self checks)
    ("simple_k2_500", SIMPLE_CASE, K2_DEVICES, 500, True, True),
    ("adami_k1_500", ADAMI_CASE, K1_DEVICE, 500, False, True),
    ("simple_k2_999_self", SIMPLE_CASE, K2_DEVICES, 999, True, False),
    ("adami_k1_999_self", ADAMI_CASE, K1_DEVICE, 999, False, False),
    # adami K = 1 at 500 steps: rc2 vs rc3 differ by one ULP of rho / acceleration on every row with identical
    # positions (a voxel-list swap between steps 200 and 500 spread over the domain); the self checks of both builds
    # at 500 and the cross-build runs at 300 / 400 locate the deterministic window
    ("adami_k1_300", ADAMI_CASE, K1_DEVICE, 300, False, True),
    ("adami_k1_400", ADAMI_CASE, K1_DEVICE, 400, False, True),
    ("adami_k1_500_self", ADAMI_CASE, K1_DEVICE, 500, False, False),
    ("adami_k1_500_rc2self", ADAMI_CASE, K1_DEVICE, 500, False, "rc2"),
    # cavity3d_1m with --canonical-lists at K = 2 overflows the ghost pool and the install tail on rc2 and rc3 alike
    # (177k migrations in 200 steps, alive 1,671,840 of 1,685,159: the canonical bootstrap without its defrag);
    # no 3-D bit-exact comparison: the 3-D layout is covered by _test_seam_layout.check_adami_chain
)
SEAM_HORIZONS = "300,900"
COST_WARMUP = 1000
COST_STEPS = 4000
COST_TRIALS = 3


def run_logged(command: list, log_path: pathlib.Path, cwd=None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(part) for part in command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=cwd or _REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        handle.write(f"\n[exit {completed.returncode} after {time.perf_counter() - started:.1f} s]\n")
    print(f"[gates] {log_path.name}: exit {completed.returncode} ({time.perf_counter() - started:.1f} s)", flush=True)
    return completed.returncode


def canonical_dump(case: str, device_map: str, steps: int, out: pathlib.Path, extension: bool,
                   repo: pathlib.Path | None, log_path: pathlib.Path) -> int:
    command = [PYTHON, str(_REPO_ROOT / "experiment/seam_audit/canonical_dump.py"), "--solver", "v7",
               "--case", case, "--device-map", device_map, "--steps", str(steps), "--canonical-lists",
               "--out", str(out)]
    if extension:
        command.append("--transport-extension")
    if repo is not None:
        command += ["--repo", str(repo)]
    return run_logged(command, log_path)


def compare(first: pathlib.Path, second: pathlib.Path, log_path: pathlib.Path) -> int:
    return run_logged([PYTHON, "-m", "experiment.seam_audit.canonical_dump", "--compare", str(first), str(second)],
                      log_path)


def stage_bitexact(arguments) -> int:
    out = pathlib.Path(arguments.out).resolve()
    rc2 = pathlib.Path(arguments.rc2).resolve()
    verdicts = {}
    for name, case, device_map, steps, extension, mode in BITEXACT_RUNS:
        if arguments.only and name not in arguments.only:
            continue
        # mode True: rc2 vs rc3; False: rc3 twice (self check); "rc2": rc2 twice (the rc2 build's own self check)
        if mode is True:
            labels, repos = ("rc2", "rc3"), (rc2, None)
        elif mode == "rc2":
            labels, repos = ("rc2a", "rc2b"), (rc2, rc2)
        else:
            labels, repos = ("rc3a", "rc3b"), (None, None)
        first, second = out / f"{name}_{labels[0]}.npz", out / f"{name}_{labels[1]}.npz"
        codes = [canonical_dump(case, device_map, steps, first, extension, repos[0], out / f"{name}_{labels[0]}.log"),
                 canonical_dump(case, device_map, steps, second, extension, repos[1], out / f"{name}_{labels[1]}.log")]
        if any(codes):
            verdicts[name] = f"dump failed (exit {codes})"
            continue
        code = compare(first, second, out / f"{name}.compare.log")
        text = (out / f"{name}.compare.log").read_text(encoding="utf-8")
        verdict = "BIT-IDENTICAL" if "BIT-IDENTICAL" in text and code == 0 else "NOT bit-identical"
        verdicts[name] = verdict
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    print("[gates] bitexact verdicts:")
    for name, verdict in verdicts.items():
        print(f"  {name:22s} {verdict}")
    return 0 if all(verdict == "BIT-IDENTICAL" for verdict in verdicts.values()) else 1


def stage_seam(arguments) -> int:
    out = pathlib.Path(arguments.out).resolve()
    dumps = out / "dumps"
    runs = (("adami_K1_t1", "1", K1_DEVICE), ("adami_K1_t2", "1", K1_DEVICE),
            ("adami_K2_t1", "2", K2_DEVICES), ("adami_K2_t2", "2", K2_DEVICES))
    for run_name, slabs, device_map in runs:
        if arguments.only and run_name not in arguments.only:
            continue
        code = run_logged([PYTHON, "-m", "experiment.seam_audit.dump_state", "--version", "v7", "--case", ADAMI_CASE,
                           "--slabs", slabs, "--device-map", device_map, "--horizons", SEAM_HORIZONS,
                           "--out-dir", str(dumps), "--run-name", run_name], out / f"{run_name}.log")
        if code:
            print(f"[gates] {run_name} failed (exit {code})")
            return code
    results = {}
    for horizon in SEAM_HORIZONS.split(","):
        for particles in ("fluid", "all"):
            analysis = out / f"analysis_N{horizon}_{particles}"
            code = run_logged([PYTHON, "-m", "experiment.seam_audit.analyze",
                               "--reference", str(dumps / f"adami_K1_t1_N{horizon}.npz"),
                               str(dumps / f"adami_K1_t2_N{horizon}.npz"),
                               "--test", str(dumps / f"adami_K2_t1_N{horizon}.npz"),
                               str(dumps / f"adami_K2_t2_N{horizon}.npz"),
                               "--particles", particles, "--out", str(analysis)], out / f"analyze_N{horizon}_{particles}.log")
            results[f"N{horizon}_{particles}"] = {"exit": code, "report": str(analysis / "report.md")}
    (out / "analyses.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(item["exit"] == 0 for item in results.values()) else 1


STEADY_PATTERN = re.compile(r"STEADY \(post-warmup \d+\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps")


def stage_cost(arguments) -> int:
    out = pathlib.Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configurations = [("simple", SIMPLE_CASE), ("adami", ADAMI_CASE)]
    results = []
    for trial in range(COST_TRIALS):
        order = configurations if trial % 2 == 0 else configurations[::-1]
        for slab_count, device_map in ((1, K1_DEVICE), (2, K2_DEVICES)):
            for label, case in order:
                name = f"{label}_k{slab_count}_t{trial}"
                log_path = out / f"{name}.log"
                code = run_logged([PYTHON, str(_REPO_ROOT / "experiment/v7/_run_v7_chain_bench.py"), "--case", case,
                                   "--weights", ",".join(["1.0"] * slab_count), "--device-map", device_map,
                                   "--warmup", str(COST_WARMUP), "--max-steps", str(COST_WARMUP + COST_STEPS)],
                                  log_path)
                text = log_path.read_text(encoding="utf-8")
                match = STEADY_PATTERN.search(text)
                fps = float(match.group(3)) if match else None
                invalid = any(key in text for key in ("INVALID", "overflow_", "drift") if f"{key}" in text and
                              re.search(rf"{key}\w*[=: ]+[1-9]", text))
                results.append({"label": label, "K": slab_count, "trial": trial, "exit": code, "steady_fps": fps,
                                "steady_steps": int(match.group(1)) if match else None, "log": log_path.name})
                print(f"[gates] {name}: exit {code}, steady {fps} fps", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["| K | simple fps (trials) | adami fps (trials) | step time adami / simple |", "|---|---|---|---|"]
    for slab_count in (1, 2):
        simple = [r["steady_fps"] for r in results if r["label"] == "simple" and r["K"] == slab_count and r["steady_fps"]]
        adami = [r["steady_fps"] for r in results if r["label"] == "adami" and r["K"] == slab_count and r["steady_fps"]]
        if simple and adami:
            ratio = (sum(simple) / len(simple)) / (sum(adami) / len(adami))
            lines.append(f"| {slab_count} | {' / '.join(f'{v:.1f}' for v in simple)} | "
                         f"{' / '.join(f'{v:.1f}' for v in adami)} | {ratio:.4f} (+{100 * (ratio - 1):.2f} %) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(r["exit"] == 0 and r["steady_fps"] for r in results) else 1


TRACE_INTERVALS = (
    # label, end tick, start ticks (the first present, in order)
    ("phase A", "a_end", ("a_start",)),
    ("phase B", "b_end", ("b_start",)),
    ("  correction_density_interior", "b_correction_density_interior_end", ("b_deep_wall_marker_end", "b_start")),
    ("  wall pass (interior)", "b_wall_extrapolate_end", ("b_correction_density_interior_end",)),
    ("  force_deep_interior", "b_force_deep_interior_end",
     ("b_wall_extrapolate_end", "b_correction_density_interior_end")),
    ("phase C", "c_end", ("c_start",)),
    ("  install + append + expand", "c_append_departed_end", ("c_start",)),
    ("  correction_density band", "c_correction_density_boundary_end",
     ("c_append_departed_end", "c_install_trailing_end", "c_install_leading_end", "c_expand_end", "c_start")),
    ("  density copy", "c_density_end", ("c_correction_density_boundary_end",)),
    ("  wall band pass", "c_wall_extrapolate_end", ("c_density_end",)),
    ("  force band", "c_force_end", ("c_wall_extrapolate_end", "c_density_end")),
    ("B -> C gap", "c_start", ("b_end",)),
)


def stage_trace(arguments) -> int:
    """Median per-step intervals (us) of the step traces under --out/<name>/steps_device.csv (the chain bench's
    --step-trace --step-trace-detail full), one column per (trace, sim); the step period from a_start."""
    import csv
    import statistics
    out = pathlib.Path(arguments.out).resolve()
    columns = {}
    for trace_directory in sorted(path for path in out.iterdir() if (path / "steps_device.csv").exists()):
        with open(trace_directory / "steps_device.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sims = sorted({row["sim"] for row in rows})
        for sim in sims:
            mine = [row for row in rows if row["sim"] == sim and row.get("a_start")]
            values = {}
            starts = [float(row["a_start"]) for row in mine if row["a_start"]]
            periods = [b - a for a, b in zip(starts, starts[1:]) if b > a]
            values["step period"] = statistics.median(periods) / 1000.0 if periods else None
            for label, end, start_candidates in TRACE_INTERVALS:
                samples = []
                for row in mine:
                    if not row.get(end):
                        continue
                    start = next((row[name] for name in start_candidates if row.get(name)), None)
                    if start is None:
                        continue
                    samples.append((float(row[end]) - float(start)) / 1000.0)
                values[label] = statistics.median(samples) if samples else None
            columns[f"{trace_directory.name} s{sim}"] = values
    labels = ["step period"] + [label for label, _, _ in TRACE_INTERVALS]
    lines = ["| interval (median us) | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(
            "-" if columns[name].get(label) is None else f"{columns[name][label]:.1f}" for name in columns) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("bitexact", "seam", "cost", "trace"))
    parser.add_argument("--out", default=None, help="log / dump directory (default logs/e40/<stage>)")
    parser.add_argument("--rc2", default="../vulkan-demo-v7-wake", help="bitexact: the rc2 checkout")
    parser.add_argument("--only", nargs="*", default=None, help="run names to run (default: all)")
    arguments = parser.parse_args()
    if arguments.out is None:
        arguments.out = str(_REPO_ROOT / "logs" / "e40" / arguments.stage)
    return {"bitexact": stage_bitexact, "seam": stage_seam, "cost": stage_cost,
            "trace": stage_trace}[arguments.stage](arguments)


if __name__ == "__main__":
    sys.exit(main())
