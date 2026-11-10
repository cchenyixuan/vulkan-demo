"""viscous_form_campaign.py - the Re = 1000 adami cavity with both numerics.viscous_form values
(docs/viscous_operator_comparison.md): cavity_runner.py runs (K = 1, v7, release switches) of the validation case
and of its laplacian variants, and a small analysis of the runs: centreline errors against Marchi et al. 2021
(the runner's MLS samples at the 28 + 28 reference points, wall frame, mean over the last --average-span time
units), kinetic energy and fluid density range.

    .venv/Scripts/python.exe -m experiment.validation.viscous_form_campaign run --resolution 250
    .venv/Scripts/python.exe -m experiment.validation.viscous_form_campaign analyse --resolution 250
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RUNS = {
    "monaghan_gingold": "cases/lid_driven_cavity_2d_n{n}_xi0p001_eps0p0025_adami/case.yaml",
    "laplacian": "cases/viscous_cavity_n{n}_laplacian/case.yaml",
    "laplacian_eps1em6": "cases/viscous_cavity_n{n}_laplacian_eps1em6/case.yaml",
}
LOGS = _REPO_ROOT / "logs" / "micropolar" / "cavity_forms"


def make_cases(resolution: int) -> None:
    """The two laplacian variants of the validation case: the same file with numerics.viscous_form: laplacian
    (and epsilon_squared_factor 1e-6), referring to the same particle files."""
    source = (_REPO_ROOT / RUNS["monaghan_gingold"].format(n=resolution)).read_text(encoding="utf-8").replace("\r\n", "\n")
    for name, factor in (("laplacian", None), ("laplacian_eps1em6", "1.0e-6")):
        text = source
        key = "  wall_boundary: adami"
        if text.count(key) != 1:
            raise RuntimeError("numerics.wall_boundary line not found once")
        text = text.replace(key, "  viscous_form: laplacian   # docs/viscous_operator_comparison.md\n" + key)
        if factor is not None:
            old = "  epsilon_squared_factor: 0.0025"
            if text.count(old) != 1:
                raise RuntimeError("epsilon_squared_factor line not found once")
            text = text.replace(old, f"  epsilon_squared_factor: {factor}")
        header = (f"# VARIANT of cases/lid_driven_cavity_2d_n{resolution}_xi0p001_eps0p0025_adami for "
                  f"docs/viscous_operator_comparison.md: numerics.viscous_form: laplacian"
                  + (f", epsilon_squared_factor {factor}" if factor else "") + "; everything else identical.\n")
        text = text.replace("schema_version: 2\n", "schema_version: 2\n\n" + header, 1)
        path = _REPO_ROOT / RUNS[name].format(n=resolution)
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def command_run(arguments) -> int:
    from experiment.validation.cavity_campaign import environment
    make_cases(arguments.resolution)
    for name in arguments.forms:
        run_dir = LOGS / f"n{arguments.resolution}_{name}"
        if (run_dir / "done.json").exists():
            print(f"[viscous_form] {run_dir.name}: done already", flush=True)
            continue
        command = [sys.executable, "-m", "experiment.validation.cavity_runner", "--solver", "v7",
                   "--case", RUNS[name].format(n=arguments.resolution), "--run-dir", str(run_dir), "--slabs", "1",
                   "--device-map", str(arguments.device), "--expect", "release", "--t-end", str(arguments.t_end),
                   "--t-max", str(arguments.t_max)]
        if run_dir.exists():
            command.append("--resume")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(run_dir.parent / f"{run_dir.name}.log", "a", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=_REPO_ROOT, env=environment("release", "v7"), stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        print(f"[viscous_form] {run_dir.name}: exit {code}", flush=True)
        if code == 0:
            (run_dir / "done.json").write_text(json.dumps({"command": command}), encoding="utf-8")
    return 0


def summarise(run_dir: pathlib.Path, average_span: float) -> dict:
    from experiment.validation import cavity_reference
    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    end_time = rows[-1]["time"]
    window = [row for row in rows if row["time"] > end_time - average_span + 1e-9]
    profiles = {"u": [], "v": []}
    for row in window:
        with np.load(run_dir / "samples" / f"s{row['step']:010d}.npz") as archive:
            profiles["u"].append(np.array(archive["u_marchi_wall_mls"]))
            profiles["v"].append(np.array(archive["v_marchi_wall_mls"]))
    reference = cavity_reference.marchi2021()
    summary = {"run": run_dir.name, "end_time": end_time, "samples": len(window)}
    for name in ("u", "v"):
        mean = np.mean(profiles[name], axis=0)
        error = mean - reference[name][1]
        summary[f"{name}_mean"] = mean.tolist()
        summary[f"{name}_l2"] = float(np.sqrt((error ** 2).mean()))
        summary[f"{name}_linf"] = float(np.abs(error).max())
        summary[f"{name}_min"], summary[f"{name}_max"] = float(mean.min()), float(mean.max())
    energy = np.array([row["kinetic_energy"] for row in window])
    spread = np.array([row["density_max"] - row["density_min"] for row in window])
    summary.update({
        "kinetic_energy_mean": float(energy.mean()), "kinetic_energy_std": float(energy.std(ddof=1)),
        "density_range_mean": float(spread.mean()), "density_range_max": float(spread.max()),
        "density_min": float(min(row["density_min"] for row in window)),
        "density_max": float(max(row["density_max"] for row in window)),
        "max_speed_max": float(max(row["max_speed"] for row in window)),
        "kinetic_energy_history": [[row["time"], row["kinetic_energy"]] for row in rows
                                   if abs(row["time"] / 5.0 - round(row["time"] / 5.0)) < 1e-6],
        "steps_per_second": float(np.median([row["fps"] for row in window])),
    })
    steady_path = run_dir / "steady.json"
    if steady_path.exists():
        summary["steady"] = json.loads(steady_path.read_text(encoding="utf-8"))
    return summary


def command_analyse(arguments) -> int:
    summaries = {}
    for name in RUNS:
        run_dir = LOGS / f"n{arguments.resolution}_{name}"
        if (run_dir / "samples.jsonl").exists():
            summaries[name] = summarise(run_dir, arguments.average_span)
    if not summaries:
        print("no runs")
        return 1
    print(f"n = {arguments.resolution}, Re = 1000, mean over the last {arguments.average_span:g} time units; errors "
          f"against Marchi et al. 2021 at the 28 + 28 points (MLS, wall frame), in units of U")
    print(f"{'run':<22}{'t_end':>8}{'L2 u':>10}{'Linf u':>10}{'L2 v':>10}{'Linf v':>10}{'u_min':>10}{'v_max':>10}"
          f"{'v_min':>10}{'E_k':>12}{'std E_k':>10}{'rho range':>11}{'max':>8}")
    for name, item in summaries.items():
        print(f"{name:<22}{item['end_time']:>8.1f}{item['u_l2']:>10.2e}{item['u_linf']:>10.2e}{item['v_l2']:>10.2e}"
              f"{item['v_linf']:>10.2e}{item['u_min']:>10.5f}{item['v_max']:>10.5f}{item['v_min']:>10.5f}"
              f"{item['kinetic_energy_mean']:>12.6f}{item['kinetic_energy_std']:>10.1e}"
              f"{item['density_range_mean']:>11.4f}{item['density_range_max']:>8.3f}")
    if "monaghan_gingold" in summaries:
        base = summaries["monaghan_gingold"]
        for name, item in summaries.items():
            if name == "monaghan_gingold":
                continue
            difference_u = np.array(item["u_mean"]) - np.array(base["u_mean"])
            difference_v = np.array(item["v_mean"]) - np.array(base["v_mean"])
            print(f"{name} - monaghan_gingold at the reference points: rms u {np.sqrt((difference_u ** 2).mean()):.2e} "
                  f"max {np.abs(difference_u).max():.2e}; rms v {np.sqrt((difference_v ** 2).mean()):.2e} max "
                  f"{np.abs(difference_v).max():.2e}; E_k ratio "
                  f"{item['kinetic_energy_mean'] / base['kinetic_energy_mean']:.5f}")
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--resolution", type=int, default=250)
    run.add_argument("--forms", nargs="+", default=list(RUNS), choices=list(RUNS))
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--t-end", type=float, default=100.0)
    run.add_argument("--t-max", type=float, default=200.0)
    run.set_defaults(handler=command_run)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("--resolution", type=int, default=250)
    analyse.add_argument("--average-span", type=float, default=20.0)
    analyse.add_argument("--json", default=None)
    analyse.set_defaults(handler=command_analyse)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
