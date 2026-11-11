"""
micropolar_stage5_cavity.py - Stage 5, section 4: the lid-driven cavity at Re = 1000 (n = 250 and 500, thesis
configuration otherwise) for the candidates for the wall and the density-diffusion term.

  thesis   cases/viscous_cavity_n{n}_laplacian_eps1em6 (the runs of 2026-11-11 under logs/micropolar/cavity_forms)
  nd       the same case with numerics.use_density_diffusion: false
  fd       the same case with numerics.delta_fluid_only: true
  ew       the same case with numerics.wall_density: eos

  run       cavity_runner.py runs (K = 1, v7, release switches; the options of the thesis runs) of the named
            candidates at one resolution on one device.
  analyse   Errors against Marchi et al. 2021 at the 28 + 28 reference points (as viscous_form_campaign.py), the
            four points nearest the lid (u) and nearest the right wall (v), the velocity into each wall and its
            relation to the wall pressure (from the snapshots of the last 20 time units), the density range and
            the level of the density.
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

THESIS_CASE = "cases/viscous_cavity_n{n}_laplacian_eps1em6/case.yaml"
CASES = {"thesis": THESIS_CASE, "nd": "cases/stage5_cavity_n{n}_nd/case.yaml",
         "fd": "cases/stage5_cavity_n{n}_fd/case.yaml", "ew": "cases/stage5_cavity_n{n}_ew/case.yaml"}
THESIS_RUNS = _REPO_ROOT / "logs" / "micropolar" / "cavity_forms"
RUNS = _REPO_ROOT / "logs" / "micropolar" / "stage5_cavity"
DENSITY, WALL_SPEED, SOUND_SPEED, EXPONENT = 1000.0, 1.0, 100.0, 7.0


def run_directory(candidate: str, resolution: int) -> pathlib.Path:
    if candidate == "thesis":
        return THESIS_RUNS / f"n{resolution}_laplacian_eps1em6"
    return RUNS / f"n{resolution}_{candidate}"


def make_cases(resolution: int) -> None:
    source = (_REPO_ROOT / THESIS_CASE.format(n=resolution)).read_text(encoding="utf-8").replace("\r\n", "\n")
    key = "  use_density_diffusion: true\n"
    if source.count(key) != 1:
        raise RuntimeError("numerics.use_density_diffusion line not found once")
    variants = {
        "nd": (source.replace(key, "  use_density_diffusion: false\n"),
               "numerics.use_density_diffusion: false"),
        "fd": (source.replace(key, key + "  delta_fluid_only: true\n"),
               "numerics.delta_fluid_only: true (density diffusion between fluid pairs only)"),
        "ew": (source.replace(key, key + "  wall_density: eos\n"),
               "numerics.wall_density: eos (adami walls store the Tait density of their pressure)"),
    }
    for name, (text, change) in variants.items():
        header = (f"# VARIANT of cases/viscous_cavity_n{resolution}_laplacian_eps1em6 (the thesis configuration) for "
                  f"docs/micropolar_design.md Section 14: {change}; everything else identical.\n")
        text = text.replace("schema_version: 2\n", "schema_version: 2\n\n" + header, 1)
        path = _REPO_ROOT / CASES[name].format(n=resolution)
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def command_run(arguments) -> int:
    from experiment.validation.cavity_campaign import environment
    make_cases(arguments.resolution)
    for candidate in arguments.candidates:
        run_dir = run_directory(candidate, arguments.resolution)
        if (run_dir / "done.json").exists():
            print(f"[stage5 cavity] {run_dir.name}: done already", flush=True)
            continue
        command = [sys.executable, "-m", "experiment.validation.cavity_runner", "--solver", "v7",
                   "--case", CASES[candidate].format(n=arguments.resolution), "--run-dir", str(run_dir), "--slabs", "1",
                   "--device-map", str(arguments.device), "--expect", "release", "--t-end", str(arguments.t_end),
                   "--t-max", str(arguments.t_max)]
        if run_dir.exists():
            command.append("--resume")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(run_dir.parent / f"{run_dir.name}.log", "a", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=_REPO_ROOT, env=environment("release", "v7"), stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        print(f"[stage5 cavity] {run_dir.name}: exit {code}", flush=True)
        if code == 0:
            (run_dir / "done.json").write_text(json.dumps({"command": command}), encoding="utf-8")
    return 0


def wall_velocities(run_dir: pathlib.Path, spacing: float, bins: int = 50, layer_spacings=(0.5, 12.0),
                    degree: int = 3) -> dict:
    """From the snapshots of a run (one per time unit over its last 20): per wall, in bins along the wall, the
    velocity of the fluid into the wall and its pressure as cubics in the wall distance over the fluid between 0.5
    and 12 dx from the wall surface, evaluated at the surface. The pressure is the Tait pressure of the density
    m / V (the snapshots keep the particle volumes; the wall particles keep rho0 and give m / rho0). The corner
    squares of side 12 dx are left out. Slope: velocity into the wall over U against wall pressure over rho0 U^2."""
    files = sorted((run_dir / "snapshots").glob("t*.npz"))
    count = degree + 1
    walls = ("bottom", "left", "right", "lid")
    sums = {wall: {"matrix": np.zeros((bins, count, count)), "velocity": np.zeros((bins, count)),
                   "pressure": np.zeros((bins, count))} for wall in walls}
    reach = layer_spacings[1] * spacing
    stiffness = DENSITY * SOUND_SPEED ** 2 / EXPONENT
    for path in files:
        with np.load(path) as archive:
            positions = archive["positions"].astype(np.float64)
            velocities = archive["velocities"].astype(np.float64)
            volumes = archive["volumes"].astype(np.float64)
            material = archive["material"]
        fluid = material == 0
        rest_volume = float(np.median(volumes[~fluid]))                     # m / rho0
        pressure = stiffness * ((rest_volume / volumes[fluid]) ** EXPONENT - 1.0)
        x, y = positions[fluid, 0], positions[fluid, 1]
        u, v = velocities[fluid, 0], velocities[fluid, 1]
        # the cavity is [-0.5, 0.5]^2; distance from each wall surface, coordinate along it, velocity into it
        geometry = {"bottom": (y + 0.5, x, -v), "left": (x + 0.5, y, -u), "right": (0.5 - x, y, u),
                    "lid": (0.5 - y, x, v)}
        for wall, (distance, along, into_wall) in geometry.items():
            layer = ((distance > layer_spacings[0] * spacing) & (distance < reach) & (np.abs(along) < 0.5 - reach))
            scaled = distance[layer] / spacing
            sector = np.minimum(((along[layer] + 0.5 - reach) / (1.0 - 2.0 * reach) * bins).astype(np.int64), bins - 1)
            powers = [scaled ** power for power in range(2 * count - 1)]
            for row in range(count):
                sums[wall]["velocity"][:, row] += np.bincount(sector, into_wall[layer] * powers[row], bins)
                sums[wall]["pressure"][:, row] += np.bincount(sector, pressure[layer] * powers[row], bins)
                for column in range(count):
                    sums[wall]["matrix"][:, row, column] += np.bincount(sector, powers[row + column], bins)
    result = {"snapshots": len(files),
              "along": (-0.5 + reach + (np.arange(bins) + 0.5) * (1.0 - 2.0 * reach) / bins).tolist()}
    all_velocity, all_pressure = [], []
    for wall in walls:
        velocity_at_wall = np.linalg.solve(sums[wall]["matrix"], sums[wall]["velocity"][:, :, None])[:, 0, 0] / WALL_SPEED
        pressure_at_wall = (np.linalg.solve(sums[wall]["matrix"], sums[wall]["pressure"][:, :, None])[:, 0, 0]
                            / (DENSITY * WALL_SPEED ** 2))
        slope, intercept = np.polyfit(pressure_at_wall, velocity_at_wall, 1)
        result[wall] = {"velocity_into_wall": velocity_at_wall.tolist(), "wall_pressure": pressure_at_wall.tolist(),
                        "slope": float(slope), "intercept": float(intercept),
                        "correlation": float(np.corrcoef(pressure_at_wall, velocity_at_wall)[0, 1]),
                        "velocity_rms": float(np.sqrt((velocity_at_wall ** 2).mean())),
                        "velocity_mean": float(velocity_at_wall.mean()),
                        "velocity_min": float(velocity_at_wall.min()), "velocity_max": float(velocity_at_wall.max()),
                        "pressure_min": float(pressure_at_wall.min()), "pressure_max": float(pressure_at_wall.max())}
        all_velocity.append(velocity_at_wall)
        all_pressure.append(pressure_at_wall)
    velocity, pressure = np.concatenate(all_velocity), np.concatenate(all_pressure)
    slope, intercept = np.polyfit(pressure, velocity, 1)
    result["all_walls"] = {"slope": float(slope), "intercept": float(intercept),
                           "correlation": float(np.corrcoef(pressure, velocity)[0, 1]),
                           "velocity_rms": float(np.sqrt((velocity ** 2).mean())),
                           "pressure_mean": float(pressure.mean())}
    return result


def summarise(run_dir: pathlib.Path, average_span: float) -> dict:
    from experiment.validation import cavity_reference
    from experiment.validation import viscous_form_campaign
    summary = viscous_form_campaign.summarise(run_dir, average_span)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    reference = cavity_reference.marchi2021()
    for name in ("u", "v"):
        coordinate, expected = np.asarray(reference[name][0]), np.asarray(reference[name][1])
        error = np.asarray(summary[f"{name}_mean"]) - expected
        # u(y) on x = 0.5: the four points nearest the lid; v(x) on y = 0.5: the four nearest the right wall
        nearest = np.argsort(coordinate)[-4:]
        others = np.setdiff1d(np.arange(coordinate.size), nearest)
        interior = (coordinate > 0.05) & (coordinate < 0.95)
        summary[f"{name}_coordinates"] = coordinate.tolist()
        summary[f"{name}_error"] = error.tolist()
        summary[f"{name}_nearest_coordinates"] = coordinate[nearest].tolist()
        summary[f"{name}_nearest_errors"] = error[nearest].tolist()
        summary[f"{name}_nearest_rms"] = float(np.sqrt((error[nearest] ** 2).mean()))
        summary[f"{name}_others_rms"] = float(np.sqrt((error[others] ** 2).mean()))
        summary[f"{name}_interior_rms"] = float(np.sqrt((error[interior] ** 2).mean()))
        summary[f"{name}_interior_points"] = int(interior.sum())
    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary["density_history"] = [[row["time"], row["density_min"], row["density_max"]] for row in rows
                                  if abs(row["time"] / 10.0 - round(row["time"] / 10.0)) < 1e-6]
    summary["spacing"] = meta["spacing"]
    summary["timestep"] = meta["dt"]
    summary["git"] = meta.get("git")
    summary["walls"] = wall_velocities(run_dir, meta["spacing"])
    # the level of the fluid density in the last snapshot (m / V against the wall particles' m / rho0)
    files = sorted((run_dir / "snapshots").glob("t*.npz"))
    levels = []
    for path in (files[0], files[-1]):
        with np.load(path) as archive:
            volumes, material = archive["volumes"].astype(np.float64), archive["material"]
        fluid = material == 0
        levels.append(float(DENSITY * (np.median(volumes[~fluid]) / volumes[fluid]).mean()))
    summary["density_mean_first_snapshot"], summary["density_mean_last_snapshot"] = levels
    summary["snapshot_times"] = [int(files[0].stem[1:]) * meta["dt"], int(files[-1].stem[1:]) * meta["dt"]]
    return summary


def command_analyse(arguments) -> int:
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    say("Lid-driven cavity, Re = 1000, thesis configuration otherwise; mean over the last "
        f"{arguments.average_span:g} time units; errors against Marchi et al. 2021 (MLS samples, wall frame), "
        "in units of U")
    say(f"{'n':>5}{'cand.':>8}{'t_end':>8}{'L2 u':>10}{'Linf u':>10}{'L2 v':>10}{'Linf v':>10}"
        f"{'u: 4 at lid':>13}{'others':>10}{'v: 4 at wall':>14}{'others':>10}{'u_min':>10}{'v_max':>10}{'v_min':>10}"
        f"{'E_k':>10}{'rho min':>10}{'rho max':>10}{'rho mean':>11}{'steady at':>10}")
    for resolution in arguments.resolutions:
        for candidate in CASES:
            run_dir = run_directory(candidate, resolution)
            if not (run_dir / "samples.jsonl").exists() or not list((run_dir / "snapshots").glob("t*.npz")):
                continue
            item = summarise(run_dir, arguments.average_span)
            item["finished"] = (run_dir / "done.json").exists()
            records[f"n{resolution}_{candidate}"] = item
            steady = item.get("steady", {}).get("steady_time")
            say(f"{resolution:>5}{candidate:>8}{item['end_time']:>8.1f}{item['u_l2']:>10.2e}{item['u_linf']:>10.2e}"
                f"{item['v_l2']:>10.2e}{item['v_linf']:>10.2e}{item['u_nearest_rms']:>13.2e}"
                f"{item['u_others_rms']:>10.2e}{item['v_nearest_rms']:>14.2e}{item['v_others_rms']:>10.2e}"
                f"{item['u_min']:>10.5f}{item['v_max']:>10.5f}{item['v_min']:>10.5f}"
                f"{item['kinetic_energy_mean']:>10.4f}{item['density_min']:>10.3f}{item['density_max']:>10.3f}"
                f"{item['density_mean_last_snapshot']:>11.4f}"
                + (f"{steady:>10.2f}" if steady is not None else f"{'-':>10}")
                + ("" if item["finished"] else "  (not finished)"))
    stopped = {}
    for resolution in arguments.resolutions:
        for candidate in CASES:
            run_dir = run_directory(candidate, resolution)
            if f"n{resolution}_{candidate}" in records or not (run_dir / "samples.jsonl").exists():
                continue
            rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
                    if line]
            log = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "run.log").exists() \
                else ""
            failures = [line.split("failed: ", 1)[1].strip() for line in log.splitlines() if "failed: " in line]
            stopped[f"n{resolution}_{candidate}"] = {
                "last_sample_time": rows[-1]["time"], "samples": len(rows), "failure": failures[-1] if failures else None,
                "max_speed_max": float(max(row["max_speed"] for row in rows)),
                "max_speed_first": rows[0]["max_speed"],
                "density_min": float(min(row["density_min"] for row in rows)),
                "density_max": float(max(row["density_max"] for row in rows)),
                "density_min_last": rows[-1]["density_min"], "density_max_last": rows[-1]["density_max"],
                "kinetic_energy_last": rows[-1]["kinetic_energy"]}
    if stopped:
        say()
        say("runs without an averaging window (stopped, or still running)")
        say(f"{'n':>5}{'cand.':>8}{'last sample at t':>18}{'largest fluid speed / U':>25}{'density min':>13}{'max':>10}"
            f"   failure")
        for key, item in stopped.items():
            resolution, candidate = key.split("_")
            say(f"{resolution[1:]:>5}{candidate:>8}{item['last_sample_time']:>18.2f}{item['max_speed_max']:>25.3f}"
                f"{item['density_min']:>13.2f}{item['density_max']:>10.2f}   {item['failure'] or 'none recorded'}")
        records["_stopped"] = stopped
    say()
    say("errors at the four points nearest the lid (u on x = 0.5) and nearest the right wall (v on y = 0.5)")
    say(f"{'n':>5}{'cand.':>8}   " + f"{'u at y =':<12}" + " " * 36 + f"{'v at x =':<12}")
    finished_records = {key: item for key, item in records.items() if not key.startswith("_")}
    for key, item in finished_records.items():
        resolution, candidate = key.split("_")
        say(f"{resolution[1:]:>5}{candidate:>8}   "
            + " ".join(f"{coordinate:.3f}: {error:+.2e}" for coordinate, error
                       in zip(item["u_nearest_coordinates"], item["u_nearest_errors"]))
            + "   |   "
            + " ".join(f"{coordinate:.3f}: {error:+.2e}" for coordinate, error
                       in zip(item["v_nearest_coordinates"], item["v_nearest_errors"])))
    say()
    say("velocity into each wall at its surface (cubic in the wall distance over 0.5 to 12 dx, 50 bins along the "
        "wall, corners left out, snapshots of the last 20 time units), over U; slope against the wall pressure over "
        "rho0 U^2 and its correlation")
    say(f"{'n':>5}{'cand.':>8}" + "".join(f"{wall + ': rms':>14}{'slope':>10}{'r':>7}"
                                           for wall in ("bottom", "left", "right", "lid"))
        + f"{'all: rms':>12}{'slope':>10}{'r':>7}{'slope / Ma':>12}{'mean p_w':>10}")
    for key, item in finished_records.items():
        resolution, candidate = key.split("_")
        walls = item["walls"]
        say(f"{resolution[1:]:>5}{candidate:>8}"
            + "".join(f"{walls[wall]['velocity_rms']:>14.2e}{walls[wall]['slope']:>+10.5f}"
                      f"{walls[wall]['correlation']:>+7.2f}" for wall in ("bottom", "left", "right", "lid"))
            + f"{walls['all_walls']['velocity_rms']:>12.2e}{walls['all_walls']['slope']:>+10.5f}"
            f"{walls['all_walls']['correlation']:>+7.2f}{walls['all_walls']['slope'] / (WALL_SPEED / SOUND_SPEED):>12.3f}"
            f"{walls['all_walls']['pressure_mean']:>+10.4f}")
    say()
    say("thesis run minus candidate: difference of the velocity into the wall (over U) against the wall pressure of "
        "the thesis run (over rho0 U^2), per wall and for the four walls together: slope, correlation")
    say(f"{'n':>5}{'cand.':>8}" + "".join(f"{wall + ': slope':>16}{'r':>7}" for wall in ("bottom", "left", "right", "lid"))
        + f"{'all: slope':>14}{'r':>7}{'slope / Ma':>12}{'rms difference':>16}")
    for key, item in finished_records.items():
        resolution, candidate = key.split("_")
        reference = records.get(f"{resolution}_thesis")
        if candidate == "thesis" or reference is None:
            continue
        row, velocities, pressures = "", [], []
        item["difference_to_thesis"] = {}
        for wall in ("bottom", "left", "right", "lid"):
            difference = (np.asarray(reference["walls"][wall]["velocity_into_wall"])
                          - np.asarray(item["walls"][wall]["velocity_into_wall"]))
            pressure = np.asarray(reference["walls"][wall]["wall_pressure"])
            slope = float(np.polyfit(pressure, difference, 1)[0])
            correlation = float(np.corrcoef(pressure, difference)[0, 1])
            item["difference_to_thesis"][wall] = {"slope": slope, "correlation": correlation}
            row += f"{slope:>+16.5f}{correlation:>+7.2f}"
            velocities.append(difference)
            pressures.append(pressure)
        difference, pressure = np.concatenate(velocities), np.concatenate(pressures)
        slope = float(np.polyfit(pressure, difference, 1)[0])
        correlation = float(np.corrcoef(pressure, difference)[0, 1])
        item["difference_to_thesis"]["all_walls"] = {"slope": slope, "correlation": correlation,
                                                     "rms": float(np.sqrt((difference ** 2).mean()))}
        say(f"{resolution[1:]:>5}{candidate:>8}" + row + f"{slope:>+14.5f}{correlation:>+7.2f}"
            f"{slope / (WALL_SPEED / SOUND_SPEED):>12.3f}{np.sqrt((difference ** 2).mean()):>16.2e}")
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--resolution", type=int, required=True)
    run.add_argument("--candidates", nargs="+", default=["nd", "fd"], choices=sorted(CASES))
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--t-end", type=float, default=60.0)
    run.add_argument("--t-max", type=float, default=120.0)
    run.set_defaults(handler=command_run)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("--resolutions", type=int, nargs="+", default=[250, 500])
    analyse.add_argument("--average-span", type=float, default=20.0)
    analyse.add_argument("--json", default=None)
    analyse.set_defaults(handler=command_analyse)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
