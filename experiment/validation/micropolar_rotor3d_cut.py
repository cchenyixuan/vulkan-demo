"""
micropolar_rotor3d_cut.py - host against card within one support of a slab cut, on a saved two-slab state of the
3-D case (docs/micropolar_design.md 16.0, Stage 7).

The acceleration check of micropolar_rotor3d_torque.py (every pair term of force.comp summed on the host over the
saved neighbours, against the card's acceleration) reads 1e-6 of the rms acceleration away from a cut and 2e-3 to
4e-3 within one support of it. This tool tests where that comes from, without a card:

  1. Is the saved pressure of a fluid particle the equation of state of its saved density? (The density is stored
     in single precision as an absolute value; one unit in its last place is 6.1e-5 kg/m^3, 0.15 Pa.)
  2. The host sum again with the pressure of the fluid neighbours across the cut taken from the equation of state
     of their saved density (single-precision arithmetic), and with that of every fluid neighbour, by the distance
     of the target from the cut.
  3. The same for targets whose neighbourhood holds no wall particle.

  python -m experiment.validation.micropolar_rotor3d_cut g40_k3_m2_k2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import micropolar_rotor3d_torque as torque                     # noqa: E402

REST_DENSITY, POWER = 1000.0, 7.0
BINS = ((-200, -8), (-8, -4), (-4, 0), (0, 4), (4, 8), (8, 200))        # distance from the cut in spacings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--directory", default="logs/micropolar/stage6_rotor3d")
    parser.add_argument("--out", default="data/micropolar/stage6/rotor3d_cut_check.txt")
    arguments = parser.parse_args()
    lines, result = [], {}

    def say(text=""):
        lines.append(text)
        print(text, flush=True)

    run_directory = _REPO_ROOT / arguments.directory / arguments.run
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    meta = run["rotor3d"]
    parameters = torque.parameters_of(run)
    path = sorted((run_directory / "states").glob("state_*.npz"))[-1]
    with np.load(path) as archive:
        raw = {name: np.array(archive[name]) for name in archive.files}
    state = torque.State(raw, parameters)
    support, spacing = parameters["support"], parameters["spacing"]
    cut = run["cuts_x"][0]
    sound = meta["speed_of_sound"]
    say(f"{arguments.run}, {path.name}: {state.position.shape[0]:,} saved rows; slabs on devices {run['device_map']}, "
        f"cut at x = {cut:.6f} m; support {support:g} m = {support / spacing:g} dx")
    result.update({"run": arguments.run, "state": path.name, "cut_x": cut, "device_map": run["device_map"],
                   "support": support, "spacing": spacing})

    # ---- 1. the saved pressure against the equation of state of the saved density --------------------------------
    density = raw["density"].astype(np.float32)
    constant = np.float32(REST_DENSITY * sound ** 2 / POWER)
    eos = (constant * (np.power(density / np.float32(REST_DENSITY), np.float32(POWER)) - np.float32(1.0))
           ).astype(np.float64)
    fluid = state.is_fluid
    saved = raw["pressure"].astype(np.float64)
    unit = float(np.spacing(np.float32(REST_DENSITY)))
    difference = eos[fluid] - saved[fluid]
    result["pressure"] = {
        "saved_rms": float(np.sqrt((saved[fluid] ** 2).mean())), "density_unit_in_the_last_place": unit,
        "pressure_of_one_density_unit": sound ** 2 * unit,
        "host_equation_of_state_minus_saved_rms": float(np.sqrt((difference ** 2).mean())),
        "host_equation_of_state_minus_saved_max": float(np.abs(difference).max()),
        "share_equal_to_the_bit": float((difference == 0.0).mean())}
    say(f"1. Fluid rows: rms of the saved pressure {result['pressure']['saved_rms']:.1f} Pa. One unit in the last place "
        f"of the density is {unit:.2e} kg/m^3, that is {sound ** 2 * unit:.3f} Pa. The equation of state of the saved "
        f"density, evaluated on the host in single precision, differs from the saved pressure by "
        f"{result['pressure']['host_equation_of_state_minus_saved_rms']:.2f} Pa rms (at most "
        f"{result['pressure']['host_equation_of_state_minus_saved_max']:.2f} Pa); "
        f"{100.0 * result['pressure']['share_equal_to_the_bit']:.0f} % of the rows agree to the bit.")

    def check(target: np.ndarray, label: str) -> dict:
        first, second, offset, distance = torque._pairs(state.position[target], state.position, support)
        first_global = target[first]
        terms = state.pair_terms(first_global, second, offset, distance)
        volume, corrected = terms["volume"], terms["corrected"]
        kernel_sum = np.bincount(first, volume * terms["kernel"], target.size)
        self_pressure = state.pressure[first_global]
        near_surface = kernel_sum[first] < 0.75
        coupling = parameters["coupling"] * volume[:, None] * np.cross(terms["spin_difference"], corrected)
        across = (state.position[first_global, 0] - cut) * (state.position[second, 0] - cut) < 0.0
        neighbour_fluid = state.is_fluid[second]
        card = state.acceleration[target]
        scale = float(np.sqrt((card ** 2).sum(axis=1).mean()))
        signed = (state.position[target, 0] - cut) / spacing
        with_wall = int((np.bincount(first, (~neighbour_fluid).astype(np.float64), target.size) > 0).sum())
        say(f"{label}: {target.size:,} targets ({with_wall:,} with a wall neighbour); rms of the card's acceleration "
            f"{scale:.3f} m/s^2")
        variants = {
            "saved pressures (the check as it is)": state.pressure[second],
            "fluid neighbours across the cut: equation of state of their saved density":
                np.where(across & neighbour_fluid, eos[second], state.pressure[second]),
            "every fluid neighbour: equation of state of its saved density":
                np.where(neighbour_fluid, eos[second], state.pressure[second]),
        }
        say("   rms of |host - card| over the rms acceleration, by the distance of the target from the cut in dx "
            "(negative: the slab of lower x)")
        say("   " + f"{'pressure of the neighbours in the host sum':<76}"
            + "".join(f"{f'{low} to {high}':>13}" for low, high in BINS))
        entry = {"targets": int(target.size), "acceleration_rms": scale, "bins": [list(pair) for pair in BINS],
                 "variants": {}}
        for name, neighbour_pressure in variants.items():
            combined = np.where((self_pressure > 0.0) | near_surface, neighbour_pressure + self_pressure,
                                neighbour_pressure - self_pressure)
            pressure = -(volume * combined / state.density[first_global])[:, None] * corrected
            total = np.column_stack([np.bincount(first, (pressure + terms["viscous"] + coupling)[:, column],
                                                 target.size) for column in range(3)])
            difference = np.sqrt(((total - card) ** 2).sum(axis=1)) / scale
            row = []
            for low, high in BINS:
                chosen = (signed >= low) & (signed < high)
                row.append(float(np.sqrt((difference[chosen] ** 2).mean())) if chosen.any() else None)
            entry["variants"][name] = row
            say("   " + f"{name:<76}" + "".join(f"{value:>13.1e}" if value is not None else f"{'-':>13}"
                                                for value in row))
        return entry

    azimuth = np.degrees(np.arctan2(state.position[:, 1], state.position[:, 0]))
    say()
    say("2. Targets of the check (fluid within one support of the rotor, a wedge of 12 degrees about the x axis)")
    result["near_the_rotor"] = check(np.nonzero(fluid & (state.to_rotor < support) & (np.abs(azimuth) < 6.0))[0],
                                     "   near the rotor")
    say()
    say("3. Targets whose whole neighbourhood is saved and holds no wall particle (between h and h + 0.9 dx from the "
        "rotor, more than h from the housing; a wedge of 40 degrees)")
    shell = (fluid & (state.to_rotor > support + 0.05 * spacing) & (state.to_rotor < support + 0.9 * spacing)
             & (state.to_housing > support + 0.6 * spacing) & (np.abs(azimuth) < 20.0))
    result["no_wall_neighbour"] = check(np.nonzero(shell)[0], "   one shell")
    target = _REPO_ROOT / arguments.out
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    target.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
