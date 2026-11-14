"""
micropolar_startup.py - Stage 8c, Step A2: the start-up of plane Couette flow on the host (docs/micropolar_design.md
Section 19). 1-D, w scalar, the signs of the solver (a rigid rotation has curl v = 2 Omega and w = Omega; for
u(y) the vorticity is omega = -du/dy):

    rho   du/dt = (mu + kappa) d2u/dy2 + kappa dw/dy
    rho j dw/dt = gamma d2w/dy2 - kappa (du/dy + 2 w)

u(0, t) = U for t > 0 (impulsive start, as the runs), u(H, t) = 0, w(0, t) = w(H, t) = 0, u = w = 0 at t = 0. The
wall stress on the sliding wall is T_yx = (mu + kappa) du/dy + kappa w = (mu + kappa) du/dy (w = 0 there).

Units: rho = 1, l = 1, U = 1, mu + kappa / 2 = 1 (nu_inf = 1), K = kappa / mu = 3: mu = 0.4, kappa = 1.2, gamma =
l^2 kappa (2 mu + kappa) / (mu + kappa) = 1.5, j = theta l^2. The excess E(t) = tau_mp(t) / tau_N(t) - 1 against
Lambda(t) / l with Lambda = (mu + kappa / 2) U / tau_N(t) of the Newtonian fluid of viscosity mu + kappa / 2 depends
on K, H / l and theta only (the problem is linear; U and the Reynolds number drop out).

Space: second-order finite differences on a grid stretched geometrically from the sliding wall (first spacing d0,
ratio r) up to a uniform spacing; time: TR-BDF2 (L-stable, second order) with steps growing geometrically from
1e-12; the convergence check halves d0, the uniform spacing and the growth of the time steps.

  solve     the curves for theta = 1, 1e-2, 1e-4 and the Newtonian fluid, two resolutions -> data/micropolar/stage8c/
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIRECTORY = "data/micropolar/stage8c"
COUPLING_RATIO = 3.0
GAP_OVER_LENGTH = 20.0
THETAS = (1.0, 1.0e-2, 1.0e-4)
LAYER_TIMES = (0.25, 1.0, 4.0)          # profiles where Lambda / l = these


def material(theta: float | None) -> dict:
    """None: the Newtonian fluid of viscosity mu + kappa / 2 = 1."""
    if theta is None:
        return {"mu": 1.0, "kappa": 0.0, "gamma": 0.0, "inertia": 1.0}
    mu = 1.0 / (1.0 + 0.5 * COUPLING_RATIO)
    kappa = COUPLING_RATIO * mu
    gamma = kappa * (2.0 * mu + kappa) / (mu + kappa)          # l = 1
    return {"mu": mu, "kappa": kappa, "gamma": gamma, "inertia": theta}


def grid(first: float, ratio: float, uniform: float, height: float = GAP_OVER_LENGTH) -> np.ndarray:
    nodes = [0.0]
    spacing = first
    while nodes[-1] < height:
        nodes.append(nodes[-1] + spacing)
        spacing = min(spacing * ratio, uniform)
    nodes = np.asarray(nodes)
    return nodes * (height / nodes[-1])                      # end exactly at H


def operators(y: np.ndarray):
    """Sparse first and second derivative operators on the interior nodes (second order on the non-uniform grid),
    and the columns that multiply the boundary values y[0] and y[-1]."""
    from scipy import sparse
    lower = y[1:-1] - y[:-2]
    upper = y[2:] - y[1:-1]
    count = y.size - 2
    second_left = 2.0 / (lower * (lower + upper))
    second_middle = -2.0 / (lower * upper)
    second_right = 2.0 / (upper * (lower + upper))
    first_left = -upper / (lower * (lower + upper))
    first_middle = (upper - lower) / (lower * upper)
    first_right = lower / (upper * (lower + upper))
    second = sparse.diags([second_left[1:], second_middle, second_right[:-1]], [-1, 0, 1], shape=(count, count))
    first = sparse.diags([first_left[1:], first_middle, first_right[:-1]], [-1, 0, 1], shape=(count, count))
    boundary = {"second_left": second_left[0], "second_right": second_right[-1],
                "first_left": first_left[0], "first_right": first_right[-1]}
    return first.tocsc(), second.tocsc(), boundary


def solve(theta: float | None, first: float, ratio: float, uniform: float, growth: float, end_time: float,
          layer_times: dict | None = None) -> dict:
    """tau(t) on the sliding wall and the profiles at the requested times (time -> label)."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    y = grid(first, ratio, uniform)
    coefficients = material(theta)
    mu, kappa, gamma, inertia = (coefficients[key] for key in ("mu", "kappa", "gamma", "inertia"))
    first_derivative, second_derivative, boundary = operators(y)
    count = y.size - 2
    identity = sparse.identity(count, format="csc")
    # M dX/dt = A X + b, X = (u, w) on the interior; u(0) = 1, u(H) = 0, w = 0 on both walls
    a = sparse.bmat([[(mu + kappa) * second_derivative, kappa * first_derivative],
                     [-kappa * first_derivative, gamma * second_derivative - 2.0 * kappa * identity]], format="csc")
    b = np.zeros(2 * count)
    b[0] = (mu + kappa) * boundary["second_left"]               # u(0) = 1 in the u equation
    b[count] = -kappa * boundary["first_left"]                  # u(0) = 1 in the w equation (- kappa du/dy)
    mass = sparse.diags(np.concatenate([np.ones(count), inertia * np.ones(count)]), format="csc")
    state = np.zeros(2 * count)
    split = 2.0 - math.sqrt(2.0)
    time_value, step = 0.0, 1.0e-12
    times, stresses, profiles = [], [], {}
    pending = sorted((layer_times or {}).items())

    def wall_stress(vector):
        u0, u1, u2 = 1.0, vector[0], vector[1]
        h1, h2 = y[1] - y[0], y[2] - y[0]
        slope = (u1 - u0) * h2 / (h1 * (h2 - h1)) - (u2 - u0) * h1 / (h2 * (h2 - h1))
        return (mu + kappa) * slope                            # negative: the fluid drags the wall back

    while time_value < end_time:
        step_now = min(step, end_time - time_value)
        # land on a requested profile time exactly
        if pending and time_value + step_now > pending[0][0]:
            step_now = pending[0][0] - time_value
        trapezoid = (mass - 0.5 * split * step_now * a).tocsc()
        stage = splu(trapezoid).solve((mass + 0.5 * split * step_now * a) @ state + split * step_now * b)
        factor = (1.0 - split) / (2.0 - split)
        bdf = (mass - factor * step_now * a).tocsc()
        right = mass @ (stage / (split * (2.0 - split)) - (1.0 - split) ** 2 / (split * (2.0 - split)) * state)
        state = splu(bdf).solve(right + factor * step_now * b)
        time_value += step_now
        times.append(time_value)
        stresses.append(-wall_stress(state))
        if pending and abs(time_value - pending[0][0]) < 1e-15 * max(1.0, time_value):
            label = pending.pop(0)[1]
            profiles[label] = {"time": time_value, "y": y.tolist(),
                               "u": [1.0] + state[:count].tolist() + [0.0],
                               "w": [0.0] + state[count:].tolist() + [0.0]}
        step *= growth
    return {"y_nodes": int(y.size), "first_spacing": float(y[1] - y[0]), "times": np.asarray(times),
            "stress": np.asarray(stresses), "profiles": profiles}


def steady_excess(theta_independent: bool = True) -> float:
    c = material(1.0)
    mu, kappa = c["mu"], c["kappa"]
    displacement = kappa / (2.0 * (mu + kappa))                    # delta* / l
    return GAP_OVER_LENGTH / (GAP_OVER_LENGTH - 2.0 * displacement * math.tanh(0.5 * GAP_OVER_LENGTH)) - 1.0


def command_solve(arguments) -> int:
    end_time = arguments.end_time
    resolutions = {"coarse": (2.0e-5, 1.02, 0.02, 1.004), "fine": (1.0e-5, 1.01, 0.01, 1.002)}
    newtonian = {}
    for label, (first, ratio, uniform, growth) in resolutions.items():
        newtonian[label] = solve(None, first, ratio, uniform, growth, end_time)
    # the times at which the Newtonian layer has Lambda / l = 0.25, 1, 4 (fine resolution, interpolated in log)
    reference = newtonian["fine"]
    layer = 1.0 / reference["stress"]                                # Lambda = (mu + kappa / 2) U / tau_N
    layer_times = {}
    for value in LAYER_TIMES:
        index = int(np.searchsorted(layer, value))
        t0, t1 = reference["times"][index - 1], reference["times"][index]
        l0, l1 = layer[index - 1], layer[index]
        layer_times[float(np.exp(np.interp(np.log(value), [np.log(l0), np.log(l1)], [np.log(t0), np.log(t1)])))] = (
            f"{value:g}")
    for label, (first, ratio, uniform, growth) in resolutions.items():
        newtonian[label] = solve(None, first, ratio, uniform, growth, end_time, layer_times)
    curves = {"newtonian": {}}
    grid_layer = np.geomspace(0.01, 1.0 / (1.0 / GAP_OVER_LENGTH), 400)       # Lambda / l up to the steady H
    result = {"coupling_ratio": COUPLING_RATIO, "gap_over_length": GAP_OVER_LENGTH, "end_time": end_time,
              "early_limit": math.sqrt((1.0 + COUPLING_RATIO) / (1.0 + 0.5 * COUPLING_RATIO)) - 1.0,
              "steady_excess_eq_5_5": steady_excess(), "layer_times": {value: time for time, value in layer_times.items()},
              "resolutions": {label: dict(zip(("first_spacing", "ratio", "uniform_spacing", "time_step_growth"), values))
                              for label, values in resolutions.items()},
              "curves": {}, "profiles": {}, "convergence": {}}
    for label in resolutions:
        result["profiles"].setdefault("newtonian", {})[label] = newtonian[label]["profiles"]
    for theta in THETAS:
        key = f"theta_{theta:g}"
        runs = {}
        for label, (first, ratio, uniform, growth) in resolutions.items():
            runs[label] = solve(theta, first, ratio, uniform, growth, end_time, layer_times)
        series = {}
        for label in resolutions:
            base = newtonian[label]
            stress_newtonian = np.interp(runs[label]["times"], base["times"], base["stress"])
            excess = runs[label]["stress"] / stress_newtonian - 1.0
            layer_of = 1.0 / stress_newtonian
            series[label] = (layer_of, excess, runs[label]["times"], runs[label]["stress"], stress_newtonian)
        fine_layer, fine_excess = series["fine"][0], series["fine"][1]
        coarse_on_fine = np.interp(np.log(fine_layer), np.log(series["coarse"][0]), series["coarse"][1])
        chosen = (fine_layer >= 0.01) & (fine_layer <= 0.99 * fine_layer.max())
        result["convergence"][key] = {"max_abs_difference_of_excess": float(np.abs(coarse_on_fine - fine_excess)[chosen].max()),
                                      "nodes": {label: runs[label]["y_nodes"] for label in resolutions}}
        order = np.argsort(fine_layer)
        result["curves"][key] = {
            "layer_over_length": grid_layer.tolist(),
            "excess": np.interp(np.log(grid_layer), np.log(fine_layer[order]), fine_excess[order]).tolist(),
            "times": series["fine"][2].tolist()[::10], "stress": series["fine"][3].tolist()[::10],
            "stress_newtonian": series["fine"][4].tolist()[::10]}
        result["profiles"][key] = {label: runs[label]["profiles"] for label in resolutions}
        early = float(np.interp(np.log(0.01), np.log(fine_layer[order]), fine_excess[order]))
        late = float(fine_excess[-1])
        result["curves"][key].update({"excess_at_layer_0p01": early, "excess_at_end": late})
        print(f"[startup] theta {theta:g}: E(Lambda = 0.01 l) = {early:.5f} (sqrt(1.6) - 1 = "
              f"{result['early_limit']:.5f}), E(end) = {late:.6f} (eq. 5.5: {result['steady_excess_eq_5_5']:.6f}); "
              f"coarse - fine max |dE| = {result['convergence'][key]['max_abs_difference_of_excess']:.2e}", flush=True)
    base = newtonian["fine"]
    # the Rayleigh layer tau = 1 / sqrt(pi t) while Lambda is far above the first spacing and far below H
    window = (base["times"] > 3.0e-6) & (base["times"] < 1.0e-2)
    result["curves"]["newtonian"] = {"times": base["times"].tolist()[::10], "stress": base["stress"].tolist()[::10],
                                     "rayleigh_check": float(np.max(np.abs(
                                         base["stress"][window] * np.sqrt(math.pi * base["times"][window]) - 1.0))),
                                     "rayleigh_check_window": [3.0e-6, 1.0e-2]}
    print(f"[startup] Newtonian: max |tau sqrt(pi t) - 1| for 3e-6 < t < 0.01 (Rayleigh): "
          f"{result['curves']['newtonian']['rayleigh_check']:.2e}; steady tau H = {base['stress'][-1] * GAP_OVER_LENGTH:.6f}")
    out_dir = _REPO_ROOT / DATA_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "startup_host.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve_parser = commands.add_parser("solve")
    solve_parser.add_argument("--end-time", type=float, default=1200.0)
    arguments = parser.parse_args()
    return command_solve(arguments)


if __name__ == "__main__":
    sys.exit(main())
