"""
wall_option_reproduction.py — E37: the release's wall_boundary adami against E36's adami_rho0 (v7-wall-bc WALL_BC = 3)
at the same resolution, CPU only.

Reads the e36_summary.json that E36's comparison tool writes for two cavity_runner runs (E36's run first, the release
run second) and prints psi_min and the three centre-line extrema as deviations from Marchi 2021 (%), their difference,
and the noise scale of the difference: E36 section 8.2's per-run scale (e36_factorial.deviations: psi_min = half the
difference of the two half-window minima, extrema = the sem) of the two runs in quadrature. The comparison tool lives
on branch v7-wall-bc only (experiment/v7/wall_bc/e36_compare.py and e36_factorial.py, 9c7c847); pass that checkout.

    # in a v7-wall-bc checkout (no .venv there: use this checkout's interpreter); --run is "label::run directory";
    # give OUT and the run directories as absolute paths (a relative OUT lands in the v7 checkout); the labels may be
    # non-ASCII, so set PYTHONIOENCODING=utf-8 when redirecting the output
    <this checkout>/.venv/Scripts/python.exe -m experiment.v7.wall_bc.e36_compare --out <absolute OUT> \\
        --run "E36 adami_rho0::<abs>/logs/e36/runs/n250_k1_v7_diag3_rho0" --run "E37 release adami::<abs release run dir>"
    # here:
    .venv/Scripts/python.exe experiment/seam_audit/wall_option_reproduction.py <absolute OUT>/e36_summary.json \\
        --v7-checkout <path>
"""
from __future__ import annotations

import argparse
import json
import math
import sys

QUANTITIES = ("psi_min", "u_min", "v_max", "v_min")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="e36_summary.json of e36_compare (the E36 run first, the release run second)")
    parser.add_argument("--v7-checkout", required=True, help="a checkout of branch v7-wall-bc (for e36_factorial)")
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.v7_checkout)
    from experiment.v7.wall_bc.e36_factorial import deviations          # noqa: E402 (the v7 checkout)

    with open(arguments.summary, encoding="utf-8") as handle:
        runs = json.load(handle)["runs"]
    if len(runs) != 2:
        sys.exit(f"expected two runs in {arguments.summary}, got {len(runs)}")
    (first_value, first_noise), (second_value, second_noise) = deviations(runs[0]), deviations(runs[1])
    for label, run in zip(("first ", "second"), runs):
        print(f"{label}: {run['label']} ({run['run']}), window {run['window'][0]:.2f}-{run['window'][1]:.2f}, "
              f"{run['samples_in_window']} samples, steady at t = {run.get('t_steady_online')}")
    print(f"| quantity | {runs[0]['label']} | {runs[1]['label']} | difference (pp) | noise (pp) |")
    print("|---|---|---|---|---|")
    for name in QUANTITIES:
        noise = math.hypot(first_noise[name], second_noise[name])
        difference = second_value[name] - first_value[name]
        print(f"| {name} | {first_value[name]:+.3f} % | {second_value[name]:+.3f} % | {difference:+.3f} | {noise:.3f} "
              f"({abs(difference) / noise:.1f} sigma) |")
    for run in runs:
        errors = run["errors"]
        print(f"{run['label']}: relative L2 u {100 * errors['u']['L2_relative']:.2f} %, "
              f"v {100 * errors['v']['L2_relative']:.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
