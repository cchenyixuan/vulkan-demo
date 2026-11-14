"""
make_all.py - run the item scripts of the pack and assemble captions.md and the index of README.md.

  python data/micropolar/ch5_draft/make_all.py            every item whose script exists
  python data/micropolar/ch5_draft/make_all.py 01 03      the named items only (the index is rebuilt from all)
  python data/micropolar/ch5_draft/make_all.py --index    captions.md and README.md only
"""
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

ITEMS = [
    ("01", "fig_01_concentric_profiles", "Concentric: v_theta(r) and w(r) against the exact solution, K = 3, m = 2 "
                                         "and m = 10, insets of both wall layers"),
    ("02", "fig_02_concentric_convergence", "Concentric: deviation of D against l / dx for both series, profile "
                                            "errors of v and w"),
    ("03", "fig_03_concentric_sweep", "Concentric: D against m for the three K, slip-form rule, ratios to exact"),
    ("04", "tab_04_concentric_rows", "Table of every concentric row"),
    ("05", "tab_05_eccentric_newtonian", "Table: Newtonian flow between eccentric cylinders against the exact "
                                         "Stokes values"),
    ("06", "fig_06_eccentric_wall_pressure", "Eccentric: wall pressure around the rotor against the Stokes "
                                             "solution, with the micropolar runs"),
    ("07", "fig_07_eccentric_fields", "Eccentric fields, eps = 0.6, K = 3, l = 10 dx: pressure, speed, "
                                      "micro-rotation, w - curl v / 2"),
    ("08", "fig_08_eccentric_prediction", "Eccentric: measured over predicted change against narrowest gap / l"),
    ("09", "tab_09_eccentric_rows", "Table of every eccentric micropolar row"),
    ("10", "fig_10_film_function", "The film function over h^3 against h / l at K = 3"),
    ("11", "fig_11_micropolar_cavity", "Micropolar cavity: centre lines against the finite-difference solution, "
                                       "fields, the slab cut"),
    ("12", "fig_12_cavity_baseline", "Newtonian cavity baseline at n = 1000 against Marchi et al."),
    ("13", "fig_13_seepage", "Seepage: velocity into the wall against wall pressure; candidates table"),
    ("14", "tab_14_solver_checks", "Tables of Sections 8 to 12: operator checks, passive label, K = 2 against "
                                   "K = 1, step limits, cost"),
    ("15", "fig_15_three_dimensional", "3-D: fields in a plane through the axis and the mid-plane profiles"),
    ("16", "fig_16_effect_map", "The map of the effect: contours of D in the plane (N, m) for the torque and the "
                                "load"),
    ("20", "fig_20_dipole_showcase", "Dipole-wall collision: vorticity of the control and of the two micropolar runs "
                                     "at four times"),
    ("21", "fig_21_dipole_tracks", "Dipole-wall collision: core trajectories and enstrophy against the benchmark"),
    ("22", "tab_22_dipole_numbers", "Dipole-wall collision: numbers against the control with the noise floor"),
    ("26", "tab_26_dipole_checks", "Dipole-wall collision: the checks of micro-inertia, saturation and resolution"),
]


def load_items() -> dict:
    found = {}
    for path in sorted((HERE / "items").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        found[entry["item"]] = entry
    return found


def write_index() -> None:
    found = load_items()
    captions = ["# Captions of the chapter-draft pack", "",
                "Written by the item scripts (`style.record`); regenerate with `make_all.py`. Numbers are quoted "
                "with the digits the noise floor of each quantity supports.", ""]
    rows = ["| no. | item | files | status | data |", "|---|---|---|---|---|"]
    for number, script, title in ITEMS:
        entries = [entry for key, entry in sorted(found.items()) if key == number or key.startswith(number)
                   and not key[len(number):len(number) + 1].isdigit()]
        if not entries:
            rows.append(f"| {number} | {title} | - | not reached | - |")
            continue
        for entry in entries:
            files = ", ".join(f"`{name}`" for name in entry["files"])
            rows.append(f"| {entry['item']} | {entry['title']} | {files} | {entry['status']} | {entry['tag']} |")
            captions += [f"## {entry['item']}. {entry['title']}", "", f"Status: {entry['status']}. Files: {files}.", "",
                         entry["caption"], ""]
    (HERE / "captions.md").write_text("\n".join(captions), encoding="utf-8", newline="\n")
    readme = (HERE / "README.head.md").read_text(encoding="utf-8").rstrip("\n")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    (HERE / "README.md").write_text(readme + f"\n\n## Index (written {stamp})\n\n" + "\n".join(rows) + "\n",
                                    encoding="utf-8", newline="\n")
    print(f"[pack] captions.md, README.md: {len(found)} items recorded")


def main() -> int:
    arguments = sys.argv[1:]
    if "--index" not in arguments:
        chosen = [item for item in ITEMS if not arguments or item[0] in arguments]
        for number, script, title in chosen:
            path = HERE / f"{script}.py"
            if not path.exists():
                continue
            code = subprocess.run([sys.executable, str(path)], cwd=REPO_ROOT).returncode
            if code:
                print(f"[pack] item {number}: exit {code}")
    write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
