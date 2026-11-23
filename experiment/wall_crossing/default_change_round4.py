"""
default_change_round4.py - the checks of wall store round 4's default change (claude/wall_store_report.md section 10):
wall_boundary crossing's items (a) (2-D: z set to 0 after the reads) and (d) (near-wall force with M_i out of the pair
sums) on by default, against the commit before (262156d, a detached checkout given by --reference).

    .venv/Scripts/python.exe -m experiment.wall_crossing.default_change_round4 bits --reference <checkout> --device 1
    .venv/Scripts/python.exe -m experiment.wall_crossing.default_change_round4 physics --reference <checkout>

bits     the fused pass and force once on a frozen crossing state (frozen_evaluate.py --sort-voxel-lists --split: the
         passes as the step records them, bit-reproducible) by: this checkout's default, again, with both items taken
         back (V7_WALL_CROSSING_ZERO_Z=0 V7_WALL_CROSSING_FORCE_FACTORED=0: must be the reference's bits), each item
         alone taken back, and the reference checkout; every output compared bit for bit with the reference's.
         States: logs/wall_store/round4/frozen/state_{n500,c3d}.npz (deleted by the caller afterwards).
physics  the dynamic runs (logs/wall_store/round4/): 2-D 250^2 cavity to t = 10 (psi_min wall frame, the three
         extrema, t = 8-10, cavity_analysis.analyse_run), the tilted hydrostatic box (largest and rms speed over
         sqrt(g H) for 5 <= t <= 10, the mean pressure at the box centre), 3-D 1M cavity to t = 2.04 (kinetic energy and
         the three centre-line extrema at the last sample, box_run's profiles), each against the reference build's runs
         and the spread of repeated runs. Tables claude/wall_store/tables/round4_*.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

ROOT = REPOSITORY_ROOT / "logs" / "wall_store" / "round4"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
NEWLINE = chr(10)
FIELDS = ("density_rate", "acceleration", "shift", "correction_inverse", "density_gradient_kernel_sum")
STATES = (("n500", "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing/case.yaml"),
          ("c3d", "cases/cavity3d_1m_crossing/case.yaml"))
# --set ad (sections 10.2 / 10.4): (a), (d) and spec const 142, each taken back; run while the default was (a)+(d)
# with the wall pairs of S in the loop (lanes and (e) not yet defaults: pinned off here)
_PINNED = {"V7_WALL_CROSSING_NEAR_WALL_LANES": "1"}
BUILDS = {"default": {}, "default_again": {},
          "zero_z_off": {"V7_WALL_CROSSING_ZERO_Z": "0"},
          "factored_off": {"V7_WALL_CROSSING_FORCE_FACTORED": "0"},
          "both_off": {"V7_WALL_CROSSING_ZERO_Z": "0", "V7_WALL_CROSSING_FORCE_FACTORED": "0"},
          # section 10.5: spec const 142 (the fused pass's wall pairs of S in the wall segment's loop by default since
          # round 4; 262156d had them after the store)
          "after_store": {"V7_WALL_CROSSING_WALL_RATE_LOOP": "after_store"},
          "both_off_after_store": {"V7_WALL_CROSSING_ZERO_Z": "0", "V7_WALL_CROSSING_FORCE_FACTORED": "0",
                                   "V7_WALL_CROSSING_WALL_RATE_LOOP": "after_store"}}
BUILDS = {name: {**_PINNED, **settings} for name, settings in BUILDS.items()}
PAIRS = (("after_store", "default"), ("both_off_after_store", "both_off"))
# --set final (sections 10.6 / 10.7): the final default (2-D: (a), (d), S in the loop, k = 4 by the rule; 3-D: (d), S in
# the loop, k = 1 by the rule), the lanes taken back, the experimental (e) added (with and without lanes), and every
# round-4 item taken back (262156d's modules with S in the loop: bit-identical to 262156d)
FINAL_BUILDS = {"default": {}, "default_again": {},
                "lanes_off": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "1"},
                "kernel_range_on": {"V7_FLUID_SEGMENT_KERNEL_RANGE": "1"},
                "kernel_range_on_lanes_off": {"V7_FLUID_SEGMENT_KERNEL_RANGE": "1",
                                              "V7_WALL_CROSSING_NEAR_WALL_LANES": "1"},
                "all_off": {"V7_WALL_CROSSING_ZERO_Z": "0", "V7_WALL_CROSSING_FORCE_FACTORED": "0",
                            "V7_WALL_CROSSING_NEAR_WALL_LANES": "1"}}
FINAL_PAIRS = (("lanes_off", "default"), ("kernel_range_on", "default"), ("kernel_range_on_lanes_off", "lanes_off"))


def evaluate(checkout: pathlib.Path, case: str, state: pathlib.Path, out: pathlib.Path, device: int,
             environment_settings: dict) -> None:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    environment.update(environment_settings)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    command = [sys.executable, str(checkout / "experiment" / "wall_crossing" / "frozen_evaluate.py"),
               "--repo", str(checkout), "--case", case, "--state", str(state), "--device", str(device),
               "--out", str(out), "--sort-voxel-lists", "--split"]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{out.name}: exit {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def compare(first: dict, second: dict) -> dict:
    result = {}
    for field in FIELDS:
        a, b = np.asarray(first[field]), np.asarray(second[field])
        if a.dtype == np.float32:
            differs = (a.view(np.uint32) != b.view(np.uint32)).reshape(a.shape[0], -1)
        else:
            differs = (a != b).reshape(a.shape[0], -1)
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = float(np.sqrt(np.mean(b.astype(np.float64) ** 2))) or 1.0
        result[field] = {"rows": int(differs.any(axis=1).sum()), "of": int(a.shape[0]),
                         "largest_over_rms": float(difference.max() / scale)}
    return result


def bits(arguments) -> int:
    reference = pathlib.Path(arguments.reference).resolve()
    out = ROOT / "bits"
    out.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ["| 算例 | 版本（对 262156d） | " + " | ".join(FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
    for label, case in STATES:
        state = ROOT / "frozen" / f"state_{label}.npz"
        evaluate(reference, case, state, out / f"{label}_reference.npz", arguments.device, {})
        base = dict(np.load(out / f"{label}_reference.npz"))
        builds, pairs = (FINAL_BUILDS, FINAL_PAIRS) if arguments.set == "final" else (BUILDS, PAIRS)
        for build, settings in builds.items():
            evaluate(REPOSITORY_ROOT, case, state, out / f"{label}_{build}.npz", arguments.device, settings)
            result = compare(dict(np.load(out / f"{label}_{build}.npz")), base)
            summary.setdefault(label, {})[build] = result
            cells = [("逐位相同" if entry["rows"] == 0
                      else f"{entry['rows']:,} / {entry['of']:,} 行；最大差 {entry['largest_over_rms']:.1e} 均方根")
                     for entry in result.values()]
            lines.append(f"| {label} | {build} | " + " | ".join(cells) + " |")
            print(lines[-1], flush=True)
        # section 10.5: spec const 142's two values against each other directly (same items)
        for first, second in pairs:
            result = compare(dict(np.load(out / f"{label}_{first}.npz")), dict(np.load(out / f"{label}_{second}.npz")))
            summary[label][f"{first}_versus_{second}"] = result
            cells = [("逐位相同" if entry["rows"] == 0
                      else f"{entry['rows']:,} / {entry['of']:,} 行；最大差 {entry['largest_over_rms']:.1e} 均方根")
                     for entry in result.values()]
            lines.append(f"| {label} | {first} 对 {second}（直接比） | " + " | ".join(cells) + " |")
            print(lines[-1], flush=True)
    suffix = "_final" if arguments.set == "final" else "_ad"
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"round4_bits{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for path in out.glob("*.npz"):
        path.unlink()
    return 0


def cavity_entry(name: str, root: pathlib.Path) -> dict | None:
    from experiment.validation.cavity_analysis import analyse_run, load_run
    run = load_run(name, root=root)
    if run is None:
        return None
    frames = any((root / name / "snapshots").glob("t*.npz"))
    analysis = analyse_run(run, average_span=2.0, tolerance=1.0e-3, window=10.0, with_stream_function=frames)
    extrema = analysis["extrema"]
    # psi_min needs the frames, which are deleted after the analysis: it is kept in <run>/psi_min.json
    cache = root / name / "psi_min.json"
    if frames:
        psi_min = (analysis.get("stream") or {}).get("psi_min_wall")
        cache.write_text(json.dumps({"psi_min_wall": psi_min}), encoding="utf-8")
    else:
        psi_min = json.loads(cache.read_text(encoding="utf-8"))["psi_min_wall"] if cache.exists() else None
    return {"psi_min": psi_min, "u_min": extrema["u_min"]["value"],
            "v_max": extrema["v_max"]["value"], "v_min": extrema["v_min"]["value"],
            "window": analysis["window"], "samples": analysis["samples_in_window"]}


def hydrostatic_entry(directory: pathlib.Path) -> dict | None:
    if not (directory / "summary.json").exists():
        return None
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with open(directory / "series.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scale = summary["scales"]["velocity_sqrt_g_extent"]
    pressure_scale = summary["scales"].get("pressure_rho_g_extent") or summary["scales"].get("pressure")
    time = np.array([float(row["time"]) for row in rows])
    chosen = time >= 5.0 - 1e-9
    return {"largest": float(np.mean([float(row["speed_maximum"]) for row, keep in zip(rows, chosen) if keep]) / scale),
            "root_mean_square": float(np.mean([float(row["speed_root_mean_square"]) for row, keep in zip(rows, chosen)
                                               if keep]) / scale),
            "pressure_level": float(np.mean([float(row["pressure_level_at_centre"]) for row, keep in zip(rows, chosen)
                                             if keep])),
            "pressure_level_over_scale": summary["final"].get("pressure_level_over_scale"),
            "pressure_scale": pressure_scale}


def box_entry(directory: pathlib.Path) -> dict | None:
    if not (directory / "profiles.json").exists():
        return None
    profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
    last = profiles[-1]
    with open(directory / "series.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1]
    u = np.asarray(last["u_along_y"]["velocity"], dtype=float)
    v = np.asarray(last["v_along_x"]["velocity"], dtype=float)
    return {"time": float(final["time"]), "kinetic_energy": float(final["kinetic_energy"]),
            "profile_time": float(last.get("time", np.nan)), "u_min": float(np.nanmin(u)),
            "v_max": float(np.nanmax(v)), "v_min": float(np.nanmin(v)),
            "density_mean": float(final["density_mean"]), "u": u.tolist(), "v": v.tolist()}


def physics(arguments) -> int:
    lines = []
    result = {}
    # 2-D 250^2 cavity to t = 10: every repeat of each build (round 3's three default runs are 262156d's defaults)
    new_names = [f"n250_default_r{index}" for index in range(1, 6)]
    old_names = [f"n250_reference_r{index}" for index in range(1, 4)]
    # sections 10.6 / 10.7: the final 2-D default ((a), (d), (e), the wall pairs of S in the loop, k = 4 by the rule)
    final_names = [f"n250_final_r{index}" for index in range(1, 3)]
    cavity = {name: cavity_entry(name, ROOT / "cavity") for name in new_names + old_names + final_names}
    round3 = json.loads((REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "lanes_physics.json").read_text(
        encoding="utf-8"))["runs"]
    rows = []
    labelled = ([(f"新缺省 (a)+(d)，第 {index} 次", cavity[name]) for index, name in enumerate(new_names, 1)]
                + [(f"最终缺省 (a)+(d)+(e)、S 合进循环、k = 4，第 {index} 次", cavity[name])
                   for index, name in enumerate(final_names, 1)]
                + [(f"262156d（参照检出），本轮第 {index} 次", cavity[name]) for index, name in enumerate(old_names, 1)]
                + [(f"262156d 的缺省，第三轮第 {index} 次", round3[f"lanes_default_r{index}"]) for index in (1, 2, 3)])
    for text, entry in labelled:
        if entry is None:
            continue
        rows.append(f"| {text} | {entry['psi_min']:.6f} | {entry['u_min']:.5f} | {entry['v_max']:.5f} | "
                    f"{entry['v_min']:.5f} |")
    old = [cavity[name] for name in old_names] + [round3[f"lanes_default_r{index}"] for index in (1, 2, 3)]
    old = [entry for entry in old if entry]
    new = [cavity[name] for name in new_names if cavity[name]]
    keys = (("psi_min", 6), ("u_min", 5), ("v_max", 5), ("v_min", 5))
    for key, digits in keys:
        result.setdefault("cavity", {})[key] = {
            "new_mean": float(np.mean([entry[key] for entry in new])), "old_mean": float(np.mean([entry[key] for entry in old])),
            "old_range": float(np.ptp([entry[key] for entry in old])), "new_range": float(np.ptp([entry[key] for entry in new])),
            "new_values": [entry[key] for entry in new], "old_values": [entry[key] for entry in old]}
    summary = result["cavity"]
    rows.append("| 差：新缺省平均 − 262156d 平均 | " + " | ".join(
        f"{summary[key]['new_mean'] - summary[key]['old_mean']:+.{digits}f}" for key, digits in keys) + " |")
    final = [cavity[name] for name in final_names if cavity[name]]
    if final:
        for key, digits in keys:
            summary[key]["final_values"] = [entry[key] for entry in final]
        rows.append("| 差：最终缺省平均 − 262156d 平均 | " + " | ".join(
            f"{np.mean([entry[key] for entry in final]) - summary[key]['old_mean']:+.{digits}f}" for key, digits in keys)
                    + " |")
    rows.append(f"| 运行间散布（极差）：262156d {len(old)} 次 / 新缺省 {len(new)} 次 | " + " | ".join(
        f"{summary[key]['old_range']:.{digits}f} / {summary[key]['new_range']:.{digits}f}" for key, digits in keys) + " |")
    lines.append("| 2-D 250² 方腔，t = 10（8–10 平均） | ψ_min（wall 帧） | u_min | v_max | v_min |" + NEWLINE
                 + "|---|---|---|---|---|" + NEWLINE + NEWLINE.join(rows))
    (TABLES / "round4_cavity.md").write_text(lines[-1] + NEWLINE, encoding="utf-8", newline=NEWLINE)
    # tilted hydrostatic box
    hydro = {name: hydrostatic_entry(ROOT / "hydrostatic" / name) for name in ("g30_default", "g30_reference",
                                                                                "g30_final")}
    round3_hydro = json.loads((REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "hydrostatic_round3.json").read_text(
        encoding="utf-8"))
    rows = []
    for text, entry in (("新缺省 (a)+(d)", hydro["g30_default"]), ("262156d（参照检出），本轮", hydro["g30_reference"]),
                        ("最终缺省 (a)+(d)+(e)、S 合进循环、k = 4", hydro["g30_final"])):
        if entry:
            rows.append(f"| {text} | {entry['largest']:.5f} | {entry['root_mean_square']:.6f} | "
                        f"{entry['pressure_level']:+.1f} |")
    from experiment.wall_crossing.hydrostatic_round3 import GROUPS
    for build, text in (("store", "壁面存储 36 次（第 9 节）"), ("tag", "标签 36 次（第 9 节）")):
        statistics = round3_hydro["statistics"][build]
        # the 36 runs' pressure level at the centre, recomputed from their histories (section 9 had no such column)
        levels = [entry["pressure_level"] for entry in (hydrostatic_entry(path) for path in GROUPS[build]) if entry]
        rows.append(f"| {text}：中位数（四分位） | {statistics['largest']['median']:.5f}（{statistics['largest']['quartiles'][0]:.5f}–"
                    f"{statistics['largest']['quartiles'][1]:.5f}） | {statistics['root_mean_square']['median']:.6f}（"
                    f"{statistics['root_mean_square']['quartiles'][0]:.6f}–{statistics['root_mean_square']['quartiles'][1]:.6f}） | "
                    f"{np.median(levels):+.1f}（{np.percentile(levels, 25):+.1f} 到 {np.percentile(levels, 75):+.1f}；"
                    f"全距 {min(levels):+.1f} 到 {max(levels):+.1f}；{len(levels)} 次） |")
    result["hydrostatic"] = hydro
    (TABLES / "round4_hydrostatic.md").write_text(
        "| 倾斜 30° 静水盒（2-D 250²，t = 10） | 最大速度 / √(gH)（5 ≤ t ≤ 10 平均） | 均方根速度 / √(gH) | 盒中心的平均压力 Pa（5 ≤ t ≤ 10） |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    # 3-D 1M to t = 2: every repeat of each build
    new_names = [f"c3d_default_r{index}" for index in range(1, 4)]
    old_names = [f"c3d_reference_r{index}" for index in range(1, 4)]
    boxes = {name: box_entry(ROOT / "box_run" / name) for name in new_names + old_names}
    present = {name: entry for name, entry in boxes.items() if entry}
    new_names = [name for name in new_names if name in present]
    old_names = [name for name in old_names if name in present]
    rows = []
    for names, text in ((new_names, "新缺省 (a)+(d)"), (old_names, "262156d")):
        for index, name in enumerate(names, 1):
            entry = present[name]
            rows.append(f"| {text}，第 {index} 次 | {entry['time']:.2f} | {entry['kinetic_energy']:.5f} | {entry['u_min']:.5f} | "
                        f"{entry['v_max']:.5f} | {entry['v_min']:.5f} | {entry['density_mean'] - 1000.0:+.5f} |")
    keys = (("kinetic_energy", 5), ("u_min", 5), ("v_max", 5), ("v_min", 5), ("density_mean", 5))

    def values(names, key):
        return [present[name][key] for name in names]
    rows.append("| 差：新缺省平均 − 262156d 平均 | | " + " | ".join(
        f"{np.mean(values(new_names, key)) - np.mean(values(old_names, key)):+.{digits}f}" for key, digits in keys) + " |")
    rows.append(f"| 运行间散布（极差）：262156d {len(old_names)} 次 / 新缺省 {len(new_names)} 次 | | " + " | ".join(
        f"{np.ptp(values(old_names, key)):.{digits}f} / {np.ptp(values(new_names, key)):.{digits}f}" for key, digits in keys)
                + " |")
    profile = {}
    for line in ("u", "v"):
        new_mean = np.mean([np.asarray(present[name][line]) for name in new_names], axis=0)
        old_mean = np.mean([np.asarray(present[name][line]) for name in old_names], axis=0)
        same_build = [float(np.sqrt(np.nanmean((np.asarray(present[first][line]) - np.asarray(present[second][line])) ** 2)))
                      for names in (new_names, old_names) for position, first in enumerate(names)
                      for second in names[position + 1:]]
        profile[line] = {"between_builds_l2": float(np.sqrt(np.nanmean((new_mean - old_mean) ** 2))),
                         "same_build_l2_largest": max(same_build), "same_build_l2_median": float(np.median(same_build))}
    result["c3d_profile"] = profile
    result["c3d"] = {name: {key: value for key, value in entry.items() if key not in ("u", "v")}
                     for name, entry in present.items()}
    (TABLES / "round4_cavity3d.md").write_text(
        "| 3-D 1M 方腔（总体积修正打开），单卡 | 时刻 | 动能（J） | u_min（竖直中心线） | v_max（水平中心线） | v_min | 平均密度 − ρ0 |"
        + NEWLINE + "|---|---|---|---|---|---|---|" + NEWLINE + NEWLINE.join(rows) + NEWLINE + NEWLINE
        + f"两个构建平均剖面之差 L2：u(y) {profile['u']['between_builds_l2']:.5f}、v(x) {profile['v']['between_builds_l2']:.5f}；"
          f"同一构建两次之差 L2（两个构建各自两两比，中位数 / 最大）：u(y) {profile['u']['same_build_l2_median']:.5f} / "
          f"{profile['u']['same_build_l2_largest']:.5f}、v(x) {profile['v']['same_build_l2_median']:.5f} / "
          f"{profile['v']['same_build_l2_largest']:.5f}（盖速为单位）。" + NEWLINE,
        encoding="utf-8", newline=NEWLINE)
    (ROOT / "physics.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    for name in ("round4_cavity.md", "round4_hydrostatic.md", "round4_cavity3d.md"):
        print((TABLES / name).read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    bits_parser = sub.add_parser("bits")
    bits_parser.add_argument("--reference", required=True)
    bits_parser.add_argument("--device", type=int, default=1)
    bits_parser.add_argument("--set", choices=("ad", "final"), default="final")
    physics_parser = sub.add_parser("physics")
    physics_parser.add_argument("--reference", required=False)
    arguments = parser.parse_args()
    return bits(arguments) if arguments.action == "bits" else physics(arguments)


if __name__ == "__main__":
    sys.exit(main())
