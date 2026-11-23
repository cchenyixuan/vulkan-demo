"""
lanes_physics.py - section 9.4 of claude/wall_store_report.md: the 2-D 250^2 cavity to t = 10 (t 8-10 averaged, the
round-1 check of section 3.2) with the default build three times (the run-to-run spread: the voxel lists are filled
in atomic order) and with k = 2, 4, 8 invocations per near-wall particle (V7_WALL_CROSSING_NEAR_WALL_LANES, cavity_run
--extra-switch). psi_min (wall frame), the three velocity extrema and the centre-line L2 against Marchi 2021.

    .venv/Scripts/python.exe -m experiment.wall_crossing.lanes_physics [--delete-frames]

Runs: logs/wall_store/round3/cavity/lanes_{default_r1,default_r2,default_r3,2,4,8}. Writes
claude/wall_store/tables/round3_lanes_physics.md and logs/wall_store/round3/lanes_physics.json; --delete-frames then
removes the runs' particle snapshots and checkpoints (the samples and histories stay).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

ROOT = REPOSITORY_ROOT / "logs" / "wall_store" / "round3"
RUNS = (("lanes_default_r1", "缺省（k = 1），第 1 次"), ("lanes_default_r2", "缺省（k = 1），第 2 次"),
        ("lanes_default_r3", "缺省（k = 1），第 3 次"), ("lanes_2", "k = 2"), ("lanes_4", "k = 4"), ("lanes_8", "k = 8"))
KEYS = ("psi_min", "u_min", "v_max", "v_min", "l2_u", "l2_v")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete-frames", action="store_true")
    arguments = parser.parse_args()
    from experiment.validation.cavity_analysis import analyse_run, load_run
    entries = {}
    for name, _ in RUNS:
        run = load_run(name, root=ROOT / "cavity")
        analysis = analyse_run(run, average_span=2.0, tolerance=1.0e-3, window=10.0, with_stream_function=True)
        extrema, errors = analysis["extrema"], analysis["errors"]
        entries[name] = {"psi_min": (analysis["stream"] or {}).get("psi_min_wall"),
                         "u_min": extrema["u_min"]["value"], "v_max": extrema["v_max"]["value"],
                         "v_min": extrema["v_min"]["value"], "l2_u": errors[("wall", "mls", "u")]["L2"],
                         "l2_v": errors[("wall", "mls", "v")]["L2"], "window": analysis["window"],
                         "samples": analysis["samples_in_window"],
                         "u_dense": analysis["statistics"]["u_dense_mls"]["mean"].tolist(),
                         "v_dense": analysis["statistics"]["v_dense_mls"]["mean"].tolist()}
    defaults = [entries[name] for name, _ in RUNS[:3]]
    mean = {key: float(np.mean([entry[key] for entry in defaults])) for key in KEYS}
    spread = {key: float(np.max([entry[key] for entry in defaults]) - np.min([entry[key] for entry in defaults]))
              for key in KEYS}
    rows = []
    for name, text in RUNS:
        entry = entries[name]
        u_difference = np.asarray(entry["u_dense"]) - np.mean([np.asarray(d["u_dense"]) for d in defaults], axis=0)
        v_difference = np.asarray(entry["v_dense"]) - np.mean([np.asarray(d["v_dense"]) for d in defaults], axis=0)
        entry["profile_difference_to_default_mean"] = {"u_l2": float(np.sqrt(np.mean(u_difference ** 2))),
                                                       "v_l2": float(np.sqrt(np.mean(v_difference ** 2)))}
        rows.append(f"| {text} | {entry['psi_min']:.6f}（{100 * (entry['psi_min'] / mean['psi_min'] - 1):+.3f} %） | "
                    f"{entry['u_min']:.5f}（{entry['u_min'] - mean['u_min']:+.5f}） | "
                    f"{entry['v_max']:.5f}（{entry['v_max'] - mean['v_max']:+.5f}） | "
                    f"{entry['v_min']:.5f}（{entry['v_min'] - mean['v_min']:+.5f}） | {entry['l2_u']:.5f} | "
                    f"{entry['l2_v']:.5f} | {entry['profile_difference_to_default_mean']['u_l2']:.5f} / "
                    f"{entry['profile_difference_to_default_mean']['v_l2']:.5f} | "
                    f"{entry['window'][0]:.1f}–{entry['window'][1]:.1f}（{entry['samples']}） |")
    rows.append(f"| 缺省三次的极差 | {spread['psi_min']:.6f}（{100 * spread['psi_min'] / abs(mean['psi_min']):.3f} %） | "
                f"{spread['u_min']:.5f} | {spread['v_max']:.5f} | {spread['v_min']:.5f} | {spread['l2_u']:.5f} | "
                f"{spread['l2_v']:.5f} |  |  |")
    header = ("| 2-D 250² 方腔，t = 10（8–10 平均） | ψ_min（wall 帧；括号：对缺省三次平均） | u_min | v_max | v_min | "
              "u(y) L2 对 Marchi | v(x) L2 对 Marchi | 密中心线剖面对缺省平均之差 L2：u / v | 平均窗口（样本数） |\n"
              "|---|---|---|---|---|---|---|---|---|")
    table = header + "\n" + "\n".join(rows) + "\n"
    (REPOSITORY_ROOT / "claude" / "wall_store" / "tables" / "round3_lanes_physics.md").write_text(
        table, encoding="utf-8", newline="\n")
    for entry in entries.values():
        entry.pop("u_dense"), entry.pop("v_dense")
    (ROOT / "lanes_physics.json").write_text(json.dumps({"runs": entries, "default_mean": mean,
                                                         "default_spread": spread}, indent=1), encoding="utf-8")
    print(table)
    if arguments.delete_frames:
        for name, _ in RUNS:
            for folder in ("snapshots", "checkpoints"):
                path = ROOT / "cavity" / name / folder
                if path.exists():
                    shutil.rmtree(path)
        print("[lanes_physics] deleted the runs' snapshots and checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
