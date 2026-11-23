"""
stress_plans.py - the run plans of the round-6 stress test (claude/wall_store_report.md section 12) for
experiment/wall_crossing/stress.py queue.

    python -m experiment.wall_crossing.stress_plans --out <directory> [--seed 6]

Writes into <directory>:
  local_L1.json   3-D 1M cavity, both cards (K = 2), from t = 0, ended by a 3 h wall-clock deadline
  local_L2.json   2-D 1000^2 cavity, both cards, from t = 0, ended after 2 h
  local_L3_device{0,1}.json   mixed short runs, one card each (shuffled rounds of every combination)
  local_L4_slot{1..4}_device{0,1}_{a,b}.json   two runs at once on each card, four slots of 15 min
  remote.json     R2 (2-D 500^2 in segments of one checkpoint interval, resumed each time, to the end of the
                  section-11 run), R1 (3-D 1M to t = 90), R3 (mixed short 3-D runs, cut by the queue's deadline)

Mixed short runs: every combination of case (2-D cavity 250^2 and 500^2, 3-D cavity 50 and 70 spacings per side,
2-D hydrostatic box with tilted gravity, 3-D 48^3 hydrostatic box with tilted gravity), volume constraint (on / off,
cavities only; off = cases/stress6_<case>_volume_off from stress.py make-cases) and near-wall lanes k (the rule /
V7_WALL_CROSSING_NEAR_WALL_LANES=1 / =4); each run takes a random number of steps, a multiple of the defrag cadence
in [2,000, 20,000]. The runs go in rounds: each round is every combination once, in a fresh random order.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LOCAL_UUIDS = ["fb83f0dafd9e1d38dc05e7ae643e22a8", "ae137c668a40f5acaab90a83f7cda175"]   # v7 devices 0 and 1
REMOTE_UUID = "a85cce699d0fb7294c33d0a5981a4666"
BOX_RUN = "experiment.wall_crossing.box_run"
CASES = {
    "c250": ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml", True),
    "c500": ("cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing/case.yaml", True),
    "c3d50": ("cases/cavity3d_n50_crossing/case.yaml", True),
    "c3d70": ("cases/cavity3d_n70_crossing/case.yaml", True),
    "h2d": ("cases/wall_crossing/hydrostatic_2d_n250_g30_crossing/case.yaml", False),
    "h3d": ("cases/wall_crossing/hydrostatic_3d_n48_gtilt_crossing/case.yaml", False),
}
LANES = {"kauto": {}, "k1": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "1"}, "k4": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "4"}}


def volume_off(case: str) -> str:
    base = pathlib.PurePosixPath(case).parent.name
    return f"cases/stress6_{base}_volume_off/case.yaml"


def timestep(case: str, cache: dict) -> float:
    if case not in cache:
        from experiment.v7.utils.case_loader_v7 import load_case_v7
        cache[case] = float(load_case_v7(REPOSITORY_ROOT / case).physics.timestep)
    return cache[case]


def box_job(label: str, case: str, device: str, uuids: list, time_value: float, cavity: bool, env: dict,
            combo: str | None = None, extra: list | None = None, **fields) -> dict:
    args = ["--case", case, "--name", "{name}", "--device", device]
    for uuid in uuids:
        args += ["--uuid", uuid]
    args += ["--time", repr(time_value)] + (extra or ["--sample-time", "0.1"])
    args += (["--stop-speed", "1.5"] if cavity else []) + ["--stop-outside"]
    return {"label": label, "module": BOX_RUN, "args": args, "env": env, "device_uuids": uuids, "combo": combo,
            **fields}


def combinations(names: list) -> list:
    combos = []
    for name in names:
        for volume in (("von", "voff") if CASES[name][1] else ("von",)):
            for lanes in LANES:
                combos.append((name, volume, lanes))
    return combos


def mixed_jobs(names: list, device: str, uuids: list, rounds: int, generator: random.Random, cache: dict) -> list:
    jobs = []
    combos = combinations(names)
    for _ in range(rounds):
        order = combos[:]
        generator.shuffle(order)
        for name, volume, lanes in order:
            case, cavity = CASES[name]
            if volume == "voff":
                case = volume_off(case)
            steps = 1000 * generator.randint(2, 20)
            dt = timestep(case, cache)
            combo = f"{name}_{volume}_{lanes}"
            jobs.append(box_job(f"{combo}_s{steps // 1000}k", case, device, uuids, (steps - 0.5) * dt, cavity,
                                LANES[lanes], combo=combo, budget_seconds=1800, steps=steps))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=6)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    generator = random.Random(arguments.seed)

    def write(name: str, jobs: list) -> None:
        (out / name).write_text(json.dumps({"jobs": jobs}, indent=1), encoding="utf-8", newline="\n")
        print(f"{name}: {len(jobs)} job(s)")

    write("local_L1.json", [box_job("L1_cavity3d_1m_dual", "cases/cavity3d_1m_crossing/case.yaml", "0,1",
                                    LOCAL_UUIDS, 1000.0, True, {}, deadline_seconds=3 * 3600,
                                    budget_seconds=3 * 3600 + 1800)])
    write("local_L2.json", [box_job("L2_n1000_dual", "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing/"
                                    "case.yaml", "0,1", LOCAL_UUIDS, 1000.0, True, {},
                                    extra=["--sample-time", "0.2"], deadline_seconds=2 * 3600,
                                    budget_seconds=2 * 3600 + 1800)])
    every_case = ["c250", "c500", "c3d50", "c3d70", "h2d", "h3d"]
    for device in (0, 1):
        write(f"local_L3_device{device}.json",
              mixed_jobs(every_case, str(device), [LOCAL_UUIDS[device]], 30, generator, cache))
    # L4: two runs at once on each card; (a, b) per slot, each repeated until the slot ends
    slots = {1: (("c250", "k1", 200000), ("c3d50", "kauto", 40000)),
             2: (("c500", "kauto", 60000), ("h3d", "kauto", 40000)),
             3: (("h2d", "kauto", 200000), ("c3d70", "kauto", 20000)),
             4: (("c250", "kauto", 200000), ("c3d50", "kauto", 40000))}
    for slot, pair in slots.items():
        for device in (0, 1):
            for part, (name, lanes, steps) in zip("ab", pair):
                case, cavity = CASES[name]
                combo = f"{name}_von_{lanes}"
                job = box_job(f"L4s{slot}{part}_{combo}", case, str(device), [LOCAL_UUIDS[device]],
                              (steps - 0.5) * timestep(case, cache), cavity, LANES[lanes], combo=combo,
                              budget_seconds=2400, steps=steps)
                write(f"local_L4_slot{slot}_device{device}_{part}.json", [job] * 60)
    # remote: R2, R1, R3
    remote = []
    n500 = "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing/case.yaml"
    checkpoint_steps = 663000           # the section-11 run's checkpoint interval (meta.json: 51 samples of 13,000)
    stop_steps = 6667000                # its stop step (t_end 100)
    cavity_args = ["--solver", "v7", "--expect", "release", "--slabs", "1", "--device-map", "0", "--require-uuid",
                   REMOTE_UUID, "--case", n500, "--run-dir", "{run_dir}", "--t-end", "100.0", "--t-max", "120.0",
                   "--average-span", "20.0"]
    segments = (stop_steps + checkpoint_steps - 1) // checkpoint_steps
    for segment in range(segments):
        last = segment == segments - 1
        remote.append({"label": f"R2_n500_segment{segment:02d}", "module": "experiment.wall_crossing.cavity_runner_entry",
                       "run_dir": "{out}/cavity/r6_r2_n500", "env_preset": "cavity_release",
                       "args": cavity_args + (["--resume"] if segment else [])
                       + ([] if last else ["--max-steps", str(checkpoint_steps)]),
                       "device_uuids": [REMOTE_UUID], "budget_seconds": 3600, "segment": segment})
    remote.append(box_job("R1_cavity3d_1m", "cases/cavity3d_1m_crossing/case.yaml", "0", [REMOTE_UUID], 90.0, True,
                          {}, extra=["--sample-time", "0.1", "--profile-time", "0.5", "--checkpoint-time", "5"],
                          budget_seconds=7 * 3600, keep_npz=True))
    remote += mixed_jobs(["c3d50", "c3d70", "h3d"], "0", [REMOTE_UUID], 40, generator, cache)
    write("remote.json", remote)
    print("timesteps:", json.dumps(cache, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
