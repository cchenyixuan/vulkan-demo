"""
poison_g1.py - E23 poison test of V6_DIAG_POISON_G1 (docs/seam_audit/v6_opt.md, "G1 去 P").

  pressure   expand_ghost_lists writes NaN into the P of every inbound G1 (inner) replica.
             G1's P^n is read by nobody (C2 / C3 read rho only; C4 overwrites G1's rho and P
             with rho^{n+1}, P^{n+1} before C5), so every field of every pool slot must stay
             finite, with drift 0 and every overflow_* / far_migration counter 0.
  density    negative control: NaN into G1's rho (read by C2 and C3) -> NaN must appear,
             which shows that the test catches a field that is read. The run stops at the
             first horizon with a non-finite value (no point running on with NaN positions).

Every run: the release set of the dimension (ab_restart.SEAM_L2 + PRODUCTION + RELEASE[d],
V6_FAST_SUBMIT=1), K = 2, devices 0,1, from rest (bootstrap_all), the production depth-2
loop (dump_state.run_frames). At every horizon (0 = right after the bootstrap) a full-pool
readback of every per-particle float buffer of set 0 (+ material): non-finite rows per buffer x
pool region x live/dead; drift from the alive own slots (mass > 0 and voxel id > 0.5;
GlobalStatus.alive_particle_count is refreshed only by initialize_voxelization / defrag); every
overflow_* counter; far_migration_count; GPU + host frame-stamp errors; material index range of
the live rows. The last horizon's full-pool arrays are saved (the dump).

Usage:
  .venv/Scripts/python.exe -m experiment.seam_audit.poison_g1 --out logs/seam_audit/opt/poison_g1
  .venv/Scripts/python.exe -m experiment.seam_audit.poison_g1 --out ... --cases cavity2d_1m --modes pressure
  .venv/Scripts/python.exe -m experiment.seam_audit.poison_g1 --dry-run      (CPU only: partition + regions)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.seam_audit.ab_restart import PRODUCTION, RELEASE, SEAM_L2  # noqa: E402

# name: (case yaml, dimension, horizons incl. 0 = after the bootstrap)
CASES = {
    "cavity2d_1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, (0, 1, 2, 5, 10, 20, 50, 100, 200)),
    "cavity3d_1m": ("cases/cavity3d_1m/case.yaml", 3, (0, 1, 2, 5, 10, 20, 50)),
}
# mode: (V6_DIAG_POISON_G1 value, extra environment, NaN expected)
MODES = {
    "pressure": ("pressure", {}, False),
    "density": ("density", {}, True),
}
# Every per-particle float buffer of set 0 (components per pool slot); material is uint.
FLOAT_BUFFERS = {
    "position_voxel_id": 4, "velocity_mass": 4, "density_pressure": 2, "density_pressure_scratch": 2,
    "acceleration": 4, "shift": 4, "correction_inverse": 8, "density_gradient_kernel_sum": 4,
    "extension_fields": 4,
}
# Live prefix of each ghost / departed region (GlobalStatus keys, receiver side).
LIVE_COUNT_KEYS = {
    "leading_inner": "replica_inner_recv_leading_count", "leading_outer": "replica_outer_recv_leading_count",
    "leading_migrant": "ghost_recv_leading_count",
    "trailing_inner": "replica_inner_recv_trailing_count", "trailing_outer": "replica_outer_recv_trailing_count",
    "trailing_migrant": "ghost_recv_trailing_count",
    "departed": "departed_count",
}
DEVICES = (0, 1)
POOL_SAFETY = 1.2
DEPTH = 2
RESULT_PREFIX = "[poison_g1] RESULT "


def run_environment(dimension: int, mode: str) -> dict:
    value, extra, _expected = MODES[mode]
    environment = {key: item for key, item in os.environ.items() if not key.startswith("V6_")}
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
                        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    environment.update(SEAM_L2)
    environment.update(PRODUCTION)
    environment.update(RELEASE[dimension])
    environment["V6_FAST_SUBMIT"] = "1"
    environment["V6_DIAG_POISON_G1"] = value
    environment.update(extra)
    return environment


def pool_regions(capacities) -> dict:
    """[first, stop) pid ranges (case_v6.Capacities layout:
    0 | leading [1, L] | own [L+1, L+O] | trailing [L+O+1, L+O+T] | departed [.., +D]);
    a two-layer ghost pool is [inner R | outer R | migrants]."""
    leading = capacities.leading_ghost_pool_size
    own = capacities.own_pool_size
    trailing = capacities.trailing_ghost_pool_size
    departed = capacities.departed_pool_size
    replica = capacities.replica_region_size
    regions = {"sentinel": (0, 1)}
    if leading:
        regions.update({"leading_inner": (1, 1 + replica), "leading_outer": (1 + replica, 1 + 2 * replica),
                        "leading_migrant": (1 + 2 * replica, 1 + leading)})
    regions["own"] = (1 + leading, 1 + leading + own)
    if trailing:
        base = 1 + leading + own
        regions.update({"trailing_inner": (base, base + replica),
                        "trailing_outer": (base + replica, base + 2 * replica),
                        "trailing_migrant": (base + 2 * replica, base + trailing)})
    if departed:
        base = 1 + leading + own + trailing
        regions["departed"] = (base, base + departed)
    total = capacities.total_pool_capacity()
    assert max(stop for _first, stop in regions.values()) == total, (regions, total)
    return regions


def inspect_sim(sim, material_count: int, check_copy: bool = True) -> dict:
    """Full-pool readback of one sim after a drained frame. check_copy=False right after the
    bootstrap: its defrag copies the zeroed defrag twins over the whole primary buffers (ghost and
    departed rows become 0) but never touches density_pressure_scratch (binding 2)."""
    capacities = sim.case.capacities
    pool = capacities.total_pool_capacity()
    raw = sim.readback_buffers_batch(list(FLOAT_BUFFERS) + ["material"])
    status = dict(sim.readback_global_status())
    health = dict(sim.readback_pool_health())
    arrays = {name: np.frombuffer(raw[name], dtype=np.float32)[:pool * components].reshape(pool, components)
              for name, components in FLOAT_BUFFERS.items()}
    material = np.frombuffer(raw["material"], dtype=np.uint32)[:pool]
    regions = pool_regions(capacities)
    live = np.zeros(pool, dtype=bool)
    own_first, own_stop = regions["own"]
    alive_own = ((arrays["velocity_mass"][own_first:own_stop, 3] > 0)
                 & (arrays["position_voxel_id"][own_first:own_stop, 3] > 0.5))
    live[own_first:own_stop] = alive_own
    live_counts = {}
    for region, (first, stop) in regions.items():
        key = LIVE_COUNT_KEYS.get(region)
        if key is None:
            continue
        count = min(int(status.get(key, 0)), stop - first)
        live_counts[region] = count
        live[first:first + count] = True
    nonfinite = {}
    for name, values in arrays.items():
        bad = ~np.isfinite(values).all(axis=1)
        for region, (first, stop) in regions.items():
            region_bad = bad[first:stop]
            region_live = live[first:stop]
            counts = (int((region_bad & region_live).sum()), int((region_bad & ~region_live).sum()))
            if counts != (0, 0):
                nonfinite[f"{name}|{region}"] = {"live": counts[0], "dead": counts[1]}
    bad_material = int(((material >= material_count) & live).sum())
    # Evidence of the C4 overwrite (step boundary): every live G1 / departed row of primary
    # density_pressure equals scratch bit for bit (rho^{n+1}, P^{n+1}); every live G2 row has P = 0
    # (the E23 unpack writes 0 for both layers; C4 never touches G2).
    primary_bits = arrays["density_pressure"].view(np.uint32)
    scratch_bits = arrays["density_pressure_scratch"].view(np.uint32)
    copy_check = {}
    for region, count in (live_counts.items() if check_copy else ()):
        first = regions[region][0]
        rows = slice(first, first + count)
        if region.endswith("_inner") or region == "departed":
            copy_check[region] = int((primary_bits[rows] != scratch_bits[rows]).any(axis=1).sum())
        elif region.endswith("_outer"):
            copy_check[region] = int((arrays["density_pressure"][rows, 1] != 0.0).sum())
    return {
        "alive_own": int(alive_own.sum()),
        "live_counts": live_counts,
        "nonfinite": nonfinite,
        "nonfinite_rows_total": int(sum(entry["live"] + entry["dead"] for entry in nonfinite.values())),
        "bad_material_live_rows": bad_material,
        "copy_check_mismatched_rows": copy_check,
        "overflow": {key: int(value) for key, value in status.items() if key.startswith("overflow_")},
        "far_migration_count": int(status.get("far_migration_count", 0)),
        "stamp_error_count": int(status.get("stamp_error_count", 0)),
        "migration_install_count": int(status.get("migration_install_count", 0)),
        "correction_fallback_count": int(status.get("correction_fallback_count", 0)),
        "pool_health": health,
        "_arrays": arrays, "_material": material,
    }


def run_worker(arguments) -> int:
    from experiment.seam_audit.dump_state import run_frames
    from experiment.seam_audit.solver_adapter import load_solver

    case_path, dimension, horizons = CASES[arguments.case_name]
    out = pathlib.Path(arguments.out) / f"{arguments.case_name}_{arguments.mode}"
    out.mkdir(parents=True, exist_ok=True)
    expected_value = MODES[arguments.mode][0]
    if os.environ.get("V6_DIAG_POISON_G1") != expected_value:
        raise RuntimeError(f"V6_DIAG_POISON_G1={os.environ.get('V6_DIAG_POISON_G1')!r}, expected {expected_value!r}")
    solver = load_solver("v6")
    global_case = solver.load_case(case_path)
    chain = solver.compute_chain_partition(global_case, [1.0, 1.0], POOL_SAFETY)
    initial_total = int(global_case.initial.positions.shape[0])
    material_count = len(global_case.materials)
    cadence = int(global_case.numerics.defrag_cadence) or 10 ** 12
    contexts, sims = [], []
    for index, device in enumerate(DEVICES):
        contexts.append(solver.Context.create(device_index=device, enable_validation=False,
                                              application_name=f"poison_g1_s{index}"))
        sims.append(solver.Simulator(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
    orchestrator = solver.Orchestrator(sims, defrag_cadence=cadence)
    orchestrator.bootstrap_all()
    report = {"case": arguments.case_name, "mode": arguments.mode, "initial_total": initial_total,
              "cuts": [int(cut) for cut in chain.cuts],
              "environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith("V6_")},
              "horizons": {}}
    defrag_log: list = []
    current = 0
    for horizon in horizons:
        if horizon > current:
            run_frames(orchestrator, current, horizon, DEPTH, cadence, defrag_log)
            current = horizon
        # no copy check right after the bootstrap or a boundary defrag: the defrag copies its zeroed twins over
        # the primary ghost / departed rows but never touches density_pressure_scratch
        check_copy = horizon > 0 and horizon % cadence != 0
        inspections = [inspect_sim(sim, material_count, check_copy=check_copy) for sim in sims]
        worker_stamp_errors = {worker.label: int(getattr(worker, "stamp_error_count", 0))
                               for worker in getattr(orchestrator, "workers", ())}
        entry = {
            "drift": sum(item["alive_own"] for item in inspections) - initial_total,
            "worker_stamp_errors": worker_stamp_errors,
            "sims": [{key: value for key, value in item.items() if not key.startswith("_")}
                     for item in inspections],
        }
        entry["nonfinite_rows_total"] = sum(item["nonfinite_rows_total"] for item in inspections)
        report["horizons"][str(horizon)] = entry
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[poison_g1] {arguments.case_name} {arguments.mode} N={horizon}: non-finite rows "
              f"{entry['nonfinite_rows_total']} drift {entry['drift']} overflow "
              f"{[item['overflow'] for item in inspections]} far "
              f"{[item['far_migration_count'] for item in inspections]}", flush=True)
        stop = MODES[arguments.mode][2] and entry["nonfinite_rows_total"] > 0
        if (horizon == horizons[-1] or stop) and arguments.save:
            dump = {}
            for index, item in enumerate(inspections):
                for name, values in item["_arrays"].items():
                    dump[f"s{index}_{name}"] = values
                dump[f"s{index}_material"] = item["_material"]
            np.savez(out / f"dump_N{horizon}.npz", **dump)
        del inspections
        if stop:
            print(f"[poison_g1] {arguments.case_name} {arguments.mode}: non-finite values at N={horizon} "
                  "(expected for this control); stopping", flush=True)
            break
    orchestrator.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()
    return 0


def verdict_of(report: dict, mode: str) -> dict:
    expected_nan = MODES[mode][2]
    horizons = report.get("horizons", {})
    first_nan = next((int(h) for h, entry in horizons.items() if entry["nonfinite_rows_total"]), None)
    clean = bool(horizons) and all(
        entry["nonfinite_rows_total"] == 0 and entry["drift"] == 0
        and not any(entry["worker_stamp_errors"].values())
        and all(sum(sim["overflow"].values()) == 0 and sim["far_migration_count"] == 0
                and sim["stamp_error_count"] == 0 and sim["bad_material_live_rows"] == 0
                and sim["correction_fallback_count"] == 0     # a NaN read absorbed by the KCG identity fallback
                and not any(sim.get("copy_check_mismatched_rows", {}).values())
                for sim in entry["sims"])
        for entry in horizons.values())
    passed = (first_nan is not None) if expected_nan else clean
    return {"mode": mode, "expected_nan": expected_nan, "first_nonfinite_horizon": first_nan,
            "clean": clean, "pass": passed}


def run_driver(arguments) -> int:
    out = pathlib.Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").unlink(missing_ok=True)
    verdict = {"runs": []}
    for case_name in arguments.cases.split(","):
        dimension = CASES[case_name][1]
        for mode in arguments.modes.split(","):
            command = [sys.executable, str(pathlib.Path(__file__).resolve()), "worker", "--case-name", case_name,
                       "--mode", mode, "--out", str(out)] + (["--save"] if arguments.save else [])
            # nothing of an earlier run in this --out may stand in for this one (a worker that dies before its
            # first horizon writes no report)
            run_directory = out / f"{case_name}_{mode}"
            for stale in [run_directory / "report.json"] + sorted(run_directory.glob("dump_N*.npz")):
                stale.unlink(missing_ok=True)
            log_path = out / f"{case_name}_{mode}.log"
            started = time.time()
            with open(log_path, "w", encoding="utf-8") as log:
                try:
                    code = subprocess.run(command, env=run_environment(dimension, mode), cwd=str(_REPO_ROOT),
                                          stdout=log, stderr=subprocess.STDOUT, timeout=arguments.timeout).returncode
                except subprocess.TimeoutExpired:
                    code = -9
            report_path = out / f"{case_name}_{mode}" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            entry = dict(verdict_of(report, mode), case=case_name, exit=code,
                         seconds=round(time.time() - started))
            if not MODES[mode][2]:
                entry["pass"] = entry["pass"] and code == 0
            verdict["runs"].append(entry)
            print(f"[poison_g1] {case_name} {mode}: {entry}", flush=True)
    verdict["pass"] = all(run["pass"] for run in verdict["runs"])
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    return 0 if verdict["pass"] else 3


def run_dry(arguments) -> int:
    """CPU only: release env in-process, partition + pool regions, no Vulkan import."""
    from experiment.seam_audit.solver_adapter import load_solver
    for case_name in arguments.cases.split(","):
        case_path, dimension, horizons = CASES[case_name]
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(run_environment(dimension, "pressure"))
        try:
            solver = load_solver("v6", include_gpu_modules=False)
            global_case = solver.load_case(str(_REPO_ROOT / case_path))
            chain = solver.compute_chain_partition(global_case, [1.0, 1.0], POOL_SAFETY)
            for index, slab in enumerate(chain.slabs):
                print(f"[poison_g1] dry {case_name} slab {index}: regions {pool_regions(slab.capacities)}")
            print(f"[poison_g1] dry {case_name}: horizons {horizons}, env "
                  f"{ {k: v for k, v in sorted(os.environ.items()) if k.startswith('V6_')} }")
        finally:
            os.environ.clear()
            os.environ.update(saved)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E23 G1 poison test")
    parser.add_argument("command", nargs="?", default="driver", choices=["driver", "worker"])
    parser.add_argument("--out", default="logs/seam_audit/opt/poison_g1")
    parser.add_argument("--cases", default="cavity2d_1m,cavity3d_1m")
    parser.add_argument("--modes", default="pressure,density")
    parser.add_argument("--case-name")
    parser.add_argument("--mode")
    parser.add_argument("--save", action="store_true", help="save the last horizon's full-pool arrays")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.dry_run:
        return run_dry(arguments)
    if arguments.command == "driver":
        return run_driver(arguments)
    try:
        code = run_worker(arguments)
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        print(RESULT_PREFIX + json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
        sys.stdout.flush()
        os._exit(1)     # no Vulkan teardown after a failure (frames in flight; dump_state policy)
    print(RESULT_PREFIX + json.dumps({"ok": True}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
