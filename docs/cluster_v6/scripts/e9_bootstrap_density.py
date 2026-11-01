"""e9_bootstrap_density.py - the bootstrap's extra density step of experiment/v7: size, restart path, the fix's residual scratch.

_record_bootstrap_compute_cmd ran correction + density and then the scratch -> primary copy, so force evaluated a^0 on
rho^0 + dt rho_dot^0 instead of rho^0 and step 1 integrated density from that value a second time. This driver measures it
on the workstation's two RTX 5090 with an external loop, for the solver in either worktree:

    old   E9_SOLVER_ROOT = the e8-3d worktree (tag v7-rc4's experiment/v7)
    new   E9_SOLVER_ROOT = this worktree (branch bootstrap-density-fix)

Modes of one run:
    bootstrap      bootstrap_all, then every alive particle's (rho, P, a, shift, v^{-1/2}): deviation from rho0 and where it sits
    trajectory     bootstrap_all + the production frame loop (depth 2, defrag at the cadence); at the record steps the 56 Marchi
                   points (MLS, wall frame, both velocity components), kinetic energy and density extrema
    poison         bootstrap_all, then density_pressure_scratch = quiet NaN in EVERY slot, then the frame loop: any live slot
                   whose scratch is not rewritten before its step's copy carries the NaN into primary and force
    restart_save   the frame loop to step 4000, checkpoint at the defrag boundary of step 2000 (cavity_runner's read_restart_state)
    restart_load   restart_all from that checkpoint: the restored state against the checkpoint, then 2000 steps

    python docs/cluster_v6/scripts/e9_bootstrap_density.py all --only old
    python docs/cluster_v6/scripts/e9_bootstrap_density.py all --only new
    python docs/cluster_v6/scripts/e9_bootstrap_density.py table
Output: docs/cluster_v6/data/e9/bootstrap_density/ (bootstrap_density.{md,csv,json}, runs/<run>/{run.json, run.log}, samples.npz);
particle arrays and checkpoints go to E9_ARRAY_DIRECTORY (not committed).
The cases are read from the rc3 worktree (the OBJ files are git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

_THIS_REPOSITORY = pathlib.Path(__file__).resolve().parents[3]
SOLVER_ROOT = pathlib.Path(os.environ.get("E9_SOLVER_ROOT", _THIS_REPOSITORY)).resolve()
sys.path.insert(0, str(SOLVER_ROOT))

from experiment.validation import cavity_runner as runner, cavity_sampling as sampling  # noqa: E402

PROJECTS = _THIS_REPOSITORY.parent
TREES = {"old": PROJECTS / "vulkan-demo-e8", "new": _THIS_REPOSITORY}
RC3_WORKTREE = PROJECTS / "vulkan-demo-v7-rc3"
CASES = {"n250_simple": RC3_WORKTREE / "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
         "n250_adami": RC3_WORKTREE / "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml",
         "n1000_adami": RC3_WORKTREE / "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml"}
POINTS_FILE = RC3_WORKTREE / "logs/validation/cavity_re1000_v7/n1000_k1_float32_xi0p001_eps0p0025_adami/points.npz"
OUTPUT_DIRECTORY = _THIS_REPOSITORY / "docs/cluster_v6/data/e9/bootstrap_density"
ARRAY_DIRECTORY = pathlib.Path(os.environ.get("E9_ARRAY_DIRECTORY", OUTPUT_DIRECTORY / "arrays"))
DEVICE_MAPS = {1: [1], 2: [0, 1]}                     # K = 1 on the headless 5090, K = 2 on both
PIPELINE_DEPTH = 2
POOL_SAFETY = 1.2
STALL_TIMEOUT_SECONDS = 120.0
TRAJECTORY_STEPS = (1, 10, 100, 1000, 10000, 33000, 100000, 333000)
TRAJECTORY_STEPS_K2 = (1, 10, 100, 1000, 10000, 33000)
POISON_STEPS = (1, 2, 3, 10, 100, 999, 1000, 1001, 1200)
RESTART_CHECKPOINT_STEP = 2000
RESTART_END_STEP = 4000
VARIANTS = {"release": {}, "unfused": {"V7_FUSED_CORRECTION_DENSITY": "0"}, "transfer_copy": {"V7_DENSITY_COPY_COMPUTE": "0"},
            "deep_wall_skip_on": {"V7_DEEP_WALL_SKIP": "1"}, "deep_wall_skip_off": {"V7_DEEP_WALL_SKIP": "0"}}
QUIET_NAN_BITS = 0x7FC00000
# written for the runs in runs/ (2026-11-01); the numbers are in the tables below
CONCLUSION = ["## 结论", "",
              "问题属实，已修，但理由是一致性，不是精度。论文已用的数据不需要重跑。", "",
              "- 问题：旧代码在融合 / 非融合、深壁跳过开 / 关、传输拷贝、K = 1 / 2、simple / adami 共 15 个组合上结果相同：26 个流体粒子被改动，全部在上角 2h 以内，"
              "最大 |ρ−ρ0|/ρ0 = 7.913e-4，最大压力 7,932 Pa = 7.9 ρ0U²；n250 与 n1000 数字相同（h/Δx 和 CFL 相同）。a^0 的最大值也被它主导：simple 215 → 38.7，adami 381 → 77.5（n250），"
              "adami 1615 → 1240（n1000）。",
              "- 修复：方案 A（启动步不做 scratch → primary 拷贝，用一个 compute 屏障替代）。15 个组合启动后流体 ρ = ρ0、流体和壁 P = 0。",
              "- 残留检查：defrag 不迁移 scratch，所以第 1 步看到的 scratch 无论如何都是过期且错位的，方案 A 没有引入新状态。毒化实测：启动后 scratch 全部写 NaN，16 个组合（含深壁跳过、传输拷贝）跑到第 1200 步，"
              "跨一次 defrag，活粒子 NaN 为 0；空槽里的 11,231 / 15,071 个 NaN 说明毒值确实经过了拷贝。方案 B 不需要。",
              "- 量级：差别只在启动瞬间显著。流体动能相对差：第 1 步 −46 % / −48 %，第 10 步 −10 %，第 100 步 −3.8e-4 / −7.0e-4，第 1000 步 −2.4e-6 / −9.5e-6；56 点瞬时速度从第 1000 步（t = 0.03）起与噪声底同量级，到 t ≈ 10 仍如此。",
              "- 重启：`restart_init` 从不走 `bootstrap_compute`，所以没有第二次密度积分，也没有第二次半步回退。实测 4 个运行重启后流体的 r、v、ρ、P、a、shift 与检查点逐位相同。adami 壁的 P 不同（K = 1 至 4.9e-4，K = 2 至 0.8 Pa），"
              "这是重启时的壁面重算，第一步 force 之前会再算一次；继续 2000 步后与不间断运行的差（≤ 1e-5）在同龄噪声内。",
              "- 未覆盖：GPU 测量只有 2-D。3-D 只有 CPU 上的录制流比较。warm 启动（2000² Re 3200）没有单独测，它走旧启动步且 ρ̇^0 不局限在角上，但同时缺的是整个压力场，这一步密度差不是主要的初始误差。", ""]


def planned_runs(tree: str) -> list[dict]:
    """Every run of one solver tree. A run's name is unique over the two trees."""
    plan = []

    def add(mode: str, case: str, slabs: int, variant: str = "release", repeat: str = "") -> None:
        name = "_".join(part for part in (tree, mode, case, f"k{slabs}", variant, repeat) if part)
        plan.append({"name": name, "tree": tree, "mode": mode, "case": case, "slabs": slabs, "variant": variant, "repeat": repeat})

    for case in CASES:
        add("bootstrap", case, 1)
    for case in ("n250_simple", "n250_adami"):
        for variant in ("unfused", "transfer_copy", "deep_wall_skip_on", "deep_wall_skip_off"):
            add("bootstrap", case, 1, variant)
        add("bootstrap", case, 2)
        add("bootstrap", case, 2, "unfused")
        for repeat in ("a", "b"):
            add("trajectory", case, 1, repeat=repeat)
    if tree == "new":
        for case in ("n250_simple", "n250_adami"):
            add("trajectory", case, 2)
            for slabs in (1, 2):
                for variant in ("release", "unfused", "deep_wall_skip_on", "transfer_copy"):
                    add("poison", case, slabs, variant)
                for repeat in ("a", "b"):
                    add("restart_save", case, slabs, repeat=repeat)
                add("restart_load", case, slabs, repeat="a")
    return plan


def advance(orchestrator, last_step: int, record_steps: tuple, on_record, on_boundary=None) -> None:
    """ChainOrchestratorV7.run_pipelined's legacy loop (it always starts at frame 0 and stops only at a defrag boundary), with
    a drain + on_record at the record steps. Same submit / wait / defrag order."""
    frame = 0
    next_wait = 0
    cadence = int(orchestrator.defrag_cadence)
    while frame < last_step:
        orchestrator._submit_frame(frame)
        frame += 1
        orchestrator._frame_count = frame
        while frame - next_wait >= PIPELINE_DEPTH:
            orchestrator._wait_frame(next_wait, STALL_TIMEOUT_SECONDS)
            next_wait += 1
        boundary = frame % cadence == 0
        if boundary or frame in record_steps:
            while next_wait < frame:
                orchestrator._wait_frame(next_wait, STALL_TIMEOUT_SECONDS)
                next_wait += 1
        if frame in record_steps:
            on_record(frame)
        if boundary:
            report = orchestrator._collect_defrag_report()
            if on_boundary is not None:
                on_boundary(frame, report)
            for sim in orchestrator.sims:
                sim.submit_defrag_and_wait()


def read_rows(sims: list) -> dict:
    """Alive own rows of the nine restart fields over every slab."""
    parts = [runner.read_restart_state(sim) for sim in sims]
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def row_order(rows: dict) -> np.ndarray:
    positions = rows["position_voxel_id"]
    velocities = rows["velocity_mass"]
    return np.lexsort((velocities[:, 1], velocities[:, 0], positions[:, 1], positions[:, 0]))


def check_invariants(orchestrator, sims: list, step: int) -> None:
    for index, sim in enumerate(sims):
        status = sim.readback_global_status()
        bad = {key: value for key, value in status.items() if key.startswith("overflow_") and value}
        for key in ("far_migration_count", "stamp_error_count"):
            if status.get(key, 0):
                bad[key] = status[key]
        if bad:
            raise RuntimeError(f"invariant violation at step {step}, sim {index}: {bad}")
    host_stamps = sum(getattr(worker, "stamp_error_count", 0) for worker in orchestrator.workers)
    if host_stamps:
        raise RuntimeError(f"host stamp errors at step {step}: {host_stamps}")


def group_summary(mask: np.ndarray, positions: np.ndarray, density: np.ndarray, rest: np.ndarray, pressure: np.ndarray,
                  acceleration: np.ndarray, corners: np.ndarray, support: float) -> dict:
    """Deviation of one particle group; the distance to the nearer lid corner is in units of h."""
    if not mask.any():
        return {"count": 0}
    relative = (density[mask] - rest[mask]) / rest[mask]
    points = positions[mask]
    corner_distance = np.min(np.linalg.norm(points[:, None, :] - corners[None, :, :], axis=2), axis=1) / support
    changed = relative != 0.0
    worst = int(np.argmax(np.abs(relative)))
    pressure_worst = int(np.argmax(np.abs(pressure[mask])))
    speed = np.linalg.norm(acceleration[mask], axis=1)
    return {"count": int(mask.sum()), "density_changed": int(changed.sum()),
            "density_changed_beyond_2h_of_a_corner": int((changed & (corner_distance > 2.0)).sum()),
            "density_changed_beyond_5h_of_a_corner": int((changed & (corner_distance > 5.0)).sum()),
            "max_relative_density_deviation": float(np.abs(relative).max()),
            "at_corner_distance_h": float(corner_distance[worst]), "position": [float(value) for value in points[worst]],
            "max_abs_pressure": float(np.abs(pressure[mask]).max()), "pressure_at_corner_distance_h": float(corner_distance[pressure_worst]),
            "pressure_changed": int((pressure[mask] != 0.0).sum()), "pressure_changed_beyond_5h_of_a_corner": int(((pressure[mask] != 0.0) & (corner_distance > 5.0)).sum()),
            "max_acceleration": float(speed.max()), "mean_acceleration": float(speed.mean())}


def run_one(specification: dict, directory: pathlib.Path) -> int:
    from experiment.v7.utils import simulator_v7 as simulator_module
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_FLUID
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition, restart_slab_rows
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7

    if not pathlib.Path(simulator_module.__file__).resolve().is_relative_to(SOLVER_ROOT):
        sys.exit(f"[bootstrap] solver imported from {simulator_module.__file__}, expected under {SOLVER_ROOT}")
    name, mode, slabs = specification["name"], specification["mode"], specification["slabs"]
    case_path = CASES[specification["case"]]
    global_case = load_case_v7(str(case_path))
    expected_total = int(global_case.initial.positions.shape[0])
    dt = float(global_case.physics.timestep)
    support = float(global_case.physics.smoothing_length)
    spacing = 2.0 * min(float(material.radius) for material in global_case.materials if float(material.radius) > 0)
    cadence = int(global_case.numerics.defrag_cadence)
    kinds = np.array([material.kind for material in global_case.materials])
    rest_densities = np.array([float(material.rest_density) for material in global_case.materials])
    fluid_groups = np.flatnonzero(kinds == KIND_FLUID).astype(np.uint32)
    lid_speed = max(float(np.linalg.norm(material.initial_velocity)) for material in global_case.materials)
    half_width = sampling.Frame.make("wall", spacing).half_width
    with np.load(POINTS_FILE) as archive:
        scale = half_width / float(sampling.Frame.make("wall", 0.001).half_width)
        point_sets = {"u_line": ("u", np.array(archive["u_marchi_wall"]) * scale), "v_line": ("v", np.array(archive["v_marchi_wall"]) * scale)}

    checkpoint_path = ARRAY_DIRECTORY / "_".join(("new_restart_save", specification["case"], f"k{slabs}", "release", specification["repeat"])) / "checkpoint.npz"
    restart_rows = None
    if mode == "restart_load":
        with np.load(checkpoint_path) as archive:
            restart_rows = {field: np.array(archive[field]) for field in archive.files if not field.startswith("_")}
            stored_offset = float(archive["_stored_density_offset"])
        global_case.initial = type(global_case.initial)(
            positions=np.ascontiguousarray(restart_rows["position_voxel_id"][:, :3], dtype=np.float32),
            velocities=np.ascontiguousarray(restart_rows["velocity_mass"][:, :3], dtype=np.float32),
            material_group=np.ascontiguousarray(restart_rows["material"], dtype=np.uint32))
    chain = compute_chain_partition(global_case, [1.0] * slabs, POOL_SAFETY)
    print(f"[bootstrap] {name}: solver {SOLVER_ROOT.name}, {expected_total:,} particles, dt {dt:.3e}, cadence {cadence}, "
          f"wall {global_case.numerics.wall_boundary}", flush=True)

    contexts, sims, readers = [], [], []
    orchestrator = None
    document: dict = {"run": specification, "solver_root": str(SOLVER_ROOT), "dt": dt, "support_radius": support, "spacing": spacing,
                      "lid_speed": lid_speed, "expected_total": expected_total, "defrag_cadence": cadence,
                      "wall_boundary": global_case.numerics.wall_boundary, "switches": {key: value for key, value in os.environ.items() if key.startswith("V7_")},
                      "records": {}}
    samples: dict[str, list] = {}
    record_steps: list[int] = []
    started = time.perf_counter()
    try:
        for index in range(slabs):
            contexts.append(VulkanContextV7.create(device_index=DEVICE_MAPS[slabs][index], enable_validation=False, application_name=f"bootstrap_s{index}"))
            sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        document["device_uuids"] = [runner.device_uuid(context) for context in contexts]
        document["fused_active"] = [bool(sim._fused_correction_density_active()) for sim in sims]
        document["deep_wall_skip_active"] = [bool(sim._deep_wall_skip_active()) for sim in sims]
        orchestrator = ChainOrchestratorV7(sims, defrag_cadence=cadence)

        if mode == "restart_load":
            rows_per_slab = restart_slab_rows(global_case, chain, restart_rows["position_voxel_id"][:, 0])
            states = [{field: np.ascontiguousarray(values[part]) for field, values in restart_rows.items()} for part in rows_per_slab]
            orchestrator.restart_all(states, stored_offset)
            del states
            restored = read_rows(sims)
            old_order, new_order = row_order(restart_rows), row_order(restored)
            fluid = np.isin(restart_rows["material"][old_order], fluid_groups)
            comparison = {}
            for field in ("position_voxel_id", "velocity_mass", "density_pressure", "acceleration", "shift"):
                saved, loaded = restart_rows[field][old_order], restored[field][new_order]
                if field == "position_voxel_id":
                    saved, loaded = saved[:, :3], loaded[:, :3]
                comparison[field] = {"fluid_bit_identical": bool(np.array_equal(saved[fluid].tobytes(), loaded[fluid].tobytes())),
                                     "wall_bit_identical": bool(np.array_equal(saved[~fluid].tobytes(), loaded[~fluid].tobytes())),
                                     "fluid_max_abs_difference": float(np.abs(saved[fluid] - loaded[fluid]).max()),
                                     "wall_max_abs_difference": float(np.abs(saved[~fluid] - loaded[~fluid]).max())}
            document["restored_vs_checkpoint"] = comparison
            print(f"[bootstrap] restored vs checkpoint: {comparison}", flush=True)
        else:
            orchestrator.bootstrap_all()

        if mode == "bootstrap":
            rows = read_rows(sims)
            order = row_order(rows)
            rows = {field: values[order] for field, values in rows.items()}
            offset = float(sims[0].stored_density_offset())
            positions = rows["position_voxel_id"][:, :2].astype(np.float64)
            density = rows["density_pressure"][:, 0].astype(np.float64) + offset
            pressure = rows["density_pressure"][:, 1].astype(np.float64)
            rest = rest_densities[rows["material"]]
            acceleration = rows["acceleration"][:, :2].astype(np.float64)
            fluid = np.isin(rows["material"], fluid_groups)
            fluid_positions = positions[fluid]
            corners = np.array([[fluid_positions[:, 0].min(), fluid_positions[:, 1].max()], [fluid_positions[:, 0].max(), fluid_positions[:, 1].max()]])
            document["alive"] = int(density.size)
            document["fluid"] = group_summary(fluid, positions, density, rest, pressure, acceleration, corners, support)
            document["wall"] = group_summary(~fluid, positions, density, rest, pressure, acceleration, corners, support)
            document["fluid_max_half_step_speed"] = float(np.linalg.norm(rows["velocity_mass"][fluid, :2], axis=1).max())
            array_directory = ARRAY_DIRECTORY / name
            array_directory.mkdir(parents=True, exist_ok=True)
            np.savez(array_directory / "bootstrap.npz", positions=positions, density=density, pressure=pressure, acceleration=acceleration,
                     shift=rows["shift"][:, :2], velocity=rows["velocity_mass"][:, :2], material=rows["material"])
            check_invariants(orchestrator, sims, 0)
            print(f"[bootstrap] fluid {document['fluid']}\n[bootstrap] wall {document['wall']}", flush=True)
        else:
            readers = [runner.CachedStateReader(sim) for sim in sims]
            if mode == "poison":
                for sim in sims:
                    buffer = sim.buffers["density_pressure_scratch"]
                    sim._staging_upload(buffer, np.full(buffer.size // 4, QUIET_NAN_BITS, dtype=np.uint32).tobytes())
                print("[bootstrap] density_pressure_scratch poisoned in every slot", flush=True)

            def poison_record() -> dict:
                """NaNs in primary over each slab's own range, in live slots and in empty ones (those show the poison reached the
                copy pass), and in every field of the alive rows."""
                live, empty = 0, 0
                for sim in sims:
                    raw = sim.readback_buffers_batch(["density_pressure", "velocity_mass", "position_voxel_id"], density="stored")
                    capacity = sim.case.capacities.total_pool_capacity()
                    first, stop = sim.own_first_pid(), sim.own_first_pid() + sim.case.capacities.own_pool_size
                    primary = np.frombuffer(raw["density_pressure"], np.float32)[:capacity * 2].reshape(capacity, 2)[first:stop]
                    velocity = np.frombuffer(raw["velocity_mass"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
                    position = np.frombuffer(raw["position_voxel_id"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
                    alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
                    bad = np.isnan(primary).any(axis=1)
                    live += int((bad & alive).sum())
                    empty += int((bad & ~alive).sum())
                rows = read_rows(sims)
                not_finite = {field: int((~np.isfinite(rows[field])).sum()) for field in
                              ("position_voxel_id", "velocity_mass", "density_pressure", "acceleration", "shift", "correction_inverse")}
                return {"live_primary_nan": live, "empty_primary_nan": empty, "alive_rows_not_finite": not_finite}

            def on_record(step: int) -> None:
                parts = [reader.read() for reader in readers]
                light = {field: np.concatenate([part[field] for part in parts]) for field in parts[0]}
                if int(light["mass"].size) != expected_total:
                    raise RuntimeError(f"drift at step {step}: alive {light['mass'].size} != {expected_total}")
                check_invariants(orchestrator, sims, step)
                fluid = np.isin(light["material"], fluid_groups)
                record_rest = rest_densities[light["material"][fluid]]
                relative = (light["density"][fluid] - record_rest) / record_rest
                record = {"time": step * dt, "alive": int(light["mass"].size), "kinetic_energy": float(0.5 * (light["mass"][fluid] * (light["velocities"][fluid] ** 2).sum(axis=1)).sum()),
                          "max_speed": float(np.sqrt((light["velocities"][fluid] ** 2).sum(axis=1).max())),
                          "max_relative_density_deviation": float(np.abs(relative).max()), "density_min": float(light["density"][fluid].min()), "density_max": float(light["density"][fluid].max())}
                if mode == "poison":
                    record.update(poison_record())
                    print(f"[bootstrap] step {step}: {record}", flush=True)
                else:
                    particles = sampling.ParticleSet(light["positions"], light["velocities"], light["mass"] / light["density"])
                    for line_name, (line, points) in point_sets.items():
                        axis = 0 if line == "u" else 1
                        strip = sampling.band(particles, axis, float(points[0, axis]), 1.05 * support)
                        sampled = sampling.interpolate(strip, points, support, tree=cKDTree(strip.positions))
                        samples.setdefault(f"{line_name}_mls", []).append(np.array(sampled["mls"]))
                    print(f"[bootstrap] step {step}: KE {record['kinetic_energy']:.9e}, max |drho|/rho0 {record['max_relative_density_deviation']:.3e}", flush=True)
                document["records"][str(step)] = record
                record_steps.append(step)

            def on_boundary(frame: int, report: list) -> None:
                if mode == "restart_save" and frame == RESTART_CHECKPOINT_STEP:
                    rows = read_rows(sims)
                    array_directory = checkpoint_path.parent
                    array_directory.mkdir(parents=True, exist_ok=True)
                    np.savez(array_directory / "checkpoint.npz", _stored_density_offset=float(sims[0].stored_density_offset()), **rows)
                    print(f"[bootstrap] checkpoint at step {frame}: {rows['material'].size:,} rows", flush=True)

            if mode == "trajectory":
                steps = TRAJECTORY_STEPS if slabs == 1 else TRAJECTORY_STEPS_K2
            elif mode == "poison":
                steps = POISON_STEPS
            elif mode == "restart_save":
                steps = (RESTART_CHECKPOINT_STEP, RESTART_END_STEP)
            else:
                steps = (RESTART_END_STEP - RESTART_CHECKPOINT_STEP,)
            advance(orchestrator, max(steps), steps, on_record, on_boundary)
    except Exception as error:
        print(f"[bootstrap] run {name} failed: {error!r}", flush=True)
        os._exit(1)                                  # frames may be in flight: the Vulkan teardown can block forever
    document["wall_seconds"] = time.perf_counter() - started
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps(document, indent=1), encoding="utf-8", newline="\n")
    if samples:
        np.savez(directory / "samples.npz", steps=np.array(record_steps), **{key: np.stack(values) for key, values in samples.items()})
    for reader in readers:
        reader.destroy()
    if orchestrator is not None:
        orchestrator.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()
    print(f"[bootstrap] run {name} done in {document['wall_seconds']:.1f} s", flush=True)
    return 0


def git_identity(tree: pathlib.Path) -> dict:
    def text(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, cwd=tree).stdout.strip()
    return {"root": str(tree), "head": text("rev-parse", "HEAD"), "branch": text("rev-parse", "--abbrev-ref", "HEAD"),
            "experiment_v7_tree": text("rev-parse", "HEAD:experiment/v7"), "dirty_experiment_v7": bool(text("status", "--porcelain", "experiment/v7"))}


def run_all(only: list[str], force: bool) -> int:
    base = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    base.update(runner.expected_environments("v7")["release"])
    base.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8"})
    (OUTPUT_DIRECTORY / "runs").mkdir(parents=True, exist_ok=True)
    failed = 0
    for tree in only:
        for specification in planned_runs(tree):
            directory = OUTPUT_DIRECTORY / "runs" / specification["name"]
            if (directory / "run.json").exists() and not force:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            environment = dict(base)
            environment.update(VARIANTS[specification["variant"]])
            environment["E9_SOLVER_ROOT"] = str(TREES[tree])
            print(f"[bootstrap] === {specification['name']} ===", flush=True)
            with open(directory / "run.log", "w", encoding="utf-8") as log:
                completed = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "one", "--specification", json.dumps(specification)],
                                           env=environment, cwd=TREES[tree], stdout=log, stderr=subprocess.STDOUT, text=True, timeout=1800)
            if completed.returncode != 0:
                failed += 1
                print(f"[bootstrap] {specification['name']} exit {completed.returncode}; see {directory / 'run.log'}", flush=True)
    print(f"[bootstrap] done, {failed} failed", flush=True)
    return 1 if failed else 0


def load_runs() -> dict:
    runs = {}
    for path in sorted((OUTPUT_DIRECTORY / "runs").glob("*/run.json")):
        runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def point_differences(first: pathlib.Path, second: pathlib.Path) -> dict:
    """step -> (max |du|, max |dv|) over the 56 points, per run pair: u at the u line, v at the v line."""
    out = {}
    with np.load(first) as a, np.load(second) as b:
        for step_index, step in enumerate(a["steps"]):
            matching = np.flatnonzero(b["steps"] == step)
            if matching.size == 0:
                continue
            other = int(matching[0])
            out[int(step)] = (float(np.abs(a["u_line_mls"][step_index, :, 0] - b["u_line_mls"][other, :, 0]).max()),
                              float(np.abs(a["v_line_mls"][step_index, :, 1] - b["v_line_mls"][other, :, 1]).max()))
    return out


def table() -> int:
    runs = load_runs()
    lines = ["# 启动步多积分一步密度：测量", "", "由 `docs/cluster_v6/scripts/e9_bootstrap_density.py table` 生成。old = v7-rc4 的 `experiment/v7`，new = 修复分支。", ""] + CONCLUSION
    bootstrap_rows = []
    for name, run in runs.items():
        specification = run["run"]
        if specification["mode"] != "bootstrap":
            continue
        fluid, wall = run["fluid"], run["wall"]
        bootstrap_rows.append({"run": name, "tree": specification["tree"], "case": specification["case"], "K": specification["slabs"], "variant": specification["variant"],
                               "fused": run["fused_active"][0], "deep_wall_skip": run["deep_wall_skip_active"][0],
                               "fluid_density_changed": fluid["density_changed"], "fluid_changed_beyond_5h": fluid["density_changed_beyond_5h_of_a_corner"],
                               "fluid_max_relative_density_deviation": fluid["max_relative_density_deviation"], "at_corner_distance_h": fluid["at_corner_distance_h"],
                               "fluid_max_abs_pressure": fluid["max_abs_pressure"], "wall_max_abs_pressure": wall["max_abs_pressure"], "wall_pressure_changed": wall["pressure_changed"],
                               "fluid_max_acceleration": fluid["max_acceleration"]})
    lines += ["## 启动后的状态（流体粒子，相对 ρ0）", "",
              "| 树 | 算例 | K | 路径 | 融合 | 改变的流体粒子 | 其中距上角 > 5h | max\\|ρ−ρ0\\|/ρ0 | 距角 (h) | 流体 max\\|P\\| | 壁 max\\|P\\| | 流体 max\\|a\\| |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(bootstrap_rows, key=lambda item: (item["case"], item["K"], item["variant"], item["tree"])):
        lines.append(f"| {row['tree']} | {row['case']} | {row['K']} | {row['variant']} | {'是' if row['fused'] else '否'} | {row['fluid_density_changed']:,} | {row['fluid_changed_beyond_5h']:,} | "
                     f"{row['fluid_max_relative_density_deviation']:.3e} | {row['at_corner_distance_h']:.2f} | {row['fluid_max_abs_pressure']:.3e} | {row['wall_max_abs_pressure']:.3e} | {row['fluid_max_acceleration']:.3e} |")
    lines.append("")

    trajectory_rows = []
    directory = OUTPUT_DIRECTORY / "runs"
    for case in ("n250_simple", "n250_adami"):
        names = {key: f"{tree}_trajectory_{case}_k1_release_{repeat}" for key, tree, repeat in
                 (("old_a", "old", "a"), ("old_b", "old", "b"), ("new_a", "new", "a"), ("new_b", "new", "b"))}
        if not all((directory / name / "samples.npz").exists() for name in names.values()):
            continue
        pairs = {"new_a - old_a": ("new_a", "old_a"), "new_b - old_b": ("new_b", "old_b"), "old_b - old_a (floor)": ("old_b", "old_a"), "new_b - new_a (floor)": ("new_b", "new_a")}
        differences = {label: point_differences(directory / names[first] / "samples.npz", directory / names[second] / "samples.npz") for label, (first, second) in pairs.items()}
        for step in sorted(next(iter(differences.values()))):
            record_old = runs[names["old_a"]]["records"][str(step)]
            record_new = runs[names["new_a"]]["records"][str(step)]
            row = {"case": case, "step": step, "time": record_old["time"], "ke_relative_difference": (record_new["kinetic_energy"] - record_old["kinetic_energy"]) / record_old["kinetic_energy"],
                   "density_deviation_old": record_old["max_relative_density_deviation"], "density_deviation_new": record_new["max_relative_density_deviation"]}
            for label, values in differences.items():
                row[label] = values[step]
            trajectory_rows.append(row)
    if trajectory_rows:
        lines += ["## 轨迹：56 点瞬时速度的最大差（/U，u 线取 u、v 线取 v；K = 1）", "",
                  "噪声底 = 同一份代码两次运行之差（体素内顺序逐次不同）。", "",
                  "| 算例 | 步 | t | new−old (a) Δu / Δv | new−old (b) Δu / Δv | 底 old Δu / Δv | 底 new Δu / Δv | KE 相对差 | max\\|ρ−ρ0\\|/ρ0 old | new |",
                  "|---|---:|---:|---|---|---|---|---:|---:|---:|"]
        for row in trajectory_rows:
            pair = lambda label: f"{row[label][0]:.2e} / {row[label][1]:.2e}"
            lines.append(f"| {row['case']} | {row['step']} | {row['time']:.3f} | {pair('new_a - old_a')} | {pair('new_b - old_b')} | {pair('old_b - old_a (floor)')} | {pair('new_b - new_a (floor)')} | "
                         f"{row['ke_relative_difference']:.2e} | {row['density_deviation_old']:.3e} | {row['density_deviation_new']:.3e} |")
        lines.append("")

    poison_rows = []
    for name, run in runs.items():
        specification = run["run"]
        if specification["mode"] != "poison":
            continue
        records = run["records"]
        poison_rows.append({"run": name, "case": specification["case"], "K": specification["slabs"], "variant": specification["variant"], "steps": max(int(step) for step in records),
                            "fused": run["fused_active"][0], "deep_wall_skip": all(run["deep_wall_skip_active"]),
                            "live_primary_nan": max(record["live_primary_nan"] for record in records.values()), "empty_primary_nan_step_1": records["1"]["empty_primary_nan"],
                            "empty_primary_nan_last": records[str(max(int(step) for step in records))]["empty_primary_nan"],
                            "alive_rows_not_finite": max(sum(record["alive_rows_not_finite"].values()) for record in records.values()),
                            "max_relative_density_deviation": max(record["max_relative_density_deviation"] for record in records.values())})
    if poison_rows:
        lines += ["## 毒化：启动后 scratch 全部写成 NaN，再跑到第 1200 步", "",
                  "| 算例 | K | 路径 | 融合 | 深壁跳过 | 活粒子 primary 的 NaN | 活粒子各字段非有限值 | 空槽 NaN（第 1 步，说明毒值到过拷贝） | 空槽 NaN（末步） | max\\|ρ−ρ0\\|/ρ0 |", "|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
        for row in sorted(poison_rows, key=lambda item: (item["case"], item["K"], item["variant"])):
            lines.append(f"| {row['case']} | {row['K']} | {row['variant']} | {'是' if row['fused'] else '否'} | {'是' if row['deep_wall_skip'] else '否'} | {row['live_primary_nan']} | {row['alive_rows_not_finite']} | "
                         f"{row['empty_primary_nan_step_1']:,} | {row['empty_primary_nan_last']:,} | {row['max_relative_density_deviation']:.3e} |")
        lines.append("")

    restart_rows = []
    for case in ("n250_simple", "n250_adami"):
        for slabs in (1, 2):
            load_name = f"new_restart_load_{case}_k{slabs}_release_a"
            save_a, save_b = f"new_restart_save_{case}_k{slabs}_release_a", f"new_restart_save_{case}_k{slabs}_release_b"
            if not all((directory / item / "samples.npz").exists() for item in (load_name, save_a, save_b)):
                continue
            with np.load(directory / load_name / "samples.npz") as loaded, np.load(directory / save_a / "samples.npz") as saved, np.load(directory / save_b / "samples.npz") as independent:
                last = int(np.flatnonzero(saved["steps"] == RESTART_END_STEP)[0])
                restart_difference = (float(np.abs(loaded["u_line_mls"][0, :, 0] - saved["u_line_mls"][last, :, 0]).max()), float(np.abs(loaded["v_line_mls"][0, :, 1] - saved["v_line_mls"][last, :, 1]).max()))
                floor = (float(np.abs(independent["u_line_mls"][last, :, 0] - saved["u_line_mls"][last, :, 0]).max()), float(np.abs(independent["v_line_mls"][last, :, 1] - saved["v_line_mls"][last, :, 1]).max()))
            restored = runs[load_name]["restored_vs_checkpoint"]
            restart_rows.append({"case": case, "K": slabs, "restored": restored, "restart_difference": restart_difference, "floor": floor})
    if restart_rows:
        lines += ["## 重启：restart_all 后的状态与检查点比较，以及继续 2000 步", "",
                  "| 算例 | K | 流体 v 逐位相同 | 流体 (ρ, P) 逐位相同 | 流体 a 逐位相同 | 流体 shift 逐位相同 | 重启 − 不间断 Δu / Δv | 噪声底 Δu / Δv |", "|---|---:|---|---|---|---|---|---|"]
        for row in restart_rows:
            restored = row["restored"]
            mark = lambda field: "是" if restored[field]["fluid_bit_identical"] else f"否 ({restored[field]['fluid_max_abs_difference']:.1e})"
            lines.append(f"| {row['case']} | {row['K']} | {mark('velocity_mass')} | {mark('density_pressure')} | {mark('acceleration')} | {mark('shift')} | "
                         f"{row['restart_difference'][0]:.2e} / {row['restart_difference'][1]:.2e} | {row['floor'][0]:.2e} / {row['floor'][1]:.2e} |")
        lines.append("")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "bootstrap_density.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    identity = {tree: git_identity(path) for tree, path in TREES.items()}
    (OUTPUT_DIRECTORY / "bootstrap_density.json").write_text(json.dumps({"solvers": identity, "bootstrap": bootstrap_rows, "trajectory": trajectory_rows, "poison": poison_rows, "restart": restart_rows}, indent=1), encoding="utf-8", newline="\n")
    if bootstrap_rows:
        with open(OUTPUT_DIRECTORY / "bootstrap_density.csv", "w", newline="\n", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
            writer.writeheader()
            writer.writerows(bootstrap_rows)
    print(f"[bootstrap] tables written: {len(bootstrap_rows)} bootstrap, {len(trajectory_rows)} trajectory, {len(poison_rows)} poison, {len(restart_rows)} restart rows", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    every = commands.add_parser("all")
    every.add_argument("--only", nargs="+", choices=("old", "new"), default=["old", "new"])
    every.add_argument("--force", action="store_true", help="repeat runs that already have a run.json")
    single = commands.add_parser("one")
    single.add_argument("--specification", required=True)
    commands.add_parser("table")
    arguments = parser.parse_args()
    if arguments.command == "all":
        return run_all(arguments.only, arguments.force)
    if arguments.command == "table":
        return table()
    specification = json.loads(arguments.specification)
    return run_one(specification, OUTPUT_DIRECTORY / "runs" / specification["name"])


if __name__ == "__main__":
    sys.exit(main())
