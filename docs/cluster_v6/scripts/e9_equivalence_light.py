"""e9_equivalence_light.py - light K = 2 vs K = 1 equivalence check on the workstation's two cards (user 2026-10-31).

Re = 1000 cavity, 1001^2 fluid particles, the constants of the paper's validation run. Every run starts from the SAME K = 1
checkpoint of that run and advances 500 steps; at steps 1, 10, 100 and 500 the instantaneous velocity is read at the 28 + 28
Marchi 2021 points (MLS, wall frame: the validation run's own points.npz and cavity_sampling.interpolate, nothing is
time averaged). K = 2 uses equal weights, which cuts the chain at the cavity's centre, so the 28 points of u(0.5, y) lie on
the seam. Two RTX 5090:  a1, a2 = K = 1 on the headless card;  b = K = 2 on both cards.

No solver code changes. The start follows cavity_runner's --start-from path (experiment/validation in the rc3 worktree):
the checkpoint's nine fields stand in for the initial condition, compute_chain_partition cuts by the state's fluid
histogram, restart_slab_rows splits the rows, ChainOrchestratorV7.restart_all restores them and runs the one restart
defrag. The frame loop is run_pipelined's legacy loop at the production depth 2, plus a drain at the record steps
(run_pipelined can only stop at a defrag boundary). The next periodic defrag is at step 1000 of the window, so the 500
steps contain none. Every record step checks the runner's invariants (alive total, overflow_*, far_migration, stamps).
The solver is this worktree's experiment/v7 (tree of tag v7-rc4), the case is read from the rc3 worktree because the
OBJ files are git-ignored (case.yaml and materials/standard.yaml have the same bytes in both worktrees, recorded).

Besides the 56-point values each record step stores sha256 hashes of the alive particles' (x, y, u, v, rho): in pid order
and sorted by (x, y, u, v), and the same two for (x, y, u, v) alone ("kinematic_*"; the first set, runs/, predates these
two). Equal raw hashes = the whole state is bit-identical, pid order included.

    python docs/cluster_v6/scripts/e9_equivalence_light.py all                          # a1, a2, b, then the tables
    python docs/cluster_v6/scripts/e9_equivalence_light.py all --subdirectory repeat    # the independent repeat
    python docs/cluster_v6/scripts/e9_equivalence_light.py table                        # the tables from the saved runs
Output: docs/cluster_v6/data/e9/equivalence_light/ (equivalence_light.{md,csv,json}, runs/ and repeat/, each
<run>/{samples.npz, run.json, run.log}).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.validation import cavity_runner as runner, cavity_sampling as sampling  # noqa: E402

RC3_WORKTREE = _REPOSITORY_ROOT.parent / "vulkan-demo-v7-rc3"
CASE_PATH = RC3_WORKTREE / "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml"
SOURCE_RUN = RC3_WORKTREE / "logs/validation/cavity_re1000_v7/n1000_k1_float32_xi0p001_eps0p0025_adami"
CHECKPOINT_PATH = SOURCE_RUN / "checkpoints/c0011907000.npz"
OUTPUT_DIRECTORY = _REPOSITORY_ROOT / "docs/cluster_v6/data/e9/equivalence_light"
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
DISPLAY_5090_UUID = "fb83f0dafd9e1d38dc05e7ae643e22a8"
RECORD_STEPS = (1, 10, 100, 500)
PIPELINE_DEPTH = 2
POOL_SAFETY = 1.2
RUNS = {"a1": (1, "1", HEADLESS_5090_UUID), "a2": (1, "1", HEADLESS_5090_UUID),
        "b": (2, "0,1", f"{DISPLAY_5090_UUID},{HEADLESS_5090_UUID}")}
COMPARISONS = (("a2", "a1"), ("b", "a1"))
# written for the two data sets in runs/ and repeat/ (2026-10-31), the ratios are in equivalence_light.csv
CONCLUSION = ["## 结论", "",
              "b 的差和噪声底同量级：第 10、100、500 步，两组数据的两个分量，b 对 a1 的 56 点最大差与 a1 对 a2 之比都在 0.65–1.53 之间。", "",
              "缝上的点没有超出噪声。同样 28 个缝上点，b 对 a1 与 a1 对 a2（K = 1 没有缝）的最大差之比在 0.60–1.40 之间。"
              "缝上 28 点与其余 27 点直接比，方向不固定，而且 K = 1 对也有同样的起伏：第一组第 500 步缝上是其余的 0.88 / 0.87 倍，"
              "重复组第 500 步是 2.35 / 2.01 倍，同组 K = 1 对在同样的点上是 1.70 / 1.75 倍；第一组第 10 步 Δu 是 2.40 倍，K = 1 对是 2.34 倍。"
              "没有缝的 K = 1 对也出现同样的差别，所以它不是缝造成的。", "",
              "分辨力：比值本身散布到约 1.5 倍（每组只有一对 K = 1，b 对 a1 与 a1 对 a2 共用 a1），这个检查分辨不了 1.5 倍以内的缝效应。", ""]


def sha256_of_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_text(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, cwd=_REPOSITORY_ROOT).stdout.strip()


def load_checkpoint(path: pathlib.Path, expected_total: int, layout: dict, dt: float) -> tuple[dict, float, int, dict]:
    """cavity_runner.load_start_state's checks (that function is in the rc3 worktree's runner, not in this tree's)."""
    with np.load(path) as archive:
        rows = {name: np.array(archive[name]) for name in archive.files if not name.startswith("_")}
        stored_offset = float(archive["_stored_density_offset"])
    if set(rows) != set(layout):
        sys.exit(f"[equivalence] {path}: fields {sorted(rows)} differ from {sorted(layout)}")
    count = int(rows["position_voxel_id"].shape[0])
    for name, (component_count, element_type) in layout.items():
        shape = (count,) if component_count == 1 else (count, component_count)
        if rows[name].shape != shape or rows[name].dtype != element_type:
            sys.exit(f"[equivalence] {path}: {name} is {rows[name].dtype} {rows[name].shape}")
    if count != expected_total:
        sys.exit(f"[equivalence] {path}: {count} rows, the case has {expected_total} particles")
    source_step = runner.step_of(path)
    meta = json.loads((path.parent.parent / "meta.json").read_text(encoding="utf-8"))
    if abs(float(meta["dt"]) - dt) > 1e-15 or int(meta["expected_total"]) != expected_total:
        sys.exit(f"[equivalence] {path}: the source run's dt / particle count differ from the case's")
    provenance = {"file": str(path), "sha256": sha256_of_file(path), "rows": count, "stored_density_offset": stored_offset,
                  "source_run": path.parent.parent.name, "source_step": source_step, "source_time": source_step * dt,
                  "source_git": meta.get("git"), "source_environment": meta.get("environment")}
    return rows, stored_offset, source_step, provenance


def state_hashes(light: dict) -> dict:
    fields = (light["positions"], light["velocities"], light["density"])
    raw = hashlib.sha256()
    for field in fields:
        raw.update(np.ascontiguousarray(field).tobytes())
    order = np.lexsort((light["velocities"][:, 1], light["velocities"][:, 0], light["positions"][:, 1], light["positions"][:, 0]))
    ordered = hashlib.sha256()
    for field in fields:
        ordered.update(np.ascontiguousarray(field[order]).tobytes())
    kinematic_fields = (light["positions"], light["velocities"])
    kinematic_raw = hashlib.sha256(b"".join(np.ascontiguousarray(field).tobytes() for field in kinematic_fields))
    kinematic_ordered = hashlib.sha256(b"".join(np.ascontiguousarray(field[order]).tobytes() for field in kinematic_fields))
    return {"pid_order": raw.hexdigest(), "sorted": ordered.hexdigest(),
            "kinematic_pid_order": kinematic_raw.hexdigest(), "kinematic_sorted": kinematic_ordered.hexdigest()}


def sample_points(light: dict, support: float, point_sets: dict) -> dict:
    """Both velocity components at every point, by the validation run's interpolation. sample_centerlines (the validation
    run's call, which keeps only the line's own component) runs too and must give the same bits."""
    particles = sampling.ParticleSet(light["positions"], light["velocities"], light["mass"] / light["density"])
    reference = sampling.sample_centerlines(particles, support, point_sets)
    out = {}
    for name, (line, points) in point_sets.items():
        axis = 0 if line == "u" else 1
        strip = sampling.band(particles, axis, float(points[0, axis]), 1.05 * support)
        sampled = sampling.interpolate(strip, points, support, tree=cKDTree(strip.positions))
        if not np.array_equal(sampled["mls"][:, 0 if line == "u" else 1], reference[name]["mls"], equal_nan=True):
            raise RuntimeError(f"{name}: interpolate and sample_centerlines disagree")
        out[name] = sampled
    return out


def run_frames(orchestrator, record_steps: tuple, on_record, depth: int = PIPELINE_DEPTH, stall_timeout_s: float = 120.0) -> None:
    """ChainOrchestratorV7.run_pipelined's legacy loop, with a drain + on_record at the record steps instead of the defrag
    boundary (no boundary falls inside the window)."""
    frame = 0
    next_wait = 0
    while frame < max(record_steps):
        orchestrator._submit_frame(frame)
        frame += 1
        orchestrator._frame_count = frame
        while frame - next_wait >= depth:
            orchestrator._wait_frame(next_wait, stall_timeout_s)
            next_wait += 1
        if frame in record_steps:
            while next_wait < frame:
                orchestrator._wait_frame(next_wait, stall_timeout_s)
                next_wait += 1
            on_record(frame)


def run_one(name: str, directory: pathlib.Path) -> int:
    slabs, device_map_text, require_uuid = RUNS[name]
    environment = runner.check_environment("release", "v7")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_FLUID
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition, restart_slab_rows
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7

    global_case = load_case_v7(str(CASE_PATH))
    expected_total = int(global_case.initial.positions.shape[0])
    dt = float(global_case.physics.timestep)
    support = float(global_case.physics.smoothing_length)
    spacing = 2.0 * min(float(material.radius) for material in global_case.materials if float(material.radius) > 0)
    fluid_groups = np.array([index for index, material in enumerate(global_case.materials) if material.kind == KIND_FLUID], dtype=np.uint32)
    lid_speed = max(float(np.linalg.norm(material.initial_velocity)) for material in global_case.materials)
    if global_case.numerics.wall_boundary != "adami":
        sys.exit("[equivalence] the validation case runs wall_boundary adami")
    source_meta = json.loads((SOURCE_RUN / "meta.json").read_text(encoding="utf-8"))
    for key, value in (("dt", dt), ("support_radius", support), ("spacing", spacing), ("expected_total", expected_total),
                       ("environment", environment), ("defrag_cadence", int(global_case.numerics.defrag_cadence))):
        if source_meta[key] != value:
            sys.exit(f"[equivalence] {key}: {value} here, {source_meta[key]} in the validation run")
    rows, stored_offset, start_step, checkpoint = load_checkpoint(CHECKPOINT_PATH, expected_total, SphSimulatorV7.RESTART_FIELD_LAYOUT, dt)
    global_case.initial = type(global_case.initial)(
        positions=np.ascontiguousarray(rows["position_voxel_id"][:, :3], dtype=np.float32),
        velocities=np.ascontiguousarray(rows["velocity_mass"][:, :3], dtype=np.float32),
        material_group=np.ascontiguousarray(rows["material"], dtype=np.uint32))
    weights = [1.0] * slabs
    chain = compute_chain_partition(global_case, weights, POOL_SAFETY)
    device_map = [int(item) for item in device_map_text.split(",")]
    cuts = list(getattr(chain, "cuts", []))
    cuts_physical = [float(global_case.grid.origin_x + cut * support) for cut in cuts]
    half_width = sampling.Frame.make("wall", spacing).half_width
    with np.load(SOURCE_RUN / "points.npz") as archive:
        point_sets = {"u_line": ("u", np.array(archive["u_marchi_wall"])), "v_line": ("v", np.array(archive["v_marchi_wall"]))}
        reference_coordinates = {"u_line": np.array(archive["u_reference_y"]), "v_line": np.array(archive["v_reference_x"])}

    print(f"[equivalence] run {name}: K = {slabs} devices {device_map}, checkpoint step {start_step} (t = {start_step * dt:.4f}), "
          f"{expected_total:,} particles, dt {dt:.3e}, cuts {cuts} x {cuts_physical}", flush=True)
    contexts, sims, readers = [], [], []
    orchestrator = None
    records: dict[int, dict] = {}
    arrays: dict[str, list] = {}
    started = time.perf_counter()
    try:
        for index in range(slabs):
            contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=False, application_name=f"equivalence_s{index}"))
            sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        uuids = [runner.device_uuid(ctx) for ctx in contexts]
        if uuids != require_uuid.split(","):
            raise RuntimeError(f"device uuids {uuids} differ from {require_uuid}")
        orchestrator = ChainOrchestratorV7(sims, defrag_cadence=int(global_case.numerics.defrag_cadence))
        rows_per_slab = restart_slab_rows(global_case, chain, rows["position_voxel_id"][:, 0])
        slab_particles = [int(part.size) for part in rows_per_slab]
        states = [{field: np.ascontiguousarray(values[part]) for field, values in rows.items()} for part in rows_per_slab]
        orchestrator.restart_all(states, stored_offset)
        del states
        readers = [runner.CachedStateReader(sim) for sim in sims]

        def on_record(step: int) -> None:
            sample_started = time.perf_counter()
            parts = [reader.read() for reader in readers]
            light = {field: np.concatenate([part[field] for part in parts]) for field in parts[0]}
            if int(light["mass"].size) != expected_total:
                raise RuntimeError(f"drift at step {step}: alive {light['mass'].size} != {expected_total}")
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
            sampled = sample_points(light, support, point_sets)
            for line in point_sets:
                for key, values in (("mls", sampled[line]["mls"]), ("shepard", sampled[line]["shepard"]),
                                    ("neighbours", sampled[line]["neighbours"]), ("mls_fallback", sampled[line]["mls_fallback"])):
                    arrays.setdefault(f"{line}_{key}", []).append(np.array(values))
            fluid = np.isin(light["material"], fluid_groups)
            records[step] = {"global_step": start_step + step, "time": (start_step + step) * dt, "alive": int(light["mass"].size),
                             "state_sha256": state_hashes(light), "max_speed": float(np.sqrt((light["velocities"][fluid] ** 2).sum(axis=1).max())),
                             "density_min": float(light["density"][fluid].min()), "density_max": float(light["density"][fluid].max()),
                             "sample_seconds": time.perf_counter() - sample_started}
            print(f"[equivalence] {name} step {step}: alive {records[step]['alive']:,}, max speed {records[step]['max_speed']:.6f}, "
                  f"state {records[step]['state_sha256']['pid_order'][:12]} / {records[step]['state_sha256']['sorted'][:12]}", flush=True)

        run_frames(orchestrator, RECORD_STEPS, on_record)
    except Exception as error:
        print(f"[equivalence] run {name} failed: {error!r}", flush=True)
        os._exit(1)                                  # frames may be in flight: the Vulkan teardown can block forever
    wall_seconds = time.perf_counter() - started
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "samples.npz", steps=np.array(RECORD_STEPS), global_steps=np.array([records[step]["global_step"] for step in RECORD_STEPS]),
             u_line_points=point_sets["u_line"][1], v_line_points=point_sets["v_line"][1],
             u_line_reference=reference_coordinates["u_line"], v_line_reference=reference_coordinates["v_line"],
             **{key: np.stack(values) for key, values in arrays.items()})
    document = {"run": name, "slabs": slabs, "device_map": device_map, "device_uuids": uuids, "weights": weights, "depth": PIPELINE_DEPTH,
                "cuts": cuts, "cuts_x_physical": cuts_physical, "cuts_x_reference": [(x + half_width) / (2.0 * half_width) for x in cuts_physical],
                "slab_particles": slab_particles, "own_first_columns": [int(g.own_global_first_column) for g in chain.geometry],
                "own_last_columns": [int(g.own_global_last_column) for g in chain.geometry],
                "ghost_layers": [int(slab.ghost_grid.ghost_layers) if hasattr(slab.ghost_grid, "ghost_layers") else None for slab in chain.slabs],
                "checkpoint": checkpoint, "start_step": start_step, "dt": dt, "support_radius": support, "spacing": spacing, "lid_speed": lid_speed,
                "expected_total": expected_total, "fluid_particles": int(np.isin(rows["material"], fluid_groups).sum()),
                "defrag_cadence": int(global_case.numerics.defrag_cadence), "use_prefix_sum_defrag": bool(global_case.numerics.use_prefix_sum_defrag),
                "wall_boundary": global_case.numerics.wall_boundary, "environment": environment,
                "case": {"path": str(CASE_PATH), "sha256": sha256_of_file(CASE_PATH),
                         "same_as_this_worktree": sha256_of_file(CASE_PATH) == sha256_of_file(_REPOSITORY_ROOT / CASE_PATH.relative_to(RC3_WORKTREE)),
                         "materials_same_as_this_worktree": sha256_of_file(RC3_WORKTREE / "materials/standard.yaml") == sha256_of_file(_REPOSITORY_ROOT / "materials/standard.yaml")},
                "solver": {"git_head": git_text("rev-parse", "HEAD"), "experiment_v7_tree": git_text("rev-parse", "HEAD:experiment/v7"),
                           "v7_rc4_tree": git_text("rev-parse", "v7-rc4:experiment/v7"), "dirty": bool(git_text("status", "--porcelain", "experiment/v7", "experiment/validation"))},
                "records": {str(step): record for step, record in records.items()}, "wall_seconds": wall_seconds,
                "unix_end": time.time()}
    (directory / "run.json").write_text(json.dumps(document, indent=1), encoding="utf-8", newline="\n")
    for reader in readers:
        reader.destroy()
    if orchestrator is not None:
        orchestrator.destroy()
    for sim in sims:
        sim.destroy()
    for ctx in contexts:
        ctx.destroy()
    print(f"[equivalence] run {name} done in {wall_seconds:.1f} s", flush=True)
    return 0


def run_all(subdirectory: str) -> int:
    runs_directory = OUTPUT_DIRECTORY / subdirectory
    if any((runs_directory / name / "run.json").exists() for name in RUNS):
        sys.exit(f"[equivalence] {runs_directory} already holds runs; move them away first")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    environment.update(runner.expected_environments("v7")["release"])
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8"})
    for name in RUNS:
        directory = runs_directory / name
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[equivalence] === run {name} ===", flush=True)
        with open(directory / "run.log", "w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "one", "--run", name, "--subdirectory", subdirectory], env=environment,
                                       cwd=_REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0:
            print(f"[equivalence] run {name} exit {completed.returncode}; see {directory / 'run.log'}", flush=True)
            return completed.returncode
        print(f"[equivalence] run {name} ok", flush=True)
    return table()


def load_runs(subdirectory: str) -> dict:
    runs = {}
    for name in RUNS:
        directory = OUTPUT_DIRECTORY / subdirectory / name
        with np.load(directory / "samples.npz") as archive:
            samples = {key: np.array(archive[key]) for key in archive.files}
        runs[name] = {"samples": samples, "document": json.loads((directory / "run.json").read_text(encoding="utf-8"))}
    first = runs["a1"]["document"]
    for name, run in runs.items():
        for key in ("start_step", "dt", "lid_speed", "expected_total", "environment"):
            if run["document"][key] != first[key]:
                sys.exit(f"[equivalence] {subdirectory}/{name}: {key} differs from a1")
        if run["document"]["checkpoint"]["sha256"] != first["checkpoint"]["sha256"]:
            sys.exit(f"[equivalence] {subdirectory}/{name}: another checkpoint")
    return runs


def maximum_with_location(delta: np.ndarray, labels: list) -> tuple[float, str]:
    position = int(np.argmax(np.abs(delta)))
    return float(abs(delta[position])), labels[position]


def comparison_rows(runs: dict, dataset: str) -> list:
    first = runs["a1"]["document"]
    lid_speed = float(first["lid_speed"])
    samples_a1 = runs["a1"]["samples"]
    u_labels = [f"u line y = {value:.7g}" for value in samples_a1["u_line_reference"]]
    v_labels = [f"v line x = {value:.7g}" for value in samples_a1["v_line_reference"]]
    labels = u_labels + v_labels
    centre_on_v_line = int(np.argmin(np.abs(samples_a1["v_line_reference"] - 0.5)))
    off_seam = [index for index in range(len(v_labels)) if index != centre_on_v_line]
    rows = []
    for later, earlier in COMPARISONS:
        one, two = runs[earlier]["samples"], runs[later]["samples"]
        for step_index, step in enumerate(int(value) for value in samples_a1["steps"]):
            delta_u_line = (two["u_line_mls"][step_index] - one["u_line_mls"][step_index]) / lid_speed
            delta_v_line = (two["v_line_mls"][step_index] - one["v_line_mls"][step_index]) / lid_speed
            max_u, where_u = maximum_with_location(np.concatenate([delta_u_line[:, 0], delta_v_line[:, 0]]), labels)
            max_v, where_v = maximum_with_location(np.concatenate([delta_u_line[:, 1], delta_v_line[:, 1]]), labels)
            seam_u, seam_u_where = maximum_with_location(delta_u_line[:, 0], u_labels)
            seam_v, seam_v_where = maximum_with_location(delta_u_line[:, 1], u_labels)
            hashes_one = runs[earlier]["document"]["records"][str(step)]["state_sha256"]
            hashes_two = runs[later]["document"]["records"][str(step)]["state_sha256"]
            rows.append({"dataset": dataset, "comparison": f"{later}-{earlier}", "step": step, "global_step": first["start_step"] + step,
                         "max_du_all56": max_u, "max_dv_all56": max_v, "at_du": where_u, "at_dv": where_v,
                         "max_du_seam28": seam_u, "max_dv_seam28": seam_v, "at_du_seam": seam_u_where, "at_dv_seam": seam_v_where,
                         "max_du_off_seam27": float(np.abs(delta_v_line[off_seam, 0]).max()), "max_dv_off_seam27": float(np.abs(delta_v_line[off_seam, 1]).max()),
                         "rms_du_seam28": float(np.sqrt(np.mean(delta_u_line[:, 0] ** 2))), "rms_dv_seam28": float(np.sqrt(np.mean(delta_u_line[:, 1] ** 2))),
                         "rms_du_off_seam27": float(np.sqrt(np.mean(delta_v_line[off_seam, 0] ** 2))), "rms_dv_off_seam27": float(np.sqrt(np.mean(delta_v_line[off_seam, 1] ** 2))),
                         "points_56_bit_identical": bool(np.array_equal(one["u_line_mls"][step_index], two["u_line_mls"][step_index])
                                                         and np.array_equal(one["v_line_mls"][step_index], two["v_line_mls"][step_index])),
                         "state_bit_identical_pid_order": hashes_one["pid_order"] == hashes_two["pid_order"],
                         "state_bit_identical_sorted": hashes_one["sorted"] == hashes_two["sorted"],
                         "kinematic_bit_identical_sorted": (hashes_one["kinematic_sorted"] == hashes_two["kinematic_sorted"]) if "kinematic_sorted" in hashes_one else None})
    return rows


def table() -> int:
    runs = load_runs("runs")
    rows = comparison_rows(runs, "runs")
    repeat_rows = comparison_rows(load_runs("repeat"), "repeat") if (OUTPUT_DIRECTORY / "repeat" / "b" / "run.json").exists() else []
    first = runs["a1"]["document"]
    b_run = runs["b"]["document"]
    steps = [int(value) for value in runs["a1"]["samples"]["steps"]]
    with open(OUTPUT_DIRECTORY / "equivalence_light.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + repeat_rows)
    document = {"rows": rows + repeat_rows, "seam": {key: b_run[key] for key in ("cuts", "cuts_x_physical", "cuts_x_reference", "slab_particles")},
                "checkpoint": first["checkpoint"], "solver": first["solver"], "steps": steps, "lid_speed": first["lid_speed"],
                "runs": {name: {key: run["document"][key] for key in ("slabs", "device_map", "device_uuids", "wall_seconds", "cuts", "slab_particles")} for name, run in runs.items()}}
    (OUTPUT_DIRECTORY / "equivalence_light.json").write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

    def value(row: dict, key: str) -> str:
        return "0" if row[key] == 0.0 else f"{row[key]:.2e}"

    def main_table(selected: list) -> list:
        by_key = {(row["comparison"], row["step"]): row for row in selected}
        out = ["| 步 | a1 对 a2 max\\|Δu\\|/U | a1 对 a2 max\\|Δv\\|/U | b 对 a1 max\\|Δu\\|/U | b 对 a1 max\\|Δv\\|/U |", "|---|---|---|---|---|"]
        for step in steps:
            floor, test = by_key[("a2-a1", step)], by_key[("b-a1", step)]
            out.append(f"| {step} | {value(floor, 'max_du_all56')} | {value(floor, 'max_dv_all56')} | {value(test, 'max_du_all56')} | {value(test, 'max_dv_all56')} |")
        return out

    def seam_table(selected: list) -> list:
        by_key = {(row["comparison"], row["step"]): row for row in selected}
        out = ["| 步 | 组 | 缝上 28 点 max\\|Δu\\|/U | 缝上 28 点 max\\|Δv\\|/U | 其余 27 点 max\\|Δu\\|/U | 其余 27 点 max\\|Δv\\|/U | 56 点最大差所在（Δu / Δv） |", "|---|---|---|---|---|---|---|"]
        for key, label in (("b-a1", "b 对 a1"), ("a2-a1", "a1 对 a2")):
            for step in steps:
                row = by_key[(key, step)]
                out.append(f"| {step} | {label} | {value(row, 'max_du_seam28')} | {value(row, 'max_dv_seam28')} | {value(row, 'max_du_off_seam27')} | {value(row, 'max_dv_off_seam27')} | {row['at_du']} / {row['at_dv']} |")
        return out

    checkpoint = first["checkpoint"]
    seam_x = b_run["cuts_x_physical"][0]
    lines = ["# 轻量版 K = 2 对 K = 1 等价性检查（2026-10-31）", "",
             f"算例 Re = 1000 方腔，1001² 流体粒子（共 {first['expected_total']:,} 粒子，常数同 5.3），每次从同一个 K = 1 检查点 `{pathlib.Path(checkpoint['file']).name}` 起跑 500 步"
             f"（第 {checkpoint['source_step']:,} 步，t = {checkpoint['source_time']:.4f}，sha256 {checkpoint['sha256'][:16]}…）。"
             "5.3 的 K = 1 运行只保留了最后两个检查点（t = 89.3025 与 99.225），t ≈ 80 的那个已不存在；定常判据在 t = 79.38 触发，这里用最近的 t = 89.3025，也是 K = 8 续算的起点。", "",
             f"求解器 `experiment/v7` 树 {first['solver']['experiment_v7_tree'][:12]}（= v7-rc4），未改；两张 RTX 5090（无 (c)）；depth 2；500 步内无周期 defrag。"
             f"瞬时速度，5.3 的 MLS 插值与 wall 坐标系，点 = 5.3 运行自己的 `points.npz`；差值除以 U = {first['lid_speed']:g}；两个分量都在全部 56 点上取。", "",
             f"K = 2 等权切口：第 {b_run['cuts'][0]} 列，x = {seam_x:+.6f} m（参考坐标 {b_run['cuts_x_reference'][0]:.5f}），离 x = 0.5 中线 {abs(seam_x) / first['spacing']:.2f} Δx = {abs(seam_x) / first['support_radius']:.2f} h，"
             f"两侧 {b_run['slab_particles'][0]:,} / {b_run['slab_particles'][1]:,} 粒子。u(0.5, y) 的 28 点都在缝上：每个点的 MLS 支撑（半径 h）跨过缝。", "",
             "## 1. 全部 56 点", ""] + main_table(rows) + ["",
             "a1 与 a2 在四个步上都不逐位相同（56 点与全状态都不同），所以噪声底不是 0。来源是 `initialize_voxelization` 用 atomicAdd 建体素列表，体素内粒子顺序每次运行不同，defrag 保留这个顺序，邻居求和的舍入因此不同。", "",
             "第 1 步这一行没有检验力：predict 只用检查点里的 v、a、δr 逐粒子更新，第 3 节的重复里三次运行第 1 步排序后的 (x, y, u, v) 逐位相同，表中 2–3e-16 是插值的 float64 舍入。", "",
             "## 2. 缝上 28 点对其余点", "",
             "其余 27 点 = v(x, 0.5) 线去掉与 u 线重合的中心点（它也在缝上）。", ""] + seam_table(rows) + [""]
    if repeat_rows:
        lines += ["## 3. 独立重复一遍（`repeat/`，同样三次运行）", ""] + main_table(repeat_rows) + ["", ""] + seam_table(repeat_rows) + [""]
        identical = [row for row in repeat_rows if row["step"] == 1]
        lines += ["第 1 步排序后的 (x, y, u, v)：" + "；".join(f"{row['comparison']} {'逐位相同' if row['kinematic_bit_identical_sorted'] else '不同'}" for row in identical) + "。", ""]
    lines += CONCLUSION
    (OUTPUT_DIRECTORY / "equivalence_light.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("all", "one", "table"))
    parser.add_argument("--run", choices=tuple(RUNS))
    parser.add_argument("--subdirectory", default="runs", choices=("runs", "repeat"))
    arguments = parser.parse_args()
    if arguments.command == "one":
        if arguments.run is None:
            parser.error("one needs --run")
        return run_one(arguments.run, OUTPUT_DIRECTORY / arguments.subdirectory / arguments.run)
    if arguments.command == "table":
        return table()
    return run_all(arguments.subdirectory)


if __name__ == "__main__":
    sys.exit(main())
