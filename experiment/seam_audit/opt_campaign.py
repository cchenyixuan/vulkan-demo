"""
opt_campaign.py - v6 optimisation campaign: interleaved local 2-GPU performance
of v6 switch sets (docs/seam_audit/v6_opt.md).

One process per run (v6, K=2, device map 0,1). Each run, on one bootstrapped
chain:
  1. fps        the production depth-2 loop without GPU timers; steady fps
                after the warmup.
  2. bytes      per link per frame: DMA = the staging size (readback = upload),
                host copy = the count-aware worker's exact byte total / frames.
  3. anatomy    BenchTimers attached (compute + transfer pools), step cmds
                re-recorded, synchronous depth-1 steps: per-frame medians of
                phase A/B/C, b->c gap, the band kernels, install /
                append_departed / list expansion, and the transport chain t_tr
                per link = sender readback DMA + worker host memcpy
                (copy_ns - wait_ns of the worker's own timestamps) + receiver
                upload DMA.
  4. invariants drift after a final defrag, GPU + host stamp errors, every
                overflow_* counter, far_migration; all pool peaks
                (PoolHealth + GlobalStatus) recorded.

Configurations are switch sets on top of the production environment
(count-aware worker, split transfer queues, cascade force, band-voxel dispatch,
ghost pool factor 0.25 in 2-D / 1.0 in 3-D). The seam configuration is part of
the switch set, not added by the driver: the presets in CONFIGS that start from
(1,2) = V6_KEEP_DEPARTED=1 V6_GHOST_LAYERS=2 contain it, and a --define
configuration gets it only through '@l2' (name=@l2;KEY=VALUE;...); without it
the run is v5-equivalent (0,1).

Driver: trials interleaved (trial-major, then case, then configuration).

Usage:
    .venv/Scripts/python.exe experiment/seam_audit/opt_campaign.py --out logs/seam_audit/opt/perf_base \\
        --configs l2,l2_lean [--cases 2d_1m,3d_8m] [--trials 3]
    ... --summarize-only --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULT_PREFIX = "[opt_perf] RESULT "

CASES = {
    # name: (case path, dimension, warmup, total steps, anatomy frames)
    "2d_narrow": ("logs/two_hop_experiment/cases/cavity_2d_10k/case.yaml", 2, 3000, 30000, 1000),
    "2d_1m":     ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, 1000, 7000, 400),
    "2d_4m":     ("cases/lid_driven_cavity_2d_4m/case.yaml", 2, 1000, 4000, 300),
    "2d_16m":    ("cases/lid_driven_cavity_2d_16m/case.yaml", 2, 500, 2500, 200),
    "3d_8m":     ("cases/cavity3d_8m/case.yaml", 3, 300, 1300, 100),
    "3d_narrow": ("cases/cavity3d_narrow/case.yaml", 3, 300, 1800, 150),
}

SEAM_L2 = {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}
SEAM_L1 = {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "1"}
CONFIGS = {
    "v5eq": {"V6_KEEP_DEPARTED": "0", "V6_GHOST_LAYERS": "1"},
    "l1": dict(SEAM_L1),
    "l2": dict(SEAM_L2),
}

ANATOMY_KEYS = (
    "phase_a_us", "phase_b_us", "phase_c_us", "b_to_c_gap_us", "a_to_b_gap_us",
    "correction_interior_us", "density_deep_interior_us", "wall_extrapolate_us", "force_deep_interior_us",
    "install_leading_us", "install_trailing_us", "append_departed_us", "expand_lists_us",
    "band_compact_us", "correction_boundary_us", "density_boundary_us", "density_copy_us",
    "force_us", "readback_leading_dma_us", "readback_trailing_dma_us",
    "upload_leading_dma_us", "upload_trailing_dma_us", "install_chain_us",
)
# per-frame series kept in the result (E23 install timing: pooled over trials, median / p95 per direction)
INSTALL_KEYS = ("expand_lists_us", "install_leading_us", "install_trailing_us", "append_departed_us",
                "install_chain_us")
# host bytes per link and frame predicted from the count words (E23 formula; packed replicas 32 B for both
# layers, 44 B per migrant without V6_TRANSPORT_EXTENSION, the two voxel lists 2 x 4 B x 2 NyNz, three count
# words and the frame stamp 16 B)
FORMULA_REPLICA_BYTES = 32
FORMULA_MIGRANT_BYTES = 44


def production_environment(dimension: int) -> dict:
    """The pre-E6b defaults (partition_v6.LEGACY_DEFAULTS) plus the production
    switches: every preset and --define names its switches on top of these (so
    'l2' stays the (1,2) seam with every optimisation off, 'release' the set it
    lists)."""
    from experiment.v6.utils.partition_v6 import LEGACY_DEFAULTS
    return {
        **LEGACY_DEFAULTS,
        "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
        "V6_WORKER_COUNT_AWARE": "1",
        "V6_GHOST_POOL_FACTOR": "0.25" if dimension == 2 else "1.0",
        "V6_SPLIT_TRANSFER_QUEUES": "1",
        "V6_CASCADE_FORCE": "1",
        "V6_BAND_VOXEL_DISPATCH": "1",
    }


def resolve_by_dimension(environment: dict, dimension: int) -> dict:
    """A value written 'd2:X|d3:Y' takes X in 2-D cases and Y in 3-D cases
    (per-dimension pool factors in one configuration)."""
    resolved = {}
    for key, value in environment.items():
        if isinstance(value, str) and value.startswith("d2:"):
            choices = dict(part.split(":", 1) for part in value.split("|"))
            value = choices[f"d{dimension}"]
        resolved[key] = value
    return resolved


def parse_definitions(definitions) -> dict:
    """--define name=KEY=VALUE;KEY=VALUE ... (a definition may start from a
    preset: name=@l2;KEY=VALUE)."""
    configs = dict(CONFIGS)
    for definition in definitions or ():
        name, _, body = definition.partition("=")
        environment = {}
        for item in body.split(";"):
            item = item.strip()
            if not item:
                continue
            if item.startswith("@"):
                environment.update(configs[item[1:]])
                continue
            key, _, value = item.partition("=")
            environment[key.strip()] = value.strip()
        configs[name.strip()] = environment
    return configs


# --------------------------------------------------------------------------- worker
def build_provenance(root: pathlib.Path, solver_file: pathlib.Path) -> dict:
    def git(*arguments):
        try:
            completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None
    status = git("status", "--porcelain", "--", "experiment/v6")
    return {"root": str(root), "solver_file": str(solver_file), "head": git("rev-parse", "--short", "HEAD"),
            "experiment_v6_dirty": None if status is None else bool(status)}     # None: git failed


def run_worker(args) -> int:
    build_root = pathlib.Path(args.build_root).resolve() if args.build_root else _REPO_ROOT
    if args.build_root:
        # the solver of another tree (OPT_BUILD_ROOT, e.g. a git worktree of the pre-change build): its
        # experiment package first on sys.path, before the first experiment.* import of this process; the
        # cases and the measurement code stay those of this tree (cwd = this tree's root)
        sys.path.insert(0, str(build_root))
    from experiment.v6.utils.bench_v6 import BenchTimer, compute_durations, split_parity_ticks
    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.orchestrator_v6 import ChainOrchestratorV6
    from experiment.v6.utils.partition_v6 import compute_chain_partition
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6
    import experiment.v6.utils.simulator_v6 as simulator_module
    import struct

    solver_file = pathlib.Path(simulator_module.__file__).resolve()
    if build_root not in solver_file.parents:
        raise RuntimeError(f"the solver resolved to {solver_file}, not under the build root {build_root}")

    global_case = load_case_v6(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    chain = compute_chain_partition(global_case, [1.0, 1.0], 1.2)
    device_map = [int(device) for device in args.device_map.split(",")]
    defrag_cadence = global_case.numerics.defrag_cadence
    contexts, sims = [], []
    result = {"case": args.case, "cuts": list(chain.cuts),
              "own_columns": [g.own_column_count for g in chain.geometry],
              "switches": {key: value for key, value in sorted(os.environ.items())
                           if key.startswith("V6_")},
              "build": build_provenance(build_root, solver_file)}
    try:
        for index in range(2):
            contexts.append(VulkanContextV6.create(device_index=device_map[index],
                                                   application_name=f"opt_perf_s{index}"))
            sims.append(SphSimulatorV6(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        with ChainOrchestratorV6(sims, defrag_cadence=defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            workers = list(orchestrator.workers)

            # ---- 1. fps (production loop, no timers) --------------------------
            pipelined = orchestrator.run_pipelined(args.steps, depth=2, warmup=args.warmup)
            result["fps"] = pipelined.get("steady_fps", pipelined["fps"])
            result["steady_frames"] = pipelined.get("steady_frames")

            # ---- 2. bytes ------------------------------------------------------
            links = {}
            for worker in workers:
                frames = worker.copy_frame_count
                links[worker.label] = {
                    "host_copy_bytes_per_frame": (worker.total_copy_bytes / frames) if frames else None,
                    "dma_bytes_per_frame": worker.staging_bytes,
                }

            # ---- 3. anatomy (timers attached, re-recorded, depth 1) ------------
            timers = []
            for index, sim in enumerate(sims):
                bench = BenchTimer(sim.ctx, label=f"s{index}")
                bench_transfer = BenchTimer(sim.ctx, label=f"s{index}_transfer",
                                            queue_family_index=sim.ctx.transfer_queue_family_index)
                sim.bench = bench
                sim.bench_transfer = bench_transfer
                sim.prepare_step_cmd_buffers()
                timers.append((bench, bench_transfer))
            per_sim_samples = [dict() for _ in sims]
            host_copy_us = {worker.label: [] for worker in workers}
            # host-byte formula on the same frames: after a depth-1 step the sender staging still holds this
            # frame's count words (the next readback comes with the next step) and worker.last_copy_bytes is
            # this frame's count-aware copy
            formula = {}
            for worker in workers:
                segments = worker.source._transport_segments[worker.source_direction]
                offsets = {}
                for segment in segments:
                    region = getattr(segment, "region", None)
                    if region in ("inner", "outer", "migrant") and region not in offsets:
                        offsets[region] = segment.count_staging_offset
                grid = worker.source.case.grid
                formula[worker.label] = {"segments": segments, "offsets": offsets,
                                         "cross_section_voxels": grid.grid_dimension_y * grid.grid_dimension_z,
                                         "frames": 0, "n0": 0, "n1": 0, "n_mig": 0, "measured": 0,
                                         "predicted": 0, "segment_model": 0, "max_abs_difference": 0,
                                         "max_abs_segment_difference": 0}
            for _ in range(args.anatomy_frames):
                record = orchestrator.step()
                frame_n = record["frame_n"]
                for worker in workers:
                    plan = formula[worker.label]
                    view = worker.source.sender_staging_view(worker.source_direction)

                    def word(offset):
                        return struct.unpack_from("<I", view, offset)[0]
                    n0 = word(plan["offsets"]["inner"]) if "inner" in plan["offsets"] else 0
                    n1 = word(plan["offsets"]["outer"]) if "outer" in plan["offsets"] else 0
                    n_mig = word(plan["offsets"]["migrant"]) if "migrant" in plan["offsets"] else 0
                    predicted = (FORMULA_REPLICA_BYTES * (n0 + n1) + FORMULA_MIGRANT_BYTES * n_mig
                                 + 16 * plan["cross_section_voxels"] + 16)
                    segment_model = sum(min(segment.size, word(segment.count_staging_offset) * segment.stride)
                                        if segment.count_staging_offset is not None else segment.size
                                        for segment in plan["segments"])
                    measured = worker.last_copy_bytes
                    plan["frames"] += 1
                    plan["n0"] += n0
                    plan["n1"] += n1
                    plan["n_mig"] += n_mig
                    plan["measured"] += measured
                    plan["predicted"] += predicted
                    plan["segment_model"] += segment_model
                    plan["max_abs_difference"] = max(plan["max_abs_difference"], abs(measured - predicted))
                    plan["max_abs_segment_difference"] = max(plan["max_abs_segment_difference"],
                                                             abs(measured - segment_model))
                for index, (bench, bench_transfer) in enumerate(timers):
                    ticks = bench.read_frame(include_defrag=False)
                    ticks.update(bench_transfer.read_frame(include_defrag=False))
                    if bench.parity_regions:
                        ticks, _previous = split_parity_ticks(ticks, frame_n % 2)
                    durations = compute_durations(ticks)
                    if durations.get("expand_lists_us") is not None and durations.get("append_departed_us") is not None:
                        # = c_append_departed_end - c_start: expand + install of every inbound direction + append
                        durations["install_chain_us"] = (durations["expand_lists_us"] + durations["append_departed_us"]
                                                         + sum(durations.get(f"install_{direction}_us") or 0.0
                                                               for direction in ("leading", "trailing")))
                    for key in ANATOMY_KEYS:
                        if durations.get(key) is not None:
                            per_sim_samples[index].setdefault(key, []).append(durations[key])
                for worker in workers:
                    stamps = worker.timestamps_for_frame(frame_n)
                    if stamps:
                        host_copy_us[worker.label].append((stamps["copy_ns"] - stamps["wait_ns"]) / 1000.0)
            result["anatomy"] = [
                {key: {"median": statistics.median(values),
                       "p90": sorted(values)[int(0.9 * (len(values) - 1))],
                       "p95": sorted(values)[int(0.95 * (len(values) - 1))], "n": len(values)}
                 for key, values in samples.items()}
                for samples in per_sim_samples]
            result["anatomy_series"] = [{key: samples[key] for key in INSTALL_KEYS if key in samples}
                                        for samples in per_sim_samples]
            # t_tr per link: s0 -> s1 = s0 readback trailing + worker + s1 upload leading
            def median_of(index, key):
                entry = result["anatomy"][index].get(key)
                return entry["median"] if entry else None
            for worker in workers:
                source = 0 if worker.label == "s0_to_s1" else 1
                destination = 1 - source
                source_direction = "trailing" if source == 0 else "leading"
                destination_direction = "leading" if destination == 1 else "trailing"
                copies = host_copy_us[worker.label]
                links[worker.label].update({
                    "readback_dma_us": median_of(source, f"readback_{source_direction}_dma_us"),
                    "host_copy_us": statistics.median(copies) if copies else None,
                    "upload_dma_us": median_of(destination, f"upload_{destination_direction}_dma_us"),
                })
                plan = formula[worker.label]
                frames = plan["frames"]
                links[worker.label]["formula"] = {
                    "frames": frames, "cross_section_voxels": plan["cross_section_voxels"],
                    "n0_mean": plan["n0"] / frames if frames else None,
                    "n1_mean": plan["n1"] / frames if frames else None,
                    "n_mig_mean": plan["n_mig"] / frames if frames else None,
                    "measured_mean": plan["measured"] / frames if frames else None,
                    "predicted_mean": plan["predicted"] / frames if frames else None,
                    "segment_model_mean": plan["segment_model"] / frames if frames else None,
                    "max_abs_difference": plan["max_abs_difference"],
                    "max_abs_segment_difference": plan["max_abs_segment_difference"]}
            result["links"] = links

            # ---- 4. invariants + pool peaks ------------------------------------
            for sim in sims:
                sim.submit_defrag_and_wait()
            alive_total, stamp_errors_gpu, overflow = 0, 0, {}
            statuses, healths = [], []
            for sim in sims:
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                statuses.append(dict(status))
                healths.append(dict(health))
                alive_total += status["alive_particle_count"]
                stamp_errors_gpu += status.get("stamp_error_count", 0)
                for name, value in status.items():
                    if name.startswith("overflow_"):
                        overflow[name] = overflow.get(name, 0) + value
            stamp_errors_host = sum(worker.stamp_error_count for worker in workers)
            far_migration = sum(status.get("far_migration_count", 0) for status in statuses)
            result["global_status"] = statuses
            result["pool_health"] = healths
            result["capacities"] = [{
                "own_pool_size": sim.case.capacities.own_pool_size,
                "leading_ghost_pool_size": sim.case.capacities.leading_ghost_pool_size,
                "trailing_ghost_pool_size": sim.case.capacities.trailing_ghost_pool_size,
                "replica_region_size": sim.case.capacities.replica_region_size,
                "departed_pool_size": sim.case.capacities.departed_pool_size,
            } for sim in sims]
            result["invariants"] = {
                "drift": alive_total - expected_total, "stamp_errors_gpu": stamp_errors_gpu,
                "stamp_errors_host": stamp_errors_host, "overflow": overflow,
                "far_migration": far_migration,
                "valid": (alive_total == expected_total and stamp_errors_gpu == 0
                          and stamp_errors_host == 0 and not any(overflow.values())
                          and far_migration == 0)}
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0 if result["invariants"]["valid"] else 3


# --------------------------------------------------------------------------- driver
def run_driver(args, configs: dict) -> int:
    out_dir = pathlib.Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("ok"):
                done.add(record["run_id"])
    case_names = args.cases.split(",") if args.cases else list(CASES)
    config_names = args.configs.split(",")
    (out_dir / "configs.json").write_text(json.dumps({name: configs[name] for name in config_names},
                                                     indent=1), encoding="utf-8")
    for trial in range(1, args.trials + 1):
        for case_name in case_names:
            case_path, dimension, warmup, steps, anatomy_frames = CASES[case_name]
            # --counterbalance: even trials run the configurations in reverse
            # order, so no configuration always holds the first position
            order = list(reversed(config_names)) if (args.counterbalance and trial % 2 == 0) else config_names
            for config_name in order:
                run_id = f"{case_name}/{config_name}/t{trial}"
                if run_id in done:
                    continue
                config = dict(configs[config_name])
                # OPT_BUILD_ROOT (reserved key, not an environment variable): run this configuration's
                # solver from another tree (e.g. a git worktree of the pre-change build)
                build_root = config.pop("OPT_BUILD_ROOT", None)
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith("V6_") and key != "OPT_BUILD_ROOT"}
                environment.update(production_environment(dimension))
                environment.update(resolve_by_dimension(config, dimension))
                command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
                           "--case", case_path, "--device-map", args.device_map,
                           "--warmup", str(warmup), "--steps", str(steps),
                           "--anatomy-frames", str(anatomy_frames)]
                if build_root:
                    command += ["--build-root", str(pathlib.Path(build_root).resolve())]
                log_path = out_dir / "logs" / (run_id.replace("/", "__") + ".log")
                started = time.time()
                print(f"[opt_perf] {run_id} ...", flush=True)
                try:
                    completed = subprocess.run(command, env=environment, capture_output=True,
                                               text=True, timeout=args.timeout, cwd=_REPO_ROOT)
                    output = completed.stdout + "\n" + completed.stderr
                    return_code = completed.returncode
                except subprocess.TimeoutExpired as error:
                    output = f"TIMEOUT after {args.timeout}s\n{error.stdout or ''}\n{error.stderr or ''}"
                    return_code = -1
                log_path.write_text(output if isinstance(output, str) else str(output), encoding="utf-8")
                result_line = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
                record = {"run_id": run_id, "case": case_name, "config": config_name, "trial": trial,
                          "return_code": return_code, "wall_s": round(time.time() - started, 1),
                          "ok": bool(result_line), "build_root": build_root}
                if result_line:
                    record["result"] = json.loads(result_line[-1][len(RESULT_PREFIX):])
                    print(f"[opt_perf]   fps={record['result']['fps']:.1f} "
                          f"valid={record['result']['invariants']['valid']} ({record['wall_s']} s)", flush=True)
                else:
                    print(f"[opt_perf]   FAILED rc={return_code}, see {log_path}", flush=True)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
    summarize(out_dir, config_names)
    return 0


def _mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(out_dir: pathlib.Path, config_order=None) -> list:
    records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record.get("ok")]
    by_key = {}
    for record in records:
        by_key.setdefault((record["case"], record["config"]), []).append(record)
    if config_order is None:
        config_order = []
        for record in records:
            if record["config"] not in config_order:
                config_order.append(record["config"])
    lines = ["# v6 optimisation campaign (interleaved trials, local 2 x RTX 5090, K = 2)", "",
             "fps: production depth-2 loop, no timers. Phase C and kernels: medians of per-frame GPU timestamps",
             "(depth-1 instrumented phase), max over sims. Per link (mean over the two links): bytes per frame",
             "(DMA = staging size, host = count-aware live copy) and t_tr = readback DMA + host memcpy + upload DMA (µs).",
             "fps ratio: trial-wise against the first configuration listed.", ""]
    header = ("| case | config | trials | valid | fps | vs first | phase C µs | corr bnd | dens bnd | force bnd | "
              "append dep. | expand | DMA KiB | host KiB | readback µs | host µs | upload µs | t_tr µs |")
    lines += [header, "|" + "---|" * 18]
    rows = []
    for case_name in [name for name in CASES if any(key[0] == name for key in by_key)]:
        reference_by_trial = {record["trial"]: record["result"]["fps"]
                              for record in by_key.get((case_name, config_order[0]), [])}
        for config_name in config_order:
            group = by_key.get((case_name, config_name))
            if not group:
                continue
            results = [record["result"] for record in group]
            fps_mean, fps_std = _mean_std([result["fps"] for result in results])
            ratios = [record["result"]["fps"] / reference_by_trial[record["trial"]]
                      for record in group if record["trial"] in reference_by_trial]
            ratio_mean, ratio_std = _mean_std(ratios)

            def anatomy_max(key):
                per_trial = []
                for result in results:
                    values = [sim.get(key, {}).get("median") for sim in result.get("anatomy", [])]
                    values = [value for value in values if value is not None]
                    if values:
                        per_trial.append(max(values))
                return _mean_std(per_trial)[0]

            def link_mean(key):
                per_trial = []
                for result in results:
                    values = [link.get(key) for link in result.get("links", {}).values()]
                    values = [value for value in values if value is not None]
                    if values:
                        per_trial.append(statistics.mean(values))
                return _mean_std(per_trial)[0]

            row = {"case": case_name, "config": config_name, "trials": len(results),
                   "valid": all(result["invariants"]["valid"] for result in results),
                   "fps_mean": fps_mean, "fps_std": fps_std, "fps_ratio": ratio_mean,
                   "fps_ratio_std": ratio_std,
                   "phase_c_us": anatomy_max("phase_c_us"),
                   "correction_boundary_us": anatomy_max("correction_boundary_us"),
                   "density_boundary_us": anatomy_max("density_boundary_us"),
                   "force_band_us": anatomy_max("force_us"),
                   "append_departed_us": anatomy_max("append_departed_us"),
                   "expand_lists_us": anatomy_max("expand_lists_us"),
                   "band_compact_us": anatomy_max("band_compact_us"),
                   "dma_bytes": link_mean("dma_bytes_per_frame"),
                   "host_bytes": link_mean("host_copy_bytes_per_frame"),
                   "readback_us": link_mean("readback_dma_us"),
                   "host_copy_us": link_mean("host_copy_us"),
                   "upload_us": link_mean("upload_dma_us")}
            parts = [row[key] for key in ("readback_us", "host_copy_us", "upload_us")]
            row["t_tr_us"] = sum(parts) if all(part is not None for part in parts) else None
            rows.append(row)

            def fmt(value, digits=1):
                return "—" if value is None else f"{value:,.{digits}f}"
            lines.append(
                f"| {case_name} | {config_name} | {len(results)} | {'yes' if row['valid'] else '**NO**'} | "
                f"{fmt(fps_mean)} ± {fmt(fps_std)} | "
                f"{fmt(100 * ratio_mean, 2) + ' %' if ratio_mean else '—'} | {fmt(row['phase_c_us'])} | "
                f"{fmt(row['correction_boundary_us'])} | {fmt(row['density_boundary_us'])} | "
                f"{fmt(row['force_band_us'])} | {fmt(row['append_departed_us'])} | {fmt(row['expand_lists_us'])} | "
                f"{fmt(row['dma_bytes'] / 1024 if row['dma_bytes'] else None)} | "
                f"{fmt(row['host_bytes'] / 1024 if row['host_bytes'] else None)} | {fmt(row['readback_us'])} | "
                f"{fmt(row['host_copy_us'])} | {fmt(row['upload_us'])} | {fmt(row['t_tr_us'])} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", default=None)
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--anatomy-frames", type=int, default=300)
    parser.add_argument("--build-root", default=None,
                        help="worker: import the solver from this tree (set per configuration with the reserved key "
                             "OPT_BUILD_ROOT=<path> in --define)")
    parser.add_argument("--out", default="logs/seam_audit/opt/perf")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--configs", default="l2")
    parser.add_argument("--define", action="append", default=[],
                        help="extra configuration: name=KEY=VALUE;KEY=VALUE (or name=@preset;KEY=VALUE)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--counterbalance", action="store_true",
                        help="reverse the configuration order on even trials (the 2026-10-03 campaigns used the "
                             "fixed order; their sub-1 %% ratios are not resolved against position effects)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cp1252 consoles
    arguments = parse_args()
    if arguments.worker:
        sys.exit(run_worker(arguments))
    if arguments.summarize_only:
        summarize(pathlib.Path(arguments.out))
        sys.exit(0)
    sys.exit(run_driver(arguments, parse_definitions(arguments.define)))
