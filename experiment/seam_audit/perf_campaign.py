"""
perf_campaign.py — v5 vs v6 seam configurations, interleaved performance campaign.

One process per run. Each run does, on the same bootstrapped chain:
  1. fps phase   — ChainOrchestrator.run_pipelined(depth 2) with no GPU timers
                   attached: steady fps after the warmup (the production loop).
  2. anatomy     — BenchTimers (compute + transfer pools) attached, step cmd
                   buffers re-recorded, then synchronous depth-1 orchestrator
                   steps; every frame's per-kernel durations are read back.
                   Medians per sim: phase A/B/C, b->c gap, the three Phase C
                   band kernels (correction_boundary, density_boundary, force
                   band), install / append_departed, readback/upload DMA.
  3. bytes       — per link per frame: host-copy bytes (count-aware worker;
                   v6 keeps exact totals, v5 is sampled every frame through the
                   orchestrator's on_frame_done hook) and the DMA staging size.
  4. invariants  — drift (after a final defrag), GPU + host stamp errors, every
                   overflow_* counter; a run with any violation is invalid.

Driver: trials are interleaved (trial-major, then case, then configuration) so
slow drifts of the machine hit every configuration alike.

Usage:
    .venv/Scripts/python.exe experiment/seam_audit/perf_campaign.py \
        --out logs/seam_audit/perf_20261002 [--cases 2d_1m,2d_4m] [--trials 3]
    .venv/Scripts/python.exe experiment/seam_audit/perf_campaign.py --summarize-only --out ...
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

CASES = {
    # name: (case path, dimension, weights, warmup, total steps, anatomy frames)
    "2d_1m":     ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, "1,1", 1000, 7000, 400),
    "2d_4m":     ("cases/lid_driven_cavity_2d_4m/case.yaml", 2, "1,1", 1000, 4000, 300),
    "2d_16m":    ("cases/lid_driven_cavity_2d_16m/case.yaml", 2, "1,1", 500, 2500, 200),
    "3d_8m":     ("cases/cavity3d_8m/case.yaml", 3, "1,1", 300, 1300, 100),
    "2d_narrow": ("logs/two_hop_experiment/cases/cavity_2d_10k/case.yaml", 2, "1,1", 3000, 30000, 1000),
}

CONFIGS = {
    "v5":       ("v5", {}),
    "v6_k0_l1": ("v6", {"V6_KEEP_DEPARTED": "0", "V6_GHOST_LAYERS": "1"}),
    "v6_k1_l1": ("v6", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "1"}),
    "v6_k1_l2": ("v6", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}),
}

ANATOMY_KEYS = (
    "phase_a_us", "phase_b_us", "phase_c_us", "b_to_c_gap_us", "a_to_b_gap_us",
    "correction_interior_us", "density_deep_interior_us", "wall_extrapolate_us", "force_deep_interior_us",
    "install_leading_us", "install_trailing_us", "append_departed_us",
    "correction_boundary_us", "density_boundary_us", "density_copy_us", "force_us",
    # v7 E39 B1 (V7_FUSED_CORRECTION_DENSITY): one fused kernel per site instead of correction + density
    "correction_density_interior_us", "correction_density_boundary_us",
    "readback_leading_dma_us", "readback_trailing_dma_us",
    "upload_leading_dma_us", "upload_trailing_dma_us",
)


def production_environment(version: str, dimension: int) -> dict:
    prefix = "V5_" if version == "v5" else "V6_"
    return {
        "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
        prefix + "WORKER_COUNT_AWARE": "1",
        prefix + "GHOST_POOL_FACTOR": "0.25" if dimension == 2 else "1.0",
        prefix + "SPLIT_TRANSFER_QUEUES": "1",
        prefix + "CASCADE_FORCE": "1",
        prefix + "BAND_VOXEL_DISPATCH": "1",
    }


def load_solver(version: str):
    if version == "v5":
        from experiment.v5.utils.bench_v5 import BenchTimer, compute_durations, split_parity_ticks
        from experiment.v5.utils.case_loader_v5 import load_case_v5 as load_case
        from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5 as Orchestrator
        from experiment.v5.utils.partition_v5 import compute_chain_partition
        from experiment.v5.utils.simulator_v5 import SphSimulatorV5 as Simulator
        from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5 as Context
    else:
        from experiment.v6.utils.bench_v6 import BenchTimer, compute_durations, split_parity_ticks
        from experiment.v6.utils.case_loader_v6 import load_case_v6 as load_case
        from experiment.v6.utils.orchestrator_v6 import ChainOrchestratorV6 as Orchestrator
        from experiment.v6.utils.partition_v6 import compute_chain_partition
        from experiment.v6.utils.simulator_v6 import SphSimulatorV6 as Simulator
        from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6 as Context
    return dict(BenchTimer=BenchTimer, compute_durations=compute_durations,
                split_parity_ticks=split_parity_ticks, load_case=load_case,
                Orchestrator=Orchestrator, compute_chain_partition=compute_chain_partition,
                Simulator=Simulator, Context=Context)


# --------------------------------------------------------------------------- worker
def run_worker(args) -> int:
    solver = load_solver(args.version)
    global_case = solver["load_case"](args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    weights = [float(weight) for weight in args.weights.split(",")]
    chain = solver["compute_chain_partition"](global_case, weights, 1.2)
    device_map = [int(device) for device in args.device_map.split(",")]
    defrag_cadence = global_case.numerics.defrag_cadence
    contexts, sims = [], []
    result = {"version": args.version, "case": args.case, "weights": weights,
              "cuts": list(chain.cuts),
              "own_columns": [g.own_column_count for g in chain.geometry]}
    try:
        for index in range(len(weights)):
            contexts.append(solver["Context"].create(
                device_index=device_map[index % len(device_map)],
                application_name=f"seam_perf_s{index}"))
            sims.append(solver["Simulator"](contexts[-1], chain.slabs[index],
                                            sync_scheme="per-direction"))
        with solver["Orchestrator"](sims, defrag_cadence=defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            workers = list(getattr(orchestrator, "workers", []))
            sampled_bytes = {worker.label: [] for worker in workers}

            def sample_copy_bytes(frame_n, _sim_index):
                for worker in workers:
                    sampled_bytes[worker.label].append(getattr(worker, "last_copy_bytes", 0))
            orchestrator.on_frame_done = sample_copy_bytes

            # ---- 1. fps (production loop, no timers) ----------------------
            pipelined = orchestrator.run_pipelined(args.steps, depth=2, warmup=args.warmup)
            result["fps"] = pipelined.get("steady_fps", pipelined["fps"])
            result["steady_frames"] = pipelined.get("steady_frames")
            result["steady_s"] = pipelined.get("steady_s")
            orchestrator.on_frame_done = None
            links = {}
            for worker in workers:
                samples = [value for value in sampled_bytes[worker.label] if value > 0]
                exact_frames = getattr(worker, "copy_frame_count", 0)
                links[worker.label] = {
                    "host_copy_bytes_per_frame_sampled":
                        statistics.mean(samples) if samples else None,
                    "host_copy_bytes_per_frame_exact":
                        (worker.total_copy_bytes / exact_frames) if exact_frames else None,
                    "dma_bytes_per_frame": worker._source_view.nbytes,
                }
            result["links"] = links

            # ---- 2. anatomy (timers attached, re-recorded, depth 1) ----------
            timers = []
            for index, sim in enumerate(sims):
                bench = solver["BenchTimer"](sim.ctx, label=f"s{index}")
                bench_transfer = solver["BenchTimer"](
                    sim.ctx, label=f"s{index}_transfer",
                    queue_family_index=sim.ctx.transfer_queue_family_index)
                sim.bench = bench
                sim.bench_transfer = bench_transfer
                sim.prepare_step_cmd_buffers()
                timers.append((bench, bench_transfer))
            per_sim_samples = [dict() for _ in sims]
            for _ in range(args.anatomy_frames):
                record = orchestrator.step()
                frame_n = record["frame_n"]
                for index, (bench, bench_transfer) in enumerate(timers):
                    ticks = bench.read_frame(include_defrag=False)
                    ticks.update(bench_transfer.read_frame(include_defrag=False))
                    if bench.parity_regions:
                        ticks, _previous = solver["split_parity_ticks"](ticks, frame_n % 2)
                    durations = solver["compute_durations"](ticks)
                    for key in ANATOMY_KEYS:
                        if key in durations and durations[key] is not None:
                            per_sim_samples[index].setdefault(key, []).append(durations[key])
            result["anatomy"] = [
                {key: {"median": statistics.median(values), "p90": sorted(values)[int(0.9 * (len(values) - 1))],
                       "n": len(values)}
                 for key, values in samples.items()}
                for samples in per_sim_samples]

            # ---- 4. invariants ------------------------------------------------
            for sim in sims:
                sim.submit_defrag_and_wait()
            alive_total = 0
            stamp_errors_gpu = 0
            overflow = {}
            seam = []
            for sim in sims:
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                alive_total += status["alive_particle_count"]
                stamp_errors_gpu += status.get("stamp_error_count", 0)
                for name, value in status.items():
                    if name.startswith("overflow_"):
                        overflow[name] = overflow.get(name, 0) + value
                seam.append({"peak_departed_count": health.get("peak_departed_count"),
                             "departed_pool_size": health.get("departed_pool_size"),
                             "far_migration_count": status.get("far_migration_count"),
                             "peak_migration_count": health.get("peak_migration_count")})
            stamp_errors_host = sum(getattr(worker, "stamp_error_count", 0) for worker in workers)
            far_migration = sum((entry.get("far_migration_count") or 0) for entry in seam)
            result["invariants"] = {
                "drift": alive_total - expected_total, "stamp_errors_gpu": stamp_errors_gpu,
                "stamp_errors_host": stamp_errors_host, "overflow": overflow,
                "far_migration": far_migration,
                "valid": (alive_total == expected_total and stamp_errors_gpu == 0
                          and stamp_errors_host == 0 and not any(overflow.values())
                          and far_migration == 0)}
            result["seam"] = seam
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    print("[seam_perf] RESULT " + json.dumps(result), flush=True)
    return 0 if result["invariants"]["valid"] else 3


# --------------------------------------------------------------------------- driver
def run_driver(args) -> int:
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
    config_names = args.configs.split(",") if args.configs else list(CONFIGS)
    for trial in range(1, args.trials + 1):
        for case_name in case_names:
            case_path, dimension, weights, warmup, steps, anatomy_frames = CASES[case_name]
            for config_name in config_names:
                run_id = f"{case_name}/{config_name}/t{trial}"
                if run_id in done:
                    continue
                version, config_env = CONFIGS[config_name]
                environment = dict(os.environ)
                if version != "v5":
                    # v6 configurations name their switches on top of the pre-E6b
                    # defaults (as in the 2026-10-02 campaign); caller-set V6_* are kept
                    from experiment.v6.utils.partition_v6 import LEGACY_DEFAULTS
                    for key, value in LEGACY_DEFAULTS.items():
                        environment.setdefault(key, value)
                environment.update(production_environment(version, dimension))
                environment.update(config_env)
                command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
                           "--version", version, "--case", case_path, "--weights", weights,
                           "--device-map", args.device_map, "--warmup", str(warmup),
                           "--steps", str(steps), "--anatomy-frames", str(anatomy_frames)]
                log_path = out_dir / "logs" / (run_id.replace("/", "__") + ".log")
                started = time.time()
                print(f"[seam_perf] {run_id} ...", flush=True)
                try:
                    completed = subprocess.run(command, env=environment, capture_output=True,
                                               text=True, timeout=args.timeout, cwd=_REPO_ROOT)
                    output = completed.stdout + "\n" + completed.stderr
                    return_code = completed.returncode
                except subprocess.TimeoutExpired as error:
                    output = f"TIMEOUT after {args.timeout}s\n{error.stdout or ''}\n{error.stderr or ''}"
                    return_code = -1
                log_path.write_text(output if isinstance(output, str) else str(output), encoding="utf-8")
                result_line = [line for line in output.splitlines() if line.startswith("[seam_perf] RESULT ")]
                record = {"run_id": run_id, "case": case_name, "config": config_name, "trial": trial,
                          "return_code": return_code, "wall_s": round(time.time() - started, 1),
                          "ok": bool(result_line)}
                if result_line:
                    record["result"] = json.loads(result_line[-1][len("[seam_perf] RESULT "):])
                    print(f"[seam_perf]   fps={record['result']['fps']:.1f} "
                          f"valid={record['result']['invariants']['valid']} ({record['wall_s']} s)", flush=True)
                else:
                    print(f"[seam_perf]   FAILED rc={return_code}, see {log_path}", flush=True)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
    summarize(out_dir)
    return 0


def _mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def correction_band_cell(row: dict, fmt) -> str:
    """The correction band column: v7 E39 B1's fused band kernel (correction + density; the density column is
    then empty) when the run has no separate correction band kernel."""
    if row.get("correction_boundary_us") is None and row.get("correction_density_boundary_us") is not None:
        return f"{fmt(row['correction_density_boundary_us'])} (fused, + density)"
    return fmt(row.get("correction_boundary_us"))


def summarize(out_dir: pathlib.Path) -> None:
    records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record.get("ok")]
    summary = {}
    for record in records:
        key = (record["case"], record["config"])
        summary.setdefault(key, []).append(record["result"])
    lines = ["# Seam configurations — performance (interleaved trials)", "",
             "fps = steady fps of the production depth-2 loop (no timers). Phase C / band kernels = medians of",
             "per-frame GPU timestamps in a depth-1 instrumented phase of the same process, max over sims.",
             "Bytes per link per frame: host copy (count-aware worker, mean over links) and DMA staging size.", ""]
    header = ("| case | config | trials | valid | fps (mean ± std) | vs v5 | phase C µs | correction_bnd µs | "
              "density_bnd µs | force_bnd µs | append_departed µs | b→c gap µs | host copy KiB/link/frame | "
              "DMA KiB/link/frame |")
    lines += [header, "|" + "---|" * 14]
    json_rows = []
    case_names = sorted({case for case, _ in summary}, key=lambda name: list(CASES).index(name) if name in CASES else 99)
    for case_name in case_names:
        reference_fps, _ = _mean_std([result["fps"] for result in summary.get((case_name, "v5"), [])])
        for config_name in CONFIGS:
            results = summary.get((case_name, config_name))
            if not results:
                continue
            fps_mean, fps_std = _mean_std([result["fps"] for result in results])

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

            host_copy = link_mean("host_copy_bytes_per_frame_exact") or link_mean("host_copy_bytes_per_frame_sampled")
            dma = link_mean("dma_bytes_per_frame")
            valid = all(result["invariants"]["valid"] for result in results)
            ratio = (fps_mean / reference_fps) if (reference_fps and fps_mean) else None
            row = {"case": case_name, "config": config_name, "trials": len(results), "valid": valid,
                   "fps_mean": fps_mean, "fps_std": fps_std, "fps_vs_v5": ratio,
                   "phase_c_us": anatomy_max("phase_c_us"),
                   "correction_boundary_us": anatomy_max("correction_boundary_us"),
                   "density_boundary_us": anatomy_max("density_boundary_us"),
                   "correction_density_boundary_us": anatomy_max("correction_density_boundary_us"),
                   "force_band_us": anatomy_max("force_us"),
                   "append_departed_us": anatomy_max("append_departed_us"),
                   "b_to_c_gap_us": anatomy_max("b_to_c_gap_us"),
                   "host_copy_bytes_per_link_frame": host_copy, "dma_bytes_per_link_frame": dma}
            json_rows.append(row)

            def fmt(value, digits=1):
                return "—" if value is None else f"{value:.{digits}f}"
            lines.append(
                f"| {case_name} | {config_name} | {len(results)} | {'yes' if valid else '**NO**'} | "
                f"{fmt(fps_mean)} ± {fmt(fps_std)} | {fmt(100 * ratio, 2) + ' %' if ratio else '—'} | "
                f"{fmt(row['phase_c_us'])} | {correction_band_cell(row, fmt)} | {fmt(row['density_boundary_us'])} | "
                f"{fmt(row['force_band_us'])} | {fmt(row['append_departed_us'])} | {fmt(row['b_to_c_gap_us'])} | "
                f"{fmt(host_copy / 1024 if host_copy else None)} | {fmt(dma / 1024 if dma else None)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(json_rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--version", default="v5")
    parser.add_argument("--case", default=None)
    parser.add_argument("--weights", default="1,1")
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--anatomy-frames", type=int, default=300)
    parser.add_argument("--out", default="logs/seam_audit/perf")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--summarize-only", action="store_true")
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
    sys.exit(run_driver(arguments))
