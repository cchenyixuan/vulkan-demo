"""
k1_chain_vs_single.py — one-card throughput of the two v6 single-GPU forms (opt side item 五).

  single   the single-buffer path (_run_v6_single_bench.py): sim.bootstrap() + one defrag (as
           the chain's bootstrap_all does), one combined step command buffer (predict +
           update_voxel + correction_all + density_all + force_all), one vkQueueSubmit + fence
           wait per step, defrag at the case cadence.
  chain1   the K = 1 chain the seam audit uses as its reference: compute_chain_partition with
           one slab, ChainOrchestratorV6, phase A / B / C as three submits per step on the
           timeline scheme, run_pipelined with depth 1.
  chain2   the same chain with depth 2 (the production pipelined loop).

Same card for every run (default device 1 = the headless 5090 in v6's discrete-first order),
validation layers off, no GPU timers, trials interleaved (trial-major, then case, then form),
fps = steady frames / wall over the measured window after the warmup. Each run is its own
process. Drift (alive count vs the case) is checked at the end of every run.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.k1_chain_vs_single --out logs/seam_audit/opt/k1_chain_vs_single
    .venv/Scripts/python.exe -m experiment.seam_audit.k1_chain_vs_single --out ... --summarize-only
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

RESULT_PREFIX = "[k1_vs_single] RESULT "
# name: (case, warmup, measured steps)
CASES = {
    "1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 1000, 6000),
    "2m": ("cases/lid_driven_cavity_2d_2m/case.yaml", 1000, 4000),
    "4m": ("cases/lid_driven_cavity_2d_4m/case.yaml", 500, 3000),
    "8m": ("cases/lid_driven_cavity_2d_8m/case.yaml", 300, 2000),
    "16m": ("cases/lid_driven_cavity_2d_16m/case.yaml", 200, 1200),
    "32m": ("cases/lid_driven_cavity_2d_32m/case.yaml", 200, 800),
}
FORMS = ("single", "chain1", "chain2")


def run_worker(args) -> int:
    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6

    case = load_case_v6(args.case)
    expected_total = int(case.initial.positions.shape[0])
    defrag_cadence = case.numerics.defrag_cadence
    result = {"form": args.form, "case": args.case, "warmup": args.warmup, "steps": args.steps,
              "device": args.device}
    context = VulkanContextV6.create(device_index=args.device, application_name="k1_vs_single")
    result["device_name"] = getattr(context, "device_name", None)
    sims = []
    try:
        if args.form == "single":
            sim = SphSimulatorV6(context, case)
            sims.append(sim)
            sim.bootstrap()
            # same start as the chain (ChainOrchestratorV6.bootstrap_all ends with
            # one defrag): particles voxel-sorted before the first step
            sim.submit_defrag_and_wait()
            sim.prepare_step_single_cmd_buffer()
            frame_n = 0
            started = None
            while frame_n < args.warmup + args.steps:
                if frame_n == args.warmup:
                    started = time.perf_counter()
                sim.submit_step_single_and_wait()
                frame_n += 1
                if frame_n % defrag_cadence == 0:
                    sim.submit_defrag_and_wait()
            elapsed = time.perf_counter() - started
            result["fps"] = args.steps / elapsed
            sim.submit_defrag_and_wait()
            alive = sim.readback_global_status()["alive_particle_count"]
        else:
            from experiment.v6.utils.orchestrator_v6 import ChainOrchestratorV6
            from experiment.v6.utils.partition_v6 import compute_chain_partition
            chain = compute_chain_partition(case, [1.0], 1.2)
            sim = SphSimulatorV6(context, chain.slabs[0], sync_scheme="per-direction")
            sims.append(sim)
            depth = 1 if args.form == "chain1" else 2
            with ChainOrchestratorV6(sims, defrag_cadence=defrag_cadence) as orchestrator:
                orchestrator.bootstrap_all()
                pipelined = orchestrator.run_pipelined(args.steps + args.warmup, depth=depth,
                                                       warmup=args.warmup)
                result["fps"] = pipelined.get("steady_fps", pipelined["fps"])
                result["steady_frames"] = pipelined.get("steady_frames")
                sim.submit_defrag_and_wait()
                alive = sim.readback_global_status()["alive_particle_count"]
        result["drift"] = alive - expected_total
    finally:
        for sim in sims:
            sim.destroy()
        context.destroy()
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0 if result["drift"] == 0 else 3


def run_driver(args) -> int:
    out_dir = pathlib.Path(args.out).resolve()
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("ok"):
                done.add(record["run_id"])
    case_names = args.cases.split(",") if args.cases else list(CASES)
    forms = args.forms.split(",") if args.forms else list(FORMS)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    for trial in range(1, args.trials + 1):
        for case_name in case_names:
            case_path, warmup, steps = CASES[case_name]
            for form in forms:
                run_id = f"{case_name}/{form}/t{trial}"
                if run_id in done:
                    continue
                command = [sys.executable, "-m", "experiment.seam_audit.k1_chain_vs_single", "--worker",
                           "--form", form, "--case", case_path, "--warmup", str(warmup),
                           "--steps", str(steps), "--device", str(args.device)]
                print(f"[k1_vs_single] {run_id} ...", flush=True)
                started = time.time()
                try:
                    completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                               timeout=args.timeout, cwd=_REPO_ROOT)
                    output = completed.stdout + "\n" + completed.stderr
                    return_code = completed.returncode
                except subprocess.TimeoutExpired as error:
                    output = f"TIMEOUT\n{error.stdout or ''}\n{error.stderr or ''}"
                    return_code = -1
                (out_dir / "logs" / (run_id.replace("/", "__") + ".log")).write_text(str(output), encoding="utf-8")
                lines = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
                record = {"run_id": run_id, "case": case_name, "form": form, "trial": trial,
                          "return_code": return_code, "wall_s": round(time.time() - started, 1),
                          "ok": bool(lines)}
                if lines:
                    record["result"] = json.loads(lines[-1][len(RESULT_PREFIX):])
                    print(f"[k1_vs_single]   fps={record['result']['fps']:.1f} drift={record['result']['drift']}",
                          flush=True)
                else:
                    print(f"[k1_vs_single]   FAILED rc={return_code}", flush=True)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
    summarize(out_dir)
    return 0


def summarize(out_dir: pathlib.Path) -> None:
    records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record.get("ok")]
    by = {}
    for record in records:
        by.setdefault((record["case"], record["form"]), {})[record["trial"]] = record["result"]
    lines = ["# K = 1 chain vs single-buffer, one RTX 5090 (interleaved trials)", "",
             "| case | single fps | chain depth 1 fps | chain depth 2 fps | chain1 / single | chain2 / single | drift |",
             "|---|---|---|---|---|---|---|"]
    for case_name in CASES:
        if (case_name, "single") not in by:
            continue
        single = by[(case_name, "single")]

        def stats(form):
            runs = by.get((case_name, form), {})
            values = [run["fps"] for run in runs.values()]
            if not values:
                return "—", None
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ratios = [runs[trial]["fps"] / single[trial]["fps"] for trial in runs if trial in single]
            return f"{mean:,.1f} ± {std:.1f}", ratios

        single_text, _ = stats("single")
        chain1_text, chain1_ratios = stats("chain1")
        chain2_text, chain2_ratios = stats("chain2")

        def ratio_text(ratios):
            if not ratios:
                return "—"
            mean = statistics.mean(ratios)
            std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
            return f"{100 * mean:.2f} ± {100 * std:.2f} %"
        drifts = {run["drift"] for form in FORMS for run in by.get((case_name, form), {}).values()}
        lines.append(f"| {case_name} | {single_text} | {chain1_text} | {chain2_text} | "
                     f"{ratio_text(chain1_ratios)} | {ratio_text(chain2_ratios)} | {sorted(drifts)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="K=1 chain vs single-buffer throughput")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--form", choices=FORMS)
    parser.add_argument("--case")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--out", default="logs/seam_audit/opt/k1_chain_vs_single")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--forms", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if args.summarize_only:
        summarize(pathlib.Path(args.out).resolve())
        return 0
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
