"""
_run_single_bench.py — headless throughput runner of v7_single (one GPU, N steps in flight).

    .venv/Scripts/python.exe experiment/v7_single/_run_single_bench.py --case cases/lid_driven_cavity_2d_gen/case.yaml \\
        --gpu-uuid ae137c668a40f5acaab90a83f7cda175 --warmup 1000 --measure 2000 --in-flight 2

fps = measured steps / wall time, both ends taken with an empty queue (every fence waited), so the value is
exact at any in-flight depth; defrag at the case's cadence inside the window, appended to the step's
submission (in-flight loop) exactly as _run_single_baseline_bench.py's fence ring does for the other
solvers. The bootstrap defrag (one defrag after the bootstrap, before the first step, as the chain's
ChainOrchestratorV7.bootstrap_all does unconditionally) is run by default; --no-defrag disables only the
periodic one, --no-bootstrap-defrag the bootstrap one (the pre-2026-10-22 behaviour of this bench under
--no-defrag: the generator's particle order was then never voxel-sorted, 17 % slower on the 8M cube; E8 job
1722130). The final status block gives the drift (alive - loaded) and every overflow counter; both must be 0.
Prints one ``RESULT {json}`` line. --validation enables the Khronos validation layer (messages are counted and
reported). The device is named by its Vulkan deviceUUID (--gpu-uuid; nvidia-smi's uuid without "GPU-") or
by its index in the discrete-first order (--device).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def resolve_device_index(gpu_uuid: str) -> int:
    """Index of the card with this UUID in the discrete-first order (a short-lived instance, no device)."""
    from vulkan import (VK_MAKE_VERSION, VkApplicationInfo, VkInstanceCreateInfo, vkCreateInstance, vkDestroyInstance)
    from experiment.v7_single.utils.vulkan_context_single import device_uuid_hex, enumerate_devices
    instance = vkCreateInstance(VkInstanceCreateInfo(pApplicationInfo=VkApplicationInfo(
        pApplicationName="v7_single_probe", apiVersion=VK_MAKE_VERSION(1, 3, 0))), None)
    try:
        uuids = [device_uuid_hex(device) for device in enumerate_devices(instance)]
    finally:
        vkDestroyInstance(instance, None)
    wanted = gpu_uuid.replace("-", "").lower()
    if wanted not in uuids:
        raise SystemExit(f"no physical device with uuid {wanted}; present: {uuids}")
    return uuids.index(wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", type=int, default=None, help="index in the discrete-first order")
    parser.add_argument("--gpu-uuid", default=None, help="Vulkan deviceUUID of the card (preferred)")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=2000)
    parser.add_argument("--in-flight", type=int, default=1)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--no-defrag", action="store_true", help="no periodic defrag (the bootstrap defrag still runs)")
    parser.add_argument("--no-bootstrap-defrag", action="store_true",
                        help="skip the defrag after the bootstrap (the generator order is then kept until the first periodic defrag)")
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()
    if arguments.gpu_uuid:
        device_index = resolve_device_index(arguments.gpu_uuid)
    elif arguments.device is not None:
        device_index = arguments.device
    else:
        raise SystemExit("pass --gpu-uuid or --device")

    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7_single.utils.runner_single import run_steps
    from experiment.v7_single.utils.simulator_single import SphSimulatorSingle
    from experiment.v7_single.utils.vulkan_context_single import VALIDATION_MESSAGE_COUNTS, VulkanContextSingle

    case = load_case_v7(arguments.case)
    expected = int(case.initial.positions.shape[0])
    cadence = 0 if arguments.no_defrag else int(case.numerics.defrag_cadence)
    ctx = VulkanContextSingle.create(device_index=device_index, enable_validation=arguments.validation,
                                     application_name="sph_v7_single_bench")
    result = {"status": "error"}
    try:
        sim = SphSimulatorSingle(ctx, case)
        print(f"[bench_single] gpu={ctx.device_name} uuid={ctx.device_uuid} particles={expected:,} "
              f"in_flight={arguments.in_flight} warmup={arguments.warmup} measure={arguments.measure} "
              f"defrag_cadence={cadence} bootstrap_defrag={int(not arguments.no_bootstrap_defrag)}", flush=True)
        t0 = time.perf_counter()
        sim.bootstrap()
        sim.prepare_step(in_flight=arguments.in_flight)
        if not arguments.no_bootstrap_defrag:
            # as ChainOrchestratorV7.bootstrap_all: voxel-sort the generator's initial order once, pipeline idle
            t_defrag = time.perf_counter()
            sim.submit_defrag_and_wait()
            print(f"[bench_single] bootstrap defrag: {1e3 * (time.perf_counter() - t_defrag):.1f} ms", flush=True)
        setup_s = time.perf_counter() - t0
        total = arguments.warmup + arguments.measure
        epoch_start = epoch_end = None
        t_start = None
        if arguments.warmup == 0:                 # the queue is empty here: the window opens now
            t_start = time.perf_counter()
            epoch_start = time.time()
        for frame in range(1, total + 1):
            sim.submit_step(with_defrag=bool(cadence) and frame % cadence == 0)
            if frame == arguments.warmup:
                sim.drain()
                t_start = time.perf_counter()
                epoch_start = time.time()
        sim.drain()
        t_end = time.perf_counter()
        epoch_end = time.time()
        elapsed = t_end - t_start
        status = sim.readback_global_status()
        overflow = {name: value for name, value in status.items() if name.startswith("overflow_") and value}
        drift = int(status["alive_particle_count"]) - expected
        result = {"status": "ok" if drift == 0 and not overflow else "bad_state", "solver": "v7_single",
                  "in_flight": arguments.in_flight, "case": arguments.case, "tag": arguments.tag,
                  "device": device_index, "gpu": ctx.device_name, "gpu_uuid": ctx.device_uuid,
                  "particles": expected, "alive": int(status["alive_particle_count"]), "drift": drift,
                  "overflow": overflow, "correction_fallback": int(status["correction_fallback_count"]),
                  "warmup": arguments.warmup, "measure": arguments.measure, "defrag_cadence": cadence,
                  "bootstrap_defrag": not arguments.no_bootstrap_defrag,
                  "setup_seconds": round(setup_s, 2), "measure_seconds": round(elapsed, 4),
                  "fps": round(arguments.measure / elapsed, 4), "us_per_step": round(elapsed / arguments.measure * 1e6, 4),
                  "epoch_start": epoch_start, "epoch_end": epoch_end,
                  "validation": dict(VALIDATION_MESSAGE_COUNTS) if arguments.validation else None,
                  "validation_enabled": ctx.validation_enabled,
                  **sim.resolution_record()}
        print(f"[bench_single] fps={result['fps']:.2f} us/step={result['us_per_step']:.1f} alive={result['alive']:,} "
              f"drift={drift:+d} overflow={overflow or 0}", flush=True)
        if arguments.validation:
            print(f"[bench_single] validation layer messages: {VALIDATION_MESSAGE_COUNTS}", flush=True)
        sim.destroy()
    finally:
        ctx.destroy()
    print("RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
