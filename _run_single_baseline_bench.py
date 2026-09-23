"""
_run_single_baseline_bench.py — headless single-GPU throughput bench used for
the paper's single-GPU baseline (V0 reference solver vs. the multi-GPU V5
solver running in single-GPU mode).

Solvers (``--solver``):
  v0  the reference single-GPU solver: utils/sph/SphSimulator, the code path
      behind _run_viewer.py minus rendering. It has no headless runner of its
      own, so this script is its benchmark entry point.
  v5  the multi-GPU solver experiment/v5/SphSimulatorV5 in single-GPU mode
      (one sim, no peer, combined step command buffer, no ghost flow).

Frames in flight (``--in-flight``):
  1   the solver's native loop: one submit + one fence wait per step
      (V0: SphSimulator.step(); V5: submit_step_single_and_wait()). The CPU
      round trip between fence wakeup and the next submit is exposed as a
      GPU idle bubble on every step.
  N>1 a fence ring over the same pre-recorded SIMULTANEOUS_USE step command
      buffer: step k+1 is submitted before step k has finished, so the GPU
      queue never drains between steps (the multi-GPU orchestrator's depth-N
      pipelining, applied to a single GPU). The solvers are not modified.

Metric: steady wall-clock throughput = measure_steps / (t_end - t_start).
Both timestamps are taken with an EMPTY GPU queue (every fence waited), so
the value is exact at any in-flight depth. No GPU timestamps, no validation
layer. Defrag runs at the case's cadence inside the measured window, the
same way for both solvers. The final alive count is compared with the loaded
particle count (drift must be 0).

Prints one line ``RESULT {json}`` on stdout.

Usage:
    .venv/Scripts/python.exe _run_single_baseline_bench.py --solver v0
        --case cases/lid_driven_cavity_2d_2m/case.yaml --device 1
        --warmup 1000 --measure 6000 --in-flight 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

UINT64_MAX = 0xFFFFFFFFFFFFFFFF


# ============================================================================
# OBJ parse cache (load time only; never inside the timed window)
# ============================================================================


def install_obj_cache(cache_dir: pathlib.Path) -> None:
    """Cache parsed OBJ vertex arrays as .npy, keyed by path + size + mtime.

    Both solvers parse OBJ files line by line in pure Python; the 32M case's
    1 GB domain.obj takes minutes per run. Caching removes that from every
    trial. Loading finishes before the timed window starts, so the metric
    is untouched."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(parse):
        def wrapper(path):
            resolved = pathlib.Path(path).resolve()
            stat = resolved.stat()
            key = hashlib.sha1(
                f"{resolved}|{stat.st_size}|{int(stat.st_mtime)}".encode()
            ).hexdigest()[:16]
            cache_path = cache_dir / f"{resolved.parent.name}_{resolved.stem}_{key}.npy"
            if cache_path.exists():
                return np.load(cache_path)
            vertices = parse(resolved)
            np.save(cache_path, vertices)
            return vertices
        return wrapper

    import utils.sph.case as v0_case_module
    v0_case_module.load_obj_vertices = cached(v0_case_module.load_obj_vertices)
    import experiment.v5.utils.case_loader_v5 as v5_loader_module
    v5_loader_module._parse_obj_vertices = cached(v5_loader_module._parse_obj_vertices)


# ============================================================================
# Device selection by UUID
# ============================================================================


def resolve_device_index(gpu_uuid_hex: str, solver: str) -> int:
    """Map a GPU's Vulkan deviceUUID to the index each context expects.

    V0's VulkanContext indexes the raw vkEnumeratePhysicalDevices order
    (on this rig: 5090, iGPU, 5090); V5's VulkanContextV5 re-sorts discrete
    GPUs first with a stable sort (5090, 5090, iGPU). The raw order is not
    stable across reboots, so the campaign pins the card by UUID (the same
    UUID nvidia-smi reports, without the "GPU-" prefix)."""
    from vulkan import (VK_MAKE_VERSION, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                        VkApplicationInfo, VkInstanceCreateInfo,
                        VkPhysicalDeviceIDProperties,
                        VkPhysicalDeviceProperties2, vkCreateInstance,
                        vkDestroyInstance, vkEnumeratePhysicalDevices,
                        vkGetInstanceProcAddr)
    application_info = VkApplicationInfo(
        pApplicationName="baseline_bench_probe", applicationVersion=1,
        pEngineName="baseline_bench_probe", engineVersion=1,
        apiVersion=VK_MAKE_VERSION(1, 1, 0))
    instance = vkCreateInstance(VkInstanceCreateInfo(
        pApplicationInfo=application_info, enabledExtensionCount=1,
        ppEnabledExtensionNames=["VK_KHR_get_physical_device_properties2"]), None)
    entries: list[tuple[int, str, bool]] = []
    try:
        get_properties2 = vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceProperties2KHR")
        for raw_index, physical_device in enumerate(
                vkEnumeratePhysicalDevices(instance)):
            id_properties = VkPhysicalDeviceIDProperties()
            properties2 = VkPhysicalDeviceProperties2(pNext=id_properties)
            get_properties2(physical_device, properties2)
            is_discrete = (properties2.properties.deviceType
                           == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            entries.append((raw_index, bytes(id_properties.deviceUUID).hex(),
                            is_discrete))
    finally:
        vkDestroyInstance(instance, None)

    wanted = gpu_uuid_hex.replace("-", "").lower()
    matches = [entry for entry in entries if entry[1] == wanted]
    if not matches:
        raise SystemExit(f"no physical device with uuid {wanted}; present: "
                         f"{[entry[1] for entry in entries]}")
    raw_index = matches[0][0]
    if solver == "v0":
        return raw_index
    discrete_first = sorted(entries, key=lambda entry: 0 if entry[2] else 1)
    return [entry[0] for entry in discrete_first].index(raw_index)


def resolve_device_index_in_subprocess(gpu_uuid_hex: str, solver: str) -> int:
    """Run resolve_device_index in a child process so the bench process
    holds exactly one VkInstance (the solver's own) for its whole life."""
    import subprocess
    completed = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         "--resolve-only", "--solver", solver, "--gpu-uuid", gpu_uuid_hex],
        capture_output=True, text=True, cwd=str(_REPO))
    if completed.returncode != 0:
        raise SystemExit("device resolution failed:\n" + completed.stdout
                         + completed.stderr)
    return int(completed.stdout.strip().splitlines()[-1])


# ============================================================================
# Fence ring: N frames in flight over one SIMULTANEOUS_USE command buffer
# ============================================================================


class FenceRing:
    """Keeps up to ``depth`` submissions in flight on one queue.

    Slot k's fence guards submission k, k+depth, k+2*depth, ...; before
    reusing a slot we wait for the submission ``depth`` frames back, so at
    most ``depth`` steps are queued on the GPU at any time."""

    def __init__(self, device, queue, depth: int):
        from vulkan import (VK_FENCE_CREATE_SIGNALED_BIT,
                            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                            VkFenceCreateInfo, vkCreateFence)
        self.device = device
        self.queue = queue
        self.depth = depth
        self.fences = [
            vkCreateFence(device, VkFenceCreateInfo(
                sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                flags=VK_FENCE_CREATE_SIGNALED_BIT), None)
            for _ in range(depth)
        ]
        self.submitted_count = 0

    def submit(self, command_buffers: list) -> None:
        from vulkan import (VK_STRUCTURE_TYPE_SUBMIT_INFO, VK_TRUE,
                            VkSubmitInfo, vkQueueSubmit, vkResetFences,
                            vkWaitForFences)
        slot = self.submitted_count % self.depth
        fence = self.fences[slot]
        vkWaitForFences(self.device, 1, [fence], VK_TRUE, UINT64_MAX)
        vkResetFences(self.device, 1, [fence])
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=len(command_buffers),
            pCommandBuffers=command_buffers,
        )
        vkQueueSubmit(self.queue, 1, submit_info, fence)
        self.submitted_count += 1

    def drain(self) -> None:
        from vulkan import VK_TRUE, vkWaitForFences
        vkWaitForFences(self.device, len(self.fences), self.fences, VK_TRUE,
                        UINT64_MAX)

    def destroy(self) -> None:
        from vulkan import vkDestroyFence
        for fence in self.fences:
            vkDestroyFence(self.device, fence, None)
        self.fences = []


# ============================================================================
# Solver adapters
# ============================================================================


class V0Adapter:
    """utils/sph/SphSimulator (reference single-GPU solver)."""

    name = "v0"
    native_step_includes_defrag = True

    def __init__(self, case_path: str, device_index: int):
        from utils.sph.case import load_case
        from utils.sph.simulator import SphSimulator
        from utils.sph.vulkan_context import VulkanContext

        t0 = time.perf_counter()
        self.case = load_case(case_path)
        self.load_seconds = time.perf_counter() - t0
        self.expected_particle_count = sum(
            int(source.vertices.shape[0]) for source in self.case.particle_sources)

        t0 = time.perf_counter()
        self.ctx = VulkanContext.create(
            application_name="sph_v0_single_baseline_bench",
            enable_validation=False,
            device_index=device_index,
        )
        self.sim = SphSimulator(self.ctx, self.case)
        self.sim.bootstrap()
        self.setup_seconds = time.perf_counter() - t0

        numerics = self.case.numerics
        self.defrag_cadence = (numerics.defrag_cadence
                               if numerics.defrag_enabled else 0)
        self.device = self.ctx.device
        self.queue = self.ctx.compute_queue
        self.physical_device = self.ctx.physical_device
        self.step_cmd = self.sim.step_cmd
        self.defrag_cmd = self.sim.defrag_cmd

    def native_step(self) -> None:
        self.sim.step()             # submit + fence wait; defrag at cadence

    def native_defrag(self) -> None:
        raise RuntimeError("V0 step() already runs defrag at cadence")

    def readback_status(self) -> dict:
        return self.sim.readback_global_status()

    def destroy(self) -> None:
        self.sim.destroy()
        self.ctx.destroy()


class V5Adapter:
    """experiment/v5/SphSimulatorV5 in single-GPU mode (no peer)."""

    name = "v5"
    native_step_includes_defrag = False

    def __init__(self, case_path: str, device_index: int):
        from experiment.v5.utils.case_loader_v5 import load_case_v5
        from experiment.v5.utils.simulator_v5 import SphSimulatorV5
        from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

        t0 = time.perf_counter()
        self.case = load_case_v5(case_path)
        self.load_seconds = time.perf_counter() - t0
        self.expected_particle_count = int(self.case.initial.positions.shape[0])

        t0 = time.perf_counter()
        self.ctx = VulkanContextV5.create(
            application_name="sph_v5_single_baseline_bench",
            enable_validation=False,
            device_index=device_index,
        )
        self.sim = SphSimulatorV5(self.ctx, self.case)   # sim.bench stays None
        self.sim.bootstrap()
        self.sim.prepare_step_single_cmd_buffer()
        if self.sim.defrag_cmd is None:
            self.sim.defrag_cmd = self.sim._record_defrag_cmd()
        self.setup_seconds = time.perf_counter() - t0

        self.defrag_cadence = int(self.case.numerics.defrag_cadence)
        self.device = self.ctx.device
        self.queue = self.ctx.compute_queue
        self.physical_device = self.ctx.physical_device
        self.step_cmd = self.sim.step_single_cmd
        self.defrag_cmd = self.sim.defrag_cmd

    def native_step(self) -> None:
        self.sim.submit_step_single_and_wait()

    def native_defrag(self) -> None:
        self.sim.submit_defrag_and_wait()

    def readback_status(self) -> dict:
        return self.sim.readback_global_status()

    def destroy(self) -> None:
        self.sim.destroy()
        self.ctx.destroy()


ADAPTERS = {"v0": V0Adapter, "v5": V5Adapter}


# ============================================================================
# Timed loop
# ============================================================================


def run_timed(adapter, warmup: int, measure: int, in_flight: int) -> dict:
    if warmup < 1:
        raise ValueError("--warmup must be >= 1 so t_start is defined")
    total = warmup + measure
    cadence = adapter.defrag_cadence
    t_start = t_end = None
    epoch_start = epoch_end = None

    if in_flight == 1:
        for frame in range(1, total + 1):
            adapter.native_step()
            if (cadence and not adapter.native_step_includes_defrag
                    and frame % cadence == 0):
                adapter.native_defrag()
            if frame == warmup:
                t_start = time.perf_counter()
                epoch_start = time.time()
        t_end = time.perf_counter()
        epoch_end = time.time()
    else:
        ring = FenceRing(adapter.device, adapter.queue, in_flight)
        try:
            for frame in range(1, total + 1):
                command_buffers = [adapter.step_cmd]
                if cadence and frame % cadence == 0:
                    command_buffers.append(adapter.defrag_cmd)
                ring.submit(command_buffers)
                if frame == warmup:
                    ring.drain()
                    t_start = time.perf_counter()
                    epoch_start = time.time()
            ring.drain()
            t_end = time.perf_counter()
            epoch_end = time.time()
        finally:
            ring.destroy()

    elapsed = t_end - t_start
    return {
        "measure_seconds": elapsed,
        "fps": measure / elapsed,
        "us_per_step": elapsed / measure * 1e6,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
    }


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="single-GPU baseline bench: V0 reference vs V5 single mode")
    parser.add_argument("--solver", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--resolve-only", action="store_true",
                        help="print the --gpu-uuid device index for --solver and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="physical device index in the SOLVER's own enumeration "
                             "order (V0 raw, V5 discrete-first); prefer --gpu-uuid")
    parser.add_argument("--gpu-uuid", type=str, default=None,
                        help="pin the card by Vulkan deviceUUID (nvidia-smi uuid "
                             "without the GPU- prefix); resolves --device per solver")
    parser.add_argument("--expect-gpu", type=str, default=None,
                        help="abort unless the selected device name contains this")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="steps before the timed window (>= 1)")
    parser.add_argument("--measure", type=int, default=2000,
                        help="steps inside the timed window")
    parser.add_argument("--in-flight", type=int, default=1,
                        help="frames in flight: 1 = native submit+wait loop, N>1 = fence ring")
    parser.add_argument("--obj-cache", type=str, default=None,
                        help="directory for cached OBJ vertex arrays (.npy)")
    parser.add_argument("--tag", type=str, default="",
                        help="free-form label copied into the RESULT record")
    parser.add_argument("--trial", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resolve_only:
        if not args.gpu_uuid:
            raise SystemExit("--resolve-only needs --gpu-uuid")
        print(resolve_device_index(args.gpu_uuid, args.solver))
        return 0
    if args.case is None:
        raise SystemExit("--case is required")
    if args.in_flight < 1:
        raise SystemExit("--in-flight must be >= 1")
    if args.obj_cache:
        install_obj_cache(pathlib.Path(args.obj_cache))

    from vulkan import vkGetPhysicalDeviceProperties

    if args.gpu_uuid:
        device_index = resolve_device_index_in_subprocess(args.gpu_uuid, args.solver)
        print(f"[baseline_bench] uuid {args.gpu_uuid} -> {args.solver} device index "
              f"{device_index}", flush=True)
    elif args.device is not None:
        device_index = args.device
    else:
        raise SystemExit("pass --gpu-uuid (preferred) or --device")

    print(f"[baseline_bench] solver={args.solver} in_flight={args.in_flight} "
          f"device={device_index} case={args.case} warmup={args.warmup} "
          f"measure={args.measure}", flush=True)

    adapter = ADAPTERS[args.solver](args.case, device_index)
    result = {"status": "error"}
    try:
        properties = vkGetPhysicalDeviceProperties(adapter.physical_device)
        gpu_name = str(properties.deviceName)
        print(f"[baseline_bench] gpu={gpu_name}  particles="
              f"{adapter.expected_particle_count:,}  load={adapter.load_seconds:.1f}s "
              f"setup={adapter.setup_seconds:.1f}s  defrag_cadence={adapter.defrag_cadence}",
              flush=True)
        if args.expect_gpu and args.expect_gpu not in gpu_name:
            raise SystemExit(f"selected device '{gpu_name}' does not contain "
                             f"'{args.expect_gpu}'; refusing to measure")

        timing = run_timed(adapter, args.warmup, args.measure, args.in_flight)
        status = adapter.readback_status()
        alive = int(status["alive_particle_count"])
        drift = alive - adapter.expected_particle_count
        overflow = (int(status.get("overflow_inside_count", 0))
                    + int(status.get("overflow_incoming_count", 0)))

        result = {
            "status": "ok" if (drift == 0 and overflow == 0) else "bad_state",
            "solver": args.solver,
            "in_flight": args.in_flight,
            "case": args.case,
            "tag": args.tag,
            "trial": args.trial,
            "device": device_index,
            "gpu_uuid": args.gpu_uuid,
            "gpu": gpu_name,
            "particles": adapter.expected_particle_count,
            "alive": alive,
            "drift": drift,
            "overflow_inside": int(status.get("overflow_inside_count", 0)),
            "overflow_incoming": int(status.get("overflow_incoming_count", 0)),
            "correction_fallback": int(status.get("correction_fallback_count", 0)),
            "warmup": args.warmup,
            "measure": args.measure,
            "defrag_cadence": adapter.defrag_cadence,
            "load_seconds": round(adapter.load_seconds, 2),
            "setup_seconds": round(adapter.setup_seconds, 2),
            **{key: (round(value, 4) if isinstance(value, float) else value)
               for key, value in timing.items()},
        }
        print(f"[baseline_bench] fps={result['fps']:.2f}  "
              f"us/step={result['us_per_step']:.1f}  alive={alive:,} "
              f"drift={drift:+d} overflow={overflow}", flush=True)
    finally:
        adapter.destroy()

    print("RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
