"""
soak.py - claude/wall_store_report.md section 11.2: soak runs of wall_boundary crossing on the cavity runner's code
path (experiment/validation/cavity_runner.py: ChainOrchestratorV7 with one slab, the release switch set, depth-2
pipelined submission, the defrag hook's status readbacks, a light-state readback every 7,000 steps) but without
samples, snapshots or checkpoints, and with a host-side watchdog.

    .venv/Scripts/python.exe -m experiment.wall_crossing.soak run --case <crossing case.yaml> --device 1 \
        --frames 200000 --label n250_k4 --env V7_WALL_CROSSING_NEAR_WALL_LANES=4 --out <directory>
    .venv/Scripts/python.exe -m experiment.wall_crossing.soak batch --plan <plan.json> --device 1 --out <directory> \
        --hours 5

Watchdog: every frame's completion is awaited in 1 s slices. A frame not done after --stall-seconds is a stall; the
watchdog then writes (once, and again at the give-up) the semaphore autopsy (vkGetSemaphoreCounterValue, never
blocks), the last submitted frame and the frames in flight, the last 64 frames' submit / completion times, three
nvidia-smi samples of the card (utilization, clocks, power, P-state, temperature, throttle reasons) and a
responsiveness probe: an empty command buffer with a fence on the slab's transfer queue (idle with one slab; not
behind the stuck compute work) and on the compute queue (behind it). A frame that completes after a stall is logged
as a recovery and the run continues; one that is not done after --give-up-seconds ends the process (exit code 2,
os._exit: a wedged device can hang the teardown).

run    one run; appends one JSON line to <out>/runs.jsonl (frames, wall time, rate, the largest gap between two
       frame completions, invariant checks, stalls with their diagnostics).
batch  the runs of a plan (a JSON list of {"case", "frames", "label", "env"}) in turn, each in its own process, cycling
       through the plan until --hours have passed (the run in progress finishes); a failed run is recorded and the
       batch goes on after 30 s.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
LIGHT_STATE_STEPS = 7000        # the cavity runner's sample interval at 250^2 (its light-state readback)


class Stall(RuntimeError):
    pass


def nvidia_smi_samples(uuid: str, count: int = 3) -> list:
    rows = []
    for _ in range(count):
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id=GPU-{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}",
                 "--query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,pstate,"
                 "temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15)
            rows.append(result.stdout.strip() or result.stderr.strip())
        except Exception as error:          # noqa: BLE001 - a diagnostic must not raise
            rows.append(f"nvidia-smi failed: {error!r}")
        time.sleep(1.0)
    return rows


def queue_probe(context, queue, pool, timeout_s: float) -> str:
    """An empty command buffer with a fence on `queue`: 'completed in x s' or 'not completed within t s'."""
    from vulkan import (VK_COMMAND_BUFFER_LEVEL_PRIMARY, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo,
                        VkFenceCreateInfo, VkSubmitInfo, vkAllocateCommandBuffers, vkBeginCommandBuffer,
                        vkCreateFence, vkEndCommandBuffer, vkGetFenceStatus, vkQueueSubmit)
    try:
        command_buffer = vkAllocateCommandBuffers(context.device, VkCommandBufferAllocateInfo(
            commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
        vkEndCommandBuffer(command_buffer)
        fence = vkCreateFence(context.device, VkFenceCreateInfo(), None)
        started = time.perf_counter()
        vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[command_buffer])], fence)
        while time.perf_counter() - started < timeout_s:
            try:
                vkGetFenceStatus(context.device, fence)
                return f"completed in {time.perf_counter() - started:.3f} s"
            except Exception as status:        # VkNotReady while pending; anything else is reported
                if type(status).__name__ != "VkNotReady":
                    return f"fence status error: {status!r}"
            time.sleep(0.01)
        return f"not completed within {timeout_s:.0f} s"
    except Exception as error:                  # noqa: BLE001
        return f"probe failed: {error!r}"


def run(arguments) -> int:
    from experiment.validation.cavity_runner import CachedStateReader, expected_environments
    for name, value in expected_environments("v7")["release"].items():
        os.environ[name] = value
    for item in arguments.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    os.chdir(REPOSITORY_ROOT)
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    case = load_case_v7(arguments.case)
    chain = compute_chain_partition(case, [1.0], 1.2)
    expected_total = int(chain.slabs[0].initial.positions.shape[0])
    cadence = int(case.numerics.defrag_cadence)
    context = VulkanContextV7.create(device_index=arguments.device, enable_validation=False,
                                     application_name=f"soak_{arguments.label}")
    from vulkan import VkPhysicalDeviceIDProperties, VkPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2
    identity = VkPhysicalDeviceIDProperties()
    vkGetPhysicalDeviceProperties2(context.physical_device, VkPhysicalDeviceProperties2(pNext=identity))
    uuid = bytes(identity.deviceUUID).hex()
    sim = SphSimulatorV7(context, chain.slabs[0], sync_scheme="per-direction")
    orchestrator = ChainOrchestratorV7([sim], defrag_cadence=cadence)
    orchestrator.bootstrap_all()
    reader = CachedStateReader(sim)
    record = {"label": arguments.label, "case": arguments.case, "device": arguments.device, "uuid": uuid,
              "environment": arguments.env, "frames_planned": arguments.frames, "lanes": sim._near_wall_lanes(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stalls": [], "status": "running"}
    recent = collections.deque(maxlen=64)       # (frame, submitted s, completed s) relative to the start
    submitted_at = {}
    started = time.perf_counter()
    last_completion = [started]
    largest_gap = [0.0, -1]

    def diagnostics(frame: int, waited: float, final: bool) -> dict:
        return {"frame": frame, "waited_s": round(waited, 1), "final": final,
                "last_submitted": max(submitted_at) if submitted_at else None,
                "in_flight": sorted(submitted_at),
                "autopsy": orchestrator.dump_stall_autopsy(frame),
                "recent": [(f, round(s, 4), round(c, 4)) for f, s, c in recent],
                "nvidia_smi": nvidia_smi_samples(uuid),
                "probe_transfer_queue": queue_probe(context, context.transfer_queue, context.transfer_command_pool, 10.0),
                "probe_compute_queue": queue_probe(context, context.compute_queue, context.command_pool, 3.0),
                "sync_state_after_probes": str(sim.sync_state())}

    def wait_frame(frame: int) -> None:
        semaphore, target = sim.sync.frame_done_op(frame)
        waiting_since = time.perf_counter()
        stall = None
        while not sim.wait_semaphore(semaphore, target, timeout_ns=int(1e9)):
            waited = time.perf_counter() - waiting_since
            if stall is None and waited > arguments.stall_seconds:
                stall = diagnostics(frame, waited, False)
                print(f"[soak] STALL frame {frame} after {waited:.1f} s: {json.dumps(stall)}", flush=True)
            if waited > arguments.give_up_seconds:
                final = diagnostics(frame, waited, True)
                record["stalls"].append({"first": stall, "final": final, "recovered": False})
                raise Stall(f"frame {frame} not done after {waited:.0f} s")
        now = time.perf_counter()
        if stall is not None:
            stall["recovered_after_s"] = round(now - waiting_since, 1)
            record["stalls"].append({"first": stall, "recovered": True})
            print(f"[soak] frame {frame} recovered after {now - waiting_since:.1f} s", flush=True)
        gap = now - last_completion[0]
        if frame > 0 and gap > largest_gap[0]:      # frame 0's wait includes the start-up
            largest_gap[0], largest_gap[1] = gap, frame
        last_completion[0] = now
        recent.append((frame, submitted_at.pop(frame) - started, now - started))

    def check_invariants(frame: int) -> None:
        report = orchestrator._collect_defrag_report()
        bad = {key: value for key, value in report[0].items()
               if (key.startswith("overflow_") or key == "far_migration") and value}
        status = sim.readback_global_status()
        bad.update({key: value for key, value in status.items()
                    if (key.startswith("overflow_") or key in ("far_migration_count", "stamp_error_count")) and value})
        if bad:
            raise RuntimeError(f"invariant violation at frame {frame}: {bad}")
        if frame % LIGHT_STATE_STEPS == 0:
            alive = int(reader.read()["mass"].size)
            if alive != expected_total:
                raise RuntimeError(f"drift at frame {frame}: alive {alive} != {expected_total}")

    if arguments.test_diagnostics:
        # check the stall diagnostics themselves once (frame 0 not yet submitted: the autopsy shows the bootstrap state)
        print(f"[soak] diagnostics test: {json.dumps(diagnostics(0, 0.0, False))}", flush=True)
    depth = 2
    submitted = completed = 0
    try:
        while submitted < arguments.frames:
            submitted_at[submitted] = time.perf_counter()
            orchestrator._submit_frame(submitted)
            submitted += 1
            orchestrator._frame_count = submitted
            while submitted - completed >= depth:
                wait_frame(completed)
                completed += 1
            if submitted % cadence == 0:
                while completed < submitted:
                    wait_frame(completed)
                    completed += 1
                check_invariants(submitted)
                sim.submit_defrag_and_wait()
            if submitted % 100000 == 0:
                print(f"[soak] {arguments.label}: frame {submitted:,} at "
                      f"{submitted / (time.perf_counter() - started):.0f} frames/s", flush=True)
        while completed < submitted:
            wait_frame(completed)
            completed += 1
        record["status"] = "ok"
    except Stall as error:
        record["status"] = f"stall: {error}"
    except Exception as error:                  # noqa: BLE001 - recorded, the batch decides
        record["status"] = f"error: {error!r}"
    elapsed = time.perf_counter() - started
    record.update({"frames_done": completed, "wall_s": round(elapsed, 1),
                   "frames_per_s": round(completed / elapsed, 1) if elapsed > 0 else 0.0,
                   "largest_completion_gap_s": round(largest_gap[0], 3), "largest_gap_frame": largest_gap[1],
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(out / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"[soak] {arguments.label}: {record['status']}, {completed:,} frames in {elapsed:.0f} s, largest gap "
          f"{largest_gap[0]:.3f} s at frame {largest_gap[1]}", flush=True)
    if record["status"].startswith("stall"):
        sys.stdout.flush()
        os._exit(2)
    try:
        sim.destroy()
        context.destroy()
    except Exception:                           # noqa: BLE001
        pass
    return 0 if record["status"] == "ok" else 1


def batch(arguments) -> int:
    plan = json.loads(pathlib.Path(arguments.plan).read_text(encoding="utf-8"))
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + arguments.hours * 3600.0
    sequence = 0
    while time.time() < deadline:
        for entry in plan:
            if time.time() >= deadline:
                break
            sequence += 1
            log = out / f"{sequence:04d}_{entry['label']}.log"
            command = [sys.executable, "-m", "experiment.wall_crossing.soak", "run", "--case", entry["case"],
                       "--device", str(arguments.device), "--frames", str(entry["frames"]), "--label", entry["label"],
                       "--out", str(out), "--stall-seconds", str(arguments.stall_seconds),
                       "--give-up-seconds", str(arguments.give_up_seconds)]
            for item in entry.get("env", []):
                command += ["--env", item]
            with open(log, "w", encoding="utf-8") as handle:
                code = subprocess.call(command, cwd=REPOSITORY_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            print(f"[soak batch] {time.strftime('%H:%M:%S')} #{sequence} {entry['label']}: exit {code}", flush=True)
            if code != 0:
                time.sleep(30)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "batch"):
        command = sub.add_parser(name)
        command.add_argument("--device", type=int, required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--stall-seconds", type=float, default=10.0)
        command.add_argument("--give-up-seconds", type=float, default=300.0)
    run_parser = sub.choices["run"]
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--frames", type=int, required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--env", action="append", default=[])
    run_parser.add_argument("--test-diagnostics", action="store_true",
                            help="run the stall diagnostics once before the first frame (to check them)")
    batch_parser = sub.choices["batch"]
    batch_parser.add_argument("--plan", required=True)
    batch_parser.add_argument("--hours", type=float, required=True)
    arguments = parser.parse_args()
    return run(arguments) if arguments.action == "run" else batch(arguments)


if __name__ == "__main__":
    sys.exit(main())
