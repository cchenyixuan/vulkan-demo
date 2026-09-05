"""
gil_and_hop_probe.py — the two final suspects for the K>=3 collapse, measured
directly on the target machine.

TEST 1 (GIL): four threads each block in vkWaitSemaphores (2 s timeout) on
semaphores nobody signals. Wall time ~2 s => the cffi call releases the GIL
(waits run concurrently); ~8 s => the GIL is held across the call and every
worker wait in the transport serializes against all others AND the submit
thread — the collapse mechanism.

TEST 2 (semaphore hop latency): host-signal value v on timeline A; an
EMPTY pre-submitted GPU batch waits A>=v and signals B=v; host waits B>=v.
The measured round trip = host->GPU wake + GPU->host wake, i.e. TWO of the
sync hops the V5 transport chain crosses ~5 times per direction per frame.
Windows reference for one hop is tens of microseconds; if a hop costs
milliseconds here, an interior slab's two chained directions overflow the
Phase B hiding window and the collapse is pure sync-hop latency.

    source ~/swq/env.sh && python remote/gil_and_hop_probe.py [device_index]
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vulkan import *  # noqa: F401,F403,E402

from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5  # noqa: E402


def create_timeline(device, initial=0):
    type_info = VkSemaphoreTypeCreateInfo(
        semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE, initialValue=initial)
    return vkCreateSemaphore(
        device, VkSemaphoreCreateInfo(pNext=type_info), None)


def host_wait(device, semaphore, value, timeout_ns):
    info = VkSemaphoreWaitInfo(semaphoreCount=1, pSemaphores=[semaphore],
                               pValues=[value])
    return vkWaitSemaphores(device, info, timeout_ns)


def test_gil(ctx) -> None:
    print("[TEST 1] four concurrent 2s vkWaitSemaphores ...", flush=True)
    semaphores = [create_timeline(ctx.device) for _ in range(4)]

    def block(semaphore):
        host_wait(ctx.device, semaphore, 1, 2_000_000_000)   # times out at 2s

    threads = [threading.Thread(target=block, args=(s,)) for s in semaphores]
    t_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - t_start
    verdict = ("GIL RELEASED (concurrent)" if elapsed < 3.5
               else "GIL HELD (serialized!)")
    print(f"[TEST 1] wall={elapsed:.2f}s -> {verdict}", flush=True)
    for semaphore in semaphores:
        vkDestroySemaphore(ctx.device, semaphore, None)


def test_hop_latency(ctx, rounds=500) -> None:
    print(f"[TEST 2] host->GPU->host semaphore round trip x{rounds} ...",
          flush=True)
    timeline_a = create_timeline(ctx.device)
    timeline_b = create_timeline(ctx.device)

    wait_info = VkSemaphoreSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, semaphore=timeline_a,
        value=0, stageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    signal_info = VkSemaphoreSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, semaphore=timeline_b,
        value=0, stageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)

    samples_us = []
    for round_index in range(1, rounds + 1):
        wait_info.value = round_index
        signal_info.value = round_index
        submit = VkSubmitInfo2(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            waitSemaphoreInfoCount=1, pWaitSemaphoreInfos=[wait_info],
            commandBufferInfoCount=0,
            signalSemaphoreInfoCount=1, pSignalSemaphoreInfos=[signal_info])
        vkQueueSubmit2(ctx.transfer_queue, 1, [submit], VK_NULL_HANDLE)

        t_start = time.perf_counter_ns()
        vkSignalSemaphore(ctx.device, VkSemaphoreSignalInfo(
            semaphore=timeline_a, value=round_index))
        host_wait(ctx.device, timeline_b, round_index, 10**10)
        samples_us.append((time.perf_counter_ns() - t_start) / 1000.0)

    samples_us.sort()
    p50 = samples_us[len(samples_us) // 2]
    p90 = samples_us[int(len(samples_us) * 0.9)]
    print(f"[TEST 2] round trip (2 hops): p50={p50:.0f}us p90={p90:.0f}us "
          f"max={samples_us[-1]:.0f}us -> per hop ~{p50/2:.0f}us", flush=True)
    vkQueueWaitIdle(ctx.transfer_queue)
    vkDestroySemaphore(ctx.device, timeline_a, None)
    vkDestroySemaphore(ctx.device, timeline_b, None)


def main() -> int:
    device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ctx = VulkanContextV5.create(device_index=device_index,
                                 enable_validation=False,
                                 application_name="gil_hop_probe")
    try:
        test_gil(ctx)
        test_hop_latency(ctx)
    finally:
        ctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
