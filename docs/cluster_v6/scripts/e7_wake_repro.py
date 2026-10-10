"""
e7_wake_repro.py — standalone reproduction of the late host wake-ups on a timeline semaphore (E7 batch 1: the
transport worker's dest guard woke ~10 ms late when a second thread waited on the same value). No solver code.

Same binding and threading as the solver: the `vulkan` package (cffi), the raw `vulkan._vulkan.lib.vkWaitSemaphores`
with fresh small structs per call (simulator_v7's V7_FAST_SUBMIT path), waiters on threading.Thread, the GIL switch
interval at 0.2 ms (the chain bench's default), vkQueueSubmit2 on the dedicated transfer queue (the readback queue).

One device (--device, discrete-first order as in the solver), one timeline semaphore. Per iteration the waiter
threads are released together (threading.Barrier), each records its start on the host clock and waits for the
iteration's value; after a random delay (--pre-signal-us) the main thread signals it, either from the host
(vkSignalSemaphore) or from the GPU (a vkQueueSubmit2 whose command buffer copies --copy-bytes device -> host and
writes a bottom-of-pipe timestamp, then signals the value). The GPU signal time is that timestamp on the host clock
(VK_KHR/EXT_calibrated_timestamps pairs, least squares over the narrowest pairs). A wait is late when it returns more
than 5 ms after the signal; only waits that began before the signal count as blocked.

Wait methods (the candidate fixes are measured in the same program):
  infinite          vkWaitSemaphores(UINT64_MAX) in every waiter (the solver today)
  timeout:<ms>      vkWaitSemaphores with that timeout in a loop (B1)
  relay_event       waiter 0 waits on the semaphore, then sets a per-iteration threading.Event the others wait on (B2)
  relay_condition   the same with a threading.Condition and a host counter (B2)
  zeropoll / counterpoll  waiter 0 sleeps (infinite); the others poll every 100 us with a zero-timeout
                    vkWaitSemaphores / vkGetSemaphoreCounterValue (can a non-blocking check steal the sleeper's wake?)
Patterns: same (all waiters on one value), mixed (the solver's transport timeline: the GPU signals odd values, waiter 0
host-signals the next even value after it wakes), split (each waiter on its own semaphore, both signalled by one
submit), distinct (waiter k waits its own value of one semaphore; one GPU signal reaches all of them), cowait (a
compute-queue batch also waits the value on the GPU, as the upload would under a GPU-side dest guard). --stagger-us
delays waiter k's start by k x stagger (the solver's reverse worker usually waits first).
A load thread (--load-duty) emulates the orchestrator's main loop: Python work in 1 ms windows with that duty cycle;
its work units per busy second, against the case without waiters, is the GIL cost of the waiters.

Usage:
    python docs/cluster_v6/scripts/e7_wake_repro.py --device 0 --out DIR [--cases all|basic|fixes|extras|NAME,...]
        [--iterations 4000] [--copy-bytes 1048576] [--pre-signal-us 200:3000] [--load-duty 0.5]
Writes DIR/waits.csv (one row per wait), DIR/cases.json (per case and waiter), DIR/summary.txt, DIR/environment.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import statistics
import sys
import threading
import time

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi
from vulkan._vulkan import lib

INFINITE_TIMEOUT = 0xFFFFFFFFFFFFFFFF
VK_SUCCESS_CODE = 0
VK_TIMEOUT_CODE = 2
LATE_NANOSECONDS = 5_000_000
TIME_DOMAIN_DEVICE = 0
TIME_DOMAIN_CLOCK_MONOTONIC = 1
TIME_DOMAIN_CLOCK_MONOTONIC_RAW = 2
TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER = 3
TIME_DOMAIN_NAMES = {TIME_DOMAIN_CLOCK_MONOTONIC: "CLOCK_MONOTONIC",
                     TIME_DOMAIN_CLOCK_MONOTONIC_RAW: "CLOCK_MONOTONIC_RAW",
                     TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER: "QUERY_PERFORMANCE_COUNTER"}
CALIBRATED_EXTENSIONS = ("VK_KHR_calibrated_timestamps", "VK_EXT_calibrated_timestamps")
CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE_EXT = 1000184000
QUERY_RING = 64


# ----------------------------------------------------------------------------- cases

def case_list(selection: str) -> list:
    basic = []
    for load in (False, True):
        for signal in ("host", "gpu"):
            for waiters in (1, 2, 4):
                basic.append({"name": f"{signal}_w{waiters}_infinite" + ("_load" if load else ""), "signal": signal,
                              "waiters": waiters, "method": "infinite", "pattern": "same", "load": load})
    extras = [
        {"name": "gpu_w2_infinite_mixed_load", "signal": "gpu", "waiters": 2, "method": "infinite", "pattern": "mixed",
         "load": True},
        {"name": "gpu_w2_infinite_split_load", "signal": "gpu", "waiters": 2, "method": "infinite", "pattern": "split",
         "load": True},
        {"name": "gpu_w2_infinite_stagger_load", "signal": "gpu", "waiters": 2, "method": "infinite", "pattern": "same",
         "load": True, "stagger_us": 500},
        {"name": "gpucompute_w2_infinite_load", "signal": "gpu", "queue": "compute", "waiters": 2, "method": "infinite",
         "pattern": "same", "load": True},
        {"name": "gpu_w2_infinite_longwait_load", "signal": "gpu", "waiters": 2, "method": "infinite", "pattern": "same",
         "load": True, "pre_signal_us": (5000.0, 25000.0), "iterations": 1000},
        {"name": "gpu_w1_infinite_cowait_load", "signal": "gpu", "waiters": 1, "method": "infinite", "pattern": "cowait",
         "load": True},
        {"name": "gpu_w2_infinite_cowait_load", "signal": "gpu", "waiters": 2, "method": "infinite", "pattern": "cowait",
         "load": True},
        {"name": "gpu_w2_infinite_distinct_load", "signal": "gpu", "waiters": 2, "method": "infinite",
         "pattern": "distinct", "load": True},
        {"name": "gpu_w4_infinite_distinct_load", "signal": "gpu", "waiters": 4, "method": "infinite",
         "pattern": "distinct", "load": True},
    ]
    fixes = []
    for poll in ("zeropoll", "counterpoll"):
        fixes.append({"name": f"gpu_w2_{poll}_load", "signal": "gpu", "waiters": 2, "method": poll, "pattern": "same",
                      "load": True})
    for waiters in (2, 4):
        for timeout_ms in ("0.1", "0.2", "0.5"):
            fixes.append({"name": f"gpu_w{waiters}_timeout{timeout_ms}_load", "signal": "gpu", "waiters": waiters,
                          "method": f"timeout:{timeout_ms}", "pattern": "same", "load": True})
        for relay in ("relay_event", "relay_condition"):
            fixes.append({"name": f"gpu_w{waiters}_{relay}_load", "signal": "gpu", "waiters": waiters,
                          "method": relay, "pattern": "same", "load": True})
    baseline = [{"name": "load_only", "signal": "host", "waiters": 0, "method": "infinite", "pattern": "same",
                 "load": True}]
    baseline_end = [dict(baseline[0], name="load_only_end")]
    groups = {"basic": basic, "extras": extras, "fixes": fixes, "baseline": baseline}
    if selection == "all":
        return baseline + basic + extras + fixes + baseline_end
    chosen = []
    for token in selection.split(","):
        if token in groups:
            chosen.extend(groups[token])
        else:
            matches = [case for case in baseline + basic + extras + fixes + baseline_end if case["name"] == token]
            if not matches:
                raise SystemExit(f"unknown case {token!r}")
            chosen.extend(matches)
    return chosen


# ----------------------------------------------------------------------------- device

class Device:
    """Instance, one physical device (discrete-first order), a compute queue and the dedicated transfer queue,
    timeline semaphores, a copy pair and a timestamp query pool."""

    def __init__(self, device_index: int, copy_bytes: int):
        application_info = VkApplicationInfo(pApplicationName="e7_wake_repro", applicationVersion=1,
                                             pEngineName="e7_wake_repro", engineVersion=1,
                                             apiVersion=VK_MAKE_VERSION(1, 3, 0))
        self.instance = vkCreateInstance(VkInstanceCreateInfo(pApplicationInfo=application_info), None)
        physical_devices = sorted(vkEnumeratePhysicalDevices(self.instance),
                                  key=lambda handle: 0 if vkGetPhysicalDeviceProperties(handle).deviceType
                                  == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1)
        self.physical_device = physical_devices[device_index]
        properties = vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = properties.deviceName
        self.timestamp_period_ns = float(properties.limits.timestampPeriod)
        self.driver_version_raw = int(properties.driverVersion)
        families = vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        self.compute_family = next(index for index, family in enumerate(families)
                                   if family.queueFlags & VK_QUEUE_COMPUTE_BIT)
        self.transfer_family = next((index for index, family in enumerate(families)
                                     if family.queueFlags & VK_QUEUE_TRANSFER_BIT
                                     and not family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)),
                                    self.compute_family)
        self.timestamp_bits = {"compute": int(families[self.compute_family].timestampValidBits),
                               "transfer": int(families[self.transfer_family].timestampValidBits)}
        available = {extension.extensionName for extension in
                     vkEnumerateDeviceExtensionProperties(self.physical_device, None)}
        self.calibrated_extension = next((name for name in CALIBRATED_EXTENSIONS if name in available), None)
        if self.calibrated_extension is None:
            raise SystemExit("no calibrated timestamps extension: the GPU signal time cannot be placed on the host clock")
        self.suffix = "KHR" if self.calibrated_extension.startswith("VK_KHR") else "EXT"
        queue_infos = [VkDeviceQueueCreateInfo(queueFamilyIndex=self.compute_family, queueCount=1,
                                               pQueuePriorities=[1.0])]
        if self.transfer_family != self.compute_family:
            queue_infos.append(VkDeviceQueueCreateInfo(queueFamilyIndex=self.transfer_family, queueCount=1,
                                                       pQueuePriorities=[1.0]))
        features_1_3 = VkPhysicalDeviceVulkan13Features(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                        synchronization2=VK_TRUE)
        features_1_2 = VkPhysicalDeviceVulkan12Features(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                        timelineSemaphore=VK_TRUE, hostQueryReset=VK_TRUE,
                                                        pNext=features_1_3)
        features_2 = VkPhysicalDeviceFeatures2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                               features=VkPhysicalDeviceFeatures(), pNext=features_1_2)
        self.device = vkCreateDevice(self.physical_device, VkDeviceCreateInfo(
            pNext=features_2, queueCreateInfoCount=len(queue_infos), pQueueCreateInfos=queue_infos,
            enabledExtensionCount=1, ppEnabledExtensionNames=[self.calibrated_extension], pEnabledFeatures=None),
            None)
        self.queues = {"compute": vkGetDeviceQueue(self.device, self.compute_family, 0),
                       "transfer": vkGetDeviceQueue(self.device, self.transfer_family, 0)}
        self.pools = {name: vkCreateCommandPool(self.device, VkCommandPoolCreateInfo(
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=family), None)
            for name, family in (("compute", self.compute_family), ("transfer", self.transfer_family))}
        self.copy_bytes = copy_bytes
        self.source_buffer, self.source_memory = self._buffer(copy_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.target_buffer, self.target_memory = self._buffer(copy_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        self.query_pool = vkCreateQueryPool(self.device, VkQueryPoolCreateInfo(queryType=VK_QUERY_TYPE_TIMESTAMP,
                                                                               queryCount=QUERY_RING), None)
        self.command_buffers = {name: self._record_ring(name) for name in ("compute", "transfer")}
        self.get_calibrated = vkGetDeviceProcAddr(self.device, f"vkGetCalibratedTimestamps{self.suffix}")
        domains_function = vkGetInstanceProcAddr(self.instance,
                                                 f"vkGetPhysicalDeviceCalibrateableTimeDomains{self.suffix}")
        domains = {int(value) for value in domains_function(self.physical_device)}
        order = ((TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER,) if sys.platform == "win32"
                 else (TIME_DOMAIN_CLOCK_MONOTONIC_RAW, TIME_DOMAIN_CLOCK_MONOTONIC))
        self.domain = next(domain for domain in order if domain in domains)
        self.host_clock_name, self.host_clock = host_clock_for(self.domain)
        self._qpc_frequency = None
        if self.domain == TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER:
            import ctypes
            frequency = ctypes.c_int64()
            ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(frequency))
            self._qpc_frequency = int(frequency.value)
        structure_type = (CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE_EXT if self.suffix == "EXT"
                          else VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR)
        self._calibration_infos = [VkCalibratedTimestampInfoKHR(sType=structure_type, timeDomain=TIME_DOMAIN_DEVICE),
                                   VkCalibratedTimestampInfoKHR(sType=structure_type, timeDomain=self.domain)]
        self.driver_name, self.driver_info = self._driver_properties()

    def _driver_properties(self):
        driver = VkPhysicalDeviceDriverProperties(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES)
        properties2 = VkPhysicalDeviceProperties2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=driver)
        try:
            vkGetPhysicalDeviceProperties2(self.physical_device, properties2)
            return ffi.string(driver.driverName).decode(), ffi.string(driver.driverInfo).decode()
        except Exception as error:                     # noqa: BLE001
            return "unknown", repr(error)

    def _buffer(self, size: int, usage: int, flags: int):
        buffer = vkCreateBuffer(self.device, VkBufferCreateInfo(size=size, usage=usage,
                                                                sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
        requirements = vkGetBufferMemoryRequirements(self.device, buffer)
        memory_properties = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        type_index = next(index for index in range(memory_properties.memoryTypeCount)
                          if requirements.memoryTypeBits & (1 << index)
                          and (memory_properties.memoryTypes[index].propertyFlags & flags) == flags)
        memory = vkAllocateMemory(self.device, VkMemoryAllocateInfo(allocationSize=requirements.size,
                                                                    memoryTypeIndex=type_index), None)
        vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    def _record_ring(self, queue_name: str) -> list:
        buffers = vkAllocateCommandBuffers(self.device, VkCommandBufferAllocateInfo(
            commandPool=self.pools[queue_name], level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=QUERY_RING))
        for slot, command_buffer in enumerate(buffers):
            vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
            if self.copy_bytes > 0:
                vkCmdCopyBuffer(command_buffer, self.source_buffer, self.target_buffer, 1,
                                [VkBufferCopy(srcOffset=0, dstOffset=0, size=self.copy_bytes)])
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.query_pool, slot)
            vkEndCommandBuffer(command_buffer)
        return list(buffers)

    def timeline(self):
        type_info = VkSemaphoreTypeCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                              semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE, initialValue=0)
        return vkCreateSemaphore(self.device, VkSemaphoreCreateInfo(pNext=type_info), None)

    def submit_signal(self, queue_name: str, slot: int, signals: list) -> None:
        """One batch: the slot's command buffer, then signal every (semaphore, value)."""
        vkResetQueryPool(self.device, self.query_pool, slot, 1)
        infos = [VkSemaphoreSubmitInfo(semaphore=semaphore, value=value,
                                       stageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) for semaphore, value in signals]
        submit = VkSubmitInfo2(commandBufferInfoCount=1,
                               pCommandBufferInfos=[VkCommandBufferSubmitInfo(
                                   commandBuffer=self.command_buffers[queue_name][slot])],
                               signalSemaphoreInfoCount=len(infos), pSignalSemaphoreInfos=infos)
        vkQueueSubmit2(self.queues[queue_name], 1, [submit], VK_NULL_HANDLE)

    def submit_wait_signal(self, queue_name: str, waits: list, signals: list) -> None:
        """A batch without command buffers: wait every (semaphore, value) on the GPU, then signal."""
        wait_infos = [VkSemaphoreSubmitInfo(semaphore=semaphore, value=value,
                                            stageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) for semaphore, value in waits]
        signal_infos = [VkSemaphoreSubmitInfo(semaphore=semaphore, value=value,
                                              stageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
                        for semaphore, value in signals]
        submit = VkSubmitInfo2(waitSemaphoreInfoCount=len(wait_infos), pWaitSemaphoreInfos=wait_infos,
                               commandBufferInfoCount=0, pCommandBufferInfos=None,
                               signalSemaphoreInfoCount=len(signal_infos), pSignalSemaphoreInfos=signal_infos)
        vkQueueSubmit2(self.queues[queue_name], 1, [submit], VK_NULL_HANDLE)

    def read_timestamp(self, slot: int) -> int:
        data = ffi.new("uint64_t[1]")                  # python-vulkan takes pData as a cdata array
        vkGetQueryPoolResults(self.device, self.query_pool, slot, 1, 8, data, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
        return int(data[0])

    def calibration_pairs(self, count: int = 24, burst: int = 4) -> list:
        """(device ns, host ns, max deviation ns), each the narrowest of a burst."""
        stamps = ffi.new("uint64_t[2]")
        pairs = []
        for _ in range(count):
            best = None
            for _ in range(burst):
                deviation = int(self.get_calibrated(self.device, 2, self._calibration_infos, stamps))
                host = int(stamps[1])
                if self._qpc_frequency is not None:
                    host = host * 1_000_000_000 // self._qpc_frequency
                sample = (int(stamps[0]) * self.timestamp_period_ns, host, deviation)
                if best is None or sample[2] < best[2]:
                    best = sample
            pairs.append(best)
            time.sleep(0.0005)
        return pairs


def host_clock_for(domain: int):
    if domain == TIME_DOMAIN_CLOCK_MONOTONIC_RAW:
        return "clock_gettime_ns(CLOCK_MONOTONIC_RAW)", lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if domain == TIME_DOMAIN_CLOCK_MONOTONIC:
        return "clock_gettime_ns(CLOCK_MONOTONIC)", lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return "time.perf_counter_ns (QPC)", time.perf_counter_ns


def fit_device_to_host(pairs: list):
    """host = intercept + slope x device over the pairs within 1 us of the narrowest."""
    floor = min(pair[2] for pair in pairs)
    kept = [pair for pair in pairs if pair[2] <= floor + 1000] or pairs
    if len(kept) < 3:
        kept = sorted(pairs, key=lambda pair: pair[2])[:3]
    device_mean = statistics.fmean(pair[0] for pair in kept)
    host_mean = statistics.fmean(pair[1] for pair in kept)
    numerator = sum((pair[0] - device_mean) * (pair[1] - host_mean) for pair in kept)
    denominator = sum((pair[0] - device_mean) ** 2 for pair in kept)
    slope = numerator / denominator if denominator > 0 else 1.0
    residuals = [pair[1] - (host_mean + slope * (pair[0] - device_mean)) for pair in kept]
    return {"slope": slope, "device_mean": device_mean, "host_mean": host_mean, "pairs": len(pairs),
            "kept": len(kept), "deviation_floor_ns": floor,
            "residual_max_ns": max(abs(value) for value in residuals)}


def to_host(fit: dict, device_ns: float) -> float:
    return fit["host_mean"] + fit["slope"] * (device_ns - fit["device_mean"])


# ----------------------------------------------------------------------------- waits

def wait_info(semaphore, value):
    semaphores = ffi.new("VkSemaphore[1]", [semaphore])
    values = ffi.new("uint64_t[1]", [value])
    info = ffi.new("VkSemaphoreWaitInfo*")
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO
    info.semaphoreCount = 1
    info.pSemaphores = semaphores
    info.pValues = values
    return info, semaphores, values


def wait_infinite(device, semaphore, value, clock):
    info, _keep_semaphores, _keep_values = wait_info(semaphore, value)
    result = lib.vkWaitSemaphores(device, info, INFINITE_TIMEOUT)
    if result != VK_SUCCESS_CODE:
        raise RuntimeError(f"vkWaitSemaphores: VkResult {result}")
    return 1, 0


def wait_timeout_loop(device, semaphore, value, clock, timeout_ns):
    """Returns (driver calls, ns spent in Python between the calls)."""
    info, _keep_semaphores, _keep_values = wait_info(semaphore, value)
    calls = 0
    python_ns = 0
    returned = None
    while True:
        if returned is not None:
            python_ns += clock() - returned
        result = lib.vkWaitSemaphores(device, info, timeout_ns)
        returned = clock()
        calls += 1
        if result == VK_SUCCESS_CODE:
            return calls, python_ns
        if result != VK_TIMEOUT_CODE:
            raise RuntimeError(f"vkWaitSemaphores: VkResult {result}")


def wait_poll(device, semaphore, value, clock, use_counter: bool):
    """Non-blocking checks every 100 us until the value is reached: zero-timeout vkWaitSemaphores, or
    vkGetSemaphoreCounterValue. Returns (calls, ns in Python between the calls)."""
    info, _keep_semaphores, _keep_values = wait_info(semaphore, value)
    counter = ffi.new("uint64_t *")
    calls = 0
    while True:
        calls += 1
        if use_counter:
            result = lib.vkGetSemaphoreCounterValue(device, semaphore, counter)
            if result != VK_SUCCESS_CODE:
                raise RuntimeError(f"vkGetSemaphoreCounterValue: VkResult {result}")
            if counter[0] >= value:
                return calls, 0
        else:
            result = lib.vkWaitSemaphores(device, info, 0)
            if result == VK_SUCCESS_CODE:
                return calls, 0
            if result != VK_TIMEOUT_CODE:
                raise RuntimeError(f"vkWaitSemaphores: VkResult {result}")
        time.sleep(0.0001)


def signal_host(device, semaphore, value):
    info = ffi.new("VkSemaphoreSignalInfo*")
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO
    info.semaphore = semaphore
    info.value = value
    result = lib.vkSignalSemaphore(device, info)
    if result != VK_SUCCESS_CODE:
        raise RuntimeError(f"vkSignalSemaphore: VkResult {result}")


# ----------------------------------------------------------------------------- one case

class LoadThread(threading.Thread):
    """Python work in 1 ms windows, busy for duty x 1 ms then asleep; counts work units in busy windows."""

    def __init__(self, duty: float, clock):
        super().__init__(name="load", daemon=True)
        self.duty = duty
        self.clock = clock
        self.stop = threading.Event()
        self.units = 0
        self.busy_ns = 0

    def run(self):
        window_ns = 1_000_000
        busy_ns = int(self.duty * window_ns)
        table = {}
        while not self.stop.is_set():
            start = self.clock()
            while self.clock() - start < busy_ns:
                for key in range(16):                  # a unit: a little dict and arithmetic work
                    table[key] = table.get(key, 0) + key * 3
                self.units += 1
            self.busy_ns += self.clock() - start
            time.sleep(max(0.0, (window_ns - busy_ns) / 1e9))


def run_case(device: Device, case: dict, iterations: int, pre_signal_us: tuple, load_duty: float, seed: int):
    clock = device.host_clock
    waiters = case["waiters"]
    pattern = case["pattern"]
    method = case["method"]
    queue_name = case.get("queue", "transfer")
    stagger_ns = int(case.get("stagger_us", 0) * 1000)
    semaphores = [device.timeline() for _ in range(2 if pattern == "split" else 1)]
    random_generator = random.Random(seed)
    pre_signal_us = case.get("pre_signal_us", pre_signal_us)
    iterations = min(iterations, case.get("iterations", iterations))
    delays = [random_generator.uniform(*pre_signal_us) / 1e6 for _ in range(iterations)]
    if pattern == "mixed":
        def value_of(iteration):
            return 2 * iteration + 1
    elif pattern == "distinct":
        def value_of(iteration):                       # the signal reaches every waiter's own value
            return (iteration + 1) * waiters
    else:
        def value_of(iteration):
            return iteration + 1

    def target_of(iteration, index):
        return iteration * waiters + index + 1 if pattern == "distinct" else value_of(iteration)
    gpu_waiter_semaphore = device.timeline() if pattern == "cowait" else None
    timeout_ns = int(float(method.split(":")[1]) * 1e6) if method.startswith("timeout:") else None
    start_barrier = threading.Barrier(waiters + 1)
    end_barrier = threading.Barrier(waiters + 1)
    records = [[None] * iterations for _ in range(waiters)]
    thread_cpu = [0] * waiters
    events = [threading.Event() for _ in range(iterations)] if method == "relay_event" else None
    condition = threading.Condition()
    relayed = {"value": 0}
    errors = []

    def waiter_body(index: int):
        try:
            cpu_start = time.thread_time_ns()
            for iteration in range(iterations):
                start_barrier.wait()
                value = target_of(iteration, index)
                semaphore = semaphores[index % len(semaphores)]
                if stagger_ns:
                    target = clock() + stagger_ns * index
                    while clock() < target:
                        time.sleep(0.00005)
                t_start = clock()
                calls, python_ns = 1, 0
                if method == "infinite" or (method.startswith("relay") and index == 0)                         or (method in ("zeropoll", "counterpoll") and index == 0):
                    calls, python_ns = wait_infinite(device.device, semaphore, value, clock)
                    if method == "relay_event":
                        events[iteration].set()
                    elif method == "relay_condition":
                        with condition:
                            relayed["value"] = value
                            condition.notify_all()
                elif timeout_ns is not None:
                    calls, python_ns = wait_timeout_loop(device.device, semaphore, value, clock, timeout_ns)
                elif method in ("zeropoll", "counterpoll"):
                    calls, python_ns = wait_poll(device.device, semaphore, value, clock, method == "counterpoll")
                elif method == "relay_event":
                    events[iteration].wait()
                elif method == "relay_condition":
                    with condition:
                        while relayed["value"] < value:
                            condition.wait()
                t_wake = clock()
                if pattern == "mixed" and index == 0:
                    signal_host(device.device, semaphore, value + 1)   # the worker_done-like host value
                records[index][iteration] = (t_start, t_wake, calls, python_ns)
                end_barrier.wait()
            thread_cpu[index] = time.thread_time_ns() - cpu_start
        except Exception as error:                     # noqa: BLE001
            errors.append(repr(error))
            start_barrier.abort()
            end_barrier.abort()

    load = LoadThread(load_duty, clock) if case["load"] else None
    if load is not None:
        load.start()
    threads = [threading.Thread(target=waiter_body, args=(index,), name=f"waiter{index}", daemon=True)
               for index in range(waiters)]
    for thread in threads:
        thread.start()
    calibration_before = device.calibration_pairs()
    signals = []
    wall_start = clock()
    try:
        for iteration in range(iterations):
            value = value_of(iteration)
            slot = iteration % QUERY_RING
            if gpu_waiter_semaphore is not None:          # pending on the GPU before the signal, like an upload
                device.submit_wait_signal("compute", [(semaphores[0], value)], [(gpu_waiter_semaphore, iteration + 1)])
            start_barrier.wait()
            time.sleep(delays[iteration])
            t_signal_host = clock()
            if case["signal"] == "host":
                for semaphore in semaphores:
                    signal_host(device.device, semaphore, value)
            else:
                device.submit_signal(queue_name, slot, [(semaphore, value) for semaphore in semaphores])
            end_barrier.wait()
            if gpu_waiter_semaphore is not None:
                wait_infinite(device.device, gpu_waiter_semaphore, iteration + 1, clock)
            gpu_ns = device.read_timestamp(slot) * device.timestamp_period_ns if case["signal"] == "gpu" else None
            signals.append((t_signal_host, gpu_ns))
    except threading.BrokenBarrierError:
        pass
    wall_ns = clock() - wall_start
    calibration_after = device.calibration_pairs()
    for thread in threads:
        thread.join(timeout=30)
    if load is not None:
        load.stop.set()
        load.join(timeout=5)
    if errors:
        raise RuntimeError(f"case {case['name']}: {errors[0]}")
    fit = fit_device_to_host(calibration_before + calibration_after) if case["signal"] == "gpu" else None
    rows = []
    for iteration, (t_signal_host, gpu_ns) in enumerate(signals):
        t_signal = to_host(fit, gpu_ns) if gpu_ns is not None else t_signal_host
        for index in range(waiters):
            t_start, t_wake, calls, python_ns = records[index][iteration]
            rows.append({"case": case["name"], "iteration": iteration, "waiter": index, "t_start": t_start,
                         "t_wake": t_wake, "t_signal": round(t_signal), "t_signal_host_call": t_signal_host,
                         "calls": calls, "python_ns": python_ns})
    for semaphore in semaphores + ([gpu_waiter_semaphore] if gpu_waiter_semaphore is not None else []):
        vkDestroySemaphore(device.device, semaphore, None)
    load_rate = (load.units / (load.busy_ns / 1e9)) if load is not None and load.busy_ns > 0 else None
    return rows, {"wall_s": wall_ns / 1e9, "thread_cpu_s": [value / 1e9 for value in thread_cpu],
                  "load_units_per_busy_s": load_rate, "clock_fit": fit}


# ----------------------------------------------------------------------------- summary

def quantile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else float("nan")


def summarize(case: dict, rows: list, extra: dict, baseline_load: float) -> dict:
    waiters = case["waiters"]
    per_waiter = []
    blocked_all, late_all, late_from_start = [], [], []
    first_late = later_late = first_count = later_count = 0
    by_iteration = {}
    for row in rows:
        by_iteration.setdefault(row["iteration"], []).append(row)
    for group in by_iteration.values():
        order = sorted(group, key=lambda row: row["t_start"])
        for rank, row in enumerate(order):
            if row["t_start"] >= row["t_signal"]:
                continue
            late = row["t_wake"] - row["t_signal"] > LATE_NANOSECONDS
            if rank == 0:
                first_count += 1
                first_late += late
            else:
                later_count += 1
                later_late += late
    total_wait_ns = sum(max(0, row["t_wake"] - row["t_start"]) for row in rows)
    for index in range(waiters):
        mine = [row for row in rows if row["waiter"] == index]
        blocked = [row for row in mine if row["t_start"] < row["t_signal"]]
        latency = [(row["t_wake"] - row["t_signal"]) / 1e3 for row in blocked]
        late = [row for row in blocked if row["t_wake"] - row["t_signal"] > LATE_NANOSECONDS]
        wait_ns = sum(max(0, row["t_wake"] - row["t_start"]) for row in mine)
        calls = sum(row["calls"] for row in mine)
        python_ns = sum(row["python_ns"] for row in mine)
        per_waiter.append({
            "waiter": index, "waits": len(mine), "blocked": len(blocked),
            "latency_us_p50": quantile(latency, 0.5), "latency_us_p99": quantile(latency, 0.99),
            "latency_us_max": max(latency) if latency else float("nan"),
            "late": len(late), "late_fraction": len(late) / len(blocked) if blocked else float("nan"),
            "driver_calls_per_wait_s": calls / (wait_ns / 1e9) if wait_ns else float("nan"),
            "wait_ns_per_driver_call": wait_ns / calls if calls else float("nan"),
            "python_share_while_waiting": python_ns / wait_ns if wait_ns else float("nan"),
            "thread_cpu_per_wall_s": extra["thread_cpu_s"][index] / extra["wall_s"] if extra["wall_s"] else None})
        blocked_all.extend(latency)
        late_all.extend(late)
        late_from_start.extend((row["t_wake"] - row["t_start"]) / 1e3 for row in late)
    histogram = {}
    for value in late_from_start:
        histogram_bin = round(int(value // 250) * 0.25, 2)
        histogram[histogram_bin] = histogram.get(histogram_bin, 0) + 1
    in_10ms_bin = sum(1 for value in late_from_start if 10_000 <= value < 10_250)
    return {
        "case": case["name"], "signal": case["signal"], "queue": case.get("queue", "transfer"), "waiters": waiters,
        "method": case["method"], "pattern": case["pattern"], "load": case["load"],
        "stagger_us": case.get("stagger_us", 0), "pre_signal_us": list(case.get("pre_signal_us", ())) or None,
        "iterations": len({row["iteration"] for row in rows}),
        "blocked": len(blocked_all), "late": len(late_all),
        "late_fraction": len(late_all) / len(blocked_all) if blocked_all else float("nan"),
        "latency_us_p50": quantile(blocked_all, 0.5), "latency_us_p99": quantile(blocked_all, 0.99),
        "latency_us_max": max(blocked_all) if blocked_all else float("nan"),
        "late_from_start_us_p01": quantile(late_from_start, 0.01),
        "late_from_start_us_p50": quantile(late_from_start, 0.5),
        "late_from_start_us_p99": quantile(late_from_start, 0.99),
        "late_in_10_00_to_10_25_ms": in_10ms_bin,
        "late_from_start_histogram_ms": dict(sorted(histogram.items())),
        "late_first_starter": [first_late, first_count], "late_later_starters": [later_late, later_count],
        "driver_calls_per_s_all_waiters": sum(row["calls"] for row in rows) / (extra["wall_s"] or 1.0),
        "total_wait_s": total_wait_ns / 1e9, "wall_s": extra["wall_s"],
        "load_units_per_busy_s": extra["load_units_per_busy_s"],
        "load_relative_to_baseline": (extra["load_units_per_busy_s"] / baseline_load
                                      if extra["load_units_per_busy_s"] and baseline_load else None),
        "per_waiter": per_waiter, "clock_fit": extra["clock_fit"]}


def format_case(summary: dict) -> str:
    lines = [f"== {summary['case']}: signal {summary['signal']} ({summary['queue']}), {summary['waiters']} waiter(s), "
             f"{summary['method']}, pattern {summary['pattern']}, load {summary['load']}, iterations "
             f"{summary['iterations']}, wall {summary['wall_s']:.1f} s"]
    if summary["waiters"]:
        lines.append(f"   blocked {summary['blocked']}, late (> 5 ms) {summary['late']} "
                     f"({100 * summary['late_fraction']:.2f} %); latency us p50 {summary['latency_us_p50']:.0f} "
                     f"p99 {summary['latency_us_p99']:.0f} max {summary['latency_us_max']:.0f}")
        if summary["late"]:
            lines.append(f"   late waits, from their start: p01 {summary['late_from_start_us_p01']:.0f} p50 "
                         f"{summary['late_from_start_us_p50']:.0f} p99 {summary['late_from_start_us_p99']:.0f} us; "
                         f"in 10.00-10.25 ms: {summary['late_in_10_00_to_10_25_ms']}; first starter late "
                         f"{summary['late_first_starter'][0]}/{summary['late_first_starter'][1]}, later starters "
                         f"{summary['late_later_starters'][0]}/{summary['late_later_starters'][1]}")
            lines.append("   late from start, 250 us bins (ms: count): " + ", ".join(
                f"{key:.2f}: {value}" for key, value in summary["late_from_start_histogram_ms"].items()))
        for waiter in summary["per_waiter"]:
            lines.append(f"   waiter {waiter['waiter']}: blocked {waiter['blocked']}, p50 {waiter['latency_us_p50']:.0f} "
                         f"p99 {waiter['latency_us_p99']:.0f} max {waiter['latency_us_max']:.0f} us, late "
                         f"{waiter['late']}; driver calls per waiting s {waiter['driver_calls_per_wait_s']:.0f} "
                         f"({waiter['wait_ns_per_driver_call'] / 1e3:.0f} us waited per call), "
                         f"Python share while waiting {100 * waiter['python_share_while_waiting']:.2f} %, thread CPU "
                         f"per wall s {waiter['thread_cpu_per_wall_s']:.3f}")
    if summary["load_units_per_busy_s"]:
        relative = summary["load_relative_to_baseline"]
        lines.append(f"   load thread: {summary['load_units_per_busy_s']:.0f} units per busy s"
                     + (f" ({100 * relative:.1f} % of the case without waiters)" if relative else ""))
    if summary["clock_fit"]:
        fit = summary["clock_fit"]
        lines.append(f"   clock fit: {fit['kept']}/{fit['pairs']} pairs, deviation floor {fit['deviation_floor_ns']} ns, "
                     f"residual max {fit['residual_max_ns']:.0f} ns")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def environment(device: Device) -> dict:
    record = {"python": sys.version, "platform": platform.platform(), "switch_interval_s": sys.getswitchinterval(),
              "device": device.device_name, "driver_name": device.driver_name, "driver_info": device.driver_info,
              "driver_version_raw": device.driver_version_raw, "calibrated_extension": device.calibrated_extension,
              "time_domain": TIME_DOMAIN_NAMES[device.domain], "host_clock": device.host_clock_name,
              "timestamp_period_ns": device.timestamp_period_ns, "timestamp_valid_bits": device.timestamp_bits,
              "transfer_family": device.transfer_family, "compute_family": device.compute_family,
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        import importlib.metadata
        record["vulkan_package"] = importlib.metadata.version("vulkan")
    except Exception:                                  # noqa: BLE001
        record["vulkan_package"] = None
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--copy-bytes", type=int, default=1 << 20)
    parser.add_argument("--pre-signal-us", default="200:3000")
    parser.add_argument("--load-duty", type=float, default=0.5)
    parser.add_argument("--switch-interval-ms", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    sys.setswitchinterval(args.switch_interval_ms / 1000.0)
    low, high = (float(value) for value in args.pre_signal_us.split(":"))
    os.makedirs(args.out, exist_ok=True)
    device = Device(args.device, args.copy_bytes)
    env = environment(device)
    with open(os.path.join(args.out, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=1)
    print(json.dumps(env), flush=True)
    cases = case_list(args.cases)
    summaries = []
    baseline_load = None
    with open(os.path.join(args.out, "waits.csv"), "w", newline="", encoding="utf-8") as handle, \
            open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as text:
        writer = None
        for number, case in enumerate(cases):
            rows, extra = run_case(device, case, args.iterations, (low, high), args.load_duty, args.seed + number)
            if case["name"] == "load_only":
                baseline_load = extra["load_units_per_busy_s"]
            summary = summarize(case, rows, extra, baseline_load)
            summaries.append(summary)
            block = format_case(summary)
            print(block, flush=True)
            text.write(block + "\n")
            text.flush()
            if rows:
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
    with open(os.path.join(args.out, "cases.json"), "w", encoding="utf-8") as handle:
        json.dump({"environment": env, "arguments": vars(args), "cases": summaries}, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
