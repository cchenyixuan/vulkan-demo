"""
caps_v6.py — E30 S1 capability query (no simulation).

Per physical device, in the same discrete-first order as VulkanContextV6:
driver (VkPhysicalDeviceDriverProperties), API version, device UUID, PCI bus
(VK_EXT_pci_bus_info), memory heaps (+ budget), every queue family (raw flags,
decoded flags, queueCount, timestampValidBits), timestampPeriod, the
calibrated-timestamp extensions and their calibrateable time domains, and the
v6 transfer-family pick (first TRANSFER family without GRAPHICS / COMPUTE; the
release default splits readback / upload over two queues of it when
queueCount >= 2).

Host side: Python clock implementations, CLOCK_MONOTONIC_RAW availability, the
kernel clocksource, the job's CPU affinity, each GPU's NUMA node and cpulist
(sysfs), and the nvidia-smi index / UUID / bus id join, with the check that the
v6 index equals the nvidia-smi index (V6_WORKER_AFFINITY is indexed by the
former, the job builds it in the latter's order).

--live-sample additionally creates a VulkanContextV6 per NVIDIA device with
VK_KHR_calibrated_timestamps and takes vkGetCalibratedTimestampsKHR samples
against every offered host domain, bracketed by the matching Python clock
(E29: does CLOCK_MONOTONIC equal time.perf_counter_ns on this host?).

Usage (from the checkout root):
    python docs/cluster_v6/scripts/caps_v6.py --out caps.json [--live-sample]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import vulkan as vk                                   # noqa: E402
from vulkan._vulkancache import ffi                   # noqa: E402

QUEUE_FLAG_NAMES = (
    (0x001, "graphics"), (0x002, "compute"), (0x004, "transfer"), (0x008, "sparse"),
    (0x010, "protected"), (0x020, "video_decode"), (0x040, "video_encode"),
    (0x100, "optical_flow"),
)
TIME_DOMAIN_NAMES = {0: "DEVICE", 1: "CLOCK_MONOTONIC", 2: "CLOCK_MONOTONIC_RAW",
                     3: "QUERY_PERFORMANCE_COUNTER"}
NVIDIA_VENDOR_ID = 0x10DE
CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE = 1000184000   # official value (python-vulkan 1.3.275.1 has 1000543000)


def as_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return ffi.string(value).decode("utf-8", "replace")
    except Exception:                                                  # noqa: BLE001
        return str(value)


def decode_queue_flags(flags: int) -> str:
    names = [name for bit, name in QUEUE_FLAG_NAMES if flags & bit]
    remainder = flags & ~sum(bit for bit, _ in QUEUE_FLAG_NAMES)
    if remainder:
        names.append(f"0x{remainder:x}")
    return "+".join(names) or "none"


def version_text(version: int) -> str:
    return f"{version >> 22}.{(version >> 12) & 0x3ff}.{version & 0xfff}"


def read_text(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def nvidia_smi_rows() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = "index,uuid,pci.bus_id,name,driver_version,memory.total,power.limit"
    try:
        output = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=60).stdout
    except Exception:                                                  # noqa: BLE001
        return []
    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        bus = parts[2].lower()
        if len(bus.split(":")[0]) == 8:                # 00000000:16:00.0 -> 0000:16:00.0
            bus = bus[4:]
        rows.append({"index": int(parts[0]), "uuid": parts[1].replace("GPU-", "").replace("-", "").lower(),
                     "pci_bus_id": bus, "name": parts[3], "driver_version": parts[4],
                     "memory_total": parts[5], "power_limit": parts[6]})
    return rows


def host_information() -> dict:
    information = {"hostname": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
                   "platform": sys.platform, "python": sys.version.split()[0],
                   "has_clock_monotonic_raw": hasattr(time, "CLOCK_MONOTONIC_RAW"),
                   "clocksource": read_text("/sys/devices/system/clocksource/clocksource0/current_clocksource")}
    for clock_name in ("perf_counter", "monotonic"):
        clock_info = time.get_clock_info(clock_name)
        information[f"clock_{clock_name}"] = {"implementation": clock_info.implementation,
                                              "resolution": clock_info.resolution}
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        information["job_cpu_count"] = len(cpus)
        information["job_cpus"] = compress_cpu_list(cpus)
    try:
        import importlib.metadata as metadata
        information["python_vulkan"] = metadata.version("vulkan")
    except Exception:                                                  # noqa: BLE001
        pass
    loader_version = vk.vkEnumerateInstanceVersion()
    information["vulkan_loader"] = version_text(loader_version)
    return information


def compress_cpu_list(cpus: list[int]) -> str:
    ranges, start, previous = [], None, None
    for cpu in cpus:
        if start is None:
            start = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            ranges.append(f"{start}-{previous}" if previous > start else f"{start}")
            start = previous = cpu
    if start is not None:
        ranges.append(f"{start}-{previous}" if previous > start else f"{start}")
    return ",".join(ranges)


def numa_information(pci_bus_id: str | None) -> dict:
    if not pci_bus_id:
        return {}
    device_path = f"/sys/bus/pci/devices/{pci_bus_id}"
    numa_node_text = read_text(f"{device_path}/numa_node")
    information = {"sysfs_present": pathlib.Path(device_path).exists(),
                   "numa_node": int(numa_node_text) if numa_node_text not in (None, "") else None,
                   "local_cpulist": read_text(f"{device_path}/local_cpulist"),
                   "current_link_speed": read_text(f"{device_path}/current_link_speed"),
                   "current_link_width": read_text(f"{device_path}/current_link_width")}
    if information["numa_node"] is not None and information["numa_node"] >= 0:
        information["node_cpulist"] = read_text(
            f"/sys/devices/system/node/node{information['numa_node']}/cpulist")
    return information


def query_devices() -> list[dict]:
    application_info = vk.VkApplicationInfo(pApplicationName="e30_caps", applicationVersion=1,
                                            pEngineName="e30_caps", engineVersion=1,
                                            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0))
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=application_info), None)
    devices = []
    try:
        domain_queries = {}
        for function_name in ("vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
                              "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"):
            try:
                domain_queries[function_name] = vk.vkGetInstanceProcAddr(instance, function_name)
            except Exception as error:                                 # noqa: BLE001
                domain_queries[function_name] = repr(error)
        raw_devices = vk.vkEnumeratePhysicalDevices(instance)
        order = sorted(range(len(raw_devices)), key=lambda raw_index: 0 if vk.vkGetPhysicalDeviceProperties(
            raw_devices[raw_index]).deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1)
        for v6_index, raw_index in enumerate(order):
            physical_device = raw_devices[raw_index]
            properties = vk.vkGetPhysicalDeviceProperties(physical_device)
            extension_names = sorted(as_text(extension.extensionName) for extension in
                                     vk.vkEnumerateDeviceExtensionProperties(physical_device, None))
            supported = {name: name in extension_names for name in (
                "VK_KHR_calibrated_timestamps", "VK_EXT_calibrated_timestamps",
                "VK_EXT_pci_bus_info", "VK_EXT_memory_budget")}
            pci_bus_info = (vk.VkPhysicalDevicePCIBusInfoPropertiesEXT()
                            if supported["VK_EXT_pci_bus_info"] else None)
            identity = vk.VkPhysicalDeviceIDProperties(pNext=pci_bus_info if pci_bus_info is not None else ffi.NULL)
            driver = vk.VkPhysicalDeviceDriverProperties(pNext=identity)
            vk.vkGetPhysicalDeviceProperties2(physical_device, vk.VkPhysicalDeviceProperties2(pNext=driver))
            conformance = driver.conformanceVersion
            driver_version = properties.driverVersion
            entry = {
                "v6_index": v6_index, "raw_index": raw_index,
                "name": as_text(properties.deviceName),
                "device_type": int(properties.deviceType),
                "vendor_id": f"0x{properties.vendorID:04x}",
                "is_nvidia_discrete": (properties.vendorID == NVIDIA_VENDOR_ID and
                                       properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
                "api_version": version_text(properties.apiVersion),
                "driver_version_raw": int(driver_version),
                "driver_version_nvidia": (f"{driver_version >> 22}.{(driver_version >> 14) & 0xff}."
                                          f"{(driver_version >> 6) & 0xff}"),
                "driver_name": as_text(driver.driverName),
                "driver_info": as_text(driver.driverInfo),
                "conformance": f"{conformance.major}.{conformance.minor}.{conformance.subminor}.{conformance.patch}",
                "device_uuid": bytes(identity.deviceUUID).hex(),
                "timestamp_period_ns": float(properties.limits.timestampPeriod),
                "timestamp_compute_and_graphics": int(properties.limits.timestampComputeAndGraphics),
                "extensions": supported,
            }
            if pci_bus_info is not None:
                entry["pci_bus_id"] = (f"{pci_bus_info.pciDomain:04x}:{pci_bus_info.pciBus:02x}:"
                                       f"{pci_bus_info.pciDevice:02x}.{pci_bus_info.pciFunction:x}")
            memory = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
            entry["heaps"] = [{"index": heap_index,
                               "gib": round(memory.memoryHeaps[heap_index].size / 2**30, 2),
                               "device_local": bool(memory.memoryHeaps[heap_index].flags & 0x1)}
                              for heap_index in range(memory.memoryHeapCount)]
            if supported["VK_EXT_memory_budget"]:
                budget = vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT()
                vk.vkGetPhysicalDeviceMemoryProperties2(physical_device,
                                                        vk.VkPhysicalDeviceMemoryProperties2(pNext=budget))
                entry["heap_budget_gib"] = [round(budget.heapBudget[heap_index] / 2**30, 2)
                                            for heap_index in range(memory.memoryHeapCount)]
            families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
            entry["queue_families"] = [{
                "index": family_index, "flags_hex": f"0x{family.queueFlags:x}",
                "flags": decode_queue_flags(family.queueFlags), "count": int(family.queueCount),
                "timestamp_valid_bits": int(family.timestampValidBits),
                "transfer_only": bool((family.queueFlags & 0x4) and not (family.queueFlags & 0x3)),
            } for family_index, family in enumerate(families)]
            compute_family = next((family["index"] for family in entry["queue_families"]
                                   if int(family["flags_hex"], 16) & 0x2), None)
            transfer_family = next((family["index"] for family in entry["queue_families"]
                                    if family["transfer_only"]), None)
            entry["v6_compute_family"] = compute_family
            entry["v6_transfer_family"] = transfer_family
            entry["v6_transfer_queue_count"] = (entry["queue_families"][transfer_family]["count"]
                                                if transfer_family is not None else None)
            entry["v6_split_transfer_possible"] = bool(transfer_family is not None
                                                       and entry["v6_transfer_queue_count"] >= 2)
            domains = {}
            for function_name, function in domain_queries.items():
                suffix = function_name[-3:]
                if not callable(function):
                    domains[suffix] = f"unavailable: {function}"
                elif not supported[f"VK_{suffix}_calibrated_timestamps"]:
                    domains[suffix] = "extension not supported"
                else:
                    try:
                        domains[suffix] = [TIME_DOMAIN_NAMES.get(int(value), int(value))
                                           for value in function(physical_device)]
                    except Exception as error:                         # noqa: BLE001
                        domains[suffix] = f"error: {error!r}"
            entry["calibrateable_domains"] = domains
            devices.append(entry)
    finally:
        vk.vkDestroyInstance(instance, None)
    return devices


def live_calibration_samples(device_entries: list[dict], repeat: int = 5) -> None:
    """vkGetCalibratedTimestampsKHR on each NVIDIA device against every offered host
    domain, bracketed by the Python clock of that domain."""
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6
    host_clocks = {1: ("time.perf_counter_ns", time.perf_counter_ns)}
    if hasattr(time, "CLOCK_MONOTONIC_RAW"):
        host_clocks[2] = ("clock_gettime_ns(CLOCK_MONOTONIC_RAW)",
                          lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
    if sys.platform == "win32":
        host_clocks[3] = ("QueryPerformanceCounter", None)
    for entry in device_entries:
        if not entry["is_nvidia_discrete"]:
            continue
        domains = entry["calibrateable_domains"].get("KHR")
        if not isinstance(domains, list):
            entry["live_sample"] = "skipped: KHR calibrated timestamps not supported"
            continue
        context = None
        results = {}
        try:
            context = VulkanContextV6.create(device_index=entry["v6_index"], enable_validation=False,
                                             application_name="e30_caps_live",
                                             extra_device_extensions=["VK_KHR_calibrated_timestamps"])
            function = vk.vkGetDeviceProcAddr(context.device, "vkGetCalibratedTimestampsKHR")
            stamps = ffi.new("uint64_t[2]")
            for domain_value, (clock_name, clock_function) in host_clocks.items():
                if TIME_DOMAIN_NAMES[domain_value] not in domains or clock_function is None:
                    continue
                samples = []
                for structure_type in (None, CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE):
                    try:
                        if structure_type is None:
                            infos = [vk.VkCalibratedTimestampInfoKHR(timeDomain=0),
                                     vk.VkCalibratedTimestampInfoKHR(timeDomain=domain_value)]
                        else:
                            infos = [vk.VkCalibratedTimestampInfoKHR(sType=structure_type, timeDomain=0),
                                     vk.VkCalibratedTimestampInfoKHR(sType=structure_type,
                                                                     timeDomain=domain_value)]
                        for _ in range(repeat):
                            before = clock_function()
                            deviation = function(context.device, 2, infos, stamps)
                            after = clock_function()
                            host_value = int(stamps[1])
                            samples.append({"host_before": before, "host_value": host_value,
                                            "host_after": after,
                                            "inside_bracket": before <= host_value <= after,
                                            "max_deviation_ns": float(int(deviation)) * entry["timestamp_period_ns"],
                                            "device_ticks": int(stamps[0])})
                        results[TIME_DOMAIN_NAMES[domain_value]] = {
                            "host_clock": clock_name,
                            "structure_type": "python-vulkan default" if structure_type is None else structure_type,
                            "inside_bracket": sum(sample["inside_bracket"] for sample in samples),
                            "samples": len(samples),
                            "max_deviation_ns_min": min(sample["max_deviation_ns"] for sample in samples),
                            "host_minus_bracket_mid_ns": [sample["host_value"] - (sample["host_before"]
                                                          + sample["host_after"]) // 2 for sample in samples]}
                        break
                    except Exception as error:                         # noqa: BLE001
                        results[TIME_DOMAIN_NAMES[domain_value]] = f"error with sType {structure_type}: {error!r}"
                        samples = []
            entry["live_sample"] = results
        except Exception as error:                                     # noqa: BLE001
            entry["live_sample"] = f"error: {error!r}"
        finally:
            if context is not None:
                context.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="E30 S1 capability query")
    parser.add_argument("--out", default=None, help="write the full report as JSON here")
    parser.add_argument("--live-sample", action="store_true",
                        help="create a context per NVIDIA device and take calibrated-timestamp samples")
    arguments = parser.parse_args()

    report = {"host": host_information(), "devices": query_devices(), "nvidia_smi": nvidia_smi_rows()}
    by_uuid = {row["uuid"]: row for row in report["nvidia_smi"]}
    mapping_ok = True
    for entry in report["devices"]:
        row = by_uuid.get(entry["device_uuid"])
        entry["nvidia_smi_index"] = row["index"] if row else None
        entry["numa"] = numa_information(entry.get("pci_bus_id"))
        if entry["is_nvidia_discrete"] and (row is None or row["index"] != entry["v6_index"]):
            mapping_ok = False
    report["v6_index_equals_nvidia_smi_index"] = mapping_ok
    if arguments.live_sample:
        live_calibration_samples(report["devices"])

    host = report["host"]
    print(f"[caps] host={host.get('hostname')} python={host.get('python')} "
          f"python_vulkan={host.get('python_vulkan')} loader={host.get('vulkan_loader')} "
          f"clocksource={host.get('clocksource')} "
          f"perf_counter={host['clock_perf_counter']['implementation']} "
          f"monotonic={host['clock_monotonic']['implementation']} "
          f"has_CLOCK_MONOTONIC_RAW={host.get('has_clock_monotonic_raw')} "
          f"job_cpus={host.get('job_cpus')}", flush=True)
    for entry in report["devices"]:
        families = " ".join(f"[{family['index']}]{family['flags_hex']}x{family['count']}"
                            f"/ts{family['timestamp_valid_bits']}" for family in entry["queue_families"])
        heap = max((heap["gib"] for heap in entry["heaps"] if heap["device_local"]), default=None)
        numa = entry.get("numa", {})
        print(f"[caps] v6[{entry['v6_index']}] smi[{entry.get('nvidia_smi_index')}] {entry['name']} "
              f"bus={entry.get('pci_bus_id')} uuid={entry['device_uuid'][:8]} "
              f"driver={entry['driver_info'] or entry['driver_version_nvidia']} api={entry['api_version']} "
              f"vram={heap}GiB period={entry['timestamp_period_ns']}ns "
              f"compute_qf={entry['v6_compute_family']} transfer_qf={entry['v6_transfer_family']}"
              f"x{entry['v6_transfer_queue_count']} split={int(entry['v6_split_transfer_possible'])} "
              f"domains_KHR={entry['calibrateable_domains'].get('KHR')} "
              f"domains_EXT={entry['calibrateable_domains'].get('EXT')} "
              f"numa={numa.get('numa_node')} cpus={numa.get('node_cpulist')} | {families}", flush=True)
        if "live_sample" in entry:
            print(f"[caps] v6[{entry['v6_index']}] live: {json.dumps(entry['live_sample'])[:600]}", flush=True)
    print(f"[caps] v6_index_equals_nvidia_smi_index={mapping_ok}", flush=True)
    if arguments.out:
        pathlib.Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[caps] wrote {arguments.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
