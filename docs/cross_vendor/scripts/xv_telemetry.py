"""xv_telemetry.py — telemetry of the RTX 5090 (nvidia-smi) and the RX 7900 XTX (ADL, atiadlxx.dll) on the cross-vendor
machine, in the CSV layout parse_run_v6.py reads (nvidia-smi --query-gpu --format=csv: a "timestamp" column
"YYYY/MM/DD HH:MM:SS.fff", "index", "memory.used [MiB]", "utilization.gpu [%]", "power.draw [W]", "temperature.gpu",
"clocks.current.sm [MHz]"), plus columns the parser ignores (memory clock, hot-spot temperature, PCIe link generation
and width, the throttle status).

Rows: index 0 = the NVIDIA card (nvidia-smi index 0), index 1 = the AMD discrete card (the VulkanContextV7
discrete-first order of this machine: [0] 5090, [1] 7900 XTX, [2] the Ryzen iGPU, which is not sampled).

NVIDIA: ONE streaming nvidia-smi process (--query-gpu ... -lms INTERVAL, the cluster harness's way) read by a thread;
one-shot nvidia-smi processes per sample returned clock / power values that lagged by ~15 s on this machine (E0,
2026-10-20), the stream does not. The NVIDIA rows carry nvidia-smi's own timestamp.

AMD: ADL2_New_QueryPMLogData_Get (the Overdrive 8 PM log) sensor ids GFXCLK (1), MEMCLK (2), edge temperature (8),
hot-spot temperature (27), GFX activity (19); VRAM in use from ADL2_Adapter_VRAMUsage_Get. No power value: the two
candidate ids (40 and 73) read a constant 3 and a constant 403 under load on this driver (26.6.1), neither credible,
so the AMD power column is left empty. The PCIe link of the AMD card is read from the PnP device properties
(DEVPKEY_PciDevice_CurrentLink*) once at start. A sleeping card reports GFXCLK 0.

Usage:
    python docs/cross_vendor/scripts/xv_telemetry.py OUT.csv [--interval 1.0] [--stop-file PATH]
Runs until the stop file exists (or Ctrl-C).
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import pathlib
import subprocess
import sys
import threading
import time

ADL_OK = 0
MAX_SENSORS = 256
ADL_SENSOR = {"gfxclk": 1, "memclk": 2, "temp_edge": 8, "activity_gfx": 19, "temp_hotspot": 27, "throttler": 35}
NVIDIA_QUERY = ("timestamp,index,pci.bus_id,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,"
                "clocks.current.sm,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active")
HEADER = ("timestamp, index, name, pci.bus_id, memory.used [MiB], memory.total [MiB], utilization.gpu [%], power.draw [W], "
          "temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], temperature.hotspot, pcie.link.gen.current, "
          "pcie.link.width.current, throttle")


class AdapterInfo(ctypes.Structure):
    _fields_ = [("iSize", ctypes.c_int), ("iAdapterIndex", ctypes.c_int), ("strUDID", ctypes.c_char * 256), ("iBusNumber", ctypes.c_int),
                ("iDeviceNumber", ctypes.c_int), ("iFunctionNumber", ctypes.c_int), ("iVendorID", ctypes.c_int),
                ("strAdapterName", ctypes.c_char * 256), ("strDisplayName", ctypes.c_char * 256), ("iPresent", ctypes.c_int),
                ("iExist", ctypes.c_int), ("strDriverPath", ctypes.c_char * 256), ("strDriverPathExt", ctypes.c_char * 256),
                ("strPNPString", ctypes.c_char * 256), ("iOSDisplayIndex", ctypes.c_int)]


class SingleSensorData(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class PMLogDataOutput(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("data", SingleSensorData * MAX_SENSORS)]


ALLOC = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int)
ctypes.windll.kernel32.HeapAlloc.restype = ctypes.c_void_p
ctypes.windll.kernel32.HeapAlloc.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t]
ctypes.windll.kernel32.GetProcessHeap.restype = ctypes.c_void_p
_alloc_callback = ALLOC(lambda size: ctypes.windll.kernel32.HeapAlloc(ctypes.windll.kernel32.GetProcessHeap(), 0, size))


class AmdCard:
    """The first ADL adapter of the discrete AMD card (vendor 1002, a name containing name_filter)."""

    def __init__(self, name_filter: str = "RX"):
        self.lib = ctypes.WinDLL("atiadlxx.dll")
        self.context = ctypes.c_void_p()
        self.lib.ADL2_Main_Control_Create.argtypes = [ALLOC, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        if self.lib.ADL2_Main_Control_Create(_alloc_callback, 1, ctypes.byref(self.context)) != ADL_OK:
            raise RuntimeError("ADL2_Main_Control_Create failed")
        count = ctypes.c_int()
        self.lib.ADL2_Adapter_NumberOfAdapters_Get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.ADL2_Adapter_NumberOfAdapters_Get(self.context, ctypes.byref(count))
        infos = (AdapterInfo * count.value)()
        self.lib.ADL2_Adapter_AdapterInfo_Get.argtypes = [ctypes.c_void_p, ctypes.POINTER(AdapterInfo), ctypes.c_int]
        self.lib.ADL2_Adapter_AdapterInfo_Get(self.context, infos, ctypes.sizeof(infos))
        self.lib.ADL2_New_QueryPMLogData_Get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(PMLogDataOutput)]
        self.lib.ADL2_Adapter_VRAMUsage_Get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.index = None
        for info in infos:
            name = info.strAdapterName.decode(errors="replace")
            if info.iVendorID == 1002 and name_filter in name:
                self.index, self.name = info.iAdapterIndex, name
                self.bus = f"{info.iBusNumber:02x}:{info.iDeviceNumber:02x}.{info.iFunctionNumber}"
                break
        if self.index is None:
            raise RuntimeError(f"no AMD adapter whose name contains {name_filter!r}")

    def sample(self) -> dict:
        output = PMLogDataOutput()
        values = {}
        if self.lib.ADL2_New_QueryPMLogData_Get(self.context, self.index, ctypes.byref(output)) == ADL_OK:
            for key, sensor in ADL_SENSOR.items():
                values[key] = output.data[sensor].value if output.data[sensor].supported else None
        vram = ctypes.c_int(-1)
        if self.lib.ADL2_Adapter_VRAMUsage_Get(self.context, self.index, ctypes.byref(vram)) == ADL_OK:
            values["vram_used_mib"] = vram.value
        return values


def amd_pcie_link() -> tuple[str, str]:
    """(generation, width) of the discrete AMD card from the PnP device properties (DEVPKEY_PciDevice_CurrentLink*)."""
    command = ("$d = Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -like '*RX*' } | Select-Object -First 1; "
               "(Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_PciDevice_CurrentLinkSpeed,"
               "DEVPKEY_PciDevice_CurrentLinkWidth | ForEach-Object { $_.Data }) -join ','")
    try:
        text = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=30).stdout.strip()
        generation, width = text.split(",")
        return generation, width
    except Exception:  # noqa: BLE001
        return "", ""


class NvidiaStream:
    """One nvidia-smi process streaming every interval; a thread keeps the latest row of each index."""

    def __init__(self, interval_ms: int):
        self.process = subprocess.Popen(["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.latest: dict = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in self.process.stdout:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 13:
                continue
            row = {"timestamp": parts[0], "index": parts[1], "bus": parts[2], "memory_used": parts[3], "memory_total": parts[4],
                   "utilization": parts[5], "power": parts[6], "temperature": parts[7], "sm": parts[8], "mem": parts[9],
                   "pcie_gen": parts[10], "pcie_width": parts[11], "throttle": parts[12]}
            with self.lock:
                self.latest[row["index"]] = row

    def take(self, index: str = "0"):
        with self.lock:
            return self.latest.get(index)

    def stop(self) -> None:
        self.process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="cross-vendor telemetry sampler")
    parser.add_argument("out")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--amd-name", default="RX")
    arguments = parser.parse_args()
    amd = AmdCard(arguments.amd_name)
    amd_gen, amd_width = amd_pcie_link()
    nvidia = NvidiaStream(max(100, int(arguments.interval * 1000)))
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_nvidia_stamp = None
    try:
        with out.open("w", encoding="utf-8", newline="") as handle:
            handle.write(HEADER + "\n")
            handle.flush()
            while True:
                if arguments.stop_file and pathlib.Path(arguments.stop_file).exists():
                    break
                started = time.time()
                stamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                row = nvidia.take("0")
                if row and row["timestamp"] != last_nvidia_stamp:
                    last_nvidia_stamp = row["timestamp"]
                    handle.write(f"{row['timestamp']}, 0, NVIDIA GeForce RTX 5090, {row['bus']}, {row['memory_used']}, {row['memory_total']}, "
                                 f"{row['utilization']}, {row['power']}, {row['temperature']}, {row['sm']}, {row['mem']}, , "
                                 f"{row['pcie_gen']}, {row['pcie_width']}, {row['throttle']}\n")
                sample = amd.sample()
                handle.write(f"{stamp}, 1, {amd.name}, {amd.bus}, {sample.get('vram_used_mib', '')}, 24576, {sample.get('activity_gfx', '')}, , "
                             f"{sample.get('temp_edge', '')}, {sample.get('gfxclk', '')}, {sample.get('memclk', '')}, "
                             f"{sample.get('temp_hotspot', '')}, {amd_gen}, {amd_width}, {sample.get('throttler', '')}\n")
                handle.flush()
                time.sleep(max(0.0, arguments.interval - (time.time() - started)))
    finally:
        nvidia.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
