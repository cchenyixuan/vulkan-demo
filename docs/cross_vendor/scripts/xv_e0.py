"""xv_e0.py — E0 of the cross-vendor campaign: the platform record and one K = 2 cross-vendor chain on 1000^2
(cavity2d_n1000) under the cluster protocol (calibration pilots, then 1500 / 500 steps, two in flight, step trace on,
seam check on). Writes docs/cross_vendor/data/e0/: platform.json, results.jsonl, logs/, weights/, traces/,
telemetry.csv, summary.json (the numbers the report quotes).

Usage (from the checkout root, the vulkan-demo venv):
    python docs/cross_vendor/scripts/xv_e0.py [--case cavity2d_n1000] [--out docs/cross_vendor/data/e0]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xv_lib  # noqa: E402

POWERSHELL_PLATFORM = r"""
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$board = Get-CimInstance Win32_BaseBoard
$dimms = Get-CimInstance Win32_PhysicalMemory | ForEach-Object { @{capacity_GiB = [math]::Round($_.Capacity / 1GB, 1); speed_MTs = $_.Speed; configured_MTs = $_.ConfiguredClockSpeed; part = $_.PartNumber.Trim() } }
$gpus = Get-CimInstance Win32_VideoController | ForEach-Object { @{name = $_.Name; driver_version = $_.DriverVersion; driver_date = ($_.DriverDate).ToString('yyyy-MM-dd'); pnp = $_.PNPDeviceID } }
$pcie = @{}
foreach ($d in Get-PnpDevice -Class Display) {
  $p = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_PciDevice_CurrentLinkSpeed,DEVPKEY_PciDevice_CurrentLinkWidth,DEVPKEY_PciDevice_MaxLinkSpeed,DEVPKEY_PciDevice_MaxLinkWidth,DEVPKEY_Device_LocationInfo
  $pcie[$d.FriendlyName] = @{ current_gen = $p[0].Data; current_width = $p[1].Data; max_gen = $p[2].Data; max_width = $p[3].Data; location = $p[4].Data }
}
@{ os = @{ caption = $os.Caption; version = $os.Version; build = $os.BuildNumber; total_memory_GiB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1) };
   cpu = @{ name = $cpu.Name.Trim(); cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors; max_clock_MHz = $cpu.MaxClockSpeed };
   board = @{ manufacturer = $board.Manufacturer; product = $board.Product }; dimms = $dimms; gpus_windows = $gpus; pcie = $pcie } | ConvertTo-Json -Depth 5
"""


def windows_platform() -> dict:
    text = subprocess.run(["powershell", "-NoProfile", "-Command", POWERSHELL_PLATFORM], capture_output=True, text=True, timeout=120).stdout
    return json.loads(text)


def vulkan_devices() -> dict:
    script = pathlib.Path(__file__).with_name("xv_probe_devices.py")
    text = subprocess.run([str(xv_lib.PYTHON), str(script)], capture_output=True, text=True, timeout=120, env=xv_lib.environment()).stdout
    return json.loads(text)


def nvidia_smi() -> list:
    text = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
                           "pcie.link.width.current,pcie.link.width.max,memory.total,power.limit,clocks.max.sm,clocks.max.memory",
                           "--format=csv"], capture_output=True, text=True, timeout=60).stdout
    rows = list(csv.reader(text.splitlines()))
    return [dict(zip([name.strip() for name in rows[0]], [value.strip() for value in row])) for row in rows[1:] if row]


def telemetry_ranges(path: pathlib.Path, start: float, end: float) -> dict:
    """Per card: min / median / max of the SM clock, memory clock, temperature, hot-spot, power, PCIe link inside [start, end]."""
    columns = {"clock": "clocks.current.sm", "memory_clock": "clocks.current.memory", "temperature": "temperature.gpu",
               "hotspot": "temperature.hotspot", "power": "power.draw", "utilization": "utilization.gpu", "memory_used": "memory.used",
               "pcie_gen": "pcie.link.gen.current", "pcie_width": "pcie.link.width.current"}
    samples = {}
    with path.open(encoding="utf-8") as handle:
        header = None
        for record in csv.reader(handle):
            if not record:
                continue
            if header is None:
                header = [name.strip() for name in record]
                continue
            values = dict(zip(header, (value.strip() for value in record)))
            try:
                stamp = datetime.datetime.strptime(values["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except (KeyError, ValueError):
                continue
            if not (start <= stamp <= end):
                continue
            entry = samples.setdefault(values["index"], {"name": values.get("name"), **{key: [] for key in columns}})
            for key, prefix in columns.items():
                column = next((name for name in values if name.startswith(prefix)), None)
                try:
                    entry[key].append(float(values[column]))
                except (TypeError, ValueError, KeyError):
                    pass
    out = {}
    for index, entry in samples.items():
        out[index] = {"name": entry["name"], "samples": len(entry["clock"])}
        for key in columns:
            series = entry[key]
            out[index][key] = {"min": min(series), "median": statistics.median(series), "max": max(series)} if series else None
    return out


def trace_summary(trace_directory: pathlib.Path) -> dict:
    """Per-card phase medians and the two links' t_own (E7 definitions: step_trace_model + transfer_queue_idle)."""
    sys.path.insert(0, str(xv_lib.REPOSITORY))
    sys.path.insert(0, str(xv_lib.REPOSITORY / "docs" / "cluster_v6" / "scripts"))
    from experiment.v7.analysis import step_trace_model          # noqa: E402
    import e7_full_summary                                       # noqa: E402
    run = step_trace_model.load_run(trace_directory)
    cards = step_trace_model.cycle_components(run)
    metrics = step_trace_model.link_metrics(run)
    idle = e7_full_summary.transfer_queue_idle(run)
    import numpy as np                                           # noqa: E402
    links = {}
    for name, values in metrics.items():
        steps = values["step"]
        readback_idle = np.array([idle.get((name, int(step)), (0.0, 0.0))[0] for step in steps])
        upload_idle = np.array([idle.get((name, int(step)), (0.0, 0.0))[1] for step in steps])
        t_own = values["t_chain"] - readback_idle - upload_idle
        links[name] = {"steps": int(len(steps)), "sender": values["sender"], "receiver": values["receiver"],
                       "t_chain_us_p50": float(statistics.median(values["t_chain"])),
                       "t_own_us_p50": float(statistics.median(t_own)),
                       "r_own_p50": float(statistics.median(t_own / values["T_B"])),
                       "readback_us_p50": float(statistics.median(values["chain_readback"] - readback_idle)),
                       "host_us_p50": float(statistics.median(values["chain_host"])),
                       "upload_us_p50": float(statistics.median(values["chain_upload"] - upload_idle))}
    return {"cards": [{key: (float(value) if isinstance(value, (int, float)) else value) for key, value in entry.items()} for entry in cards],
            "links": links}


def main() -> int:
    parser = argparse.ArgumentParser(description="E0 cross-vendor smoke + platform record")
    parser.add_argument("--case", default="cavity2d_n1000")
    parser.add_argument("--out", default=str(xv_lib.REPOSITORY / "docs" / "cross_vendor" / "data" / "e0"))
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    experiment = xv_lib.Experiment(out)

    record = {"recorded_at": datetime.datetime.now().isoformat(timespec="seconds"), "provenance": xv_lib.provenance(),
              "windows": windows_platform(), "nvidia_smi": nvidia_smi(), "vulkan": vulkan_devices(),
              "python": {"version": sys.version, "vulkan": importlib.metadata.version("vulkan"), "numpy": importlib.metadata.version("numpy"),
                         "cffi": importlib.metadata.version("cffi"), "executable": str(xv_lib.PYTHON)},
              "protocol": {"steps": xv_lib.STEPS, "warmup": xv_lib.WARMUP, "depth": 2, "device_map": xv_lib.DEVICE_MAP, "common": xv_lib.COMMON,
                           "switch_interval_ms": xv_lib.SWITCH_INTERVAL_MS, "worker_affinity": "unset (default)"}}
    (out / "platform.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print("platform.json written", flush=True)

    experiment.start_telemetry()
    summary = {"case": arguments.case, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        weights_file, cal_row, cal_record = experiment.calibrate(arguments.case, f"cal_{arguments.case}_K2", timeout=900)
        summary["calibration"] = {"status": cal_row["status"], "weights": cal_record["weights"], "cuts": cal_record["cuts"],
                                  "rounds": [{"round": r["round"], "weights": r["weights"], "cuts": r["cuts"], "omega": r["omega"],
                                              "next_cuts": r["next_cuts"], "changed": r["changed"], "busy_us": r["busy_us"], "fluid": r["fluid"]}
                                             for r in cal_record["rounds"]]}
        chain_label = f"{arguments.case}_K2_trace"
        row = experiment.chain(arguments.case, chain_label, weights_file=weights_file, timeout=900, trace=True)
        summary["chain"] = {key: row.get(key) for key in ("label", "status", "rc", "steady_fps", "steady_fps_nodefrag", "reasons", "final_line",
                                                           "alive_start", "alive_end", "seam_checks", "defrag_events_summary", "stages",
                                                           "build_seconds", "telemetry_loop", "dest_guard_fallback", "dest_guard_released",
                                                           "validation_messages", "partition", "solver_switches", "solver_tag")}
        summary["chain"]["invariants"] = {key: row.get(key) for key in row if key.startswith(("overflow", "drift", "far_", "stamp", "nan", "init_clamp"))}
        stages = row.get("stages") or {}
        loop_start = (stages.get("loop_start") or {}).get("epoch")
        loop_end = (stages.get("loop_end") or {}).get("epoch")
        time.sleep(2)
        experiment.stop_telemetry()
        if loop_start and loop_end:
            summary["telemetry_loop_ranges"] = telemetry_ranges(experiment.telemetry, loop_start, loop_end)
        try:
            summary["trace"] = trace_summary(out / "traces" / chain_label)
        except Exception as error:  # noqa: BLE001
            summary["trace_error"] = repr(error)
    except xv_lib.RunFailed as error:
        summary["failed"] = str(error)
        experiment.stop_telemetry()
    finally:
        summary["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str)[:6000])
    return 1 if "failed" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
