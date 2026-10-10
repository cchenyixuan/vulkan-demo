"""
_collect_platform.py — the measurement platform's facts for the paper's hardware description ("§3 补测" item 1):
CPU, memory, Windows build, both RTX 5090s (PCIe link, memory, power limit, display), driver, Vulkan loader /
SDK / per-device apiVersion and driverVersion, Python and package versions, glslc. Writes platform.json and
platform.md (a paragraph + a table). Sources: WMI (Win32_Processor, Win32_PhysicalMemory, Win32_OperatingSystem,
the CurrentVersion registry key, Win32_BaseBoard), nvidia-smi --query-gpu, vulkaninfo --summary, the file version
of vulkan-1.dll, VULKAN_SDK + glslc --version, importlib.metadata, git.

    python experiment/v7_single/_collect_platform.py --out docs/seam_audit/v7_single/platform.json \
        --out-md docs/seam_audit/v7_single/platform.md [--telemetry logs/v7_single/section3/telemetry.csv]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"


def run(command: list[str], timeout: int = 120) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                               errors="replace")
    return completed.stdout


def powershell_json(script: str):
    text = run(["powershell", "-NoProfile", "-Command", script + " | ConvertTo-Json -Depth 3"])
    text = text.strip()
    return json.loads(text) if text else None


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def wmi() -> dict:
    cpu = as_list(powershell_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                                  "NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize"))
    dimms = as_list(powershell_json("Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,PartNumber,"
                                    "Capacity,Speed,ConfiguredClockSpeed,DeviceLocator,SMBIOSMemoryType"))
    os_info = powershell_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,"
                              "OSArchitecture")
    version = powershell_json("Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion' | "
                              "Select-Object DisplayVersion,CurrentBuild,UBR")
    board = powershell_json("Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product")
    system = powershell_json("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory")
    return {"cpu": [{"name": c["Name"].strip(), "cores": c["NumberOfCores"], "threads": c["NumberOfLogicalProcessors"],
                     "max_clock_mhz_wmi": c["MaxClockSpeed"], "l2_cache_kb": c["L2CacheSize"], "l3_cache_kb": c["L3CacheSize"]}
                    for c in cpu],
            "memory_modules": [{"manufacturer": (d["Manufacturer"] or "").strip(), "part_number": (d["PartNumber"] or "").strip(),
                                "capacity_bytes": d["Capacity"], "speed_mt_s": d["Speed"],
                                "configured_mt_s": d["ConfiguredClockSpeed"], "slot": d["DeviceLocator"],
                                "smbios_memory_type": d.get("SMBIOSMemoryType")} for d in dimms],
            "total_physical_memory_bytes": system["TotalPhysicalMemory"] if system else None,
            "os": {"caption": os_info["Caption"], "version": os_info["Version"], "build": os_info["BuildNumber"],
                   "architecture": os_info["OSArchitecture"], "display_version": version["DisplayVersion"],
                   "full_build": f"{version['CurrentBuild']}.{version['UBR']}"},
            "board": {"manufacturer": board["Manufacturer"], "product": board["Product"]}}


def nvidia() -> dict:
    fields = ["index", "name", "uuid", "driver_version", "pci.bus_id", "pcie.link.gen.max", "pcie.link.gen.gpumax",
              "pcie.link.gen.current", "pcie.link.width.max", "pcie.link.width.current", "memory.total",
              "power.limit", "power.default_limit", "power.max_limit", "display_active", "clocks.max.sm",
              "clocks.max.memory", "vbios_version"]
    text = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        record = dict(zip(fields, values))
        record["uuid_hex"] = record["uuid"].replace("GPU-", "").replace("-", "").lower()
        record["role"] = "headless (measured)" if record["uuid_hex"] == HEADLESS_5090_UUID else "display"
        gpus.append(record)
    cuda = re.search(r"CUDA Version\s*:\s*([\d.]+)", run(["nvidia-smi", "-q", "-i", "0"]))
    return {"gpus": gpus, "cuda_version": cuda.group(1) if cuda else None}


def vulkan() -> dict:
    text = run(["vulkaninfo", "--summary"])
    instance = re.search(r"Vulkan Instance Version:\s*([\d.]+)", text)
    devices = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU") and stripped.endswith(":"):
            current = {"id": stripped.rstrip(":")}
            devices.append(current)
        elif current is not None and "=" in stripped:
            key, _, value = stripped.partition("=")
            key = key.strip()
            if key in ("apiVersion", "driverVersion", "deviceName", "driverName", "driverInfo", "conformanceVersion",
                       "deviceUUID", "deviceType"):
                current[key] = value.strip()
    loader = run(["powershell", "-NoProfile", "-Command",
                  "(Get-Item 'C:\\Windows\\System32\\vulkan-1.dll').VersionInfo.FileVersion"]).strip()
    sdk = os.environ.get("VULKAN_SDK", "")
    glslc = run([str(pathlib.Path(sdk) / "Bin" / "glslc.exe"), "--version"]).strip() if sdk else ""
    return {"instance_version": instance.group(1) if instance else None, "loader_dll_version": loader,
            "sdk": sdk, "sdk_version": pathlib.Path(sdk).name if sdk else None,
            "glslc_version": [line.strip() for line in glslc.splitlines() if line.strip()], "devices": devices}


def python_versions() -> dict:
    import importlib.metadata as metadata
    packages = {}
    for name in ("vulkan", "numpy", "scipy", "matplotlib", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "python_full": sys.version, "executable": sys.executable,
            "packages": packages, "platform": platform.platform()}


def git() -> dict:
    def out(*arguments):
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": out("rev-parse", "HEAD"), "describe": out("describe", "--tags", "--always"),
            "branch": out("rev-parse", "--abbrev-ref", "HEAD")}


def telemetry_link(path: pathlib.Path, index: int = 1) -> dict | None:
    """PCIe link state and SM clock of the measured card while loaded (power >= 300 W) in a campaign's telemetry."""
    if not path.exists():
        return None
    import csv
    gens, widths, clocks, powers = [], [], [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 10 or not row[1].strip().isdigit() or int(row[1]) != index:
                continue
            try:
                power = float(row[4].split()[0])
                clock = float(row[2].split()[0])
            except ValueError:
                continue
            if power < 300:
                continue
            gens.append(row[8].strip()); widths.append(row[9].strip()); clocks.append(clock); powers.append(power)
    if not clocks:
        return None
    from collections import Counter
    return {"samples_loaded": len(clocks), "pcie_gen_current_mode": Counter(gens).most_common(1)[0][0],
            "pcie_width_current_mode": Counter(widths).most_common(1)[0][0],
            "sm_mhz_mean": sum(clocks) / len(clocks), "power_w_mean": sum(powers) / len(powers),
            "source": str(path)}


def markdown(p: dict) -> str:
    cpu = p["wmi"]["cpu"][0]
    dimms = p["wmi"]["memory_modules"]
    os_info = p["wmi"]["os"]
    measured = next((g for g in p["nvidia"]["gpus"] if g["role"].startswith("headless")), p["nvidia"]["gpus"][0])
    display = next((g for g in p["nvidia"]["gpus"] if g["role"] == "display"), None)
    nv_device = next((d for d in p["vulkan"]["devices"] if d.get("deviceUUID", "").replace("-", "") == measured["uuid_hex"]), {})
    total_gib = p["wmi"]["total_physical_memory_bytes"] / 2**30
    dimm_text = ", ".join(f"{d['capacity_bytes'] / 2**30:.0f} GB {d['part_number']} ({d['slot']}, {d['configured_mt_s']} MT/s"
                          + (f" of {d['speed_mt_s']} rated" if d["speed_mt_s"] != d["configured_mt_s"] else "") + ")"
                          for d in dimms)
    link = p.get("telemetry_link")
    lines = ["# 平台(v7_single 测量机器,§3 补测第 1 项)", "",
             f"采集时间 {p['collected']},提交 {p['git']['describe']}(`{p['git']['commit'][:12]}`)。",
             "",
             f"**一句话**:{cpu['name']}({cpu['cores']} 核 {cpu['threads']} 线程,WMI 最大时钟 {cpu['max_clock_mhz_wmi']} MHz,"
             f"L3 {cpu['l3_cache_kb'] // 1024} MB),{total_gib:.1f} GiB 内存({dimm_text}),"
             f"{p['wmi']['board']['manufacturer']} {p['wmi']['board']['product']},"
             f"{os_info['caption']} {os_info['display_version']}(build {os_info['full_build']});"
             f"两张 {measured['name']}(各 {measured['memory.total']},功耗上限 {measured['power.limit']},"
             f"PCIe Gen {measured['pcie.link.gen.max']} x{measured['pcie.link.width.max']} 槽位、当前链路宽度 x{measured['pcie.link.width.current']}),"
             f"测量用的是无头的那张(PCI {measured['pci.bus_id']},uuid {measured['uuid_hex'][:8]}…"
             + (f";显示器接在 PCI {display['pci.bus_id']} 的那张" if display else "") + ");"
             f"驱动 {measured['driver_version']}(Vulkan driverVersion {nv_device.get('driverVersion', '?')},"
             f"apiVersion {nv_device.get('apiVersion', '?')},conformance {nv_device.get('conformanceVersion', '?')}),"
             f"Vulkan loader {p['vulkan']['loader_dll_version']}(SDK {p['vulkan']['sdk_version']},"
             f"vulkaninfo 实例版本 {p['vulkan']['instance_version']});"
             f"Python {p['python']['python']},python-vulkan {p['python']['packages']['vulkan']},"
             f"numpy {p['python']['packages']['numpy']},scipy {p['python']['packages']['scipy']},"
             f"glslc {p['vulkan']['glslc_version'][0] if p['vulkan']['glslc_version'] else '?'}。", ""]
    if link:
        lines += [f"负载下(功耗 ≥ 300 W 的遥测样本 {link['samples_loaded']} 个,`{link['source']}`):"
                  f"PCIe 链路 Gen {link['pcie_gen_current_mode']} x{link['pcie_width_current_mode']},"
                  f"SM 平均 {link['sm_mhz_mean']:.0f} MHz,功耗平均 {link['power_w_mean']:.0f} W。", ""]
    lines += ["| 项 | 值 |", "|---|---|",
              f"| CPU | {cpu['name']}:{cpu['cores']} 核 / {cpu['threads']} 线程,WMI MaxClockSpeed {cpu['max_clock_mhz_wmi']} MHz,"
              f"L2 {cpu['l2_cache_kb'] // 1024} MB,L3 {cpu['l3_cache_kb'] // 1024} MB |",
              f"| 内存 | {total_gib:.1f} GiB 可用;{dimm_text} |",
              f"| 主板 | {p['wmi']['board']['manufacturer']} {p['wmi']['board']['product']} |",
              f"| Windows | {os_info['caption']} {os_info['display_version']},版本 {os_info['version']},build {os_info['full_build']},{os_info['architecture']} |"]
    for g in p["nvidia"]["gpus"]:
        lines.append(f"| GPU {g['index']}({g['role']}) | {g['name']},PCI {g['pci.bus_id']},uuid {g['uuid_hex']},"
                     f"显存 {g['memory.total']},功耗上限 {g['power.limit']}(默认 {g['power.default_limit']},最大 {g['power.max_limit']}),"
                     f"PCIe Gen max {g['pcie.link.gen.max']}(GPU max {g['pcie.link.gen.gpumax']}),采集时 Gen {g['pcie.link.gen.current']};"
                     f"宽度 max x{g['pcie.link.width.max']},当前 x{g['pcie.link.width.current']};显示输出 {g['display_active']};"
                     f"最高 SM / 显存时钟 {g['clocks.max.sm']} / {g['clocks.max.memory']};VBIOS {g['vbios_version']} |")
    lines.append(f"| 驱动 | {measured['driver_version']}(CUDA {p['nvidia']['cuda_version']}) |")
    for d in p["vulkan"]["devices"]:
        lines.append(f"| vulkaninfo {d['id']} | {d.get('deviceName')}:apiVersion {d.get('apiVersion')},driverVersion {d.get('driverVersion')},"
                     f"driverInfo {d.get('driverInfo')},conformance {d.get('conformanceVersion')},uuid {d.get('deviceUUID')} |")
    lines += [f"| Vulkan loader / SDK | vulkan-1.dll {p['vulkan']['loader_dll_version']};VULKAN_SDK {p['vulkan']['sdk']};"
              f"vulkaninfo 实例版本 {p['vulkan']['instance_version']} |",
              f"| glslc | {'; '.join(p['vulkan']['glslc_version'])} |",
              f"| Python | {p['python']['python']}({p['python']['executable']});"
              + ", ".join(f"{name} {version}" for name, version in p["python"]["packages"].items()) + " |",
              f"| 提交 | {p['git']['branch']} {p['git']['describe']} {p['git']['commit']} |", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/seam_audit/v7_single/platform.json")
    parser.add_argument("--out-md", default="docs/seam_audit/v7_single/platform.md")
    parser.add_argument("--telemetry", default=None, help="a campaign telemetry.csv for the loaded link state")
    arguments = parser.parse_args()
    record = {"collected": datetime.datetime.now().isoformat(timespec="seconds"), "wmi": wmi(), "nvidia": nvidia(),
              "vulkan": vulkan(), "python": python_versions(), "git": git(),
              "telemetry_link": telemetry_link(pathlib.Path(arguments.telemetry)) if arguments.telemetry else None}
    out = _REPOSITORY_ROOT / arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    (_REPOSITORY_ROOT / arguments.out_md).write_text(markdown(record), encoding="utf-8")
    print(f"[platform] -> {out}, {arguments.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
