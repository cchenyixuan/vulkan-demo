"""
bringup_check_v6.py — E30 S0: bring-up validation of the v6 solver on a node
(the N56 cluster, or the local rig for the pre-run).

Port of remote/bringup_check.py (v5) with the E30 changes:
  - the case stage never generates anything: it checks that the case's .obj
    files are present (symlinks into the v5 checkout on N56) and never touches
    a case.yaml (2026-09-04 incident: a regenerated 1M case silently changed
    gamma, c and the materials);
  - contexts are created only on the discrete NVIDIA devices (on N56 the
    loader also lists llvmpipe), in the v6 discrete-first order;
  - the single-card stage is the chain bench at K=1 (the path the S3/S4 K=1
    runs use), the second stage K=2 on devices 0,1; both through
    docs/cluster_v6/scripts/run_chain_v6.py (switch interval 0.2 ms, effective
    configuration, init clamp count) and judged by parse_run_v6.py;
  - every child streams its full output to a log file (no capture, so a hang
    or a kill leaves the log), has its own timeout, and the report goes to the
    --log-dir (node-local /dev/shm on N56), never into the checkout.

Stages: env, case, k1, k2, all (env -> case -> k1 -> k2, stop at the first
hard failure).

--solver v7 (E39; e30_lib.sh passes it as $E30_SOLVER_ARGS when E30_SOLVER=v7)
checks experiment/v7 instead: its runtime modules and SPIR-V, VulkanContextV7,
the V7_* environment, and the chain stages run with --solver v7. Default v6.

    python remote/bringup_check_v6.py --stage all --log-dir /dev/shm/scxm138/e30_<job>/bringup
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

SCRIPTS_DIRECTORY = _REPOSITORY_ROOT / "docs" / "cluster_v6" / "scripts"
SOLVERS = ("v6", "v7")
RUNTIME_MODULE_NAMES = ("vulkan_context", "simulator", "orchestrator", "partition", "case_loader", "case",
                        "transport", "bench", "sync_scheme", "phase_trace")


def runtime_modules(solver: str) -> tuple[str, ...]:
    """experiment.<solver>.utils.<name>_<solver> of every runtime module, in the v6 order."""
    return tuple(f"experiment.{solver}.utils.{name}_{solver}" for name in RUNTIME_MODULE_NAMES)


RUNTIME_MODULES = runtime_modules("v6")
SHADER_MODULES = ("bootstrap_half_kick", "initialize_voxelization", "predict", "update_voxel", "ghost_send",
                  "install_migrations", "correction", "density", "force", "defrag", "append_departed",
                  "expand_ghost_lists", "band_compact")
NVIDIA_VENDOR_ID = 0x10DE


class Report:
    def __init__(self, log_directory: pathlib.Path):
        self.log_directory = log_directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.path = log_directory / "bringup_report.txt"

    def log(self, text: str) -> None:
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def run_streamed(report: Report, command: list[str], log_name: str, timeout_seconds: int) -> tuple[int, pathlib.Path]:
    """Run a child with its stdout+stderr streamed to log_directory/log_name."""
    log_path = report.log_directory / log_name
    report.log(f"  $ {' '.join(command)}   > {log_path}")
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, timeout=timeout_seconds)
            return completed.returncode, log_path
        except subprocess.TimeoutExpired:
            report.log(f"  TIMEOUT after {timeout_seconds}s (partial log kept)")
            return 124, log_path
        except FileNotFoundError as error:
            report.log(f"  NOT FOUND: {error}")
            return 127, log_path


def run_captured(command: list[str], timeout_seconds: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds,
                                   encoding="utf-8", errors="replace")
        return completed.returncode, (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except FileNotFoundError as error:
        return 127, repr(error)


def count_nvidia_discrete_devices() -> int:
    import vulkan as vk
    application_info = vk.VkApplicationInfo(pApplicationName="bringup_count", applicationVersion=1,
                                            pEngineName="bringup", engineVersion=1,
                                            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0))
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=application_info), None)
    try:
        count = 0
        for physical_device in vk.vkEnumeratePhysicalDevices(instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical_device)
            if (properties.vendorID == NVIDIA_VENDOR_ID
                    and properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU):
                count += 1
        return count
    finally:
        vk.vkDestroyInstance(instance, None)


def stage_environment(report: Report, arguments: argparse.Namespace) -> bool:
    report.log(f"\n===== ENV STAGE {datetime.datetime.now().isoformat(timespec='seconds')} =====")
    report.log(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    report.log(f"platform : {platform.platform()}  host={platform.node()}")
    commit_file = _REPOSITORY_ROOT / "COMMIT"
    if commit_file.exists():
        report.log(f"COMMIT   : {commit_file.read_text(encoding='utf-8').strip()}")
    elif (_REPOSITORY_ROOT / ".git").exists():
        code, output = run_captured(["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"])
        report.log(f"git HEAD : {output.strip()}")
    solver = arguments.solver
    prefix = solver.upper() + "_"
    variables = sorted(name for name in os.environ if name.startswith(prefix))
    report.log(f"{prefix}* in environment: {', '.join(variables) if variables else 'none'}")

    ok = True
    for module_name in ("numpy", "yaml", "vulkan", "cffi"):
        try:
            module = __import__(module_name)
            report.log(f"import {module_name:<10}: OK ({getattr(module, '__version__', '?')})")
        except Exception as error:                                     # noqa: BLE001
            report.log(f"import {module_name:<10}: FAIL — {error!r}")
            ok = False
    for module_name in runtime_modules(solver):
        try:
            __import__(module_name)
            report.log(f"import {module_name}: OK")
        except Exception as error:                                     # noqa: BLE001
            report.log(f"import {module_name}: FAIL — {error!r}")
            ok = False

    if sys.platform.startswith("linux"):
        import ctypes.util
        loader = ctypes.util.find_library("vulkan")
        report.log(f"libvulkan (find_library): {loader or 'NOT FOUND'}  "
                   f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
        for icd_directory in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
            directory = pathlib.Path(icd_directory)
            entries = sorted(entry.name for entry in directory.glob("*.json")) if directory.exists() else []
            report.log(f"ICDs {icd_directory}: {entries or 'none'}")
    try:
        import vulkan as vk
        version = vk.vkEnumerateInstanceVersion()
        report.log(f"Vulkan loader instance version: {version >> 22}.{(version >> 12) & 0x3ff}.{version & 0xfff}")
    except Exception as error:                                         # noqa: BLE001
        report.log(f"vkEnumerateInstanceVersion: FAIL — {error!r}")
        ok = False

    if shutil.which("nvidia-smi"):
        code, output = run_captured(["nvidia-smi",
                                     "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit",
                                     "--format=csv"])
        report.log(output.strip())
        code, output = run_captured(["nvidia-smi", "topo", "-m"])
        report.log(output.strip() if code == 0 else f"nvidia-smi topo -m: rc={code} (not fatal) {output.strip()[:200]}")
    else:
        report.log("nvidia-smi: not on PATH")

    shader_directory = _REPOSITORY_ROOT / "experiment" / solver / "shaders" / "spv"
    missing = []
    for name in SHADER_MODULES:
        path = shader_directory / f"{name}.comp.spv"
        if not path.exists():
            missing.append(name)
            continue
        report.log(f"spv {name}.comp.spv {hashlib.sha256(path.read_bytes()).hexdigest()}")
    if missing:
        report.log(f"SPIR-V missing: {missing}")
        ok = False

    context_class_name = f"VulkanContext{solver.upper()}"
    try:
        context_class = getattr(importlib.import_module(f"experiment.{solver}.utils.vulkan_context_{solver}"),
                                context_class_name)
        device_count = count_nvidia_discrete_devices()
        report.log(f"NVIDIA discrete devices: {device_count}")
        if device_count == 0:
            ok = False
        for device_index in range(device_count):
            context = context_class.create(device_index=device_index, enable_validation=False,
                                           application_name=f"bringup_{solver}")
            split = context.transfer_queue_upload is not context.transfer_queue
            report.log(f"{context_class_name}[{device_index}]: OK — {context.device_name} "
                       f"(compute qf {context.compute_queue_family_index}, "
                       f"transfer qf {context.transfer_queue_family_index}, "
                       f"transfer queues {'2 (split)' if split else '1 (shared)'})")
            if not split:
                report.log(f"  WARNING: device {device_index} runs one shared transfer queue")
            context.destroy()
        if arguments.expected_devices is not None and device_count != arguments.expected_devices:
            report.log(f"device count {device_count} != expected {arguments.expected_devices}")
            ok = False
    except Exception as error:                                         # noqa: BLE001
        report.log(f"{context_class_name} enumeration: FAIL — {error!r}")
        ok = False

    report.log(f"ENV STAGE: {'PASS' if ok else 'FAIL'}")
    return ok


def case_object_files(case_path: pathlib.Path) -> list[pathlib.Path]:
    import yaml
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    geometry = document.get("geometry", {}) or {}
    names = [geometry.get("frame")] + [entry.get("file") for entry in geometry.get("particles", []) or []]
    return [case_path.parent / name for name in names if name]


def stage_case(report: Report, arguments: argparse.Namespace) -> bool:
    report.log("\n===== CASE STAGE (check only; nothing is generated) =====")
    case_path = _REPOSITORY_ROOT / arguments.case
    if not case_path.exists():
        report.log(f"case yaml missing: {case_path}")
        report.log("CASE STAGE: FAIL")
        return False
    report.log(f"case yaml: {case_path} (symlink={case_path.is_symlink()})")
    ok = not case_path.is_symlink()
    if case_path.is_symlink():
        report.log("  case.yaml must be a real file in this checkout (never a link into another checkout)")
    for object_path in case_object_files(case_path):
        if not object_path.exists():
            report.log(f"  MISSING {object_path.name}")
            ok = False
            continue
        target = os.path.realpath(object_path)
        report.log(f"  {object_path.name}: {object_path.stat().st_size:,} B"
                   + (f" -> {target}" if object_path.is_symlink() else ""))
    report.log(f"CASE STAGE: {'PASS' if ok else 'FAIL'}")
    return ok


def chain_stage(report: Report, arguments: argparse.Namespace, label: str, weights: str, device_map: str) -> bool:
    report.log(f"\n===== {label.upper()} STAGE (chain bench, weights {weights}, devices {device_map}, "
               f"{arguments.steps} steps) =====")
    solver_arguments = ["--solver", arguments.solver] if arguments.solver != "v6" else []
    command = [sys.executable, "-u", str(SCRIPTS_DIRECTORY / "run_chain_v6.py"), *solver_arguments, "--",
               "--case", arguments.case, "--weights", weights, "--device-map", device_map,
               "--sync-scheme", "per-direction", "--depth", "2", "--pool-safety", "1.2",
               "--max-steps", str(arguments.steps), "--warmup", str(arguments.warmup),
               "--defrag-cadence", str(arguments.defrag_cadence)]
    start = time.time()
    return_code, log_path = run_streamed(report, command, f"bringup_{label}.log", arguments.timeout)
    end = time.time()
    parse_command = [sys.executable, str(SCRIPTS_DIRECTORY / "parse_run_v6.py"), *solver_arguments,
                     "--log", str(log_path),
                     "--label", f"bringup_{label}", "--rc", str(return_code), "--start", str(start),
                     "--end", str(end), "--results", str(report.log_directory / "bringup_results.jsonl"),
                     "--node", platform.node()]
    parse_code, parse_output = run_captured(parse_command, timeout_seconds=120)
    report.log(parse_output.strip())
    passed = parse_code == 0
    report.log(f"{label.upper()} STAGE: {'PASS' if passed else 'FAIL'} (rc={return_code})")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="E30 S0 bring-up for the v6 solver")
    parser.add_argument("--stage", default="all", choices=["env", "case", "k1", "k2", "all"])
    parser.add_argument("--log-dir", default=str(_REPOSITORY_ROOT / "logs" / "bringup_v6"))
    parser.add_argument("--case", default="cases/lid_driven_cavity_2d_n1000/case.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--defrag-cadence", type=int, default=100,
                        help="in-run defrag every N steps so 300 steps cross the drained-defrag path")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per chain stage")
    parser.add_argument("--expected-devices", type=int, default=None,
                        help="fail the env stage unless exactly this many NVIDIA discrete devices are visible")
    parser.add_argument("--solver", choices=SOLVERS, default="v6",
                        help="solver directory experiment/<solver> to check (E39: v7; default v6)")
    arguments = parser.parse_args()
    os.chdir(_REPOSITORY_ROOT)
    report = Report(pathlib.Path(arguments.log_dir))

    stages = {
        "env": [lambda: stage_environment(report, arguments)],
        "case": [lambda: stage_case(report, arguments)],
        "k1": [lambda: chain_stage(report, arguments, "k1", "1", "0")],
        "k2": [lambda: chain_stage(report, arguments, "k2", "1,1", "0,1")],
    }
    stages["all"] = stages["env"] + stages["case"] + stages["k1"] + stages["k2"]
    for stage in stages[arguments.stage]:
        if not stage():
            report.log(f"\nBRINGUP: FAILED — report: {report.path}")
            return 1
    report.log(f"\nBRINGUP: ALL PASS — report: {report.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
