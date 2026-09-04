"""
bringup_check.py — one-command validation suite for a NEW machine (the
air-gapped Linux GPU server, or any fresh rig).

Everything it learns goes to stdout AND ``bringup_report.txt`` (append),
so a single file can be rsync'd back for remote diagnosis.

Stages (--stage, default ``all``):
    env     python/platform, imports, libvulkan + ICD inventory,
            nvidia-smi + topology, Vulkan device enumeration via the
            project's own VulkanContextV5 (one context per GPU)
    case    generate the 1M 2D cavity if absent (obj files are gitignored,
            so a fresh checkout has only case.yaml)
    single  single-GPU bench, 2000 steps on device 0 (fps + conservation)
    chain   K=2 chain across devices 0,1 (seam + drift; requires >=2 GPUs)
    all     env -> case -> single -> chain, stop at first hard failure

    .venv/bin/python remote/bringup_check.py --stage all
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

REPORT = _REPO / "bringup_report.txt"
BRINGUP_CASE = "cases/lid_driven_cavity_2d/case.yaml"

_ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
}


def log(text: str) -> None:
    print(text, flush=True)
    with open(REPORT, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_logged(cmd: list[str], timeout_s: int = 1800) -> tuple[int, str]:
    log(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=_REPO, env=_ENV, timeout=timeout_s,
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout_s}s")
        return 124, ""
    except FileNotFoundError as error:
        log(f"  NOT FOUND: {error}")
        return 127, ""
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def stage_env() -> bool:
    log(f"\n===== ENV STAGE {datetime.datetime.now().isoformat(timespec='seconds')} =====")
    log(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    log(f"platform : {platform.platform()}")

    ok = True
    for module_name in ("numpy", "yaml", "vulkan", "matplotlib"):
        try:
            module = __import__(module_name)
            log(f"import {module_name:<11}: OK "
                f"({getattr(module, '__version__', '?')})")
        except Exception as error:                     # noqa: BLE001
            hard = module_name != "matplotlib"         # plots are optional
            log(f"import {module_name:<11}: {'FAIL' if hard else 'missing (optional)'} — {error!r}")
            ok = ok and not hard

    if sys.platform.startswith("linux"):
        import ctypes.util
        loader = ctypes.util.find_library("vulkan")
        log(f"libvulkan: {loader or 'NOT FOUND (need vulkan-loader / libvulkan1)'}")
        ok = ok and loader is not None
        for icd_dir in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
            entries = sorted(pathlib.Path(icd_dir).glob("*.json")) \
                if pathlib.Path(icd_dir).exists() else []
            log(f"ICDs {icd_dir}: {[e.name for e in entries] or 'none'}")

    if shutil.which("nvidia-smi"):
        code, output = run_logged(["nvidia-smi",
                                   "--query-gpu=index,name,memory.total,driver_version",
                                   "--format=csv"])
        log(output.strip())
        code, output = run_logged(["nvidia-smi", "topo", "-m"])
        log(output.strip())
    else:
        log("nvidia-smi: not on PATH")

    # Full-path check: create + destroy a context on every physical device.
    try:
        from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5
        device_index = 0
        while True:
            try:
                ctx = VulkanContextV5.create(device_index=device_index,
                                             enable_validation=False,
                                             application_name="bringup")
            except RuntimeError as error:
                if device_index == 0:
                    log(f"VulkanContextV5[0]: FAIL — {error}")
                    ok = False
                break
            log(f"VulkanContextV5[{device_index}]: OK — {ctx.device_name} "
                f"(compute qf {ctx.compute_queue_family_index}, "
                f"transfer qf {ctx.transfer_queue_family_index})")
            ctx.destroy()
            device_index += 1
        log(f"usable Vulkan devices: {device_index}")
    except Exception as error:                          # noqa: BLE001
        log(f"VulkanContextV5 enumeration: FAIL — {error!r}")
        ok = False

    log(f"ENV STAGE: {'PASS' if ok else 'FAIL'}")
    return ok


def stage_case() -> bool:
    log("\n===== CASE STAGE =====")
    case_dir = _REPO / "cases/lid_driven_cavity_2d"
    if (case_dir / "domain.obj").exists():
        log("1M case present — skip generation")
        return True
    code, output = run_logged(
        [sys.executable, "utils/geometry/_demo_cavity_case.py",
         "--half", "500", "--out", "cases/lid_driven_cavity_2d",
         "--no-preview"], timeout_s=900)
    log(output[-1500:])
    log(f"CASE STAGE: {'PASS' if code == 0 else 'FAIL'}")
    return code == 0


def stage_single() -> bool:
    log("\n===== SINGLE-GPU STAGE (2000 steps, device 0) =====")
    code, output = run_logged(
        [sys.executable, "experiment/v5/_run_v5_single_bench.py",
         "--case", BRINGUP_CASE, "--device", "0",
         "--max-steps", "2000", "--warmup", "500",
         "--bench-window", "1500"], timeout_s=1800)
    log(output[-2000:])
    passed = code == 0 and "WARN: alive drift" not in output
    log(f"SINGLE STAGE: {'PASS' if passed else 'FAIL'}")
    return passed


def stage_chain() -> bool:
    log("\n===== CHAIN STAGE (K=2 on devices 0,1; 1500 steps) =====")
    code, output = run_logged(
        [sys.executable, "experiment/v5/_run_v5_chain_bench.py",
         "--case", BRINGUP_CASE, "--weights", "1,1", "--device-map", "0,1",
         "--sync-scheme", "per-direction",
         "--max-steps", "1500", "--warmup", "500"], timeout_s=1800)
    log(output[-2500:])
    passed = code == 0 and "drift=0" in output
    log(f"CHAIN STAGE: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["env", "case", "single", "chain", "all"])
    args = parser.parse_args()

    stages = {"env": [stage_env], "case": [stage_case],
              "single": [stage_single], "chain": [stage_chain],
              "all": [stage_env, stage_case, stage_single, stage_chain]}
    for stage in stages[args.stage]:
        if not stage():
            log(f"\nBRINGUP: FAILED at {stage.__name__} — report: {REPORT}")
            return 1
    log(f"\nBRINGUP: ALL PASS — report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
