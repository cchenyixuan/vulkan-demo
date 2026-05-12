"""
compile_shaders_v2.py — compile all V2 compute shaders.

Mirrors experiment/v1/compile_shaders_v1.py but operates on
experiment/v2/shaders/*.comp and emits experiment/v2/shaders/spv/.

Same glslc flags as V0 / V1 (`--target-env=vulkan1.2`, `-O`, include path
into the V2 shader dir for `common.glsl` / `helpers.glsl`). Keeping target
env at vulkan1.2 matches VkApplicationInfo.apiVersion and pins SPIR-V at
1.5 — vulkan1.3 trips validation + slows down ~15%.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v2/compile_shaders_v2.py
"""

import glob
import os
import subprocess
import sys


GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"

V2_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V2_SPV_DIR = V2_SHADER_DIR + "/spv"


def _run_glslc(source: str, output: str) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        "-O",
        "-I", V2_SHADER_DIR,
        source,
        "-o", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v2_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V2_SHADER_DIR):
        sys.exit(f"V2 shader dir not found: {V2_SHADER_DIR}")
    os.makedirs(V2_SPV_DIR, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V2_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v2] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V2_SPV_DIR, f"{name}.spv")
        print(f"[v2] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    print(f"[v2] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v2_shaders()
