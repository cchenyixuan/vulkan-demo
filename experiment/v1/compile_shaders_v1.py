"""
compile_shaders_v1.py — compile all V1 compute shaders.

Mirrors compile_shaders.py at repo root but operates on
experiment/v1/shaders/*.comp and emits experiment/v1/shaders/spv/.

Uses the SAME glslc flags as the V0 compile path (`--target-env=vulkan1.2`,
`-O`, includes pointed at the V1 shader dir for `common.glsl` /
`helpers.glsl`). Same target env as V0 keeps SPIR-V version at 1.5, which
the application's VkApplicationInfo.apiVersion=Vulkan1.2 expects — avoids
"SPIR-V 1.6 vs Vulkan 1.2" validation errors that we hit when V1 was
mistakenly compiled with --target-env=vulkan1.3.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v1/compile_shaders_v1.py
"""

import glob
import os
import subprocess
import sys


GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"

V1_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V1_SPV_DIR = V1_SHADER_DIR + "/spv"


def _run_glslc(source: str, output: str) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        "-O",
        "-I", V1_SHADER_DIR,
        source,
        "-o", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v1_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V1_SHADER_DIR):
        sys.exit(f"V1 shader dir not found: {V1_SHADER_DIR}")
    os.makedirs(V1_SPV_DIR, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V1_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v1] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V1_SPV_DIR, f"{name}.spv")
        print(f"[v1] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    print(f"[v1] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v1_shaders()
