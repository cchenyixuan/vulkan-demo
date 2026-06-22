"""
compile_shaders_v4.py — compile all V4 compute shaders.

Mirrors experiment/v1/compile_shaders_v1.py but operates on
experiment/v4/shaders/*.comp and emits experiment/v4/shaders/spv/.

Same glslc flags as V0 / V1 (`--target-env=vulkan1.2`, `-O`, include path
into the V4 shader dir for `common.glsl` / `helpers.glsl`). Keeping target
env at vulkan1.2 matches VkApplicationInfo.apiVersion and pins SPIR-V at
1.5 — vulkan1.3 trips validation + slows down ~15%.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v4/compile_shaders_v4.py
"""

import glob
import os
import subprocess
import sys


GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"

V4_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V4_SPV_DIR = V4_SHADER_DIR + "/spv"


def _run_glslc(source: str, output: str) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        "-O",
        "-I", V4_SHADER_DIR,
        source,
        "-o", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v4_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V4_SHADER_DIR):
        sys.exit(f"V4 shader dir not found: {V4_SHADER_DIR}")
    os.makedirs(V4_SPV_DIR, exist_ok=True)
    render_spv_dir = os.path.join(V4_SPV_DIR, "render")
    os.makedirs(render_spv_dir, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V4_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v4] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V4_SPV_DIR, f"{name}.spv")
        print(f"[v4] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    # Render shaders (.vert / .frag) live in shaders/render/
    render_sources = sorted(
        glob.glob(os.path.join(V4_SHADER_DIR, "render", "*.vert"))
        + glob.glob(os.path.join(V4_SHADER_DIR, "render", "*.frag")))
    for source in render_sources:
        name = os.path.basename(source)
        output = os.path.join(render_spv_dir, f"{name}.spv")
        print(f"[v4/render] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    print(f"[v4] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v4_shaders()
