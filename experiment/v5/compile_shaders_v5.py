"""
compile_shaders_v5.py — compile all V5 compute shaders.

Mirrors experiment/v1/compile_shaders_v1.py but operates on
experiment/v5/shaders/*.comp and emits experiment/v5/shaders/spv/.

Same glslc flags as V0 / V1 (`--target-env=vulkan1.2`, `-O`, include path
into the V5 shader dir for `common.glsl` / `helpers.glsl`). Keeping target
env at vulkan1.2 matches VkApplicationInfo.apiVersion and pins SPIR-V at
1.5 — vulkan1.3 trips validation + slows down ~15%.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v5/compile_shaders_v5.py
"""

import glob
import os
import subprocess
import sys


def _find_glslc() -> str:
    """Locate glslc cross-platform: $VULKAN_SDK layout first (Windows Bin/
    glslc.exe, Linux bin/glslc or x86_64/bin/glslc), then PATH. On the
    air-gapped server glslc is NOT required at all — the offline kit ships
    precompiled .spv (SPIR-V is platform-independent); this script only
    runs where shaders are (re)compiled."""
    import shutil as _shutil
    sdk = os.environ.get("VULKAN_SDK")
    candidates = []
    if sdk:
        candidates += [sdk + "/Bin/glslc.exe", sdk + "/bin/glslc",
                       sdk + "/x86_64/bin/glslc"]
    else:
        candidates += ["C:/VulkanSDK/1.4.341.1/Bin/glslc.exe",
                       "C:/VulkanSDK/1.4.350.0/Bin/glslc.exe"]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    from_path = _shutil.which("glslc")
    if from_path:
        return from_path
    return candidates[0]   # keep the old error message path


GLSLC = _find_glslc()

V5_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V5_SPV_DIR = V5_SHADER_DIR + "/spv"


def _run_glslc(source: str, output: str) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        "-O",
        "-I", V5_SHADER_DIR,
        source,
        "-o", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v5_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V5_SHADER_DIR):
        sys.exit(f"V5 shader dir not found: {V5_SHADER_DIR}")
    os.makedirs(V5_SPV_DIR, exist_ok=True)
    render_spv_dir = os.path.join(V5_SPV_DIR, "render")
    os.makedirs(render_spv_dir, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V5_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v4] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V5_SPV_DIR, f"{name}.spv")
        print(f"[v4] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    # Render shaders (.vert / .frag) live in shaders/render/
    render_sources = sorted(
        glob.glob(os.path.join(V5_SHADER_DIR, "render", "*.vert"))
        + glob.glob(os.path.join(V5_SHADER_DIR, "render", "*.frag")))
    for source in render_sources:
        name = os.path.basename(source)
        output = os.path.join(render_spv_dir, f"{name}.spv")
        print(f"[v4/render] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    print(f"[v4] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v5_shaders()
