"""
compile_shaders_v7.py — compile all V5 compute shaders.

Mirrors experiment/v1/compile_shaders_v1.py but operates on
experiment/v7/shaders/*.comp and emits experiment/v7/shaders/spv/.

Same glslc flags as V0 / V1 (`--target-env=vulkan1.2`, `-O`, include path
into the V5 shader dir for `common.glsl` / `helpers.glsl`). Keeping target
env at vulkan1.2 matches VkApplicationInfo.apiVersion and pins SPIR-V at
1.5 — vulkan1.3 trips validation + slows down ~15%.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v7/compile_shaders_v7.py
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

V7_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V7_SPV_DIR = V7_SHADER_DIR + "/spv"


# E39 B4: (output name, source, preprocessor macros): a source compiled a
# second time with -D<macro> into spv/<output name>.comp.spv. The variant's code
# sits in the source behind #ifdef, so the plain build of the source (no macro)
# is the same SPIR-V as before.
SHADER_VARIANTS = (
    ("correction_deep_wall_skip", "correction.comp", ("DEEP_WALL_SKIP",)),
    ("density_deep_wall_skip", "density.comp", ("DEEP_WALL_SKIP",)),
    # E39 B1: the fused correction + density kernel with B4's deep-wall skip
    ("correction_density_deep_wall_skip", "correction_density.comp", ("DEEP_WALL_SKIP",)),
    # E39 B3: a workgroup segment of force_deep_interior_scratch (spec constant 115 = its first thread)
    ("force_segment", "force.comp", ("FORCE_SEGMENT",)),
    # wall_boundary crossing (case.yaml numerics.wall_boundary: crossing; plan C, the static wall store): the fused
    # correction + density pass and force with a fluid segment (the pool's neighbours) and a wall segment (the wall
    # store's), shaders/wall_boundary.glsl. A crossing slab builds every correction_density_* and force_* pipeline
    # from these two modules; spec const 131 (the dispatch class) gives organisation (A) one pipeline or (B) the
    # interior and near-wall pipelines. simple / adami keep the plain modules.
    ("correction_density_wall_crossing", "correction_density.comp", ("WALL_CROSSING",)),
    ("force_wall_crossing", "force.comp", ("WALL_CROSSING",)),
    # DIAGNOSTIC (V7_WALL_CROSSING_DIFFUSION_SUMMED=1): the density diffusion of the wall neighbours applied after the
    # loop as the particle-pool version did, to measure the pair-by-pair form against it (claude/wall_store_report.md)
    ("correction_density_wall_crossing_diffusion_summed", "correction_density.comp",
     ("WALL_CROSSING", "WALL_CROSSING_DIFFUSION_SUMMED")),
)


def glslc_command(source: str, output: str, macros=()) -> list:
    """The glslc invocation of every build (this script and the SPIR-V
    freshness check of _test_seam_layout.py)."""
    return [GLSLC, "--target-env=vulkan1.2", "-O", *(f"-D{macro}" for macro in macros),
            "-I", V7_SHADER_DIR, source, "-o", output]


def _run_glslc(source: str, output: str, macros=()) -> None:
    command = glslc_command(source, output, macros)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v7_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V7_SHADER_DIR):
        sys.exit(f"V5 shader dir not found: {V7_SHADER_DIR}")
    os.makedirs(V7_SPV_DIR, exist_ok=True)
    render_spv_dir = os.path.join(V7_SPV_DIR, "render")
    os.makedirs(render_spv_dir, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V7_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v4] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V7_SPV_DIR, f"{name}.spv")
        print(f"[v4] {name}")
        _run_glslc(source, output)
        n_compiled += 1
    for variant_name, source_name, macros in SHADER_VARIANTS:
        output = os.path.join(V7_SPV_DIR, f"{variant_name}.comp.spv")
        print(f"[v4] {variant_name} ({source_name} -D{' -D'.join(macros)})")
        _run_glslc(os.path.join(V7_SHADER_DIR, source_name), output, macros)
        n_compiled += 1

    # Render shaders (.vert / .frag) live in shaders/render/
    render_sources = sorted(
        glob.glob(os.path.join(V7_SHADER_DIR, "render", "*.vert"))
        + glob.glob(os.path.join(V7_SHADER_DIR, "render", "*.frag")))
    for source in render_sources:
        name = os.path.basename(source)
        output = os.path.join(render_spv_dir, f"{name}.spv")
        print(f"[v4/render] {name}")
        _run_glslc(source, output)
        n_compiled += 1

    print(f"[v4] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v7_shaders()
