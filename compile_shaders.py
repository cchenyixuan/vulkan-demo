import subprocess
import sys
import os
import glob

GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SHADER_DIR = os.path.join(REPO_ROOT, "shaders")

# Phase 1 rendering + particle migration demo shaders (no #include, no sph/)
PHASE1_SHADERS = [
    "particle.vert",
    "particle.frag",
    "particle_update.comp",
]

# SPH V0 compute shaders live under shaders/sph/ and use #include "common.glsl".
# Any *.comp under shaders/sph/ is compiled (excluding names starting with "_"
# which are reserved for smoke tests like _test_common.comp).
SPH_SHADER_DIR = os.path.join(SHADER_DIR, "sph")
SPH_INCLUDE_DIR = SPH_SHADER_DIR


def _run_glslc(source_path: str, output_path: str, include_dirs: list, optimize: bool) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        source_path,
        "-o", output_path,
    ]
    if optimize:
        command.append("-O")
    for include_dir in include_dirs:
        command.extend(["-I", include_dir])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"ERROR compiling {os.path.relpath(source_path, REPO_ROOT)}:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)


def compile_phase1_shaders() -> None:
    for name in PHASE1_SHADERS:
        source = os.path.join(SHADER_DIR, name)
        output = source + ".spv"
        print(f"[phase1] {name} -> {name}.spv")
        _run_glslc(source, output, include_dirs=[], optimize=False)


def compile_sph_shaders(optimize: bool = True) -> None:
    if not os.path.isdir(SPH_SHADER_DIR):
        print(f"[sph] {SPH_SHADER_DIR} does not exist, skipping.")
        return

    sph_sources = sorted(glob.glob(os.path.join(SPH_SHADER_DIR, "*.comp")))
    for source in sph_sources:
        name = os.path.basename(source)
        # Skip files starting with underscore (reserved for smoke tests / drafts)
        if name.startswith("_"):
            print(f"[sph] skip {name} (underscore-prefixed)")
            continue

        output = source + ".spv"
        rel_source = os.path.relpath(source, REPO_ROOT)
        print(f"[sph]    {rel_source}")
        _run_glslc(
            source,
            output,
            include_dirs=[SPH_INCLUDE_DIR],
            optimize=optimize,
        )


def compile_smoke_tests() -> None:
    """Compile any shaders/sph/_test_*.comp as a sanity check for common.glsl."""
    if not os.path.isdir(SPH_SHADER_DIR):
        return
    test_sources = sorted(glob.glob(os.path.join(SPH_SHADER_DIR, "_test_*.comp")))
    for source in test_sources:
        name = os.path.basename(source)
        output = source + ".spv"
        rel_source = os.path.relpath(source, REPO_ROOT)
        print(f"[smoke]  {rel_source}")
        _run_glslc(
            source,
            output,
            include_dirs=[SPH_INCLUDE_DIR],
            optimize=False,
        )


if __name__ == "__main__":
    compile_phase1_shaders()
    compile_smoke_tests()
    compile_sph_shaders(optimize=True)
    print("All shaders compiled successfully.")
