"""
compile_shaders_v8.py — compile the experiment/v8 kernels (the v7 kernels + the periodic domain) and write the
SPIR-V manifest experiment/v7_single's loader verifies (spirv_manifest.verify_directory on this directory).

Same glslc flags as experiment/v7/compile_shaders_v7.py (`--target-env=vulkan1.2 -O`, include path = this
directory). The sources are the files of experiment/v8/shaders: every *.comp plus the -D variants below.

    .venv/Scripts/python.exe experiment/v8/compile_shaders_v8.py
"""

from __future__ import annotations

import glob
import hashlib
import os
import pathlib
import subprocess
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.v7.compile_shaders_v7 import GLSLC    # noqa: E402  (the same compiler lookup)

V8_SHADER_DIRECTORY = pathlib.Path(__file__).resolve().parent / "shaders"
V8_SPV_DIRECTORY = V8_SHADER_DIRECTORY / "spv"
MANIFEST_NAME = "MANIFEST.txt"

# (output name, source, macros): v7's SHADER_VARIANTS that a single slab can load (force_segment is a
# multi-GPU phase-C variant and is not built here).
SHADER_VARIANTS = (
    ("correction_deep_wall_skip", "correction.comp", ("DEEP_WALL_SKIP",)),
    ("density_deep_wall_skip", "density.comp", ("DEEP_WALL_SKIP",)),
    ("correction_density_deep_wall_skip", "correction_density.comp", ("DEEP_WALL_SKIP",)),
)


def glslc_command(source: pathlib.Path, output: pathlib.Path, macros=()) -> list[str]:
    return [GLSLC, "--target-env=vulkan1.2", "-O", *(f"-D{macro}" for macro in macros),
            "-I", str(V8_SHADER_DIRECTORY), str(source), "-o", str(output)]


def run_glslc(source: pathlib.Path, output: pathlib.Path, macros=()) -> None:
    result = subprocess.run(glslc_command(source, output, macros), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {source.name}:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def build_list() -> list[tuple[str, str, tuple[str, ...]]]:
    """(output name, source file name, macros) of every build, sources sorted, variants after."""
    builds = []
    for source in sorted(glob.glob(str(V8_SHADER_DIRECTORY / "*.comp"))):
        name = os.path.basename(source)
        if name.startswith("_"):
            continue
        builds.append((name[:-len(".comp")], name, ()))
    builds += list(SHADER_VARIANTS)
    return builds


def glslc_version() -> str:
    result = subprocess.run([GLSLC, "--version"], capture_output=True, text=True)
    return result.stdout.strip()


def write_manifest(builds: list[tuple[str, str, tuple[str, ...]]], git_commit: str) -> pathlib.Path:
    lines = [
        "SPIR-V manifest of experiment/v8/shaders/spv (v8: the v7 kernels + the periodic domain, experiment/v8/README.md)",
        "",
        f"compiler: {GLSLC}; glslc --version:",
        *(f"  {line}" for line in glslc_version().splitlines()),
        "flags: glslc --target-env=vulkan1.2 -O [-D<macro>] -I experiment/v8/shaders experiment/v8/shaders/<source> "
        "-o experiment/v8/shaders/spv/<name>.comp.spv",
        "  (experiment/v8/compile_shaders_v8.py: every *.comp whose name does not start with '_', plus the -D variants of",
        "  SHADER_VARIANTS)",
        f"sources: experiment/v8/shaders/*.comp, *.glsl at commit {git_commit}",
        "scope: the compute SPIR-V of experiment/v8. Regenerate with any shader change (experiment/v7_single's",
        "  spirv_manifest.verify_directory refuses a file whose size or sha256 differs).",
        "",
        "sha256                                                             bytes  file  (source)",
    ]
    for output_name, source_name, macros in builds:
        path = V8_SPV_DIRECTORY / f"{output_name}.comp.spv"
        payload = path.read_bytes()
        source_text = source_name + "".join(f" -D{macro}" for macro in macros)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {len(payload):6d}  {path.name}  ({source_text})")
    manifest = V8_SPV_DIRECTORY / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=_REPOSITORY_ROOT, check=True).stdout.strip() or "(unknown)"
    except Exception:
        return "(unknown)"


def compile_v8_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK.")
    V8_SPV_DIRECTORY.mkdir(parents=True, exist_ok=True)
    builds = build_list()
    for output_name, source_name, macros in builds:
        print(f"[v8] {output_name}" + (f" ({source_name} -D{' -D'.join(macros)})" if macros else ""))
        run_glslc(V8_SHADER_DIRECTORY / source_name, V8_SPV_DIRECTORY / f"{output_name}.comp.spv", macros)
    manifest = write_manifest(builds, git_head())
    print(f"[v8] compiled {len(builds)} shaders; manifest {manifest}")


if __name__ == "__main__":
    compile_v8_shaders()
