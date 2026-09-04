"""
build_offline_kit.py — build the AIR-GAPPED deployment kit for the GPU server.

Runs on the Windows dev box (network available). Produces
``dist/v5_offline_kit_<date>.tar.gz`` containing everything the air-gapped
Linux server needs — it can NOT pip/apt/download anything:

    repo.tar            git archive of HEAD (tracked files only)
    spv/                freshly compiled SPIR-V for experiment/v5 shaders
                        (SPIR-V is platform-independent; the server never
                        needs glslc or the Vulkan SDK)
    wheels/cp310..cp313/  manylinux wheels for numpy / PyYAML / vulkan /
                        matplotlib / scipy + pip/setuptools/wheel, one dir
                        per CPython minor version (server version unknown
                        until first contact)
    install_kit.sh      server-side installer (also inside repo.tar; copied
                        to kit root so it runs before extraction)
    bringup_check.py    server-side validation suite (same duplication)

Server-side flow (through the tailscale subnet route):
    rsync dist/v5_offline_kit_*.tar.gz user@172.21.156.9:~/
    ssh user@172.21.156.9
    tar xzf v5_offline_kit_*.tar.gz && cd v5_offline_kit && bash install_kit.sh

    .venv/Scripts/python.exe remote/build_offline_kit.py
    .venv/Scripts/python.exe remote/build_offline_kit.py --skip-wheels   # repo/spv refresh only
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import shutil
import subprocess
import sys
import tarfile

_REPO = pathlib.Path(__file__).resolve().parents[1]

# Server runtime dependencies. Viewers (glfw/pyrr) are deliberately NOT
# included — the server is headless; slice rendering uses matplotlib Agg.
RUNTIME_PACKAGES = ["numpy", "PyYAML", "matplotlib", "scipy"]
BOOTSTRAP_PACKAGES = ["pip", "setuptools", "wheel"]
# python-vulkan publishes a universal (pure-python, cffi ABI-mode) build —
# downloaded separately because --only-binary would reject an sdist.
UNIVERSAL_PACKAGES = ["vulkan"]
PYTHON_MINOR_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


def run(cmd: list[str], **kwargs) -> None:
    print("  $", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wheels", action="store_true",
                        help="reuse previously downloaded wheels/")
    parser.add_argument("--out-dir", default="dist")
    args = parser.parse_args()

    stage = _REPO / args.out_dir / "v5_offline_kit"
    stage.mkdir(parents=True, exist_ok=True)

    print("[kit] 1/4 compile shaders (SPIR-V is platform-independent)")
    run([sys.executable, str(_REPO / "experiment/v5/compile_shaders_v5.py")],
        cwd=_REPO)
    spv_source = _REPO / "experiment/v5/shaders/spv"
    spv_stage = stage / "spv"
    if spv_stage.exists():
        shutil.rmtree(spv_stage)
    shutil.copytree(spv_source, spv_stage)
    spv_count = len(list(spv_stage.glob("*.spv")))
    print(f"[kit] {spv_count} .spv staged")

    print("[kit] 2/4 git archive HEAD -> repo.tar")
    run(["git", "archive", "--format=tar", "-o", str(stage / "repo.tar"),
         "HEAD"], cwd=_REPO)

    if not args.skip_wheels:
        print("[kit] 3/4 download manylinux wheels per CPython version")
        for version in PYTHON_MINOR_VERSIONS:
            tag = "cp" + version.replace(".", "")
            wheel_dir = stage / "wheels" / tag
            wheel_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "pip", "download",
                 "--dest", str(wheel_dir),
                 "--platform", "manylinux2014_x86_64",
                 "--python-version", version,
                 "--only-binary=:all:", "--quiet",
                 *RUNTIME_PACKAGES, *BOOTSTRAP_PACKAGES])
            run([sys.executable, "-m", "pip", "download",
                 "--dest", str(wheel_dir),
                 "--python-version", version,
                 "--no-deps", "--no-binary=:all:", "--quiet",
                 *UNIVERSAL_PACKAGES])
    else:
        print("[kit] 3/4 SKIPPED (reusing wheels/)")

    print("[kit] 4/4 stage server scripts + pack")
    # Force LF: a CRLF install_kit.sh (Windows autocrlf checkout) breaks
    # bash on the server with "\r: command not found".
    for name in ("remote/install_kit.sh", "remote/bringup_check.py"):
        content = (_REPO / name).read_bytes().replace(b"\r\n", b"\n")
        (stage / pathlib.Path(name).name).write_bytes(content)

    stamp = datetime.date.today().strftime("%Y%m%d")
    tarball = _REPO / args.out_dir / f"v5_offline_kit_{stamp}.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(stage, arcname="v5_offline_kit")
    size_mb = tarball.stat().st_size / 1e6
    print(f"[kit] DONE: {tarball}  ({size_mb:.0f} MB)")
    print(f"[kit] ship it:  rsync -P {tarball.name} user@172.21.156.9:~/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
