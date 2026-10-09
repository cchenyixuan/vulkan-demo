"""
spirv_manifest.py — the kernels of v7_single are experiment/v7's SPIR-V, verified against its MANIFEST.txt.

v7_single compiles nothing and copies nothing: every pipeline is built from a file of
experiment/v7/shaders/spv, and before the first module is created every file the manifest lists is
hashed (sha256) and compared with the manifest. A missing file, a size or digest mismatch, or a
manifest that cannot be parsed stops the process (SystemExit) — a v7_single run never runs a kernel
the v7 manifest does not vouch for.

The manifest format is the one experiment/v7/compile_shaders_v7.py writes: a free-text header, then
one line per file ``<sha256>  <bytes>  <file>  (<source>)``.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
V7_SHADER_DIRECTORY = _REPOSITORY_ROOT / "experiment" / "v7" / "shaders" / "spv"
MANIFEST_NAME = "MANIFEST.txt"
_ENTRY = re.compile(r"^([0-9a-f]{64})\s+(\d+)\s+(\S+\.spv)\s+\((.*)\)\s*$")

_verified_directories: dict[pathlib.Path, dict[str, str]] = {}


def parse_manifest(text: str) -> dict[str, tuple[str, int, str]]:
    """{file name: (sha256, bytes, source)} of every entry line of a manifest."""
    entries: dict[str, tuple[str, int, str]] = {}
    for line in text.splitlines():
        match = _ENTRY.match(line.strip())
        if match:
            entries[match.group(3)] = (match.group(1), int(match.group(2)), match.group(4))
    return entries


def verify_directory(directory: pathlib.Path = V7_SHADER_DIRECTORY) -> dict[str, str]:
    """Hash every file the directory's manifest lists; return {file name: sha256}. Verified once per
    process and directory (the result is cached); any mismatch raises SystemExit."""
    directory = pathlib.Path(directory).resolve()
    cached = _verified_directories.get(directory)
    if cached is not None:
        return cached
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"[v7_single] SPIR-V manifest missing: {manifest_path}")
    entries = parse_manifest(manifest_path.read_text(encoding="utf-8"))
    if not entries:
        raise SystemExit(f"[v7_single] SPIR-V manifest {manifest_path} lists no file")
    problems = []
    digests: dict[str, str] = {}
    for file_name, (expected_sha256, expected_bytes, _source) in sorted(entries.items()):
        path = directory / file_name
        if not path.exists():
            problems.append(f"{file_name}: missing")
            continue
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or actual_sha256 != expected_sha256:
            problems.append(f"{file_name}: {len(payload)} B sha256 {actual_sha256[:16]}..., manifest says "
                            f"{expected_bytes} B {expected_sha256[:16]}...")
            continue
        digests[file_name] = actual_sha256
    if problems:
        raise SystemExit(f"[v7_single] experiment/v7 SPIR-V differs from its manifest ({manifest_path}):\n  "
                         + "\n  ".join(problems))
    _verified_directories[directory] = digests
    return digests


def load_spirv(shader_name: str, directory: pathlib.Path = V7_SHADER_DIRECTORY) -> bytes:
    """Bytes of ``<shader_name>.comp.spv`` after the directory's manifest check."""
    digests = verify_directory(directory)
    file_name = f"{shader_name}.comp.spv"
    if file_name not in digests:
        raise SystemExit(f"[v7_single] {file_name} is not listed in {pathlib.Path(directory) / MANIFEST_NAME}")
    return (pathlib.Path(directory) / file_name).read_bytes()


def spirv_digests(directory: pathlib.Path = V7_SHADER_DIRECTORY) -> dict[str, str]:
    """{file name: sha256} of the verified directory (for run metadata)."""
    return dict(verify_directory(directory))
