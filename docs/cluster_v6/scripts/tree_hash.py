"""
tree_hash.py — the git tree hash of a deployed directory (archive deployment: there is no .git), to compare with
`git rev-parse <commit>:<path>` (E7 provenance: experiment/v7 of the deployment against tag v7-rc1).

Git's object format, computed without git: blob = sha1("blob <size>\\0" + content); tree = sha1("tree <size>\\0" +
entries), each entry "<mode> <name>\\0<20-byte sha>", sorted by name with directories compared as "<name>/";
modes 100755 (owner may execute), 100644, 120000 (symbolic link), 40000 (directory); empty directories are not
stored. Left out: __pycache__ directories and *.pyc files (written by Python at run time, never tracked).

Usage:
    python docs/cluster_v6/scripts/tree_hash.py DIR [--expect SHA] [--list]
prints the tree hash (exit 1 when --expect differs); --list prints "<mode> <blob sha> <path>" per file, the
format of `git ls-tree -r <commit> -- <path>` without the type column, for finding the file that differs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys


def skipped(name: str) -> bool:
    return name == "__pycache__" or name.endswith(".pyc")


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree_sha(directory: str, prefix: str, listing: list) -> tuple[str | None, int]:
    """(tree sha or None for an empty tree, number of files) of DIRECTORY; LISTING collects every file."""
    entries = []
    files = 0
    for entry in os.scandir(directory):
        if skipped(entry.name):
            continue
        relative = f"{prefix}{entry.name}"
        if entry.is_symlink():
            mode, sha, is_directory = "120000", blob_sha(os.readlink(entry.path).encode()), False
            files += 1
            listing.append((mode, sha, relative))
        elif entry.is_dir():
            sha, count = tree_sha(entry.path, relative + "/", listing)
            if sha is None:
                continue
            mode, is_directory = "40000", True
            files += count
        else:
            with open(entry.path, "rb") as handle:
                sha = blob_sha(handle.read())
            mode = "100755" if os.stat(entry.path).st_mode & stat.S_IXUSR else "100644"
            is_directory = False
            files += 1
            listing.append((mode, sha, relative))
        entries.append((entry.name.encode() + (b"/" if is_directory else b""), mode, entry.name, sha))
    if not entries:
        return None, 0
    entries.sort(key=lambda item: item[0])
    body = b"".join(mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(sha) for _, mode, name, sha in entries)
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest(), files


def main() -> int:
    parser = argparse.ArgumentParser(description="git tree hash of a directory without git")
    parser.add_argument("directory")
    parser.add_argument("--expect", default=None, help="exit 1 when the hash differs from this sha")
    parser.add_argument("--list", action="store_true", help="also print every file's mode, blob sha and path")
    arguments = parser.parse_args()
    listing: list = []
    sha, files = tree_sha(arguments.directory, "", listing)
    if arguments.list:
        for mode, blob, relative in sorted(listing, key=lambda item: item[2]):
            print(f"{mode} {blob} {relative}")
    print(sha if sha is not None else "empty")
    if arguments.expect is not None and sha != arguments.expect:
        print(f"tree_hash: {arguments.directory} = {sha} ({files} files) != expected {arguments.expect}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
