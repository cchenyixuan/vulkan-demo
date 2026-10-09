"""
obj_cache_build.py — E7: build the run_chain_v6.py --obj-cache files of cases ahead of the jobs (CPU only).

The chain bench parses every .obj of a case in pure Python (case_loader_<solver>._parse_obj_vertices: a list
of float tuples, then one float32 array). For the E7 cases that is minutes per run (about 1 GB of .obj per
minute) and, for the 255.6M case, tens of GB of Python objects. run_chain_v6.py --obj-cache DIR stores each
parsed array as DIR/<stem>_<key>.npy (key = sha1 of resolved path, size, mtime_ns) and later runs load it.
This script fills such a DIR before the jobs run, so that no billed run parses text.

  - The cache key and the file written are run_chain_v6.install_obj_cache's own: the solver's parser is
    replaced by a chunked one BEFORE install_obj_cache wraps it, so the wrapper computes the key and saves.
  - The chunked parser does what _parse_obj_vertices does, line by line (strip '#' comments, keep 'v' lines
    with at least 3 coordinates, float() each token, convert to float32), but converts every CHUNK lines to a
    float32 array, so the Python objects alive at once stay bounded (the 255.6M case otherwise holds about
    37 GB of tuples). float() then the float64 -> float32 cast is element by element, so the result is bit
    for bit the reference parser's; --verify runs the reference parser on the same file and compares (use it
    on small and medium files).
  - Prints per file: vertices, parse seconds, cache path and size, hit or built; and at the end the process's
    peak resident set (ru_maxrss).

--check only looks the cache files up (run_chain_v6.obj_cache_file, no parse, no load) and exits 1 when one
is missing: the job's stage-in check. With --stage-to DIR it also copies every cache file it found to DIR
under the same name (the key depends only on the .obj, so the copies are hits from any directory): a job
reads the shared cache from ~/run once and its runs load node-local copies.

Usage (from the checkout root):
    python docs/cluster_v6/scripts/obj_cache_build.py --solver v7 --obj-cache DIR --case CASE.yaml [--case ...]
        [--chunk 10000000] [--verify] [--check [--stage-to DIR]]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import yaml

_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIRECTORY.parents[2]
for entry in (str(_REPOSITORY_ROOT), str(_SCRIPT_DIRECTORY)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_chain_v6  # noqa: E402  (the harness: install_obj_cache, solver_module)


def peak_resident_text() -> str:
    """This process's peak resident set (Linux ru_maxrss is in KiB); n/a where the resource module is missing."""
    try:
        import resource
    except ImportError:
        return "n/a"
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2:.2f} GiB"


def chunked_parse_obj_vertices(path, chunk_lines: int) -> np.ndarray:
    """_parse_obj_vertices with bounded memory: same line rules, float() per token, float32 per chunk."""
    chunks: list[np.ndarray] = []
    vertices: list[tuple[float, float, float]] = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            hash_position = raw_line.find("#")
            if hash_position != -1:
                raw_line = raw_line[:hash_position]
            tokens = raw_line.split()
            if not tokens or tokens[0] != "v":
                continue
            if len(tokens) < 4:
                continue
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            if len(vertices) >= chunk_lines:
                chunks.append(np.asarray(vertices, dtype=np.float32))
                vertices = []
    if vertices:
        chunks.append(np.asarray(vertices, dtype=np.float32))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def case_obj_paths(case_path: pathlib.Path) -> list[pathlib.Path]:
    """Every .obj the loader reads for this case: geometry.particles[*].file and geometry.frame."""
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    geometry = document["geometry"]
    names = [entry["file"] for entry in geometry["particles"]] + [geometry["frame"]]
    paths = []
    for name in names:
        path = (case_path.parent / name).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="E7: build the --obj-cache files of cases ahead of the jobs")
    parser.add_argument("--solver", choices=run_chain_v6.SOLVERS, default="v6")
    parser.add_argument("--obj-cache", required=True, metavar="DIR")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--chunk", type=int, default=10_000_000, help="lines per float32 conversion")
    parser.add_argument("--verify", action="store_true",
                        help="also run the solver's own parser on each file and require identical arrays")
    parser.add_argument("--check", action="store_true", help="only report whether every cache file exists")
    parser.add_argument("--stage-to", default=None, metavar="DIR",
                        help="with --check: copy the cache files found to DIR (same names)")
    arguments = parser.parse_args()

    if arguments.check:
        import shutil
        directory = pathlib.Path(arguments.obj_cache)
        stage = pathlib.Path(arguments.stage_to) if arguments.stage_to else None
        if stage is not None:
            stage.mkdir(parents=True, exist_ok=True)
        copied_bytes = 0
        started = time.time()
        missing = 0
        for case_text in arguments.case:
            for obj_path in case_obj_paths(pathlib.Path(case_text)):
                if not obj_path.exists():
                    print(f"[obj_cache] check {obj_path}: MISSING .obj", flush=True)
                    missing += 1
                    continue
                target = run_chain_v6.obj_cache_file(directory, obj_path)
                state = f"cached {target.name} ({target.stat().st_size:,} B)" if target.exists() else "NOT CACHED"
                missing += 0 if target.exists() else 1
                if stage is not None and target.exists() and not (stage / target.name).exists():
                    shutil.copyfile(target, stage / target.name)
                    copied_bytes += target.stat().st_size
                    state += " -> staged"
                print(f"[obj_cache] check {pathlib.Path(case_text).parent.name}/{obj_path.name}: {state}", flush=True)
        if stage is not None:
            elapsed = time.time() - started
            print(f"[obj_cache] staged {copied_bytes:,} B to {stage} in {elapsed:.1f} s "
                  f"({copied_bytes / max(elapsed, 1e-9) / 1e6:.0f} MB/s)", flush=True)
        print(f"[obj_cache] check: {missing} missing", flush=True)
        return 1 if missing else 0

    case_loader = run_chain_v6.solver_module(arguments.solver, "case_loader")
    reference_parser = case_loader._parse_obj_vertices
    case_loader._parse_obj_vertices = lambda path: chunked_parse_obj_vertices(path, arguments.chunk)
    run_chain_v6.install_obj_cache(arguments.obj_cache, arguments.solver)
    cached_parser = case_loader._parse_obj_vertices
    directory = pathlib.Path(arguments.obj_cache)

    failures = 0
    for case_text in arguments.case:
        case_path = pathlib.Path(case_text)
        for obj_path in case_obj_paths(case_path):
            if not obj_path.exists():
                print(f"[obj_cache] MISSING {obj_path}", flush=True)
                failures += 1
                continue
            before = set(directory.glob(f"{obj_path.stem}_*.npy"))
            started = time.time()
            vertices = cached_parser(obj_path)
            elapsed = time.time() - started
            after = set(directory.glob(f"{obj_path.stem}_*.npy"))
            created = sorted(after - before)
            state = f"built {created[0].name} ({created[0].stat().st_size:,} B)" if created else "hit (already cached)"
            print(f"[obj_cache] {case_path.parent.name}/{obj_path.name}: {vertices.shape[0]:,} vertices "
                  f"{elapsed:.1f}s {state}", flush=True)
            if arguments.verify:
                started = time.time()
                reference = reference_parser(obj_path)
                identical = (reference.dtype == vertices.dtype and reference.shape == vertices.shape
                             and np.array_equal(reference.view(np.uint32), vertices.view(np.uint32)))
                print(f"[obj_cache] verify {obj_path.name}: {'IDENTICAL' if identical else 'DIFFERENT'} "
                      f"(reference parse {time.time() - started:.1f}s)", flush=True)
                failures += 0 if identical else 1
                del reference
            del vertices
    print(f"[obj_cache] done: peak resident set {peak_resident_text()}, failures {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
