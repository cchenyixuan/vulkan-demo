"""
e8_dump_state.py — the seam-audit dump worker (experiment/seam_audit/dump_state.py) with the E7 .obj cache of
run_chain_v6.py installed first (--obj-cache DIR; the case loader then reads the staged .npy vertex arrays instead
of parsing the .obj text), so the phase-1 field comparisons do not parse 8M-particle text files on the node.

Usage:
    python docs/cluster_v6/scripts/e8_dump_state.py [--obj-cache DIR] -- <dump_state.py arguments>
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))


def main() -> int:
    arguments = sys.argv[1:]
    cache = None
    if "--" in arguments:
        own, rest = arguments[:arguments.index("--")], arguments[arguments.index("--") + 1:]
    else:
        own, rest = [], arguments
    if "--obj-cache" in own:
        cache = own[own.index("--obj-cache") + 1]
    version = rest[rest.index("--version") + 1] if "--version" in rest else "v5"
    if cache:
        import run_chain_v6
        run_chain_v6.install_obj_cache(cache, version)
        print(f"[e8] obj cache {cache} installed for {version}", flush=True)
    from experiment.seam_audit import dump_state
    return dump_state.main(rest)


if __name__ == "__main__":
    sys.exit(main())
