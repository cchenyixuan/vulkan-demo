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
    status = dump_state.main(rest)
    tag_devices(rest)
    return status


def tag_devices(rest: list[str]) -> None:
    """E8: every dump of this run gets a per-particle ``device`` array (the physical device of the owning slab,
    from the sidecar's device_map) next to its ``slab`` array; the npz is rewritten in place (uncompressed)."""
    import json
    import numpy as np
    if "--out-dir" not in rest or "--run-name" not in rest:
        return
    out_dir = pathlib.Path(rest[rest.index("--out-dir") + 1])
    run_name = rest[rest.index("--run-name") + 1]
    for sidecar in sorted(out_dir.glob(f"{run_name}_N*.json")):
        npz = sidecar.with_suffix(".npz")
        if not npz.exists():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            device_map = [int(item) for item in meta.get("device_map", [])]
            arrays = dict(np.load(npz))
            if "device" in arrays or not device_map or "slab" not in arrays:
                continue
            arrays["device"] = np.asarray(device_map, dtype=np.int16)[arrays["slab"].astype(np.int64)]
            temporary = npz.with_name(npz.stem + ".device.partial.npz")
            np.savez(temporary, **arrays)
            temporary.replace(npz)
            print(f"[e8] {npz.name}: device array added ({arrays['device'].size:,} particles, devices {sorted(set(device_map))})", flush=True)
        except Exception as error:                                   # noqa: BLE001
            print(f"[e8] {npz.name}: device tagging failed: {error!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
