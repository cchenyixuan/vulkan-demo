"""
e8_single_bench.py — run the v7_single throughput bench of ANOTHER deployment (E8_SINGLE_REPO, branch v7-single:
experiment/v7_single on v7-rc1's experiment/v7) from this deployment's job, with this harness's .obj cache installed
into that deployment's case loader (the staged .npy files are keyed by the particle files' resolved path, which both
deployments share through the cases/ link).

The v7_single solver imports experiment.v7.utils.* of its OWN tree (the case model, the loader, the SPIR-V directory it
checks against its MANIFEST), so that tree must come first on sys.path and this deployment's experiment package must not
be imported before it. run_chain_v6.py is loaded by file path (it puts this deployment's root on sys.path at import,
which is removed again before anything from `experiment` is imported).

Usage:
    python docs/cluster_v6/scripts/e8_single_bench.py --repo DIR [--obj-cache DIR] -- <_run_single_bench.py arguments>
"""
from __future__ import annotations

import importlib.util
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve()
THIS_ROOT = HERE.parents[3]


def main() -> int:
    arguments = sys.argv[1:]
    if "--" in arguments:
        own, rest = arguments[:arguments.index("--")], arguments[arguments.index("--") + 1:]
    else:
        own, rest = arguments, []
    repo = pathlib.Path(own[own.index("--repo") + 1]).resolve()
    cache = own[own.index("--obj-cache") + 1] if "--obj-cache" in own else None
    bench = repo / "experiment" / "v7_single" / "_run_single_bench.py"
    if not bench.exists():
        raise SystemExit(f"no v7_single bench at {bench}")
    for name in list(sys.modules):
        if name == "experiment" or name.startswith("experiment."):
            del sys.modules[name]
    sys.path = [entry for entry in sys.path if pathlib.Path(entry or ".").resolve() != THIS_ROOT]
    sys.path.insert(0, str(repo))
    if cache:
        specification = importlib.util.spec_from_file_location("run_chain_v6_for_single", HERE.parent / "run_chain_v6.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)             # inserts THIS_ROOT at sys.path[0]: removed below
        sys.path = [entry for entry in sys.path if pathlib.Path(entry or ".").resolve() != THIS_ROOT]
        for name in list(sys.modules):
            if name == "experiment" or name.startswith("experiment."):
                del sys.modules[name]
        module.install_obj_cache(cache, "v7")               # wraps repo's experiment.v7.utils.case_loader_v7
        print(f"[e8] obj cache {cache} installed into {repo}", flush=True)
    print(f"[e8] v7_single bench {bench} (sys.path[0] = {sys.path[0]})", flush=True)
    sys.argv = [str(bench)] + rest
    try:
        runpy.run_path(str(bench), run_name="__main__")
    except SystemExit as exit_request:
        code = exit_request.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
