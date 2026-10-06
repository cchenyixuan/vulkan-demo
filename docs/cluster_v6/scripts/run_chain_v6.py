"""
run_chain_v6.py — E30 harness around experiment/v6/_run_v6_chain_bench.py.

Runs the chain bench inside this process (runpy), so the GIL switch interval
set here applies to its transport worker threads, after printing

  - the switch interval in effect,
  - the effective v6 configuration, resolved by v6's own functions from the
    environment of this process (partition_v6.configured_*, the simulator
    module switches, the transfer-queue and worker defaults),
  - the numerics the case yaml carries (xi, epsilon_squared_factor, c, power,
    defrag cadence, capacities) — the runner itself prints none of them.

After the run it prints each simulator's initialization_seam_clamp_count from
the last global-status readback (the bench prints only the overflow_* fields).
Nothing in experiment/v6 is modified; every V6_* switch keeps its code default
unless the caller's environment sets it.

Usage (from the checkout root):
    python docs/cluster_v6/scripts/run_chain_v6.py [--switch-interval-ms 0.2] -- <chain bench arguments>
    python docs/cluster_v6/scripts/run_chain_v6.py --config-only [--case CASE.yaml ...]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys
import types

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

CHAIN_BENCH_PATH = _REPOSITORY_ROOT / "experiment" / "v6" / "_run_v6_chain_bench.py"


def parse_arguments(argument_list: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argument_list:
        split_index = argument_list.index("--")
        own_arguments, bench_arguments = argument_list[:split_index], argument_list[split_index + 1:]
    else:
        own_arguments, bench_arguments = argument_list, []
    parser = argparse.ArgumentParser(description="E30 harness for the v6 chain bench")
    parser.add_argument("--switch-interval-ms", type=float, default=0.2,
                        help="sys.setswitchinterval for this process (v5 jobs used 0.2 ms)")
    parser.add_argument("--config-only", action="store_true",
                        help="print the effective configuration and exit (no Vulkan work)")
    parser.add_argument("--case", action="append", default=[],
                        help="case yaml(s) whose numerics to print (with --config-only)")
    parser.add_argument("--obj-cache", default=None, metavar="DIR",
                        help="cache parsed .obj vertex arrays as .npy files in DIR (node-local); later runs "
                             "of the same files skip the pure-Python parse. Harness only: the loader's "
                             "result is the same array")
    return parser.parse_args(own_arguments), bench_arguments


def install_obj_cache(cache_directory: str) -> None:
    """Wrap case_loader_v6._parse_obj_vertices with a .npy cache keyed by resolved path, size and mtime."""
    import hashlib
    import numpy as np
    from experiment.v6.utils import case_loader_v6
    original_parser = case_loader_v6._parse_obj_vertices
    directory = pathlib.Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)

    def parse_obj_vertices_cached(path):
        resolved = pathlib.Path(path).resolve()
        status = resolved.stat()
        key = hashlib.sha1(f"{resolved}|{status.st_size}|{status.st_mtime_ns}".encode()).hexdigest()[:20]
        target = directory / f"{resolved.stem}_{key}.npy"
        if target.exists():
            vertices = np.load(target)
            print(f"[e30] obj cache hit {resolved.name} ({vertices.shape[0]:,} vertices)", flush=True)
            return vertices
        vertices = original_parser(path)
        temporary = directory / f"{resolved.stem}_{key}.{os.getpid()}.tmp.npy"
        np.save(temporary, vertices)
        os.replace(temporary, target)
        return vertices

    case_loader_v6._parse_obj_vertices = parse_obj_vertices_cached


def case_argument(bench_arguments: list[str]) -> str:
    for index, argument in enumerate(bench_arguments):
        if argument == "--case" and index + 1 < len(bench_arguments):
            return bench_arguments[index + 1]
        if argument.startswith("--case="):
            return argument.split("=", 1)[1]
    return "cases/lid_driven_cavity_2d/case.yaml"     # the bench's own default


def print_case_numerics(case_path: str) -> None:
    import yaml
    path = pathlib.Path(case_path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:                                        # noqa: BLE001
        print(f"[e30-config] case {case_path}: unreadable ({error!r})", flush=True)
        return
    physics = document.get("physics", {}) or {}
    numerics = document.get("numerics", {}) or {}
    regularization = numerics.get("regularization", {}) or {}
    capacities = document.get("capacities", {}) or {}
    print(f"[e30-config] case {case_path}: dimension={physics.get('dimension')} h={physics.get('h')} "
          f"speed_of_sound={physics.get('speed_of_sound')} power={physics.get('power')} "
          f"xi={regularization.get('xi', 'default(0.001)')} "
          f"epsilon_squared_factor={numerics.get('epsilon_squared_factor', 'default(0.0025)')} "
          f"defrag_cadence={numerics.get('defrag_cadence')} pool_size={capacities.get('pool_size')} "
          f"max_per_voxel={capacities.get('max_per_voxel')} max_incoming={capacities.get('max_incoming')} "
          f"yaml_is_symlink={path.is_symlink()}", flush=True)


def print_effective_configuration(case_paths: list[str]) -> None:
    from experiment.v6.utils import partition_v6, simulator_v6
    variables = sorted(name for name in os.environ if name.startswith("V6_"))
    print(f"[e30-config] V6_* in environment: "
          + (", ".join(f"{name}={os.environ[name]}" for name in variables) if variables else "none"),
          flush=True)
    print("[e30-config] partition: "
          f"ghost_layers={partition_v6.configured_ghost_layers()} "
          f"keep_departed={int(partition_v6.configured_keep_departed())} "
          f"lean_transport={int(partition_v6.configured_lean_transport())} "
          f"transport_extension={int(partition_v6.configured_transport_extension())} "
          f"compact_ghost_lists={int(partition_v6.configured_compact_ghost_lists())} "
          f"packed_replicas={int(partition_v6.configured_packed_replicas())} "
          f"init_seam_clamp={int(partition_v6.configured_init_seam_clamp())} "
          f"delta_density={int(partition_v6.configured_delta_density())} "
          f"band_widths={','.join(map(str, partition_v6.configured_band_widths()))} "
          f"ghost_self={','.join(partition_v6.configured_ghost_self_kernels())}", flush=True)
    for dimension in (2, 3):
        stand_in_case = types.SimpleNamespace(physics=types.SimpleNamespace(dimension=dimension))
        print(f"[e30-config] pools {dimension}-D: "
              f"ghost_factor={partition_v6.configured_ghost_pool_factor(stand_in_case)} "
              f"migrant_factor={partition_v6.configured_migrant_pool_factor(stand_in_case)} "
              f"departed_face_fraction={partition_v6.configured_departed_face_fraction(stand_in_case)}",
              flush=True)
    print("[e30-config] simulator: "
          f"cascade_force={int(simulator_v6._CASCADE_FORCE)} "
          f"band_voxel_dispatch={int(simulator_v6._BAND_VOXEL_DISPATCH)} "
          f"band_slot_lanes={simulator_v6._BAND_SLOT_LANES} "
          f"band_compact={int(simulator_v6._BAND_COMPACT)} "
          f"fast_submit={int(simulator_v6._FAST_SUBMIT)} "
          f"phase_a_no_wait={int(simulator_v6._PHASE_A_NO_WAIT)} "
          f"submit_lock_scope={simulator_v6._SUBMIT_LOCK_SCOPE}", flush=True)
    print("[e30-config] transport/context: "
          f"worker_count_aware={os.environ.get('V6_WORKER_COUNT_AWARE', '1')} "
          f"split_transfer_queues={os.environ.get('V6_SPLIT_TRANSFER_QUEUES', '1')} "
          f"concurrent_buffers={os.environ.get('V6_CONCURRENT_BUFFERS', '1')} "
          f"pool_peaks={os.environ.get('V6_POOL_PEAKS', '0')} "
          f"per_sim_pipeline={os.environ.get('V6_PER_SIM_PIPELINE', '0')} "
          f"worker_affinity={'set' if os.environ.get('V6_WORKER_AFFINITY') else 'unset'}", flush=True)
    for case_path in case_paths:
        print_case_numerics(case_path)


def install_clamp_count_recorder() -> dict:
    """Wrap SphSimulatorV6.readback_global_status to remember each simulator's
    latest initialization_seam_clamp_count (first-call order = slab order: the
    bootstrap reads every sim in index order)."""
    from experiment.v6.utils import simulator_v6
    original_readback = simulator_v6.SphSimulatorV6.readback_global_status
    recorded: dict = {}

    def readback_global_status_recording(simulator, *arguments, **keyword_arguments):
        status = original_readback(simulator, *arguments, **keyword_arguments)
        device_index = getattr(getattr(simulator, "ctx", None), "physical_device_index", -1)
        recorded[id(simulator)] = (device_index,
                                   status.get("initialization_seam_clamp_count"),
                                   status.get("overflow_initialization_outside"))
        return status

    simulator_v6.SphSimulatorV6.readback_global_status = readback_global_status_recording
    return recorded


def main() -> int:
    arguments, bench_arguments = parse_arguments(sys.argv[1:])
    os.chdir(_REPOSITORY_ROOT)
    sys.setswitchinterval(arguments.switch_interval_ms / 1000.0)
    print(f"[e30] switchinterval_s={sys.getswitchinterval()} pid={os.getpid()} "
          f"cwd={os.getcwd()}", flush=True)
    if arguments.config_only:
        print_effective_configuration(arguments.case)
        return 0

    print_effective_configuration([case_argument(bench_arguments)])
    if arguments.obj_cache:
        install_obj_cache(arguments.obj_cache)
    recorded = install_clamp_count_recorder()
    sys.argv = [str(CHAIN_BENCH_PATH)] + bench_arguments
    exit_code = 0
    try:
        runpy.run_path(str(CHAIN_BENCH_PATH), run_name="__main__")
    except SystemExit as exit_request:
        code = exit_request.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        for slab_index, (device_index, clamp_count, outside_count) in enumerate(recorded.values()):
            print(f"[e30] sim{slab_index} (dev{device_index}): "
                  f"initialization_seam_clamp_count={clamp_count} "
                  f"overflow_initialization_outside={outside_count}", flush=True)
        sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
