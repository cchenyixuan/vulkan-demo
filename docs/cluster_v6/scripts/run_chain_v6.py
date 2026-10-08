"""
run_chain_v6.py — E30 harness around experiment/v6/_run_v6_chain_bench.py.

Runs the chain bench inside this process (runpy) and forwards its own
--switch-interval-ms to it: the bench sets the GIL switch interval at the start
of its main (it then applies to the transport worker threads), so this value
is the only one; passing --switch-interval-ms after "--" as well is refused.
Before the run it prints

  - the switch interval forwarded to the bench,
  - the effective v6 configuration, resolved by v6's own functions from the
    environment of this process (partition_v6.configured_*, the simulator
    module switches, the transfer-queue and worker defaults),
  - the numerics the case yaml carries (xi, epsilon_squared_factor, c, power,
    defrag cadence, capacities) — the runner itself prints none of them.

After the run it prints each simulator's initialization_seam_clamp_count from
the last global-status readback (the bench prints only the overflow_* fields).
Nothing in experiment/v6 is modified; every V6_* switch keeps its code default
unless the caller's environment sets it.

--solver v7 (E39; default v6) runs experiment/v7/_run_v7_chain_bench.py
instead: the configuration is resolved by v7's own modules (partition_v7,
simulator_v7) under the V7_* prefix, the header gains a line
"[e30] solver=v7 bench=...", and one more configuration line names the v7
switches beyond the v6 set (the E39 switches of v7's own registry
simulator_v7.configured_v7_switches() when it exists, every argument-free
partition_v7.configured_* function the lines do not print, and every other
module-level constant of partition_v7 / simulator_v7 assigned from a V7_*
variable, read from the module source), each as v7 resolves it. With the
default everything printed is the v6 output, unchanged.

Usage (from the checkout root):
    python docs/cluster_v6/scripts/run_chain_v6.py [--switch-interval-ms 0.2] [--solver v7] -- <chain bench arguments>
    python docs/cluster_v6/scripts/run_chain_v6.py --config-only [--solver v7] [--case CASE.yaml ...]
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import runpy
import sys
import types

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

SOLVERS = ("v6", "v7")
# what the v6 configuration lines print; additional_switches lists the rest of a newer solver's switches
PRINTED_PARTITION_FUNCTIONS = frozenset({
    "configured_ghost_layers", "configured_keep_departed", "configured_lean_transport",
    "configured_transport_extension", "configured_compact_ghost_lists", "configured_packed_replicas",
    "configured_init_seam_clamp", "configured_delta_density", "configured_band_widths",
    "configured_ghost_self_kernels"})
PRINTED_SIMULATOR_CONSTANTS = frozenset({
    "_CASCADE_FORCE", "_BAND_VOXEL_DISPATCH", "_BAND_SLOT_LANES", "_BAND_COMPACT", "_FAST_SUBMIT",
    "_PHASE_A_NO_WAIT", "_SUBMIT_LOCK_SCOPE"})


def chain_bench_path(solver: str) -> pathlib.Path:
    return _REPOSITORY_ROOT / "experiment" / solver / f"_run_{solver}_chain_bench.py"


def solver_module(solver: str, name: str):
    """experiment.<solver>.utils.<name>_<solver> (the module layout of v6 and its forks)."""
    return importlib.import_module(f"experiment.{solver}.utils.{name}_{solver}")


def parse_arguments(argument_list: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argument_list:
        split_index = argument_list.index("--")
        own_arguments, bench_arguments = argument_list[:split_index], argument_list[split_index + 1:]
    else:
        own_arguments, bench_arguments = argument_list, []
    parser = argparse.ArgumentParser(description="E30 harness for the v6 chain bench")
    parser.add_argument("--switch-interval-ms", type=float, default=0.2,
                        help="forwarded to the chain bench, which applies it with sys.setswitchinterval "
                             "(default 0.2 ms, as the v5 jobs)")
    parser.add_argument("--config-only", action="store_true",
                        help="print the effective configuration and exit (no Vulkan work)")
    parser.add_argument("--case", action="append", default=[],
                        help="case yaml(s) whose numerics to print (with --config-only)")
    parser.add_argument("--obj-cache", default=None, metavar="DIR",
                        help="cache parsed .obj vertex arrays as .npy files in DIR (node-local); later runs "
                             "of the same files skip the pure-Python parse. Harness only: the loader's "
                             "result is the same array")
    parser.add_argument("--solver", choices=SOLVERS, default="v6",
                        help="solver directory experiment/<solver>: its chain bench, its configuration "
                             "resolvers and its environment prefix (E39: v7 = the v7-perf fork, V7_* "
                             "switches; default v6)")
    return parser.parse_args(own_arguments), bench_arguments


def install_obj_cache(cache_directory: str, solver: str = "v6") -> None:
    """Wrap case_loader_<solver>._parse_obj_vertices with a .npy cache keyed by resolved path, size and mtime."""
    import hashlib
    import numpy as np
    case_loader = solver_module(solver, "case_loader")
    original_parser = case_loader._parse_obj_vertices
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

    case_loader._parse_obj_vertices = parse_obj_vertices_cached


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


def switch_value_text(value) -> str:
    """A resolved switch as the configuration lines print it: booleans as 0 / 1, sequences comma-joined."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (tuple, list)):
        return ",".join(map(str, value))
    return str(value)


def environment_constants(module, prefix: str) -> list[tuple[str, list[str]]]:
    """(name, variables) of every module-level constant of ``module`` whose assignment reads an environment
    variable named <prefix>* (a call whose first argument is such a string literal), in source order."""
    import ast
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        variables = sorted({call.args[0].value for call in ast.walk(value)
                            if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Constant)
                            and isinstance(call.args[0].value, str) and call.args[0].value.startswith(prefix)})
        if variables:
            constants += [(target.id, variables) for target in targets if isinstance(target, ast.Name)]
    return constants


def additional_switches(solver: str, partition, simulator) -> list[str]:
    """name=value of the switches of a newer solver that the v6 lines do not print, as the solver resolves
    them: first the solver's own switch registry when it has one (simulator_<solver>.configured_<solver>_
    switches(), {variable: value}, the line the chain bench prints too), then every argument-free
    partition_<solver>.configured_* function (the pool factors take the case and have their own lines), then
    every other module-level constant of partition_<solver> / simulator_<solver> assigned from a <PREFIX>*
    variable, labelled with the variable(s) it reads."""
    import inspect
    prefix = solver.upper() + "_"
    entries = []
    registry_function = getattr(simulator, f"configured_{solver}_switches", None)
    registry = registry_function() if callable(registry_function) else {}
    entries += [f"{variable}={switch_value_text(value)}" for variable, value in registry.items()]
    for name, function in inspect.getmembers(partition, inspect.isfunction):
        if (not name.startswith("configured_") or function.__module__ != partition.__name__
                or name in PRINTED_PARTITION_FUNCTIONS):
            continue
        if any(parameter.default is inspect.Parameter.empty
               and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
               for parameter in inspect.signature(function).parameters.values()):
            continue
        entries.append(f"{name[len('configured_'):]}={switch_value_text(function())}")
    for module in (partition, simulator):
        for name, variables in environment_constants(module, prefix):
            if (module is simulator and name in PRINTED_SIMULATOR_CONSTANTS) or set(variables) <= set(registry):
                continue
            entries.append(f"{name.strip('_').lower()}({','.join(variables)})="
                           f"{switch_value_text(getattr(module, name))}")
    return entries


def print_effective_configuration(case_paths: list[str], solver: str = "v6") -> None:
    partition = solver_module(solver, "partition")
    simulator = solver_module(solver, "simulator")
    prefix = solver.upper() + "_"
    variables = sorted(name for name in os.environ if name.startswith(prefix))
    print(f"[e30-config] {prefix}* in environment: "
          + (", ".join(f"{name}={os.environ[name]}" for name in variables) if variables else "none"),
          flush=True)
    print("[e30-config] partition: "
          f"ghost_layers={partition.configured_ghost_layers()} "
          f"keep_departed={int(partition.configured_keep_departed())} "
          f"lean_transport={int(partition.configured_lean_transport())} "
          f"transport_extension={int(partition.configured_transport_extension())} "
          f"compact_ghost_lists={int(partition.configured_compact_ghost_lists())} "
          f"packed_replicas={int(partition.configured_packed_replicas())} "
          f"init_seam_clamp={int(partition.configured_init_seam_clamp())} "
          f"delta_density={int(partition.configured_delta_density())} "
          f"band_widths={','.join(map(str, partition.configured_band_widths()))} "
          f"ghost_self={','.join(partition.configured_ghost_self_kernels())}", flush=True)
    for dimension in (2, 3):
        stand_in_case = types.SimpleNamespace(physics=types.SimpleNamespace(dimension=dimension))
        print(f"[e30-config] pools {dimension}-D: "
              f"ghost_factor={partition.configured_ghost_pool_factor(stand_in_case)} "
              f"migrant_factor={partition.configured_migrant_pool_factor(stand_in_case)} "
              f"departed_face_fraction={partition.configured_departed_face_fraction(stand_in_case)}",
              flush=True)
    print("[e30-config] simulator: "
          f"cascade_force={int(simulator._CASCADE_FORCE)} "
          f"band_voxel_dispatch={int(simulator._BAND_VOXEL_DISPATCH)} "
          f"band_slot_lanes={simulator._BAND_SLOT_LANES} "
          f"band_compact={int(simulator._BAND_COMPACT)} "
          f"fast_submit={int(simulator._FAST_SUBMIT)} "
          f"phase_a_no_wait={int(simulator._PHASE_A_NO_WAIT)} "
          f"submit_lock_scope={simulator._SUBMIT_LOCK_SCOPE}", flush=True)
    print("[e30-config] transport/context: "
          f"worker_count_aware={os.environ.get(prefix + 'WORKER_COUNT_AWARE', '1')} "
          f"split_transfer_queues={os.environ.get(prefix + 'SPLIT_TRANSFER_QUEUES', '1')} "
          f"concurrent_buffers={os.environ.get(prefix + 'CONCURRENT_BUFFERS', '1')} "
          f"pool_peaks={os.environ.get(prefix + 'POOL_PEAKS', '0')} "
          f"per_sim_pipeline={os.environ.get(prefix + 'PER_SIM_PIPELINE', '0')} "
          f"worker_affinity={'set' if os.environ.get(prefix + 'WORKER_AFFINITY') else 'unset'}", flush=True)
    if solver != "v6":
        entries = additional_switches(solver, partition, simulator)
        print(f"[e30-config] {solver} switches beyond the v6 set: " + (" ".join(entries) if entries else "none"),
              flush=True)
    for case_path in case_paths:
        print_case_numerics(case_path)


def install_clamp_count_recorder(solver: str = "v6") -> dict:
    """Wrap SphSimulator<V>.readback_global_status (the solver's simulator class) to
    remember each simulator's latest initialization_seam_clamp_count (first-call
    order = slab order: the bootstrap reads every sim in index order)."""
    simulator_module = solver_module(solver, "simulator")
    simulator_class = getattr(simulator_module, f"SphSimulator{solver.upper()}")
    original_readback = simulator_class.readback_global_status
    recorded: dict = {}

    def readback_global_status_recording(simulator, *arguments, **keyword_arguments):
        status = original_readback(simulator, *arguments, **keyword_arguments)
        device_index = getattr(getattr(simulator, "ctx", None), "physical_device_index", -1)
        recorded[id(simulator)] = (device_index,
                                   status.get("initialization_seam_clamp_count"),
                                   status.get("overflow_initialization_outside"))
        return status

    simulator_class.readback_global_status = readback_global_status_recording
    return recorded


def main() -> int:
    arguments, bench_arguments = parse_arguments(sys.argv[1:])
    # any spelling argparse would take for the bench's --switch-interval-ms (it accepts unique prefixes, "--sw")
    if any(len(argument.split("=", 1)[0]) >= 4 and "--switch-interval-ms".startswith(argument.split("=", 1)[0])
           for argument in bench_arguments):
        sys.exit("pass --switch-interval-ms before '--' only: this script forwards its own value to the bench")
    if any(argument.split("=", 1)[0] == "--solver" for argument in bench_arguments):
        sys.exit("pass --solver before '--': it selects the bench, the bench itself has no such option")
    chain_bench = chain_bench_path(arguments.solver)
    if not chain_bench.exists():
        sys.exit(f"--solver {arguments.solver}: no chain bench at {chain_bench} (experiment/{arguments.solver} "
                 f"not deployed?)")
    os.chdir(_REPOSITORY_ROOT)
    # The bench applies it (sys.setswitchinterval at the start of its main) and prints it in its header.
    print(f"[e30] switchinterval_s={arguments.switch_interval_ms / 1000.0:.6g} (forwarded to the bench) "
          f"pid={os.getpid()} cwd={os.getcwd()}", flush=True)
    if arguments.solver != "v6":           # the v6 header stays as it was; parse_run_v6.py reads the solver here
        print(f"[e30] solver={arguments.solver} bench={chain_bench.relative_to(_REPOSITORY_ROOT).as_posix()}",
              flush=True)
    if arguments.config_only:
        print_effective_configuration(arguments.case, arguments.solver)
        return 0

    print_effective_configuration([case_argument(bench_arguments)], arguments.solver)
    if arguments.obj_cache:
        install_obj_cache(arguments.obj_cache, arguments.solver)
    recorded = install_clamp_count_recorder(arguments.solver)
    # forwarded last: argparse keeps the last occurrence, so this value wins even past the refusal above
    sys.argv = ([str(chain_bench)] + bench_arguments
                + ["--switch-interval-ms", repr(arguments.switch_interval_ms)])
    exit_code = 0
    try:
        runpy.run_path(str(chain_bench), run_name="__main__")
    except SystemExit as exit_request:
        code = exit_request.code
        if isinstance(code, str):
            print(code, file=sys.stderr, flush=True)          # sys.exit("message"): show why the bench stopped
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
