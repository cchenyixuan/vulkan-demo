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

E7 (E15): with <PREFIX>POOL_PEAKS=1 in the environment the transport workers
record every frame's demand of each ghost-pool region of their link (the
sender's allocation counter: inner / outer replicas and migrants, or one mixed
region with a single ghost layer), but nothing prints it. Then this script
collects the workers as they are constructed and, after the run, prints one
line per link and region:
  [e30] pool <link> <region>: capacity=C peak=P frame_of_peak=F p999=Q mean=M last=L frames=N occupancy=P/C
(frames counted from the first recorded frame; the departed pool and the
install tail are printed by the bench itself on every run). Without the
switch nothing is installed and the output is unchanged.

E7, solvers other than v6 (the v6 output stays as it was): the chain build is
timed in stages, each line with this process's host memory from
/proc/self/status (n/a where there is none):
  [e30] stage <name>: t=<seconds since the bench started>s VmRSS=<GiB>GiB VmHWM=<GiB>GiB
for case_loaded (load_case_<solver> returned: the read-in, bench imports
included), partition_done (compute_chain_partition returned), bootstrap_start
(contexts and simulators built), bootstrap_end, loop_start (= the chain build
time: everything before the first timed frame), loop_end and exit (after the
bench's post-run checks). With --weights auto the pilots report too (the last
line of a stage is the timed chain's).
After the run each simulator's last pool-health readback is printed raw (the
bench prints pool_used only to 0.1 %):
  [e30] sim<i> (dev<d>) pool_health: peak_tail_high_water=... own_pool_size=... free_margin=...
      peak_migration_count=... peak_departed_count=... departed_pool_size=...

E7 full campaign (solvers other than v6):
  --barrier DIR --barrier-parties N   a start barrier for processes launched together (a K = 1
      reference set, the K = 2 pairs of a pair reference, a pre-check group): after its bootstrap
      each process writes a token into DIR (node-local) and waits until N tokens are there (or
      --barrier-timeout seconds passed), then starts its loop, so that the timed windows coincide:
        [e30] barrier dir=... parties=N arrived_epoch=... released_epoch=... waited=...s seen=n status=ok|timeout
      (not with --weights auto: its pilot chains would take the barrier);
  the stage lines end with epoch=<seconds since 1970> (aligns runs with the telemetry);
  --defrag-log   every in-loop defrag's report, one line per slab, and every defrag's duration:
        [e30] defrag f<frame> sim<i>: interval_migration=... alive=... overflow_install_tail=... ...
        [e30] defrag_time sim<i>: wall_ms=... [gpu_us=...]   (gpu_us with --anatomy timers attached)
  with --anatomy in the bench arguments every duration the anatomy computes (the bench prints a fixed
  subset) and the install sum c_start .. c_append_departed_end, one line before each [anatomy] line:
        [e30] anatomy_all call=<n>: predict=... expand_lists=... install_leading=... ... install_sum=...
  with <PREFIX>POOL_PEAKS=1 the region demand per 1000 recorded frames as well (E15 developed flow):
        [e30] pool_series <link> <region>: capacity=C window=1000 peaks=p1,p2,...

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
import time
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
    parser.add_argument("--barrier", default=None, metavar="DIR",
                        help="E7: hold the loop after the bootstrap until --barrier-parties processes reached it "
                             "(token files in DIR)")
    parser.add_argument("--barrier-parties", type=int, default=0)
    parser.add_argument("--barrier-timeout", type=float, default=1800.0,
                        help="seconds a process waits at the barrier before it starts anyway (status=timeout)")
    parser.add_argument("--defrag-log", action="store_true",
                        help="E7: print every in-loop defrag's per-slab report and each defrag's duration")
    parser.add_argument("--solver", choices=SOLVERS, default="v6",
                        help="solver directory experiment/<solver>: its chain bench, its configuration "
                             "resolvers and its environment prefix (E39: v7 = the v7-perf fork, V7_* "
                             "switches; default v6)")
    return parser.parse_args(own_arguments), bench_arguments


def obj_cache_file(directory: pathlib.Path, path) -> pathlib.Path:
    """The cache file of one .obj: DIR/<stem>_<key>.npy, key = sha1(resolved path | size | mtime_ns)[:20]."""
    import hashlib
    resolved = pathlib.Path(path).resolve()
    status = resolved.stat()
    key = hashlib.sha1(f"{resolved}|{status.st_size}|{status.st_mtime_ns}".encode()).hexdigest()[:20]
    return directory / f"{resolved.stem}_{key}.npy"


def install_obj_cache(cache_directory: str, solver: str = "v6") -> None:
    """Wrap case_loader_<solver>._parse_obj_vertices with a .npy cache keyed by resolved path, size and mtime."""
    import numpy as np
    case_loader = solver_module(solver, "case_loader")
    original_parser = case_loader._parse_obj_vertices
    directory = pathlib.Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)

    def parse_obj_vertices_cached(path):
        resolved = pathlib.Path(path).resolve()
        target = obj_cache_file(directory, resolved)
        if target.exists():
            vertices = np.load(target)
            print(f"[e30] obj cache hit {resolved.name} ({vertices.shape[0]:,} vertices)", flush=True)
            return vertices
        vertices = original_parser(path)
        temporary = directory / f"{target.stem}.{os.getpid()}.tmp.npy"
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


def host_memory_text() -> str:
    """VmRSS and VmHWM (peak) of this process in GiB from /proc/self/status (Linux); n/a elsewhere."""
    values = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in ("VmRSS", "VmHWM"):
                    values[name] = int(rest.split()[0]) / 1024 ** 2          # kB -> GiB
    except OSError:
        pass
    return " ".join(f"{name}={values[name]:.2f}GiB" if name in values else f"{name}=n/a"
                    for name in ("VmRSS", "VmHWM"))


def report_stage(stage: str, started: float) -> None:
    print(f"[e30] stage {stage}: t={time.perf_counter() - started:.2f}s {host_memory_text()} "
          f"epoch={time.time():.3f}", flush=True)


def barrier_wait(directory: str, parties: int, timeout_seconds: float) -> None:
    """E7 start barrier: write this process's token into DIRECTORY, wait until PARTIES tokens are there (or the
    timeout passed), print one line. The token names carry host and pid; the directory is fresh per group."""
    import socket
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    arrived = time.time()
    token = path / f"{socket.gethostname()}_{os.getpid()}.ready"
    temporary = path / f".{token.name}.tmp"
    temporary.write_text(f"{arrived:.6f}\n", encoding="utf-8")
    os.replace(temporary, token)
    status, seen = "timeout", 0
    while True:
        seen = sum(1 for _ in path.glob("*.ready"))
        if seen >= parties:
            status = "ok"
            break
        if time.time() - arrived > timeout_seconds:
            break
        time.sleep(0.02)
    released = time.time()
    print(f"[e30] barrier dir={directory} parties={parties} arrived_epoch={arrived:.3f} "
          f"released_epoch={released:.3f} waited={released - arrived:.2f}s seen={seen} status={status}", flush=True)


def install_stage_reporter(solver: str, started: float, barrier=None) -> None:
    """E7: wrap ChainOrchestrator<V>.bootstrap_all and run_pipelined so that the start and end of the bootstrap
    and of the timed loop are reported (report_stage). With --weights auto the pilot chains report too; the
    timed chain's lines come last. BARRIER = (directory, parties, timeout): the first loop waits there first."""
    orchestrator_module = solver_module(solver, "orchestrator")
    chain_class = getattr(orchestrator_module, f"ChainOrchestrator{solver.upper()}")
    original_bootstrap = chain_class.bootstrap_all
    original_run = chain_class.run_pipelined
    barrier_passed: list = []

    def bootstrap_all_reporting(orchestrator, *arguments, **keyword_arguments):
        report_stage("bootstrap_start", started)
        result = original_bootstrap(orchestrator, *arguments, **keyword_arguments)
        report_stage("bootstrap_end", started)
        return result

    def run_pipelined_reporting(orchestrator, *arguments, **keyword_arguments):
        if barrier is not None and not barrier_passed:
            barrier_passed.append(True)
            barrier_wait(*barrier)
        report_stage("loop_start", started)
        try:
            return original_run(orchestrator, *arguments, **keyword_arguments)
        finally:
            report_stage("loop_end", started)

    chain_class.bootstrap_all = bootstrap_all_reporting
    chain_class.run_pipelined = run_pipelined_reporting

    # the read-in and the partition (the bench and the calibration import both inside their functions, after
    # these module attributes are replaced)
    case_loader = solver_module(solver, "case_loader")
    partition = solver_module(solver, "partition")
    loader_name = f"load_case_{solver}"
    original_load = getattr(case_loader, loader_name)
    original_partition = partition.compute_chain_partition

    def load_case_reporting(*arguments, **keyword_arguments):
        result = original_load(*arguments, **keyword_arguments)
        report_stage("case_loaded", started)
        return result

    def compute_chain_partition_reporting(*arguments, **keyword_arguments):
        result = original_partition(*arguments, **keyword_arguments)
        report_stage("partition_done", started)
        return result

    setattr(case_loader, loader_name, load_case_reporting)
    partition.compute_chain_partition = compute_chain_partition_reporting


def install_pool_health_recorder(solver: str) -> dict:
    """E7 (E15): wrap SphSimulator<V>.readback_pool_health so that each simulator's last pool-health readback
    (the bench reads it once per slab after the run, in slab order) is kept with its device index; the bench
    prints the tail's high-water mark only as pool_used to 0.1 % of the own pool, too coarse for the free
    margin of large slabs."""
    simulator_module = solver_module(solver, "simulator")
    simulator_class = getattr(simulator_module, f"SphSimulator{solver.upper()}")
    original_readback = simulator_class.readback_pool_health
    recorded: dict = {}

    def readback_pool_health_recording(simulator, *arguments, **keyword_arguments):
        health = original_readback(simulator, *arguments, **keyword_arguments)
        device_index = getattr(getattr(simulator, "ctx", None), "physical_device_index", -1)
        recorded[id(simulator)] = (device_index, dict(health))
        return health

    simulator_class.readback_pool_health = readback_pool_health_recording
    return recorded


def print_pool_health(recorded: dict) -> None:
    """[e30] sim<i> (dev<d>) pool_health: the raw watermarks (never reset, survive defrag; common.glsl
    PoolHealthBuffer): peak_tail_high_water (largest alive + install-tail occupancy ever demanded, the quantity
    the install overflow guard checks), own_pool_size, free_margin (own pool - that peak), peak_migration_count
    (deepest migrant install tail in any one defrag interval), peak_departed_count (most migrants sent away in
    one frame), departed_pool_size."""
    names = ("peak_tail_high_water", "own_pool_size", "free_margin", "peak_migration_count",
             "peak_departed_count", "departed_pool_size")
    for slab_index, (device_index, health) in enumerate(recorded.values()):
        print(f"[e30] sim{slab_index} (dev{device_index}) pool_health: "
              + " ".join(f"{name}={health.get(name)}" for name in names), flush=True)


def install_pool_peak_recorder(solver: str = "v6") -> list:
    """E7 (E15): wrap transport_<solver>.GhostMigrationWorker.__init__ so that every worker the bench constructs is
    kept here (construction order = the orchestrator's link order); their region_counts / region_capacity are
    read after the run by print_pool_peaks."""
    transport = solver_module(solver, "transport")
    worker_class = transport.GhostMigrationWorker
    original_initializer = worker_class.__init__
    constructed_workers: list = []

    def initializer_recording(worker, *arguments, **keyword_arguments):
        original_initializer(worker, *arguments, **keyword_arguments)
        constructed_workers.append(worker)

    worker_class.__init__ = initializer_recording
    return constructed_workers


def print_pool_peaks(workers: list) -> None:
    """One line per link and recorded region: capacity (slots), peak demand, the recorded frame of the peak, the
    99.9th percentile (nearest rank), mean, last value, number of frames and peak / capacity."""
    for worker in workers:
        capacities = getattr(worker, "region_capacity", {}) or {}
        for region, counts in (getattr(worker, "region_counts", {}) or {}).items():
            capacity = capacities.get(region)
            if not counts:
                print(f"[e30] pool {worker.label} {region}: capacity={capacity} frames=0", flush=True)
                continue
            peak = max(counts)
            ordered = sorted(counts)
            rank = max(1, -(-999 * len(ordered) // 1000))          # nearest rank: ceil(0.999 n)
            occupancy = f"{peak / capacity:.4f}" if capacity else "n/a"
            print(f"[e30] pool {worker.label} {region}: capacity={capacity} peak={peak} "
                  f"frame_of_peak={counts.index(peak)} p999={ordered[rank - 1]} "
                  f"mean={sum(counts) / len(counts):.1f} last={counts[-1]} frames={len(counts)} "
                  f"occupancy={occupancy}", flush=True)
            window = 1000
            print(f"[e30] pool_series {worker.label} {region}: capacity={capacity} window={window} peaks="
                  + ",".join(str(max(counts[start:start + window])) for start in range(0, len(counts), window)),
                  flush=True)


def report_value_text(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def install_defrag_log(solver: str) -> None:
    """E7 --defrag-log: wrap ChainOrchestrator<V>.run_pipelined so that its on_defrag callback first prints the
    report the orchestrator collected at that defrag (one line per slab: the interval's migration count, alive,
    the never-reset pool watermarks, the overflow counters), and SphSimulator<V>.submit_defrag_and_wait so that
    each defrag's wall time is printed (and its GPU time, defrag_end - defrag_start, when a BenchTimer is
    attached, i.e. with --anatomy). Slab index = order of the first defrag call (the bootstrap's, in slab order)."""
    orchestrator_module = solver_module(solver, "orchestrator")
    chain_class = getattr(orchestrator_module, f"ChainOrchestrator{solver.upper()}")
    simulator_class = getattr(solver_module(solver, "simulator"), f"SphSimulator{solver.upper()}")
    original_run = chain_class.run_pipelined
    original_defrag = simulator_class.submit_defrag_and_wait
    slab_order: dict = {}

    def run_pipelined_logging(orchestrator, *arguments, **keyword_arguments):
        callback = keyword_arguments.get("on_defrag")

        def on_defrag_logging(frame_n, report):
            for index, entry in enumerate(report or []):
                print(f"[e30] defrag f{frame_n} sim{index}: "
                      + " ".join(f"{name}={report_value_text(value)}" for name, value in entry.items()), flush=True)
            if callback is not None:
                return callback(frame_n, report)
            return None

        keyword_arguments["on_defrag"] = on_defrag_logging
        return original_run(orchestrator, *arguments, **keyword_arguments)

    def submit_defrag_and_wait_timed(simulator, *arguments, **keyword_arguments):
        index = slab_order.setdefault(id(simulator), len(slab_order))
        started = time.perf_counter()
        result = original_defrag(simulator, *arguments, **keyword_arguments)
        wall_ms = (time.perf_counter() - started) * 1e3
        gpu_text = ""
        bench = getattr(simulator, "bench", None)
        if bench is not None and hasattr(bench, "read_frame"):
            try:
                ticks = bench.read_frame(include_defrag=True)
                if "defrag_start" in ticks and "defrag_end" in ticks:
                    gpu_text = f" gpu_us={(ticks['defrag_end'] - ticks['defrag_start']) / 1000.0:.1f}"
            except Exception as error:                                # noqa: BLE001
                gpu_text = f" gpu_us=unreadable({type(error).__name__})"
        print(f"[e30] defrag_time sim{index}: wall_ms={wall_ms:.3f}{gpu_text}", flush=True)
        return result

    chain_class.run_pipelined = run_pipelined_logging
    simulator_class.submit_defrag_and_wait = submit_defrag_and_wait_timed


INSTALL_END_LABELS = ("c_append_departed_end", "c_install_trailing_end", "c_install_leading_end", "c_expand_end")


def install_anatomy_full(solver: str) -> None:
    """E7 (--anatomy in the bench arguments): wrap bench_<solver>.compute_durations so that every duration it
    computes is printed (the bench's [anatomy] line keeps a fixed subset: no expand_lists / append_departed /
    band_compact / ghost sends ...), plus install_sum = the last install tick - c_start (expand_ghost_lists +
    install_migrations per direction + append_departed with their barriers, v6_opt.md "install 三个 kernel").
    The bench imports compute_durations inside its main, after this module attribute is replaced; it computes
    one frame per slab in slab order and prints its [anatomy] line right after, so call n pairs with the n-th
    [anatomy] line."""
    bench_module = solver_module(solver, "bench")
    original_compute = bench_module.compute_durations
    calls = [0]

    def compute_durations_printing(ticks):
        durations = original_compute(ticks)
        calls[0] += 1
        install_text = ""
        last = next((label for label in INSTALL_END_LABELS if label in ticks), None)
        if last is not None and "c_start" in ticks:
            install_text = f" install_sum={(ticks[last] - ticks['c_start']) / 1000.0:.3f} install_last={last}"
        print(f"[e30] anatomy_all call={calls[0]}: "
              + " ".join(f"{name[:-3] if name.endswith('_us') else name}={value:.3f}"
                         for name, value in durations.items() if isinstance(value, (int, float)))
              + install_text, flush=True)
        return durations

    bench_module.compute_durations = compute_durations_printing


def main() -> int:
    arguments, bench_arguments = parse_arguments(sys.argv[1:])
    # any spelling argparse would take for the bench's --switch-interval-ms (it accepts unique prefixes, "--sw")
    if any(len(argument.split("=", 1)[0]) >= 4 and "--switch-interval-ms".startswith(argument.split("=", 1)[0])
           for argument in bench_arguments):
        sys.exit("pass --switch-interval-ms before '--' only: this script forwards its own value to the bench")
    if any(argument.split("=", 1)[0] == "--solver" for argument in bench_arguments):
        sys.exit("pass --solver before '--': it selects the bench, the bench itself has no such option")
    if arguments.barrier is not None:
        if arguments.solver == "v6" or arguments.barrier_parties < 1:
            sys.exit("--barrier needs --barrier-parties >= 1 and a solver other than v6 (the stage reporter holds it)")
        if "--weights=auto" in bench_arguments or any(
                argument == "--weights" and index + 1 < len(bench_arguments) and bench_arguments[index + 1] == "auto"
                for index, argument in enumerate(bench_arguments)):
            sys.exit("--barrier with --weights auto: the first pilot chain would take the barrier")
    if arguments.defrag_log and arguments.solver == "v6":
        sys.exit("--defrag-log is an E7 option (solvers other than v6)")
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
    pool_peak_workers = (install_pool_peak_recorder(arguments.solver)
                         if os.environ.get(arguments.solver.upper() + "_POOL_PEAKS", "0") == "1" else None)
    bench_started = time.perf_counter()
    pool_health = None
    if arguments.solver != "v6":
        barrier = ((arguments.barrier, arguments.barrier_parties, arguments.barrier_timeout)
                   if arguments.barrier is not None else None)
        install_stage_reporter(arguments.solver, bench_started, barrier)
        pool_health = install_pool_health_recorder(arguments.solver)
        if arguments.defrag_log:
            install_defrag_log(arguments.solver)
        if "--anatomy" in bench_arguments:
            install_anatomy_full(arguments.solver)
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
        if pool_health is not None:
            print_pool_health(pool_health)
        if pool_peak_workers is not None:
            print_pool_peaks(pool_peak_workers)
        if arguments.solver != "v6":
            report_stage("exit", bench_started)
        sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
