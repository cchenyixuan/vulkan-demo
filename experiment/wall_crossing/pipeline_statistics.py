"""
pipeline_statistics.py - the driver's compile statistics (register count, binary size ...) of a case's compute
pipelines through VK_KHR_pipeline_executable_properties (the method of docs/wall_bc/scripts/w3_pipeline_stats.py
on branch v7-wall, for this tree's wall options).

It builds the real K = 1 simulator (SphSimulatorV7 on the slab of a one-slab chain partition: the SPIR-V and
specialisation a production K = 1 run uses) on one device with the extension enabled and every compute pipeline
created with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, then prints the statistics of the pipelines whose
name starts with one of --filter. Nothing is dispatched. The statistic names are the driver's.

    .venv/Scripts/python.exe experiment/wall_crossing/pipeline_statistics.py --case cases/.../case.yaml --device 1 \
        [--filter correction_density_all,force_all] [--out logs/wall_crossing/statistics/x.json] [--repo CHECKOUT]

--repo runs another checkout's solver (the base commit's worktree): that tree goes first on sys.path, so start
this script by its path (not with -m) and give --case relative to that checkout or absolute.
"""
import argparse
import json
import os
import pathlib
import sys


def compare_reports(first_path: str, second_path: str) -> int:
    """Are two reports' simulators made of the same pipelines: the same pipeline names, each with the same
    SPIR-V module (sha256) and specialization entries, the same buffer specs, and the same driver statistics
    for the pipelines both list. Exit code 0 = identical."""
    first = json.loads(pathlib.Path(first_path).read_text(encoding="utf-8"))
    second = json.loads(pathlib.Path(second_path).read_text(encoding="utf-8"))
    problems = []
    names_first, names_second = set(first["recipes"]), set(second["recipes"])
    if names_first != names_second:
        problems.append(f"pipeline sets differ: only first {sorted(names_first - names_second)}, "
                        f"only second {sorted(names_second - names_first)}")
    for name in sorted(names_first & names_second):
        if first["recipes"][name]["module_sha256"] != second["recipes"][name]["module_sha256"]:
            problems.append(f"{name}: SPIR-V module differs")
        if first["recipes"][name]["entries"] != second["recipes"][name]["entries"]:
            problems.append(f"{name}: specialization entries differ")
    if first["buffer_specs"] != second["buffer_specs"]:
        problems.append("buffer specs differ")
    for name in sorted(set(first["pipelines"]) & set(second["pipelines"])):
        if first["pipelines"][name] != second["pipelines"][name]:
            problems.append(f"{name}: driver statistics differ")
    print(f"[pipeline_statistics] {first_path} vs {second_path}: {len(names_first & names_second)} pipelines, "
          f"{len(first['buffer_specs'])} buffers")
    for problem in problems:
        print(f"  DIFFERENT: {problem}")
    print("  -> " + ("IDENTICAL (modules, specialization entries, buffer specs, driver statistics)"
                     if not problems else f"{len(problems)} difference(s)"))
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=False)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--filter", default="correction_density_all,correction_density_interior,force_all,"
                                            "force_deep_interior_scratch,wall_extrapolate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[2]))
    parser.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"), default=None,
                        help="compare two reports written with --out: the pipelines' modules and specialization "
                             "entries, the buffer specs and the register counts (no GPU)")
    arguments, _ = parser.parse_known_args()
    if arguments.compare:
        return compare_reports(*arguments.compare)
    arguments = parser.parse_args()
    if not arguments.case or arguments.device is None:
        parser.error("--case and --device are required (or --compare FIRST SECOND)")

    out_path = pathlib.Path(arguments.out).resolve() if arguments.out else None
    repository = pathlib.Path(arguments.repo).resolve()
    if "experiment" in sys.modules:
        sys.exit("experiment/ is already imported: start this script by its path, not with -m")
    sys.path.insert(0, str(repository))
    os.chdir(repository)
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")

    import vulkan as vk
    from vulkan import ffi
    from experiment.v7.utils import simulator_v7 as simulator_module
    from experiment.v7.utils import vulkan_context_v7 as context_module
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    if pathlib.Path(simulator_module.__file__).resolve().parents[3] != repository:
        sys.exit(f"simulator_v7 came from {simulator_module.__file__}, not from {repository}")

    # the feature the extension needs, chained behind the context's Vulkan 1.3 features (kept alive here)
    extension_features = vk.VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR(pipelineExecutableInfo=vk.VK_TRUE)
    original_features = context_module.VkPhysicalDeviceVulkan13Features

    def features_with_extension(*positional, **keywords):
        keywords.setdefault("pNext", extension_features)
        return original_features(*positional, **keywords)
    context_module.VkPhysicalDeviceVulkan13Features = features_with_extension

    original_create_info = simulator_module.VkComputePipelineCreateInfo

    def create_info_with_statistics(*positional, **keywords):
        keywords["flags"] = keywords.get("flags", 0) | vk.VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
        return original_create_info(*positional, **keywords)
    simulator_module.VkComputePipelineCreateInfo = create_info_with_statistics

    # what each pipeline is made of: the sha256 of its SPIR-V module and its specialization entries, recorded as
    # the simulator creates them (two builds whose every pipeline has the same module and entries, and the same
    # buffer specs, hand the driver the same input)
    import hashlib
    module_hashes: dict = {}
    original_create_module = simulator_module.vkCreateShaderModule

    def create_module_recording(device, create_info, allocator):
        module = original_create_module(device, create_info, allocator)
        code = bytes(ffi.buffer(create_info.pCode, create_info.codeSize)) if not isinstance(
            create_info.pCode, (bytes, bytearray)) else bytes(create_info.pCode)
        module_hashes[id(module)] = hashlib.sha256(code).hexdigest()
        return module
    simulator_module.vkCreateShaderModule = create_module_recording

    pipeline_recipes: list = []
    original_create_pipeline = simulator_module.SphSimulatorV7._create_pipeline

    def create_pipeline_recording(self, shader, entries):
        pipeline = original_create_pipeline(self, shader, entries)
        pipeline_recipes.append((pipeline, module_hashes.get(id(shader)),
                                 [[int(constant), kind, (float(value) if kind == "f" else int(value))]
                                  for constant, kind, value in entries]))
        return pipeline
    simulator_module.SphSimulatorV7._create_pipeline = create_pipeline_recording

    case = load_case_v7(arguments.case)
    chain = compute_chain_partition(case, [1.0])
    context = context_module.VulkanContextV7.create(
        device_index=arguments.device, enable_validation=False, application_name="wall_crossing_pipeline_statistics",
        extra_device_extensions=[vk.VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME])
    simulator = simulator_module.SphSimulatorV7(context, chain.slabs[0])
    get_properties = vk.vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutablePropertiesKHR")
    get_statistics = vk.vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutableStatisticsKHR")

    def text(value) -> str:
        try:
            return value if isinstance(value, str) else ffi.string(value).decode("utf-8", "replace")
        except Exception:
            return str(value)

    def statistic_value(statistic):
        value_format = int(statistic.format)
        if value_format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
            return bool(statistic.value.b32)
        if value_format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
            return int(statistic.value.i64)
        if value_format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
            return int(statistic.value.u64)
        return float(statistic.value.f64)

    try:
        from experiment.validation.cavity_runner import device_uuid
        uuid = device_uuid(context)
    except Exception:
        uuid = None
    wanted = [name for name in arguments.filter.split(",") if name]
    report = {"case": arguments.case, "repository": str(repository).replace("\\", "/"),
              "device": arguments.device, "device_name": getattr(context, "device_name", None), "device_uuid": uuid,
              "dimension": int(case.physics.dimension), "wall_boundary": case.numerics.wall_boundary,
              "pipelines": {}}
    print(f"[pipeline_statistics] {pathlib.Path(arguments.case).parent.name}: wall_boundary "
          f"{case.numerics.wall_boundary}, {int(case.physics.dimension)}-D, {len(simulator.pipelines)} pipelines, "
          f"device uuid {uuid}")
    for name in sorted(simulator.pipelines):
        if not any(name.startswith(prefix) for prefix in wanted):
            continue
        pipeline = simulator.pipelines[name]
        rows = []
        for index, executable in enumerate(get_properties(context.device, vk.VkPipelineInfoKHR(pipeline=pipeline))):
            statistics = get_statistics(context.device,
                                        vk.VkPipelineExecutableInfoKHR(pipeline=pipeline, executableIndex=index))
            row = {"executable": text(executable.name), "subgroup_size": int(executable.subgroupSize),
                   "statistics": {text(statistic.name): statistic_value(statistic) for statistic in statistics}}
            rows.append(row)
            print(f"  {name} [{row['executable']}]: "
                  + ", ".join(f"{key} = {value}" for key, value in row["statistics"].items()))
        report["pipelines"][name] = rows
    # every pipeline of the simulator (not only the filtered ones): module hash + specialization entries, and
    # the buffer specs; compare two reports with --compare
    recipes = {}
    for name, pipeline in simulator.pipelines.items():
        for recorded, module_hash, entries in pipeline_recipes:
            if recorded is pipeline:
                recipes[name] = {"module_sha256": module_hash, "entries": entries}
                break
    report["recipes"] = recipes
    report["buffer_specs"] = [[spec.name, int(spec.set_index), int(spec.binding), int(spec.size), int(spec.usage)]
                              for spec in simulator._buffer_specs]
    missing = sorted(set(simulator.pipelines) - set(recipes))
    print(f"[pipeline_statistics] recipes of {len(recipes)} pipelines recorded"
          + (f" (not created through _create_pipeline: {missing})" if missing else ""))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        print(f"[pipeline_statistics] wrote {out_path}")
    simulator.destroy()
    context.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
