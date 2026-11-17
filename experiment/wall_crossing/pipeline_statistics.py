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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--filter", default="correction_density_all,correction_density_interior,force_all,"
                                            "force_deep_interior_scratch,wall_extrapolate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[2]))
    arguments = parser.parse_args()

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
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        print(f"[pipeline_statistics] wrote {out_path}")
    simulator.destroy()
    context.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
