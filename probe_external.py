"""Cross-vendor external memory / semaphore capability probe.

Queries each discrete GPU for which Win32 external handle types can be exported and
imported. The intersection of NV's EXPORT set and AMD's IMPORT set (and vice versa)
determines whether Phase 2 shared-memory migration is feasible.
"""
from vulkan import *


INSTANCE_EXTENSIONS = [
    "VK_KHR_get_physical_device_properties2",
    "VK_KHR_external_memory_capabilities",
    "VK_KHR_external_semaphore_capabilities",
]


MEMORY_HANDLE_TYPES = [
    ("OPAQUE_WIN32",      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT),
    ("OPAQUE_WIN32_KMT",  VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT),
    ("D3D11_TEXTURE",     VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT),
    ("D3D11_TEXTURE_KMT", VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT),
    ("D3D12_HEAP",        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT),
    ("D3D12_RESOURCE",    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT),
]

SEMAPHORE_HANDLE_TYPES = [
    ("OPAQUE_WIN32",     VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT),
    ("OPAQUE_WIN32_KMT", VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT),
    ("D3D12_FENCE",      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT),
]


def create_probe_instance():
    app_info = VkApplicationInfo(
        pApplicationName="probe",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="probe",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 1, 0),
    )
    info = VkInstanceCreateInfo(
        pApplicationInfo=app_info,
        enabledExtensionCount=len(INSTANCE_EXTENSIONS),
        ppEnabledExtensionNames=INSTANCE_EXTENSIONS,
    )
    return vkCreateInstance(info, None)


def mem_features_str(flags):
    out = []
    if flags & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT:
        out.append("DEDICATED")
    if flags & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT:
        out.append("EXPORT")
    if flags & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT:
        out.append("IMPORT")
    return "+".join(out) if out else "-"


def sem_features_str(flags):
    out = []
    if flags & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT:
        out.append("EXPORT")
    if flags & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT:
        out.append("IMPORT")
    return "+".join(out) if out else "-"


def probe_memory(inst_fns, physical_device):
    """Returns dict: handle_type_name -> (features, compatible_bits)."""
    results = {}
    for name, handle_type in MEMORY_HANDLE_TYPES:
        ext_info = VkPhysicalDeviceExternalBufferInfo(
            flags=0,
            usage=(
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            ),
            handleType=handle_type,
        )
        try:
            props = inst_fns["vkGetPhysicalDeviceExternalBufferPropertiesKHR"](
                physical_device, ext_info,
            )
            emp = props.externalMemoryProperties
            results[name] = (emp.externalMemoryFeatures, emp.compatibleHandleTypes)
        except Exception as e:
            results[name] = (None, None)
    return results


def probe_semaphore(inst_fns, physical_device):
    results = {}
    for name, handle_type in SEMAPHORE_HANDLE_TYPES:
        info = VkPhysicalDeviceExternalSemaphoreInfo(handleType=handle_type)
        try:
            props = inst_fns["vkGetPhysicalDeviceExternalSemaphorePropertiesKHR"](
                physical_device, info,
            )
            results[name] = (props.externalSemaphoreFeatures, props.compatibleHandleTypes)
        except Exception as e:
            results[name] = (None, None)
    return results


def load_instance_functions(instance):
    names = [
        "vkGetPhysicalDeviceExternalBufferPropertiesKHR",
        "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR",
    ]
    fns = {}
    for name in names:
        fns[name] = vkGetInstanceProcAddr(instance, name)
    return fns


def main():
    instance = create_probe_instance()
    inst_fns = load_instance_functions(instance)

    devices = []
    for dev in vkEnumeratePhysicalDevices(instance):
        props = vkGetPhysicalDeviceProperties(dev)
        if props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            continue
        devices.append((props.deviceName, dev))

    if len(devices) < 2:
        print(f"Need ≥2 discrete GPUs; found {len(devices)}")
        return

    mem_results = []
    sem_results = []
    for name, dev in devices:
        mem_results.append((name, probe_memory(inst_fns, dev)))
        sem_results.append((name, probe_semaphore(inst_fns, dev)))

    # Print memory table
    print("=" * 90)
    print("EXTERNAL MEMORY (usage=STORAGE_BUFFER|TRANSFER_SRC|TRANSFER_DST)")
    print("=" * 90)
    header = f"{'handle type':<20}"
    for name, _ in mem_results:
        header += f"  {name[:32]:<32}"
    print(header)
    print("-" * 90)
    for handle_name, _ in MEMORY_HANDLE_TYPES:
        row = f"{handle_name:<20}"
        for _, results in mem_results:
            features, _ = results[handle_name]
            if features is None:
                row += f"  {'(query failed)':<32}"
            else:
                row += f"  {mem_features_str(features):<32}"
        print(row)

    # Print semaphore table
    print()
    print("=" * 90)
    print("EXTERNAL SEMAPHORE")
    print("=" * 90)
    header = f"{'handle type':<20}"
    for name, _ in sem_results:
        header += f"  {name[:32]:<32}"
    print(header)
    print("-" * 90)
    for handle_name, _ in SEMAPHORE_HANDLE_TYPES:
        row = f"{handle_name:<20}"
        for _, results in sem_results:
            features, _ = results[handle_name]
            if features is None:
                row += f"  {'(query failed)':<32}"
            else:
                row += f"  {sem_features_str(features):<32}"
        print(row)

    # Summary: what's viable for cross-vendor sharing?
    print()
    print("=" * 90)
    print("CROSS-VENDOR VIABILITY")
    print("=" * 90)
    print("A handle type is viable if one GPU can EXPORT and the other can IMPORT.")
    print()
    print("MEMORY:")
    for handle_name, _ in MEMORY_HANDLE_TYPES:
        flags_0 = mem_results[0][1][handle_name][0] or 0
        flags_1 = mem_results[1][1][handle_name][0] or 0
        e0 = bool(flags_0 & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)
        i0 = bool(flags_0 & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)
        e1 = bool(flags_1 & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)
        i1 = bool(flags_1 & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)
        dir_01 = e0 and i1
        dir_10 = e1 and i0
        if dir_01 and dir_10:
            verdict = "BIDIRECTIONAL"
        elif dir_01:
            verdict = f"{mem_results[0][0][:10]} -> {mem_results[1][0][:10]} only"
        elif dir_10:
            verdict = f"{mem_results[1][0][:10]} -> {mem_results[0][0][:10]} only"
        else:
            verdict = "not viable"
        print(f"  {handle_name:<20}  {verdict}")

    print()
    print("SEMAPHORE:")
    for handle_name, _ in SEMAPHORE_HANDLE_TYPES:
        flags_0 = sem_results[0][1][handle_name][0] or 0
        flags_1 = sem_results[1][1][handle_name][0] or 0
        e0 = bool(flags_0 & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)
        i0 = bool(flags_0 & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT)
        e1 = bool(flags_1 & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)
        i1 = bool(flags_1 & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT)
        dir_01 = e0 and i1
        dir_10 = e1 and i0
        if dir_01 and dir_10:
            verdict = "BIDIRECTIONAL"
        elif dir_01:
            verdict = f"{sem_results[0][0][:10]} -> {sem_results[1][0][:10]} only"
        elif dir_10:
            verdict = f"{sem_results[1][0][:10]} -> {sem_results[0][0][:10]} only"
        else:
            verdict = "not viable"
        print(f"  {handle_name:<20}  {verdict}")

    vkDestroyInstance(instance, None)


if __name__ == "__main__":
    main()
