"""List the Vulkan physical devices of this machine (raw enumeration order) with the properties the platform record needs."""
import json
import sys

import vulkan as vk

application = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO, pApplicationName="probe", applicationVersion=1,
                                   pEngineName="probe", engineVersion=1, apiVersion=vk.VK_MAKE_VERSION(1, 1, 0))
instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=application), None)
loader_version = vk.vkEnumerateInstanceVersion()


def version_string(value):
    return f"{value >> 22}.{(value >> 12) & 0x3ff}.{value & 0xfff}"


def nvidia_driver_string(value):
    return f"{value >> 22}.{(value >> 14) & 0xff}.{(value >> 6) & 0xff}.{value & 0x3f}"


rows = []
for index, device in enumerate(vk.vkEnumeratePhysicalDevices(instance)):
    properties = vk.vkGetPhysicalDeviceProperties(device)
    row = {"index": index, "name": properties.deviceName, "type": int(properties.deviceType),
           "api": version_string(properties.apiVersion), "driver_raw": int(properties.driverVersion),
           "vendor": hex(properties.vendorID), "device_id": hex(properties.deviceID),
           "timestamp_period_ns": float(properties.limits.timestampPeriod)}
    extensions = {e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(device, None)}
    row["has_pci_bus_info"] = "VK_EXT_pci_bus_info" in extensions
    row["has_calibrated_timestamps"] = "VK_EXT_calibrated_timestamps" in extensions or "VK_KHR_calibrated_timestamps" in extensions
    row["has_external_memory_win32"] = "VK_KHR_external_memory_win32" in extensions
    try:
        driver = vk.VkPhysicalDeviceDriverProperties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES)
        pci = vk.VkPhysicalDevicePCIBusInfoPropertiesEXT(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT, pNext=driver)
        ids = vk.VkPhysicalDeviceIDProperties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, pNext=pci)
        properties2 = vk.VkPhysicalDeviceProperties2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=ids)
        getter = vk.vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2")
        vk.vkGetPhysicalDeviceProperties2(device, properties2)
        row["uuid"] = bytes(ids.deviceUUID).hex()
        row["pci"] = f"{pci.pciDomain:04x}:{pci.pciBus:02x}:{pci.pciDevice:02x}.{pci.pciFunction}"
        row["driver_name"] = driver.driverName
        row["driver_info"] = driver.driverInfo
        row["conformance"] = f"{driver.conformanceVersion.major}.{driver.conformanceVersion.minor}.{driver.conformanceVersion.subminor}.{driver.conformanceVersion.patch}"
    except Exception as error:  # noqa: BLE001
        row["properties2_error"] = repr(error)
    memory = vk.vkGetPhysicalDeviceMemoryProperties(device)
    row["device_local_heaps_GiB"] = [round(memory.memoryHeaps[i].size / 2**30, 2) for i in range(memory.memoryHeapCount)
                                     if memory.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT]
    row["driver_decoded"] = nvidia_driver_string(row["driver_raw"]) if properties.vendorID == 0x10DE else version_string(row["driver_raw"])
    queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
    row["queue_families"] = [f"{q.queueCount}x{'G' if q.queueFlags & 1 else ''}{'C' if q.queueFlags & 2 else ''}{'T' if q.queueFlags & 4 else ''}" for q in queues]
    rows.append(row)

print(json.dumps({"loader_api": version_string(loader_version), "python": sys.version.split()[0], "devices": rows}, indent=1, ensure_ascii=False))
