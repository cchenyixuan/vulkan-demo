"""
vulkan_context_single.py — the Vulkan session of v7_single: one compute queue, one command pool.

What v7's VulkanContextV7 has and this one does not (docs/seam_audit/v7_single.md, inventory 1):
the transfer-only queue family with its two queues and command pool (the host-staged ghost
transport's DMA engines), the timelineSemaphore and vertexPipelineStoresAndAtomics device features
(frame pipelining across sims; the viewer), the optional calibrated-timestamps extension (step
traces). Kept: apiVersion 1.3, the discrete-first stable device order (the same index names the
same card as in v7), synchronization2 (every barrier is a vkCmdPipelineBarrier2), the optional
validation layer with a debug messenger that also counts its messages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi

VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"
DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils"
TARGET_API_VERSION = VK_MAKE_VERSION(1, 3, 0)

# Messages the debug messenger received in this process, by severity name; the runners print them.
VALIDATION_MESSAGE_COUNTS: dict[str, int] = {"ERROR": 0, "WARNING": 0, "INFO": 0, "VERBOSE": 0}


@ffi.callback(
    "VkBool32(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, "
    "VkDebugUtilsMessengerCallbackDataEXT *, void *)")
def _debug_callback(severity, message_type, callback_data, user_data):
    if severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        label = "ERROR"
    elif severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        label = "WARNING"
    elif severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        label = "INFO"
    else:
        label = "VERBOSE"
    VALIDATION_MESSAGE_COUNTS[label] += 1
    message = ffi.string(callback_data.pMessage).decode("utf-8", errors="replace")
    print(f"[Vulkan {label}] {message}", file=sys.stderr, flush=True)
    return VK_FALSE


def validation_message_total() -> int:
    return sum(VALIDATION_MESSAGE_COUNTS.values())


def _validation_layer_available() -> bool:
    return any(layer.layerName == VALIDATION_LAYER_NAME for layer in vkEnumerateInstanceLayerProperties())


def _find_compute_queue_family(physical_device) -> Optional[int]:
    for index, properties in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
        if properties.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return index
    return None


def enumerate_devices(instance) -> list:
    """Physical devices in v7's discrete-first stable order (the raw order is not stable across reboots)."""
    devices = vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise RuntimeError("vkEnumeratePhysicalDevices returned no devices")
    return sorted(devices, key=lambda device: 0 if vkGetPhysicalDeviceProperties(device).deviceType
                  == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1)


def device_uuid_hex(physical_device) -> str:
    id_properties = VkPhysicalDeviceIDProperties()
    vkGetPhysicalDeviceProperties2(physical_device, VkPhysicalDeviceProperties2(pNext=id_properties))
    return bytes(id_properties.deviceUUID).hex()


@dataclass
class VulkanContextSingle:
    instance: object
    physical_device: object
    device: object
    compute_queue: object
    compute_queue_family_index: int
    command_pool: object
    device_name: str
    physical_device_index: int
    device_uuid: str
    validation_enabled: bool = False
    _debug_messenger: Optional[object] = None
    _memory_properties: Optional[object] = field(default=None, repr=False)
    _destroyed: bool = field(default=False, repr=False)

    @classmethod
    def create(cls, device_index: Optional[int] = None, enable_validation: bool = False,
               application_name: str = "sph_v7_single") -> "VulkanContextSingle":
        validation_active = enable_validation and _validation_layer_available()
        if enable_validation and not validation_active:
            print("[VulkanContextSingle] WARNING: validation layer requested but not available", file=sys.stderr)
        layers = [VALIDATION_LAYER_NAME] if validation_active else []
        extensions = [DEBUG_UTILS_EXTENSION] if validation_active else []
        instance = vkCreateInstance(VkInstanceCreateInfo(
            pApplicationInfo=VkApplicationInfo(
                pApplicationName=application_name, applicationVersion=VK_MAKE_VERSION(0, 1, 0),
                pEngineName="sph_v7_single", engineVersion=VK_MAKE_VERSION(0, 1, 0),
                apiVersion=TARGET_API_VERSION),
            enabledLayerCount=len(layers), ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions), ppEnabledExtensionNames=extensions), None)
        debug_messenger = None
        if validation_active:
            create_messenger = vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
            if create_messenger is not None:
                debug_messenger = create_messenger(instance, VkDebugUtilsMessengerCreateInfoEXT(
                    messageSeverity=(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                     | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT),
                    messageType=(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                 | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                 | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT),
                    pfnUserCallback=_debug_callback), None)

        devices = enumerate_devices(instance)
        print(f"[VulkanContextSingle] physical devices ({len(devices)}, discrete-first stable order):")
        for index, device in enumerate(devices):
            properties = vkGetPhysicalDeviceProperties(device)
            print(f"  [{index}] {properties.deviceName}  type={properties.deviceType}  "
                  f"compute family {_find_compute_queue_family(device)}")
        if device_index is None:
            device_index = next((index for index, device in enumerate(devices)
                                 if _find_compute_queue_family(device) is not None), None)
            if device_index is None:
                raise RuntimeError("no physical device exposes a compute queue family")
        if not 0 <= device_index < len(devices):
            raise RuntimeError(f"device_index={device_index} out of range (only {len(devices)} present)")
        physical_device = devices[device_index]
        queue_family_index = _find_compute_queue_family(physical_device)
        if queue_family_index is None:
            raise RuntimeError(f"device [{device_index}] has no compute queue family")
        properties = vkGetPhysicalDeviceProperties(physical_device)
        device_name = properties.deviceName
        print(f"[VulkanContextSingle] selected: [{device_index}] {device_name} on queue_family[{queue_family_index}]")

        features_1_3 = VkPhysicalDeviceVulkan13Features(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, synchronization2=VK_TRUE)
        features_2 = VkPhysicalDeviceFeatures2(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, features=VkPhysicalDeviceFeatures(),
            pNext=features_1_3)
        device = vkCreateDevice(physical_device, VkDeviceCreateInfo(
            pNext=features_2,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[VkDeviceQueueCreateInfo(queueFamilyIndex=queue_family_index, queueCount=1,
                                                       pQueuePriorities=[1.0])],
            enabledLayerCount=0, ppEnabledLayerNames=[],
            enabledExtensionCount=0, ppEnabledExtensionNames=[],
            pEnabledFeatures=None), None)
        compute_queue = vkGetDeviceQueue(device, queue_family_index, 0)
        command_pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=queue_family_index), None)
        return cls(instance=instance, physical_device=physical_device, device=device,
                   compute_queue=compute_queue, compute_queue_family_index=queue_family_index,
                   command_pool=command_pool, device_name=device_name, physical_device_index=device_index,
                   device_uuid=device_uuid_hex(physical_device), validation_enabled=validation_active,
                   _debug_messenger=debug_messenger,
                   _memory_properties=vkGetPhysicalDeviceMemoryProperties(physical_device))

    def find_memory_type(self, type_bits: int, required_properties: int, preferred_properties: int = 0) -> int:
        """First memory type of ``type_bits`` with ``required_properties`` (and ``preferred_properties``
        when any type offers them), as in v7."""
        memory = self._memory_properties
        if preferred_properties:
            wanted = required_properties | preferred_properties
            for index in range(memory.memoryTypeCount):
                if type_bits & (1 << index) and (memory.memoryTypes[index].propertyFlags & wanted) == wanted:
                    return index
        for index in range(memory.memoryTypeCount):
            if (type_bits & (1 << index)
                    and (memory.memoryTypes[index].propertyFlags & required_properties) == required_properties):
                return index
        raise RuntimeError(f"no memory type satisfies type_bits=0x{type_bits:x} required=0x{required_properties:x}")

    def submit_and_wait(self, command_buffer) -> None:
        """One-shot submit with a fresh fence (bootstrap, uploads, readbacks; never the step loop)."""
        fence = vkCreateFence(self.device, VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        try:
            vkQueueSubmit(self.compute_queue, 1, VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[command_buffer]), fence)
            vkWaitForFences(self.device, 1, [fence], VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        finally:
            vkDestroyFence(self.device, fence, None)

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self.device is not None:
            vkDeviceWaitIdle(self.device)
            vkDestroyCommandPool(self.device, self.command_pool, None)
            vkDestroyDevice(self.device, None)
            self.device = None
        if self._debug_messenger is not None:
            destroy_messenger = vkGetInstanceProcAddr(self.instance, "vkDestroyDebugUtilsMessengerEXT")
            if destroy_messenger is not None:
                destroy_messenger(self.instance, self._debug_messenger, None)
            self._debug_messenger = None
        if self.instance is not None:
            vkDestroyInstance(self.instance, None)
            self.instance = None
        self._destroyed = True

    def __enter__(self) -> "VulkanContextSingle":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
