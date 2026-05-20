"""
vulkan_context_v2.py — V2-internal Vulkan platform layer.

V2-specific differences vs V0/V1's utils/sph/vulkan_context.py:

  - apiVersion = 1.3  (V0 used 1.2). 1.3 promotes sync2 / vkQueueSubmit2 to
    core, so V2 can call them without enabling VK_KHR_synchronization2 as an
    extension. Shader target-env stays at vulkan1.2 (SPIR-V 1.5) per V1 perf
    memory; apiVersion and shader target-env are independent.

  - Two device features enabled via pNext-chained VkPhysicalDeviceFeatures2:
        VkPhysicalDeviceVulkan12Features.timelineSemaphore     = VK_TRUE
        VkPhysicalDeviceVulkan13Features.synchronization2      = VK_TRUE

  - find_memory_type accepts (required, preferred) so HOST_CACHED can degrade
    gracefully when a device doesn't expose the combo (cross-vendor robustness;
    cf. docs/sph_v2_design.md §14.5).

  - Per V2 isolation: this module does NOT import utils/sph/. Implementation
    is parallel to V0's, not a subclass. Bug fixes here do not propagate to
    V0/V1 and vice versa, by design.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi


# ============================================================================
# Constants
# ============================================================================

VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"
DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils"

# V2 uses Vulkan 1.3 to get sync2 + timeline as core (no extension dance).
TARGET_API_VERSION = VK_MAKE_VERSION(1, 3, 0)


# ============================================================================
# Debug messenger
# ============================================================================

@ffi.callback(
    "VkBool32(VkDebugUtilsMessageSeverityFlagBitsEXT, "
    "VkDebugUtilsMessageTypeFlagsEXT, "
    "VkDebugUtilsMessengerCallbackDataEXT *, "
    "void *)"
)
def _debug_callback(severity, message_type, callback_data, user_data):
    if severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        label = "\033[91mERROR\033[0m"
    elif severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        label = "\033[93mWARNING\033[0m"
    elif severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        label = "\033[94mINFO\033[0m"
    else:
        label = "VERBOSE"
    message = ffi.string(callback_data.pMessage).decode("utf-8")
    print(f"[Vulkan {label}] {message}", file=sys.stderr)
    return VK_FALSE


def _check_validation_layer_support() -> bool:
    available = vkEnumerateInstanceLayerProperties()
    return any(layer.layerName == VALIDATION_LAYER_NAME for layer in available)


# ============================================================================
# Physical device pretty-printing + selection
# ============================================================================

_DEVICE_TYPE_NAMES = {
    VK_PHYSICAL_DEVICE_TYPE_OTHER:          "OTHER",
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "INTEGRATED",
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   "DISCRETE",
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    "VIRTUAL",
    VK_PHYSICAL_DEVICE_TYPE_CPU:            "CPU",
}


def _summarize_queue_family(queue_family_properties) -> str:
    flags = queue_family_properties.queueFlags
    parts = []
    if flags & VK_QUEUE_GRAPHICS_BIT:       parts.append("graphics")
    if flags & VK_QUEUE_COMPUTE_BIT:        parts.append("compute")
    if flags & VK_QUEUE_TRANSFER_BIT:       parts.append("transfer")
    if flags & VK_QUEUE_SPARSE_BINDING_BIT: parts.append("sparse")
    return f"{'+'.join(parts) if parts else 'none'} (count={queue_family_properties.queueCount})"


def _find_compute_queue_family(physical_device) -> Optional[int]:
    for index, props in enumerate(
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
        if props.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return index
    return None


def _select_physical_device(instance, requested_device_index: Optional[int]):
    physical_devices = vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        raise RuntimeError("vkEnumeratePhysicalDevices returned no devices")

    print(f"[VulkanContextV2] available physical devices ({len(physical_devices)}):")

    auto_selected_index = None
    auto_selected_queue_family_index = None
    for device_index, physical_device in enumerate(physical_devices):
        properties = vkGetPhysicalDeviceProperties(physical_device)
        device_type = _DEVICE_TYPE_NAMES.get(properties.deviceType, "UNKNOWN")
        api_version = (f"{VK_VERSION_MAJOR(properties.apiVersion)}."
                       f"{VK_VERSION_MINOR(properties.apiVersion)}."
                       f"{VK_VERSION_PATCH(properties.apiVersion)}")
        queue_family_index = _find_compute_queue_family(physical_device)
        print(f"  [{device_index}] {properties.deviceName}  "
              f"type={device_type}  api={api_version}")
        for family_index, family_properties in enumerate(
                vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
            marker = "  <- compute pick" if family_index == queue_family_index else ""
            print(f"        queue_family[{family_index}]: "
                  f"{_summarize_queue_family(family_properties)}{marker}")
        if requested_device_index is None and auto_selected_index is None and queue_family_index is not None:
            auto_selected_index = device_index
            auto_selected_queue_family_index = queue_family_index

    if requested_device_index is not None:
        if not (0 <= requested_device_index < len(physical_devices)):
            raise RuntimeError(
                f"device_index={requested_device_index} out of range "
                f"(only {len(physical_devices)} present)")
        candidate = physical_devices[requested_device_index]
        candidate_queue_family_index = _find_compute_queue_family(candidate)
        if candidate_queue_family_index is None:
            props = vkGetPhysicalDeviceProperties(candidate)
            raise RuntimeError(
                f"requested device_index={requested_device_index} "
                f"({props.deviceName}) has no compute queue family")
        selected_index, selected_queue_family_index = requested_device_index, candidate_queue_family_index
    else:
        if auto_selected_index is None:
            raise RuntimeError(
                "no physical device exposes a queue family with VK_QUEUE_COMPUTE_BIT")
        selected_index, selected_queue_family_index = auto_selected_index, auto_selected_queue_family_index

    selected_props = vkGetPhysicalDeviceProperties(physical_devices[selected_index])
    print(f"[VulkanContextV2] selected: [{selected_index}] {selected_props.deviceName} "
          f"on queue_family[{selected_queue_family_index}]")
    return physical_devices[selected_index], selected_queue_family_index


# ============================================================================
# VulkanContextV2
# ============================================================================

@dataclass
class VulkanContextV2:
    """V2 compute-only Vulkan session handle bundle.

    Use ``VulkanContextV2.create(...)``; destroy via ``destroy()`` or
    context manager.
    """
    instance: object
    physical_device: object
    device: object
    compute_queue: object
    compute_queue_family_index: int
    command_pool: object
    device_name: str

    _validation_enabled: bool = False
    _debug_messenger: Optional[object] = None
    _memory_properties: Optional[object] = field(default=None, repr=False)
    _destroyed: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        application_name: str = "sph_v2",
        enable_validation: bool = True,
        extra_instance_extensions: Optional[list] = None,
        extra_device_extensions: Optional[list] = None,
        device_index: Optional[int] = None,
    ) -> "VulkanContextV2":
        # ---- Validation availability ------------------------------------
        validation_supported = _check_validation_layer_support()
        if enable_validation and not validation_supported:
            print(f"[VulkanContextV2] WARNING: validation layer requested but "
                  f"not available; continuing without.", file=sys.stderr)
        validation_active = enable_validation and validation_supported

        layers = [VALIDATION_LAYER_NAME] if validation_active else []
        extensions = [DEBUG_UTILS_EXTENSION] if validation_active else []
        if extra_instance_extensions:
            extensions.extend(extra_instance_extensions)

        # ---- Instance ---------------------------------------------------
        application_info = VkApplicationInfo(
            pApplicationName=application_name,
            applicationVersion=VK_MAKE_VERSION(0, 1, 0),
            pEngineName="sph_v2",
            engineVersion=VK_MAKE_VERSION(0, 1, 0),
            apiVersion=TARGET_API_VERSION,
        )
        instance_create_info = VkInstanceCreateInfo(
            pApplicationInfo=application_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        )
        instance = vkCreateInstance(instance_create_info, None)

        # ---- Debug messenger (optional) ---------------------------------
        debug_messenger = None
        if validation_active:
            create_fn = vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
            if create_fn is not None:
                messenger_info = VkDebugUtilsMessengerCreateInfoEXT(
                    messageSeverity=(
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT),
                    messageType=(
                        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT),
                    pfnUserCallback=_debug_callback,
                )
                debug_messenger = create_fn(instance, messenger_info, None)

        # ---- Physical device + compute queue family ---------------------
        physical_device, compute_queue_family_index = _select_physical_device(instance, device_index)
        device_name = vkGetPhysicalDeviceProperties(physical_device).deviceName

        # ---- Logical device ---------------------------------------------
        # V2 must enable BOTH timelineSemaphore (1.2 core feature) and
        # synchronization2 (1.3 core feature). Both default to FALSE per
        # spec; opt-in via pNext-chained VkPhysicalDeviceFeatures2.
        queue_create_info = VkDeviceQueueCreateInfo(
            queueFamilyIndex=compute_queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )

        # pNext chain (innermost first; chain pNext from outer to inner):
        #   VkDeviceCreateInfo.pNext ─→ Features2
        #     Features2.pNext        ─→ Vulkan12Features  (timelineSemaphore)
        #       Vulkan12Features.pNext ─→ Vulkan13Features (synchronization2)
        features_1_3 = VkPhysicalDeviceVulkan13Features(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            synchronization2=VK_TRUE,
        )
        features_1_2 = VkPhysicalDeviceVulkan12Features(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            timelineSemaphore=VK_TRUE,
            pNext=features_1_3,
        )
        # vertexPipelineStoresAndAtomics: V0/V1 already use this for the
        # render path (vert reads SoA SSBOs). We carry it forward so the
        # V2 render pipeline (added Phase 5+) doesn't need a feature dance.
        features_2 = VkPhysicalDeviceFeatures2(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            features=VkPhysicalDeviceFeatures(
                vertexPipelineStoresAndAtomics=VK_TRUE,
            ),
            pNext=features_1_2,
        )

        device_extension_list = list(extra_device_extensions) if extra_device_extensions else []
        device_create_info = VkDeviceCreateInfo(
            pNext=features_2,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledLayerCount=0,
            ppEnabledLayerNames=[],
            enabledExtensionCount=len(device_extension_list),
            ppEnabledExtensionNames=device_extension_list,
            # pEnabledFeatures MUST be NULL when pNext chain contains
            # VkPhysicalDeviceFeatures2 (Vulkan spec rule).
            pEnabledFeatures=None,
        )
        device = vkCreateDevice(physical_device, device_create_info, None)
        compute_queue = vkGetDeviceQueue(device, compute_queue_family_index, 0)

        # ---- Command pool -----------------------------------------------
        command_pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=compute_queue_family_index,
        ), None)

        memory_properties = vkGetPhysicalDeviceMemoryProperties(physical_device)

        return cls(
            instance=instance,
            physical_device=physical_device,
            device=device,
            compute_queue=compute_queue,
            compute_queue_family_index=compute_queue_family_index,
            command_pool=command_pool,
            device_name=device_name,
            _validation_enabled=validation_active,
            _debug_messenger=debug_messenger,
            _memory_properties=memory_properties,
        )

    # ------------------------------------------------------------------

    def find_memory_type(
        self,
        type_bits: int,
        required_properties: int,
        preferred_properties: int = 0,
    ) -> int:
        """Return first memory type matching ``type_bits`` and providing all
        ``required_properties`` flags.

        If ``preferred_properties`` is non-zero, first try to find a type that
        also provides those flags; on failure, fall back to required-only.
        Caller can detect the fallback by checking memoryTypes[returned].propertyFlags.

        Raises RuntimeError if no type matches required.
        """
        # First pass: required + preferred (if preferred != 0)
        if preferred_properties:
            want = required_properties | preferred_properties
            for i in range(self._memory_properties.memoryTypeCount):
                if not (type_bits & (1 << i)):
                    continue
                if (self._memory_properties.memoryTypes[i].propertyFlags & want) == want:
                    return i
            # Fall through to required-only

        for i in range(self._memory_properties.memoryTypeCount):
            if not (type_bits & (1 << i)):
                continue
            if (self._memory_properties.memoryTypes[i].propertyFlags & required_properties) == required_properties:
                return i

        raise RuntimeError(
            f"no memory type satisfies type_bits=0x{type_bits:x} "
            f"required=0x{required_properties:x} preferred=0x{preferred_properties:x}")

    def memory_type_flags(self, type_index: int) -> int:
        """propertyFlags of memoryTypes[type_index]. Useful after
        find_memory_type to confirm whether preferred fallback fired."""
        return self._memory_properties.memoryTypes[type_index].propertyFlags

    def submit_and_wait(self, command_buffer) -> None:
        """Fence-wait submit; for bootstrap / readback / defrag (not in
        timeline). Mirrors V0's helper."""
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        fence = vkCreateFence(self.device, VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        try:
            vkQueueSubmit(self.compute_queue, 1, submit_info, fence)
            vkWaitForFences(self.device, 1, [fence], VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        finally:
            vkDestroyFence(self.device, fence, None)

    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self.device is not None:
            vkDeviceWaitIdle(self.device)
        if self.command_pool is not None:
            vkDestroyCommandPool(self.device, self.command_pool, None)
            self.command_pool = None
        if self.device is not None:
            vkDestroyDevice(self.device, None)
            self.device = None
        if self._debug_messenger is not None:
            destroy_fn = vkGetInstanceProcAddr(
                self.instance, "vkDestroyDebugUtilsMessengerEXT")
            if destroy_fn is not None:
                destroy_fn(self.instance, self._debug_messenger, None)
            self._debug_messenger = None
        if self.instance is not None:
            vkDestroyInstance(self.instance, None)
            self.instance = None
        self._destroyed = True

    def __enter__(self) -> "VulkanContextV2":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
