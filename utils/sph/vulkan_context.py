"""
vulkan_context.py — generic Vulkan platform setup (instance + device + queues
+ command pool). Knows nothing about SPH; future renderer / multi-GPU work
can reuse this layer as-is.

V0 scope:
  - Headless single GPU (no surface / swapchain / GLFW dependency)
  - Pick first physical device that exposes a queue family with COMPUTE_BIT
    (graphics+compute combined is fine; dedicated compute is not required)
  - Validation layers + debug messenger always on by default
  - One command pool on the chosen compute queue family

Public surface:
    VulkanContext.create(application_name=..., enable_validation=True)
    ctx.find_memory_type(type_bits, required_properties)
    ctx.submit_and_wait(command_buffer)
    ctx.destroy()
    with VulkanContext.create() as ctx: ...
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

from vulkan import *
from vulkan._vulkancache import ffi


# ============================================================================
# Constants
# ============================================================================

VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"
DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils"

# Vulkan 1.2 — matches glslc --target-env=vulkan1.2 in compile_shaders.py.
TARGET_API_VERSION = VK_MAKE_VERSION(1, 2, 0)


# ============================================================================
# Debug messenger
# ============================================================================

# The CFFI signature string MUST match VkDebugUtilsMessengerCallbackEXT exactly.
# Keeping the callback at module scope (not nested inside a function) prevents
# Python from garbage-collecting the cffi handle while Vulkan still holds it.
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
    available_names = {layer.layerName for layer in available}
    return VALIDATION_LAYER_NAME in available_names


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
    """One-line summary of a queue family's capabilities + count."""
    flags = queue_family_properties.queueFlags
    parts = []
    if flags & VK_QUEUE_GRAPHICS_BIT:       parts.append("graphics")
    if flags & VK_QUEUE_COMPUTE_BIT:        parts.append("compute")
    if flags & VK_QUEUE_TRANSFER_BIT:       parts.append("transfer")
    if flags & VK_QUEUE_SPARSE_BINDING_BIT: parts.append("sparse")
    cap_string = "+".join(parts) if parts else "none"
    return f"{cap_string} (count={queue_family_properties.queueCount})"


def _find_compute_queue_family(physical_device) -> Optional[int]:
    """Return the index of the first queue family supporting VK_QUEUE_COMPUTE_BIT,
    or None if none does."""
    queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for index, properties in enumerate(queue_families):
        if properties.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return index
    return None


def _select_physical_device(instance):
    """Enumerate all physical devices, print a summary line per device, return
    (physical_device, compute_queue_family_index) for the first one with a
    compute-capable queue family. Prints the selection at the end.

    Raises RuntimeError if no GPU exposes a compute queue.
    """
    physical_devices = vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        raise RuntimeError("vkEnumeratePhysicalDevices returned no devices")

    print(f"[VulkanContext] available physical devices ({len(physical_devices)}):")

    selected_index = None
    selected_queue_family = None

    for device_index, physical_device in enumerate(physical_devices):
        properties = vkGetPhysicalDeviceProperties(physical_device)
        device_type = _DEVICE_TYPE_NAMES.get(properties.deviceType, "UNKNOWN")
        api_version = (
            f"{VK_VERSION_MAJOR(properties.apiVersion)}."
            f"{VK_VERSION_MINOR(properties.apiVersion)}."
            f"{VK_VERSION_PATCH(properties.apiVersion)}")
        queue_family_index = _find_compute_queue_family(physical_device)

        print(f"  [{device_index}] {properties.deviceName}  "
              f"type={device_type}  api={api_version}")
        for qf_index, qf_props in enumerate(
                vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
            marker = "  <- compute pick" if qf_index == queue_family_index else ""
            print(f"        queue_family[{qf_index}]: "
                  f"{_summarize_queue_family(qf_props)}{marker}")

        if selected_index is None and queue_family_index is not None:
            selected_index = device_index
            selected_queue_family = queue_family_index

    if selected_index is None:
        raise RuntimeError(
            "no physical device exposes a queue family with VK_QUEUE_COMPUTE_BIT")

    selected = physical_devices[selected_index]
    selected_props = vkGetPhysicalDeviceProperties(selected)
    print(f"[VulkanContext] selected: [{selected_index}] {selected_props.deviceName} "
          f"on queue_family[{selected_queue_family}]")

    return selected, selected_queue_family


# ============================================================================
# Public API
# ============================================================================


@dataclass
class VulkanContext:
    """All handles needed for a compute-only Vulkan session.

    Use ``VulkanContext.create(...)`` to construct; ``destroy()`` (or the ``with``
    statement) tears down in reverse order. Cached memory properties enable
    fast ``find_memory_type`` lookups during buffer allocation.
    """
    instance: object                                # VkInstance
    physical_device: object                         # VkPhysicalDevice
    device: object                                  # VkDevice
    compute_queue: object                           # VkQueue
    compute_queue_family_index: int
    command_pool: object                            # VkCommandPool

    # Internals (kept on the dataclass so destroy() can clean up in order)
    _validation_enabled: bool = False
    _debug_messenger: Optional[object] = None       # VkDebugUtilsMessengerEXT
    _memory_properties: Optional[object] = field(default=None, repr=False)
    _destroyed: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        application_name: str = "sph_v0",
        enable_validation: bool = True,
        extra_instance_extensions: Optional[list] = None,
        extra_device_extensions: Optional[list] = None,
    ) -> "VulkanContext":
        # ---- 1. Decide validation availability --------------------------
        validation_supported = _check_validation_layer_support()
        if enable_validation and not validation_supported:
            print(
                f"[VulkanContext] WARNING: VK_LAYER_KHRONOS_validation requested "
                f"but not available; continuing without validation. Install the "
                f"Vulkan SDK validation layers to enable.",
                file=sys.stderr)
        validation_active = enable_validation and validation_supported

        layers = [VALIDATION_LAYER_NAME] if validation_active else []
        extensions = [DEBUG_UTILS_EXTENSION] if validation_active else []
        if extra_instance_extensions:
            extensions.extend(extra_instance_extensions)

        # ---- 2. Create VkInstance ---------------------------------------
        application_info = VkApplicationInfo(
            pApplicationName=application_name,
            applicationVersion=VK_MAKE_VERSION(0, 1, 0),
            pEngineName="sph_v0",
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

        # ---- 3. Debug messenger (optional) ------------------------------
        debug_messenger = None
        if validation_active:
            create_fn = vkGetInstanceProcAddr(
                instance, "vkCreateDebugUtilsMessengerEXT")
            if create_fn is not None:
                messenger_create_info = VkDebugUtilsMessengerCreateInfoEXT(
                    messageSeverity=(
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                    ),
                    messageType=(
                        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                    ),
                    pfnUserCallback=_debug_callback,
                )
                debug_messenger = create_fn(instance, messenger_create_info, None)

        # ---- 4. Pick a physical device + compute queue family -----------
        physical_device, compute_queue_family_index = _select_physical_device(instance)

        # ---- 5. Create logical device + grab the queue ------------------
        queue_create_info = VkDeviceQueueCreateInfo(
            queueFamilyIndex=compute_queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_extension_list = list(extra_device_extensions) if extra_device_extensions else []
        # Enable vertexPipelineStoresAndAtomics: Vulkan 1.0 core feature,
        # universally supported on desktop GPUs. Permits the vertex stage to
        # bind storage buffers without each declaration being marked
        # `readonly`. Our render vert shader (particle.vert) reads sim's
        # set 0 SSBOs via `#include "common.glsl"`, where the buffers are
        # declared read-write (compute stages need to write). Declaring them
        # `readonly` only inside the vert would require either macro tricks or
        # local redeclaration; enabling this feature is the cleaner path. The
        # shader still does no writes — the feature only relaxes validation.
        device_features = VkPhysicalDeviceFeatures(
            vertexPipelineStoresAndAtomics=VK_TRUE,
        )
        device_create_info = VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledLayerCount=0,
            ppEnabledLayerNames=[],
            enabledExtensionCount=len(device_extension_list),
            ppEnabledExtensionNames=device_extension_list,
            pEnabledFeatures=device_features,
        )
        device = vkCreateDevice(physical_device, device_create_info, None)
        compute_queue = vkGetDeviceQueue(device, compute_queue_family_index, 0)

        # ---- 6. Command pool on that queue family -----------------------
        command_pool_create_info = VkCommandPoolCreateInfo(
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=compute_queue_family_index,
        )
        command_pool = vkCreateCommandPool(device, command_pool_create_info, None)

        # ---- 7. Cache memory properties for find_memory_type ------------
        memory_properties = vkGetPhysicalDeviceMemoryProperties(physical_device)

        return cls(
            instance=instance,
            physical_device=physical_device,
            device=device,
            compute_queue=compute_queue,
            compute_queue_family_index=compute_queue_family_index,
            command_pool=command_pool,
            _validation_enabled=validation_active,
            _debug_messenger=debug_messenger,
            _memory_properties=memory_properties,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def find_memory_type(self, type_bits: int, required_properties: int) -> int:
        """Return the index of the first memory type compatible with ``type_bits``
        (a bitmask from VkMemoryRequirements.memoryTypeBits) and providing at
        least ``required_properties`` (e.g. VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        """
        for i in range(self._memory_properties.memoryTypeCount):
            type_bit = 1 << i
            if not (type_bits & type_bit):
                continue
            flags = self._memory_properties.memoryTypes[i].propertyFlags
            if (flags & required_properties) == required_properties:
                return i
        raise RuntimeError(
            f"no memory type satisfies type_bits=0x{type_bits:x} "
            f"required_properties=0x{required_properties:x}")

    def submit_and_wait(self, command_buffer) -> None:
        """Submit one command buffer to the compute queue, fence-wait, clean up.
        Used for one-shot work (initial uploads, bootstrap, readback)."""
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        fence_create_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vkCreateFence(self.device, fence_create_info, None)
        try:
            vkQueueSubmit(self.compute_queue, 1, submit_info, fence)
            # Timeout = u64::max → wait forever (V0 single sync; no overlap yet).
            vkWaitForFences(self.device, 1, [fence], VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        finally:
            vkDestroyFence(self.device, fence, None)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._destroyed:
            return
        # Idle wait so any pending work is finished before tearing handles down.
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

    def __enter__(self) -> "VulkanContext":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
