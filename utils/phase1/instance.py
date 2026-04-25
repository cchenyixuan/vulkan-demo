import ctypes
import sys
import glfw
from vulkan import *
from vulkan._vulkancache import ffi

ENABLE_VALIDATION = True
VALIDATION_LAYERS = ["VK_LAYER_KHRONOS_validation"]


def _check_validation_layer_support():
    available = vkEnumerateInstanceLayerProperties()
    available_names = {layer.layerName for layer in available}
    for layer in VALIDATION_LAYERS:
        if layer not in available_names:
            return False
    return True


@ffi.callback(
    "VkBool32(VkDebugUtilsMessageSeverityFlagBitsEXT, "
    "VkDebugUtilsMessageTypeFlagsEXT, "
    "VkDebugUtilsMessengerCallbackDataEXT *, "
    "void *)"
)
def _debug_callback(severity, msg_type, callback_data, user_data):
    if severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        label = "\033[91mERROR\033[0m"
    elif severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        label = "\033[93mWARNING\033[0m"
    else:
        label = "\033[94mINFO\033[0m"
    message = ffi.string(callback_data.pMessage).decode("utf-8")
    print(f"[Vulkan {label}] {message}", file=sys.stderr)
    return VK_FALSE


def create_instance():
    app_info = VkApplicationInfo(
        pApplicationName="Vulkan Triangle",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_API_VERSION_1_0,
    )
    extensions = list(glfw.get_required_instance_extensions())
    layers = []

    use_validation = ENABLE_VALIDATION and _check_validation_layer_support()
    if use_validation:
        layers = VALIDATION_LAYERS
        extensions.append(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)

    create_info = VkInstanceCreateInfo(
        pApplicationInfo=app_info,
        enabledExtensionCount=len(extensions),
        ppEnabledExtensionNames=extensions,
        enabledLayerCount=len(layers),
        ppEnabledLayerNames=layers,
    )
    return vkCreateInstance(create_info, None)


def create_debug_messenger(instance):
    func = vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
    if func is None:
        return None
    create_info = VkDebugUtilsMessengerCreateInfoEXT(
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
    return func(instance, create_info, None)


def create_surface(instance, window):
    surface_ptr = ctypes.c_void_p()
    result = glfw.create_window_surface(instance, window, None, ctypes.pointer(surface_ptr))
    if result != 0:
        raise RuntimeError(f"Failed to create window surface: {result}")
    return ffi.cast("VkSurfaceKHR", surface_ptr.value)


def load_instance_functions(instance):
    names = [
        "vkGetPhysicalDeviceSurfaceSupportKHR",
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        "vkGetPhysicalDeviceSurfaceFormatsKHR",
        "vkGetPhysicalDeviceSurfacePresentModesKHR",
        "vkDestroySurfaceKHR",
        "vkDestroyDebugUtilsMessengerEXT",
    ]
    fns = {}
    for name in names:
        try:
            fns[name] = vkGetInstanceProcAddr(instance, name)
        except Exception:
            pass
    return fns
