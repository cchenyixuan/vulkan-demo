import ctypes
import glfw
from vulkan import *
from vulkan._vulkancache import ffi


def create_instance():
    app_info = VkApplicationInfo(
        pApplicationName="Vulkan Triangle",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_API_VERSION_1_0,
    )
    extensions = glfw.get_required_instance_extensions()
    create_info = VkInstanceCreateInfo(
        pApplicationInfo=app_info,
        enabledExtensionCount=len(extensions),
        ppEnabledExtensionNames=extensions,
    )
    return vkCreateInstance(create_info, None)


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
    ]
    fns = {}
    for name in names:
        fns[name] = vkGetInstanceProcAddr(instance, name)
    return fns
