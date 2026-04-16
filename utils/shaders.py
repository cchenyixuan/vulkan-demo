import os
from vulkan import *


SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders")


def load_spirv(filename):
    path = os.path.join(SHADER_DIR, filename)
    with open(path, "rb") as f:
        return f.read()


def create_shader_module(device, spirv_code):
    create_info = VkShaderModuleCreateInfo(
        codeSize=len(spirv_code),
        pCode=spirv_code,
    )
    return vkCreateShaderModule(device, create_info, None)
