import os
import pathlib
from vulkan import *


SHADER_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "shaders" / "spv")


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
