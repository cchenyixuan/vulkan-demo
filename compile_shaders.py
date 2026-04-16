import subprocess
import sys
import os

GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"
SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

SHADERS = [
    "triangle.vert",
    "triangle.frag",
]


def compile_shaders():
    for name in SHADERS:
        src = os.path.join(SHADER_DIR, name)
        out = src + ".spv"
        print(f"Compiling {name} -> {name}.spv")
        result = subprocess.run([GLSLC, src, "-o", out], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR compiling {name}:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"  OK")
    print("All shaders compiled successfully.")


if __name__ == "__main__":
    compile_shaders()
