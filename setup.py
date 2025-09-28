import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

print(f"this_dir:{this_dir}")

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True


# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("CK_TILE_FORCE_CXX11_ABI", "FALSE") == "TRUE"
USE_TRITON_ROCM = os.getenv("CK_TILE_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
SKIP_CK_BUILD = os.getenv("CK_TILE_SKIP_CK_BUILD", "TRUE") == "TRUE" if USE_TRITON_ROCM else False

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))
    
def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )

def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")

def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

if IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_rocm_home_none("ck_tile")
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)

    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    if archs != ['native']:
        cc_flag = [f"--offload-arch={arch}" for arch in archs]
    else:
        arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
        cc_flag = [f"--offload-arch={arch}"]

    cc_flag += ["-O3", "-std=c++17",
                "-fgpu-flush-denormals-to-zero",
                "-DCK_ENABLE_BF16",
                "-DCK_ENABLE_BF8",
                "-DCK_ENABLE_FP16",
                "-DCK_ENABLE_FP32",
                "-DCK_ENABLE_FP64",
                "-DCK_ENABLE_FP8",
                "-DCK_ENABLE_INT8",
                "-DCK_USE_XDL",
                "-DUSE_PROF_API=1",
                "-D__HIP_PLATFORM_HCC__=1",
                "-Wno-unknown-pragmas",
                "-Wno-comment"]
    
    cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

    hip_version = get_hip_version()
    if hip_version > Version('5.5.00000'):
        cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
    if hip_version > Version('5.7.23302'):
        cc_flag += ["-fno-offload-uniform-block"]
    if hip_version > Version('6.1.40090'):
        cc_flag += ["-mllvm", "-enable-post-misched=0"]
    if hip_version > Version('6.2.41132'):
        cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                    "-mllvm", "-amdgpu-function-calls=false"]
    if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
        cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

    cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

    sources = [
        "csrc/python_api.cpp",
        "csrc/apis/group_gemm/group_gemm.cpp",
        "csrc/kernels/group_gemm_kernel/grouped_gemm_kernel.cpp",
        "csrc/apis/gemm/gemm.cpp",
        "csrc/kernels/gemm_kernel/gemm_kernel.cpp",
        "csrc/apis/batched_gemm/batched_gemm.cpp",
        "csrc/kernels/batched_gemm_kernel/batched_gemm_kernel.cpp",
        "csrc/apis/flatmm/flatmm.cpp",
        "csrc/kernels/flatmm_kernel/flatmm_kernel.cpp",
        "csrc/apis/layernorm2d/layernorm2d.cpp",
        "csrc/kernels/layernorm_2d/layernorm_2d_fwd_kernel.cpp"
    ]

    rename_cpp_to_cu(sources)

    renamed_sources = [
        "csrc/python_api.cu",
        "csrc/apis/group_gemm/group_gemm.cu",
        "csrc/kernels/group_gemm_kernel/grouped_gemm_kernel.cu",
        "csrc/apis/gemm/gemm.cu",
        "csrc/kernels/gemm_kernel/gemm_kernel.cu",
        "csrc/apis/batched_gemm/batched_gemm.cu",
        "csrc/kernels/batched_gemm_kernel/batched_gemm_kernel.cu",
        "csrc/apis/flatmm/flatmm.cu",
        "csrc/kernels/flatmm_kernel/flatmm_kernel.cu",
        "csrc/apis/layernorm2d/layernorm2d.cu",
        "csrc/kernels/layernorm_2d/layernorm_2d_fwd_kernel.cu"
    ]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17", "-Wno-unknown-pragmas","-Wno-comment"] + generator_flag,
        "nvcc": cc_flag + generator_flag
    }

    include_dirs = [
        Path(this_dir) / "ck_tile" / "include",
        Path(ROCM_HOME) / "include"
    ]

    ext_modules.append(
        CUDAExtension(
            name="ck_tile_python",
            sources=renamed_sources,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        )
    )

setup(
    name="ck_tile_python",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "pybind11","einops"],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],

)