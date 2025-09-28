import os
import subprocess

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
import ck_tile_cpp
from ck_tile_cpp import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
)

# GEMM APIs
from ck_tile_cpp import (
    # Grouped GEMM
    grouped_gemm,
    create_test_data,
    benchmark_grouped_gemm,
    validate_grouped_gemm,
    GroupedGemmArgs,
)

# Layout APIs
from ck_tile_cpp import (
    transform_sf_into_required_layout,
    get_default_strides,
    is_valid_layout,
)

# Runtime APIs
from ck_tile_cpp import (
    get_gpu_info,
    check_hip_error,
    sync_gpu,
)

# Python modules
from . import grouped_gemm as py_grouped_gemm
from .grouped_gemm import (
    create_simple_grouped_gemm_args,
)

# Some utils
from . import utils

# Initialize CPP modules
def _find_hip_home() -> str:
    hip_home = os.environ.get('HIP_HOME') or os.environ.get('ROCM_HOME')
    if hip_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                hipcc = subprocess.check_output(['which', 'hipcc'], stderr=devnull).decode().rstrip('\r\n')
                hip_home = os.path.dirname(os.path.dirname(hipcc))
        except Exception:
            hip_home = '/opt/rocm'
            if not os.path.exists(hip_home):
                hip_home = None
    assert hip_home is not None
    return hip_home

# Initialize the C++ module
ck_tile_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_hip_home()                           # HIP home
)

# Version info
__version__ = "1.0.0"

# Export main APIs
__all__ = [
    # C++ APIs
    'set_num_sms',
    'get_num_sms', 
    'set_tc_util',
    'get_tc_util',
    'grouped_gemm',
    'create_test_data',
    'benchmark_grouped_gemm',
    'validate_grouped_gemm',
    'GroupedGemmArgs',
    'transform_sf_into_required_layout',
    'get_default_strides',
    'is_valid_layout',
    'get_gpu_info',
    'check_hip_error',
    'sync_gpu',
    
    # Python APIs
    'create_simple_grouped_gemm_args',
]
