#include <pybind11/pybind11.h>
#include <torch/python.h>


#include "apis/group_gemm/group_gemm.hpp"
#include "apis/gemm/gemm.hpp"
#include "apis/batched_gemm/batched_gemm.hpp"
#include "apis/flatmm/flatmm.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME ck_tile_python
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CK_TILE C++ library";

    ck_tile_cpp::group_gemm::register_group_gemm_apis(m);
    ck_tile_cpp::gemm::register_gemm_apis(m);
    ck_tile_cpp::batched_gemm::register_batched_gemm_apis(m);
    ck_tile_cpp::flatmm::register_flatmm_apis(m);

}
