#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#include "ck_tile/ops/gemm.hpp"

namespace ck_tile_cpp {

namespace gemm {

struct GemmArgs {
    int Ms;        // M dimensions
    int Ns;        // N dimensions  
    int Ks;        // K dimensions
    std::string dtype; 
    bool validate; 
    int warmup;               
    int repeat;
    bool persistent;            
};

torch::Tensor gemm_api(
    torch::Tensor& A_tensor,
    torch::Tensor& B_tensor,
    const GemmArgs& args);

void register_gemm_apis(pybind11::module& m);

} // namespace gemm

} // namespace ck_tile
