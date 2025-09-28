#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/flatmm.hpp"

namespace ck_tile_cpp {

namespace flatmm {

struct FlatmmArgs {
    int Ms;        // M dimensions
    int Ns;        // N dimensions  
    int Ks;        // K dimensions
    std::string dtype; 
    bool validate; 
    int warmup;               
    int repeat;           
};

torch::Tensor flatmm_api(
    torch::Tensor& A_tensor,
    torch::Tensor& B_tensor,
    const FlatmmArgs& args);

void register_flatmm_apis(pybind11::module& m);

} // namespace gemm

} // namespace ck_tile
