#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#include "ck_tile/ops/gemm.hpp"

using grouped_gemm_kargs = ck_tile::GemmHostArgs</*NumDTensor = 0*/>;

namespace ck_tile_cpp {

namespace group_gemm {

struct GroupGemmArgs {
    std::vector<int> Ms;        // M dimensions
    std::vector<int> Ns;        // N dimensions  
    std::vector<int> Ks;        // K dimensions
    int group_count;            
    std::string dtype;          
    bool validate;              
    int warmup;                 
    int repeat;                 
};

std::vector<torch::Tensor> grouped_gemm_api(
    std::vector<torch::Tensor>& A_tensors,
    std::vector<torch::Tensor>& B_tensors,
    const GroupGemmArgs& args);

void register_group_gemm_apis(pybind11::module& m);

} // namespace group_gemm

} // namespace ck_tile
