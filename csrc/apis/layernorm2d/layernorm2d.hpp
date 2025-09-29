#pragma once

#include <torch/extension.h>
#include <string>

namespace ck_tile_cpp {

namespace layernorm2d {

struct Layernorm2dArgs {
    int m = 0;                    // m dimension
    int n = 0;                    // n dimension
    std::string prec_i = "fp16";  // input precision
    std::string prec_o = "auto";  // output precision
    std::string prec_sm = "auto"; // x-scale precision
    std::string prec_sy = "auto"; // y-scale precision
    bool save_mean_var = false;   // save mean/variance
    int xbias = 0;                // add bias: 0=no, 1=add
    int fused_add = 0;            // fused-add: 0=no, 1=preadd+store, 2=preadd only
    int fused_quant = 0;          // fused-quant: 0=no, 1=smooth-dynamic-quant, 2=dynamic-quant
    float epsilon = 1e-5f;        // epsilon value
    int warmup = 5;               // warmup iterations
    int repeat = 20;              // repeat iterations
    bool validate = true;         // enable validation
};

torch::Tensor layernorm2d_api(
    torch::Tensor& x_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args);

torch::Tensor layernorm2d_with_residual_api(
    torch::Tensor& x_tensor,
    torch::Tensor& x_residual_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args);

torch::Tensor layernorm2d_with_bias_api(
    torch::Tensor& x_tensor,
    torch::Tensor& x_bias_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args);

void register_layernorm2d_apis(pybind11::module& m);

} // namespace layernorm2d

} // namespace ck_tile_cpp
