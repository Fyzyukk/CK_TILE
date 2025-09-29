#include "layernorm2d.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include "../../kernels/layernorm_2d/layernorm_2d_fwd_kernel.hpp"
#include <hip/hip_runtime.h>
#include <iostream>

extern "C" float layernorm2d_fwd_fp16_fp16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s);

extern "C" float layernorm2d_fwd_bf16_bf16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s);

extern "C" float layernorm2d_fwd_fp16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s);

extern "C" float layernorm2d_fwd_bf16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s);

namespace ck_tile_cpp {

namespace layernorm2d {

torch::Tensor layernorm2d_api(
    torch::Tensor& x_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args) {

    TORCH_CHECK(x_tensor.is_cuda(), "x_tensor must be on CUDA device");
    TORCH_CHECK(gamma_tensor.is_cuda(), "gamma_tensor must be on CUDA device");
    TORCH_CHECK(beta_tensor.is_cuda(), "beta_tensor must be on CUDA device");
    
    if (!x_tensor.is_contiguous()) {
        x_tensor = x_tensor.contiguous();
    }
    if (!gamma_tensor.is_contiguous()) {
        gamma_tensor = gamma_tensor.contiguous();
    }
    if (!beta_tensor.is_contiguous()) {
        beta_tensor = beta_tensor.contiguous();
    }

    const int m = args.m > 0 ? args.m : x_tensor.size(0);
    const int n = args.n > 0 ? args.n : x_tensor.size(1);
    
    TORCH_CHECK(x_tensor.size(0) == m && x_tensor.size(1) == n, 
                "x_tensor shape mismatch: expected (", m, ", ", n, "), got ", x_tensor.sizes());
    TORCH_CHECK(gamma_tensor.size(0) == n, 
                "gamma_tensor shape mismatch: expected (", n, "), got ", gamma_tensor.sizes());
    TORCH_CHECK(beta_tensor.size(0) == n, 
                "beta_tensor shape mismatch: expected (", n, "), got ", beta_tensor.sizes());

    torch::ScalarType output_dtype;
    std::string prec_o = args.prec_o;
    if (prec_o == "auto") {
        prec_o = args.prec_i;
    }
    
    if (prec_o == "fp16") {
        output_dtype = torch::kFloat16;
    } else if (prec_o == "bf16") {
        output_dtype = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    } else if (prec_o == "int8") {
        output_dtype = torch::kInt8;
    } else if (prec_o == "fp8") {
        output_dtype = static_cast<torch::ScalarType>(c10::ScalarType::Float8_e4m3fn);
    } else {
        TORCH_CHECK(false, "Unsupported output precision: ", prec_o);
    }

    auto y_tensor = torch::zeros({m, n}, 
                                 torch::TensorOptions()
                                 .dtype(output_dtype)
                                 .device(x_tensor.device())
                                 .memory_format(torch::MemoryFormat::Contiguous));

    int x_stride = n;  
    int y_stride = n; 

    ck_tile::Layernorm2dFwdHostArgs layernorm_args;
    layernorm_args.p_x = x_tensor.data_ptr();
    layernorm_args.p_x_residual = nullptr;  
    layernorm_args.p_sm_scale = nullptr;   
    layernorm_args.p_x_bias = nullptr;      
    layernorm_args.p_gamma = gamma_tensor.data_ptr();
    layernorm_args.p_beta = beta_tensor.data_ptr();
    layernorm_args.p_y = y_tensor.data_ptr();
    layernorm_args.p_y_residual = nullptr;  
    layernorm_args.p_y_scale = nullptr;     
    layernorm_args.p_mean = nullptr;        
    layernorm_args.p_inv_std = nullptr;     
    layernorm_args.epsilon = args.epsilon;
    layernorm_args.m = m;
    layernorm_args.n = n;
    layernorm_args.x_row_stride = x_stride;
    layernorm_args.x_residual_row_stride = x_stride;
    layernorm_args.y_row_stride = y_stride;
    layernorm_args.y_residual_row_stride = y_stride;

    ck_tile::stream_config s{nullptr, true, 1, args.warmup, args.repeat, true, true, 10};

    const auto kBF16 = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    const auto kFP8 = static_cast<torch::ScalarType>(c10::ScalarType::Float8_e4m3fn);

    if (x_tensor.dtype() == torch::kFloat16 && y_tensor.dtype() == torch::kFloat16) {
        layernorm2d_fwd_fp16_fp16(layernorm_args, s);
    } else if (x_tensor.dtype() == kBF16 && y_tensor.dtype() == kBF16) {
        layernorm2d_fwd_bf16_bf16(layernorm_args, s);
    } else if (x_tensor.dtype() == torch::kFloat16 && y_tensor.dtype() == torch::kInt8) {
        layernorm2d_fwd_fp16_int8(layernorm_args, s);
    } else if (x_tensor.dtype() == kBF16 && y_tensor.dtype() == torch::kInt8) {
        layernorm2d_fwd_bf16_int8(layernorm_args, s);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination for LayerNorm2D");
    }

    return y_tensor;
}

torch::Tensor layernorm2d_with_residual_api(
    torch::Tensor& x_tensor,
    torch::Tensor& x_residual_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args) {

    TORCH_CHECK(x_tensor.is_cuda(), "x_tensor must be on CUDA device");
    TORCH_CHECK(x_residual_tensor.is_cuda(), "x_residual_tensor must be on CUDA device");
    TORCH_CHECK(gamma_tensor.is_cuda(), "gamma_tensor must be on CUDA device");
    TORCH_CHECK(beta_tensor.is_cuda(), "beta_tensor must be on CUDA device");
    
    if (!x_tensor.is_contiguous()) {
        x_tensor = x_tensor.contiguous();
    }
    if (!x_residual_tensor.is_contiguous()) {
        x_residual_tensor = x_residual_tensor.contiguous();
    }
    if (!gamma_tensor.is_contiguous()) {
        gamma_tensor = gamma_tensor.contiguous();
    }
    if (!beta_tensor.is_contiguous()) {
        beta_tensor = beta_tensor.contiguous();
    }

    const int m = args.m > 0 ? args.m : x_tensor.size(0);
    const int n = args.n > 0 ? args.n : x_tensor.size(1);
    

    TORCH_CHECK(x_tensor.size(0) == m && x_tensor.size(1) == n, 
                "x_tensor shape mismatch: expected (", m, ", ", n, "), got ", x_tensor.sizes());
    TORCH_CHECK(x_residual_tensor.size(0) == m && x_residual_tensor.size(1) == n, 
                "x_residual_tensor shape mismatch: expected (", m, ", ", n, "), got ", x_residual_tensor.sizes());
    TORCH_CHECK(gamma_tensor.size(0) == n, 
                "gamma_tensor shape mismatch: expected (", n, "), got ", gamma_tensor.sizes());
    TORCH_CHECK(beta_tensor.size(0) == n, 
                "beta_tensor shape mismatch: expected (", n, "), got ", beta_tensor.sizes());

    torch::ScalarType output_dtype;
    std::string prec_o = args.prec_o;
    if (prec_o == "auto") {
        prec_o = args.prec_i;
    }
    
    if (prec_o == "fp16") {
        output_dtype = torch::kFloat16;
    } else if (prec_o == "bf16") {
        output_dtype = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    } else if (prec_o == "int8") {
        output_dtype = torch::kInt8;
    } else if (prec_o == "fp8") {
        output_dtype = static_cast<torch::ScalarType>(c10::ScalarType::Float8_e4m3fn);
    } else {
        TORCH_CHECK(false, "Unsupported output precision: ", prec_o);
    }

    auto y_tensor = torch::zeros({m, n}, 
                                 torch::TensorOptions()
                                 .dtype(output_dtype)
                                 .device(x_tensor.device())
                                 .memory_format(torch::MemoryFormat::Contiguous));

    torch::Tensor y_residual_tensor;
    if (args.fused_add == 1) {
        y_residual_tensor = torch::zeros({m, n}, 
                                         torch::TensorOptions()
                                         .dtype(x_tensor.dtype())
                                         .device(x_tensor.device())
                                         .memory_format(torch::MemoryFormat::Contiguous));
    }

    int x_stride = n;
    int xr_stride = n;
    int y_stride = n;
    int yr_stride = n;

    ck_tile::Layernorm2dFwdHostArgs layernorm_args;
    layernorm_args.p_x = x_tensor.data_ptr();
    layernorm_args.p_x_residual = x_residual_tensor.data_ptr();
    layernorm_args.p_sm_scale = nullptr;
    layernorm_args.p_x_bias = nullptr;
    layernorm_args.p_gamma = gamma_tensor.data_ptr();
    layernorm_args.p_beta = beta_tensor.data_ptr();
    layernorm_args.p_y = y_tensor.data_ptr();
    layernorm_args.p_y_residual = (args.fused_add == 1) ? y_residual_tensor.data_ptr() : nullptr;
    layernorm_args.p_y_scale = nullptr;
    layernorm_args.p_mean = nullptr;
    layernorm_args.p_inv_std = nullptr;
    layernorm_args.epsilon = args.epsilon;
    layernorm_args.m = m;
    layernorm_args.n = n;
    layernorm_args.x_row_stride = x_stride;
    layernorm_args.x_residual_row_stride = xr_stride;
    layernorm_args.y_row_stride = y_stride;
    layernorm_args.y_residual_row_stride = yr_stride;

    ck_tile::stream_config s{nullptr, true, 1, args.warmup, args.repeat, true, true, 10};

    const auto kBF16 = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);

    if (x_tensor.dtype() == torch::kFloat16 && y_tensor.dtype() == torch::kFloat16) {
        layernorm2d_fwd_fp16_fp16(layernorm_args, s);
    } else if (x_tensor.dtype() == kBF16 && y_tensor.dtype() == kBF16) {
        layernorm2d_fwd_bf16_bf16(layernorm_args, s);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination for LayerNorm2D with residual");
    }

    return y_tensor;
}

torch::Tensor layernorm2d_with_bias_api(
    torch::Tensor& x_tensor,
    torch::Tensor& x_bias_tensor,
    torch::Tensor& gamma_tensor,
    torch::Tensor& beta_tensor,
    const Layernorm2dArgs& args) {

    TORCH_CHECK(x_tensor.is_cuda(), "x_tensor must be on CUDA device");
    TORCH_CHECK(x_bias_tensor.is_cuda(), "x_bias_tensor must be on CUDA device");
    TORCH_CHECK(gamma_tensor.is_cuda(), "gamma_tensor must be on CUDA device");
    TORCH_CHECK(beta_tensor.is_cuda(), "beta_tensor must be on CUDA device");
    
    if (!x_tensor.is_contiguous()) {
        x_tensor = x_tensor.contiguous();
    }
    if (!x_bias_tensor.is_contiguous()) {
        x_bias_tensor = x_bias_tensor.contiguous();
    }
    if (!gamma_tensor.is_contiguous()) {
        gamma_tensor = gamma_tensor.contiguous();
    }
    if (!beta_tensor.is_contiguous()) {
        beta_tensor = beta_tensor.contiguous();
    }

    const int m = args.m > 0 ? args.m : x_tensor.size(0);
    const int n = args.n > 0 ? args.n : x_tensor.size(1);
    
    TORCH_CHECK(x_tensor.size(0) == m && x_tensor.size(1) == n, 
                "x_tensor shape mismatch: expected (", m, ", ", n, "), got ", x_tensor.sizes());
    TORCH_CHECK(x_bias_tensor.size(0) == n, 
                "x_bias_tensor shape mismatch: expected (", n, "), got ", x_bias_tensor.sizes());
    TORCH_CHECK(gamma_tensor.size(0) == n, 
                "gamma_tensor shape mismatch: expected (", n, "), got ", gamma_tensor.sizes());
    TORCH_CHECK(beta_tensor.size(0) == n, 
                "beta_tensor shape mismatch: expected (", n, "), got ", beta_tensor.sizes());

    torch::ScalarType output_dtype;
    std::string prec_o = args.prec_o;
    if (prec_o == "auto") {
        prec_o = args.prec_i;
    }
    
    if (prec_o == "fp16") {
        output_dtype = torch::kFloat16;
    } else if (prec_o == "bf16") {
        output_dtype = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    } else {
        TORCH_CHECK(false, "Unsupported output precision: ", prec_o);
    }

    auto y_tensor = torch::zeros({m, n}, 
                                 torch::TensorOptions()
                                 .dtype(output_dtype)
                                 .device(x_tensor.device())
                                 .memory_format(torch::MemoryFormat::Contiguous));

    int x_stride = n;

    ck_tile::Layernorm2dFwdHostArgs layernorm_args;
    layernorm_args.p_x = x_tensor.data_ptr();
    layernorm_args.p_x_residual = nullptr;
    layernorm_args.p_sm_scale = nullptr;
    layernorm_args.p_x_bias = x_bias_tensor.data_ptr();
    layernorm_args.p_gamma = gamma_tensor.data_ptr();
    layernorm_args.p_beta = beta_tensor.data_ptr();
    layernorm_args.p_y = y_tensor.data_ptr();
    layernorm_args.p_y_residual = nullptr;
    layernorm_args.p_y_scale = nullptr;
    layernorm_args.p_mean = nullptr;
    layernorm_args.p_inv_std = nullptr;
    layernorm_args.epsilon = args.epsilon;
    layernorm_args.m = m;
    layernorm_args.n = n;
    layernorm_args.x_row_stride = x_stride;
    layernorm_args.x_residual_row_stride = x_stride;
    layernorm_args.y_row_stride = x_stride;
    layernorm_args.y_residual_row_stride = x_stride;

    ck_tile::stream_config s{nullptr, true, 1, args.warmup, args.repeat, true, true, 10};
    const auto kBF16 = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);

    if (x_tensor.dtype() == torch::kFloat16 && y_tensor.dtype() == torch::kFloat16) {
        layernorm2d_fwd_fp16_fp16(layernorm_args, s);
    } else if (x_tensor.dtype() == kBF16 && y_tensor.dtype() == kBF16) {
        layernorm2d_fwd_bf16_bf16(layernorm_args, s);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination for LayerNorm2D with bias");
    }

    return y_tensor;
}

void register_layernorm2d_apis(pybind11::module& m) {
    pybind11::class_<Layernorm2dArgs>(m, "Layernorm2dArgs")
        .def(pybind11::init<>())
        .def_readwrite("m", &Layernorm2dArgs::m)
        .def_readwrite("n", &Layernorm2dArgs::n)
        .def_readwrite("prec_i", &Layernorm2dArgs::prec_i)
        .def_readwrite("prec_o", &Layernorm2dArgs::prec_o)
        .def_readwrite("prec_sm", &Layernorm2dArgs::prec_sm)
        .def_readwrite("prec_sy", &Layernorm2dArgs::prec_sy)
        .def_readwrite("save_mean_var", &Layernorm2dArgs::save_mean_var)
        .def_readwrite("xbias", &Layernorm2dArgs::xbias)
        .def_readwrite("fused_add", &Layernorm2dArgs::fused_add)
        .def_readwrite("fused_quant", &Layernorm2dArgs::fused_quant)
        .def_readwrite("epsilon", &Layernorm2dArgs::epsilon)
        .def_readwrite("warmup", &Layernorm2dArgs::warmup)
        .def_readwrite("repeat", &Layernorm2dArgs::repeat)
        .def_readwrite("validate", &Layernorm2dArgs::validate);

    m.def("layernorm2d_api", &layernorm2d_api, 
          "Perform LayerNorm2D operations",
          pybind11::arg("x_tensor"), pybind11::arg("gamma_tensor"), 
          pybind11::arg("beta_tensor"), pybind11::arg("args"));
    
    m.def("layernorm2d_with_residual_api", &layernorm2d_with_residual_api, 
          "Perform LayerNorm2D operations with residual connection",
          pybind11::arg("x_tensor"), pybind11::arg("x_residual_tensor"), 
          pybind11::arg("gamma_tensor"), pybind11::arg("beta_tensor"), pybind11::arg("args"));
    
    m.def("layernorm2d_with_bias_api", &layernorm2d_with_bias_api, 
          "Perform LayerNorm2D operations with bias",
          pybind11::arg("x_tensor"), pybind11::arg("x_bias_tensor"), 
          pybind11::arg("gamma_tensor"), pybind11::arg("beta_tensor"), pybind11::arg("args"));
}

} // namespace layernorm2d

} // namespace ck_tile_cpp
