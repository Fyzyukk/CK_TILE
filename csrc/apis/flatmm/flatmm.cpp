#include "flatmm.hpp"
#include <hip/hip_runtime.h>
#include <iostream>

extern "C" float flatmm_fp16_fp16_fp16(
    const ck_tile::FlatmmHostArgs<>& args,
    const ck_tile::stream_config& s);

extern "C" float flatmm_bf16_bf16_bf16(
    const ck_tile::FlatmmHostArgs<>& args,
    const ck_tile::stream_config& s);

extern "C" float flatmm_fp8_fp8_fp16(
    const ck_tile::FlatmmHostArgs<>& args,
    const ck_tile::stream_config& s);

extern "C" float flatmm_bf8_bf8_fp16(
    const ck_tile::FlatmmHostArgs<>& args,
    const ck_tile::stream_config& s);

namespace ck_tile_cpp {

namespace flatmm {

torch::Tensor pre_shuffle_b_tensor(torch::Tensor& B_tensor, const std::string& dtype) {

    int N_Warp_Tile, K_Warp_Tile;
    
    if (dtype == "fp16") {

        N_Warp_Tile = 32;
        K_Warp_Tile = 16;  
    } else if (dtype == "bf16") {
        N_Warp_Tile = 32;
        K_Warp_Tile = 16;  
    } else {
        N_Warp_Tile = 16;
        K_Warp_Tile = 32;  
    }
    
    int K = B_tensor.size(0); 
    int N = B_tensor.size(1);
    
    int divisor = (N_Warp_Tile == 32) ? 2 : 4;
    
    if (N % N_Warp_Tile != 0 || K % K_Warp_Tile != 0) {
        return B_tensor.t().contiguous();
    }
    
    int n_tiles = N / N_Warp_Tile;
    int k_tiles = K / K_Warp_Tile;
    int k_tile_div = K_Warp_Tile / divisor;
    
    auto B_reshaped = B_tensor.view({n_tiles, N_Warp_Tile, k_tiles, divisor, k_tile_div});
    
    auto B_permuted = B_reshaped.permute({0, 2, 3, 1, 4});
    
    auto B_shuffled = B_permuted.contiguous().view({K, N});
    
    return B_shuffled;
}

torch::Tensor flatmm_api(
    torch::Tensor& A_tensor,
    torch::Tensor& B_tensor,
    const FlatmmArgs& args) {

    TORCH_CHECK(A_tensor.is_cuda(), "A tensors must be on device");
    if (!A_tensor.is_contiguous()) {
            A_tensor = A_tensor.contiguous();
    }

    TORCH_CHECK(B_tensor.is_cuda(), "B tensors must be on device");
    
    if (B_tensor.stride(0) == 1 && B_tensor.stride(1) == B_tensor.size(0)) {
        B_tensor = pre_shuffle_b_tensor(B_tensor, args.dtype);
    } else {
        B_tensor = B_tensor.t().contiguous();
        B_tensor = pre_shuffle_b_tensor(B_tensor, args.dtype);
    }

    const int M = args.Ms;
    const int N = args.Ns;
    const int K = args.Ks;

    int stride_A = ck_tile::get_default_stride(M, K, 0, ck_tile::bool_constant<true>{});
    int stride_B = ck_tile::get_default_stride(K, N, 0, ck_tile::bool_constant<false>{}); 
    int stride_C = ck_tile::get_default_stride(M, N, 0, ck_tile::bool_constant<true>{});

    int n_warmup = args.warmup;
    int n_repeat = args.repeat;
    
    torch::ScalarType c_dtype;
    if (args.dtype == "fp16") {
        c_dtype = torch::kFloat16;
    } 
    else if (args.dtype == "bf16") 
    {
        c_dtype = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    }

    auto C_tensor = torch::zeros({M, N}, 
                                torch::TensorOptions()
                                 .dtype(c_dtype)
                                 .device(A_tensor.device())
                                 .memory_format(torch::MemoryFormat::Contiguous));
    
    ck_tile::FlatmmHostArgs<> gemm_desc;

    gemm_desc.a_ptr = A_tensor.data_ptr();
    gemm_desc.b_ptr = B_tensor.data_ptr();
    gemm_desc.e_ptr = C_tensor.data_ptr();
    gemm_desc.M = M;
    gemm_desc.N = N;
    gemm_desc.K = K;
    gemm_desc.stride_A = stride_A;
    gemm_desc.stride_B = stride_B;
    gemm_desc.stride_E = stride_C;
    gemm_desc.k_batch = 1;
    
    ck_tile::stream_config s{nullptr, true, 1, args.warmup, args.repeat, true, true, 10};

    const auto kBF16 = static_cast<torch::ScalarType>(c10::ScalarType::BFloat16);
    const auto kFP8 = static_cast<torch::ScalarType>(c10::ScalarType::Float8_e4m3fn);
    const auto kBF8 = static_cast<torch::ScalarType>(c10::ScalarType::Float8_e5m2);

    if (A_tensor.dtype() == torch::kFloat16 && B_tensor.dtype() == torch::kFloat16 && C_tensor.dtype() == torch::kFloat16) {
        flatmm_fp16_fp16_fp16(gemm_desc, s);
    } 
    else if (A_tensor.dtype() == kBF16 && B_tensor.dtype() == kBF16 && C_tensor.dtype() == kBF16) {
        flatmm_bf16_bf16_bf16(gemm_desc, s);
    }
    else if (A_tensor.dtype() == kFP8 && B_tensor.dtype() == kFP8 && C_tensor.dtype() == torch::kFloat16) {
        flatmm_fp8_fp8_fp16(gemm_desc, s);
    }
    else if (A_tensor.dtype() == kBF8 && B_tensor.dtype() == kBF8 && C_tensor.dtype() == torch::kFloat16) {
        flatmm_bf8_bf8_fp16(gemm_desc, s);
    }
    else{
        TORCH_CHECK(false, "Unsupported dtype combination for GEMM");
    }

    return C_tensor;
    
}

void register_flatmm_apis(pybind11::module& m) {
    pybind11::class_<FlatmmArgs>(m, "FlatmmArgs")
        .def(pybind11::init<>())
        .def_readwrite("Ms", &FlatmmArgs::Ms)
        .def_readwrite("Ns", &FlatmmArgs::Ns)
        .def_readwrite("Ks", &FlatmmArgs::Ks)
        .def_readwrite("dtype", &FlatmmArgs::dtype)
        .def_readwrite("validate", &FlatmmArgs::validate)
        .def_readwrite("warmup", &FlatmmArgs::warmup)
        .def_readwrite("repeat", &FlatmmArgs::repeat);
    

    m.def("flatmm_api", &flatmm_api, 
          "Perform GEMM operations with interface",
          pybind11::arg("A_tensor"), pybind11::arg("B_tensor"), pybind11::arg("args"));
    
}

} // namespace flatmm

} // namespace ck_tile_cpp
