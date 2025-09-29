#include "layernorm_2d_fwd_kernel.hpp"
#include "ck_tile/ops/layernorm2d.hpp"

extern "C" float layernorm2d_fwd_fp16_fp16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    return 0.0f;
}

extern "C" float layernorm2d_fwd_bf16_bf16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    return 0.0f;
}

extern "C" float layernorm2d_fwd_fp16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {

    return 0.0f;
}

extern "C" float layernorm2d_fwd_bf16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    return 0.0f;
}