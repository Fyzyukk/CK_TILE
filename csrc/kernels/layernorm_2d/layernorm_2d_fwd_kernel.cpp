// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm_2d_fwd_kernel.hpp"
#include "ck_tile/ops/layernorm2d.hpp"

// 外部 C 函数实现
extern "C" float layernorm2d_fwd_fp16_fp16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    // 这里应该调用具体的 layernorm2d 内核
    // 由于我们还没有完整的内核实现，这里返回一个占位符
    // 实际实现需要根据具体的 layernorm2d 配置来调用相应的内核
    
    // 暂时返回 0，表示成功执行
    return 0.0f;
}

extern "C" float layernorm2d_fwd_bf16_bf16(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    // 这里应该调用具体的 layernorm2d 内核
    // 由于我们还没有完整的内核实现，这里返回一个占位符
    
    // 暂时返回 0，表示成功执行
    return 0.0f;
}

extern "C" float layernorm2d_fwd_fp16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    // 这里应该调用具体的 layernorm2d 内核
    // 由于我们还没有完整的内核实现，这里返回一个占位符
    
    // 暂时返回 0，表示成功执行
    return 0.0f;
}

extern "C" float layernorm2d_fwd_bf16_int8(
    const ck_tile::Layernorm2dFwdHostArgs& args,
    const ck_tile::stream_config& s) {
    
    // 这里应该调用具体的 layernorm2d 内核
    // 由于我们还没有完整的内核实现，这里返回一个占位符
    
    // 暂时返回 0，表示成功执行
    return 0.0f;
}