// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/layernorm2d.hpp"

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