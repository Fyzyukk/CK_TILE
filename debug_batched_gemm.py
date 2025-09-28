#!/usr/bin/env python3
"""
调试 Batched GEMM 的步长和布局问题
"""

import torch
import ck_tile_python
import numpy as np

def debug_stride_and_layout():
    """调试步长和布局问题"""
    print("=" * 60)
    print("调试 Batched GEMM 步长和布局问题")
    print("=" * 60)
    
    # 创建小规模的测试数据
    batch_count, M, N, K = 2, 4, 4, 4
    dtype = torch.float16
    
    print(f"测试规格: batch_count={batch_count}, M={M}, N={N}, K={K}")
    
    # 创建简单的测试数据
    A = torch.arange(batch_count * M * K, dtype=dtype, device='cuda').reshape(batch_count, M, K)
    B = torch.arange(batch_count * K * N, dtype=dtype, device='cuda').reshape(batch_count, K, N)
    
    print(f"\nA tensor:")
    print(f"  shape: {A.shape}")
    print(f"  stride: {A.stride()}")
    print(f"  data: {A.cpu().numpy()}")
    
    print(f"\nB tensor:")
    print(f"  shape: {B.shape}")
    print(f"  stride: {B.stride()}")
    print(f"  data: {B.cpu().numpy()}")
    
    # 使用 PyTorch 计算参考结果
    expected = torch.bmm(A, B)
    print(f"\nPyTorch 参考结果:")
    print(f"  shape: {expected.shape}")
    print(f"  data: {expected.cpu().numpy()}")
    
    # 创建参数
    args = ck_tile_python.BatchedGemmArgs()
    args.Ms = M
    args.Ns = N
    args.Ks = K
    args.batched_count = batch_count
    args.dtype = "fp16"
    args.validate = False
    args.warmup = 1
    args.repeat = 1
    
    # 调用我们的实现
    try:
        result = ck_tile_python.batched_gemm_api(A, B, args)
        print(f"\nck_tile_python 结果:")
        print(f"  shape: {result.shape}")
        print(f"  data: {result.cpu().numpy()}")
        
        # 比较结果
        diff = torch.abs(result - expected)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n差异分析:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("  ✅ 结果正确")
        else:
            print("  ❌ 结果不正确")
            print(f"  详细差异: {diff.cpu().numpy()}")
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

def debug_stride_calculation():
    """调试步长计算"""
    print("=" * 60)
    print("调试步长计算")
    print("=" * 60)
    
    M, N, K = 4, 4, 4
    
    # 模拟 ck_tile::get_default_stride 的计算
    # Row major: stride = col
    # Column major: stride = row
    
    stride_A_row = K  # Row major: M x K, stride = K
    stride_B_col = K  # Column major: K x N, stride = K  
    stride_C_row = N  # Row major: M x N, stride = N
    
    print(f"步长计算:")
    print(f"  A (Row Major, M={M} x K={K}): stride = {stride_A_row}")
    print(f"  B (Column Major, K={K} x N={N}): stride = {stride_B_col}")
    print(f"  C (Row Major, M={M} x N={N}): stride = {stride_C_row}")
    
    # 批次步长
    batch_stride_A = M * K
    batch_stride_B = K * N
    batch_stride_C = M * N
    
    print(f"\n批次步长:")
    print(f"  batch_stride_A = M * K = {M} * {K} = {batch_stride_A}")
    print(f"  batch_stride_B = K * N = {K} * {N} = {batch_stride_B}")
    print(f"  batch_stride_C = M * N = {M} * {N} = {batch_stride_C}")

def debug_tensor_layout():
    """调试张量布局"""
    print("=" * 60)
    print("调试张量布局")
    print("=" * 60)
    
    # 创建测试张量
    A = torch.randn(2, 4, 4, dtype=torch.float16, device='cuda')
    B = torch.randn(2, 4, 4, dtype=torch.float16, device='cuda')
    
    print(f"A tensor (Row Major):")
    print(f"  shape: {A.shape}")
    print(f"  stride: {A.stride()}")
    print(f"  is_contiguous: {A.is_contiguous()}")
    
    # 转置 B 矩阵为 Column Major
    B_col = B.transpose(-2, -1).contiguous()
    print(f"\nB tensor (Column Major):")
    print(f"  shape: {B_col.shape}")
    print(f"  stride: {B_col.stride()}")
    print(f"  is_contiguous: {B_col.is_contiguous()}")
    
    # 比较结果
    result1 = torch.bmm(A, B)  # 原始 B
    result2 = torch.bmm(A, B_col)  # 转置后的 B
    
    print(f"\n结果比较:")
    print(f"  torch.bmm(A, B): {result1.cpu().numpy()}")
    print(f"  torch.bmm(A, B_col): {result2.cpu().numpy()}")
    
    diff = torch.abs(result1 - result2)
    max_diff = torch.max(diff).item()
    print(f"  最大差异: {max_diff:.6f}")

def main():
    """主函数"""
    print("🔍 开始调试 Batched GEMM")
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    # 调试步长计算
    debug_stride_calculation()
    
    # 调试张量布局
    debug_tensor_layout()
    
    # 调试完整流程
    debug_stride_and_layout()

if __name__ == "__main__":
    main()
