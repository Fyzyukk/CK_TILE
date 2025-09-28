#!/usr/bin/env python3
"""
测试 ck_tile_python 模块的 GEMM 测试脚本
"""

import torch
import ck_tile_python
import time

def test_module_import():
    """测试模块导入"""
    print("=" * 50)
    print("测试模块导入...")
    try:
        print(f"成功导入 ck_tile_python 模块")
        print(f"可用函数: {dir(ck_tile_python)}")
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False

def test_gemm_args():
    """测试 GemmArgs 结构"""
    print("=" * 50)
    print("测试 GemmArgs 结构...")
    try:
        # 创建参数
        args = ck_tile_python.GemmArgs()
        args.Ms = 2048
        args.Ns = 2048
        args.Ks = 2048
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 2
        args.repeat = 5
        args.persistent = False
        
        print(f"参数创建成功:")
        print(f"  Ms: {args.Ms}")
        print(f"  Ns: {args.Ns}")
        print(f"  Ks: {args.Ks}")
        print(f"  dtype: {args.dtype}")
        print(f"  persistent: {args.persistent}")
        return args
    except Exception as e:
        print(f"参数创建失败: {e}")
        return None

def create_test_tensors(args):
    """创建测试张量"""
    print("=" * 50)
    print("创建测试张量...")
    try:
        # 确定数据类型
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        
        M, N, K = args.Ms, args.Ns, args.Ks
        
        # 创建随机测试数据
        A = torch.randn(M, K, dtype=dtype, device='cuda')
        B = torch.randn(K, N, dtype=dtype, device='cuda')
        
        print(f"矩阵尺寸: A({M}x{K}), B({K}x{N}) -> C({M}x{N})")
        print(f"A tensor: shape={A.shape}, stride={A.stride()}")
        print(f"B tensor: shape={B.shape}, stride={B.stride()}")
        
        return A, B
    except Exception as e:
        print(f"张量创建失败: {e}")
        return None, None

def test_gemm_api(A_tensor, B_tensor, args):
    """测试 gemm_api 函数"""
    print("=" * 50)
    print("测试 gemm_api 函数...")
    try:
        # 调用我们的实现
        start_time = time.time()
        C_tensor = ck_tile_python.gemm_api(A_tensor, B_tensor, args)
        end_time = time.time()
        
        print(f"执行成功！耗时: {end_time - start_time:.4f} 秒")
        print(f"返回结果张量形状: {C_tensor.shape}")
        
        # 检查结果形状
        expected_M, expected_N = args.Ms, args.Ns
        actual_M, actual_N = C_tensor.shape
        print(f"期望形状 ({expected_M}, {expected_N}), 实际形状 ({actual_M}, {actual_N})")
        
        if (actual_M, actual_N) != (expected_M, expected_N):
            print(f"❌ 形状不匹配！")
            return None
        else:
            print(f"✅ 形状正确")
        
        return C_tensor
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensor, B_tensor, C_tensor, args):
    """验证结果正确性"""
    print("=" * 50)
    print("验证结果正确性...")
    try:
        # 使用 PyTorch 计算期望结果
        expected = torch.mm(A_tensor, B_tensor)
        actual = C_tensor
        
        # 比较结果
        is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
        
        if is_close:
            print("✅ 结果正确")
        else:
            print("❌ 结果不正确")
            # 计算差异
            diff = torch.abs(actual - expected)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            return False
        
        return True
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def benchmark_comparison(A_tensor, B_tensor, args):
    """性能对比测试"""
    print("=" * 50)
    print("性能对比测试...")
    try:
        # 测试我们的实现
        print("测试 ck_tile_python 实现...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            C_tensor = ck_tile_python.gemm_api(A_tensor, B_tensor, args)
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        # 测试 PyTorch 实现
        print("测试 PyTorch 实现...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            pytorch_result = torch.mm(A_tensor, B_tensor)
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        # 计算性能指标
        total_flops = 2.0 * args.Ms * args.Ns * args.Ks
        
        ck_tile_gflops = (total_flops / ck_tile_time) / 1e9
        pytorch_gflops = (total_flops / pytorch_time) / 1e9
        
        print(f"结果对比:")
        print(f"  ck_tile_python: {ck_tile_time/args.repeat*1000:.2f} ms/iter, {ck_tile_gflops:.2f} GFLOPS")
        print(f"  PyTorch:        {pytorch_time/args.repeat*1000:.2f} ms/iter, {pytorch_gflops:.2f} GFLOPS")
        print(f"  加速比:         {pytorch_time/ck_tile_time:.2f}x")
        
        return True
    except Exception as e:
        print(f"性能测试失败: {e}")
        return False

def test_fp16_only():
    """测试 fp16 数据类型"""
    print("=" * 50)
    print("测试 fp16 数据类型...")
    
    try:
        # 创建参数
        args = ck_tile_python.GemmArgs()
        args.Ms = 512
        args.Ns = 512
        args.Ks = 512
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 1
        args.repeat = 1
        args.persistent = False
        
        # 创建 fp16 张量
        A = torch.randn(args.Ms, args.Ks, dtype=torch.float16, device='cuda')
        B = torch.randn(args.Ks, args.Ns, dtype=torch.float16, device='cuda')
        
        print(f"创建 fp16 张量: A({args.Ms}x{args.Ks}), B({args.Ks}x{args.Ns})")
        print(f"A tensor: shape={A.shape}, dtype={A.dtype}")
        print(f"B tensor: shape={B.shape}, dtype={B.dtype}")
        
        # 测试
        C = ck_tile_python.gemm_api(A, B, args)
        print(f"✅ fp16 GEMM 测试成功")
        print(f"结果张量: shape={C.shape}, dtype={C.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ fp16 GEMM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试 ck_tile_python GEMM 模块")
    print("=" * 50)
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法进行测试")
        return
    
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name()}")
    
    # 测试模块导入
    if not test_module_import():
        return
    
    # 测试参数创建
    args = test_gemm_args()
    if args is None:
        return
    
    # 创建测试张量
    A_tensor, B_tensor = create_test_tensors(args)
    if A_tensor is None or B_tensor is None:
        return
    
    # 测试主要功能
    C_tensor = test_gemm_api(A_tensor, B_tensor, args)
    if C_tensor is None:
        return
    
    # 验证结果
    if args.validate:
        is_correct = validate_results(A_tensor, B_tensor, C_tensor, args)
        if not is_correct:
            print("❌ 验证失败，结果不正确")
            return
    
    # 性能测试
    benchmark_comparison(A_tensor, B_tensor, args)
    
    # 测试 fp16 数据类型
    test_fp16_only()
    
    print("=" * 50)
    print("🎉 所有测试完成！")

if __name__ == "__main__":
    main()
