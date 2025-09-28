#!/usr/bin/env python3
"""
测试 ck_tile_python 模块的 Batched GEMM 测试脚本
使用 M=512, K=2048, N=1024, batch_count=8
"""

import torch
import ck_tile_python
import time
import numpy as np

def test_module_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试模块导入...")
    try:
        print(f"成功导入 ck_tile_python 模块")
        print(f"可用函数: {[f for f in dir(ck_tile_python) if not f.startswith('_')]}")
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False

def create_test_args():
    """创建测试参数"""
    print("=" * 60)
    print("创建测试参数...")
    try:
        args = ck_tile_python.BatchedGemmArgs()
        args.Ms = 512
        args.Ns = 1024
        args.Ks = 2048
        args.batched_count = 8
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 2
        args.repeat = 5
        
        print(f"参数创建成功:")
        print(f"  M (行数): {args.Ms}")
        print(f"  N (列数): {args.Ns}")
        print(f"  K (内积维度): {args.Ks}")
        print(f"  批次数量: {args.batched_count}")
        print(f"  数据类型: {args.dtype}")
        print(f"  预热次数: {args.warmup}")
        print(f"  重复次数: {args.repeat}")
        return args
    except Exception as e:
        print(f"参数创建失败: {e}")
        return None

def create_test_tensors(args):
    """创建测试张量"""
    print("=" * 60)
    print("创建测试张量...")
    try:
        M, N, K = args.Ms, args.Ns, args.Ks
        batch_count = args.batched_count
        dtype = torch.float16
        
        print(f"张量规格:")
        print(f"  A: ({batch_count}, {M}, {K}) - 批次矩阵A")
        print(f"  B: ({batch_count}, {K}, {N}) - 批次矩阵B")
        print(f"  C: ({batch_count}, {M}, {N}) - 批次矩阵C")
        
        # 创建随机测试数据
        A = torch.randn(batch_count, M, K, dtype=dtype, device='cuda')
        B = torch.randn(batch_count, K, N, dtype=dtype, device='cuda')
        
        print(f"\n张量信息:")
        print(f"A tensor: shape={A.shape}, dtype={A.dtype}, device={A.device}")
        print(f"  stride: {A.stride()}")
        print(f"  is_contiguous: {A.is_contiguous()}")
        
        print(f"B tensor: shape={B.shape}, dtype={B.dtype}, device={B.device}")
        print(f"  stride: {B.stride()}")
        print(f"  is_contiguous: {B.is_contiguous()}")
        
        # 计算内存使用量
        A_memory = A.numel() * A.element_size() / 1024 / 1024  # MB
        B_memory = B.numel() * B.element_size() / 1024 / 1024  # MB
        print(f"\n内存使用:")
        print(f"  A tensor: {A_memory:.2f} MB")
        print(f"  B tensor: {B_memory:.2f} MB")
        print(f"  总计: {A_memory + B_memory:.2f} MB")
        
        return A, B
    except Exception as e:
        print(f"张量创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_batched_gemm_api(A_tensor, B_tensor, args):
    """测试 batched_gemm_api 函数"""
    print("=" * 60)
    print("测试 batched_gemm_api 函数...")
    try:
        print("调用 ck_tile_python.batched_gemm_api...")
        
        # 调用我们的实现
        torch.cuda.synchronize()
        start_time = time.time()
        
        C_tensor = ck_tile_python.batched_gemm_api(A_tensor, B_tensor, args)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ 执行成功！")
        print(f"  耗时: {(end_time - start_time)*1000:.2f} ms")
        print(f"  返回张量形状: {C_tensor.shape}")
        print(f"  返回张量类型: {C_tensor.dtype}")
        print(f"  返回张量设备: {C_tensor.device}")
        
        # 检查结果形状
        expected_shape = (args.batched_count, args.Ms, args.Ns)
        actual_shape = C_tensor.shape
        print(f"\n形状验证:")
        print(f"  期望形状: {expected_shape}")
        print(f"  实际形状: {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"  ✅ 形状正确")
        else:
            print(f"  ❌ 形状不匹配！")
            return None
        
        return C_tensor
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensor, B_tensor, C_tensor, args):
    """验证结果正确性"""
    print("=" * 60)
    print("验证结果正确性...")
    try:
        print("使用 PyTorch torch.bmm() 计算参考结果...")
        
        # 使用 PyTorch 计算期望结果
        torch.cuda.synchronize()
        start_time = time.time()
        expected = torch.bmm(A_tensor, B_tensor)  # 批次矩阵乘法
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        print(f"PyTorch 计算耗时: {pytorch_time*1000:.2f} ms")
        
        actual = C_tensor
        
        print(f"\n结果比较:")
        print(f"  ck_tile 结果形状: {actual.shape}")
        print(f"  PyTorch 结果形状: {expected.shape}")
        print(f"  ck_tile 数据类型: {actual.dtype}")
        print(f"  PyTorch 数据类型: {expected.dtype}")
        
        # 比较结果
        print(f"\n数值验证:")
        is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
        
        if is_close:
            print("  ✅ 结果正确")
        else:
            print("  ❌ 结果不正确")
            
            # 计算详细差异
            diff = torch.abs(actual - expected)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            print(f"  标准差: {std_diff:.6f}")
            
            # 检查有多少元素差异较大
            large_diff_count = torch.sum(diff > 1e-2).item()
            total_elements = diff.numel()
            large_diff_ratio = large_diff_count / total_elements * 100
            
            print(f"  差异 > 1e-2 的元素: {large_diff_count}/{total_elements} ({large_diff_ratio:.2f}%)")
            
            return False
        
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_performance(A_tensor, B_tensor, args):
    """性能基准测试"""
    print("=" * 60)
    print("性能基准测试...")
    try:
        # 计算理论性能指标
        total_flops = 2.0 * args.batched_count * args.Ms * args.Ns * args.Ks
        print(f"理论计算量: {total_flops/1e12:.2f} TFLOPs")
        
        # 测试 ck_tile_python 实现
        print(f"\n测试 ck_tile_python 实现 ({args.repeat} 次)...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(args.repeat):
            C_tensor = ck_tile_python.batched_gemm_api(A_tensor, B_tensor, args)
            if i == 0:  # 第一次的结果用于验证
                first_result = C_tensor.clone()
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        # 测试 PyTorch 实现
        print(f"测试 PyTorch 实现 ({args.repeat} 次)...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(args.repeat):
            pytorch_result = torch.bmm(A_tensor, B_tensor)
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        # 计算性能指标
        ck_tile_avg_time = ck_tile_time / args.repeat
        pytorch_avg_time = pytorch_time / args.repeat
        
        ck_tile_gflops = (total_flops / ck_tile_avg_time) / 1e9
        pytorch_gflops = (total_flops / pytorch_avg_time) / 1e9
        
        print(f"\n性能对比结果:")
        print(f"  ck_tile_python:")
        print(f"    平均耗时: {ck_tile_avg_time*1000:.2f} ms/iter")
        print(f"    性能: {ck_tile_gflops:.2f} GFLOPS")
        print(f"    吞吐量: {total_flops/ck_tile_avg_time/1e12:.2f} TFLOPs/s")
        
        print(f"  PyTorch:")
        print(f"    平均耗时: {pytorch_avg_time*1000:.2f} ms/iter")
        print(f"    性能: {pytorch_gflops:.2f} GFLOPS")
        print(f"    吞吐量: {total_flops/pytorch_avg_time/1e12:.2f} TFLOPs/s")
        
        speedup = pytorch_avg_time / ck_tile_avg_time
        print(f"\n加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  ✅ ck_tile_python 比 PyTorch 快 {speedup:.2f}x")
        else:
            print(f"  ⚠️  PyTorch 比 ck_tile_python 快 {1/speedup:.2f}x")
        
        return first_result
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_usage():
    """测试内存使用情况"""
    print("=" * 60)
    print("内存使用测试...")
    try:
        # 获取初始内存状态
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # 创建大张量
        batch_count, M, N, K = 8, 512, 1024, 2048
        A = torch.randn(batch_count, M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(batch_count, K, N, dtype=torch.float16, device='cuda')
        
        after_creation = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # 执行计算
        args = ck_tile_python.BatchedGemmArgs()
        args.Ms, args.Ns, args.Ks = M, N, K
        args.batched_count = batch_count
        args.dtype = "fp16"
        args.warmup = 1
        args.repeat = 1
        
        C = ck_tile_python.batched_gemm_api(A, B, args)
        
        after_computation = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        print(f"内存使用情况:")
        print(f"  初始内存: {initial_memory:.2f} MB")
        print(f"  创建张量后: {after_creation:.2f} MB")
        print(f"  计算完成后: {after_computation:.2f} MB")
        print(f"  张量占用: {after_creation - initial_memory:.2f} MB")
        print(f"  计算额外占用: {after_computation - after_creation:.2f} MB")
        
        # 清理
        del A, B, C
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试 ck_tile_python Batched GEMM 模块")
    print("📊 测试规格: M=512, K=2048, N=1024, batch_count=8")
    print("=" * 60)
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法进行测试")
        return
    
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name()}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # 测试模块导入
    if not test_module_import():
        return
    
    # 创建测试参数
    args = create_test_args()
    if args is None:
        return
    
    # 创建测试张量
    A_tensor, B_tensor = create_test_tensors(args)
    if A_tensor is None or B_tensor is None:
        return
    
    # 测试主要功能
    C_tensor = test_batched_gemm_api(A_tensor, B_tensor, args)
    if C_tensor is None:
        return
    
    # 验证结果
    if args.validate:
        is_correct = validate_results(A_tensor, B_tensor, C_tensor, args)
        if not is_correct:
            print("❌ 验证失败，结果不正确")
            return
    
    # 性能测试
    benchmark_result = benchmark_performance(A_tensor, B_tensor, args)
    if benchmark_result is None:
        return
    
    # 内存使用测试
    test_memory_usage()
    
    print("=" * 60)
    print("🎉 所有测试完成！")
    print("📈 测试总结:")
    print("   ✅ 模块导入成功")
    print("   ✅ 参数创建成功") 
    print("   ✅ 张量创建成功")
    print("   ✅ API 调用成功")
    if args.validate:
        print("   ✅ 结果验证通过")
    print("   ✅ 性能测试完成")
    print("   ✅ 内存测试完成")

if __name__ == "__main__":
    main()
