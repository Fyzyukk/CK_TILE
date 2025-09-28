#!/usr/bin/env python3
import torch
import ck_tile_python
import time

def test_module_import():
    try:
        print(f"成功导入 ck_tile_python 模块")
        print(f"可用函数: {dir(ck_tile_python)}")
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False

def test_simple_gemm_args():
    """测试 SimpleGemmArgs 结构"""
    print("=" * 50)
    print("测试 SimpleGemmArgs 结构...")
    try:
        # 创建参数 - 增加更大的尺寸测试
        args = ck_tile_python.GroupGemmArgs()
        args.Ms = [512, 1024, 2048, 4096]
        args.Ns = [512, 1024, 2048, 4096] 
        args.Ks = [512, 1024, 2048, 7168]
        args.group_count = 4
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 2
        args.repeat = 5
        
        print(f"参数创建成功:")
        print(f"  Ms: {args.Ms}")
        print(f"  Ns: {args.Ns}")
        print(f"  Ks: {args.Ks}")
        print(f"  group_count: {args.group_count}")
        print(f"  dtype: {args.dtype}")
        return args
    except Exception as e:
        print(f"参数创建失败: {e}")
        return None

def create_test_tensors(args):
    """创建测试张量"""
    print("=" * 50)
    print("创建测试张量...")
    try:
        A_tensors = []
        B_tensors = []
        
        # 确定数据类型
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        
        for i in range(args.group_count):
            M, N, K = args.Ms[i], args.Ns[i], args.Ks[i]
            
            # 创建随机测试数据
            A = torch.randn(M, K, dtype=dtype, device='cuda')
            # 创建 B 矩阵为列主布局 (转置后连续化)
            B = torch.randn(K, N, dtype=dtype, device='cuda')
            
            A_tensors.append(A)
            B_tensors.append(B)
            
            print(f"  组 {i}: A({M}x{K}), B({K}x{N}) -> C({M}x{N})")
        
        return A_tensors, B_tensors
    except Exception as e:
        print(f"张量创建失败: {e}")
        return None, None

def test_grouped_gemm_simple(A_tensors, B_tensors, args):
    """测试 grouped_gemm_simple 函数"""
    print("=" * 50)
    print("测试 grouped_gemm_simple 函数...")
    try:
        # 调用我们的实现
        start_time = time.time()
        C_tensors = ck_tile_python.grouped_gemm_api(A_tensors, B_tensors, args)
        end_time = time.time()
        
        print(f"执行成功！耗时: {end_time - start_time:.4f} 秒")
        print(f"返回了 {len(C_tensors)} 个结果张量")
        
        # 检查结果形状
        for i, C in enumerate(C_tensors):
            expected_M, expected_N = args.Ms[i], args.Ns[i]
            actual_M, actual_N = C.shape
            print(f"  组 {i}: 期望形状 ({expected_M}, {expected_N}), 实际形状 ({actual_M}, {actual_N})")
            
            if (actual_M, actual_N) != (expected_M, expected_N):
                print(f"  ❌ 形状不匹配！")
                return False
            else:
                print(f"  ✅ 形状正确")
        
        return C_tensors
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensors, B_tensors, C_tensors, args):
    """验证结果正确性"""
    print("=" * 50)
    print("验证结果正确性...")
    try:
        all_correct = True
        
        for i in range(args.group_count):
            # 使用 PyTorch 计算期望结果
            expected = torch.mm(A_tensors[i], B_tensors[i])
            actual = C_tensors[i]
            
            # 比较结果
            is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
            
            if is_close:
                print(f"  组 {i}: ✅ 结果正确")
            else:
                print(f"  组 {i}: ❌ 结果不正确")
                print(f"    最大差异: {torch.max(torch.abs(actual - expected)).item():.6f}")
                all_correct = False
        
        return all_correct
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def benchmark_comparison(A_tensors, B_tensors, args):
    """性能对比测试"""
    print("=" * 50)
    print("性能对比测试...")
    try:
        # 测试我们的实现
        print("测试 ck_tile_python 实现...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            C_tensors = ck_tile_python.grouped_gemm_api(A_tensors, B_tensors, args)
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        # 测试 PyTorch 实现
        print("测试 PyTorch 实现...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            pytorch_results = [torch.mm(A_tensors[i], B_tensors[i]) for i in range(args.group_count)]
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        # 计算性能指标
        total_flops = sum(2.0 * args.Ms[i] * args.Ns[i] * args.Ks[i] for i in range(args.group_count))
        
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

def main():
    """主测试函数"""
    print("开始测试 ck_tile_python 模块")
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
    args = test_simple_gemm_args()
    if args is None:
        return
    
    # 创建测试张量
    A_tensors, B_tensors = create_test_tensors(args)
    if A_tensors is None or B_tensors is None:
        return
    
    # 测试主要功能
    C_tensors = test_grouped_gemm_simple(A_tensors, B_tensors, args)
    if C_tensors is None:
        return
    
    # 验证结果
    if args.validate:
        is_correct = validate_results(A_tensors, B_tensors, C_tensors, args)
        if not is_correct:
            print("❌ 验证失败，结果不正确")
            return
    
    # 性能测试
    benchmark_comparison(A_tensors, B_tensors, args)
    
    print("=" * 50)
    print("🎉 所有测试完成！")

if __name__ == "__main__":
    main()
