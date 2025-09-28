#!/usr/bin/env python3
"""
LayerNorm2D 测试脚本
测试 CK_TILE 的 LayerNorm2D 实现
"""

import torch
import time
import numpy as np
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import ck_tile_python
    print("成功导入 ck_tile_python 模块")
except ImportError as e:
    print(f"导入 ck_tile_python 失败: {e}")
    sys.exit(1)

def create_test_tensors(m, n, dtype=torch.float16, device='cuda'):
    """创建测试张量"""
    try:
        # 创建输入张量 x (m, n)
        x = torch.randn(m, n, dtype=dtype, device=device)
        
        # 创建 gamma 和 beta 参数 (n,)
        gamma = torch.ones(n, dtype=dtype, device=device)
        beta = torch.zeros(n, dtype=dtype, device=device)
        
        print(f"创建张量: x={x.shape}, gamma={gamma.shape}, beta={beta.shape}")
        return x, gamma, beta
    except Exception as e:
        print(f"张量创建失败: {e}")
        return None, None, None

def validate_results(y_ck, y_ref, tolerance=1e-2):
    """验证结果"""
    try:
        # 计算差异
        diff = torch.abs(y_ck - y_ref)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"验证结果:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  容差阈值: {tolerance}")
        
        # 检查是否在容差范围内
        is_valid = max_diff < tolerance
        print(f"  验证结果: {'通过' if is_valid else '失败'}")
        
        return is_valid
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def test_layernorm2d_api(x_tensor, gamma_tensor, beta_tensor, args):
    """测试 layernorm2d_api 函数"""
    print("=" * 50)
    print("测试 layernorm2d_api 函数...")
    try:
        # 调用我们的实现
        start_time = time.time()
        y_tensor = ck_tile_python.layernorm2d_api(x_tensor, gamma_tensor, beta_tensor, args)
        end_time = time.time()
        
        print(f"执行成功！耗时: {end_time - start_time:.4f} 秒")
        print(f"返回结果张量形状: {y_tensor.shape}")
        
        # 检查结果形状
        expected_shape = (args.m, args.n)
        if y_tensor.shape != expected_shape:
            print(f"形状不匹配: 期望 {expected_shape}, 实际 {y_tensor.shape}")
            return False
        
        # 使用 PyTorch 的 LayerNorm 作为参考
        print("计算参考结果...")
        x_cpu = x_tensor.cpu().float()
        gamma_cpu = gamma_tensor.cpu().float()
        beta_cpu = beta_tensor.cpu().float()
        
        # PyTorch LayerNorm 计算
        y_ref = torch.nn.functional.layer_norm(x_cpu, [args.n], gamma_cpu, beta_cpu, eps=args.epsilon)
        y_ref = y_ref.to(y_tensor.dtype).to(y_tensor.device)
        
        # 验证结果
        is_valid = validate_results(y_tensor, y_ref, tolerance=1e-2)
        
        return is_valid
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layernorm2d_with_residual_api(x_tensor, x_residual_tensor, gamma_tensor, beta_tensor, args):
    """测试带残差连接的 layernorm2d_api"""
    print("=" * 50)
    print("测试 layernorm2d_with_residual_api 函数...")
    try:
        # 调用我们的实现
        start_time = time.time()
        y_tensor = ck_tile_python.layernorm2d_with_residual_api(
            x_tensor, x_residual_tensor, gamma_tensor, beta_tensor, args)
        end_time = time.time()
        
        print(f"执行成功！耗时: {end_time - start_time:.4f} 秒")
        print(f"返回结果张量形状: {y_tensor.shape}")
        
        # 使用 PyTorch 作为参考
        print("计算参考结果...")
        x_cpu = x_tensor.cpu().float()
        x_residual_cpu = x_residual_tensor.cpu().float()
        gamma_cpu = gamma_tensor.cpu().float()
        beta_cpu = beta_tensor.cpu().float()
        
        # 先加上残差，然后进行 LayerNorm
        x_combined = x_cpu + x_residual_cpu
        y_ref = torch.nn.functional.layer_norm(x_combined, [args.n], gamma_cpu, beta_cpu, eps=args.epsilon)
        y_ref = y_ref.to(y_tensor.dtype).to(y_tensor.device)
        
        # 验证结果
        is_valid = validate_results(y_tensor, y_ref, tolerance=1e-2)
        
        return is_valid
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layernorm2d_with_bias_api(x_tensor, x_bias_tensor, gamma_tensor, beta_tensor, args):
    """测试带偏置的 layernorm2d_api"""
    print("=" * 50)
    print("测试 layernorm2d_with_bias_api 函数...")
    try:
        # 调用我们的实现
        start_time = time.time()
        y_tensor = ck_tile_python.layernorm2d_with_bias_api(
            x_tensor, x_bias_tensor, gamma_tensor, beta_tensor, args)
        end_time = time.time()
        
        print(f"执行成功！耗时: {end_time - start_time:.4f} 秒")
        print(f"返回结果张量形状: {y_tensor.shape}")
        
        # 使用 PyTorch 作为参考
        print("计算参考结果...")
        x_cpu = x_tensor.cpu().float()
        x_bias_cpu = x_bias_tensor.cpu().float()
        gamma_cpu = gamma_tensor.cpu().float()
        beta_cpu = beta_tensor.cpu().float()
        
        # 先加上偏置，然后进行 LayerNorm
        x_with_bias = x_cpu + x_bias_cpu.unsqueeze(0)  # 广播偏置
        y_ref = torch.nn.functional.layer_norm(x_with_bias, [args.n], gamma_cpu, beta_cpu, eps=args.epsilon)
        y_ref = y_ref.to(y_tensor.dtype).to(y_tensor.device)
        
        # 验证结果
        is_valid = validate_results(y_tensor, y_ref, tolerance=1e-2)
        
        return is_valid
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_layernorm2d(x_tensor, gamma_tensor, beta_tensor, args, num_iterations=100):
    """性能基准测试"""
    print("=" * 50)
    print("性能基准测试...")
    
    # 预热
    for _ in range(10):
        _ = ck_tile_python.layernorm2d_api(x_tensor, gamma_tensor, beta_tensor, args)
    
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        _ = ck_tile_python.layernorm2d_api(x_tensor, gamma_tensor, beta_tensor, args)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"平均执行时间: {avg_time * 1000:.2f} ms")
    
    # 计算吞吐量
    total_elements = args.m * args.n
    throughput = total_elements / avg_time / 1e9  # GFLOPs
    print(f"吞吐量: {throughput:.2f} GFLOPs")

def test_different_dtypes():
    """测试不同数据类型"""
    print("=" * 50)
    print("测试不同数据类型...")
    
    test_cases = [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]
    
    for dtype_name, torch_dtype in test_cases:
        print(f"\n测试 {dtype_name} 数据类型...")
        
        # 创建参数
        args = ck_tile_python.Layernorm2dArgs()
        args.m = 1024
        args.n = 4096
        args.prec_i = dtype_name
        args.prec_o = dtype_name
        args.epsilon = 1e-5
        args.warmup = 5
        args.repeat = 20
        args.validate = True
        
        # 创建张量
        x, gamma, beta = create_test_tensors(args.m, args.n, torch_dtype, 'cuda')
        if x is None:
            continue
        
        # 测试
        success = test_layernorm2d_api(x, gamma, beta, args)
        print(f"{dtype_name} 测试结果: {'通过' if success else '失败'}")

def main():
    """主函数"""
    print("LayerNorm2D 测试开始...")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name()}")
    
    # 基本测试参数
    args = ck_tile_python.Layernorm2dArgs()
    args.m = 1024
    args.n = 4096
    args.prec_i = "fp16"
    args.prec_o = "fp16"
    args.epsilon = 1e-5
    args.warmup = 5
    args.repeat = 20
    args.validate = True
    
    print(f"测试参数: M={args.m}, N={args.n}, dtype={args.prec_i}")
    
    # 创建测试张量
    x, gamma, beta = create_test_tensors(args.m, args.n, torch.float16, 'cuda')
    if x is None:
        print("张量创建失败，退出测试")
        return
    
    # 基本功能测试
    print("\n" + "="*60)
    print("基本功能测试")
    print("="*60)
    
    success1 = test_layernorm2d_api(x, gamma, beta, args)
    
    # 残差连接测试
    print("\n" + "="*60)
    print("残差连接测试")
    print("="*60)
    
    x_residual = torch.randn(args.m, args.n, dtype=torch.float16, device='cuda')
    args.fused_add = 1  # 启用残差连接
    success2 = test_layernorm2d_with_residual_api(x, x_residual, gamma, beta, args)
    
    # 偏置测试
    print("\n" + "="*60)
    print("偏置测试")
    print("="*60)
    
    x_bias = torch.randn(args.n, dtype=torch.float16, device='cuda')
    args.xbias = 1  # 启用偏置
    args.fused_add = 0  # 关闭残差连接
    success3 = test_layernorm2d_with_bias_api(x, x_bias, gamma, beta, args)
    
    # 性能测试
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)
    
    args.xbias = 0  # 关闭偏置
    benchmark_layernorm2d(x, gamma, beta, args)
    
    # 不同数据类型测试
    print("\n" + "="*60)
    print("不同数据类型测试")
    print("="*60)
    
    test_different_dtypes()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    print(f"基本功能测试: {'通过' if success1 else '失败'}")
    print(f"残差连接测试: {'通过' if success2 else '失败'}")
    print(f"偏置测试: {'通过' if success3 else '失败'}")
    
    all_passed = success1 and success2 and success3
    print(f"\n总体结果: {'所有测试通过' if all_passed else '部分测试失败'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
