#!/usr/bin/env python3
import torch
import ck_tile_python
import time
import numpy as np

def test_module_import():
    print("=" * 60)
    try:
        print(f"List ck_tile_python: {[f for f in dir(ck_tile_python) if not f.startswith('_')]}")
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

def create_test_args():
    print("=" * 60)
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
        
        print(f"  M : {args.Ms}")
        print(f"  N : {args.Ns}")
        print(f"  K : {args.Ks}")
        print(f"  batched_count: {args.batched_count}")
        print(f"  dtype: {args.dtype}")
        print(f"  warmup: {args.warmup}")
        print(f"  repate: {args.repeat}")
        return args
    except Exception as e:
        print(f"error: {e}")
        return None

def create_test_tensors(args):
    print("=" * 60)
    try:
        M, N, K = args.Ms, args.Ns, args.Ks
        batch_count = args.batched_count
        dtype = torch.float16
        
        print(f"  A: ({batch_count}, {M}, {K}) - Batched_A")
        print(f"  B: ({batch_count}, {K}, {N}) - Batched_B")
        print(f"  C: ({batch_count}, {M}, {N}) - Batched_C")
        
        A = torch.randn(batch_count, M, K, dtype=dtype, device='cuda')
        B = torch.randn(batch_count, K, N, dtype=dtype, device='cuda')
        
        print(f"A tensor: shape={A.shape}, dtype={A.dtype}, device={A.device}")
        print(f"  stride: {A.stride()}")
        print(f"  is_contiguous: {A.is_contiguous()}")
        
        print(f"B tensor: shape={B.shape}, dtype={B.dtype}, device={B.device}")
        print(f"  stride: {B.stride()}")
        print(f"  is_contiguous: {B.is_contiguous()}")
        
        return A, B
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_batched_gemm_api(A_tensor, B_tensor, args):
    print("=" * 60)
    try:
        print(" ck_tile_python.batched_gemm_api...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        C_tensor = ck_tile_python.batched_gemm_api(A_tensor, B_tensor, args)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"  Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"  Tensor Shape: {C_tensor.shape}")
        print(f"  Tensor dtype: {C_tensor.dtype}")
        print(f"  Tensor Device: {C_tensor.device}")
        
        expected_shape = (args.batched_count, args.Ms, args.Ns)
        actual_shape = C_tensor.shape
        print(f"  Expected Shape: {expected_shape}")
        print(f"  Actually Shape: {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"  Shape Correct")
        else:
            print(f"  Shape error")
            return None
        
        return C_tensor
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensor, B_tensor, C_tensor, args):
    print("=" * 60)
    try:
        print(" PyTorch torch.bmm()...")
        torch.cuda.synchronize()
        start_time = time.time()
        expected = torch.bmm(A_tensor, B_tensor)  
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        print(f"PyTorch Time: {pytorch_time*1000:.2f} ms")
        
        actual = C_tensor
        
        print(f"  ck_tile Shape: {actual.shape}")
        print(f"  PyTorch Shape: {expected.shape}")
        print(f"  ck_tile dtype: {actual.dtype}")
        print(f"  PyTorch dtype: {expected.dtype}")
        
        is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
        
        if is_close:
            print("  correct")
        else:
            print("  error")
            
            diff = torch.abs(actual - expected)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            
            print(f"  Max: {max_diff:.6f}")
            print(f"  Avg: {mean_diff:.6f}")
            print(f"  Stable: {std_diff:.6f}")
            
            large_diff_count = torch.sum(diff > 1e-2).item()
            total_elements = diff.numel()
            large_diff_ratio = large_diff_count / total_elements * 100
            
            print(f"  diff: {large_diff_count}/{total_elements} ({large_diff_ratio:.2f}%)")
            
            return False
        
        return True
    except Exception as e:
        print(f" error: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_performance(A_tensor, B_tensor, args):
    print("=" * 60)
    try:
        total_flops = 2.0 * args.batched_count * args.Ms * args.Ns * args.Ks
        print(f"Flops: {total_flops/1e12:.2f} TFLOPs")
        
        print(f" ck_tile_python  ({args.repeat} )...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(args.repeat):
            C_tensor = ck_tile_python.batched_gemm_api(A_tensor, B_tensor, args)
            if i == 0: 
                first_result = C_tensor.clone()
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        print(f" PyTorch ({args.repeat} )...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(args.repeat):
            pytorch_result = torch.bmm(A_tensor, B_tensor)
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        ck_tile_avg_time = ck_tile_time / args.repeat
        pytorch_avg_time = pytorch_time / args.repeat
        
        ck_tile_gflops = (total_flops / ck_tile_avg_time) / 1e9
        pytorch_gflops = (total_flops / pytorch_avg_time) / 1e9
        
        print(f"  ck_tile_python:")
        print(f"    Avg Time: {ck_tile_avg_time*1000:.2f} ms/iter")
        print(f"    Flops: {ck_tile_gflops:.2f} GFLOPS")
        print(f"    Throght: {total_flops/ck_tile_avg_time/1e12:.2f} TFLOPs/s")
        
        print(f"  PyTorch:")
        print(f"    Avg Time: {pytorch_avg_time*1000:.2f} ms/iter")
        print(f"    Flops: {pytorch_gflops:.2f} GFLOPS")
        print(f"    Throught: {total_flops/pytorch_avg_time/1e12:.2f} TFLOPs/s")
        
        speedup = pytorch_avg_time / ck_tile_avg_time
        
        if speedup > 1.0:
            print(f"   ck_tile_python   {speedup:.2f}x")
        else:
            print(f"   PyTorch  {1/speedup:.2f}x")
        
        return first_result
    except Exception as e:
        print(f" error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    
    if not torch.cuda.is_available():
        print("No Devices")
        return
    
    print(f"Devices : {torch.cuda.get_device_name()}")

    if not test_module_import():
        return
    
    args = create_test_args()
    if args is None:
        return
    
    A_tensor, B_tensor = create_test_tensors(args)
    if A_tensor is None or B_tensor is None:
        return
    
    C_tensor = test_batched_gemm_api(A_tensor, B_tensor, args)
    if C_tensor is None:
        return
    
    if args.validate:
        is_correct = validate_results(A_tensor, B_tensor, C_tensor, args)
        if not is_correct:
            print("error")
            return
    
    # benchmark_result = benchmark_performance(A_tensor, B_tensor, args)
    # if benchmark_result is None:
    #     return   
    print("=" * 60)

if __name__ == "__main__":
    main()
