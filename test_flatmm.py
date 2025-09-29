#!/usr/bin/env python3
import torch
import ck_tile_python
import time

def test_module_import():
    print("=" * 50)
    try:
        print(f"List ck_tile_python: {dir(ck_tile_python)}")
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

def test_flatmm_args():
    print("=" * 50)
    try:
        args = ck_tile_python.FlatmmArgs()
        args.Ms = 256
        args.Ns = 256
        args.Ks = 128
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 50
        args.repeat = 100
        
        print(f"  Ms: {args.Ms}")
        print(f"  Ns: {args.Ns}")
        print(f"  Ks: {args.Ks}")
        print(f"  dtype: {args.dtype}")
        return args
    except Exception as e:
        print(f"error: {e}")
        return None

def create_test_tensors(args):
    print("=" * 50)
    try:
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        
        M, N, K = args.Ms, args.Ns, args.Ks
        
        A = torch.randn(M, K, dtype=dtype, device='cuda')
        B = torch.randn(K, N, dtype=dtype, device='cuda')
        
        print(f"Matrix Shape: A({M}x{K}), B({K}x{N}) -> C({M}x{N})")
        print(f"A tensor: shape={A.shape}, stride={A.stride()}")
        print(f"B tensor: shape={B.shape}, stride={B.stride()}")
        
        return A, B
    except Exception as e:
        print(f"error: {e}")
        return None, None

def test_flatmm_api(A_tensor, B_tensor, args):
    print("=" * 50)
    try:
        start_time = time.time()
        C_tensor = ck_tile_python.flatmm_api(A_tensor, B_tensor, args)
        end_time = time.time()
        
        print(f"Time: {end_time - start_time:.4f} s")
        print(f"C_tensor Shape: {C_tensor.shape}")
        
        expected_M, expected_N = args.Ms, args.Ns
        actual_M, actual_N = C_tensor.shape
        print(f"Shape ({expected_M}, {expected_N}), Actually Shape ({actual_M}, {actual_N})")
        
        if (actual_M, actual_N) != (expected_M, expected_N):
            print(f"Shape error")
            return None
        else:
            print(f"Shape correct")
        
        return C_tensor
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensor, B_tensor, C_tensor, args):
    print("=" * 50)
    try:
        expected = torch.mm(A_tensor, B_tensor)
        actual = C_tensor
        
        is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
        
        if is_close:
            print("correct")
        else:
            print("error")
            diff = torch.abs(actual - expected)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            print(f"  Max: {max_diff:.6f}")
            print(f"  Avg: {mean_diff:.6f}")
            return False
        
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

def benchmark_comparison(A_tensor, B_tensor, args):
    print("=" * 50)
    try:
        print(" ck_tile_python...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            C_tensor = ck_tile_python.flatmm_api(A_tensor, B_tensor, args)
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        print(" PyTorch...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            pytorch_result = torch.mm(A_tensor, B_tensor)
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        total_flops = 2.0 * args.Ms * args.Ns * args.Ks
        
        ck_tile_gflops = (total_flops / ck_tile_time) / 1e9
        pytorch_gflops = (total_flops / pytorch_time) / 1e9
        
        print(f"  ck_tile_python: {ck_tile_time/args.repeat*1000:.2f} ms/iter, {ck_tile_gflops:.2f} GFLOPS")
        print(f"  PyTorch:        {pytorch_time/args.repeat*1000:.2f} ms/iter, {pytorch_gflops:.2f} GFLOPS")
        
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

def main():
    
    if not torch.cuda.is_available():
        print("No Devices")
        return
    
    print(f"Devices : {torch.cuda.get_device_name()}")
    
    if not test_module_import():
        return
    
    args = test_flatmm_args()
    if args is None:
        return
    
    A_tensor, B_tensor = create_test_tensors(args)
    if A_tensor is None or B_tensor is None:
        return
    
    C_tensor = test_flatmm_api(A_tensor, B_tensor, args)
    if C_tensor is None:
        return
    
    if args.validate:
        is_correct = validate_results(A_tensor, B_tensor, C_tensor, args)
        if not is_correct:
            print("error")
            return
    
    # benchmark_comparison(A_tensor, B_tensor, args)    
    print("=" * 50)

if __name__ == "__main__":
    main()
