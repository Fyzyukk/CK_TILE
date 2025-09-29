#!/usr/bin/env python3
import torch
import ck_tile_python
import time

def test_module_import():
    try:
        print(f"List ck_tile_python: {dir(ck_tile_python)}")
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

def test_group_gemm_args():
    print("=" * 50)
    try:
        args = ck_tile_python.GroupGemmArgs()
        args.Ms = [512, 1024, 2048, 4096]
        args.Ns = [512, 1024, 2048, 4096] 
        args.Ks = [512, 1024, 2048, 7168]
        args.group_count = 4
        args.dtype = "fp16"
        args.validate = True
        args.warmup = 2
        args.repeat = 5
        
        print(f"  Ms: {args.Ms}")
        print(f"  Ns: {args.Ns}")
        print(f"  Ks: {args.Ks}")
        print(f"  group_count: {args.group_count}")
        print(f"  dtype: {args.dtype}")
        return args
    except Exception as e:
        print(f"error: {e}")
        return None

def create_test_tensors(args):
    print("=" * 50)
    try:
        A_tensors = []
        B_tensors = []
        
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        
        for i in range(args.group_count):
            M, N, K = args.Ms[i], args.Ns[i], args.Ks[i]
            
            A = torch.randn(M, K, dtype=dtype, device='cuda')
            B = torch.randn(K, N, dtype=dtype, device='cuda')
            
            A_tensors.append(A)
            B_tensors.append(B)
            
            print(f"  group {i}: A({M}x{K}), B({K}x{N}) -> C({M}x{N})")
        
        return A_tensors, B_tensors
    except Exception as e:
        print(f"error: {e}")
        return None, None

def test_grouped_gemm_simple(A_tensors, B_tensors, args):
    print("=" * 50)
    try:
        start_time = time.time()
        C_tensors = ck_tile_python.grouped_gemm_api(A_tensors, B_tensors, args)
        end_time = time.time()
        
        print(f"Time: {end_time - start_time:.4f} s")
        print(f" {len(C_tensors)} Tensors")

        for i, C in enumerate(C_tensors):
            expected_M, expected_N = args.Ms[i], args.Ns[i]
            actual_M, actual_N = C.shape
            print(f"  group {i}: Shape ({expected_M}, {expected_N}), Actually Shape ({actual_M}, {actual_N})")
            
            if (actual_M, actual_N) != (expected_M, expected_N):
                print(f"Shape error")
                return False
            else:
                print(f"Shape correct")
        
        return C_tensors
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_results(A_tensors, B_tensors, C_tensors, args):
    print("=" * 50)
    try:
        all_correct = True
        
        for i in range(args.group_count):
            expected = torch.mm(A_tensors[i], B_tensors[i])
            actual = C_tensors[i]
            

            is_close = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
            
            if is_close:
                print(f"  group {i}: correct")
            else:
                print(f"  group {i}: error")
                print(f"    Max: {torch.max(torch.abs(actual - expected)).item():.6f}")
                all_correct = False
        
        return all_correct
    except Exception as e:
        print(f"erroer: {e}")
        return False

def benchmark_comparison(A_tensors, B_tensors, args):
    print("=" * 50)
    try:
        print(" ck_tile_python...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            C_tensors = ck_tile_python.grouped_gemm_api(A_tensors, B_tensors, args)
        
        torch.cuda.synchronize()
        ck_tile_time = time.time() - start_time
        
        print(" PyTorch...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(args.repeat):
            pytorch_results = [torch.mm(A_tensors[i], B_tensors[i]) for i in range(args.group_count)]
        
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
    
        total_flops = sum(2.0 * args.Ms[i] * args.Ns[i] * args.Ks[i] for i in range(args.group_count))
        
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
    
    args = test_group_gemm_args()
    if args is None:
        return
    
    A_tensors, B_tensors = create_test_tensors(args)
    if A_tensors is None or B_tensors is None:
        return
    
    C_tensors = test_grouped_gemm_simple(A_tensors, B_tensors, args)
    if C_tensors is None:
        return
    
    if args.validate:
        is_correct = validate_results(A_tensors, B_tensors, C_tensors, args)
        if not is_correct:
            print("error")
            return
    
    # benchmark_comparison(A_tensors, B_tensors, args)  
    print("=" * 50)

if __name__ == "__main__":
    main()
