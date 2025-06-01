import torch
import CUDAExtension

def test_cuda_kernel():
    """Test the CUDA kernel binding"""
    print("Testing CUDA kernel binding...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping CUDA tests")
        return
    
    # Create test tensors on GPU
    device = torch.device('cuda')
    a = torch.randn(1000, device=device, dtype=torch.float32)
    b = torch.randn(1000, device=device, dtype=torch.float32)
    
    # Test our custom CUDA kernel
    print("Testing add_cuda...")
    result_cuda = CUDAExtension.add_cuda(a, b)
    
    # Compare with PyTorch's native addition
    expected = a + b
    
    # Check if results are close
    if torch.allclose(result_cuda, expected, rtol=1e-5, atol=1e-5):
        print("✓ CUDA kernel test passed!")
    else:
        print("✗ CUDA kernel test failed!")
        print(f"Max difference: {torch.max(torch.abs(result_cuda - expected))}")

def test_cpu_kernel():
    """Test the CPU kernel binding"""
    print("Testing CPU kernel binding...")
    
    # Create test tensors on CPU
    a = torch.randn(1000, dtype=torch.float32)
    b = torch.randn(1000, dtype=torch.float32)
    
    # Test our custom CPU kernel
    print("Testing add_cpu...")
    result_cpu = CUDAExtension.add_cpu(a, b)
    
    # Compare with PyTorch's native addition
    expected = a + b
    
    # Check if results are close
    if torch.allclose(result_cpu, expected, rtol=1e-5, atol=1e-5):
        print("✓ CPU kernel test passed!")
    else:
        print("✗ CPU kernel test failed!")
        print(f"Max difference: {torch.max(torch.abs(result_cpu - expected))}")

def test_auto_dispatch():
    """Test the automatic CPU/CUDA dispatch"""
    print("Testing automatic dispatch...")
    
    # Test CPU
    a_cpu = torch.randn(100, dtype=torch.float32)
    b_cpu = torch.randn(100, dtype=torch.float32)
    result_cpu = CUDAExtension.add_fwd(a_cpu, b_cpu)
    expected_cpu = a_cpu + b_cpu
    
    if torch.allclose(result_cpu, expected_cpu, rtol=1e-5, atol=1e-5):
        print("✓ CPU auto-dispatch test passed!")
    else:
        print("✗ CPU auto-dispatch test failed!")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        a_cuda = torch.randn(100, device=device, dtype=torch.float32)
        b_cuda = torch.randn(100, device=device, dtype=torch.float32)
        result_cuda = CUDAExtension.add_fwd(a_cuda, b_cuda)
        expected_cuda = a_cuda + b_cuda
        
        if torch.allclose(result_cuda, expected_cuda, rtol=1e-5, atol=1e-5):
            print("✓ CUDA auto-dispatch test passed!")
        else:
            print("✗ CUDA auto-dispatch test failed!")

def benchmark_performance():
    """Simple performance comparison"""
    if not torch.cuda.is_available():
        print("CUDA not available for benchmarking")
        return
    
    print("\nPerformance comparison:")
    device = torch.device('cuda')
    
    # Large tensors for meaningful timing
    size = 1000000  # 1M elements
    a = torch.randn(size, device=device, dtype=torch.float32)
    b = torch.randn(size, device=device, dtype=torch.float32)
    
    # Warm up
    for _ in range(10):
        _ = CUDAExtension.add_cuda(a, b)
        _ = a + b
    
    torch.cuda.synchronize()
    
    # Time custom kernel
    import time
    start = time.time()
    for _ in range(100):
        result_custom = CUDAExtension.add_cuda(a, b)
    torch.cuda.synchronize()
    custom_time = time.time() - start
    
    # Time PyTorch native
    start = time.time()
    for _ in range(100):
        result_native = a + b
    torch.cuda.synchronize()
    native_time = time.time() - start
    
    print(f"Custom kernel time: {custom_time:.4f}s")
    print(f"PyTorch native time: {native_time:.4f}s")
    print(f"Speedup: {native_time/custom_time:.2f}x")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print("\n" + "="*50)
    
    test_cpu_kernel()
    print()
    test_cuda_kernel()
    print()
    test_auto_dispatch()
    print()
    benchmark_performance()
