# CUDAExtension/__init__.py
"""
CUDA Extension for PyTorch - Custom kernel operations
"""

# Import the compiled extension
try:
    from . import cuda_kernel_add
    
    # Re-export functions for easier access
    add_cuda = cuda_kernel_add.add_cuda
    add_cpu = cuda_kernel_add.add_cpu
    add_fwd = cuda_kernel_add.add_fwd
    
    __all__ = ['add_cuda', 'add_cpu', 'add_fwd', 'cuda_kernel_add']
    
except ImportError as e:
    print(f"Warning: Could not import CUDA extension: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    raise

__version__ = "0.1.0"
