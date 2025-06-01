# CUDAExtension/__init__.py
"""
CUDA Extension for PyTorch - Custom kernel operations
"""
import torch    # This is required otherwise import will fail.

# Import the compiled extension
try:
    from . import custom_extension
    
    # Re-export functions for easier access
    add_cuda = custom_extension.add_cuda
    add_cpu = custom_extension.add_cpu
    add_fwd = custom_extension.add_fwd
    
    __all__ = ['add_cuda', 'add_cpu', 'add_fwd', 'custom_extension']
    
except ImportError as e:
    print(f"Warning: Could not import CUDA extension: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    raise

__version__ = "0.1.0"
