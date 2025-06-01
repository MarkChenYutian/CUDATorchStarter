"""
CUDA kernel addition operations
"""
from __future__ import annotations
import torch
__all__ = ['add_cpu', 'add_cuda', 'add_fwd']
def add_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two CPU tensors element-wise.
    
    Args:
        a (torch.Tensor): First CPU tensor
        b (torch.Tensor): Second CPU tensor (must have same shape as a)
    """
def add_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two CUDA tensors element-wise.
    
    Args:
        a (torch.Tensor): First CUDA tensor
        b (torch.Tensor): Second CUDA tensor (must have same shape as a)
    """
def add_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise with automatic device dispatch.
    
    Automatically chooses CUDA or CPU implementation based on input tensors.
    
    Args:
        a (torch.Tensor): First tensor
        b (torch.Tensor): Second tensor (must have same shape and device as a)
    """
