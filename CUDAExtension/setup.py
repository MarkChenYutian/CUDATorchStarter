from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA version and architecture info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

setup(
    name='cuda_kernel_add',
    ext_modules=[
        CUDAExtension(
            name='cuda_kernel_add',
            sources=[
                'src/binding.cpp',  # Python binding (relative to CUDAExtension/)
                'src/kernel_add.cu'  # CUDA kernel (relative to CUDAExtension/)
            ],
            include_dirs=[
                'include',  # Your header files (relative to CUDAExtension/)
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-arch=sm_80',  # Adjust for your GPU architecture
                    '-arch=sm_86',
                    '-arch=sm_89',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
