from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

setup(
    name='custom_extension',
    ext_modules=[
        CUDAExtension(
            name='custom_extension',
            sources=[
                'src/binding.cpp',
                'src/kernel_add.cu'
            ],
            include_dirs=[
                'include',
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-arch=sm_80',
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
