from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# from pybind11 import get_cmake_dir
import pybind11
import torch
import subprocess
import os

# Get CUDA version and architecture info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

class CustomBuildExt(BuildExtension):
    def run(self):
        # First run the normal build
        super().run()
        # Then generate type stubs using pybind11-stubgen
        self.generate_pybind_stubs()
    
    def generate_pybind_stubs(self):
        """Generate .pyi type stub files using pybind11-stubgen"""
        try:
            print("Generating type stubs with pybind11-stubgen...")
            
            # Try to run pybind11-stubgen
            cmd = [
                "python", "-m", "pybind11_stubgen", 
                "CUDAExtension",
                "--output-dir=.",
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="..")
            
            if result.returncode == 0:
                print("✓ Generated type stubs with pybind11-stubgen")
                
                # Move the generated stub to the right location
                if os.path.exists("custom_extension.pyi"):
                    print("✓ Type stub generated: custom_extension.pyi")
                else:
                    print("⚠ Warning: Stub file not found after generation")
            else:
                print("⚠ pybind11-stubgen failed, No type stub will be generated")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ pybind11-stubgen not available. No type stub will be generated.")
    

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
                pybind11.get_include(),  # Add pybind11 headers
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
        'build_ext': CustomBuildExt
    },
    # Include stub files in the package
    package_data={
        '': ['*.pyi'],
    },
    zip_safe=False,
    # Dependencies for stub generation
    setup_requires=[
        'pybind11-stubgen',  # For automatic stub generation
    ],
    install_requires=[
        'torch',
    ],
)
