cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CUDAExtension LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# 1) Project Configuration
# ----------------------------------------------------------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/libtorch" CACHE PATH "Path to LibTorch")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

# Find required packages
find_package(Torch REQUIRED)
message(STATUS "Found LibTorch: ${TORCH_LIBRARIES}")

# Ensure we're using the same CXX flags as LibTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ----------------------------------------------------------------------------
# 2) Include directories
# ----------------------------------------------------------------------------
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

# ----------------------------------------------------------------------------
# 3) Create a library for the CUDA kernel (shared between main and tests)
# ----------------------------------------------------------------------------
set(KERNEL_SOURCES
    src/kernel_add.cu
)

# Create a static library containing the CUDA kernels
add_library(cuda_kernels STATIC ${KERNEL_SOURCES})

# Set CUDA properties for the library
set_target_properties(cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link LibTorch to the kernel library
target_link_libraries(cuda_kernels PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(cuda_kernels PRIVATE cxx_std_17)

# ----------------------------------------------------------------------------
# 4) Main executable
# ----------------------------------------------------------------------------
add_executable(cuda_torch_app src/main.cpp)

# Link the main app with the kernel library
target_link_libraries(cuda_torch_app PRIVATE cuda_kernels "${TORCH_LIBRARIES}")
target_compile_features(cuda_torch_app PRIVATE cxx_std_17)

# ----------------------------------------------------------------------------
# 5) Add tests subdirectory
# ----------------------------------------------------------------------------
add_subdirectory(tests)

# ----------------------------------------------------------------------------
# 6) Optional: Add custom targets for convenience
# ----------------------------------------------------------------------------
# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS cuda_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# If you want to see verbose nvcc/C++ flags, uncomment:
# set(CMAKE_VERBOSE_MAKEFILE ON)
