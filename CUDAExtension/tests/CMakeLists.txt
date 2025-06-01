# tests/CMakeLists.txt

# Set path to local Google Test installation
set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/lib/googletest")

# Try to find Google Test first (in case it's installed system-wide)
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Use local Google Test installation
    message(STATUS "Using local Google Test from: ${GTEST_ROOT}")
    
    # Add Google Test as a subdirectory
    add_subdirectory(${GTEST_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/googletest)
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    # Set the targets (these are created by add_subdirectory above)
    set(GTEST_LIBRARIES gtest gtest_main)
    set(GTEST_INCLUDE_DIRS ${GTEST_ROOT}/googletest/include)
else()
    message(STATUS "Found system Google Test")
    set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
endif()

# ----------------------------------------------------------------------------
# Test executable
# ----------------------------------------------------------------------------
set(TEST_SOURCES
    test_main.cpp
    test_kernel_add.cpp
)

add_executable(cuda_tests ${TEST_SOURCES})

# Link with Google Test, our kernel library, and LibTorch
target_link_libraries(cuda_tests PRIVATE 
    ${GTEST_LIBRARIES}
    cuda_kernels 
    "${TORCH_LIBRARIES}"
)

# Set C++ standard
target_compile_features(cuda_tests PRIVATE cxx_std_17)

# Include directories
target_include_directories(cuda_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# ----------------------------------------------------------------------------
# Register tests with CTest
# ----------------------------------------------------------------------------
include(GoogleTest)
gtest_discover_tests(cuda_tests)
