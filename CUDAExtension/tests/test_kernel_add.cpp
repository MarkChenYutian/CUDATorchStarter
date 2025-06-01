#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "kernel_add.h"
#include <cmath>

class KernelAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        // Set deterministic random seed
        torch::manual_seed(12345);
        torch::cuda::manual_seed(12345);
    }
    
    // Helper function to check tensor equality with tolerance
    void expectTensorsEqual(const torch::Tensor& a, const torch::Tensor& b, float tolerance = 1e-6) {
        ASSERT_EQ(a.sizes(), b.sizes()) << "Tensor shapes don't match";
        
        auto diff = torch::abs(a - b);
        auto max_diff = torch::max(diff).item<float>();
        
        EXPECT_LT(max_diff, tolerance) 
            << "Maximum difference " << max_diff << " exceeds tolerance " << tolerance;
    }
};

// Test basic functionality with small tensors
TEST_F(KernelAddTest, BasicAddition) {
    auto a = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kCUDA);
    auto b = torch::tensor({{5.0, 6.0}, {7.0, 8.0}}, torch::kCUDA);
    
    auto result_custom = add_cuda(a, b);
    auto result_torch = torch::add(a, b);
    
    expectTensorsEqual(result_custom, result_torch);
    
    // Also check specific values
    auto expected = torch::tensor({{6.0, 8.0}, {10.0, 12.0}}, torch::kCUDA);
    expectTensorsEqual(result_custom, expected);
}

// Test with different tensor sizes
TEST_F(KernelAddTest, DifferentSizes) {
    std::vector<std::vector<int64_t>> shapes = {
        {1, 1},      // 1x1
        {4, 4},      // 4x4  
        {10, 10},    // 10x10
        {32, 32},    // 32x32
        {100, 100},  // 100x100
        {1000},      // 1D tensor with 1000 elements
        {50, 20}     // Non-square matrix
    };
    
    for (const auto& shape : shapes) {
        auto a = torch::rand(shape, torch::TensorOptions().device(torch::kCUDA));
        auto b = torch::rand(shape, torch::TensorOptions().device(torch::kCUDA));
        
        auto result_custom = add_cuda(a, b);
        auto result_torch = torch::add(a, b);
        
        expectTensorsEqual(result_custom, result_torch);
    }
}

// Test with zero tensors
TEST_F(KernelAddTest, ZeroTensors) {
    auto a = torch::zeros({5, 5}, torch::kCUDA);
    auto b = torch::ones({5, 5}, torch::kCUDA);
    
    auto result = add_cuda(a, b);
    auto expected = torch::ones({5, 5}, torch::kCUDA);
    
    expectTensorsEqual(result, expected);
}

// Test with negative numbers
TEST_F(KernelAddTest, NegativeNumbers) {
    auto a = torch::tensor({{-1.0, -2.0}, {-3.0, -4.0}}, torch::kCUDA);
    auto b = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kCUDA);
    
    auto result = add_cuda(a, b);
    auto expected = torch::zeros({2, 2}, torch::kCUDA);
    
    expectTensorsEqual(result, expected);
}

// Test large tensors to stress test the kernel
TEST_F(KernelAddTest, LargeTensors) {
    const int size = 1024;  // 1024x1024 = ~1M elements
    
    auto a = torch::rand({size, size}, torch::TensorOptions().device(torch::kCUDA));
    auto b = torch::rand({size, size}, torch::TensorOptions().device(torch::kCUDA));
    
    auto result_custom = add_cuda(a, b);
    auto result_torch = torch::add(a, b);
    
    expectTensorsEqual(result_custom, result_torch);
}

// Test edge case: single element tensor
TEST_F(KernelAddTest, SingleElement) {
    auto a = torch::tensor({5.0}, torch::kCUDA);
    auto b = torch::tensor({3.0}, torch::kCUDA);
    
    auto result = add_cuda(a, b);
    auto expected = torch::tensor({8.0}, torch::kCUDA);
    
    expectTensorsEqual(result, expected);
}

// Test with very small numbers (precision test)
TEST_F(KernelAddTest, SmallNumbers) {
    auto a = torch::full({3, 3}, 1e-7, torch::TensorOptions().device(torch::kCUDA));
    auto b = torch::full({3, 3}, 2e-7, torch::TensorOptions().device(torch::kCUDA));
    
    auto result = add_cuda(a, b);
    auto expected = torch::full({3, 3}, 3e-7, torch::TensorOptions().device(torch::kCUDA));
    
    expectTensorsEqual(result, expected, 1e-10);  // Tighter tolerance for small numbers
}

// Performance comparison test (optional)
TEST_F(KernelAddTest, PerformanceComparison) {
    const int size = 512;
    auto a = torch::rand({size, size}, torch::TensorOptions().device(torch::kCUDA));
    auto b = torch::rand({size, size}, torch::TensorOptions().device(torch::kCUDA));
    
    // Warm up
    for (int i = 0; i < 5; ++i) {
        add_cuda(a, b);
        torch::add(a, b);
    }
    
    cudaDeviceSynchronize();  // Ensure all operations complete
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto result = add_cuda(a, b);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto custom_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto result = torch::add(a, b);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    
    auto torch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Custom kernel time: " << custom_time.count() << " μs" << std::endl;
    std::cout << "PyTorch kernel time: " << torch_time.count() << " μs" << std::endl;
    
    // This is just informational - we don't assert performance
    SUCCEED();
}

// Error handling tests
TEST_F(KernelAddTest, ErrorHandling) {
    auto a_cuda = torch::rand({4, 4}, torch::kCUDA);
    auto b_cpu = torch::rand({4, 4}, torch::kCPU);
    auto c_cuda = torch::rand({3, 3}, torch::kCUDA);
    
    // Test device mismatch (should throw)
    EXPECT_THROW(add_cuda(a_cuda, b_cpu), std::exception);
    
    // Test size mismatch (should throw)  
    EXPECT_THROW(add_cuda(a_cuda, c_cuda), std::exception);
}

// Test the dispatch function add_fwd
TEST_F(KernelAddTest, DispatchFunction) {
    // Test CUDA path
    auto a_cuda = torch::rand({4, 4}, torch::kCUDA);
    auto b_cuda = torch::rand({4, 4}, torch::kCUDA);
    
    auto result_cuda = add_fwd(a_cuda, b_cuda);
    auto expected_cuda = torch::add(a_cuda, b_cuda);
    expectTensorsEqual(result_cuda, expected_cuda);
    
    // Test CPU path
    auto a_cpu = torch::rand({4, 4}, torch::kCPU);
    auto b_cpu = torch::rand({4, 4}, torch::kCPU);
    
    auto result_cpu = add_fwd(a_cpu, b_cpu);
    auto expected_cpu = torch::add(a_cpu, b_cpu);
    expectTensorsEqual(result_cpu, expected_cpu);
}
