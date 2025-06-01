#include "kernel_add.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// Function to print tensor with nice formatting
void print_tensor(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n";
    std::cout << std::fixed << std::setprecision(4);
    
    // Convert to CPU if it's on CUDA for printing
    auto cpu_tensor = tensor.to(torch::kCPU);
    auto accessor = cpu_tensor.accessor<float, 2>();
    
    for (int i = 0; i < cpu_tensor.size(0); ++i) {
        std::cout << "[";
        for (int j = 0; j < cpu_tensor.size(1); ++j) {
            std::cout << std::setw(8) << accessor[i][j];
            if (j < cpu_tensor.size(1) - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "CUDA Tensor Addition Test\n";
    std::cout << "========================\n\n";
    
    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Please check your PyTorch CUDA installation.\n";
        return -1;
    }
    
    std::cout << "CUDA is available. Device count: " << torch::cuda::device_count() << "\n\n";
    
    // Set random seed for reproducible results
    torch::manual_seed(42);
    
    // Create two 4x4 random tensors on CPU first
    auto a_cpu = torch::rand({4, 4}, torch::kFloat);
    auto b_cpu = torch::rand({4, 4}, torch::kFloat);
    
    std::cout << "Input tensors (CPU):\n";
    print_tensor(a_cpu, "Tensor A");
    print_tensor(b_cpu, "Tensor B");
    
    // Move tensors to CUDA
    auto a_cuda = a_cpu.to(torch::kCUDA);
    auto b_cuda = b_cpu.to(torch::kCUDA);
    
    std::cout << "Tensors moved to CUDA device\n\n";
    
    // 1. Perform addition using PyTorch's built-in add function
    std::cout << "1. Using PyTorch's built-in add function:\n";
    auto result_torch = torch::add(a_cuda, b_cuda);
    print_tensor(result_torch, "PyTorch Result (A + B)");
    
    // 2. Perform addition using our custom CUDA kernel
    std::cout << "2. Using custom CUDA kernel:\n";
    auto result_custom = add_cuda(a_cuda, b_cuda);
    print_tensor(result_custom, "Custom CUDA Result (A + B)");
    
    // 3. Verify results are the same (within floating point tolerance)
    std::cout << "3. Verification:\n";
    auto diff = torch::abs(result_torch - result_custom);
    auto max_diff = torch::max(diff).item<float>();
    
    std::cout << "Maximum absolute difference: " << std::scientific << max_diff << "\n";
    
    if (max_diff < 1e-6) {
        std::cout << "✓ Results match! Custom CUDA kernel is working correctly.\n";
    } else {
        std::cout << "✗ Results don't match. There might be an issue with the custom kernel.\n";
        print_tensor(diff, "Difference tensor");
    }
    
    std::cout << "\nTest completed successfully!\n";
    
    return 0;
}
