#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

// Custom test environment to initialize CUDA
class CudaTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available. Device count: " << torch::cuda::device_count() << std::endl;
            // Set random seed for reproducible tests
            torch::manual_seed(42);
            torch::cuda::manual_seed(42);
        } else {
            std::cout << "CUDA is not available. Some tests may be skipped." << std::endl;
        }
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Add custom test environment
    ::testing::AddGlobalTestEnvironment(new CudaTestEnvironment);
    
    return RUN_ALL_TESTS();
}
