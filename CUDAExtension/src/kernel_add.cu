#include "kernel_add.h"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int64_t numel) 
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}


torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "add_cuda: `a` must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "add_cuda: `b` must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "add_cuda: `a` and `b` must be the same size");

    auto c = torch::empty_like(a);
    int64_t numel = a.numel();

    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    add_kernel<<<blocks, threads>>>( 
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        numel
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add_kernel launch failed: ", cudaGetErrorString(err));

    return c;
}

torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cpu(), "add_cpu: `a` must be a CPU tensor");
    TORCH_CHECK(b.device().is_cpu(), "add_cpu: `b` must be a CPU tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "add_cpu: `a` and `b` must be the same size");

    return a + b;  // Use PyTorch's built-in addition for CPU tensors
}

torch::Tensor add_fwd(torch::Tensor a, torch::Tensor b) {
    if (a.device().is_cuda() && b.device().is_cuda()) {
        return add_cuda(a, b);
    } else if (a.device().is_cpu() && b.device().is_cpu()) {
        return add_cpu(a, b);
    } else {
        TORCH_CHECK(false, "add_fwd: `a` and `b` must be both CUDA or both CPU tensors");
    }
}
