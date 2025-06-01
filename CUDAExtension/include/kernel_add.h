#pragma once
#include <torch/torch.h>

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b);

torch::Tensor add_fwd(torch::Tensor a, torch::Tensor b);
