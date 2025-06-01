#include <torch/extension.h>
#include "kernel_add.h"

// Python bindings with detailed type information for stub generation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernel addition operations";
    
    m.def("add_cuda", &add_cuda, 
          "Add two CUDA tensors element-wise.\n\n"
          "Args:\n"
          "    a (torch.Tensor): First CUDA tensor\n"
          "    b (torch.Tensor): Second CUDA tensor (must have same shape as a)",
          py::arg("a"), py::arg("b"),
          py::return_value_policy::move);
    
    m.def("add_cpu", &add_cpu, 
          "Add two CPU tensors element-wise.\n\n"
          "Args:\n"
          "    a (torch.Tensor): First CPU tensor\n"
          "    b (torch.Tensor): Second CPU tensor (must have same shape as a)",
          py::arg("a"), py::arg("b"),
          py::return_value_policy::move);
    
    m.def("add_fwd", &add_fwd, 
          "Add two tensors element-wise with automatic device dispatch.\n\n"
          "Automatically chooses CUDA or CPU implementation based on input tensors.\n\n"
          "Args:\n"
          "    a (torch.Tensor): First tensor\n"
          "    b (torch.Tensor): Second tensor (must have same shape and device as a)",
          py::arg("a"), py::arg("b"),
          py::return_value_policy::move);
}
