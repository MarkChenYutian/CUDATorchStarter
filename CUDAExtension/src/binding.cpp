#include <torch/extension.h>
#include "kernel_add.h"

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernel addition operations";
    
    m.def("add_cuda", &add_cuda, "Add two CUDA tensors",
          py::arg("a"), py::arg("b"));
    
    m.def("add_cpu", &add_cpu, "Add two CPU tensors", 
          py::arg("a"), py::arg("b"));
    
    m.def("add_fwd", &add_fwd, "Add two tensors (auto-dispatch CPU/CUDA)",
          py::arg("a"), py::arg("b"));
}
