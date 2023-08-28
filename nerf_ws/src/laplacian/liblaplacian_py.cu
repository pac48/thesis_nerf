#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "liblaplacian.hpp"

void solve(torch::Tensor &t) {
    int64_t num_dims = t.dim();
    if (num_dims != 3) {
        std::stringstream strstr;
        strstr << "number of dims not equal to 3" << std::endl;
        strstr << "got: " << num_dims << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int64_t x_dims = t.size(0);
    int64_t y_dims = t.size(1);

    float *t_ptr = t.data().data_ptr<float>();
    internal::solve(t_ptr, x_dims, y_dims);
}

PYBIND11_MODULE(laplacian_solver_py, m) {
    m.def("solve", solve);
}