#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "liblaplacian.hpp"

void forward(torch::Tensor &X) {
    int64_t num_dims = X.dim();
    if (num_dims != 3) {
        std::stringstream strstr;
        strstr << "number of dims not equal to 3" << std::endl;
        strstr << "got: " << num_dims << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int64_t x_dims = X.size(0);
    int64_t y_dims = X.size(1);

    float *X_ptr = X.data().data_ptr<float>();
    internal::forward(X_ptr, x_dims, y_dims);
}

void backward(torch::Tensor &X, torch::Tensor & dL_dout, torch::Tensor & dL_dV, torch::Tensor & dL_dC) {
    int64_t num_dims = X.dim();
    if (num_dims != 3) {
        std::stringstream strstr;
        strstr << "number of dims not equal to 3" << std::endl;
        strstr << "got: " << num_dims << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int64_t x_dims = X.size(0);
    int64_t y_dims = X.size(1);

    const float * const X_ptr = X.data().data_ptr<float>();
    const float * const dL_dout_ptr = dL_dout.data().data_ptr<float>();
    float *dL_dV_ptr = dL_dV.data().data_ptr<float>();
    float *dL_dC_ptr = dL_dC.data().data_ptr<float>();

    internal::backward(X_ptr, dL_dout_ptr, dL_dV_ptr, dL_dC_ptr, x_dims, y_dims);
}

PYBIND11_MODULE(laplacian_solver_py, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}