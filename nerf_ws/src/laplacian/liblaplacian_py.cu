#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "liblaplacian.hpp"

void forward(torch::Tensor &X, const torch::Tensor & indexes) {
    //TODO should validate indexes
    int64_t num_dims = X.dim();
    if (num_dims < 3 || num_dims > 4) {
        std::stringstream strstr;
        strstr << "expected input tensor to be either 2d or 3d" << std::endl;
        strstr << "got: " << num_dims << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int64_t x_dims = X.size(0);
    int64_t y_dims = X.size(1);
    int64_t z_dims = num_dims == 4 ? X.size(2) : 1;
    int64_t indexes_len = indexes.size(0);

    if (indexes.dim() != 1 ) {
        std::stringstream strstr;
        strstr << "expected index tensor to be either 1d" << std::endl;
        strstr << "got: " << indexes.dim() << std::endl;
        throw std::runtime_error(strstr.str());
    }

    auto *X_ptr = X.data().data_ptr<float>();
    const int * const indexes_ptr = indexes.data().data_ptr<int>();
    internal::forward(X_ptr, x_dims, y_dims, z_dims, indexes_ptr, indexes_len);
}

void backward(const torch::Tensor &X, const torch::Tensor & indexes, const torch::Tensor &dL_dout, torch::Tensor &dL_dV, torch::Tensor &dL_dC) {
    int64_t num_dims = X.dim();
    if (num_dims < 3 || num_dims > 4) {
        std::stringstream strstr;
        strstr << "expected input tensor to be either 2d or 3d" << std::endl;
        strstr << "got: " << num_dims << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int64_t x_dims = X.size(0);
    int64_t y_dims = X.size(1);
    int64_t z_dims = num_dims == 4 ? X.size(2) : 1;

    float const *const X_ptr = X.data().data_ptr<float>();
    const int * const indexes_ptr = indexes.data().data_ptr<int>();
    int64_t indexes_len = indexes.size(0);
    float const *const dL_dout_ptr = dL_dout.data().data_ptr<float>();
    auto *dL_dV_ptr = dL_dV.data().data_ptr<float>();
    auto *dL_dC_ptr = dL_dC.data().data_ptr<float>();

    internal::backward(X_ptr, x_dims, y_dims, z_dims, indexes_ptr, indexes_len, dL_dout_ptr, dL_dV_ptr, dL_dC_ptr);
}

PYBIND11_MODULE(laplacian_solver_py, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}