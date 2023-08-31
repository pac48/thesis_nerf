#include <sstream>
#include <iostream>
#include "liblaplacian.hpp"

namespace internal {
    constexpr int blockSize = 256;

    __global__ void forward_kernel(float *X_ptr, int x_dims, int y_dims) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % x_dims;
        int ind_y = idx / x_dims;
        constexpr int stride = 4;

        if (ind_x > 0 && ind_x < x_dims - 1 && ind_y > 0 && ind_y < y_dims - 1) {

            const int left_idx = stride * idx - stride;
            const int up_idx = stride * idx - stride * x_dims;
            const int right_idx = stride * idx + stride;
            const int down_idx = stride * idx + stride * x_dims;
            const int up_left_idx = up_idx - stride;
            const int up_right_idx = up_idx + stride;
            const int down_left_idx = down_idx - stride;
            const int down_right_idx = down_idx + stride;

            float val = X_ptr[up_idx] * exp(-X_ptr[up_idx]) + X_ptr[down_idx] * exp(-X_ptr[down_idx]) +
                        X_ptr[left_idx] * exp(-X_ptr[left_idx]) + X_ptr[right_idx] * exp(-X_ptr[right_idx]) +
                        X_ptr[up_left_idx] * exp(-X_ptr[up_left_idx]) +
                        X_ptr[up_right_idx] * exp(-X_ptr[up_right_idx]) +
                        X_ptr[down_left_idx] * exp(-X_ptr[down_left_idx]) +
                        X_ptr[down_right_idx] * exp(-X_ptr[down_right_idx]);
            val = val / (exp(-X_ptr[up_idx]) + exp(-X_ptr[down_idx]) + exp(-X_ptr[left_idx]) + exp(-X_ptr[right_idx]) +
                         exp(-X_ptr[up_left_idx]) + exp(-X_ptr[up_right_idx]) + exp(-X_ptr[down_left_idx]) +
                         exp(-X_ptr[down_right_idx]));
            float &cost = X_ptr[stride * idx + 3];
            val += cost;

            __syncthreads();

            if (X_ptr[stride * idx + 1] < 0.5) {
                X_ptr[stride * idx] = val;
            } else if (X_ptr[stride * idx + 1] < 1.5) {
                X_ptr[stride * idx] = X_ptr[stride * idx + 2];
            }
        }
    }


    __global__ void
    backward_kernel(const float *const X_ptr, const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr,
                    int x_dims, int y_dims) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % x_dims;
        int ind_y = idx / x_dims;
        constexpr int stride = 4;

        if (ind_x > 0 && ind_x < x_dims - 1 && ind_y > 0 && ind_y < y_dims - 1) {

            if (X_ptr[stride * idx + 1] < 0.5) {
                const int indexes[8] = {stride * idx - stride,
                                        stride * idx - stride * x_dims,
                                        stride * idx + stride,
                                        stride * idx + stride * x_dims,
                                        stride * idx - stride * x_dims - stride,
                                        stride * idx - stride * x_dims + stride,
                                        stride * idx + stride * x_dims - stride,
                                        stride * idx + stride * x_dims + stride};


                float dout_dC = 1;
                dL_dC_ptr[idx] = dout_dC * dL_dout_ptr[idx];

                double dL_dV_ptr_idx = 0;

                double common_sum = 0;
                for (int ind: indexes) {
                    common_sum += exp(-X_ptr[ind]);
                }
                for (int i = 0; i < 8; i++) {
                    const int left_idx = indexes[(0 + i) % 8];
                    const int up_idx = indexes[(1 + i) % 8];
                    const int right_idx = indexes[(2 + i) % 8];
                    const int down_idx = indexes[(3 + i) % 8];
                    const int up_left_idx = indexes[(4 + i) % 8];
                    const int up_right_idx = indexes[(5 + i) % 8];
                    const int down_left_idx = indexes[(6 + i) % 8];
                    const int down_right_idx = indexes[(7 + i) % 8];

                    double tmp = X_ptr[up_left_idx] * exp(-2 * X_ptr[up_left_idx]) +
                                 X_ptr[left_idx] * exp(-X_ptr[up_left_idx] - X_ptr[left_idx]) +
                                 X_ptr[down_left_idx] * exp(-X_ptr[up_left_idx] - X_ptr[down_left_idx]) +
                                 X_ptr[up_idx] * exp(-X_ptr[up_left_idx] - X_ptr[up_idx]) +
                                 X_ptr[down_idx] * exp(-X_ptr[up_left_idx] - X_ptr[down_idx]) +
                                 X_ptr[up_right_idx] * exp(-X_ptr[up_left_idx] - X_ptr[up_right_idx]) +
                                 X_ptr[right_idx] * exp(-X_ptr[up_left_idx] - X_ptr[right_idx]) +
                                 X_ptr[down_right_idx] * exp(-X_ptr[up_left_idx] - X_ptr[down_right_idx]) +
                                 (-X_ptr[up_left_idx] + 1.0) * common_sum * exp(-X_ptr[up_left_idx]);
                    dL_dV_ptr_idx += tmp * dL_dout_ptr[up_left_idx/stride];  // (90000, 90000)* (90000,1)
                }

                dL_dV_ptr_idx /= (common_sum * common_sum);
                dL_dV_ptr[idx] = dL_dV_ptr_idx;

            } else if (X_ptr[stride * idx + 1] < 1.5) {
                dL_dC_ptr[idx] = 0.0;
                dL_dV_ptr[idx] = 0.0;
            }
        }
    }

    void run_forward_kernel(float *X_ptr, int x_dims, int y_dims) {
        int num_elements = x_dims * y_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;
        forward_kernel<<<gridSize, blockSize>>>(X_ptr, x_dims, y_dims);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::stringstream strstr;
            strstr << "run_kernel launch failed" << std::endl;
            strstr << "numThreads: " << numThreads << std::endl;
            strstr << "dimBlock: " << blockSize << std::endl;
            strstr << "dimGrid: " << gridSize << std::endl;
            strstr << cudaGetErrorString(error);
            throw strstr.str();
        }
    }

    void forward(float *X_ptr, int x_dims, int y_dims) {
        run_forward_kernel(X_ptr, x_dims, y_dims);
    }

    void
    run_backward_kernel(const float *const X_ptr, const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr,
                        int x_dims, int y_dims) {
        int num_elements = x_dims * y_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;
        backward_kernel<<<gridSize, blockSize>>>(X_ptr, dL_dout_ptr, dL_dV_ptr, dL_dC_ptr, x_dims, y_dims);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::stringstream strstr;
            strstr << "run_kernel launch failed" << std::endl;
            strstr << "numThreads: " << numThreads << std::endl;
            strstr << "dimBlock: " << blockSize << std::endl;
            strstr << "dimGrid: " << gridSize << std::endl;
            strstr << cudaGetErrorString(error);
            throw strstr.str();
        }
    }

    void
    backward(const float *const X_ptr, const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr, int x_dims,
             int y_dims) {
        run_backward_kernel(X_ptr, dL_dout_ptr, dL_dV_ptr, dL_dC_ptr, x_dims, y_dims);
    }

}