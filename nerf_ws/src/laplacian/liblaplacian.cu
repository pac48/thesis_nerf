#include <sstream>
#include <iostream>
#include "liblaplacian.hpp"

namespace internal {
    constexpr int blockSize = 256;

    template<int SIZE>
    __global__ void
    forward_kernel(float *X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr) {
        constexpr int stride = 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % x_dims;
        int ind_y = idx / x_dims;
        int ind_z = ind_y / (y_dims);

        if (ind_x > 0 && ind_x < x_dims - 1 && ind_y > 0 && ind_y < y_dims - 1 &&
            (z_dims == 1 || (ind_z > 0 && ind_z < z_dims - 1))) {
            const int index_offset = stride * idx;
            double offset = 0;

#pragma unroll
            for (int i = 0; i < SIZE; i++) {
                const int index = index_offset + stride * indexes_ptr[i];
                offset += X_ptr[index];
            }
            offset = offset / SIZE;

            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < SIZE; i++) {
                const int index = index_offset + stride * indexes_ptr[i];
                numerator += X_ptr[index] * exp(offset - X_ptr[index]);
                denominator += exp(offset - X_ptr[index]);
            }
            float &cost = X_ptr[index_offset + 3];
            numerator = (numerator / denominator) + cost;

            __syncthreads();

            if (X_ptr[stride * idx + 1] < 0.5) {
                X_ptr[stride * idx] = numerator;
            } else if (X_ptr[stride * idx + 1] < 1.5) {
                X_ptr[stride * idx] = X_ptr[stride * idx + 2];
            }
        }
    }

    template<int SIZE>
    __global__ void
    backward_kernel(const float *const X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr,
                    const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr) {
        constexpr int stride = 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % x_dims;
        int ind_y = idx / x_dims;
        int ind_z = ind_y / (y_dims);


        if (ind_x > 0 && ind_x < x_dims - 1 && ind_y > 0 && ind_y < y_dims - 1 &&
            (z_dims == 1 || (ind_z > 0 && ind_z < z_dims - 1))) {

            if (X_ptr[stride * idx + 1] < 0.5) {

                float dout_dC = 1;
                dL_dC_ptr[idx] = dout_dC * dL_dout_ptr[idx];

                const int index_offset = stride * idx;
                double offset = 0;
#pragma unroll
                for (int i = 0; i < SIZE; i++) {
                    const int index = index_offset + stride * indexes_ptr[i];
                    offset += X_ptr[index];
                }
                offset = offset / SIZE;

                double dL_dV_ptr_idx = 0;
                double common_sum = 0;
#pragma unroll
                for (int i = 0; i < SIZE; i++) {
                    const int index = index_offset + stride * indexes_ptr[i];
                    common_sum += exp(offset - X_ptr[index]);
                }
#pragma unroll
                for (int i = 0; i < SIZE; i++) {
                    const int index = index_offset + stride * indexes_ptr[i];
                    double tmp = (-X_ptr[index] + 1.0) * common_sum * exp(offset - X_ptr[index]);
#pragma unroll
                    for (int j = 0; j < SIZE; j++) {
                        const int index2 = index_offset + stride * indexes_ptr[j];
                        tmp += X_ptr[index] * exp(2 * offset - X_ptr[index] - X_ptr[index2]);
                    }
                    dL_dV_ptr_idx += (tmp / (common_sum * common_sum)) * dL_dout_ptr[index / stride];
                }

                dL_dV_ptr[idx] = dL_dV_ptr_idx;

            } else if (X_ptr[stride * idx + 1] < 1.5) {
                dL_dC_ptr[idx] = 0.0;
                dL_dV_ptr[idx] = 0.0;
            }
        }
    }

    void run_forward_kernel(float *X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr,
                            int indexes_len) {

        int num_elements = x_dims * y_dims * z_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;
        if (indexes_len == 8) {
            forward_kernel<8><<<gridSize, blockSize>>>(X_ptr, x_dims, y_dims, z_dims, indexes_ptr);
        } else if (indexes_len == 26) {
            forward_kernel<26><<<gridSize, blockSize>>>(X_ptr, x_dims, y_dims, z_dims, indexes_ptr);
        } else {
            std::stringstream strstr;
            strstr << "only 2d and 3d is supported" << std::endl;
            throw strstr.str();
        }

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

    void forward(float *X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr, int indexes_len) {
        run_forward_kernel(X_ptr, x_dims, y_dims, z_dims, indexes_ptr, indexes_len);
    }

    void
    run_backward_kernel(const float *const X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr,
                        int indexes_len, const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr) {
        int num_elements = x_dims * y_dims * z_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;

        if (indexes_len == 8) {
            backward_kernel<8><<<gridSize, blockSize>>>(X_ptr, x_dims, y_dims, z_dims, indexes_ptr,
                                                        dL_dout_ptr, dL_dV_ptr, dL_dC_ptr);
        } else if (indexes_len == 26) {
            backward_kernel<26><<<gridSize, blockSize>>>(X_ptr, x_dims, y_dims, z_dims, indexes_ptr,
                                                         dL_dout_ptr, dL_dV_ptr, dL_dC_ptr);
        } else {
            std::stringstream strstr;
            strstr << "only 2d and 3d is supported" << std::endl;
            throw strstr.str();
        }


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
    backward(const float *const X_ptr, int x_dims, int y_dims, int z_dims, const int *const indexes_ptr,
             int indexes_len, const float *const dL_dout_ptr, float *dL_dV_ptr, float *dL_dC_ptr) {
        run_backward_kernel(X_ptr, x_dims, y_dims, z_dims, indexes_ptr, indexes_len, dL_dout_ptr, dL_dV_ptr,
                            dL_dC_ptr);
    }

}