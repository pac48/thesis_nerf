#include <sstream>
#include <iostream>
#include "liblaplacian.hpp"

namespace internal {
    constexpr int blockSize = 256;

    __global__ void kernel(float *vec, int x_dims, int y_dims) {
        __shared__ float s[4 * blockSize];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % x_dims;
        int ind_y = idx / x_dims;
        constexpr int stride = 3;

        if (ind_x > 0 && ind_x < x_dims - 1 && ind_y > 0 && ind_y < y_dims - 1) {

            const int left_idx = stride * idx - stride;
            const int up_idx = stride * idx - stride * x_dims;
            const int right_idx = stride * idx + stride;
            const int down_idx = stride * idx + stride * x_dims;

            const int up_left_idx = up_idx - stride;
            const int up_right_idx = up_idx + stride;
            const int down_left_idx = down_idx - stride;
            const int down_right_idx = down_idx + stride;

            const unsigned int shared_idx = 4 * (threadIdx.x % blockSize);
            s[shared_idx] = vec[up_idx];
            s[shared_idx + 1] = vec[left_idx];
            s[shared_idx + 2] = vec[right_idx];
            s[shared_idx + 3] = vec[down_idx];

//            float val = min(min(vec[up_idx], vec[down_idx]), min(vec[left_idx], vec[right_idx])) + .1f;
//            float val_diag =
//                    min(min(vec[up_left_idx], vec[up_right_idx]), min(vec[down_left_idx], vec[down_right_idx])) + .141f;
//            val = min(val, val_diag);

            float val = vec[up_idx] * exp(-vec[up_idx]) + vec[down_idx] * exp(-vec[down_idx]) +
                        vec[left_idx] * exp(-vec[left_idx]) + vec[right_idx] * exp(-vec[right_idx]) +
                        vec[up_left_idx] * exp(-vec[up_left_idx]) + vec[up_right_idx] * exp(-vec[up_right_idx]) +
                        vec[down_left_idx] * exp(-vec[down_left_idx]) + vec[down_right_idx] * exp(-vec[down_right_idx]);
            val = val / (exp(-vec[up_idx]) + exp(-vec[down_idx]) + exp(-vec[left_idx]) + exp(-vec[right_idx]) +
                         exp(-vec[up_left_idx]) + exp(-vec[up_right_idx]) + exp(-vec[down_left_idx]) +
                         exp(-vec[down_right_idx]));
            val += .005;

            __syncthreads();
//            4*u(x,y) = u(x-1,y)+u(x+1,y)+u(x,y+1)+u(x,y-1);
//            0 = u(x-1,y)+u(x+1,y)+u(x,y+1)+u(x,y-1) - 4*u(x,y);
//            0 = nabla u(x,y)
            if (vec[stride * idx + 1] < 0.5) {
                // no constraint
//                vec[stride * idx] = 0.25 * (s[shared_idx] + s[shared_idx + 1] + s[shared_idx + 2] + s[shared_idx + 3]);
                vec[stride * idx] = val;
            } else if (vec[stride * idx + 1] < 1.5) {
                // temperature constraints
                vec[stride * idx] = vec[stride * idx + 2];
            } else if (vec[stride * idx + 1] < 2.5) {
                // flux x constraints
                // T[n]-T[n-1]/dx = q
                // T[n+1]-T[n-1]/2dx = q
                float &flux = vec[stride * idx + 2];
                if (ind_x > 1) {
                    vec[stride * idx] = s[shared_idx + 1] + flux;
                } else {
                    vec[stride * idx] = s[shared_idx + 2] + flux;
                }
            } else if (vec[stride * idx + 1] < 3.5) {
                // flux y constraints
                float &flux = vec[stride * idx + 2];
                if (ind_y > 1) {
                    vec[stride * idx] = s[shared_idx] + flux;
                } else {
                    vec[stride * idx] = s[shared_idx + 3] + flux;
                }
            } else if (vec[stride * idx + 1] < 4.5) {
                // source/sink constraints
                float &flux = vec[stride * idx + 2];
                vec[stride * idx] =
                        0.25 * (s[shared_idx] + s[shared_idx + 1] + s[shared_idx + 2] + s[shared_idx + 3] + flux);
            }
        }
    }

    void run_kernel(float *vec, int x_dims, int y_dims) {
        int num_elements = x_dims * y_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(vec, x_dims, y_dims);

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

    void solve(float *data_ptr, int x_dims, int y_dims) {
        run_kernel(data_ptr, x_dims, y_dims);
    }
}