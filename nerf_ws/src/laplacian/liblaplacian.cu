#include <sstream>
#include <iostream>
#include "liblaplacian.hpp"

namespace internal {
    constexpr int blockSize = 256;

    __global__ void kernel(float *vec, int x_dims, int num_elements) {
        __shared__ float s[4 * blockSize];
        int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        const int left_idx = idx - 2;
        const int up_idx = idx - 2 * x_dims;
        const int right_idx = idx + 2;
        const int down_idx = idx + 2 * x_dims;

        if (idx < num_elements && left_idx % (2*x_dims) > 0 && up_idx > 0  && down_idx < num_elements && vec[idx + 1] < 1.0) {

            //TODO use shared mem
            const unsigned int shared_idx = 4 * (threadIdx.x % blockSize);
            s[shared_idx] = vec[up_idx];
            s[shared_idx + 1] = vec[left_idx];
            s[shared_idx + 2] = vec[right_idx];
            s[shared_idx + 3] = vec[down_idx];
            __syncthreads();
            vec[idx] = 0.25 * (s[shared_idx] + s[shared_idx + 1] + s[shared_idx + 2] + s[shared_idx + 3]);

//            const float val =  .25*(vec[up_idx]+vec[left_idx]+vec[right_idx]+vec[down_idx]);
//            __syncthreads();
//            vec[idx] = val;

        }
    }

    void run_kernel(float *vec, int x_dims, int y_dims) {
        int num_elements = x_dims * y_dims;

        int numThreads = num_elements;
        int gridSize = (numThreads + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(vec, x_dims, 2 * num_elements);

//        dim3 dimBlock(256, 1, 1);
//        dim3 dimGrid(ceil((float) num_elements / dimBlock.x));
//        kernel<<<dimGrid, dimBlock>>>(vec, num_elements);


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