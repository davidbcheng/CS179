/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Blur_cuda.cuh"


__global__
void
cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int N, int blur_v_size) {

    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < N) {
        float result = 0;

        // Similar to the CPU implementation, split into two cases
        if (index < blur_v_size) {
            for (int j = 0; j <= index; j++) {
                result += raw_data[index - j] * blur_v[j]; 
            }
        }
        else {
            for (int j = 0; j < blur_v_size; j++) {
                result += raw_data[index - j] * blur_v[j]; 
            }
        }

        // After result is computed, store it in global memory
        out_data[index] = result;

        // Compute next index for arbitrary amount of threads
        index += blockDim.x * gridDim.x;
    }
}


void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int N,
        const unsigned int blur_v_size) {
        
    /* Call the kernel with the specified parameters*/
    cudaBlurKernel<<<blocks, threadsPerBlock>>> (raw_data, blur_v, out_data,
        N, blur_v_size);
}
