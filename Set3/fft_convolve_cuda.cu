/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* shared, int tid)
{
    if(blockSize >= 64) shared[tid] = fmaxf(shared[tid], shared[tid + 32]);
    if(blockSize >= 32) shared[tid] = fmaxf(shared[tid], shared[tid + 16]);
    if(blockSize >= 16) shared[tid] = fmaxf(shared[tid], shared[tid + 8]);
    if(blockSize >=  8) shared[tid] = fmaxf(shared[tid], shared[tid + 4]);
    if(blockSize >=  4) shared[tid] = fmaxf(shared[tid], shared[tid + 2]);
    if(blockSize >=  2) shared[tid] = fmaxf(shared[tid], shared[tid + 1]);
}


__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < padded_length)
    {
        out_data[index].x = (raw_data[index].x * impulse_v[index].x -
            raw_data[index].y * impulse_v[index].y) / padded_length;

        out_data[index].y = (raw_data[index].x * impulse_v[index].y +
            raw_data[index].y * impulse_v[index].x) / padded_length;

        // Compute next index for arbitrary amount of threads
        index += blockDim.x * gridDim.x;
    }


}

template <unsigned int blockSize>
__global__ void cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;


    shared[tid] = 0.0;
    while(i + blockDim.x < padded_length)
    {
        float max = fmaxf(abs(out_data[i].x), abs(out_data[i + blockDim.x].x));
        shared[tid] = fmaxf(shared[tid], max);
        // Compute next index for arbitrary amount of threads
        i += (2 * blockDim.x) * gridDim.x;
    }
    
    __syncthreads();

    // if (blockSize >= 512) 
    // {
    //     if (tid < 256) 
    //     {   
    //         shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 256]));
    //     }
    //     __syncthreads(); 
    // }

    // if (blockSize >= 256)
    // {
    //     if (tid < 128) 
    //     {
    //         shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 128]));
    //     }
    //     __syncthreads(); 
    // }

    // if (blockSize >= 128)
    // {
    //     if (tid < 64)
    //     {
    //         shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 64]));
    //     }    
    //     __syncthreads(); 
    // }

    if (blockSize >= 512) { if (tid < 256) { shared[tid] = fmaxf(shared[tid], shared[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { shared[tid] = fmaxf(shared[tid], shared[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { shared[tid] = fmaxf(shared[tid], shared[tid + 64]); } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(shared, tid);

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}


__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while(index < padded_length)
    {
        out_data[index].x = out_data[index].x / *(max_abs_val);
        // Compute next index for arbitrary amount of threads
        index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>> (raw_data, impulse_v,
        out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    unsigned int sharedSize = threadsPerBlock * sizeof(float);
    switch(threadsPerBlock)
    {
        case 512:
            cudaMaximumKernel<512><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 256:
            cudaMaximumKernel<256><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 128:
            cudaMaximumKernel<128><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 64:
            cudaMaximumKernel<64><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 32:
            cudaMaximumKernel<32><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 16:
            cudaMaximumKernel<16><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 8:
            cudaMaximumKernel<8><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 4:
            cudaMaximumKernel<4><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 2:
            cudaMaximumKernel<2><<<blocks, threadsPerBlock, sharedSize >>> (out_data, max_abs_val, padded_length); break;
        case 1:
            cudaMaximumKernel<1><<<blocks, threadsPerBlock, sharedSize >>>  (out_data, max_abs_val, padded_length); break;
    }
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>> (out_data, max_abs_val, 
        padded_length);
}
