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

/**
 * warpReduce is used in MaximumKernel to find the max within a given warp. This
 * is used in the sequential addressing reduction
 */
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

/**
 * cudaProdScaleKernel reads input data, impulse data, and writes the 
 * pointwise multiplication of the two into out_data.
 *
 * We are using this pointwise multiplication within the implementation of
 * the Circular Convolution Theorem, where
 * y[n] = sum_{k=0}^{N-1} x[k] h[(n-k) mod N], where n is the input size,
 * y is the output, x is the input, and h is the impulse, can be written as
 * IFFT ( FFT(x) .* FFT(h))

 * At the point that ProdScale Kernel is called, we have already performed
 * FFT on both the input, x, and the impulse, h through cuFFT. In ProdScale
 * Kernel, we compute the the pointwise multiplication between FFT(x) and FFT(h)
 * . We also scale it by a factor of N, which is the padded_length since 
 * the output of IDFT(DFT(x)) does not return x, but it returns Nx. Thus, we 
 * have to scale by N.
 */
__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {

    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure index does not go past the length of our data
    while(index < padded_length)
    {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i

        // The real part of the output is calculated by (ac - bd)
        out_data[index].x = (raw_data[index].x * impulse_v[index].x -
            raw_data[index].y * impulse_v[index].y) / padded_length;

        // The complex part of the output is calculated by (ad + bc)
        out_data[index].y = (raw_data[index].x * impulse_v[index].y +
            raw_data[index].y * impulse_v[index].x) / padded_length;

        // Compute next index for arbitrary amount of threads
        index += blockDim.x * gridDim.x;
    }
}

/**
 * The MaximumKernel takes in the out_data and a pointer to memory, max_abs_val.
 * The kernel is responsible for checking each value of the out_data and 
 * finding the maximum value and saving that value into max_abs_val. 
 *
 * Reduction and Optimization Strategies ("Optimizing Parallel Reduction in CUDA):
 * 
 * Tree-based approach:
 * We want to find the maximum value by checking all the values in out_data.
 * The tree-based approach is that at each level we check pairs of values and
 * taking the max of each pair. Thus, at the next level we can just pairs
 * the max of each of the first pairs, and so on. This way we are dividing 
 * the number of elements we are checking on each level by 2. This is much
 * better than the serial version that you might find on the CPU, which is
 * to keep a max pointer and update it as we find larger values through the data.
 * because we can parallelize each level. We will have log(n) levels where n
 * is the the number of elements
 *
 * Sequential Addressing:
 * We implement sequential addressing with the tree-based approach to minimize
 * divergence and bank conflicts. This entails splitting the data that is in 
 * the block into two halves. Each thread finds the max between "shared[threadId]"
 * value and the corresponding value in the second half. For example, if the
 * block size is 512, the first thread in the warp will find the max between
 * "shared[0]" and "shared[0 + 512 / 2]" and stores it in "shared[0]"
 * . The next step, since the maximum value is bound to be within the first half
 * we can repeat the same procedure of splitting it in a half and finding the
 * max between the two halves. Thus, as explained in Tree based approch, we
 * are recursively dividing the data by 2 each iteration. Since the threads is
 * accessing the data sequentially. Thread 0 will check shared[0], 1 will check
 * shared[1], 2 will check shared[2], ... Or they will check the corresponding
 * value in the second half of the data sequentially. Thus, this will prevent
 * any bank conflicts because it is stride 1. We are also minimizing divergence,
 * since we are each thread the index of the thread (using thread 0 , 1, 2...)
 * , and utilizing each thread at each level, so each warp is not blocked
 * with thread divergence
 *
 * Loop Unrolling:
 * We want to limit instruction overhead. Thus, we will unloop all the for loops
 * to have less checks and variables. We know the block size is at most 512
 * threads as specified by the GPU. We are also assuming power of 2 block sizes.
 * Thus, we can unroll for a specific block size. Instead of iteratively
 * dividing the size by 2 in sequential Addressing, we have a set amount of 
 * instructions, having the same logic, but hopefully faster. We can also
 * use templates to have an arbitrary sized blockSize defined by user that
 * we dont know at compile time.
 *
 * Multiple block checks when loading into shared memory:
 * We could possibly perform this tree approach within each block level and
 * work with global memory. However, this is inefficient. To efficiently
 * load the data into shared memory from global memory, for each corresponding
 * thread index that is executing the code, we consider the elements with
 * that index in two blocks at a time. We compare the two and set the element
 * in shared memory at that index to be the larger of the two. We repeat 
 * this process until all the blocks have been read. 
 */
template <unsigned int blockSize>
__global__ void cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    // Declare shared memory
    extern __shared__ float shared[];

    // tid is the threadId in the current warp
    unsigned int tid = threadIdx.x;

    // i is the index overall within global memory
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    // gridSize is the size of the blockSize * the number of grids * 2 
    // because we are checking two blocks at once
    unsigned int gridSize = 2 * blockSize * gridDim.x;

    shared[tid] = 0.0;
    // Multiple block checks when loading into shared memory:
    // We check each pair of blocks at once and then store max into shared
    while(i + blockSize < padded_length)
    {
        // We are using abs to find the highest amplitude regardless of sign
        float max = fmaxf(abs(out_data[i].x), abs(out_data[i + blockSize].x));
        shared[tid] = fmaxf(shared[tid], max);
        // Compute next index for arbitrary amount of threads
        i += gridSize;
    }
    
    // Block level sync with all threads
    __syncthreads();

    /* Checking each block size if its within the range, then we 
     * Find the max between the first half and the second half and put
     * the max in the first half. Then we do the same with the first half
     * Note assuming blockSize is multiple of two
     */
    if (blockSize >= 512) 
    {
        if (tid < 256) 
        {   
            shared[tid] = fmaxf(shared[tid], shared[tid + 256]);
        }
        __syncthreads(); 
    }

    if (blockSize >= 256)
    {
        if (tid < 128) 
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + 128]);
        }
        __syncthreads(); 
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + 64]);
        }    
        __syncthreads(); 
    }

    // If tid < 32, this is within a given warp, so call warpReduce
    if (tid < 32) warpReduce<blockSize>(shared, tid);

    // The max is contained in the first element, so we use atomic Max 
    // to set the max_abs_val value to shared[0]
    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/**
 * cudaDivideKernel is called after the maximum amplitude value is found.
 * It is responsible for normalizing the data by dividing all the data by
 * the maximum amplitude. This allows decreasing the magnitude of each data 
 * point within the range of [-1, 1]. This is better than clipping values
 * over 1 or under -1.
 */
__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    // Determine the index of the output data we are writing to by
    // the block id and the thread id    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while(index < padded_length)
    {
        // Normalize data by max amplitude
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
    // Calling the Maximum Kernel to find max amplitude signal    

    // Shared memory has threadsPerBlock elements
    unsigned int sharedSize = threadsPerBlock * sizeof(float);

    // Pass in blockSize into cudaMaximumKernel
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
        
    // Calling the Divide Kernel 
    cudaDivideKernel<<<blocks, threadsPerBlock>>> (out_data, max_abs_val, 
        padded_length);
}
