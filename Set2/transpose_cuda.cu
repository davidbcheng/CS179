#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

/**
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
*/


/**
 * Each block of the naive transpose handles a 64x64 submatrix of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 submatrix of shape (32, 4), then we have
 * a matricies of shape (2 blocks, 16 blocks).
 * Warp 0 handles submatrix (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */

/**
 * In this naive approach there is a problem with non coalesced memory accesses
 * The problems occur with line "output[j + n * i] = input[i + n * j];"
 * 
 * We can break this line up into two instructions: the loading of the element
 * in input and the writing of the element in output.
 *
 * Input (Good):
 * Each thread in a given warp has the same threadIdx.y, so at one given
 * instruction, all the threads will have the same threadIdx.y. However,
 * threads in the warp have threadIdx.x values of 0, 1, 2, ..., 31 respectively.
 * This means that for the load of "input[i + n * j]". Each thread will have
 * the same j value and i values corresponding to whichever threadIdx.x value
 * it has. Thus, this is stride 1, and will only stretch along 32 elements
 * sequentially in input. Since each element is a float, this will stretch
 * across 128 bytes and thus will only take up one cache line, thereby using
 * coalesced memory. The increments of j do not change this because we are only
 * interested in one given instruction at a time and j is constant within a
 * given instruction
 *
 * Output (bad):
 * Same as before, j is consistent throughout threads, but i has corresponding
 * values of 0, 1, 2, ..., 31 respectively depending on block number.
 * A problem occurs with "output[j + n * i]". Assume without loss of generality
 * that blockIdx.x is 0. Thus, i ranges from 0 to 31. Therefore, the elements
 * of output range from output[j + 0] to output[j + 31 * n]. Assuming n is
 * 64, we have a range that is spread across 1985 elements, which corresponds
 * to 7940 bytes. This turns out to be 62.03 cache lines, so we need to load in
 * 63 cache lines, which is non coalesced memory access

 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
  // TODO: do not modify code, just comment on suboptimal accesses

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
  // TODO: Modify transpose kernel to use shared memory. All global memory
  // reads and writes should be coalesced. Minimize the number of shared
  // memory bank conflicts (0 bank conflicts should be possible using
  // padding). Again, comment on all sub-optimal accesses.

  __shared__ float data[4160];

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  int data_i = i % 64;
  int data_j = j % 64;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    data[data_i + 65 * data_j] = input[i + n * j];
    data_j += 1;
  }

  __syncthreads();
  j -= 4;
  data_j -= 4;

  for (; j < end_j; j++) {
    // output[i + n * j] = data[data_j + 65 * data_i];
    output[i + n * j] = data[data_i + 65 * data_j];
    data_j += 1;
  }
  
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  // TODO: This should be based off of your shmemTransposeKernel.
  // Use any optimization tricks discussed so far to improve performance.
  // Consider ILP and loop unrolling.

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

void cudaTranspose(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type) {
  if (type == NAIVE) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == SHMEM) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == OPTIMAL) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else {
    // unknown type
    assert(false);
  }
}