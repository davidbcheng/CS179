#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

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
  // Calculate where in matrix we are tranposing
  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  // Read from input (i, j) and write to output (j, i)
  for (; j < end_j; j++) {
  	// Input is stride 1 accessing global memory and is doing a coalesced
  	// memory access
  	// Output is stride n accessing global memory and is doing non coalesced
  	// memory access
    output[j + n * i] = input[i + n * j];
  }
}

/**
 * The idea behind shared memory is that it is much faster to access than
 * global memory and all the threads in a given block can read and write to 
 * it. Thus, if we read and write from shared memory, we dont have to worry
 * about accessing the slower global memory and reading / writing coalesced 
 * memory. However, we do have to worry about bank conflicts, which is when
 * in a given instruction, two threads access two different addresses that map
 * to the same index in the 32 slotted bank in shared memory

 * The basics of the algorithm is that each warp copies over a 32 by 4 submatrix
 * into the relative position of the shared memory. After this, we will write
 * to different parts of the global output array, so we want all the input
 * that we want to tranpose in our shared memory. Thus, we will call syncthreads
 * which is a block level command that will wait for all the threads in a given
 * block to reach the syncthreads line before continuing. Then, we will 
 * read from shared memory into the output in the same 32 by 4 submatrix fashion
 * but with with the block coordinates swapped. 

 * Optimization Notes
 * We are using only stride 1 to read from input, where each warp stretches
 * over 128 bytes, thereby reading from only one cache line and having coalesced
 * access. Similarly, we are only using stride 1 to write into output.
 *
 * To avoid bank conflicts, we are using a 65 by 64 array. To write each row
 * tranposed into a column, we increment by 65. By using a stride 65, we have
 * that stride gcd(65, 32) == 1, so we avoid bank conflicts.
 */
__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
  // Allocate shared memory for 65 x 64 element array
  // 65 is for padding, so we can avoid bank conflicts (gcd(65, 32) = 1)
  __shared__ float data[4160];

  // Compute indicies for reading from input
  const int in_i = threadIdx.x + 64 * blockIdx.x;
  int in_j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_in_j = in_j + 4;

  // Compute relative position in shared memory
  int shared_i = threadIdx.x;
  int shared_j = in_j % 64;

  // Copy over input into shared memory in the same position
  // Note that in a given instruction j is constant and i is stride one
  // from 0 to 31. Thus each instruction will read from global memory in one
  // cache line and thus be coalesced memory access
  for (; in_j < end_in_j; in_j++) {
    data[shared_i + 65 * shared_j] = input[in_i + n * in_j];
    shared_j += 1;
  }

  // Decrement j back to original shared j value
  shared_j -= 4;

  __syncthreads();

  // We want to write the shared memory into the tranposed block. Thus,
  // we swap the blockIdx.x with the blockIdx.y and compute the relative 
  // position for writing to output
  int out_i = threadIdx.x + 64 * blockIdx.y;
  int out_j = 4 * threadIdx.y + 64 * blockIdx.x;
  int end_out_j = out_j + 4;

  // We are writing from shared memory to output
  // The data in shared memory is read from in stride 65, so in the bank,
  // this is essentially a stride 1. Thus, we avoid bank conflicts
  // In addition, out_i is varied from some value x + 0 to x + 31, while
  // out_j is constant. Thus, we are writing to output in one cache line
  // Thus, our memory is coalesced.
  for (; out_j < end_out_j; out_j++) {
    output[out_i + n * out_j] = data[shared_j + 65 * shared_i];
    shared_j += 1;
  }
  
}


/**
 * We will be updated our shmemTranposeKernel code, but optimized.
 * The two big optimizations were ILP and loop unrolling.
 * 
 * For ILP, we moved all the reads before their dependencies, so that
 * we could parallelize them without scheduling more IO.
 *
 * For loop unrolling, we expanded each for loop into 4 different lines
 * with each one covering an iteration of the for loop.
 * 
 * This provided a 10% increase in performance from the shmem GPU. 
 * It is faster than GPU memcpy for size 512 size. and around 1.1 times
 * the time for GPU memcpy for the other sizes

 */
__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  // Allocate shared memory for 65 x 64 element array
  // 65 is for padding, so we can avoid bank conflicts (gcd(65, 32) = 1)
  __shared__ float data[4160];

  // Compute indicies for reading from input
  const int in_i = threadIdx.x + 64 * blockIdx.x;
  int in_j = 4 * threadIdx.y + 64 * blockIdx.y;

  // Compute relative position in shared memory
  int shared_i = threadIdx.x;
  // Division is generally than multiplication, so we replace "in_j % 64"
  // with the equivlanet "4 * threadIdx.y"
  int shared_j = 4 * threadIdx.y;

  // We want to write the shared memory into the tranposed block. Thus,
  // we swap the blockIdx.x with the blockIdx.y and compute the relative 
  // position for writing to output
  int out_i = threadIdx.x + 64 * blockIdx.y;
  int out_j = 4 * threadIdx.y + 64 * blockIdx.x;

  // Copy over input into shared memory in the same position
  // Note that in a given instruction j is constant and i is stride one
  // from 0 to 31. Thus each instruction will read from global memory in one
  // cache line and thus be coalesced memory access
  data[shared_i + 65 * (shared_j)] = input[in_i + n * (in_j)];
  data[shared_i + 65 * (shared_j + 1)] = input[in_i + n * (in_j  + 1)];
  data[shared_i + 65 * (shared_j + 2)] = input[in_i + n * (in_j + 2)];
  data[shared_i + 65 * (shared_j + 3)] = input[in_i + n * (in_j + 3)];

  __syncthreads();

  // We are writing from shared memory to output
  // The data in shared memory is read from in stride 65, so in the bank,
  // this is essentially a stride 1. Thus, we avoid bank conflicts
  // In addition, out_i is varied from some value x + 0 to x + 31, while
  // out_j is constant. Thus, we are writing to output in one cache line
  // Thus, our memory is coalesced.
  output[out_i + n * (out_j)] = data[shared_j + 65 * shared_i];
  output[out_i + n * (out_j + 1)] = data[(shared_j + 1) + 65 * shared_i];
  output[out_i + n * (out_j + 2)] = data[(shared_j + 2) + 65 * shared_i];
  output[out_i + n * (out_j + 3)] = data[(shared_j + 3) + 65 * shared_i];
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