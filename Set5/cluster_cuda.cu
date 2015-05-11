#include <cassert>
#include <cuda_runtime.h>
#include "cluster_cuda.cuh"
#include <stdio.h>
#include <float.h>

// This assumes address stores the average of n elements atomically updates
// address to store the average of n + 1 elements (the n elements as well as
// val). This might be useful for updating cluster centers.
// modified from http://stackoverflow.com/a/17401122
__device__ 
float atomicUpdateAverage(float* address, int n, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    float next_val = (n * __int_as_float(assumed) + val) / (n + 1);
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(next_val));
  } while (assumed != old);
  return __int_as_float(old);
}

// computes the distance squared between vectors a and b where vectors have
// length size and stride stride.
__device__ 
float squared_distance(float *a, float *b, int stride, int size) {
  float dist = 0.0;
  for (int i=0; i < size; i++) {
    float diff = a[stride * i] - b[stride * i];
    dist += diff * diff;
  }
  return dist;
}

/*
 * Notationally, all matrices are column majors, so if I say that matrix Z is
 * of size m * n, then the stride in the m axis is 1. For purposes of
 * optimization (particularly coalesced accesses), you can change the format of
 * any array.
 *
 * clusters is a REVIEW_DIM * k array containing the location of each of the k
 * cluster centers.
 *
 * cluster_counts is a k element array containing how many data points are in 
 * each cluster.
 *
 * k is the number of clusters.
 *
 * data is a REVIEW_DIM * batch_size array containing the batch of reviews to
 * cluster. Note that each review is contiguous (so elements 0 through 49 are
 * review 0, ...)
 *
 * output is a batch_size array that contains the index of the cluster to which
 * each review is the closest to.
 *
 * batch_size is the number of reviews this kernel must handle.
 */
__global__
void sloppyClusterKernel(float *clusters, int *cluster_counts, int k, 
                          float *data, int *output, int batch_size) {
  // Compute which index we are accessing in our output depending on our 
  // thread index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Until we reach end of the data size
  while (index < batch_size)
  {
    // Find minimum distance, so set min distance to max first
    float min_dist = FLT_MAX;
    int clusterIndex = -1;

    // Find the current review data
    float *currRev = data + index * REVIEW_DIM;

    // For all the center points, find the distance and update the min distance
    // and the min cluster center index
    for(int i = 0; i < k; i++)
    {
      float * currCenter = clusters + i * REVIEW_DIM;
      float currDist = squared_distance(currRev, currCenter, 1, REVIEW_DIM);
      if(currDist < min_dist)
      {
        min_dist = currDist;
        clusterIndex = i;
      }
    } 

    // Set the review to the minimum distance cluster. 
    output[index] = clusterIndex; 
    
    // Update all the dimensions for the center with the updated point
    for(int i = 0; i < REVIEW_DIM; i++)
    {
      float * centerDim = clusters + i + clusterIndex * REVIEW_DIM;
      float * update = data + i + clusterIndex * REVIEW_DIM;
      atomicUpdateAverage(centerDim, cluster_counts[clusterIndex], *update);
    }

    // Add 1 to the number of points that the cluster has
    atomicAdd(cluster_counts + clusterIndex, 1);

    // Handle arbitrary amount of threads
    index += blockDim.x * gridDim.x;
  } 
}


void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = 0;

  sloppyClusterKernel<<<
    block_size, 
    grid_size, 
    shmem_bytes, 
    stream>>>(clusters, cluster_counts, k, data, output, batch_size);
}
