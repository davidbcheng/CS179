#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

float hostToDeviceTime = -1;
float kernelTime = -1;
float deviceToHostTime = -1;

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
void readLSAReview(string review_str, float *output) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
  int review_idx_start;
  int batch_size;
  int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
  printerArg *arg = static_cast<printerArg *>(userData);

  for (int i=0; i < arg->batch_size; i++) {
    printf("%d: %d\n", 
	   arg->review_idx_start + i, 
	   arg->cluster_assignments[i]);
  }

  delete arg;
}

void cluster(int k, int batch_size) {
  // cluster centers
  float *d_clusters;

  // how many points lie in each cluster
  int *d_cluster_counts;

  // allocate memory for cluster centers and counts
  gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

  // randomly initialize cluster centers
  float *clusters = new float[k * REVIEW_DIM];
  gaussianFill(clusters, k * REVIEW_DIM);
  gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
		       cudaMemcpyHostToDevice));

  // initialize cluster counts to 0
  gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));
  
  // Allocate space and array for the two input buffers for host and device
  // Each one is of size batch_size * REVIEW_DIM because we want batch_size
  // number of review for each input
  float * data = new float[batch_size * REVIEW_DIM];
  float * data1 = new float[batch_size * REVIEW_DIM];

  float * d_data;
  float * d_data1;

  gpuErrChk(cudaMalloc(&d_data, batch_size * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_data1, batch_size * REVIEW_DIM * sizeof(float)));

  // Allocate space for two output buffers for host and device each
  // output is a batch_size array that contains the index of the cluster to which
  // each review is the closest to.
  int * output = new int[batch_size];
  int * output1 = new int[batch_size];

  int * d_output;
  int * d_output1;

  gpuErrChk(cudaMalloc(&d_output, batch_size * sizeof(int)));
  gpuErrChk(cudaMalloc(&d_output1, batch_size * sizeof(int)));

  // Create two streams to parallelize code further
  cudaStream_t s[2];
  cudaStreamCreate(&s[0]); cudaStreamCreate(&s[1]);

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  for (string review_str; getline(cin, review_str); review_idx++) {
    // Use readLSAReview to read review into our host input buffer
    // If it is in the first half of batch_size * 2 then we write it to the
    // first input buffer
    // If it is in the second half of the batch_size * 2, then we write it
    // to the second input buffer
    if(review_idx % (batch_size * 2) < batch_size)
    {
       readLSAReview(review_str, data + REVIEW_DIM * (review_idx % batch_size));
    }
    else
    {
       readLSAReview(review_str, data1 + REVIEW_DIM * (review_idx % batch_size));
    }

    // Once both input buffers have filled, with two batches
    if((review_idx + 1) % (batch_size * 2) == 0)
    {
      // Want to block until the stream finished all its operations
      cudaStreamSynchronize(s[0]);

      // START_TIMER();

      // Want to copy data from first input buffer onto GPU, then run the 
      // kernel to compute sloppy k means clustering on the first input data
      // buffer, then copy computed output from gpu to cpu.
      cudaMemcpyAsync(d_data, data, REVIEW_DIM * batch_size * sizeof(float),
        cudaMemcpyHostToDevice, s[0]);

      // For timing
      // cudaStreamSynchronize(s[0]);
      // STOP_RECORD_TIMER(hostToDeviceTime);
      // START_TIMER();

      cudaCluster(d_clusters, d_cluster_counts, k, d_data,
        d_output, batch_size, s[0]);

      cudaStreamSynchronize(s[0]);

      // For Timing
      // STOP_RECORD_TIMER(kernelTime);
      // START_TIMER();

      cudaMemcpyAsync(output, d_output, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, s[0]);

      // For Timing
      // cudaStreamSynchronize(s[0]);
      // STOP_RECORD_TIMER(deviceToHostTime);

      // Check indicies of output using printerArgs
      printerArg * args = new printerArg;   

      args->review_idx_start = review_idx - batch_size * 2 + 1;
      args->batch_size = batch_size;
      args->cluster_assignments = output;

      // Use cudaStreamAddCallBack which calls the printerCallback function
      // which is called on host after stream is done
      cudaStreamAddCallback(s[0], printerCallback, (void*) args, 0);

      // Want to block until stream finished all its operations
      cudaStreamSynchronize(s[1]);

      // Want to copy data from first input buffer onto GPU, then run the 
      // kernel to compute sloppy k means clustering on the first input data
      // buffer, then copy computed output from gpu to cpu.      
      cudaMemcpyAsync(d_data1, data1, REVIEW_DIM * batch_size * sizeof(float),
        cudaMemcpyHostToDevice, s[1]);
      cudaCluster(d_clusters, d_cluster_counts, k, d_data1,
        d_output1, batch_size, s[1]);
      cudaMemcpyAsync(output1, d_output1, batch_size * sizeof(int),
        cudaMemcpyDeviceToHost, s[1]);

      // Check indicies of output using printerArgs
      printerArg * args1 = new printerArg;

      args1->review_idx_start = review_idx - batch_size + 1;
      args1->batch_size = batch_size;
      args1->cluster_assignments = output1;

      // Use cudaStreamAddCallBack which calls the printerCallback function
      // which is called on host after stream is done
      cudaStreamAddCallback(s[1], printerCallback, (void*) args1, 0);
    }
  }

  // wait for everything to end on GPU before final summary
  gpuErrChk(cudaDeviceSynchronize());

  // retrieve final cluster locations and counts
  int *cluster_counts = new int[k];
  gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
		       cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
		       cudaMemcpyDeviceToHost));


  // print cluster summaries
  for (int i=0; i < k; i++) {
    printf("Cluster %d, population %d\n", i, cluster_counts[i]);
    printf("[");
    for (int j=0; j < REVIEW_DIM; j++) {
      printf("%.4e,", clusters[i * REVIEW_DIM + j]);
    }
    printf("]\n\n");
  }

  // free cluster data
  gpuErrChk(cudaFree(d_clusters));
  gpuErrChk(cudaFree(d_cluster_counts));
  delete[] cluster_counts;
  delete[] clusters;

  // Free Input and Output streams on host and on device
  delete[] data;
  delete[] data1;
  delete[] output;
  delete[] output1;

  gpuErrChk(cudaFree(d_data));
  gpuErrChk(cudaFree(d_data1));
  gpuErrChk(cudaFree(d_output));
  gpuErrChk(cudaFree(d_output1));

  // Destroy streams
  cudaStreamDestroy(s[0]);
  cudaStreamDestroy(s[1]);
}

int main() {
  // cluster(5, 32);
  cluster(5, 32);
  printf("Host to Device: %f\n", hostToDeviceTime);
  printf("Kernel: %f\n", kernelTime);
  printf("Device To Host: %f\n", deviceToHostTime);
  return 0;
}
