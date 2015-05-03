
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#define PI 3.14159265358979
#define EPSILON 0.00001


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}
__global__
void
cudaHighPassKernel(cufftComplex *raw_data, const int sinogram_width, const int nAngles) {
    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    const int length = nAngles * sinogram_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int center = sinogram_width / 2.0;


    while(index < length)
    {   
	int relative_dist = abs((float) (index % sinogram_width - center));
	float scalingFactor = (1.0 - (float) relative_dist / center);
        raw_data[index].x *= scalingFactor;
        raw_data[index].y *= scalingFactor;
	//printf("%f, %d\n", scalingFactor, abs(relative_index - center));
        index += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaBackProjectionKernel(float *sinogram, float *result, int size_result,
    int nAngles, int sinogram_width, int side_length) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;

    while (index < side_length * side_length) {
	int x_p = index % side_length;
	int y_p = index / side_length;
	
    	float x_0 = (float) x_p - (side_length / 2);
    	float y_0 = (side_length / 2) - (float) y_p;

   	int angle;
    	for (angle = 0; angle < nAngles; ++angle)
   	{
            float theta = ((float) angle / nAngles) * PI;
            int d;
            if (fabsf(theta) < EPSILON)
            {
                d = x_0;
            }
            else if (fabsf(theta - PI / 2) < EPSILON)
            {
                d = y_0;
            }
            else
            {
                float m = -1 * cosf(theta) / sinf(theta);
                float q = -1 / m;

                float x_1 = (y_0 - m * x_0) / (q - m);
                float y_1 = q * x_1;
                d = floorf(sqrtf(x_1 * x_1 + y_1 * y_1));

                if ((q > 0 && x_1 < 0) || (q < 0 && x_1 > 0))
                {
                    d = -1 * d;
                }
            }
            result[index] += sinogram[d + sinogram_width / 2 + sinogram_width * angle];
        }
	index += blockDim.x * gridDim.x;
    }
}

__global__
void cudaConvertToReal(cufftComplex * c_nums, float * r_nums, int length) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    while(index < length)
    {
        r_nums[index] = c_nums[index].x;
	index += blockDim.x * gridDim.x;
    }    
}

int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */

    cudaMalloc((void **) &dev_sinogram_cmplx, sinogram_width*nAngles*sizeof(cufftComplex));
    cudaMalloc((void **) &dev_sinogram_float, sinogram_width*nAngles*sizeof(float));

    cudaMemcpy(dev_sinogram_cmplx, sinogram_host,
        sizeof(cufftComplex) * sinogram_width * nAngles, cudaMemcpyHostToDevice);



    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    cufftHandle plan;
    int batch = nAngles;
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, batch);

    printf("Loaded Data\n");

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    printf("Completed FFT_forward\n");

    cudaHighPassKernel<<<nBlocks, threadsPerBlock>>> (dev_sinogram_cmplx,
        sinogram_width, nAngles);

    printf("Completed HighPass FIlter\n");

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);

    printf("Completed FFT INVERSe\n");

    cudaConvertToReal<<<nBlocks, threadsPerBlock>>> (dev_sinogram_cmplx, dev_sinogram_float,
	sinogram_width * nAngles);

    printf("Converted to Real\n");

    cufftDestroy(plan);

    cudaFree(dev_sinogram_cmplx);
    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */

    cudaMalloc((void **) &output_dev, size_result);
    cudaMemset(output_dev, 0, size_result);

    cudaBackProjectionKernel <<<nBlocks, threadsPerBlock>>> (dev_sinogram_float,
     output_dev, size_result, nAngles, sinogram_width, height);

    cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost);
    printf("Completed Back Projection\n");
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);

    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



