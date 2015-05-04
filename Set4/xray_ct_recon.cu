
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

texture<float, 2, cudaReadModeElementType> texreference;

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

// The high pass filter takes in sinogram data after FFT is operated on it
// and scales each value by its distance from the center of each sinogram width
__global__
void
cudaHighPassKernel(cufftComplex *raw_data, const int sinogram_width, const int nAngles) {
    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Center of sinogram
    int center = sinogram_width / 2.0;
    // Total number of data points in data
    const int length = nAngles * sinogram_width;

    while(index < length)
    {   
        // Relative distance is the distance from the current index to the 
        // nearest center
        int relative_dist = abs((float) (index % sinogram_width - center));

        // Scaling making it so that the middle is 1 and the sides are 0, 
        // decreasing linearly
        float scalingFactor = (1.0 - (float) relative_dist / center);

        // Multiply Both Componenets
        raw_data[index].x *= scalingFactor;
        raw_data[index].y *= scalingFactor;

        // Handle arbitrary amount of threads
        index += blockDim.x * gridDim.x;
    }
}

// BackProjection takes in sinogram data and its dimensions and a 
// result float array and will return the aggregated data into the result
// array 
__global__
void
cudaBackProjectionKernel(float *sinogram, float *result, 
    int nAngles, int sinogram_width, int width, int height) {
    // Determine the index of the output data we are writing to by
    // the block id and the thread id
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    // While index < last pixel
    while (index < width * height) 
    {
        // Compute pixel dimensions
        int x_p = index % width;
        int y_p = index / width;
       
        // Convert into geometric dimensions
        float x_0 = (float) x_p - (width / 2);
        float y_0 = (height / 2) - (float) y_p;
    
        // For each angle (given point), we sum up all the values that
        // go through that point and angle
   	    int angle;
        for (angle = 0; angle < nAngles; ++angle)
   	    {
            // Equally spaced out thetas
            float theta = ((float) angle / nAngles) * PI;

            int d;
            // If Theta == 0 (prevent divison of 0)
            if (fabsf(theta) < EPSILON)
            {
                d = x_0;
            }
            // If theta == PI / 2 (prevent division of 0)
            else if (fabsf(theta - PI / 2) < EPSILON)
            {
                d = y_0;
            }
            else
            {
                // Calculate slope and recipricol 
                float m = -1 * cosf(theta) / sinf(theta);
                float q = -1 / m;

                // Calculate intersection point and distance
                float x_1 = (y_0 - m * x_0) / (q - m);
                float y_1 = q * x_1;
                d = floorf(sqrtf(x_1 * x_1 + y_1 * y_1));

                // Determine whether the distance is negative with respect
                // to the line or positive
                if ((q > 0 && x_1 < 0) || (q < 0 && x_1 > 0))
                {
                    d = -1 * d;
                }
            }

            // Increment the pixel by the data through that distance and angle
            // result[index] += sinogram[d + sinogram_width / 2 + sinogram_width * angle];
            result[index] += tex2D(texreference, d + sinogram_width / 2, angle);
        }

	   index += blockDim.x * gridDim.x;
    }
}


// cudaConvertToReal takes in a list of complex numbers, a list of type floats,
// and the length of them both and returns the real portions of the complex
// numbers into the list of type floats
__global__
void cudaConvertToReal(cufftComplex * c_nums, float * r_nums, int length) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    while(index < length)
    {
        // Take the real portion of the complex number and put it into r_nums
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

    /* Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */

    cudaMalloc((void **) &dev_sinogram_cmplx, sinogram_width*nAngles*sizeof(cufftComplex));
    cudaMalloc((void **) &dev_sinogram_float, sinogram_width*nAngles*sizeof(float));

    cudaMemcpy(dev_sinogram_cmplx, sinogram_host,
        sizeof(cufftComplex) * sinogram_width * nAngles, cudaMemcpyHostToDevice);

    // Create plan to do forward and inverse FFT. 
    cufftHandle plan;

    // We will do nAngles batches with each plan doing sinogram_width work
    int batch = nAngles;
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, batch);

    printf("Loaded Data\n");

    // Complete the forward FFT to make the data symmetric and in frequency
    // domain
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    printf("Completed FFT_forward\n");

    // Scale the sinogram data according to the high pass filter described
    // in class
    cudaHighPassKernel<<<nBlocks, threadsPerBlock>>> (dev_sinogram_cmplx,
        sinogram_width, nAngles);

    printf("Completed HighPass Filter\n");

    // Compute the inverse FFT to convert the data back to the original format
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);

    printf("Completed FFT Inverse\n");

    // Since we have complex data and we want to convert it all to real, we
    // use a separate kernel to convert the data into floats.
    cudaConvertToReal<<<nBlocks, threadsPerBlock>>> (dev_sinogram_cmplx,
        dev_sinogram_float, sinogram_width * nAngles);

    printf("Converted to Real\n");

    // Finished with cuFFT, so we destroy plan and complex data
    cufftDestroy(plan);
    cudaFree(dev_sinogram_cmplx);

    float * dmatrix;

    cudaArray* carray;
    cudaChannelFormatDesc channel;

    cudaMalloc((void **) &dmatrix, sizeof(float) * height * width);

    channel = cudaCreateChannelDesc<float>();

    cudaMallocArray(&carray, &channel, width, height);

    cudaMemcpyToArray(carray, 0, 0, dev_sinogram_float, sinogram_width * nAngles, cudaMemcpyHostToDevice);

    texreference.filterMode = cudaFilterModePoint;
    texreference.addressMode[0] = cudaAddressModeWrap;
    texreference.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(texreference, carray);

    // Allocate space for the output image and set the output to 0
    cudaMalloc((void **) &output_dev, size_result);
    cudaMemset(output_dev, 0, size_result);

    // Occupy the image with the data from sinogram using Back Projection 
    cudaBackProjectionKernel <<<nBlocks, threadsPerBlock>>> (dev_sinogram_float,
     output_dev, nAngles, sinogram_width, width, height);

    cudaUnbindTexture(texreference);

    // After computing the output image, we move it back to CPU memory
    cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost);


    printf("Completed Back Projection\n");

    cudaFree(dmatrix);
    cudaFreeArray(carray);
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



