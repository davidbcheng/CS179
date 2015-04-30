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
/////////////////////////////////////////////////////////////////////////

//                     1

/////////////////////////////////////////////////////////////////////////
__global__
void
cudaMaximumKernel1(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    shared[tid] = 0.0;
    while(i < padded_length)
    {
        shared[tid] = fmaxf(shared[tid], out_data[i].x);
        // Compute next index for arbitrary amount of threads
        i += blockDim.x * gridDim.x;
    }
    
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }   

        __syncthreads();
    }

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/////////////////////////////////////////////////////////////////////////

//                     2

/////////////////////////////////////////////////////////////////////////

__global__
void
cudaMaximumKernel2(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    shared[tid] = 0.0;
    while(i < padded_length)
    {
        shared[tid] = fmaxf(shared[tid], out_data[i].x);
        // Compute next index for arbitrary amount of threads
        i += blockDim.x * gridDim.x;
    }
    
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            shared[index] = fmaxf(shared[index], shared[index + s]);
        }

        __syncthreads();
    }

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/////////////////////////////////////////////////////////////////////////

//                     3

/////////////////////////////////////////////////////////////////////////

__global__
void
cudaMaximumKernel3(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    shared[tid] = 0.0;
    while(i < padded_length)
    {
        shared[tid] = fmaxf(shared[tid], out_data[i].x);
        // Compute next index for arbitrary amount of threads
        i += blockDim.x * gridDim.x;
    }
    
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/////////////////////////////////////////////////////////////////////////

//                     4

/////////////////////////////////////////////////////////////////////////

__global__
void
cudaMaximumKernel4(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;


    shared[tid] = 0.0;
    while(i < padded_length)
    {
        float max = fmaxf(out_data[i].x, out_data[i + blockDim.x].x);
        shared[tid] = fmaxf(shared[tid], max);
        // Compute next index for arbitrary amount of threads
        i += (2 * blockDim.x) * gridDim.x;
    }
    
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/////////////////////////////////////////////////////////////////////////

//                     5

/////////////////////////////////////////////////////////////////////////

__device__ void warpReduce(volatile float* shared, int tid)
{
    shared[tid] = fmaxf(shared[tid], shared[tid + 32]);
    shared[tid] = fmaxf(shared[tid], shared[tid + 16]);
    shared[tid] = fmaxf(shared[tid], shared[tid + 8]);
    shared[tid] = fmaxf(shared[tid], shared[tid + 4]);
    shared[tid] = fmaxf(shared[tid], shared[tid + 2]);
    shared[tid] = fmaxf(shared[tid], shared[tid + 1]);
}

__global__
void
cudaMaximumKernel5(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;


    shared[tid] = 0.0;
    while(i < padded_length)
    {
        float max = fmaxf(abs(out_data[i].x), abs(out_data[i + blockDim.x].x));
        shared[tid] = fmaxf(shared[tid], max);
        // Compute next index for arbitrary amount of threads
        i += (2 * blockDim.x) * gridDim.x;
    }
    
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + s]));
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduce(shared, tid);
    }

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}

/////////////////////////////////////////////////////////////////////////

//                     6

/////////////////////////////////////////////////////////////////////////

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

    if (blockSize >= 512) 
    {
        if (tid < 256) 
        {   
            shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 256]));
        }
        __syncthreads(); 
    }

    if (blockSize >= 256)
    {
        if (tid < 128) 
        {
            shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 128]));
        }
        __syncthreads(); 
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            shared[tid] = fmaxf(abs(shared[tid]), abs(shared[tid + 64]));
        }    
        __syncthreads(); 
    }

    if (tid < 32) warpReduce<blockSize>(shared, tid);

    if(tid == 0) atomicMax(max_abs_val, shared[0]);
}
