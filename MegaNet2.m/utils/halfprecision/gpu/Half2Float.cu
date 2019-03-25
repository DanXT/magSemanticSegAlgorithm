#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_fp16.h>

/*
 * Device code
 */
void __global__ float2half(__half const * const A, float * const B, int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] =  __half2float(A[i]);
    }
}
/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    __half const *d_A;
    float *d_B = NULL;
    int N;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
	
	/* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=1) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, "input is not a GPU array");
    }
    

    A = mxGPUCreateFromMxArray(prhs[0]);

    /*
     * Verify that A really is a float array before extracting the pointer.
     */
    if (mxGPUGetClassID(A) != mxUINT16_CLASS) {
        mexErrMsgIdAndTxt(errId, "A is not a single class");
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_A = (__half const *)(mxGPUGetDataReadOnly(A));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxSINGLE_CLASS,
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float*)mxGPUGetData(B);

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    N = (int)(mxGPUGetNumberOfElements(A));
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
}
