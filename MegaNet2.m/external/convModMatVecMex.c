#include <mex.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int i, k, nex, nft, nind, ycol, acol;
    if (nrhs<4)
         mexErrMsgTxt("Number of arguments must be 4: Y,dA,I,J!");
    
    const double* Y  = mxGetData(prhs[0]);
    const double* dA = mxGetData(prhs[1]);
    const int* I     = (int *) mxGetData(prhs[2]);
    const int* J     = (int *) mxGetData(prhs[3]);

    nex = mxGetM(prhs[0]);
    nft = mxGetN(prhs[0]);
    
    if ((nex != mxGetM(prhs[1])) || (nft != mxGetN(prhs[1])))
        mexErrMsgTxt("Size of Y and dA must agree!");
    
    nind = mxGetNumberOfElements(prhs[2]);
    
    if (nind != mxGetNumberOfElements(prhs[3]))
        mexErrMsgTxt("Size of I and J must agree!");

    mwSize dims[2]; dims[0] = nind; dims[1] = 1;
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double* t =  mxGetData(plhs[0]);
    
    #pragma omp parallel for default(shared) private(i,k,ycol,acol)
    for (i=0; i < nind; i++){
        ycol = nex*(I[i]-1);
        acol = nex*(J[i]-1);
         for ( k=0; k<nex; k++){
             t[i] += Y[k+ycol]*dA[k+acol];
         }
    }
  }


