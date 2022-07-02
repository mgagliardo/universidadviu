/* ljForce.cu */

#include "ljForce.h"
//#include <cuda_runtime.h>

__global__ void ljForce(float *a, float *b)
{
    //does nothing for now    
    
}

extern "C" void lunch_ljForce_kernel(real_t *pot, int nLocalBoxes, int *nAtoms, int *gridSize, int *gid, real3 *r, real_t *U, real3 *f, int sz)
{
    float *A_d;
    float *B_d;


    dim3 Dimgrid(1,1,1);
    dim3 Dimblock(512,1,1);

    cudaMalloc((void**)&A_d, sizeof(float));
    cudaMalloc((void**)&B_d, sizeof(float));
    
    ljForce<<<Dimgrid, Dimblock>>>(A_d, B_d);
    
    cudaFree(A_d);
    cudaFree(B_d);    
}
