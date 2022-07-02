#include <stdio.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream> //for cout
#include <sstream>

#include "support.h"
#include "kernel.cu"

//#define DEBUGGING 1

#define L_Matrix_mat(row,col) matrix[((row)*numRows + (col))]
void transpose_matrix(double* matrix, int numRows)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            double a = L_Matrix_mat(i,j);
            L_Matrix_mat(i,j) = L_Matrix_mat(j,i);
            L_Matrix_mat(j,i) = a;
        }
    }
}


int main(int argc, char* argv[])
{
    Timer timer;
    
    double* matL_d;
    double* vecX_d;
    double* vecB_d;
    double* vecX_h;

    cudaError_t cuda_ret;
    
    //argv[1] - mode
    //argv[2] - dataset
    
    int mode, dataset;
    if(argc == 3)
    {
        mode = atoi(argv[1]);
        dataset = atoi(argv[2]);
    }
    else
    {
        printf("\n    Invalid input parameters."
               "\n    Usage: ./solver <m> <d>      # Mode: m, Dataset: d"
               "\n\n");
        exit(0);
    }
    
    //matL*vecX = vecB
    //matL and vecB are known, need to solver for vecX
    
    std::vector<double> matL_h;
    std::vector<double> vecB_h;
    std::vector<double> vecX_actual;
    
    std::stringstream ss;
    ss << "datasets/dataset" << dataset << "/matL.csv";
    loadCSV(ss.str(), matL_h);
    
    ss.str("");
    ss << "datasets/dataset" << dataset << "/vecX.csv";
    loadCSV(ss.str(), vecX_actual);
    
    ss.str("");
    ss << "datasets/dataset" << dataset << "/vecB.csv";
    loadCSV(ss.str(), vecB_h);
    
    //allocate space for the calculated vecB on the host
    vecX_h = (double*)malloc(vecX_actual.size()*sizeof(double));

    if (mode == 5)
        transpose_matrix(&matL_h[0], vecB_h.size());
    
#ifdef DEBUGGING
    printf("Actual vecX..."); fflush(stdout);
    for(int i=0; i<int(vecX_actual.size()); i++)
        std::cout << vecX_actual[i] << "\n";
#endif

    // allocate device variables
    if (mode != 0) //if not running CPU version
    {
        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);
        
        cuda_ret = cudaMalloc((void**)&matL_d,(matL_h.size())*sizeof(double));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
        
        cuda_ret = cudaMalloc((void**)&vecB_d,(vecB_h.size())*sizeof(double));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
        
        cuda_ret = cudaMalloc((void**)&vecX_d,(vecX_actual.size())*sizeof(double));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    
    // copy host variables to device
    if (mode != 0) //if not running CPU version
    {
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);
        
        cuda_ret = cudaMemcpy(matL_d,&matL_h[0],(matL_h.size())*sizeof(double), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
            FATAL("Unable to copy memory to the device");
        }
        
        cuda_ret = cudaMemcpy(vecB_d,&vecB_h[0],(vecB_h.size())*sizeof(double), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
            FATAL("Unable to copy memory to the device");
        }
        
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    printf("Size of L: %d,%d\n", vecB_h.size(), vecB_h.size());
    
    // launch the kernel
    printf("Launching kernel ");
    
    if(mode == 0) //CPU
    {
        printf("(CPU)...");fflush(stdout);
        startTime(&timer);
        
        cpu_solver(&matL_h[0], vecX_h,  &vecB_h[0], vecB_h.size());
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#ifdef DEBUGGING
        printf("Computed vecX..."); fflush(stdout);
        for(int i=0; i<int(vecX_actual.size()); i++)
            std::cout << vecX_h[i] << "\n";
#endif

    }
    else if (mode == 1) //GPU Simple
    {
        printf("(GPU - Simple)...");fflush(stdout);
        startTime(&timer);

        gpu_simple_solver(matL_d, vecX_d,  vecB_d, vecB_h.size());
        cudaDeviceSynchronize();
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else if (mode == 2) //GPU Complex
    {
        printf("(GPU - Complex)...");fflush(stdout);
        startTime(&timer);
        
        gpu_complex_solver(matL_d, vecX_d,  vecB_d, vecB_h.size());
        cudaDeviceSynchronize();
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else if (mode == 3) //GPU Complex Other
    {
        printf("(GPU - Complex Optimized)...");fflush(stdout);
        startTime(&timer);
        
        gpu_complex_solver_optimized(matL_d, vecX_d,  vecB_d, vecB_h.size());
        cudaDeviceSynchronize();
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else if (mode == 4) //GPU Complex Other-shared mem
    {
        printf("(GPU - Complex Optimized w Sh Mem)...");fflush(stdout);
        startTime(&timer);
        
        gpu_complex_solver_optimized_sh(matL_d, vecX_d,  vecB_d, vecB_h.size());
        cudaDeviceSynchronize();
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else if (mode == 5) //GPU Complex Transposed
    {
        printf("(GPU - Complex Transposed)...");fflush(stdout);
        startTime(&timer);
        
        gpu_complex_solver_transposed(matL_d, vecX_d,  vecB_d, vecB_h.size());
        cudaDeviceSynchronize();
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }
    else
    {
        printf("Invalid mode!\n");
        exit(0);
    }
    
    // copy device variables to host
    if (mode != 0) //if not running CPU version
    {
        printf("Copying data from device to host..."); fflush(stdout);
        //cudaDeviceSynchronize();
        
        startTime(&timer);
        
        cuda_ret = cudaMemcpy(vecX_h, vecX_d, vecX_actual.size() * sizeof(double),cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
        
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        
#ifdef DEBUGGING
        printf("Computed vecX..."); fflush(stdout);
        for(int i=0; i<int(vecX_actual.size()); i++)
            std::cout << vecX_h[i] << "\n";
#endif
    }

    //verify the results
    printf("Verifying results..."); fflush(stdout);
    verifyResults(vecX_h, &vecX_actual[0], vecX_actual.size());

    //free memory
    free (vecX_h);
    if (mode != 0) //if not running CPU version
    {
        cudaFree(matL_d);
        cudaFree(vecX_d);
        cudaFree(vecB_d);
    }
    
}


