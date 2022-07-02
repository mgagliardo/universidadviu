
#define BLOCK_SIZE 1024

//numRows = numCols since L is square
//               y,x
#define L_Matrix(row,col) matL[((row)*numRows + (col))]

__global__ void gpu_simple_solver_kernel(double* matL, double* vecX, double* vecB, int numRows, int i)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= numRows)
        return;
    
    //update the B value for every thread by subtracting off the known x (which was calculating last iteration)
    //multiplied by the corresponding L element
    if (i != 0)
        vecB[idx] = vecB[idx] - L_Matrix(idx,i-1)*vecX[i-1];
    
    if (idx == i)
    {
        vecX[i] = vecB[i] / L_Matrix(i,i);
    }
}

__global__ void gpu_square_update_kernel(double* matL, double* vecX, double* vecB, int numRows)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    
    int y = idy*2;
    int x = idx*2;
    int top_tri_idx = y;
    
    if (x == 0)
    {
        vecB[y+1] = (vecB[y+1] - L_Matrix(top_tri_idx+1,top_tri_idx)/L_Matrix(top_tri_idx,top_tri_idx)*vecB[y])/L_Matrix(top_tri_idx+1,top_tri_idx+1);
        vecB[y] = vecB[y]/L_Matrix(top_tri_idx,top_tri_idx);
    }
    
    if (idx >= numRows/2 || idy >= numRows/2)
        return;
    
    if(idy <= idx)
        return;
    
    //element 1,0 (y,x) (row,col)
    L_Matrix(y+1,x) = (L_Matrix(y+1,x) - L_Matrix(top_tri_idx+1,top_tri_idx)/L_Matrix(top_tri_idx,top_tri_idx)*L_Matrix(y,x))/L_Matrix(top_tri_idx+1,top_tri_idx+1);
    
    //element 1,1 (y,x) (row,col)
    L_Matrix(y+1,x+1) = (L_Matrix(y+1,x+1) - L_Matrix(top_tri_idx+1,top_tri_idx)/L_Matrix(top_tri_idx,top_tri_idx)*L_Matrix(y,x+1))/L_Matrix(top_tri_idx+1,top_tri_idx+1);
    
    //element 0,0 (y,x) (row,col)
    L_Matrix(y,x) = L_Matrix(y,x)/L_Matrix(top_tri_idx,top_tri_idx);
    
    //element 0,1 (y,x) (row,col)
    L_Matrix(y,x+1) = L_Matrix(y,x+1)/L_Matrix(top_tri_idx,top_tri_idx);
}

__global__ void gpu_square_solve_kernel_simple(double* matL, double* vecX, double* vecB, int numRows, int i)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int col_index = i*2;
    if (col_index >= numRows)
        return;
        
    int row_index = idx;

    if (row_index < (i+1)*2 || row_index >= numRows)
        return;
    
    double value = L_Matrix(row_index,col_index)*vecB[col_index] + L_Matrix(row_index,col_index+1)*vecB[col_index + 1];
    vecB[row_index] = vecB[row_index] - value;
    
}


__global__ void gpu_square_solve_kernel_optimized(double* matL, double* vecX, double* vecB, int numRows,  int i)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    
    //eliminates all of the upper elements in matrix (amount increases as i increases)
    if (idx >= (idy-(idy%i)-i) && idx < numRows-1)
        return;
    
    //eliminates the rows that we should not be modifying
    if (idy % (i*2) < i)
        return;
    
    //bounds check for if a block goes outside of bounds
    if (idy >= numRows && idx >= numRows)
        return;
    
    //update vecB using the last column of threads
    if (idx == numRows-1)
    {
        double value = vecB[idy];
        int offset = idy % i;
        
        for (int j = 0; j < i; j++)
        {
            int column = idy-offset-j-1;
            value = value - vecB[column]*L_Matrix(idy,column);
            //L_Matrix(idy,column) = 0;
        }
        
        vecB[idy] = value;
    }
    else //update the L matrix values
    {
        double value = L_Matrix(idy,idx);
        int offset = idy % i;
        
        for (int j = 0; j < i; j++)
        {
            int column = idy-offset-j-1;
            value = value - L_Matrix(column,idx)*L_Matrix(idy,column);
        }
        
        L_Matrix(idy,idx) = value;
    }
}

__global__ void gpu_square_solve_kernel_optimized_sh(double* matL, double* vecX, double* vecB, int numRows,  int i)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    
    if (idx >= (idy-(idy%i)-i) && idx < numRows-32)
        return;
    
    if (idy % (i*2) < i)
        return;
    
    if (idy >= numRows && idx >= numRows)
        return;
    
    __shared__ double dsb_row_elements[32][32];
    __shared__ double dsb_row_multipliers[32][32];
    int offset = idy % i;
    //int sh_offset = idx % i;
    if (idx >= numRows-32) //threads that do not modify the matrix
    {
        double value;
        if (idx == numRows - 1)
            value = vecB[idy];
        
        for (int k = 0; k < i; k +=32)
        {
            __syncthreads();
            //load values from shared memory
            dsb_row_multipliers[threadIdx.y][threadIdx.x] = L_Matrix(idy, idy - offset - (32+k) + threadIdx.x);
            if (idx == numRows - 1)
                dsb_row_elements[threadIdx.y][0] =  vecB[idy - offset - (32+k) + threadIdx.y];
            __syncthreads();
            
            if (idx == numRows - 1) // only use the last thread to update vec values
            {
                for (int j = 0; j < 32; j++)
                {
                    value = value - dsb_row_multipliers[threadIdx.y][j]*dsb_row_elements[j][0];
                }
            }
        }
        
        if (idx == numRows - 1)
            vecB[idy] = value;
    }
    else
    {
        //loop through tiles
        double value = L_Matrix(idy,idx);
        for (int k = 0; k < i; k +=32)
        {
            __syncthreads();
            //load values from shared memory
            dsb_row_elements[threadIdx.y][threadIdx.x] =  L_Matrix(idy - offset - (32+k) + threadIdx.y, idx);
            dsb_row_multipliers[threadIdx.y][threadIdx.x] = L_Matrix(idy, idy - offset - (32+k) + threadIdx.x);
            __syncthreads();
            
            for (int j = 0; j < 32; j++)
                value -= dsb_row_multipliers[threadIdx.y][j]*dsb_row_elements[j][threadIdx.x];
        }
        
        L_Matrix(idy,idx) = value;
    }
    
}

void cpu_solver(double* matL, double* vecX, double* vecB, int numRows)
{
    for (int i = 0; i < numRows; i++)
    {
        double val = vecB[i];
        for (int j = 0; j < i; j++)
        {
            val = val - L_Matrix(i,j)*vecX[j];
        }
        vecX[i] = val / L_Matrix(i,i);
    }
}


void gpu_simple_solver(double* matL, double* vecX, double* vecB, int numRows)
{
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int numBlocks = (numRows - 1)/numThreadsPerBlock + 1;
    
    for (int i = 0; i < numRows; i++)
        gpu_simple_solver_kernel<<<numBlocks , numThreadsPerBlock>>>(matL, vecX, vecB, numRows, i);
}


void gpu_complex_solver(double* matL, double* vecX, double* vecB, int numRows)
{
    dim3 dimGrid((numRows/2-1)/32+1,(numRows/2-1)/32+1,1);
    dim3 dimBlock(32,32,1);
    
    gpu_square_update_kernel<<<dimGrid,dimBlock>>>(matL, vecX, vecB, numRows);
    
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int numBlocks = (numRows - 1)/numThreadsPerBlock + 1;
    
    for (int i = 0; i < (numRows / 2); i++)
    {
        gpu_square_solve_kernel_simple<<<numBlocks , numThreadsPerBlock>>>(matL, vecX, vecB, numRows, i);
    }
    
    //copy B to X for the verification code in main.cu
    cudaMemcpy(vecX, vecB, numRows * sizeof(double),cudaMemcpyDeviceToDevice);
}

void gpu_complex_solver_optimized(double* matL, double* vecX, double* vecB, int numRows)
{
    dim3 dimGrid((numRows/2-1)/32+1,(numRows/2-1)/32+1,1);
    dim3 dimBlock(32,32,1);
    
    gpu_square_update_kernel<<<dimGrid,dimBlock>>>(matL, vecX, vecB, numRows);
 
    dim3 dimGrid2((numRows-1)/32+1,(numRows-1)/32+1,1);
    dim3 dimBlock2(32,32,1);
    
    for (int i = 2; i < numRows; i *= 2)
    {
        gpu_square_solve_kernel_optimized<<<dimGrid2,dimBlock2>>>(matL, vecX, vecB, numRows, i);
    }
    
    //copy B to X for the verification code in main.cu
    cudaMemcpy(vecX, vecB, numRows * sizeof(double),cudaMemcpyDeviceToDevice);
}

void gpu_complex_solver_optimized_sh(double* matL, double* vecX, double* vecB, int numRows)
{
    dim3 dimGrid((numRows/2-1)/32+1,(numRows/2-1)/32+1,1);
    dim3 dimBlock(32,32,1);
    
    gpu_square_update_kernel<<<dimGrid,dimBlock>>>(matL, vecX, vecB, numRows);
    
    dim3 dimGrid2((numRows-1)/32+1,(numRows-1)/32+1,1);
    dim3 dimBlock2(32,32,1);
    
    for (int i = 2; i < numRows; i *= 2)
    {
        if (i < 32)
            gpu_square_solve_kernel_optimized<<<dimGrid2,dimBlock2>>>(matL, vecX, vecB, numRows, i);
        else
            gpu_square_solve_kernel_optimized_sh<<<dimGrid2,dimBlock2>>>(matL, vecX, vecB, numRows, i);
    }
    
    //copy B to X for the verification code in main.cu
    cudaMemcpy(vecX, vecB, numRows * sizeof(double),cudaMemcpyDeviceToDevice);
}





#define L_Matrix_t(col,row) matL[((row)*numRows + (col))]
__global__ void gpu_square_update_kernel_transposed(double* matL, double* vecX, double* vecB, int numRows)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    
    int y = idy*2;
    int x = idx*2;
    int top_tri_idx = y;
    
    if (x == 0)
    {
        vecB[y+1] = (vecB[y+1] - L_Matrix_t(top_tri_idx+1,top_tri_idx)/L_Matrix_t(top_tri_idx,top_tri_idx)*vecB[y])/L_Matrix_t(top_tri_idx+1,top_tri_idx+1);
        vecB[y] = vecB[y]/L_Matrix_t(top_tri_idx,top_tri_idx);
    }
    
    if (idx >= numRows/2 || idy >= numRows/2)
        return;
    
    if(idy <= idx)
        return;
    
    //element 1,0 (y,x) (row,col)
    L_Matrix_t(y+1,x) = (L_Matrix_t(y+1,x) - L_Matrix_t(top_tri_idx+1,top_tri_idx)/L_Matrix_t(top_tri_idx,top_tri_idx)*L_Matrix_t(y,x))/L_Matrix_t(top_tri_idx+1,top_tri_idx+1);
    
    //element 1,1 (y,x) (row,col)
    L_Matrix_t(y+1,x+1) = (L_Matrix_t(y+1,x+1) - L_Matrix_t(top_tri_idx+1,top_tri_idx)/L_Matrix_t(top_tri_idx,top_tri_idx)*L_Matrix_t(y,x+1))/L_Matrix_t(top_tri_idx+1,top_tri_idx+1);
    
    //element 0,0 (y,x) (row,col)
    L_Matrix_t(y,x) = L_Matrix_t(y,x)/L_Matrix_t(top_tri_idx,top_tri_idx);
    
    //element 0,1 (y,x) (row,col)
    L_Matrix_t(y,x+1) = L_Matrix_t(y,x+1)/L_Matrix_t(top_tri_idx,top_tri_idx);
}



__global__ void gpu_square_solve_kernel_simple_transposed(double* matL, double* vecX, double* vecB, int numRows, int i)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int col_index = i*2;
    if (col_index >= numRows)
        return;
    
    int row_index = idx;
    
    if (row_index < (i+1)*2 || row_index >= numRows)
        return;
    
    double value = L_Matrix_t(row_index,col_index)*vecB[col_index] + L_Matrix_t(row_index,col_index+1)*vecB[col_index + 1];
    vecB[row_index] = vecB[row_index] - value;
    
}



void gpu_complex_solver_transposed(double* matL, double* vecX, double* vecB, int numRows)
{
    dim3 dimGrid((numRows/2-1)/32+1,(numRows/2-1)/32+1,1);
    dim3 dimBlock(32,32,1);
    
    gpu_square_update_kernel_transposed<<<dimGrid,dimBlock>>>(matL, vecX, vecB, numRows);
    
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int numBlocks = (numRows - 1)/numThreadsPerBlock + 1;
    
    for (int i = 0; i < (numRows / 2); i++)
    {
        gpu_square_solve_kernel_simple_transposed<<<numBlocks , numThreadsPerBlock>>>(matL, vecX, vecB, numRows, i);
    }
    
    //copy B to X for the verification code in main.cu
    cudaMemcpy(vecX, vecB, numRows * sizeof(double),cudaMemcpyDeviceToDevice);
}
