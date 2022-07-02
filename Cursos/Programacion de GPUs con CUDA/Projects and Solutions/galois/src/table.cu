#include "gf_gpu.h"

__global__ void table_w08_kernel(unsigned size, uint8_t *data, uint8_t *table)
{
  __shared__ uint8_t s_table[256];
  //load the table into shared memory
  if(threadIdx.x < 256)
    s_table[threadIdx.x] = table[threadIdx.x];
  __syncthreads();

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size)
  {
    data[id] = s_table[data[id]];
    id += gridDim.x * blockDim.x;
  }
}

/*
//maybe do larger tables but they won't fit in shared memory
//16-bit word would fit because we're doing lazy tables, but 32-bit would not
__global__ void table_w08_kernel(unsigned size, uint8_t *data, uint8_t *table)
{
  //load the table into shared memory
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size)
  {
    data[id] = table[data[id]];
    id += gridDim.x * blockDim.x;
  }
}
*/

void table_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  void *cuda_table;

  //generate the table
  Timer t;
  printf("Generating table... ");
  startTime(&t);
  uint8_t *table = (uint8_t *) malloc(256);
  cudaError_t cuda_ret = cudaMalloc((void**)&cuda_table, 256);
  if(cuda_ret != cudaSuccess)
    usage(cudaGetErrorString(cuda_ret));

  for(int i = 0; i < 256; i++)
    table[i] = multiply_single_w08(c, i);

  cuda_ret = cudaMemcpy(cuda_table, table, 256, cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess)
    usage(cudaGetErrorString(cuda_ret));
  free(table);
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s)\n", elapsedTime(&t));

  printf("TABLE 08 launching... ");
  startTime(&t);
  table_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes, (uint8_t *) data,
      (uint8_t *) cuda_table);
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s) %f MB/s\n", elapsedTime(&t),
      (float) bytes / (1024 * 1024) / elapsedTime(&t));
  cudaFree(cuda_table);
}
