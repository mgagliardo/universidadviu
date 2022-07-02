#include "gf_gpu.h"

#define MASK 0xf
__global__ void split_w08_kernel(unsigned size, uint8_t *data, uint8_t *tables)
{
  __shared__ uint8_t s_tables[32];
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadIdx.x < 32)
    s_tables[threadIdx.x] = tables[threadIdx.x];
  __syncthreads();

  while(id < size)
  {
    uint8_t temp = data[id];
    data[id] = s_tables[temp & MASK]
             ^ s_tables[((temp >> 4) & MASK) + 16];
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void split_w08_kernel(unsigned size, uint8_t *data, uint8_t *tables)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size)
  {
    uint8_t temp = data[id];
    data[id] = tables[temp & MASK] ^ tables[((temp >> 4) & MASK) + 16];
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void split_w16_kernel(unsigned size, uint16_t *data, uint16_t *tables)
{
  __shared__ uint16_t s_tables[64];
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadIdx.x < 64)
    s_tables[threadIdx.x] = tables[threadIdx.x];
  __syncthreads();

  while(id < size)
  {
    uint16_t temp = data[id];
    data[id] = s_tables[temp & MASK]
             ^ s_tables[((temp >> 4) & MASK) + 16]
             ^ s_tables[((temp >> 8) & MASK) + 32]
             ^ s_tables[((temp >> 12) & MASK) + 48];
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void split_w16_kernel(unsigned size, uint16_t *data, uint16_t *tables)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size)
  {
    uint16_t temp = data[id];
    data[id] = tables[temp & MASK];
    data[id] ^= tables[((temp >> 4) & MASK) + 16];
    data[id] ^= tables[((temp >> 8) & MASK) + 32];
    data[id] ^= tables[((temp >> 12) & MASK) + 48];
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void split_w32_kernel(unsigned size, uint32_t *data, uint32_t *tables)
{
  __shared__ uint32_t s_tables[128];
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadIdx.x < 128)
    s_tables[threadIdx.x] = tables[threadIdx.x];
  __syncthreads();

  while(id < size)
  {
    uint32_t temp = data[id];
    data[id] = s_tables[temp & MASK]
             ^ s_tables[((temp >> 4) & MASK) + 16]
             ^ s_tables[((temp >> 8) & MASK) + 32]
             ^ s_tables[((temp >> 12) & MASK) + 48]
             ^ s_tables[((temp >> 16) & MASK) + 64]
             ^ s_tables[((temp >> 20) & MASK) + 80]
             ^ s_tables[((temp >> 24) & MASK) + 96]
             ^ s_tables[((temp >> 28) & MASK) + 112];
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void split_w32_kernel(unsigned size, uint32_t *data, uint32_t *tables)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size)
  {
    uint32_t temp = data[id];
    data[id] = tables[temp & MASK]; //probably should be a temp var
    data[id] ^= tables[((temp >> 4) & MASK) + 16];
    data[id] ^= tables[((temp >> 8) & MASK) + 32];
    data[id] ^= tables[((temp >> 12) & MASK) + 48];
    data[id] ^= tables[((temp >> 16) & MASK) + 64];
    data[id] ^= tables[((temp >> 20) & MASK) + 80];
    data[id] ^= tables[((temp >> 24) & MASK) + 96];
    data[id] ^= tables[((temp >> 28) & MASK) + 112];
    id += gridDim.x * blockDim.x;
  }
}
*/

void split_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  //setup the tables
  void *cuda_tables;
  Timer t;
  if(w == 8)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint8_t tables[16*2];
    for(uint32_t i = 0; i < 16; i++)
      for(int j = 0; j < 2; j++)
        tables[i + 16*j] = multiply_single_w08(c, i << (j*4));

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_tables,
        sizeof(uint8_t) * 16 * 2);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));

    cuda_ret = cudaMemcpy(cuda_tables, tables, sizeof(uint8_t) * 16 * 2,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("SPLIT 08 launching... ");
    startTime(&t);
    split_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes, (uint8_t *) data,
        (uint8_t *) cuda_tables);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
      (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }
  else if(w == 16)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint16_t tables[16*4];
    for(uint32_t i = 0; i < 16; i++)
      for(int j = 0; j < 4; j++)
        tables[i + 16*j] = multiply_single_w16(c, i << (j*4));

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_tables,
        sizeof(uint16_t) * 16 * 4);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));

    cuda_ret = cudaMemcpy(cuda_tables, tables, sizeof(uint16_t) * 16 * 4,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("SPLIT 16 launching... ");
    startTime(&t);
    split_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes/2, (uint16_t *) data,
        (uint16_t *) cuda_tables);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }
  else if(w == 32)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint32_t tables[16*8];
    for(uint32_t i = 0; i < 16; i++)
      for(int j = 0; j < 8; j++)
        tables[i + 16*j] = multiply_single_w32(c, i << (j*4));

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_tables,
        sizeof(uint32_t) * 16 * 8);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));

    cuda_ret = cudaMemcpy(cuda_tables, tables, sizeof(uint32_t) * 16 * 8,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("SPLIT 32 launching... ");
    startTime(&t);
    split_w32_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes/4, (uint32_t *) data,
        (uint32_t *) cuda_tables);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }

  cudaFree(cuda_tables);
}
