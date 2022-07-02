#include "gf_gpu.h"


__global__ void group_w08_kernel(unsigned size, uint8_t *data, uint16_t *mtable,
    uint16_t *rtable)
{
  __shared__ uint16_t smtable[16];
  __shared__ uint16_t srtable[16];
  if(threadIdx.x < 16) {
    smtable[threadIdx.x] = mtable[threadIdx.x];
    srtable[threadIdx.x] = rtable[threadIdx.x];
  }
  __syncthreads();

  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint8_t mask = 0xf;
    uint16_t temp = 0;
    #pragma unroll
    for(unsigned i = 0; i < 2; i++, data[id] >>= 4)
      temp ^= smtable[data[id] & mask] << (i*4);
    #pragma unroll
    for(unsigned i = 3; i >= 2; i--)
      temp ^= srtable[(temp >> (i*4))] << ((i-2)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}


/*
//Unoptimized
__global__ void group_w08_kernel(unsigned size, uint8_t *data, uint16_t *mtable,
    uint16_t *rtable)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint8_t mask = 0xf;
    uint16_t temp = 0;
    for(unsigned i = 0; i < 2; i++, data[id] >>= 4)
      temp ^= mtable[data[id] & mask] << (i*4);
    for(unsigned i = 3; i >= 2; i--)
      temp ^= rtable[(temp >> (i*4))] << ((i-2)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}
*/


__global__ void group_w16_kernel(unsigned size, uint16_t *data,
    uint32_t *mtable, uint32_t *rtable)
{
  __shared__ uint32_t smtable[16];
  __shared__ uint32_t srtable[16];
  if(threadIdx.x < 16) {
    smtable[threadIdx.x] = mtable[threadIdx.x];
    srtable[threadIdx.x] = rtable[threadIdx.x];
  }
  __syncthreads();

  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint16_t mask = 0xf;
    uint32_t temp = 0;
    #pragma unroll
    for(unsigned i = 0; i < 4; i++, data[id] >>= 4)
      temp ^= smtable[data[id] & mask] << (i*4);
    #pragma unroll
    for(unsigned i = 7; i >= 4; i--)
      temp ^= srtable[(temp >> (i*4))] << ((i-4)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void group_w16_kernel(unsigned size, uint16_t *data,
    uint32_t *mtable, uint32_t *rtable)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint16_t mask = 0xf;
    uint32_t temp = 0;
    for(unsigned i = 0; i < 4; i++, data[id] >>= 4)
      temp ^= mtable[data[id] & mask] << (i*4);
    for(unsigned i = 7; i >= 4; i--)
      temp ^= rtable[(temp >> (i*4))] << ((i-4)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void group_w32_kernel(unsigned size, uint32_t *data,
    uint64_t *mtable, uint64_t *rtable)
{
  __shared__ uint64_t smtable[16];
  __shared__ uint64_t srtable[16];
  if(threadIdx.x < 16) {
    smtable[threadIdx.x] = mtable[threadIdx.x];
    srtable[threadIdx.x] = rtable[threadIdx.x];
  }
  __syncthreads();


  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint32_t mask = 0xf;
    uint64_t temp = 0;
    #pragma unroll
    for(unsigned i = 0; i < 8; i++, data[id] >>= 4)
      temp ^= smtable[data[id] & mask] << (i*4);
    #pragma unroll
    for(unsigned i = 15; i >= 8; i--)
      temp ^= srtable[(temp >> (i*4))] << ((i-8)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}


/*
//Unoptimized
__global__ void group_w32_kernel(unsigned size, uint32_t *data,
    uint64_t *mtable, uint64_t *rtable)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  while(id < size)
  {
    uint32_t mask = 0xf;
    uint64_t temp = 0;
    for(unsigned i = 0; i < 8; i++, data[id] >>= 4)
      temp ^= mtable[data[id] & mask] << (i*4);
    for(unsigned i = 15; i >= 8; i--)
      temp ^= rtable[(temp >> (i*4))] << ((i-8)*4);
    data[id] = temp;
    id += gridDim.x * blockDim.x;
  }
}
*/

void group_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  void *cuda_mtable;
  void *cuda_rtable;

  Timer t;
  //This might should be user configurable but I'm just making this do 4 steps
  //per pass
  if(w == 8)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint16_t mtable[16];
    uint16_t rtable[16];

    for(unsigned i = 0; i < 16; i++)
      mtable[i] = multiply_single_w08(c, i);

    for(unsigned i = 0; i < 16; i++)
    {
      rtable[i] = 0;
      for(unsigned j = 0; j < 4; j++)
        if(i & (1 << j))
          rtable[i] ^= IP08 << j;
    }

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_mtable,
        sizeof(uint16_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMalloc((void**)&cuda_rtable,
        sizeof(uint16_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    
    cuda_ret = cudaMemcpy(cuda_mtable, mtable, sizeof(uint16_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMemcpy(cuda_rtable, rtable, sizeof(uint16_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("GROUP 08 launching... ");
    startTime(&t);
    group_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes, (uint8_t *) data, 
        (uint16_t *) cuda_mtable, (uint16_t *) cuda_rtable);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }
  else if(w == 16)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint32_t mtable[16];
    uint32_t rtable[16];

    for(unsigned i = 0; i < 16; i++)
      mtable[i] = multiply_single_w16(c, i);

    for(unsigned i = 0; i < 16; i++)
    {
      rtable[i] = 0;
      for(unsigned j = 0; j < 4; j++)
        if(i & (1 << j))
          rtable[i] ^= IP16 << j;
    }

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_mtable,
        sizeof(uint32_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMalloc((void**)&cuda_rtable,
        sizeof(uint32_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    
    cuda_ret = cudaMemcpy(cuda_mtable, mtable, sizeof(uint32_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMemcpy(cuda_rtable, rtable, sizeof(uint32_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("GROUP 16 launching... ");
    startTime(&t);
    group_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes/2, (uint16_t *) data, 
        (uint32_t *) cuda_mtable, (uint32_t *) cuda_rtable);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }
  else if(w == 32)
  {
    printf("Generating tables... ");
    startTime(&t);
    uint64_t mtable[16];
    uint64_t rtable[16];

    for(unsigned i = 0; i < 16; i++)
      mtable[i] = multiply_single_w32(c, i);

    for(unsigned i = 0; i < 16; i++)
    {
      rtable[i] = 0;
      for(unsigned j = 0; j < 4; j++)
        if(i & (1 << j))
          rtable[i] ^= IP32 << j;
    }

    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_mtable,
        sizeof(uint64_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMalloc((void**)&cuda_rtable,
        sizeof(uint64_t) * 16);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    
    cuda_ret = cudaMemcpy(cuda_mtable, mtable, sizeof(uint64_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMemcpy(cuda_rtable, rtable, sizeof(uint64_t) * 16,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("GROUP 32 launching... ");
    startTime(&t);
    group_w32_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(bytes/4, (uint32_t *) data, 
        (uint64_t *) cuda_mtable, (uint64_t *) cuda_rtable);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));
  }

  cudaFree(cuda_mtable);
  cudaFree(cuda_rtable);
}
