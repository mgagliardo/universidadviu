#include "gf_gpu.h"

__global__ void log_w08_kernel(uint64_t c, unsigned size, uint8_t *data,
    uint8_t *log_table, uint8_t *antilog_table)
{
  __shared__ uint8_t shared_log[256];
  __shared__ uint8_t shared_alog[256];
  if(threadIdx.x < 256) {
    shared_log[threadIdx.x] = log_table[threadIdx.x];
    shared_alog[threadIdx.x] = antilog_table[threadIdx.x];
  }
  __syncthreads();

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned log_c = shared_log[c]; //maybe just pass in the log of c?
  while(id < size)
  {
    if(data[id] != 0)
      data[id] = shared_alog[(log_c + shared_log[data[id]]) % 255];
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void log_w08_kernel(uint64_t c, unsigned size, uint8_t *data,
    uint8_t *log_table, uint8_t *antilog_table)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned log_c = log_table[c]; //maybe just pass in the log of c?
  while(id < size)
  {
    if(data[id] != 0)
      data[id] = antilog_table[(log_c + log_table[data[id]]) % 255];
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void log_w16_kernel(uint64_t c, unsigned size, uint16_t *data,
    uint16_t *log_table, uint16_t *antilog_table)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned log_c = log_table[c];
  while(id < size)
  {
    if(data[id] != 0)
      data[id] = antilog_table[(log_c + log_table[data[id]]) % 65535];
    id += gridDim.x * blockDim.x;
  }
}

void log_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  //If c is 0, maybe just do a memcpy? or throw an error instead?
  void *cuda_log_table;
  void *cuda_antilog_table;
  Timer t;

  if(w == 8)
  {
    printf("Generating log tables... ");
    startTime(&t);
    uint8_t *log_table = (uint8_t *) malloc(256);
    uint8_t *antilog_table = (uint8_t *) malloc(256);
	//uint8_t *antilog_table = (uint8_t *) malloc(512);
    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_log_table, 256);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMalloc((void**)&cuda_antilog_table, 256);
    //cuda_ret = cudaMalloc((void**)&cuda_antilog_table, 512);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));

    unsigned count = 0;
    uint16_t num = 1;
    do
    {
      log_table[num] = count;
      antilog_table[count++] = num;

      num <<= 1;
      if(0x100 & num)
        num ^= IP08;
    } while(num != 1);
	//memcpy(antilog_table + 255, antilog_table, 255);
	//antilog_table[510] = 0;

    cuda_ret = cudaMemcpy(cuda_log_table, log_table, 256,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMemcpy(cuda_antilog_table, antilog_table, 256,
        cudaMemcpyHostToDevice);
    //cuda_ret = cudaMemcpy(cuda_antilog_table, antilog_table, 512,
    //    cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    free(log_table);
    free(antilog_table);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("LOG 08 launching... ");
    startTime(&t);
    log_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes, (uint8_t *)data,
        (uint8_t *)cuda_log_table, (uint8_t *)cuda_antilog_table);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));

    cudaFree(cuda_log_table);
    cudaFree(cuda_antilog_table);
  }
  else if(w == 16)
  {
    printf("Generating log tables... ");
    startTime(&t);
    uint16_t *log_table = (uint16_t *) malloc(sizeof(uint16_t)*256*256);
    uint16_t *antilog_table = (uint16_t *) malloc(sizeof(uint16_t)*256*256);
    cudaError_t cuda_ret = cudaMalloc((void**)&cuda_log_table, sizeof(uint16_t)*256*256);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMalloc((void**)&cuda_antilog_table, sizeof(uint16_t)*256*256);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));

    unsigned count = 0;
    uint32_t num = 1;
    do
    {
      log_table[num] = count;
      antilog_table[count++] = num;

      num <<= 1;
      if(0x10000 & num)
        num ^= IP16;
    } while(num != 1);

    cuda_ret = cudaMemcpy(cuda_log_table, log_table, sizeof(uint16_t)*256*256,
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    cuda_ret = cudaMemcpy(cuda_antilog_table, antilog_table,
        sizeof(uint16_t)*256*256, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
      usage(cudaGetErrorString(cuda_ret));
    free(log_table);
    free(antilog_table);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s)\n", elapsedTime(&t));

    printf("LOG 16 launching... ");
    startTime(&t);
    log_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/2, (uint16_t *)data,
        (uint16_t *)cuda_log_table, (uint16_t *)cuda_antilog_table);
    cudaDeviceSynchronize();
    stopTime(&t);
    printf("(%f s) %f MB/s\n", elapsedTime(&t),
        (float) bytes / (1024 * 1024) / elapsedTime(&t));

    cudaFree(cuda_log_table);
    cudaFree(cuda_antilog_table);
  }
}
