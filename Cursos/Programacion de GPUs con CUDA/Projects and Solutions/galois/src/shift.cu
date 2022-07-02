#include "gf_gpu.h"

__global__ void shift_w08_kernel(uint64_t c, unsigned size, uint8_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int num_bits;
  __shared__ int set_bits[8];

  //Find out which bits are set
  if(!threadIdx.x) {
    num_bits = 0;
    for(int i = 0; i < 8; ++i)
      if(1 << i & c)
        set_bits[num_bits++] = i;
  }
  __syncthreads();

  while(id < size)
  {
    uint16_t answer = 0;
    uint16_t element = data[id];
    for(unsigned i = 0; i < num_bits; ++i)
      //if((1 << i) & c)
      answer ^= element << set_bits[i];
    for(unsigned i = 14; i >= 8; --i)
      if((1 << i) & answer)
        answer ^= IP08 << (i-8);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void shift_w08_kernel(uint64_t c, unsigned size, uint8_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  while(id < size)
  {
    uint16_t answer = 0;
    uint16_t element = data[id];
    for(unsigned i = 0; i < 8; ++i)
      if((1 << i) & c)
        answer ^= element << i;
    for(unsigned i = 14; i >= 8; --i)
      if((1 << i) & answer)
        answer ^= IP08 << (i-8);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void shift_w16_kernel(uint64_t c, unsigned size, uint16_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int num_bits;
  __shared__ int set_bits[16];

  //Find out which bits are set
  if(!threadIdx.x) {
    num_bits = 0;
    for(int i = 0; i < 16; ++i)
      if(1 << i & c)
        set_bits[num_bits++] = i;
  }
  __syncthreads();

  while(id < size)
  {
    uint32_t answer = 0;
    uint32_t element = data[id];
    for(unsigned i = 0; i < num_bits; ++i)
      //if((1 << i) & c)
      answer ^= element << set_bits[i];
    for(unsigned i = 30; i >= 16; --i)
      if((1 << i) & answer)
        answer ^= IP16 << (i-16);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void shift_w16_kernel(uint64_t c, unsigned size, uint16_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  while(id < size)
  {
    uint32_t answer = 0;
    uint32_t element = data[id];
    for(unsigned i = 0; i < 16; ++i)
      if((1 << i) & c)
        answer ^= element << i;
    for(unsigned i = 30; i >= 16; --i)
      if((1 << i) & answer)
        answer ^= IP16 << (i-16);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void shift_w32_kernel(uint64_t c, unsigned size, uint32_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int num_bits;
  __shared__ int set_bits[32];

  //Find out which bits are set
  if(!threadIdx.x) {
    num_bits = 0;
    for(int i = 0; i < 32; ++i)
      if(1ULL << i & c)
        set_bits[num_bits++] = i;
  }
  __syncthreads();

  while(id < size)
  {
    uint64_t answer = 0;
    uint64_t element = data[id];
    for(unsigned i = 0; i < num_bits; ++i)
      //if((1ULL << i) & c)
      answer ^= element << set_bits[i];
    for(unsigned i = 62; i >= 32; --i)
      if((1ULL << i) & answer)
        answer ^= IP32 << (i-32);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void shift_w32_kernel(uint64_t c, unsigned size, uint32_t *data)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  while(id < size)
  {
    uint64_t answer = 0;
    uint64_t element = data[id];
    for(unsigned i = 0; i < 32; ++i)
      if((1ULL << i) & c)
        answer ^= element << i;
    for(unsigned i = 62; i >= 32; --i)
      if((1ULL << i) & answer)
        answer ^= IP32 << (i-32);
    data[id] = answer;
    id += gridDim.x * blockDim.x;
  }
}
*/

void shift_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  printf("SHIFT %02i launching... ", w);
  Timer t;
  startTime(&t); //This is a little off because w == 32 will cause 2 more ifs
  if(w == 8)
    shift_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes, (uint8_t *)data);
  else if(w == 16)
    shift_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/2, (uint16_t *)data);
  else if(w == 32)
    shift_w32_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/4, (uint32_t *)data);
  cudaDeviceSynchronize();
  stopTime(&t); //Superpower I wish I had
  printf("(%f s) %f MB/s\n", elapsedTime(&t),
      (float) bytes / (1024 * 1024) / elapsedTime(&t));
}

void check_answer_w08(uint64_t c, unsigned size, uint8_t *data, uint8_t *answer)
{
  printf("Verifying...");
  Timer t;
  startTime(&t);
  int i;
  for(i = 0; i < size; ++i)
    if(multiply_single_w08(c, data[i]) != answer[i])
      break;
  stopTime(&t);
  printf("%sCORRECT (%f s)\n", (i == size) ? "" : "IN", elapsedTime(&t));
}

void check_answer_w16(uint64_t c, unsigned size, uint16_t *data,
    uint16_t *answer)
{
  printf("Verifying...");
  Timer t;
  startTime(&t);
  int i;
  for(i = 0; i < size; ++i)
    if(multiply_single_w16(c, data[i]) != answer[i])
      break;
  stopTime(&t);
  printf("%sCORRECT (%f s)\n", (i == size) ? "" : "IN", elapsedTime(&t));
}

void check_answer_w32(uint64_t c, unsigned size, uint32_t *data,
    uint32_t *answer)
{
  printf("Verifying...");
  Timer t;
  startTime(&t);
  int i;
  for(i = 0; i < size; ++i)
    if(multiply_single_w32(c, data[i]) != answer[i])
      break;
  stopTime(&t);
  printf("%sCORRECT (%f s)\n", (i == size) ? "" : "IN", elapsedTime(&t));
}

uint8_t multiply_single_w08(uint16_t a, uint16_t b)
{
  uint16_t temp = 0;
  for(unsigned j = 0; j < 8; j++)
    if((1 << j) & b)
      temp ^= a << j;
  for(unsigned j = 14; j >= 8; j--)
    if((1 << j) & temp)
      temp ^= IP08 << (j-8);
  return temp;
}

uint16_t multiply_single_w16(uint32_t a, uint32_t b)
{
  uint32_t temp = 0;
  for(unsigned j = 0; j < 16; j++)
    if((1 << j) & b)
      temp ^= a << j;
  for(unsigned j = 30; j >= 16; j--)
    if((1 << j) & temp)
      temp ^= IP16 << (j-16);
  return temp;
}

uint32_t multiply_single_w32(uint64_t a, uint64_t b)
{
  uint64_t temp = 0;
  for(unsigned j = 0; j < 32; j++)
    if((1ULL << j) & b)
      temp ^= a << j;
  for(unsigned j = 62; j >= 32; j--)
    if((1ULL << j) & temp)
      temp ^= IP32 << (j-32);
  return temp;
}
