#include "gf_gpu.h"


__global__ void bytwop_w08_kernel(uint64_t c, unsigned size, uint8_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint8_t prod = 0;
    //uint8_t pmask = 0x80;
    uint8_t cmask = 0x80;
    int8_t sprod;
    uint8_t element = data[id];
    while(cmask)
    {
      sprod = prod;
      prod = (prod << 1) ^ (IP08 & (sprod >> 7));
      if(c & cmask)
        prod ^= element;
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void bytwop_w08_kernel(uint64_t c, unsigned size, uint8_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint8_t prod = 0;
    uint8_t pmask = 0x80;
    uint8_t cmask = 0x80;
    while(cmask)
    {
      if(prod & pmask)
        prod = (prod << 1) ^ IP08;
      else
        prod <<= 1;

      if(c & cmask)
        prod ^= data[id];
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void bytwop_w16_kernel(uint64_t c, unsigned size, uint16_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint16_t prod = 0;
    //uint16_t pmask = 0x8000;
    uint16_t cmask = 0x8000;
    int16_t sprod;
    uint16_t element = data[id];
    while(cmask)
    {
      sprod = prod;
      prod = (prod << 1) ^ (IP16 & (sprod >> 15));
      if(c & cmask)
        prod ^= element;
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void bytwop_w16_kernel(uint64_t c, unsigned size, uint16_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint16_t prod = 0;
    uint16_t pmask = 0x8000;
    uint16_t cmask = 0x8000;
    while(cmask)
    {
      if(prod & pmask)
        prod = (prod << 1) ^ IP16;
      else
        prod <<= 1;

      if(c & cmask)
        prod ^= data[id];
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void bytwop_w32_kernel(uint64_t c, unsigned size, uint32_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint32_t prod = 0;
    //uint32_t pmask = 0x80000000;
    uint32_t cmask = 0x80000000;
    int32_t sprod;
    uint32_t element = data[id];
    while(cmask)
    {
      sprod = prod;
      prod = (prod << 1) ^ (IP32 & (sprod >> 31));
      if(c & cmask)
        prod ^= element;
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

/*
//Unoptimized
__global__ void bytwop_w32_kernel(uint64_t c, unsigned size, uint32_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint32_t prod = 0;
    uint32_t pmask = 0x80000000;
    uint32_t cmask = 0x80000000;
    while(cmask)
    {
      if(prod & pmask)
        prod = (prod << 1) ^ IP32;
      else
        prod <<= 1;

      if(c & cmask)
        prod ^= data[id];
      cmask >>= 1;
    }
    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}
*/

__global__ void bytwob_w08_kernel(uint64_t c, unsigned size, uint8_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint8_t prod = 0;
    uint8_t cc = c;
    uint8_t cmask = 0x80;
    uint8_t element = data[id];
    while(1)
    {
      if(cc & 1)
        prod ^= element;
      cc >>= 1;
      if(cc == 0)
        break;

      if(element & cmask)
        element = (element << 1) ^ IP08;
      else
        element <<= 1;
    }

    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

__global__ void bytwob_w16_kernel(uint64_t c, unsigned size, uint16_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint16_t prod = 0;
    uint16_t cc = c;
    uint16_t cmask = 0x8000;
    uint16_t element = data[id];
    while(1)
    {
      if(cc & 1)
        prod ^= element;
      cc >>= 1;
      if(cc == 0)
        break;

      if(element & cmask)
        element = (element << 1) ^ IP16;
      else
        element <<= 1;
    }

    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

__global__ void bytwob_w32_kernel(uint64_t c, unsigned size, uint32_t *data)
{
  unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

  while(id < size)
  {
    uint32_t prod = 0;
    uint32_t cc = c;
    uint32_t cmask = 0x80000000;
    uint32_t element = data[id];
    while(1)
    {
      if(cc & 1)
        prod ^= element;
      cc >>= 1;
      if(cc == 0)
        break;

      if(element & cmask)
        element = (element << 1) ^ IP32;
      else
        element <<= 1;
    }

    data[id] = prod;
    id += gridDim.x * blockDim.x;
  }
}

void bytwob_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  Timer t;
  printf("BYTWO_b %02i launching... ", w);
  startTime(&t);
  if(w == 8)
    bytwob_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes, (uint8_t *) data);
  else if(w == 16)
    bytwob_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/2, (uint16_t *) data);
  else if(w == 32)
    bytwob_w32_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/4, (uint32_t *) data);
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s) %f MB/s\n", elapsedTime(&t),
      (float) bytes / (1024*1024) / elapsedTime(&t));
}

void bytwop_launch(unsigned w, uint64_t c, unsigned bytes, void *data)
{
  Timer t;
  printf("BYTWO_p %02i launching... ", w);
  startTime(&t);
  if(w == 8)
    bytwop_w08_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes, (uint8_t *) data);
  else if(w == 16)
    bytwop_w16_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/2, (uint16_t *) data);
  else if(w == 32)
    bytwop_w32_kernel <<<GRID_SIZE, BLOCK_SIZE>>>(c, bytes/4, (uint32_t *) data);
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s) %f MB/s\n", elapsedTime(&t),
      (float) bytes / (1024 * 1024) / elapsedTime(&t));
}
