#include "gf_gpu.h"

void *random_data(unsigned seed, unsigned bytes)
{
  if(bytes % 4)
    usage("Bytes must be multiple of 4");

  srand(seed);
  uint32_t *rv = (uint32_t *)malloc(bytes);
  unsigned index = 0;
  while(bytes)
  {
    rv[index++] = rand();
    bytes -= 4;
  }

  return (void *)rv;
}
