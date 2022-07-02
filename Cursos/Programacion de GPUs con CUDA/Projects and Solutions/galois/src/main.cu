#include "gf_gpu.h"

void usage(const char *msg)
{
  if(msg)
    fprintf(stderr, "ERROR: %s\n", msg);
  fprintf(stderr, "usage: gf_gpu w c seed bytes iterations technique\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  if(argc < 7)
    usage(NULL);
  unsigned w, bytes, iterations;
  int seed;
  uint64_t c;

  if(!sscanf(argv[1], "%u", &w))
    usage("Couldn't read w");
  if(!sscanf(argv[2], "%lu", &c))
    usage("Couldn't read c");
  if(!sscanf(argv[3], "%d", &seed))
    usage("Couldn't read seed");
  if(!sscanf(argv[4], "%u", &bytes))
    usage("Couldn't read bytes");
  if(!sscanf(argv[5], "%u", &iterations))
    usage("Couldn't read iterations");
  if(c >= (1ULL << w))
    usage("c is not an element of the field");

  cudaDeviceReset(); //Reset for nvprof
  Timer t;
  printf("Generating data... ");
  startTime(&t);
  void *data = random_data(seed, bytes);
  void *cuda_data;
  cudaError_t cuda_ret = cudaMalloc((void**)&cuda_data, bytes);
  if(cuda_ret != cudaSuccess)
    usage(cudaGetErrorString(cuda_ret));

  cuda_ret = cudaMemcpy(cuda_data, data, bytes, cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess)
    usage(cudaGetErrorString(cuda_ret));
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s)\n", elapsedTime(&t));

  //Let the kernel launchers time themselves because they have different setups
  if(!strcmp(argv[6], "SHIFT"))
    for(int i = 0; i < iterations; i++)
      shift_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "TABLE"))
    table_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "LOG"))
    log_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "BYTWO_b"))
    bytwob_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "BYTWO_p"))
    bytwop_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "SPLIT"))
    split_launch(w, c, bytes, cuda_data);
  else if(!strcmp(argv[6], "GROUP"))
    group_launch(w, c, bytes, cuda_data);
  else
    usage("Invalid technique");

  printf("Copying answer to host... ");
  startTime(&t);
  void *answer = malloc(bytes);
  cuda_ret = cudaMemcpy(answer, cuda_data, bytes, cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess)
    usage(cudaGetErrorString(cuda_ret));
  cudaDeviceSynchronize();
  stopTime(&t);
  printf("(%f s)\n", elapsedTime(&t));

  if(w == 8)
    check_answer_w08(c, bytes, (uint8_t *)data, (uint8_t *)answer);
  else if(w == 16)
    check_answer_w16(c, bytes/2, (uint16_t *)data, (uint16_t *)answer);
  else if(w == 32)
    check_answer_w32(c, bytes/4, (uint32_t *)data, (uint32_t *)answer);

  free(answer);
  free(data);
  cudaFree(cuda_data);
  return 0;
}
