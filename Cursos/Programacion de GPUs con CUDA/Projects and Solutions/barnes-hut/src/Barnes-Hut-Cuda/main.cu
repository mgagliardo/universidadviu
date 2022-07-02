#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.cu"

/******************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}


/******************************************************************************/

// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}


/******************************************************************************/

int main(int argc, char *argv[])
{
  register int i, run, blocks;
  int nnodes, nbodies, step, timesteps;
  register double runtime;
  register float dtime, dthf, epssq, itolsq;
  float time, timing[7];
  cudaEvent_t start, stop;
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz;

  int  *sortl, *childl, *countl, *startl;
  float *massl;
  float *posxl, *posyl, *poszl;
  float *velxl, *velyl, *velzl;
  float *accxl, *accyl, *acczl;
  float *maxxl, *maxyl, *maxzl;
  float *minxl, *minyl, *minzl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;

  // perform some checks

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (argc != 4) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps device\n");
    exit(-1);
  }

  

  const int dev = atoi(argv[3]);
  if ((dev < 0) || (deviceCount <= dev)) {
    fprintf(stderr, "There is no device %d\n", dev);
    exit(-1);
  }
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  
  
  

  blocks = deviceProp.multiProcessorCount;
//  fprintf(stderr, "blocks = %d\n", blocks);

  
  if (MAXDEPTH > WARPSIZE) {
    fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
    exit(-1);
  }
  if ((T1 <= 0) || (T1 & (T1-1) != 0)) {
    fprintf(stderr, "T1 must be greater than zero and a power of two\n");
    exit(-1);
  }

  // set L1/shared memory configuration
  cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(InitializationKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(InitializationKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(CoGKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ForceKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(UpdateKernel, cudaFuncCachePreferL1);
  cudaGetLastError();  // reset error value
  for (run = 0; run < 1; run++) {
    for (i = 0; i < 7; i++) timing[i] = 0.0f;

    nbodies = atoi(argv[1]);
    if (nbodies < 1) {
      fprintf(stderr, "nbodies is too small: %d\n", nbodies);
      exit(-1);
    }
    if (nbodies > (1 << 30)) {
      fprintf(stderr, "nbodies is too large: %d\n", nbodies);
      exit(-1);
    }
    nnodes = nbodies * 2;
    if (nnodes < 1024*blocks) nnodes = 1024*blocks;
    while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
    nnodes--;

    timesteps = atoi(argv[2]);
    dtime = 0.025;  dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;
    itolsq = 1.0f / (0.5 * 0.5);

    // allocate memory

    if (run == 0) {
      printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

      mass = (float *)malloc(sizeof(float) * nbodies);
      if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
      posx = (float *)malloc(sizeof(float) * nbodies);
      if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
      posy = (float *)malloc(sizeof(float) * nbodies);
      if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
      posz = (float *)malloc(sizeof(float) * nbodies);
      if (posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
      velx = (float *)malloc(sizeof(float) * nbodies);
      if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
      vely = (float *)malloc(sizeof(float) * nbodies);
      if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
      velz = (float *)malloc(sizeof(float) * nbodies);
      if (velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}

      //if (cudaSuccess != cudaMalloc((void **)&errl, sizeof(int))) fprintf(stderr, "could not allocate errd\n");  CudaTest("couldn't allocate errd");
      if (cudaSuccess != cudaMalloc((void **)&childl, sizeof(int) * (nnodes+1) * 8)) fprintf(stderr, "could not allocate childd\n");  CudaTest("couldn't allocate childd");
      if (cudaSuccess != cudaMalloc((void **)&massl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate massd\n");  CudaTest("couldn't allocate massd");
      if (cudaSuccess != cudaMalloc((void **)&posxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posxd\n");  CudaTest("couldn't allocate posxd");
      if (cudaSuccess != cudaMalloc((void **)&posyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posyd\n");  CudaTest("couldn't allocate posyd");
      if (cudaSuccess != cudaMalloc((void **)&poszl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate poszd\n");  CudaTest("couldn't allocate poszd");
      if (cudaSuccess != cudaMalloc((void **)&velxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velxd\n");  CudaTest("couldn't allocate velxd");
      if (cudaSuccess != cudaMalloc((void **)&velyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velyd\n");  CudaTest("couldn't allocate velyd");
      if (cudaSuccess != cudaMalloc((void **)&velzl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velzd\n");  CudaTest("couldn't allocate velzd");
      if (cudaSuccess != cudaMalloc((void **)&accxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate accxd\n");  CudaTest("couldn't allocate accxd");
      if (cudaSuccess != cudaMalloc((void **)&accyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate accyd\n");  CudaTest("couldn't allocate accyd");
      if (cudaSuccess != cudaMalloc((void **)&acczl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate acczd\n");  CudaTest("couldn't allocate acczd");
      if (cudaSuccess != cudaMalloc((void **)&countl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate countd\n");  CudaTest("couldn't allocate countd");
      if (cudaSuccess != cudaMalloc((void **)&startl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate startd\n");  CudaTest("couldn't allocate startd");
      if (cudaSuccess != cudaMalloc((void **)&sortl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate sortd\n");  CudaTest("couldn't allocate sortd");

      if (cudaSuccess != cudaMalloc((void **)&maxxl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate maxxd\n");  CudaTest("couldn't allocate maxxd");
      if (cudaSuccess != cudaMalloc((void **)&maxyl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate maxyd\n");  CudaTest("couldn't allocate maxyd");
      if (cudaSuccess != cudaMalloc((void **)&maxzl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate maxzd\n");  CudaTest("couldn't allocate maxzd");
      if (cudaSuccess != cudaMalloc((void **)&minxl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate minxd\n");  CudaTest("couldn't allocate minxd");
      if (cudaSuccess != cudaMalloc((void **)&minyl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate minyd\n");  CudaTest("couldn't allocate minyd");
      if (cudaSuccess != cudaMalloc((void **)&minzl, sizeof(float) * blocks * F1)) fprintf(stderr, "could not allocate minzd\n");  CudaTest("couldn't allocate minzd");
    }

    // generate input

    drndset(7);
    rsc = (3 * 3.1415926535897932384626433832795) / 16;
    vsc = sqrt(1.0 / rsc);
    for (i = 0; i < nbodies; i++) {
      mass[i] = 1.0 / nbodies;
      r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);
      posx[i] = x * scale;
      posy[i] = y * scale;
      posz[i] = z * scale;

      do {
        x = drnd();
        y = drnd() * 0.1;
      } while (y > x*x * pow(1 - x*x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r*r));
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      velx[i] = x * scale;
      vely[i] = y * scale;
      velz[i] = z * scale;
    }

    if (cudaSuccess != cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of mass to device failed\n");  CudaTest("mass copy to device failed");
    if (cudaSuccess != cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to device failed\n");  CudaTest("posx copy to device failed");
    if (cudaSuccess != cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to device failed\n");  CudaTest("posy copy to device failed");
    if (cudaSuccess != cudaMemcpy(poszl, posz, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posz to device failed\n");  CudaTest("posz copy to device failed");
    if (cudaSuccess != cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velx to device failed\n");  CudaTest("velx copy to device failed");
    if (cudaSuccess != cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of vely to device failed\n");  CudaTest("vely copy to device failed");
    if (cudaSuccess != cudaMemcpy(velzl, velz, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velz to device failed\n");  CudaTest("velz copy to device failed");

    // run timesteps (launch GPU kernels)

    cudaEventCreate(&start);  cudaEventCreate(&stop);  
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);

    cudaEventRecord(start, 0);
    InitializationKernel<<<1, 1>>>();
    cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
    timing[0] += time;
    CudaTest("kernel 0 launch failed");
	printf("Start of timestep \n");
	
    for (step = 0; step < timesteps; step++)
	{
	
	printf("TIMESTEP = %d \n", step);
      cudaEventRecord(start, 0);
      BoundingBoxKernel<<<blocks * F1, T1>>>(nnodes, nbodies, startl, childl, massl, posxl, posyl, poszl, maxxl, maxyl, maxzl, minxl, minyl, minzl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[1] += time;
      CudaTest("kernel 1 launch failed");

      cudaEventRecord(start, 0);
      InitializationKernel1<<<blocks * 1, 1024>>>(nnodes, nbodies, childl);
      TreeBuildingKernel<<<blocks * F2, T2>>>(nnodes, nbodies, childl, posxl, posyl, poszl);
      InitializationKernel2<<<blocks * 1, 1024>>>(nnodes, startl, massl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[2] += time;
      CudaTest("kernel 2 launch failed");

      cudaEventRecord(start, 0);
      CoGKernel<<<blocks * F3, T3>>>(nnodes, nbodies, countl, childl, massl, posxl, posyl, poszl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[3] += time;
      CudaTest("kernel 3 launch failed");

      cudaEventRecord(start, 0);
      SortKernel<<<blocks * F4, T4>>>(nnodes, nbodies, sortl, countl, startl, childl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[4] += time;
      CudaTest("kernel 4 launch failed");

      cudaEventRecord(start, 0);
      ForceKernel<<<blocks * F5, T5>>>(nnodes, nbodies, dthf, itolsq, epssq, sortl, childl, massl, posxl, posyl, poszl, velxl, velyl, velzl, accxl, accyl, acczl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[5] += time;
      CudaTest("kernel 5 launch failed");

      cudaEventRecord(start, 0);
      UpdateKernel<<<blocks * F6, T6>>>(nbodies, dtime, dthf, posxl, posyl, poszl, velxl, velyl, velzl, accxl, accyl, acczl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[6] += time;
      CudaTest("kernel 6 launch failed");
    }
    CudaTest("kernel launch failed");
    cudaEventDestroy(start);  cudaEventDestroy(stop);

    // transfer result back to CPU
    //if (cudaSuccess != cudaMemcpy(&error, errl, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from device failed\n");  CudaTest("err copy from device failed");
    if (cudaSuccess != cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posx from device failed\n");  CudaTest("posx copy from device failed");
    if (cudaSuccess != cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posy from device failed\n");  CudaTest("posy copy from device failed");
    if (cudaSuccess != cudaMemcpy(posz, poszl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posz from device failed\n");  CudaTest("posz copy from device failed");
    if (cudaSuccess != cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of velx from device failed\n");  CudaTest("velx copy from device failed");
    if (cudaSuccess != cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of vely from device failed\n");  CudaTest("vely copy from device failed");
    if (cudaSuccess != cudaMemcpy(velz, velzl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of velz from device failed\n");  CudaTest("velz copy from device failed");

    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec + endtime.tv_usec/1000000.0 - starttime.tv_sec - starttime.tv_usec/1000000.0;

    printf("runtime: %.4lf s  (", runtime);
    time = 0;
    for (i = 1; i < 7; i++) {
      printf(" %.1f ", timing[i]);
      time += timing[i];
    }
     printf(") = %.1f ms\n", time);
  }

  // print output
  i = 0;
//  for (i = 0; i < nbodies; i++) {
//    printf("end position of body 0 %.2e %.2e %.2e\n", posx[i], posy[i], posz[i]);
//  }

  free(mass);
  free(posx);
  free(posy);
  free(posz);
  free(velx);
  free(vely);
  free(velz);

  //cudaFree(errl);
  cudaFree(childl);
  cudaFree(massl);
  cudaFree(posxl);
  cudaFree(posyl);
  cudaFree(poszl);
  cudaFree(countl);
  cudaFree(startl);

  cudaFree(maxxl);
  cudaFree(maxyl);
  cudaFree(maxzl);
  cudaFree(minxl);
  cudaFree(minyl);
  cudaFree(minzl);

  return 0;
}
