#ifndef SUPPORT_H
#define SUPPORT_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CALL(x) if ((x)!=cudaSuccess) {       \
	printf("Error at %s:%d\n", __FILE__, __LINE__);\
	return x;}

#define CURAND_CALL(x) if ((x)!=CURAND_STATUS_SUCCESS) {\
	printf("Error at %s:%d\n", __FILE__, __LINE__);     \
	return x;}

#define CUBLAS_CALL(x) if ((x)!=CUBLAS_STATUS_SUCCESS) {\
	printf("Error at %s:%d\n", __FILE__, __LINE__);     \
	return x;}

#define MAX_KEYLEN 50


/**
  * Compute the factorial of an integral number.
  *
  * @param[in] x Input value
  * @return      Factorial of input value
**/
float fact(unsigned int x);


/**
  * Based on a bin array, random strings are generated as the list of keys.
  * The list of keys is written to a file.
  *
  * @param[in] fn    Output filename for all keys
  * @param[in] dfn   Output filename for distinct keys
  * @param[in] keySz Specify the constant or random size of keys 
  * @param[in] nbins Number of distinct keys
  * @param[in] bins  Array containing the number of each key
  * @return          Error code
**/
int writeKeysToFile(const char *fn, const char *dfn, size_t keySz, size_t nbins, unsigned int *bins);


/**
  * Read list of keys from a file.
  *
  * @param[in]  fn    Input filename
  * @param[out] keys  List of keys
  * @param[out] nbins Bin count 
  * @return           Error code
**/
int readKeysFromFile(const char *fn, char **keys, size_t *nbins);


/**
  * Display information of keys.
**/
void printKeyInfo(size_t nbins, unsigned int *bins, size_t nkeys, char *keys, unsigned int *keyPos);

#endif // SUPPORT_H

