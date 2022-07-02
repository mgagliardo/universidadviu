#ifndef BLOOM_H
#define BLOOM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>

#include "../parseArgs.h"
#include "../support.h"


/**
* Responsible for inserting words into the bloom filter.
*/
extern cudaError_t insertWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device);

/**
* Responsible for inserting words into the PBF bloom filter.
*/
extern cudaError_t insertWordsPBF(char* dev_bloom,int size,char* words,
	unsigned int* offsets,int numWords,int numBytes,int numHashes,int device,float prob);


/**
* Responsible for querying words inserted into the bloom filter
*/
extern cudaError_t queryWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
		char* result);

/**
* Responsible for querying words inserted into the PBF bloom filter
*/
extern cudaError_t queryWordsPBF(char* dev_bloom,int size,char* words,
	unsigned int* offsets,int numWords,int numBytes,int numHashes,int device,
		int* result);

#endif // BLOOM_H

