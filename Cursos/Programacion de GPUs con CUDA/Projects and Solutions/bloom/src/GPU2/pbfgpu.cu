#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../support.h"
#include "../pbf.h"
#include "../hash.h"
#include "../parseArgs.h"
#include "../rng.h"
#include "bloom.h"


/**
* Main function
*/
int main(int argc,char** argv){
	printf("Running GPU PBF...\n");

	//Does the user need help?
	if(wasArgSpecified("--help",argv,argc)!=0){
		printHelp();
		return 0;
	}

/////// CONFIG ////////
    char allKeyFile[128];
    char distinctKeyFile[128];
	strcpy(allKeyFile, argv[argc - 2]);
	strcpy(distinctKeyFile, argv[argc - 1]);
	clock_t start, end;
	double tt;


	//Initialize with default configuration.
	BloomOptions_t bloomOptions_t;
	setDefault(&bloomOptions_t);

	//Parse the user's configuration
	getConfiguration(&bloomOptions_t,argv,argc);
///////////////////////////


/////// LOAD ALL KEYS ////////
	//Read input keys
    size_t nbins, nkeys;
    unsigned int *bins = NULL;
    unsigned int *keyPos = NULL;
    char *keys = NULL;

	int err = loadRS(allKeyFile, &nbins, &bins, &nkeys, &keys, &keyPos);
	if (err) {
	  printf("Error in loading RNG data\n");
	  return 0;
	}

	printKeyInfo(nbins, bins, nkeys, keys, keyPos);
///////////////////////////


/////// CREATE HOST PBF ////////
    bloomOptions_t.numKeys = nkeys;
	bloomOptions_t.prob = calculatePBFProb((int)(nkeys / nbins), (int)nkeys);
	bloomOptions_t.size = calculatePBFSize(bloomOptions_t.numHashes, (int)nkeys,bloomOptions_t.prob);

	//Show the user the configuration.
	showDetails(&bloomOptions_t);

	//Create the bloom filter being used, and initailize with all 0's.
	char* bloom = (char*)calloc(bloomOptions_t.size, sizeof(char));
///////////////////////////


/////// CREATE GPU PBF ////////
    char *dbloom = NULL;
	size_t bloomBytes = bloomOptions_t.size * sizeof(char);
	CUDA_CALL(cudaMalloc((void **)&dbloom, bloomBytes));
	CUDA_CALL(cudaMemset(dbloom, 0, bloomBytes));
///////////////////////////


/////// INSERT PBF ////////
	int total_keys = nkeys;
	start = clock();
	insertWordsPBF(dbloom, bloomOptions_t.size, keys, keyPos, nkeys, bloomBytes, bloomOptions_t.numHashes, bloomOptions_t.device, bloomOptions_t.prob);
	end = clock();
	tt = (double)(end - start) / CLOCKS_PER_SEC;
	printf("GPU insert time: %.2f sec\n", tt);

	//Read the actual frequency of keys
	int *actual = (int *)malloc(nbins * sizeof(int));
	memcpy(actual, bins, nbins * sizeof(int));

    free(bins); bins = NULL;
    free(keyPos); keyPos = NULL;
    free(keys); keys = NULL;
////////////////////////////
	
	
/////// LOAD DISTINCT KEYS ////////
	err = loadRS(distinctKeyFile, &nbins, &bins, &nkeys, &keys, &keyPos);
	if (err) {
	  printf("Error in loading RNG data\n");
	  return 0;
	}
	printKeyInfo(nbins, bins, nkeys, keys, keyPos);
////////////////////////////


/////// QUERY PBF ////////
	//Query PBF
	int distinct_keys = nkeys;
	int *results = (int *)calloc(distinct_keys,sizeof(int));
	start = clock();
	queryWordsPBF(dbloom, bloomOptions_t.size, keys, keyPos, nkeys, bloomBytes, bloomOptions_t.numHashes, bloomOptions_t.device, results);
	end = clock();
	tt = (double)(end - start) / CLOCKS_PER_SEC;
	printf("GPU query time: %.2f sec\n", tt);

    free(bins); bins = NULL;
    free(keyPos); keyPos = NULL;
    free(keys); keys = NULL;
////////////////////////////

			
	//Write the result to output file
	//data format: index, number of 1s, calculated frequency, actual frequency, relative error
	if(bloomOptions_t.pbfOutput){
		FILE* outputFile = fopen(bloomOptions_t.pbfOutput,"w");
		writeStats(outputFile,actual,results,distinct_keys, bloomOptions_t.numHashes,bloomOptions_t.prob,total_keys,bloomOptions_t.size);
		fclose(outputFile);
	}
	
	free(actual);
	free(results);
	free(bloom);

	cudaFree(dbloom);

	return 0;
}
