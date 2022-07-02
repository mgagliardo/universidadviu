#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../support.h"
#include "../pbf.h"
#include "../hash.h"
#include "../parseArgs.h"
#include "../rng.h"


/**
* Inserts words into the pbf.
*/
void insertWords(char *bloom, BloomOptions_t *bloomOptions_t, size_t nkeys, char *keys, unsigned int *keyPos, float prob)
{
	int i;			
	for (i = 0; i < nkeys; ++i) {
		int j, value; 
		unsigned long firstValue, secondValue;
		float temp;
		omp_set_num_threads(bloomOptions_t->numThreads);
		#pragma omp parallel for private(firstValue, secondValue, value, temp)
		for (j = 0; j < bloomOptions_t->numHashes; ++j) {
			firstValue = djb2HashOffset(keys, (int)keyPos[i]) % bloomOptions_t->size;
			secondValue = sdbmHashOffset(keys, (int)keyPos[i]) % bloomOptions_t->size;
			value = (firstValue + (j * j * secondValue) % bloomOptions_t->size) % bloomOptions_t->size;
			temp = (float)rand() / RAND_MAX;					
			if (temp < prob)	
				bloom[value] = 1;
		}
	}
}

/**
* Responsible for querying the bloom filter.
*/
void queryWords(char *bloom, BloomOptions_t *bloomOptions_t, size_t nkeys, char *keys, unsigned int *keyPos, int *results) {	
	int i;
	for(i = 0; i < nkeys; ++i) {
		int j, value, count = 0;
		unsigned long firstValue, secondValue;
		#pragma omp parallel for private(firstValue, secondValue, value) reduction(+:count)
		for(j = 0; j < bloomOptions_t->numHashes; ++j) {
			firstValue = djb2HashOffset(keys, keyPos[i]) % bloomOptions_t->size;
			secondValue = sdbmHashOffset(keys, keyPos[i]) % bloomOptions_t->size;
			value = (firstValue + (j * j * secondValue) % bloomOptions_t->size) % bloomOptions_t->size;
			count += bloom[value];
		}				
		results[i] = count;		
	}
}

/**
* Main function
*/
int main(int argc,char** argv){
	printf("Running CPU OpenMP PBF...\n");

	//Does the user need help?
	if(wasArgSpecified("--help",argv,argc)!=0){
		printHelp();
		return 0;
	}

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

/////// INSERT PBF ////////
	//Read input keys and insert to PBF
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

/// Create Bloom filter ///
    bloomOptions_t.numKeys = nkeys;
	bloomOptions_t.prob = calculatePBFProb((int)(nkeys / nbins), (int)nkeys);
	bloomOptions_t.size = calculatePBFSize(bloomOptions_t.numHashes, (int)nkeys,bloomOptions_t.prob);

	//Show the user the configuration.
	showDetails(&bloomOptions_t);

	//Create the bloom filter being used, and initailize with all 0's.
	char* bloom = (char*)calloc(bloomOptions_t.size, sizeof(char));
///////////////////////////


	int total_keys = nkeys;
	start = clock();
	insertWords(bloom, &bloomOptions_t, nkeys, keys, keyPos, bloomOptions_t.prob);
	end = clock();
	tt = (double)(end - start) / CLOCKS_PER_SEC;
	printf("OMP insert time: %.2f\n", tt);

	//Read the actual frequency of keys
	int *actual = (int *)malloc(nbins * sizeof(int));
	memcpy(actual, bins, nbins * sizeof(int));

    free(bins); bins = NULL;
    free(keyPos); keyPos = NULL;
    free(keys); keys = NULL;
////////////////////////////
	
/////// QUERY PBF ////////
	//Query PBF
	err = loadRS(distinctKeyFile, &nbins, &bins, &nkeys, &keys, &keyPos);
	if (err) {
	  printf("Error in loading RNG data\n");
	  return 0;
	}
	printKeyInfo(nbins, bins, nkeys, keys, keyPos);

	int distinct_keys = nkeys;
	int *results = (int *)calloc(distinct_keys,sizeof(int));
	start = clock();
	queryWords(bloom, &bloomOptions_t, nkeys, keys, keyPos, results);
	end = clock();
	tt = (double)(end - start) / CLOCKS_PER_SEC;
	printf("OMP query time: %.2f\n", tt);

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

	return 0;
}
