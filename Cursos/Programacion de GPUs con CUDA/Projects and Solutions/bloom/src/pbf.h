#ifndef PBF_H
#define PBF_H

#include <stdio.h>
#include <stdlib.h>

typedef struct BloomOptions {
	unsigned int size;
	int numHashes;
	int numKeys;
	int device;
	char *fileName;
	char *pbfOutput;
	int freq;
	float prob;
	unsigned char numThreads;
} BloomOptions_t;

void writeStats(FILE *, int *, int *, int, int, float, int, int);
int calculatePBFSize(int, int, float);
float calculatePBFProb(int, int);

#endif // PBF_H

