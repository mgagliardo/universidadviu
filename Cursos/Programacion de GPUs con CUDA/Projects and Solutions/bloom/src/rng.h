#ifndef RNG_H
#define RNG_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "support.h"

#define PI 3.14159265358979323846

/** 
* Type of random sequence
**/
typedef enum rngType {UNIFORM = 0, NORMAL, POISSON, RANDOM} distrType_t;


/**
* Parameters for random sequence generator
**/
typedef struct rngParams {
	unsigned long long int seed;
	unsigned int keySz;  // 0 = random, # = constant
	float scale;
	float a;  // (Normal, Poisson)mean, (Random)max repeats
	float b;  // (Normal)stddev
	float low; 
	float high;
} distrParams_t;


/**
  * Execute a RNG with specified probability disttribution for occurrences of keys.
  *
  * @param[in]  type   Probability distribution (uniform, normal, Poisson, random)
  * @param[in]  fn     Output filename
  * @param[in]  dfn    Output filename
  * @param[in]  nbins  Number of distinct keys
  * @param[in]  params RNG parameters (scaling, seed, mean, stddev, low, high)
  * @return            Error code
**/
int genRS(distrType_t type, const char *fn, const char *dfn, size_t nbins, distrParams_t *params);


/**
  * Load list of random strings into array structures.
  *
  * @param[in]  fn     Output filename
  * @param[out] nbins  Number of distinct keys
  * @param[out] bins   Array containing the occurrences of each key
  * @param[out] nkeys  Number of total keys
  * @param[out] keys   List of keys
  * @param[out] keyPos Array containing the initial position of each key 
  * @return            Error code
**/
int loadRS(const char *fn, size_t *nbins, unsigned **bins, size_t *nkeys, char **keys, unsigned int **keyPos);

#endif // RNG_H

