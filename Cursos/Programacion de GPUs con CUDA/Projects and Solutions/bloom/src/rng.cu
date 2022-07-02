#include "rng.h"
#include "support.h"

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
int loadRS(const char *fn, size_t *nbins, unsigned **bins, size_t *nkeys, char **keys, unsigned int **keyPos)
{
   int err = 0;
   char *lkeys = NULL;
   size_t szKeys;
   char pkey[MAX_KEYLEN];
   size_t lnbins = 0;
   size_t lnkeys = 0;

   // Generate keys file 
   err = readKeysFromFile(fn, &(*keys), &lnbins);
   if (err)
     return err;
	
   // Get copy of keys
   szKeys = strlen((*keys));
   lkeys = (char *)malloc((szKeys + 1) * sizeof(char));
   strcpy(lkeys, (*keys));

   // Find the number of distinct and total keys
   char *key = strtok(lkeys, ",");
   if (!key) {
     *nbins = 0;
	 *nkeys = 0;
	 (*bins) = NULL;
	 (*keyPos) = NULL;
	 (*keys) = NULL;
	 return -1;
   }

   // Count keys
   while (key) {
     lnkeys++;
     key = strtok(NULL, ",");
   }
   lnkeys--;

   // Fill bins and key position arrays
   *nbins = lnbins;
   *nkeys = lnkeys;
   (*bins) = (unsigned int *)malloc(lnbins * sizeof(unsigned int));
   (*keyPos) = (unsigned int *)malloc(lnkeys * sizeof(unsigned int));

   // Get copy of keys
   lkeys = (char *)malloc((szKeys + 1) * sizeof(char));
   strcpy(lkeys, (*keys));

   // Get key and a copy
   key = strtok(lkeys, ",");
   strcpy(pkey, key);

   unsigned int bin = 0; 
   unsigned int pos = 0;
   unsigned int i = 0, j = 0;
   while (key) {
	 (*keyPos)[j++] = pos;
	 pos += strlen(key) + 1;
     if (strcmp(pkey, key) != 0) {
       strcpy(pkey, key);
	   (*bins)[i++] = bin;
	   bin = 0;
	 }
     bin++;
	 key = strtok(NULL, ",");
   }
   (*bins)[i] = bin;
   free(lkeys); lkeys = NULL;

   return err;
}


/**
* Generates a list of random strings (or numbers) where the number of occurrences
* of the strings has a uniform distribution. The length of the strings can be set
* as constant or randomized.
* 
* @param[in]  fn     Output filename
* @param[in]  dfn    Output filename
* @param[in]  nbins  Number of distinct keys
* @param[in]  params RNG parameters (scaling, seed)
* @return            Error code
**/
int uniformDistribution(const char *fn, const char *dfn, size_t nbins, distrParams_t *params)
{
   int err = 0;
   float scale;
   size_t i;
   unsigned int keySz;

   // RNG default and parameterized values
   if (params) {
	 scale = params->scale;
     srand(params->seed);
	 keySz = params->keySz;
   }
   else {
     scale = 100.0;
     srand(time(NULL));
	 keySz = 4;
   }

   // Generate bins array
   // Uniform RNG
   unsigned int *bins = (unsigned int *)malloc(nbins * sizeof(unsigned int));
   const unsigned int pdf = (unsigned int)lroundf((1.0 / nbins) * scale);
   long unsigned int nkeys = 0;
   for (i = 0; i < nbins; ++i) {
     bins[i] = pdf;
	 nkeys += pdf;
   }

   // Generate keys file 
   printf("Generating %lu keys, %zu bins\n", nkeys, nbins);
   err = writeKeysToFile(fn, dfn, keySz, nbins, bins);   
   free(bins); bins = NULL;
   if (err)
     return -1;

   return err;
}


/**
* Compute the normal (Gaussian) probability density function at each input value.
* 
* @param[in] mean   Normal mean
* @param[in] stddev Normal standard deviation
* @param[in] x      Input value
* @return           PDF at value x
**/
float normalPDF(float mean, float stddev, float x)
{
   const double pi2 = 2 * PI;
   double tmp1 = 1.0 / (stddev * sqrt(pi2));
   double tmp2 = -pow((x - mean), 2) / (2 * pow(stddev, 2));
   return (float)(tmp1 * exp(tmp2));
}


/**
* Generates a list of random strings (or numbers) where the number of occurrences
* of the strings has a normal (Gaussian) distribution. The length of the strings 
* can be set as constant or randomized.
* 
* @param[in]  fn     Output filename
* @param[in]  dfn    Output filename
* @param[in]  nbins  Number of distinct keys
* @param[in]  params RNG parameters (scaling, seed, mean, stddev)
* @return            Error code
**/
int normalDistribution(const char *fn, const char *dfn, size_t nbins, distrParams_t *params)
{
   int err = 0;
   float scale, mean, stddev, low, high;
   size_t i;
   unsigned int keySz;

   // RNG default and parameterized values
   if (params) {
	 scale = params->scale;
     srand(params->seed);
	 mean = params->a;
	 stddev = params->b;
	 low = params->low;
	 high = params->high;
	 keySz = params->keySz;
   }
   else {
     scale = 100.0;
     srand(time(NULL));
	 mean = 0.0;
	 stddev = 1.0;
	 low = -10.0;
	 high = 10.0;
	 keySz = 4;
   }

   // Generate range values for probability distribution 
   float step = (high - low) / nbins;
   float *range = (float *)malloc(nbins * sizeof(float));
   float fval = low;
   for (i = 0; i < nbins; ++i) {
     range[i] = fval;
	 fval += step;
   }

   // Generate bins array
   // Normal (Gaussian) RNG
   unsigned int *bins = (unsigned int *)malloc(nbins * sizeof(unsigned int));
   long unsigned int nkeys = 0;
   for (i = 0; i < nbins; ++i) {
     const unsigned int pdf = (unsigned int)lroundf(normalPDF(mean, stddev, range[i]) * scale);
     bins[i] = pdf;
	 nkeys += pdf;
   }
   free(range); range = NULL;
   
   // Generate keys array
   printf("Generating %lu keys, %zu\n", nkeys, nbins);
   err = writeKeysToFile(fn, dfn, keySz, nbins, bins);   
   free(bins); bins = NULL;
   if (err)
     return -1;

   return err;
} 


/**
* Compute the Poisson probability density function at each input value.
* 
* @param[in] mean   Mean
* @param[in] x      Input value
* @return           PDF at value x
**/
float poissonPDF(float mean, unsigned int x)
{
   double tmp = (pow(mean, x) / fact(x));
   return (float)(tmp * exp(-mean));
}


/**
* Generates a list of random strings (or numbers) where the number of occurrences
* of the strings has a Poisson distribution. 
* 
* @param[in]  fn     Output filename
* @param[in]  dfn    Output filename
* @param[in]  nbins  Number of distinct keys
* @param[in]  params RNG parameters (scaling, mean)
* @return            Error code
**/
int poissonDistribution(const char *fn, const char *dfn, size_t nbins, distrParams_t *params)
{
   int err = 0;
   float scale, mean, low;
   size_t i;
   size_t keySz;

   // RNG default and parameterized values
   if (params) {
	 scale = params->scale;
     srand(params->seed);
	 mean = params->a;
	 low = params->low;
	 keySz = params->keySz;
   }
   else {
     scale = 100.0;
     srand(time(NULL));
	 mean = 5.0;
	 low = 0.0;
	 keySz = 4;
   }

   // Generate range values for probability distribution 
   unsigned int *range = (unsigned int *)malloc(nbins * sizeof(unsigned int));
   unsigned int uval = low;
   for (i = 0; i < nbins; ++i) {
     range[i] = uval++;
   }

   // Generate bins array
   // Poisson RNG
   unsigned int *bins = (unsigned int *)malloc(nbins * sizeof(unsigned int));
   long unsigned int nkeys = 0;
   for (i = 0; i < nbins; ++i) {
     const unsigned int pdf = (unsigned int)lroundf(poissonPDF(mean, range[i]) * scale);
     bins[i] = pdf;
	 nkeys += pdf; 
   }
   free(range); range = NULL;
   
   // Generate keys file 
   printf("Generating %lu keys, %zu\n", nkeys, nbins);
   err = writeKeysToFile(fn, dfn, keySz, nbins, bins);   
   free(bins); bins = NULL;
   if (err)
     return -1;

   return err;
} 


/**
* Generates a list of random strings (or numbers) where the number of occurrences
* of the strings has a random distribution. 
* 
* @param[in]  fn     Output filename 
* @param[in]  dfn    Output filename 
* @param[in]  nbins  Number of distinct keys
* @param[in]  params RNG parameters (scaling, seed)
* @return            Error code
**/
int randomDistribution(const char *fn, const char *dfn, size_t nbins, distrParams_t *params)
{
   int err = 0;
   unsigned int maxReps;
   size_t i;
   size_t keySz;

   // RNG default and parameterized values
   if (params) {
     srand(params->seed);
	 maxReps = (unsigned int)params->a;
	 keySz = params->keySz;
   }
   else {
     srand(time(NULL));
	 maxReps = 10;
	 keySz = 4;
   }

   // Generate bins array
   // Random RNG
   unsigned int *bins = (unsigned int *)malloc(nbins * sizeof(unsigned int));
   long unsigned int nkeys = 0;
   for (i = 0; i < nbins; ++i) {
     const unsigned int pdf = (unsigned int)rand() % maxReps + 1;
     bins[i] = pdf;
     nkeys += pdf;
   }

   // Generate keys file 
   printf("Generating %lu keys, %zu\n", nkeys, nbins);
   err = writeKeysToFile(fn, dfn, keySz, nbins, bins);   
   free(bins); bins = NULL;

   return err;
}


/**
* Execute a RNG with specified probability disttribution for occurrences of keys.
**/
int genRS(distrType_t type, const char *fn, const char *dfn, size_t nbins, distrParams_t *params)
{
   int err = 0;

   switch(type) {
     case UNIFORM:
            printf("Distribution: uniform\n");
            err = uniformDistribution(fn, dfn, nbins, params);
            break;
     
     case NORMAL:
            printf("Distribution: normal (Gaussian)\n");
            err = normalDistribution(fn, dfn, nbins, params);
            break;

     case POISSON:
            printf("Distribution: Poisson\n");
            err = poissonDistribution(fn, dfn, nbins, params);
            break;

     case RANDOM:
            printf("Distribution: random\n");
            err = randomDistribution(fn, dfn, nbins, params);
            break;

     default:
            printf("Error: Invalid distribution type selected.\n");
			err = -1;
            break;    
   }

   return err;
}

