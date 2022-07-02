#include "support.h"


/**
* Compute the factorial of an integral number.
**/
float fact(unsigned int x)
{
   unsigned int i;
   float fval = 1;
   for (i = 1; i <= x; ++i)
     fval *= i;

   return fval;
}


/**
* Generate a random string with a specified length in bytes.
* @param[in]  nbytes Length of random string in bytes 
* @return            Random string
**/
char * generateRandomString(size_t nbytes)
{
   char *str = (char *)malloc((nbytes) * sizeof(char));
   size_t i;

   // Generate uniform characters for each random sequence
   for (i = 0; i < nbytes - 1; ++i) {
     //str[i] = (char)(rand() % 94 + 45);
     str[i] = (char)(rand() % 93 + 33);
	 while (str[i] == ',') 
       str[i] = (char)(rand() % 93 + 33);
   }
   str[nbytes - 1] = ',';

   return str;
}


/**
* Based on a bin array, random strings are generated as the list of keys.
* The list of keys is written to a file.
**/
int writeKeysToFile(const char *fn, const char *dfn, size_t keySz, size_t nbins, unsigned int *bins)
{
   int err = 0;
   size_t i, j;
   size_t kbytes;
   FILE *fd = fopen(fn, "w");
   FILE *dfd = fopen(dfn, "w");
   FILE *bfd = fopen("pdf.txt", "w");

   // Write bin count at beginning
   fprintf(fd, "%zu\n", nbins);
   fprintf(dfd, "%zu\n", nbins);

   kbytes = keySz;
   for (i = 0; i < nbins; ++i) {
     if (keySz == 0)
       kbytes = (unsigned int)rand() % MAX_KEYLEN + 1;

     // Write bins of keys to output file, for plotting purposes
	 fprintf(bfd, "%u,", bins[i]);

     // Check if last iteration of bins
	 if (i == nbins - 1) {
       fwrite("\n", sizeof(char), 1, bfd);

       // Skip if no occurrences
	   if (bins[i] == 0) {
         fwrite("\n", sizeof(char), 1, fd);
	     continue;
       }
     }

     // Skip if no occurrences
	 if (bins[i] == 0)
	   continue;

     // Generate random sequence
     char *key = generateRandomString(kbytes);

     // Write each distinct key to output file
     fwrite(key, sizeof(char), kbytes, dfd);
	 if (i == nbins - 1)
       fwrite("\n", sizeof(char), 1, dfd);

     // Write all keys to output file
     for (j = 0; j < bins[i]; ++j) {
       fwrite(key, sizeof(char), kbytes, fd);
	   if ((i == nbins - 1) && (j == bins[i] - 1))
         fwrite("\n", sizeof(char), 1, fd);
     }

     free(key);
   }
   fclose(fd); fd = NULL;
   fclose(dfd); dfd = NULL;
   fclose(bfd); bfd = NULL;
   
   return err;
}


/**
* Read list of keys from a file.
*
* @param[in]  fn    Input filename
* @param[out] keys  List of keys
* @param[out] nbins Bin count 
* @return           Error code
**/
int readKeysFromFile(const char *fn, char **keys, size_t *nbins)
{
   int err = 0;

   FILE *fd = fopen(fn,"r");
   if (!fd)
     return -1;

   fseek(fd, 0L, SEEK_END);
   size_t fsz = ftell(fd);
   rewind(fd);

   if (!fsz) {
     (*keys) = NULL;
	 return -1;
   }

   // Read bin count 
   char b[128];
   fgets(b, 128, fd);
   *nbins = (size_t)strtoul(b, NULL, 10);
   fsz -= strlen(b);

   // Read keys
   (*keys) = (char *)malloc((fsz + 1) * sizeof(char));
   size_t bW = fread((*keys), fsz, 1, fd);
   (*keys)[fsz] = '\0';
   fclose(fd);

   if (bW != 1) {
     printf("Error reading keys\n");
     err = -1;
   }

   return err;
}

/**
* Display information of keys.
**/
void printKeyInfo(size_t nbins, unsigned int *bins, size_t nkeys, char *keys, unsigned int *keyPos)
{
	size_t i;
	printf("Number of bins: %zu\n", nbins);
/*
	for (i = 0; i < nbins; ++i)
	   printf("%u, ", bins[i]);
    printf("\n\n");

	printf("Positions of keys: %zu\n", nkeys);
	for (i = 0; i < nbins; ++i)
      printf("%u, ", keyPos[i]);
    printf("\n\n");
*/

	printf("Number of keys: %zu\n", nkeys);
//	printf("%s\n\n", keys);
	printf("\n");
}

