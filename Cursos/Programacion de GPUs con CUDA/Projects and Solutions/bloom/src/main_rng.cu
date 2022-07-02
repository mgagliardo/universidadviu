#include <stdio.h>
#include <stdlib.h>

#include "support.h"
#include "rng.h"
#include "parseArgs.h"


int main(int argc, char **argv)
{
	int err = 0;
    distrType_t type;
	const char *fn, *dfn;
	size_t nbins; 
	distrParams_t params;

    //Does the user need help?
    if(wasArgSpecified("--help",argv,argc)!=0){
	  printf("Usage: RNG [--help]\n");
	  printf("Usage: RNG pdf\n");
	  printf("  pdf = 0, Uniform\n");
	  printf("  pdf = 1, Normal\n");
	  printf("  pdf = 2, Poisson\n");
	  printf("  pdf = 3, Random\n");
      return 0;
    }

	int sel = 0;
    if (argc > 1)
	  sel = atoi(argv[1]);

    params.keySz = 8;
	params.seed = 12345;
    switch(sel) {
        case 0:
          type = UNIFORM;
		  fn = "inputs/unif_data.txt";
		  dfn = "inputs/unif_distinct.txt";
          nbins = 10;
	      params.scale = 1000.0;
	      break;

	    case 1:
          type = NORMAL;
		  fn = "inputs/norm_data.txt";
		  dfn = "inputs/norm_distinct.txt";
          nbins = 20000;
	      params.scale = 1000.0;
	      params.a = 0.0;     // mean
	      params.b = 1.0;     // standard deviation
	      params.low = -10.0; // low bound of input value
	      params.high = 10.0; // high bound of input value
	      break;

	    case 2:
          type = POISSON;
		  fn = "inputs/pois_data.txt";
		  dfn = "inputs/pois_distinct.txt";
          nbins = 200;
	      params.scale = 700000.0;
	      params.a = 50.0;   // mean
	      params.low = 0.0; // low bound of input value
	      break;

	    case 3:
          type = RANDOM;
		  fn = "inputs/rand_data.txt";
		  dfn = "inputs/rand_distinct.txt";
          nbins = 1000;
	      params.a = 100.0; // max reps
	      break;

	    default:
		  printf("Invalid argument\n");
		  return 0;
	}

    err = genRS(type, fn, dfn, nbins, &params);
    if (err) {
	  printf("Error in RNG\n");
	  return 0;
    }

	return 0;
}

