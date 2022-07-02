#include "parseArgs.h"
#include "pbf.h"

/**
* Prints the help about the program.
* 
*/
void printHelp(){
	printf("\n===============\n");
	printf("--size [num] -s [num] The size of the bloom filter in bits. \n");
	printf("--hashes [num] -k [num] The number of hash functions per word inserted \n");
	printf("--numKeys [num] -n [num] Holds the size of the batch being inserted. \n");
	printf("-f [num] Threshold frequency. \n");
	printf("--help\n");
	printf("--file [fileName] [fileName] Where the bloom filter should be outputted to. \n");
	printf("--prob [float] -p [float] The probabiltiy  (PBF Only) \n");
	printf("--pbfOutput [fileName] The output of the PBF stats. (PBF)\n");
	printf("--numThreads [num] -t [num] The number of executing threads. \n");
	printf("\n===============\n");
}

/**
* Determines if a particular argument was specified.
* @param argument The argument being looked for.
* @param args The arguments the user specified.
* @param argc The number of arguments the user specified.
*/
int wasArgSpecified(const char* argument,char** args,int argc){
	int i = 0;
	for(i = 0; args[i]!=0; i++){
		if(strcmp(argument,args[i])==0){
			return 1;
		}
	}
	return 0;
}

/**
* Returns the value of the argument..
* @param argument The argument being searched for.
* @param args The argument list specified by the user.
* @param argc The number of arguments specifeid.
* @return Returns the specified value of the argument if it exists.
*/
char* getArgValue(const char* argument,char** args,int argc){
	int i = 0;
	for(i = 0; args[i]!=0 ; i++){
		if(strcmp(argument,args[i])==0){
			if(i<argc-1){
				return args[i+1];
			}
		}
	}
	return 0;
}

/**
* Parse the bloom filter options specified by the user.
* @param bloomOptions The options to be created.
* @param args The arguments specified by the user.
* @param argc The number of arguments specified by the user. 
*/
void getConfiguration(BloomOptions_t* bloomOptions,char** args,int argc){
	char* value = getArgValue("--size",args,argc);
	if(value!=0){
		bloomOptions->size = atoi(value);		
	}
	value = getArgValue("-m",args,argc);
	if(value!=0){
		bloomOptions->size = atoi(value);
	}
	value = getArgValue("--hashes",args,argc); 
	if(value!=0){
		bloomOptions->numHashes = atoi(value);	
	}
	value = getArgValue("-k",args,argc); 
	if(value!=0){
		bloomOptions->numHashes = atoi(value);
	}
	value = getArgValue("--numKeys",args,argc);
	if(value!=0){
		bloomOptions->numKeys = atoi(value);	
	}
	value = getArgValue("-n",args,argc);
	if(value!=0){
		bloomOptions->numKeys = atoi(value);
	}
	value = getArgValue("--file",args,argc);
	if(value!=0){
		bloomOptions->fileName = value;
	}
	value = getArgValue("-f",args,argc);
	if(value!=0){
		bloomOptions->freq = atoi(value);
	}
	value = getArgValue("--prob",args,argc); 
	if(value!=0){
		bloomOptions->prob = atof(value);	
	}
	value = getArgValue("-p",args,argc); 
	if(value!=0){
		bloomOptions->prob = atof(value);
	}
	value = getArgValue("--pbfOutput",args,argc);
	if(value!=0){
		bloomOptions->pbfOutput = value;
	}
	value = getArgValue("--numThreads",args,argc);
	if(value!=0){
		bloomOptions->numThreads = atoi(value);
	}
	value = getArgValue("-t",args,argc);
	if(value!=0){
		bloomOptions->numThreads = atoi(value);
	}
}

/**
* Sets the default parameters of the bloom filter (if a parameter is not specified this value will be used
* @param bloomOptions A pointer to the bloom filter being used.
*/
void setDefault(BloomOptions_t* bloomOptions){
	bloomOptions->size = 100000;	
	bloomOptions->numHashes = 100;
	bloomOptions->numKeys = 1000;
	bloomOptions->device = 0;
	bloomOptions->fileName = 0;
	bloomOptions->pbfOutput = 0;
	bloomOptions->prob = .2f;
	bloomOptions->freq = 100;
	bloomOptions->numThreads = 1;
}


/**
* Show the detail of the bloom filter.
*/ 
void showDetails(BloomOptions_t* bloomOptions){
	printf("PBF parameters\n");
	printf("size: %i \n",bloomOptions->size);
	printf("numHashes: %i \n",bloomOptions->numHashes);
	printf("numKeys: %i \n",bloomOptions->numKeys);
	printf("Probability (PBF): %.6f \n",bloomOptions->prob);	
	printf("Threshold frequency: %d \n",bloomOptions->freq);
	printf("Number of threads: %i \n",bloomOptions->numThreads);
	printf("\n");
}

/**
* Responsible for writing the bloom filter to a file.
* @param *bloomOptions A pointer to the options used to describe the bloom filter.
* @param *bloom A pointer to the bloom filter. 
*/
void writeBloomFilterToFile(BloomOptions_t* bloomOptions,char* bloom){
	//Copy the characters, not the values.
	char* outputBloom = (char*)malloc(sizeof(char)*bloomOptions->size);
	int i = 0;
	for(i = 0; i<bloomOptions->size;i++){
		outputBloom[i] = (bloom[i])? '1':'0';
	}
	FILE* bloomFile = fopen(bloomOptions->fileName,"w+");
	fwrite(outputBloom,sizeof(char),bloomOptions->size,bloomFile);
	fclose(bloomFile);
	free(outputBloom);
}
