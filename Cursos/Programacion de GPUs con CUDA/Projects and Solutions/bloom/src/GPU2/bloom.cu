#include "bloom.h"


/**
* Responsible for calculating the dimenions of the gpu layout being used.
* @param numWords
* @param numHash
* @param device
*/
dim3 calculateThreadDimensions(int numWords,int numHash,
	cudaDeviceProp* deviceProps){
	if(numWords == 0 || numHash == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}
		
	//Firstly, solve for the max number of words that 
	//Can be processed in one thread block.
	int maxWordPerBlock = deviceProps->maxThreadsPerBlock/numHash;

	//Check to see if the user demanded more hash functions than
	//A single block can support. If so, only one word per block
	//Will be processed.	
	if(maxWordPerBlock ==0){
		maxWordPerBlock = 1;
		numHash = deviceProps->maxThreadsPerBlock;
	}

    //Try to group the words into sets of 32.
	int wordsPerBlock = 32*(maxWordPerBlock/32);
	if(wordsPerBlock ==0)
	  wordsPerBlock = maxWordPerBlock;

	//If all the words can fit in one block.
	if(numWords<=maxWordPerBlock)
		wordsPerBlock = numWords;	

	dim3 threadDimensions(wordsPerBlock,numHash);
	return threadDimensions;
}

/**
* Responsible for calculating the thread dimensions of the gpu layout.
* @param threadDimensions the dimensions of the thread block.
* @param device The id of the device being used.
*/
dim3 calculateBlockDimensions(dim3 threadDimensions,int numWords,int numHash,
	cudaDeviceProp* deviceProps){
	if(numWords == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}

	//Calculate the number of blocks needed to process all of the words.
	int numBlocksNeeded = numWords/threadDimensions.x;
	if(numWords%threadDimensions.x!=0)
		numBlocksNeeded++;

	//Hard coded due to hydra glitch.
	int maxGridSizeX = 65535;
	int numBlocksPerRow;	

	//If we only need part of the first row...	
	if(numBlocksNeeded<=maxGridSizeX)
		numBlocksPerRow = numBlocksNeeded;
	//If we need one or more rows...
	else{
		numBlocksPerRow = maxGridSizeX;
	}
	//Calculate the number of rows needed.		
	int numRows = numBlocksNeeded/numBlocksPerRow;
	if(numBlocksNeeded%numBlocksPerRow!=0){
		numRows++;
	}

	//Add rows for extra hash functions > 1024.	
	numRows = numRows*(numHash/deviceProps->maxThreadsPerBlock)+ 
		numRows*(numHash%deviceProps->maxThreadsPerBlock>0 ? 1 : 0);	
	
	if(numRows>deviceProps->maxGridSize[1]){
		printf("Too many rows requested %i, \n",numRows);
		printf("Blocks Per Row %i \n",numBlocksPerRow);
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		return dim3(0,0);
	}
	
	return dim3(numBlocksPerRow,numRows);
}

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the djb2 hash in long format.
*/
__device__ unsigned long djb2Hash(unsigned char* str,int start){
	unsigned long hash = 5381;
	int c;
	while(str[start]!=','){
		c = (int)str[start];
		hash = ((hash<<5)+hash)+c;
		start++;
	}	
	return hash;
}

/**
* Calculates the sdbm hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the sdbm hash in long format.
*/
__device__ unsigned long sdbmHash(unsigned char* str,int start){
	unsigned long hash = 0;
	int c = 0;
	while(str[start]!=','){
		c = (int)str[start];
		hash = c+(hash<<6)+(hash<<16)-hash;
		start++;
	}
	return hash;
}

__device__ int calculateCurrentWord(int numRowsPerHash){
	int numThreadsPrevRows = (blockDim.x*gridDim.x)*(blockIdx.y/numRowsPerHash)+
		blockDim.x*blockIdx.x;
	return  threadIdx.x+numThreadsPrevRows;
}

__device__ int calculateIndex(char* dev_bloom,int size,char* dev_words,
	int wordStartingPosition,int numHash,int numRowsPerHash){	

	unsigned long firstValue = djb2Hash((unsigned char*)dev_words,wordStartingPosition)%size;	
	unsigned long secondValue = sdbmHash((unsigned char*)dev_words,wordStartingPosition)%size;
	int fy = ((blockIdx.y%numRowsPerHash)*(blockDim.y-1)+threadIdx.y);
	if(fy < numHash) {
	  secondValue = (secondValue*fy*fy)%size;
	  return (firstValue+secondValue)%size;
	}
	return -1;
}

/**
* Multiply with carry. Psuedo random number generation.
* @param m_w The first seed.
* @param m_z The second seed.
*/
__device__ unsigned int get_random(unsigned long m_w,unsigned long m_z){
	int i = 0;
	for(;i<100;i++){
		m_z = 36969 * (m_z & 65535) + (m_z >> 16);
		m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	}
	return (unsigned int)((m_z << 16) + m_w)%1000000;
} 

/**
* Responsible for inserting words using the gpu.
* @param dev_bloom The bloom filter being used.
* @param dev_size The size of the bloom filter being used.
* @param dev_words The words being inserted.
* @param dev_positions The starting positions of the words.
* @param dev_numWords The number of words being inserted.
* @param numHashes The number of hash functions used.
*/
__global__ void insertWordsGpuPBF(char* dev_bloom,int size,char* dev_words,
	unsigned int* dev_positions,int numWords,int numHashes,int numRowsPerHash,float prob, int randOffset){
	int currentWord = calculateCurrentWord(numRowsPerHash);

    if(currentWord < numWords) {
	  int wordStartingPosition = (int)dev_positions[currentWord]; 	
	  int setIdx = calculateIndex(dev_bloom,size,dev_words,wordStartingPosition,numHashes,numRowsPerHash);
	  int fy = ((blockIdx.y%numRowsPerHash)*(blockDim.y-1)+threadIdx.y);
	  unsigned int randVal = get_random(randOffset,(unsigned long)(randOffset*setIdx+fy+currentWord));
	  float calcProb = (float)randVal/1000000.0f;
	
	  //If the number of hash functions was exceeded.
	  if(setIdx >= 0) {
	    if(calcProb < prob){
	      dev_bloom[setIdx] = 1;
	    }
	  }
    }
}

__global__ void insertWordsGpuPBF2(char* dev_bloom,int size,char* dev_words,unsigned int* dev_positions,int numWords,int numHashes,float prob, int randOffset, int wordsPerBlock){
     
   int kIter = (numHashes + blockDim.y - 1) / blockDim.y;
   int nIter = (wordsPerBlock + blockDim.x - 1) / blockDim.x;
   int cw = blockIdx.x * wordsPerBlock;

   // Iterate random sequences
   int i;
   for (i = 0; i < nIter; ++i) {
     if (cw < numWords) {
	   int wordStartingPos = dev_positions[cw];

       // Iterate hashes
       int j, y = threadIdx.y;
	   for (j = 0; j < kIter; ++j) {
	     if (y < numHashes) {

           // Compute BF index
	       unsigned long int fv = djb2Hash((unsigned char*)dev_words,wordStartingPos) % size;	
           unsigned long int sv = sdbmHash((unsigned char*)dev_words,wordStartingPos) % size;
		   sv = (sv * y * y) % size;
	       int setIdx = (fv + sv) % size;
		   unsigned int randVal = get_random(randOffset, (unsigned long int)(randOffset * setIdx + y + cw));
		   float p = (float)randVal / 1000000;

		   // Set element based on probability
           if (p < prob) {
             dev_bloom[setIdx] = 1;
		   }
		 }
		 y += blockDim.y;
	   }
	 }
     cw += blockDim.x;
   }
}


__global__ void queryWordsGpuPBF2(char* dev_bloom,int size,char* dev_words,unsigned int* dev_positions,int* dev_results,int numWords,int numHashes,int wordsPerBlock)
{
   int kIter = (numHashes + blockDim.y - 1) / blockDim.y;
   int nIter = (wordsPerBlock + blockDim.x - 1) / blockDim.x;
   int cw = blockIdx.x * wordsPerBlock;

   // Iterate random sequences
   int i;
   for (i = 0; i < nIter; ++i) {
     if (cw < numWords) {
	   int wordStartingPos = dev_positions[cw];

       // Iterate hashes
       int j, y = threadIdx.y;
	   for (j = 0; j < kIter; ++j) {
	     if (y < numHashes) {

           // Compute BF index
	       unsigned long int fv = djb2Hash((unsigned char*)dev_words,wordStartingPos) % size;	
           unsigned long int sv = sdbmHash((unsigned char*)dev_words,wordStartingPos) % size;
		   sv = (sv * y * y) % size;
	       int getIdx = (fv + sv) % size;

		   // Accumulate 
           atomicAdd(&dev_results[cw],dev_bloom[getIdx]);
		 }
		 y += blockDim.y;
	   }
	 }
     cw += blockDim.x;
   }
}


/**
* Responsible for querying words using the gpu.
*/
__global__ void queryWordsGpuPBF(char* dev_bloom,int size,char* dev_words,
	unsigned int* dev_positions,int* dev_results,int numWords,int numHashes,
	int numRowsPerHash){

	int currentWord = calculateCurrentWord(numRowsPerHash);
	if(currentWord < numWords) {
	  int wordStartingPosition = (int)dev_positions[currentWord]; 
      int getIdx = calculateIndex(dev_bloom,size,dev_words,wordStartingPosition,numHashes,numRowsPerHash);
	  if(getIdx >= 0) {
	    atomicAdd(&dev_results[currentWord], dev_bloom[getIdx]);
	  }
	}
}


/**
* Responsible for inserting words into the PBF bloom filter.
*/
cudaError_t insertWordsPBF(char* dev_bloom,int size,char* words,
	unsigned int* offsets,int numWords,int numBytes,int numHashes,int device,float prob){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

   unsigned int wordsPerBlock = 100;
   unsigned int thx, thy, bkx;
   unsigned int maxth = deviceProps.maxThreadsPerBlock;
   thy = (maxth > numHashes) ? numHashes : maxth;
   thx = maxth / thy;
   bkx = (numWords + wordsPerBlock - 1) / wordsPerBlock;
   dim3 blockDimensions(bkx, 1, 1);
   dim3 threadDimensions(thx, thy, 1);
	printf("Dimensions calculated: \n");
	printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
	printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);

	//Allocate the information.
	unsigned int* dev_offsets;
	CUDA_CALL(cudaMalloc((void **)&dev_offsets, numWords * sizeof(unsigned int)));
	CUDA_CALL(cudaMemcpy(dev_offsets, offsets, numWords * sizeof(unsigned int), cudaMemcpyHostToDevice));
		
	char* dev_words;
	CUDA_CALL(cudaMalloc((void **)&dev_words, numBytes * sizeof(char)));
	CUDA_CALL(cudaMemcpy(dev_words, words, numWords * sizeof(char), cudaMemcpyHostToDevice));

	//Actually insert the words.
	//srand(time(0));
    int randOffset = rand();
	printf("GPU insert words\n");
	insertWordsGpuPBF2<<<blockDimensions,threadDimensions>>>(dev_bloom,size,dev_words,dev_offsets,numWords,numHashes,prob,randOffset,wordsPerBlock);
	cudaThreadSynchronize();

	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		return error;
	}

    cudaFree(dev_offsets);
    cudaFree(dev_words);

	return cudaSuccess;			 				

}

cudaError_t queryWordsPBF(char* dev_bloom,int size,char* words,
	unsigned int* offsets,int numWords,int numBytes,int numHashes,int device,
		int* results){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

   unsigned int wordsPerBlock = 100;
   unsigned int thx, thy, bkx;
   unsigned int maxth = deviceProps.maxThreadsPerBlock;
   thy = (maxth > numHashes) ? numHashes : maxth;
   thx = maxth / thy;
   bkx = (numWords + wordsPerBlock - 1) / wordsPerBlock;
   dim3 blockDimensions(bkx, 1, 1);
   dim3 threadDimensions(thx, thy, 1);
	printf("Dimensions calculated: \n");
	printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
	printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);

	unsigned int* dev_offsets;
	CUDA_CALL(cudaMalloc((void **)&dev_offsets, numWords * sizeof(unsigned int)));
	CUDA_CALL(cudaMemcpy(dev_offsets, offsets, numWords * sizeof(unsigned int), cudaMemcpyHostToDevice));
		
	char* dev_words;
	CUDA_CALL(cudaMalloc((void **)&dev_words, numBytes * sizeof(char)));
	CUDA_CALL(cudaMemcpy(dev_words, words, numWords * sizeof(char), cudaMemcpyHostToDevice));

	int* dev_results;
	CUDA_CALL(cudaMalloc((void **)&dev_results, numWords * sizeof(int)));
	CUDA_CALL(cudaMemcpy(dev_results, results, numWords * sizeof(int), cudaMemcpyHostToDevice));

	//Actually query the words.
	queryWordsGpuPBF2<<<blockDimensions,threadDimensions>>>(dev_bloom,size,dev_words,dev_offsets,dev_results,numWords,numHashes,wordsPerBlock);
	cudaThreadSynchronize();
		
	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		return error;
	}

	CUDA_CALL(cudaMemcpy(results, dev_results, numWords * sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(dev_offsets);
	cudaFree(dev_words);
	cudaFree(dev_results);

	return cudaSuccess;			 				
}
