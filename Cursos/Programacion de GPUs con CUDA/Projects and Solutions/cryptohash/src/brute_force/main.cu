#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "support.h"
#include "kernel.cu"

void find_pass(char * password, char * found_pass, int max_letters, int * found_flag);

int main(int argc, char *argv[]) {
	Timer timer;
	
	printf("Setting up the problem and allocating variables...\n");
	startTime(&timer);

	char * pass_d;
	char * pass_h;
	char * found_h;
	char * found_d;
	char * characters;
	int * found_flag;
	
	/* Allocate character array, password on device/host */
	pass_h = (char *) malloc(sizeof(char) * MAX_PASS_LENGTH);
	found_h = (char *) malloc(sizeof(char) * MAX_PASS_LENGTH);
	characters = (char *) malloc(sizeof(char) * (NUM_LETTERS+1));

	cudaMalloc((void **) &pass_d, sizeof(char)*MAX_PASS_LENGTH);
	cudaMalloc((void **) &found_d, sizeof(char)*MAX_PASS_LENGTH);
	cudaMalloc((void **) &found_flag, sizeof(int));
	cudaMemset(pass_d, 0, sizeof(char)*MAX_PASS_LENGTH);
	cudaMemset(found_d, 0, sizeof(char)*MAX_PASS_LENGTH);
	cudaMemset(found_flag, 0, sizeof(int));

	if(argc == 1) {
		pass_h = "hello";
	} else if(argc == 2) {
		pass_h = argv[1];
	}

	characters = "abcdefghijklmnopqrstuvwxyz";	
	
	cudaMemcpyToSymbol(chars_c, characters, sizeof(char)*(NUM_LETTERS+1));
	cudaMemcpy(pass_d, pass_h, sizeof(char)*MAX_PASS_LENGTH, cudaMemcpyHostToDevice);
	
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	cudaDeviceSynchronize();
	
	printf("\nTrying to find password: %s\n\n", pass_h);	
	printf("Launching kernel...\n"); fflush(stdout);
	startTime(&timer);

	find_pass(pass_d, found_d, 7, found_flag);
	
	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	cudaMemcpy(found_h, found_d, sizeof(char)*MAX_PASS_LENGTH, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	printf("Printing found password on next line\n");
	printf("%s", found_h);
	/*
	free(pass_h);
	free(found_h);
	free(characters);
	
	cudaFree(pass_d);
	cudaFree(found_d);
	cudaFree(found_flag);*/
}

void find_pass(char * password, char * found_pass, int max_letters, int * found_flag) {
	int i;
	if(max_letters == 0)
		return;
	for(i=0; i < max_letters; i++) {
		uint64_t total_words = NUM_LETTERS;
		int j;
		for(j=0; j < i; j++) {
			total_words *= NUM_LETTERS;
		}
		printf("Total number of words: %lu\n", total_words);
		uint64_t words_per_thread = (total_words-1) / (BLOCK_SIZE*NUM_BLOCKS) + 1;
		printf("Number of words per thread: %lu\n\n", words_per_thread);
		int * found_flag_h;
		found_flag_h = (int *) malloc(sizeof(int));
		dim3 dimGrid, dimBlock;
		int number_of_blocks;
		if(total_words / BLOCK_SIZE > NUM_BLOCKS) {
			number_of_blocks = NUM_BLOCKS;
		} else {
			number_of_blocks = (total_words - 1) / BLOCK_SIZE + 1;
		}

		dimGrid = dim3(number_of_blocks,1,1);
		dimBlock = dim3(BLOCK_SIZE,1,1);
		
		brute_recovery<<<dimGrid, dimBlock>>>(password, i+1, words_per_thread,
						      total_words, found_pass, found_flag);
		cudaMemcpy(found_flag_h, found_flag, sizeof(int), cudaMemcpyDeviceToHost);
		if(*found_flag_h) {
			printf("Password found!\n");
			return;
		}
	}
	printf("Returning normally...not found\n");
}


