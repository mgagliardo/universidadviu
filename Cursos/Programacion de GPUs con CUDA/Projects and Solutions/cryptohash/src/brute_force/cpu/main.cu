#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "support.h"

#define NUM_LETTERS	26
#define MAX_PASS_LENGTH	11
#define MAX_LETTERS	8


int main(int argc, char *argv[]) {
	Timer timer;
	printf("Setting up the problem and allocating variables...\n");
	startTime(&timer);

	char * password;
	char * found_password;
	char * characters;
	int * found_flag;

	password = (char *) malloc(sizeof(char) * MAX_PASS_LENGTH);
	found_password = (char *) malloc(sizeof(char) * MAX_PASS_LENGTH);
	characters = (char *) malloc(sizeof(char) * (NUM_LETTERS+1));
	found_flag = (int *) malloc(sizeof(int));

	if(argc == 1) {
		password = "hello";
	} else if(argc == 2) {
		password = argv[1];
	}

	characters = "abcdefghijklmnopqrstuvwxyz";

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	cudaDeviceSynchronize();

	printf("\nTrying to find password: %s\n\n", password);
	printf("Launching CPU password finder...\n"); fflush(stdout);
	startTime(&timer);

	int i;
	for(i=0; i < MAX_LETTERS; i++) {
		uint64_t total_words = NUM_LETTERS;
		int j;
		for(j=0; j<i; j++) {
			total_words *= NUM_LETTERS;
		}
		printf("Total number of words: %lu\n\n", total_words);

		uint64_t curr_word = 0;
		uint64_t k;
		char word[MAX_PASS_LENGTH] = "";
		int check = 0;

		for(k=0; k<total_words; k++) {
			curr_word = k;
			for(j=0; j<i; j++) {
				word[i-1-j] = characters[(curr_word % NUM_LETTERS)];
				curr_word /= NUM_LETTERS;
			}
			word[i+1] = '\0';

			check = 1;
			for(j=0; j<i+1; j++) {
				if(password[j] != word[j]){
					check = 0;
				}
			}
			if(check) {
				printf("Password is found!\n");
				printf("Found: %s\n", word);
				stopTime(&timer); printf("%f s\n", elapsedTime(timer));
				return 0;
			}
		}
	}
	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}

