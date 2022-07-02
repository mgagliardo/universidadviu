#define NUM_LETTERS	26
#define MAX_PASS_LENGTH	11
#define BLOCK_SIZE	1024
#define NUM_BLOCKS	128
__constant__ char chars_c[NUM_LETTERS+1];


/* Brute force password recovery */
__global__ void brute_recovery(char * password, int length, uint64_t words_per_thread, uint64_t total_words, char * found_pass, int * found) {	

	int tx = threadIdx.x + blockIdx.x * blockDim.x; // overall thread #
	uint64_t start_word = tx*words_per_thread; // the starting index of password this thread checks
	if(start_word >= total_words)	// exit if the thread is out of bounds
		return;

	/* initialize variables */
	uint64_t k;
	int i;
	uint64_t curr_word;
	char word[MAX_PASS_LENGTH] = "";
	int check;

	/* Each thread checks all the passwords it's assigned to */
	for(k=0; k<words_per_thread; k++) {		
		i=0;
		curr_word = start_word + k;
		if(curr_word >= total_words)
			break;
		/* construct the curr_word #'s password */
		for(i=0; i<length; i++) {
			word[length-1-i] = chars_c[(curr_word % NUM_LETTERS)];
			curr_word /= NUM_LETTERS;
		}
		word[i+1] = '\0';	
		
		/* check to see if the current password is the desired password */
		check = 1;
		for(i=0; i<length+1; i++) {
			if(password[i] != word[i]){
				check = 0;
			}
		}
		/* if found, fill found_pass with the found password and return */
		if(check) {
			*found = 1;
			for(i=0; i<length+1; i++){
				found_pass[i] = word[i];
			}
			return;
		}
	}
}

