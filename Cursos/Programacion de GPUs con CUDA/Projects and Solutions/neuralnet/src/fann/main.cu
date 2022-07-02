#include <stdio.h>

#include "support.h"
#include "fann.h"

#define MAX_HIDDEN_LAYERS 100

void cpu_fann_run(struct fann * ann, fann_type * input, unsigned int num_output, fann_type * output);
void cpu_custom_run(unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_fann_run(unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_opt1_run(unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_fann_run_opt_shared(unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_fann_run_opt_reduction(unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_fann_run_opt_full(unsigned int, unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void gpu_fann_run_opt_full_no_hidden_val(unsigned int, unsigned int, unsigned int *, float *, float *, unsigned int *, float *, unsigned int *, float *);
void cpu_compute_MSE(unsigned int num_output, float *desired_output,
  float *actual_output, float *error_output, float *mse);
void cpu_backpropagate_MSE(unsigned int num_layers, unsigned int *num_neurons_layers, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *error_output, float *error_hiddens);
void gpu_backpropagate_MSE(unsigned int num_layers, unsigned int *num_neurons_layers, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *error_output, float *error_hiddens);

bool change_data(struct fann *, fann_type *, unsigned int, unsigned int, unsigned int, float *, float *, unsigned int *);

int main(int argc, char* argv[])
{
    Timer timer;
    unsigned int i;
    
    fann_type *calc_out;
	unsigned int num_input = 200;
	unsigned int num_output = 100;
	unsigned int num_layers = 6; // default value
	unsigned int num_neurons_hidden[MAX_HIDDEN_LAYERS] = {500, 1000, 1000, 500}; // default value

	//float desired_error = (const float) 0;
	//const unsigned int max_epochs = 1000;
	//const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *data;
	
	float *input_h, *output_h, *weight_h;
    float *hidden_h;
    unsigned int *hidden_ptr_h;
	unsigned int num_weight;
	unsigned int *weight_idx_h;	
	
	float *input_d, *output_d, *weight_d;
    float *hidden_d;
    unsigned int *hidden_ptr_d;
	unsigned int *weight_idx_d;
    unsigned int *num_layers_d;
	unsigned int *layers_d;
	float *output_error_d, *hiddens_error_d;
	cudaError_t cuda_ret;

	enum Mode {CPU = 1, CPU_WITH_CHANGED_DATA, GPU, GPU_OPT1, CPU_COMPUTE_MSE, CPU_BACKPROP_MSE, GPU_OPT_SHARED=10, GPU_OPT_REDUCTION, GPU_OPT_FULL, GPU_OPT_FULL_NO_HIDDEN_VAL, GPU_BACKPROP_MSE=20};
	enum Mode mode;

	if(argc == 2) {
		mode = (enum Mode) atoi(argv[1]);
	}
  else if (argc > 3) {
    mode = (enum Mode) atoi(argv[1]);
    num_layers = atoi(argv[2]);
    if (num_layers > MAX_HIDDEN_LAYERS) {
      printf("Hidden layers count can only be up to 100.\n");
      return 1;
    }
    if (argc != 3+num_layers) {
      printf("Hidden layers parameters doesn't match.\n");
      return 1;
    }
    for (unsigned int n=0; n<num_layers; n++) {
      num_neurons_hidden[n] = atoi(argv[3+n]);
    }
    num_layers += 2; // accomodate input and output layers
  }
  else {
		printf("\n    Invalid input parameters."
		"\n"
    "\n    Usage: ./fann <m>                   # Mode: m, n_hidden: 4, hidden_1: 500, hidden_2: 1000, hidden_3: 1000, hidden_4: 500"
		"\n           ./fann <m> <N h1 h2 ... hn>  # Mode: m, n_hidden: N, hidden_1: h1, hidden_2: h2, ... hidden_N: hn"
														   "\n"
		"\n    Modes: 1 = CPU FANN original"
		"\n           2 = CPU with changed data structure"
		"\n           3 = GPU naive"
    "\n           10 = GPU optimization 1"
    "\n           11 = GPU optimization 2"
    "\n           12 = GPU optimization 3"
    "\n           13 = GPU optimization 4"
	    "\n"
		"\n\n");
		exit(0);
	}

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
  unsigned int num_hidden = num_layers - 2;
  unsigned int max_neurons_hidden = 0;
  unsigned int total_neurons = 0;
  for (unsigned int i=0; i<num_hidden; i++) {
    total_neurons += num_neurons_hidden[i];
    if (max_neurons_hidden < num_neurons_hidden[i])
      max_neurons_hidden = num_neurons_hidden[i];
  }

  // set random seed to a certain number, so we can reproduce the random numbers
  srand(0xC00DA);

  //printf("Creating network.\n");
  unsigned int layers[num_layers];
  layers[0] = num_input;
  for (unsigned int i=0; i<num_hidden; i++) {
    layers[i+1] = num_neurons_hidden[i];
  }
  layers[num_layers-1] = num_output;

  
  
  // Initialize host variables ----------------------------------------------
  
  ann = fann_create_standard_array(num_layers, layers);

  calc_out = (float*) malloc(num_output*sizeof(float));
  float *hiddens_error_calc = (float*) malloc((total_neurons+num_hidden)*sizeof(float));
  
	num_weight = (num_input+1)*num_neurons_hidden[0] + // input*hidden_0
    (num_neurons_hidden[num_hidden-1]+1)*num_output; // hidden_(n-1)*output
  for (int n=0; n<num_hidden-1; n++) {
    num_weight += (num_neurons_hidden[n]+1)*num_neurons_hidden[n+1];
  }
	input_h = (float*) malloc((num_input+1)*sizeof(float));
	output_h = (float*) malloc((num_output+1)*sizeof(float));
  hidden_h = (float*) malloc((total_neurons+num_hidden) * sizeof(float));
  hidden_ptr_h = (unsigned int*) malloc((num_hidden+1) * sizeof(unsigned int));
	weight_h = (float*) malloc(num_weight*sizeof(float));
	weight_idx_h = (unsigned int*) malloc((num_layers)*sizeof(unsigned int));
  float *output_error_h = (float*) malloc((num_output+1)*sizeof(float));
  float *hiddens_error_h = (float*) malloc((total_neurons+num_hidden)*sizeof(float));
  float mse_h;

  // update hidden_ptr_h
  hidden_ptr_h[0] = 0;
  for (unsigned int i=0; i<num_hidden; i++) {
    hidden_ptr_h[1+i] = hidden_ptr_h[i] + num_neurons_hidden[i] + 1;
  }
	
	data = fann_read_train_from_file("fann/test.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	
	if (change_data(ann, data->input[0], num_input, num_output, num_layers, input_h, weight_h, weight_idx_h))
		return 1;

	if (ann->connection_rate < 1) {
		printf("No support un-fully connected layer\n");
	}
	
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Layers configuration:");
    for (unsigned int i=0; i<num_layers; i++) {
      printf("  %d", layers[i]);
    }
    printf("\n");
    
	// unsigned int k;
	// printf("Input\n  ");
	// for (unsigned int i=0; i<num_input+1; i++) {
	// 	printf("%f ", input_h[i]);
	// }
	// printf("\n");
	// printf("Weight_index\n  ");
	// for (k=0; k<num_layers; k++) {
	// 	printf("%d ", weight_idx_h[k]);
	// }
	// printf("\n");
	// printf("Weight\n");
	// printf("  First layer\n");
	// for (unsigned int i=0; i<num_input+1; i++) {
	// 	printf("    ");
	// 	for (unsigned int j=0; j<num_neurons_hidden[0]; j++) {
	// 		printf("%f ", weight_h[i*num_neurons_hidden[0]+j]);
	// 	}
	// 	printf("\n");
	// }
	// for (k=0; k<num_layers-3; k++) {
	// 	printf("  %dth layer\n", k+2);
	// 	for (unsigned int i=0; i<num_neurons_hidden[k+1]+1; i++) {
	// 		printf("    ");
	// 		for (unsigned int j=0; j<num_neurons_hidden[k]; j++) {
	// 			printf("%f ", weight_h[weight_idx_h[k+1]+i*num_neurons_hidden[k]+j]);
	// 		}
	// 		printf("\n");
	// 	}
	// }
	// printf("  Last layer\n");
	// for (unsigned int i=0; i<num_neurons_hidden[0]+1; i++) {
	// 	printf("    ");
	// 	for (unsigned int j=0; j<num_output; j++) {
	// 		printf("%f ", weight_h[weight_idx_h[k+1]+i*num_output+j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");


    if (mode >= GPU_BACKPROP_MSE) {
		printf("Doing foward computation and MSE..."); fflush(stdout);
		startTime(&timer);
		cpu_custom_run(num_layers, layers, input_h, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_h);
		cpu_compute_MSE(num_output, data->output[0], output_h, output_error_h, &mse_h);
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    
    // Allocate device variables ----------------------------------------------

	if(mode != CPU && mode != CPU_WITH_CHANGED_DATA && mode != CPU_COMPUTE_MSE && mode != CPU_BACKPROP_MSE) {
		printf("Allocating device variables..."); fflush(stdout);
		startTime(&timer);

		if (mode < GPU_BACKPROP_MSE) {
			cuda_ret = cudaMalloc((void**)&input_d, (num_input+1)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&output_d, (num_output+1)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&weight_d, num_weight*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&hidden_d, (total_neurons+num_hidden)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&hidden_ptr_d, (num_hidden+1)*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&weight_idx_d, num_layers*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&num_layers_d, sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&layers_d, num_layers*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
		}
		else{
			cuda_ret = cudaMalloc((void**)&num_layers_d, sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&layers_d, num_layers*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

			cuda_ret = cudaMalloc((void**)&weight_d, num_weight*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

			cuda_ret = cudaMalloc((void**)&weight_idx_d, num_layers*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

			cuda_ret = cudaMalloc((void**)&hidden_d, (total_neurons+num_hidden)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
	
			cuda_ret = cudaMalloc((void**)&hidden_ptr_d, (num_hidden+1)*sizeof(unsigned int));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

			cuda_ret = cudaMalloc((void**)&output_error_d, (num_output+1)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

			cuda_ret = cudaMalloc((void**)&hiddens_error_d, (total_neurons+num_hidden)*sizeof(float));
			if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
		}

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}
	
	
	
   // Copy host variables to device ------------------------------------------

	if(mode != CPU && mode != CPU_WITH_CHANGED_DATA && mode != CPU_COMPUTE_MSE && mode != CPU_BACKPROP_MSE) {
		printf("Copying data from host to device..."); fflush(stdout);
		startTime(&timer);

		if (mode < GPU_BACKPROP_MSE) {
			cuda_ret = cudaMemcpy(input_d, input_h, (num_input+1)*sizeof(float), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(weight_d, weight_h, num_weight*sizeof(float), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(hidden_ptr_d, hidden_ptr_h, (num_hidden+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(weight_idx_d, weight_idx_h, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(num_layers_d, &num_layers, sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(layers_d, layers, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
		}
		else {
			cuda_ret = cudaMemcpy(num_layers_d, &num_layers, sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(layers_d, layers, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
			
			cuda_ret = cudaMemcpy(weight_d, weight_h, num_weight*sizeof(float), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
	
			cuda_ret = cudaMemcpy(weight_idx_d, weight_idx_h, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}

			cuda_ret = cudaMemcpy(hidden_d, hidden_h, (total_neurons+num_hidden)*sizeof(float), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
			
			cuda_ret = cudaMemcpy(hidden_ptr_d, hidden_ptr_h, (num_hidden+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
			
			cuda_ret = cudaMemcpy(output_error_d, output_error_h, (num_output+1)*sizeof(float), cudaMemcpyHostToDevice);
			if(cuda_ret != cudaSuccess) {
			  FATAL("Unable to copy memory to the device");
			}
		}

		cudaDeviceSynchronize();
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}

	
	
	// Launch kernel ----------------------------------------------------------
    
	printf("Launching kernel ");
	if (mode == CPU) {
		printf("(CPU)...");fflush(stdout);
		startTime(&timer);
		cpu_fann_run(ann, data->input[0], num_output, output_h);
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}
	else if (mode == CPU_WITH_CHANGED_DATA) {
		printf("(CPU with custom structure)...");fflush(stdout);
		startTime(&timer);
		cpu_custom_run(num_layers, layers, input_h, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_h);
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}
	else if (mode == GPU) {
		printf("(GPU)...");fflush(stdout);
		startTime(&timer);
		gpu_fann_run(num_layers, layers, input_d, weight_d, weight_idx_h, hidden_d, hidden_ptr_h, output_d);
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}
	else if (mode == GPU_OPT1) {
		printf("(GPU opt 1)...");fflush(stdout);
		startTime(&timer);
		gpu_opt1_run(num_layers, layers, input_d, weight_d, weight_idx_h, hidden_d, hidden_ptr_h, output_d);
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	}
  else if (mode == GPU_OPT_SHARED) {
    printf("(GPU opt shared)...");fflush(stdout);
    startTime(&timer);
    gpu_fann_run_opt_shared(num_layers, layers, input_d, weight_d, weight_idx_h, hidden_d, hidden_ptr_h, output_d);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else if (mode == GPU_OPT_REDUCTION) {
    printf("(GPU opt reduction)...");fflush(stdout);
    startTime(&timer);
    gpu_fann_run_opt_reduction(num_layers, layers, input_d, weight_d, weight_idx_h, hidden_d, hidden_ptr_h, output_d);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else if (mode == GPU_OPT_FULL) {
    printf("(GPU opt full)...");fflush(stdout);
    startTime(&timer);
    gpu_fann_run_opt_full(num_layers, max_neurons_hidden, layers_d, input_d, weight_d, weight_idx_d, hidden_d, hidden_ptr_d, output_d);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else if (mode == GPU_OPT_FULL_NO_HIDDEN_VAL) {
    printf("(GPU opt full no hidden value)...");fflush(stdout);
    startTime(&timer);
    gpu_fann_run_opt_full_no_hidden_val(num_layers, max_neurons_hidden, layers_d, input_d, weight_d, weight_idx_d, hidden_d, hidden_ptr_d, output_d);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else if (mode == CPU_COMPUTE_MSE) {
    cpu_custom_run(num_layers, layers, input_h, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_h);
    printf("(CPU compute MSE)...");fflush(stdout);
    startTime(&timer);
    cpu_compute_MSE(num_output, data->output[0], output_h, output_error_h, &mse_h);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Verifying computed MSE ... ");
    cpu_fann_run(ann, data->input[0], num_output, calc_out);
    fann_compute_MSE(ann, data->output[0]);
    float ann_mse = fann_get_MSE(ann);
    if (abs(ann_mse-mse_h)>0.001)
      printf("incorrect! computed mse %f, expecting %f\n", mse_h, ann_mse);
    else
      printf("Correct.\n");

    struct fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
    const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
    fann_type *error_it = ann->train_errors;
    error_it += last_layer_begin - first_neuron;
    for (i=0; i<num_output; i++)
      if (abs(error_it[i]-output_error_h[i])>0.001)
        break;

    printf("Verifying output difference ... ");
    if (i<num_output)
      printf("Incorrect results at output %d (expecting %f actual %f).\n", i,
        error_it[i], output_error_h[i]);
    else
      printf("Correct.\n");
  }
  else if (mode == CPU_BACKPROP_MSE) {
    cpu_custom_run(num_layers, layers, input_h, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_h);
    cpu_compute_MSE(num_output, data->output[0], output_h, output_error_h, &mse_h);
    printf("(CPU backpropagate MSE)...");fflush(stdout);
    startTime(&timer);
    cpu_backpropagate_MSE(num_layers, layers, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_error_h, hiddens_error_h);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else if (mode == GPU_BACKPROP_MSE) {
	    printf("(GPU backpropagate MSE)...");fflush(stdout);
	    startTime(&timer);
	    gpu_backpropagate_MSE(num_layers, layers, weight_d, weight_idx_h, hidden_d, hidden_ptr_h, output_error_d, hiddens_error_d);
	    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  }
  else {
    printf("Unsupported mode %d.\n", mode);
  }
	
	
	
    // Copy device variables from host ----------------------------------------

    if(mode != CPU && mode != CPU_WITH_CHANGED_DATA && mode != CPU_COMPUTE_MSE && mode != CPU_BACKPROP_MSE) {

        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);

		if (mode < GPU_BACKPROP_MSE) {
			cuda_ret = cudaMemcpy(output_h, output_d, num_output*sizeof(float), cudaMemcpyDeviceToHost);
			if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
		}
		else {
			cuda_ret = cudaMemcpy(hiddens_error_h, hiddens_error_d, (total_neurons+num_hidden)*sizeof(float), cudaMemcpyDeviceToHost);
			if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
		}

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }
    
    
    
    // Verify correctness -----------------------------------------------------
	
	if (mode < GPU_BACKPROP_MSE) {
		printf("Verifying results..."); fflush(stdout);
		startTime(&timer);
		cpu_fann_run(ann, data->input[0], num_output, calc_out);
		for (i=0; i<num_output; i++)
			if (abs(calc_out[i]-output_h[i])>0.001) {
				printf("%d: %f vs %f\n", i, calc_out[i], output_h[i]);
				break;
			}
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
		if (i<num_output)
			printf("Incorrect results.\n");
		else
			printf("Correct.\n");
	}
	else {
		printf("Verifying results..."); fflush(stdout);
		startTime(&timer);
	    cpu_backpropagate_MSE(num_layers, layers, weight_h, weight_idx_h, hidden_h, hidden_ptr_h, output_error_h, hiddens_error_calc);

		for (i=0; i<total_neurons+num_hidden; i++)
			if (abs(hiddens_error_calc[i]-hiddens_error_h[i])>0.001) {
				printf("%d: %f vs %f\n", i, hiddens_error_calc[i], hiddens_error_h[i]);
				break;
			}
		stopTime(&timer); printf("%f s\n", elapsedTime(timer));
		if (i<total_neurons+num_hidden)
			printf("Incorrect results.\n");
		else
			printf("Correct.\n");
	    }

	fann_destroy_train(data);
	fann_destroy(ann);
	
	
	
    // Free memory ------------------------------------------------------------
	free(input_h);
	free(output_h);
	free(weight_h);
	free(hidden_h);
	free(hidden_ptr_h);
	free(weight_idx_h);	
	if(mode != CPU && mode != CPU_WITH_CHANGED_DATA && mode != CPU_COMPUTE_MSE && mode != CPU_BACKPROP_MSE) {
		cudaFree(input_d);
		cudaFree(output_d);
		cudaFree(weight_d);
		cudaFree(hidden_d);
		cudaFree(hidden_ptr_d);
		cudaFree(weight_idx_d);
		cudaFree(num_layers_d);
		cudaFree(layers_d);
	}

	return 0;
}
                                                    

bool change_data(struct fann *ann, fann_type * input, unsigned int num_input, unsigned int num_output, unsigned int num_layers,
		float *input_h, float *weight_h, unsigned int *weight_idx_h) {
	struct fann_neuron *neuron_it, *last_neuron;
	unsigned int i, max_num_connections, total_num_weight, idx_weight;
	fann_type *weights;
	struct fann_layer *layer_it, *last_layer;

	max_num_connections = 0;
	total_num_weight = 0;
	weight_idx_h[0] = 0;
	idx_weight = 1;

	if (num_input!=ann->num_input) {
		printf("Numbers of input are mismatched (%d, %d)\n", num_input, ann->num_input);
		return true;
	}
	if (num_output!=ann->num_output) {
		printf("Numbers of output are mismatched (%d, %d)\n", num_output, ann->num_output);
		return true;
	}
	for (i = 0; i != num_input; i++) {
		input_h[i] = input[i];
	}
	input_h[i] = 1;

	last_layer = ann->last_layer;
	for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++, idx_weight++) {
		unsigned int num_neuron = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron) - 1;
		unsigned int idx_neuron = 0;
		unsigned int num_connections;
		last_neuron = layer_it->last_neuron;
		for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++, idx_neuron++) {
			num_connections = neuron_it->last_con - neuron_it->first_con;
			weights = ann->weights + neuron_it->first_con;
			
			if (max_num_connections<num_connections)
				max_num_connections = num_connections;
			if (ann->connection_rate >= 1) {				
				for (i = 0; i != num_connections; i++) {
					weight_h[total_num_weight + i*num_neuron + idx_neuron] = weights[i];
				}
			}
			else {
				printf("No support un-fully connected layer\n");
				return true;
			}
		}
		total_num_weight += max_num_connections*num_neuron;
		weight_idx_h[idx_weight] = total_num_weight;
	}

	return false;
}




