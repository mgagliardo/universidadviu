/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "fann.h"
typedef float fann_type;
#define fann_activation_switch(activation_function, value, result) \
switch(activation_function) \
{ \
	case FANN_LINEAR: \
		result = (fann_type)value; \
        break; \
	case FANN_LINEAR_PIECE: \
		result = (fann_type)((value < 0) ? 0 : (value > 1) ? 1 : value); \
        break; \
	case FANN_LINEAR_PIECE_SYMMETRIC: \
		result = (fann_type)((value < -1) ? -1 : (value > 1) ? 1 : value); \
        break; \
	case FANN_SIGMOID: \
		result = (fann_type)fann_sigmoid_real(value); \
        break; \
	case FANN_SIGMOID_SYMMETRIC: \
		result = (fann_type)fann_sigmoid_symmetric_real(value); \
        break; \
	case FANN_SIGMOID_SYMMETRIC_STEPWISE: \
		result = (fann_type)fann_stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, value); \
        break; \
	case FANN_SIGMOID_STEPWISE: \
		result = (fann_type)fann_stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, value); \
        break; \
	case FANN_THRESHOLD: \
		result = (fann_type)((value < 0) ? 0 : 1); \
        break; \
	case FANN_THRESHOLD_SYMMETRIC: \
		result = (fann_type)((value < 0) ? -1 : 1); \
        break; \
	case FANN_GAUSSIAN: \
		result = (fann_type)fann_gaussian_real(value); \
        break; \
	case FANN_GAUSSIAN_SYMMETRIC: \
		result = (fann_type)fann_gaussian_symmetric_real(value); \
        break; \
	case FANN_ELLIOT: \
		result = (fann_type)fann_elliot_real(value); \
	    break; \
	case FANN_ELLIOT_SYMMETRIC: \
		result = (fann_type)fann_elliot_symmetric_real(value); \
        break; \
	case FANN_SIN_SYMMETRIC: \
		result = (fann_type)fann_sin_symmetric_real(value); \
        break; \
	case FANN_COS_SYMMETRIC: \
		result = (fann_type)fann_cos_symmetric_real(value); \
        break; \
	case FANN_SIN: \
		result = (fann_type)fann_sin_real(value); \
        break; \
	case FANN_COS: \
		result = (fann_type)fann_cos_real(value); \
        break; \
	case FANN_GAUSSIAN_STEPWISE: \
        result = 0; \
        break; \
}

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/
__device__ fann_type fann_activation_derived_device(unsigned int activation_function,
								  fann_type steepness, fann_type value, fann_type sum)
{
	switch (activation_function)
	{
		case FANN_LINEAR:
		case FANN_LINEAR_PIECE:
		case FANN_LINEAR_PIECE_SYMMETRIC:
			return (fann_type) fann_linear_derive(steepness, value);
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			value = fann_clip(value, 0.01f, 0.99f);
			return (fann_type) fann_sigmoid_derive(steepness, value);
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			value = fann_clip(value, -0.98f, 0.98f);
			return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
		case FANN_GAUSSIAN:
			/* value = fann_clip(value, 0.01f, 0.99f); */
			return (fann_type) fann_gaussian_derive(steepness, value, sum);
		case FANN_GAUSSIAN_SYMMETRIC:
			/* value = fann_clip(value, -0.98f, 0.98f); */
			return (fann_type) fann_gaussian_symmetric_derive(steepness, value, sum);
		case FANN_ELLIOT:
			value = fann_clip(value, 0.01f, 0.99f);
			return (fann_type) fann_elliot_derive(steepness, value, sum);
		case FANN_ELLIOT_SYMMETRIC:
			value = fann_clip(value, -0.98f, 0.98f);
			return (fann_type) fann_elliot_symmetric_derive(steepness, value, sum);
		case FANN_SIN_SYMMETRIC:
			return (fann_type) fann_sin_symmetric_derive(steepness, sum);
		case FANN_COS_SYMMETRIC:
			return (fann_type) fann_cos_symmetric_derive(steepness, sum);
		case FANN_SIN:
			return (fann_type) fann_sin_derive(steepness, sum);
		case FANN_COS:
			return (fann_type) fann_cos_derive(steepness, sum);
	}
	return 0;
}

__global__ void gpu_run_kernel_single_step(float *weight, unsigned int weight_start_idx, unsigned int num_neurons_prev_layers, unsigned int num_neurons_curr_layers, float *hidden_prev, float *hidden_curr) {

	unsigned int j;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;
	
	// make sure bias neuron is set to 1
	hidden_prev[num_neurons_prev_layers] = 1;

	if (idx < num_neurons_curr_layers) {
		float neuron_sum = 0;
		for (j=0; j<=num_neurons_prev_layers; j++) {
			neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+idx]*hidden_prev[j];
		}
		neuron_sum = steepness*neuron_sum;
		float max_sum = 150/steepness;
		if(neuron_sum > max_sum)
			neuron_sum = max_sum;
		else if(neuron_sum < -max_sum)
			neuron_sum = -max_sum;

		fann_activation_switch(activation_function, neuron_sum, hidden_curr[idx]);
	}
}

__global__ void gpu_run_kernel_single_step_opt1(float *weight, unsigned int weight_start_idx, unsigned int num_neurons_prev_layers, unsigned int num_neurons_curr_layers, float *hidden_prev, float *hidden_curr) {

	unsigned int j;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;
	
	if (idx == 0) // make sure bias neuron is set to 1
		hidden_prev[num_neurons_prev_layers] = 1;

	float max_sum = 150/steepness;
	if (idx < num_neurons_curr_layers) {
		float neuron_sum = 0;
		for (j=0; j<=num_neurons_prev_layers; j++) {
			neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+idx]*hidden_prev[j];
		}
		neuron_sum = steepness*neuron_sum;
		if(neuron_sum > max_sum)
			neuron_sum = max_sum;
		else if(neuron_sum < -max_sum)
			neuron_sum = -max_sum;

		fann_activation_switch(activation_function, neuron_sum, hidden_curr[idx]);
	}
}

__global__ void gpu_run_kernel_opt_shared_single_step(float *weight, unsigned int weight_start_idx, unsigned int num_neurons_prev_layers, unsigned int num_neurons_curr_layers, float *hidden_prev, float *hidden_curr) {

	unsigned int i,j;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;

	__shared__ float hidden_prev_s[4096];

	// collectively read hidden_prev data into shared memory
	for (i=threadIdx.x; i<num_neurons_prev_layers; i+=blockDim.x) {
		hidden_prev_s[i] = hidden_prev[i];
	}

	if (threadIdx.x == 0) // make sure bias neuron is set to 1
		hidden_prev_s[num_neurons_prev_layers] = 1;
	
	__syncthreads();

	if (idx < num_neurons_curr_layers) {
		float neuron_sum = 0;
		float result;

		for (j=0; j<=num_neurons_prev_layers; j++) {
			neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+idx]*hidden_prev_s[j];
		}
		neuron_sum = steepness*neuron_sum;
		float max_sum = 150/steepness;
		if(neuron_sum > max_sum)
			neuron_sum = max_sum;
		else if(neuron_sum < -max_sum)
			neuron_sum = -max_sum;

		fann_activation_switch(activation_function, neuron_sum, result);
		hidden_curr[idx] = result;
	}
}

__global__ void gpu_run_kernel_opt_reduction_single_step(float *weight, unsigned int weight_start_idx, unsigned int num_neurons_prev_layers, unsigned int num_neurons_curr_layers, float *hidden_prev, float *hidden_curr) {

	unsigned int i,j;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int b_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;
	float neuron_sum = 0;

	__shared__ float hidden_prev_s[4096];
	__shared__ float sum_s[1024];

	// collectively read hidden_prev data into shared memory
	for (i=b_idx; i<num_neurons_prev_layers; i+=blockDim.x*blockDim.y) {
		hidden_prev_s[i] = hidden_prev[i];
	}

	if (threadIdx.x == 0) // make sure bias neuron is set to 1
		hidden_prev_s[num_neurons_prev_layers] = 1;
	
	for (i=b_idx; i<1024; i+=blockDim.x*blockDim.y) {
		sum_s[i] = 0;
	}

	__syncthreads();

	if (idx < num_neurons_curr_layers) {

		for (j=threadIdx.y; j<=num_neurons_prev_layers; j+=blockDim.y) {
			neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+idx]*hidden_prev_s[j];
		}
		sum_s[b_idx] = neuron_sum;
	}

	__syncthreads();

	if (idx < num_neurons_curr_layers) {
		unsigned int stride = blockDim.y/2;
		while(stride > 0) {
			if (threadIdx.y < stride) {
				sum_s[threadIdx.x+threadIdx.y*blockDim.x] += sum_s[threadIdx.x+(threadIdx.y+stride)*blockDim.x];
			}
			stride = stride/2;
		}
	}

	__syncthreads();

	if (idx < num_neurons_curr_layers) {
		if (threadIdx.y == 0) {
			float result;

			neuron_sum = sum_s[threadIdx.x];
			neuron_sum = steepness*neuron_sum;
			float max_sum = 150/steepness;
			if(neuron_sum > max_sum)
				neuron_sum = max_sum;
			else if(neuron_sum < -max_sum)
				neuron_sum = -max_sum;

			fann_activation_switch(activation_function, neuron_sum, result);
			hidden_curr[idx] = result;
		}
	}
}

__global__ void gpu_run_kernel_opt_full(unsigned int num_layers, unsigned int max_neurons, unsigned int *num_neurons_layers, float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int i,j, k;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;
	__shared__ float hidden_prev_s[4096];
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden) : (output);

	for (k=0; k<num_layers-1; k++) {
		unsigned int weight_start_idx = weight_idx[k];
		unsigned int num_neurons_prev_layers = num_neurons_layers[k];
		unsigned int num_neurons_curr_layers = num_neurons_layers[k+1];
		
		__syncthreads();

		// collectively read hidden_prev data into shared memory
		for (i=threadIdx.x; i<num_neurons_prev_layers; i+=blockDim.x) {
			hidden_prev_s[i] = hidden_prev[i];
		}

		if (threadIdx.x == 0) // make sure bias neuron is set to 1
			hidden_prev_s[num_neurons_prev_layers] = 1;
		
		__syncthreads();

		for (i=threadIdx.x; i<num_neurons_curr_layers; i+=blockDim.x) {
			float neuron_sum = 0;
			float result;

			for (j=0; j<=num_neurons_prev_layers; j++) {
				neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+i]*hidden_prev_s[j];
			}
			neuron_sum = steepness*neuron_sum;
			float max_sum = 150/steepness;
			if(neuron_sum > max_sum)
				neuron_sum = max_sum;
			else if(neuron_sum < -max_sum)
				neuron_sum = -max_sum;

			fann_activation_switch(activation_function, neuron_sum, result);
			hidden_curr[i] = result;
		}

		hidden_prev = hidden + hidden_ptr[k];
		if (k < num_layers-3)
			hidden_curr = hidden + hidden_ptr[k+1];
		else
			hidden_curr = output;
	}
}

__global__ void gpu_run_kernel_opt_full_no_hidden_val(unsigned int num_layers, unsigned int max_neurons, unsigned int *num_neurons_layers, float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int i,j, k;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	__shared__ float hidden_1_s[4096];
	__shared__ float hidden_2_s[4096];
	fann_type steepness = 1;
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden_1_s) : (output);

	for (k=0; k<num_layers-1; k++) {
		unsigned int weight_start_idx = weight_idx[k];
		unsigned int num_neurons_prev_layers = num_neurons_layers[k];
		unsigned int num_neurons_curr_layers = num_neurons_layers[k+1];
		
		__syncthreads();

		if (threadIdx.x == 0) // make sure bias neuron is set to 1
			hidden_prev[num_neurons_prev_layers] = 1;

		for (i=threadIdx.x; i<num_neurons_curr_layers; i+=blockDim.x) {
			float neuron_sum = 0;
			float result;

			for (j=0; j<=num_neurons_prev_layers; j++) {
				neuron_sum += weight[weight_start_idx+j*(num_neurons_curr_layers)+i]*hidden_prev[j];
			}
			neuron_sum = steepness*neuron_sum;
			float max_sum = 150/steepness;
			if(neuron_sum > max_sum)
				neuron_sum = max_sum;
			else if(neuron_sum < -max_sum)
				neuron_sum = -max_sum;

			fann_activation_switch(activation_function, neuron_sum, result);
			hidden_curr[i] = result;
		}

		hidden_prev = hidden_curr;
		if (k < num_layers-3)
			hidden_curr = (hidden_curr == hidden_1_s) ? hidden_2_s : hidden_1_s;
		else
			hidden_curr = output;
	}
}

__global__ void gpu_backpropagate_MSE_single_step(float *weight, unsigned int weight_start_idx, unsigned int num_neurons_prev_layers, unsigned int num_neurons_curr_layers, float *hidden_prev, float *error_prev_layer, float *error_curr_layer) {

	unsigned int i, j;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
    /*	for (i=0; i<num_neurons_curr_layers; i++) {
		float tmp_error = error_curr_layer[i];
		for (j=0; j<num_neurons_prev_layers; j++) {
			error_prev_layer[j] += tmp_error * weight[weight_start_idx+j*(num_neurons_curr_layers)+i];
		}
		error_prev_layer[i] *= fann_activation_derived_device(FANN_SIGMOID_SYMMETRIC_STEPWISE, 1.0, hidden_prev[i], 0);
	}
*/

	if (idx < num_neurons_curr_layers) {
		float tmp_error = error_curr_layer[idx];
		for (j=0; j<num_neurons_prev_layers; j++) {
			error_prev_layer[j] += tmp_error * weight[weight_start_idx+j*(num_neurons_curr_layers)+idx];
		}
		error_prev_layer[idx] *= fann_activation_derived_device(FANN_SIGMOID_SYMMETRIC_STEPWISE, 1.0, hidden_prev[idx], 0);
	}	
	
}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_fann_run(struct fann *ann, fann_type *input, unsigned int num_output, fann_type *output) {
	fann_type * calc_out = fann_run(ann, input);

	for (int i=0; i<num_output; i++)
		output[i] = calc_out[i];
}

void cpu_custom_run(unsigned int num_layers, unsigned int *num_neurons_layers,
	float *input_h, float *weight_h, unsigned int *weight_idx_h, float *hidden_h, unsigned int *hidden_ptr_h, float *output_h) {
	
	unsigned int i, j, k;
	float *hidden_prev = &input_h[0];
	float *hidden_curr = (num_layers > 2) ? (&hidden_h[0]) : (output_h);
	float max_sum, neuron_sum;
	unsigned int activation_function = FANN_SIGMOID_SYMMETRIC;
	fann_type steepness = 1;

	for (k=0; k<num_layers-1; k++) {
		hidden_prev[num_neurons_layers[k]] = 1;
		for (i=0; i<num_neurons_layers[k+1]; i++) {
			neuron_sum = 0;
			for (j=0; j<num_neurons_layers[k]+1; j++) {
				neuron_sum += weight_h[weight_idx_h[k]+j*(num_neurons_layers[k+1])+i]*hidden_prev[j];
			}
			neuron_sum = steepness*neuron_sum;
			max_sum = 150/steepness;
			if(neuron_sum > max_sum)
				neuron_sum = max_sum;
			else if(neuron_sum < -max_sum)
				neuron_sum = -max_sum;

			fann_activation_switch(activation_function, neuron_sum, hidden_curr[i]);
		}
		hidden_prev = &hidden_h[hidden_ptr_h[k]];
		if (k < num_layers-3)
			hidden_curr = &hidden_h[hidden_ptr_h[k+1]];
		else
			hidden_curr = output_h;
	}
}

void gpu_fann_run(unsigned int num_layers, unsigned int *num_neurons_layers,
	float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int k;
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden) : (output);

	for (k=0; k<num_layers-1; k++) {
	    const unsigned int numThreadsPerBlock = 512;
	    const unsigned int numBlocks = (num_neurons_layers[k+1] - 1)/numThreadsPerBlock + 1;
		gpu_run_kernel_single_step <<< numBlocks , numThreadsPerBlock >>>
			(weight, weight_idx[k], num_neurons_layers[k], num_neurons_layers[k+1], hidden_prev, hidden_curr);

		hidden_prev = hidden + hidden_ptr[k];
		if (k < num_layers-3)
			hidden_curr = hidden + hidden_ptr[k+1];
		else
			hidden_curr = output;
	}
}

void gpu_opt1_run(unsigned int num_layers, unsigned int *num_neurons_layers,
	float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int k;
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden) : (output);

	for (k=0; k<num_layers-1; k++) {
	    const unsigned int numThreadsPerBlock = 512;
	    const unsigned int numBlocks = (num_neurons_layers[k+1] - 1)/numThreadsPerBlock + 1;
		gpu_run_kernel_single_step_opt1 <<< numBlocks , numThreadsPerBlock >>>
			(weight, weight_idx[k], num_neurons_layers[k], num_neurons_layers[k+1], hidden_prev, hidden_curr);

		hidden_prev = hidden + hidden_ptr[k];
		if (k < num_layers-3)
			hidden_curr = hidden + hidden_ptr[k+1];
		else
			hidden_curr = output;
	}
}

void gpu_fann_run_opt_shared(unsigned int num_layers, unsigned int *num_neurons_layers,
	float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int k;
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden) : (output);

	for (k=0; k<num_layers-1; k++) {
	    const unsigned int numThreadsPerBlock = 512;
	    const unsigned int numBlocks = (num_neurons_layers[k+1] - 1)/numThreadsPerBlock + 1;
		gpu_run_kernel_opt_shared_single_step <<< numBlocks , numThreadsPerBlock >>>
			(weight, weight_idx[k], num_neurons_layers[k], num_neurons_layers[k+1], hidden_prev, hidden_curr);

		hidden_prev = hidden + hidden_ptr[k];
		if (k < num_layers-3)
			hidden_curr = hidden + hidden_ptr[k+1];
		else
			hidden_curr = output;
	}
}

void gpu_fann_run_opt_reduction(unsigned int num_layers, unsigned int *num_neurons_layers,
	float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

	unsigned int k;
	float *hidden_prev = input;
	float *hidden_curr = (num_layers > 2) ? (hidden) : (output);

	for (k=0; k<num_layers-1; k++) {
	    const dim3 numThreadsPerBlock(8,128);
	    const dim3 numBlocks((num_neurons_layers[k+1] - 1)/numThreadsPerBlock.x + 1);
		gpu_run_kernel_opt_reduction_single_step <<< numBlocks , numThreadsPerBlock >>>
			(weight, weight_idx[k], num_neurons_layers[k], num_neurons_layers[k+1], hidden_prev, hidden_curr);

		hidden_prev = hidden + hidden_ptr[k];
		if (k < num_layers-3)
			hidden_curr = hidden + hidden_ptr[k+1];
		else
			hidden_curr = output;
	}
}

void gpu_fann_run_opt_full(unsigned int num_layers, unsigned int max_neurons, unsigned int *num_neurons_layers, float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

    const unsigned int numThreadsPerBlock = 1024;
    const unsigned int numBlocks = 1;
	gpu_run_kernel_opt_full <<< numBlocks , numThreadsPerBlock >>>
		(num_layers, max_neurons, num_neurons_layers, input, weight, weight_idx, hidden, hidden_ptr, output);
}

void gpu_fann_run_opt_full_no_hidden_val(unsigned int num_layers, unsigned int max_neurons, unsigned int *num_neurons_layers, float *input, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *output) {

    const unsigned int numThreadsPerBlock = 1024;
    const unsigned int numBlocks = 1;
	gpu_run_kernel_opt_full_no_hidden_val <<< numBlocks , numThreadsPerBlock >>>
		(num_layers, max_neurons, num_neurons_layers, input, weight, weight_idx, hidden, hidden_ptr, output);
}

void cpu_compute_MSE(unsigned int num_output, float *desired_output,
	float *actual_output, float *error_output, float *mse) {
	unsigned int i;
	float total_error_squared = 0;

	for (i=0; i<num_output; i++) {
		float neuron_value = actual_output[i];
		float neuron_desired_value = desired_output[i];
		float neuron_diff = neuron_desired_value - neuron_value;
		neuron_diff /= 2;
		total_error_squared += (neuron_diff * neuron_diff);

		if(neuron_diff < -.9999999)
			neuron_diff = -17.0;
		else if(neuron_diff > .9999999)
			neuron_diff = 17.0;
		else
			neuron_diff = (fann_type) log((1.0 + neuron_diff) / (1.0 - neuron_diff));

		error_output[i] = neuron_diff * fann_activation_derived(FANN_SIGMOID_SYMMETRIC_STEPWISE, 1.0, neuron_value, 0);
	}

	*mse = total_error_squared/num_output;
}

void cpu_backpropagate_MSE(unsigned int num_layers, unsigned int *num_neurons_layers, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *error_output, float *error_hiddens) {
	fann_type tmp_error;
	unsigned int i;
	unsigned int j;
	unsigned int k;

	fann_type *error_curr_layer = error_output;
	fann_type *error_prev_layer;
	fann_type *hidden_prev;

	/* go through all the layers, from last to first.
	 * And propagate the error backwards */
	for (k=num_layers-1; k>1; k--)
	{
		/* for each connection in this layer, propagate the error backwards */
		error_prev_layer = &error_hiddens[hidden_ptr[k-2]];
		hidden_prev = &hidden[hidden_ptr[k-2]];

		for (i=0; i<num_neurons_layers[k]; i++) {
			tmp_error = error_curr_layer[i];
			for (j=0; j<num_neurons_layers[k-1]; j++) {
				error_prev_layer[j] += tmp_error * weight[weight_idx[k-1]+j*(num_neurons_layers[k])+i];
			}
			error_prev_layer[i] *= fann_activation_derived(FANN_SIGMOID_SYMMETRIC_STEPWISE, 1.0, hidden_prev[i], 0);
		}

		error_curr_layer = error_prev_layer;
	}
}

void gpu_backpropagate_MSE(unsigned int num_layers, unsigned int *num_neurons_layers, float *weight, unsigned int *weight_idx, float *hidden, unsigned int *hidden_ptr, float *error_output, float *error_hiddens) {
	fann_type tmp_error;
	unsigned int i;
	unsigned int j;
	unsigned int k;

	fann_type *error_curr_layer = error_output;
	fann_type *error_prev_layer;
	fann_type *hidden_prev;

	/* go through all the layers, from last to first.
	 * And propagate the error backwards */
	for (k=num_layers-1; k>1; k--)
	{
		/* for each connection in this layer, propagate the error backwards */
		error_prev_layer = &error_hiddens[hidden_ptr[k-2]];
		hidden_prev = &hidden[hidden_ptr[k-2]];

	    const unsigned int numThreadsPerBlock = 512;
	    const unsigned int numBlocks = (num_neurons_layers[k] - 1)/numThreadsPerBlock + 1;
		gpu_backpropagate_MSE_single_step <<< numBlocks , numThreadsPerBlock >>>
	    //gpu_backpropagate_MSE_single_step <<< 1, 1 >>>
			(weight, weight_idx[k-1], num_neurons_layers[k-1], num_neurons_layers[k], hidden_prev, error_prev_layer, error_curr_layer);
		
		error_curr_layer = error_prev_layer;
	}
}

void cpu_update_weights()
{
#if 0
	struct fann_neuron *neuron_it, *last_neuron, *prev_neurons;
	fann_type tmp_error, delta_w, *weights;
	struct fann_layer *layer_it;
	unsigned int i, j, k;
	unsigned int num_connections;

	/* store some variabels local for fast access */
	const float learning_rate = 0.7f;
    const float learning_momentum = ann->learning_momentum;        
	struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
	struct fann_layer *first_layer = ann->first_layer;
	const struct fann_layer *last_layer = ann->last_layer;
	fann_type *error_begin = ann->train_errors;
	fann_type *deltas_begin, *weights_deltas;

	deltas_begin = ann->prev_weights_deltas;
	prev_neurons = first_neuron;

	for (k=1; k<num_layers; k++)
	//for(layer_it = (first_layer + 1); layer_it != last_layer; layer_it++)
	{
		//last_neuron = layer_it->last_neuron;
		//prev_neurons = (layer_it - 1)->first_neuron;
		for (j=0; j<num_neurons_layers[k]; j++)
		//for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			tmp_error = error_hiddens[hidden_ptr[k-1]] * learning_rate;
			num_connections = neuron_it->last_con - neuron_it->first_con;
			weights = ann->weights + neuron_it->first_con;
			weights_deltas = deltas_begin + neuron_it->first_con;
			for(i = 0; i != num_connections; i++)
			{
				delta_w = tmp_error * prev_neurons[i].value + learning_momentum * weights_deltas[i];
				weights[i] += delta_w ;
				weights_deltas[i] = delta_w;
			}
		}
	}
#endif
}
