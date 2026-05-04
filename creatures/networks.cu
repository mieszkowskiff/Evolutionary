#include "creatures/networks.cuh"
#include "constants.h"

Networks::Networks(int8_t* input_neurons_n, int8_t* output_neurons_n, int creature_n) {
    this->input_neurons_n = input_neurons_n;
    cudaMalloc(&hidden_neurons_n, MAX_CREATURE_N * sizeof(int8_t));
    this->output_neurons_n = output_neurons_n;

    cudaMalloc(&first_matrix, MAX_CREATURE_N * MAX_SENSORS * MAX_HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&second_matrix, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * MAX_OUTPUT_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&bias, MAX_CREATURE_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&hidden_neuron_values, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&output_neuron_values, MAX_CREATURE_N * MAX_OUTPUT_NEURONS * sizeof(__nv_fp8_e4m3));

}

Networks::~Networks() {
    cudaFree(hidden_neurons_n);
    cudaFree(first_matrix);
    cudaFree(second_matrix);
    cudaFree(bias);
    cudaFree(hidden_neuron_values);
    cudaFree(output_neuron_values); 
}

__device__ void Networks::AddRandomNetwork(int creature_index, curandState &state) {
    int input_neurons = input_neurons_n[creature_index];
    int hidden_neurons = hidden_neurons_n[creature_index];
    int output_neurons = output_neurons_n[creature_index];
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < hidden_neurons; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < input_neurons; sensor_idx++) {
            size_t idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            first_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < output_neurons; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < hidden_neurons; hidden_idx++) {
            size_t idx = get_second_matrix_idx(creature_index, output_idx, hidden_idx);
            second_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < hidden_neurons; hidden_idx++) {
        size_t idx = (creature_index * MAX_HIDDEN_NEURONS) + hidden_idx;
        bias[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
    }
}

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (creature_idx * MAX_HIDDEN_NEURONS * MAX_SENSORS) + (hidden_idx * MAX_SENSORS) + sensor_idx;
}

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (creature_idx * MAX_OUTPUT_NEURONS * MAX_HIDDEN_NEURONS) + (output_idx * MAX_HIDDEN_NEURONS) + hidden_idx;
}