#include "creatures/networks.cuh"
#include "constants.h"
#include <iostream>

Networks::Networks(curandState* state, int creatures_n) {
    cudaMalloc(&first_matrix, MAX_CREATURE_N * MAX_SENSORS * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&second_matrix, MAX_CREATURE_N * HIDDEN_NEURONS * MAX_OUTPUT_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&bias, MAX_CREATURE_N * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&hidden_neuron_values, MAX_CREATURE_N * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&output_neuron_values, MAX_CREATURE_N * MAX_OUTPUT_NEURONS * sizeof(__nv_fp8_e4m3));

    Networks* d_this;
    cudaMalloc(&d_this, sizeof(Networks));
    cudaMemcpy(d_this, this, sizeof(Networks), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    InitializeRandomNetworks<<<(creatures_n + 255) / 256, 256>>>(d_this, creatures_n, state);
    cudaDeviceSynchronize();
}

Networks::~Networks() {
    cudaFree(first_matrix);
    cudaFree(second_matrix);
    cudaFree(bias);
    cudaFree(hidden_neuron_values);
    cudaFree(output_neuron_values); 
}

__global__ void InitializeRandomNetworks(Networks* networks, int creatures_n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < creatures_n) {
        networks->AddRandomNetwork(idx, states[idx]);
    }
}

__device__ void Networks::AddRandomNetwork(int creature_index, curandState &state) {
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < MAX_SENSORS; sensor_idx++) {
            size_t idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            first_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < MAX_OUTPUT_NEURONS; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
            size_t idx = get_second_matrix_idx(creature_index, output_idx, hidden_idx);
            second_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
        size_t idx = (creature_index * HIDDEN_NEURONS) + hidden_idx;
        bias[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
    }
}

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (creature_idx * HIDDEN_NEURONS * MAX_SENSORS) + (hidden_idx * MAX_SENSORS) + sensor_idx;
}

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (creature_idx * MAX_OUTPUT_NEURONS * HIDDEN_NEURONS) + (output_idx * HIDDEN_NEURONS) + hidden_idx;
}