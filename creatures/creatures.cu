#include "creatures/creatures.cuh"
#include "constants.h"
#include <stdio.h>


Creatures::Creatures(curandState* state, int count) {
    this->count = count;

    h_data = new CreatureData;

    cudaMalloc(&h_data->x, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->y, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->energy, MAX_CREATURE_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&h_data->sensor_x, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_y, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_type, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));

    cudaMalloc(&h_data->first_matrix, MAX_CREATURE_N * SENSORS_N * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->second_matrix, MAX_CREATURE_N * HIDDEN_NEURONS * ACTIONS_N * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->bias, MAX_CREATURE_N * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&h_data->action_x, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_y, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_type, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));

    cudaMalloc(&h_data->hidden_neuron_values, MAX_CREATURE_N * HIDDEN_NEURONS * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->output_neuron_values, MAX_CREATURE_N * ACTIONS_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&d_data, sizeof(CreatureData));
    cudaMemcpy(d_data, h_data, sizeof(CreatureData), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    InitializeRandomCreatures<<<(count + 255) / 256, 256>>>(d_data, count, state);
    cudaDeviceSynchronize();
}

Creatures::~Creatures() {
    cudaFree(h_data->x);
    cudaFree(h_data->y);
    cudaFree(h_data->energy);

    cudaFree(h_data->sensor_x);
    cudaFree(h_data->sensor_y);
    cudaFree(h_data->sensor_type);

    cudaFree(h_data->first_matrix);
    cudaFree(h_data->second_matrix);
    cudaFree(h_data->bias);

    cudaFree(h_data->action_x);
    cudaFree(h_data->action_y);
    cudaFree(h_data->action_type);

    cudaFree(h_data->hidden_neuron_values);
    cudaFree(h_data->output_neuron_values);

    cudaFree(h_data);
    delete h_data;
}

__global__ void InitializeRandomCreatures(CreatureData* creatures, int count, curandState* states) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    curandState state = states[creature_index];

    //Initialize position and energy
    creatures->x[creature_index] = curand(&state) % WIDTH;
    creatures->y[creature_index] = curand(&state) % HEIGHT;
    creatures->energy[creature_index] = __nv_fp8_e4m3(1.0f);

    // Initialize sensors
    for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
        AddRandomSensors(creatures, creature_index, sensor_idx, state);
    }

    // Initialize network
    AddRandomNetwork(creatures, creature_index, state);

    // Initialize actions
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        SetRandomAction(creatures, creature_index, action_idx, state);
    }
}

__device__ void AddRandomSensors(CreatureData* creatures, int creature_index, int sensor_index, curandState& state) {
        float x_normal = curand_normal(&state) * SENSOR_STDDEV;
        float y_normal = curand_normal(&state) * SENSOR_STDDEV;
        
        int8_t x = static_cast<int8_t>(roundf(x_normal));
        int8_t y = static_cast<int8_t>(roundf(y_normal));
        int8_t type = curand(&state) % 10; // 0: food, 1: danger, 2: creature, 3-9: empty

        creatures->sensor_x[sensor_index * MAX_CREATURE_N + creature_index] = x;
        creatures->sensor_y[sensor_index * MAX_CREATURE_N + creature_index] = y;
        creatures->sensor_type[sensor_index * MAX_CREATURE_N + creature_index] = type;
}

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, curandState &state) {
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
            size_t idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            creatures->first_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < ACTIONS_N; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
            size_t idx = get_second_matrix_idx(creature_index, output_idx, hidden_idx);
            creatures->second_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS; hidden_idx++) {
        size_t idx = (creature_index * HIDDEN_NEURONS) + hidden_idx;
        creatures->bias[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
    }
}

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (creature_idx * HIDDEN_NEURONS * SENSORS_N) + (hidden_idx * SENSORS_N) + sensor_idx;
}

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (creature_idx * ACTIONS_N * HIDDEN_NEURONS) + (output_idx * HIDDEN_NEURONS) + hidden_idx;
}

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, curandState& state) {
    float x_normal = curand_normal(&state) * ACTION_STDDEV;
    float y_normal = curand_normal(&state) * ACTION_STDDEV;
    
    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = curand(&state) % 10; // 0: move, 1: eat, 2: attack, 3: reproduce, 4-9 no action (placeholder)

    creatures->action_x[action_index * MAX_CREATURE_N + creature_index] = x;
    creatures->action_y[action_index * MAX_CREATURE_N + creature_index] = y;
    creatures->action_type[action_index * MAX_CREATURE_N + creature_index] = type;
}

