#include "initialization/initialization.cuh"
#include "constants.h"
#include <curand_kernel.h>
#include "utils/utils.cuh"

__global__ void initialize_random_states(curandState* random_states, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    curand_init(1290, idx, 0, &random_states[idx]);
}

__global__ void initialize_creatures(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int* creature_sensors_n,
    int* creature_hidden_neurons_n,
    int* creature_sensor_x,
    int* creature_sensor_y,
    int* creature_sensor_type,
    curandState* random_states,
    int creature_n,
    int map_width,
    int map_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    creature_x[idx] = curand(&random_states[idx]) % map_width;
    creature_y[idx] = curand(&random_states[idx]) % map_height;
    creature_energy[idx] = INITIAL_CREATURE_ENERGY;

    creature_sensors_n[idx] = 5;
    creature_hidden_neurons_n[idx] = 6;

    creature_sensor_x[idx + 0 * MAX_CREATURE_N] = 0;
    creature_sensor_y[idx + 0 * MAX_CREATURE_N] = 0;
    creature_sensor_type[idx + 0 * MAX_CREATURE_N] = BIT_FOOD;

    creature_sensor_x[idx + 1 * MAX_CREATURE_N] = 1;
    creature_sensor_y[idx + 1 * MAX_CREATURE_N] = 0;
    creature_sensor_type[idx + 1 * MAX_CREATURE_N] = BIT_FOOD;

    creature_sensor_x[idx + 2 * MAX_CREATURE_N] = -1;
    creature_sensor_y[idx + 2 * MAX_CREATURE_N] = 0;
    creature_sensor_type[idx + 2 * MAX_CREATURE_N] = BIT_FOOD;

    creature_sensor_x[idx + 3 * MAX_CREATURE_N] = 0;
    creature_sensor_y[idx + 3 * MAX_CREATURE_N] = 1;
    creature_sensor_type[idx + 3 * MAX_CREATURE_N] = BIT_FOOD;

    creature_sensor_x[idx + 4 * MAX_CREATURE_N] = 0;
    creature_sensor_y[idx + 4 * MAX_CREATURE_N] = -1;
    creature_sensor_type[idx + 4 * MAX_CREATURE_N] = BIT_FOOD;

    for (int i = 0; i < creature_hidden_neurons_n[idx]; i++) {
        
    }
}

__global__ void initialize_


