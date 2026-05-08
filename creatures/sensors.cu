#include "creatures/sensors.cuh"
#include "constants.h"
#include <stdio.h>


Sensors::Sensors(curandState* state, int creatures_n) {
    cudaMalloc(&sensor_x, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));
    cudaMalloc(&sensor_y, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));
    cudaMalloc(&sensor_type, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));

    Sensors* d_this;
    cudaMalloc(&d_this, sizeof(Sensors));
    cudaMemcpy(d_this, this, sizeof(Sensors), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    SetSensors<<<(creatures_n + 255) / 256, 256>>>(d_this, creatures_n, state); 
    cudaDeviceSynchronize();
}

Sensors::~Sensors() {
    cudaFree(sensor_x);
    cudaFree(sensor_y);
    cudaFree(sensor_type);
}

__device__ void Sensors::AddRandomSensors(int creature_index, int sensor_index, curandState& state) {
        float x_normal = curand_normal(&state) * SENSOR_STDDEV;
        float y_normal = curand_normal(&state) * SENSOR_STDDEV;
        
        int8_t x = static_cast<int8_t>(roundf(x_normal));
        int8_t y = static_cast<int8_t>(roundf(y_normal));
        int8_t type = curand(&state) % 10; // 0: food, 1: danger, 2: creature, 3-9: empty

        sensor_x[sensor_index * MAX_CREATURE_N + creature_index] = x;
        sensor_y[sensor_index * MAX_CREATURE_N + creature_index] = y;
        sensor_type[sensor_index * MAX_CREATURE_N + creature_index] = type;
}


__global__ void SetSensors(Sensors* sensors, int creatures_n, curandState* state) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= creatures_n) return;

    for(int sensor_index = 0; sensor_index < MAX_SENSORS; sensor_index++) {
        sensors->AddRandomSensors(creature_index, sensor_index, state[creature_index]);
    }
}