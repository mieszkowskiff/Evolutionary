#include "creatures/sensors.cuh"
#include "constants.h"


Sensors::Sensors(curandState* state, int creatures_n) {

    cudaMalloc(&sensor_x, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));
    cudaMalloc(&sensor_y, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));
    cudaMalloc(&sensor_type, MAX_CREATURE_N * MAX_SENSORS * sizeof(int8_t));

    dim3 blockSize(16, 16);
    dim3 gridSize((creatures_n + blockSize.x - 1) / blockSize.x, (MAX_SENSORS + blockSize.y - 1) / blockSize.y);

    cudaDeviceSynchronize();
    SetSensors<<<gridSize, blockSize>>>(creatures_n, state);
}

Sensors::~Sensors() {
    cudaFree(sensor_x);
    cudaFree(sensor_y);
    cudaFree(sensor_type);
}

__device__ void Sensors::AddRandomSensor(int creature_index, int sensor_index, curandState& state) {

    float x_normal = curand_normal(&state) * SENSOR_STDDEV;
    float y_normal = curand_normal(&state) * SENSOR_STDDEV;
    
    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = curand(&state) % 10; // 0: food, 1: danger, 2: creature, 3-9: empty

    sensor_x[sensor_index * MAX_CREATURE_N + creature_index] = x;
    sensor_y[sensor_index * MAX_CREATURE_N + creature_index] = y;
    sensor_type[sensor_index * MAX_CREATURE_N + creature_index] = type;
}


__global__ void Sensors::SetSensors(int creatures_n, curandState* state) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    int sensor_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (creature_index >= creatures_n) return;
    AddRandomSensor(creature_index, sensor_index, state[creature_index]);
}