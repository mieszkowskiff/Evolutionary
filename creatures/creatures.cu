#include "creatures/creatures.cuh"
#include "constants.h"
#include <stdio.h>


Creatures::Creatures(curandState* state, int count) : sensors(state, count), networks(state, count), actions(state, count) {
    this->count = count;
    cudaMalloc(&x, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&y, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&energy, MAX_CREATURE_N * sizeof(__nv_fp8_e4m3));
}

Creatures::~Creatures() {
    cudaFree(x);
    cudaFree(y);
    cudaFree(energy);
    sensors.~Sensors();
    networks.~Networks();
    actions.~Actions();
}

__global__ void InitializeRandomCreatures(Creatures* creatures, curandState* states) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= creatures->count) return;

    curandState state = states[creature_index];

    // Initialize position and energy
    creatures->x[creature_index] = curand(&state) % WIDTH;
    creatures->y[creature_index] = curand(&state) % HEIGHT;
    creatures->energy[creature_index] = __nv_fp8_e4m3(1.0f);
}