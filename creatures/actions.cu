#include "creatures/actions.cuh"
#include "constants.h"



Actions::Actions(curandState* state, int creatures_n) {
    cudaMalloc(&action_x, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));
    cudaMalloc(&action_y, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));
    cudaMalloc(&action_type, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));

    Actions* d_this;
    cudaMalloc(&d_this, sizeof(Actions));
    cudaMemcpy(d_this, this, sizeof(Actions), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    SetActions<<<(creatures_n + 255) / 256, 256>>>(d_this, creatures_n, state);
    cudaDeviceSynchronize();
}

Actions::~Actions() {
    cudaFree(action_x);
    cudaFree(action_y);
    cudaFree(action_type);
}

__device__ void Actions::SetRandomAction(int creature_index, int action_index, curandState& state) {
    float x_normal = curand_normal(&state) * ACTION_STDDEV;
    float y_normal = curand_normal(&state) * ACTION_STDDEV;
    
    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = curand(&state) % 10; // 0: move, 1: eat, 2: attack, 3: reproduce, 4-9 no action (placeholder)
}

__global__ void SetActions(Actions* actions, int creatures_n, curandState* state) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (creature_index >= creatures_n) return;

    for (int action_index = 0; action_index < MAX_ACTIONS; action_index++) {
        actions->SetRandomAction(creature_index, action_index, state[creature_index]);
    }
}

