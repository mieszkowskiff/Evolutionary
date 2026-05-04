#include "creatures/actions.cuh"
#include "constants.h"

Actions::Actions() {
    cudaMalloc(&actions_n, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&action_x, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));
    cudaMalloc(&action_y, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));
    cudaMalloc(&action_type, MAX_CREATURE_N * MAX_ACTIONS * sizeof(int8_t));
}

Actions::~Actions() {
    cudaFree(actions_n);
    cudaFree(action_x);
    cudaFree(action_y);
    cudaFree(action_type);
}