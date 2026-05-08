#include "map.cuh"
#include "constants.h"

Map::Map() {

    size_t bytes = WIDTH * HEIGHT * sizeof(__nv_fp8_e4m3);

    cudaMalloc(&food, bytes);
    cudaMalloc(&danger, bytes);
    cudaMalloc(&creature, bytes);

    cudaMemset(food, 0, bytes);
    cudaMemset(danger, 0, bytes);
    cudaMemset(creature, 0, bytes);
}

Map::~Map() {
    cudaFree(food);
    cudaFree(danger);
    cudaFree(creature);
}

__device__ int Map::get_cell_index(int x, int y) {
    int nx = x;
    int ny = y;

    if (nx < 0) {
        nx += WIDTH;
    } else if (nx >= WIDTH) {
        nx -= WIDTH;
    }

    if (ny < 0) {
        ny += HEIGHT;
    } else if (ny >= HEIGHT) {
        ny -= HEIGHT;
    }

    return ny * WIDTH + nx;
}

__global__ void place_food(Map* map, int max_food_count, curandState* random_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_food_count) return;

    int rand_x = (int)(curand_uniform(&random_states[idx]) * WIDTH);
    int rand_y = (int)(curand_uniform(&random_states[idx]) * HEIGHT);

    map->food[map->get_cell_index(rand_x, rand_y)] = (__nv_fp8_e4m3)1.0f;
}

__device__ __nv_fp8_e4m3 Map::get_cell(int layer, int index) const {
        switch (layer) {
            case 0: return food[index];
            case 1: return danger[index];
            case 2: return creature[index];
            default: return (__nv_fp8_e4m3)0.0f; 
        }
    }