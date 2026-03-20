#include <iostream>
#include <curand_kernel.h>

#define seed 1234
#define FOOD_SPAWN_RATE 0.1f





#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \

__global__ void initialize_map(
    unsigned long long int* map,
    curandState* random_states,
    int map_width,
    int map_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= map_width * map_height) return;

    curand_init(seed, idx, 0, &random_states[idx]);

    float rand_val = curand_uniform(&random_states[idx]);

    // 1 - food
    // 0 - empty
    map[idx] = (unsigned long long int)(rand_val < FOOD_SPAWN_RATE ? 1 : 0); 
}


int main() {
    printf("Starting simulation...\n");

    // Map
    int map_width = 1024;
    int map_height = 1024;
    unsigned long long int* d_map;
    CUDA_CHECK(cudaMalloc(&d_map, map_width * map_height * sizeof(unsigned long long int)));

    // Randomness source
    curandState* d_random_states;
    CUDA_CHECK(cudaMalloc(&d_random_states, map_width * map_height * sizeof(curandState)));

    // Creatures
    int creature_n = 1024; // number of creatures
    unsigned int* d_creature_x;
    unsigned int* d_creature_y;
    unsigned long long int* d_creature_energy;
    CUDA_CHECK(cudaMalloc(&d_creature_x, creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_y, creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_energy, creature_n * sizeof(unsigned long long int)));

    printf("Initializing map...\n");
    initialize_map<<<(map_width * map_height + 255) / 256, 256>>>(d_map, d_random_states, map_width, map_height);

    CUDA_CHECK(cudaFree(d_map));
    CUDA_CHECK(cudaFree(d_creature_x));
    CUDA_CHECK(cudaFree(d_creature_y));
    CUDA_CHECK(cudaFree(d_creature_energy));
    CUDA_CHECK(cudaFree(d_random_states));
    return 0;
}