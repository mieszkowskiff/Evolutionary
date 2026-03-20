#include <iostream>
#include <curand_kernel.h>
#include <csignal>
#include <unistd.h>

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

#define seed 1234
#define FOOD_SPAWN_RATE 0.1f
#define INITIAL_CREATURE_ENERGY 100

// bit flags for map cells
#define   BIT_FOOD 0
#define   BIT_CREATURE 1

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \


volatile std::sig_atomic_t interrupted = 0;

extern "C" void signal_handler(int signum) {
    interrupted = 1;
}

__global__ void initialize_random_states(curandState* random_states, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    curand_init(seed, idx, 0, &random_states[idx]);
}


__global__ void initialize_map(
    unsigned int* map,
    curandState* random_states,
    int map_width,
    int map_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= map_width * map_height) return;

    float rand_val = curand_uniform(&random_states[idx]);

    map[idx] = 0; // Clear cell
    map[idx] |= (rand_val < FOOD_SPAWN_RATE ? 1 << BIT_FOOD : 0);

}

__global__ void initialize_creatures(
    unsigned int* creature_x,
    unsigned int* creature_y,
    unsigned int* creature_energy,
    curandState* random_states,
    int creature_n,
    int map_width,
    int map_height,
    unsigned int* map
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    creature_x[idx] = curand(&random_states[idx]) % map_width;
    creature_y[idx] = curand(&random_states[idx]) % map_height;
    creature_energy[idx] = INITIAL_CREATURE_ENERGY;
    atomicOr(&map[creature_y[idx] * map_width + creature_x[idx]], 1 << BIT_CREATURE);
}


int main() {
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    printf("Starting simulation...\n");

    // Map
    int map_width = 1024;
    int map_height = 1024;
    unsigned int* d_map; // d_map is considered as a bit vector (see CellFlags)

    CUDA_CHECK(cudaMalloc(&d_map, map_width * map_height * sizeof(unsigned int)));

    // Creatures
    int creature_n = 1024; // number of creatures
    unsigned int* d_creature_x;
    unsigned int* d_creature_y;
    unsigned int* d_creature_energy;
    CUDA_CHECK(cudaMalloc(&d_creature_x, creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_y, creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_energy, creature_n * sizeof(unsigned int)));

    // Randomness source
    curandState* d_random_states;
    CUDA_CHECK(cudaMalloc(&d_random_states, map_width * map_height * sizeof(curandState)));

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(map_width, map_height);
    #endif

    printf("Initializing random states...\n");
    initialize_random_states<<<(map_width * map_height + 255) / 256, 256>>>(
        d_random_states, 
        map_width * map_height
    );
    cudaDeviceSynchronize();

    printf("Initializing map...\n");


    initialize_map<<<(map_width * map_height + 255) / 256, 256>>>(
        d_map, 
        d_random_states, 
        map_width, 
        map_height
    );

    cudaDeviceSynchronize();

    

    printf("Initializing creatures...\n");
    initialize_creatures<<<(creature_n + 255) / 256, 256>>>(
        d_creature_x, 
        d_creature_y, 
        d_creature_energy, 
        d_random_states, 
        creature_n, 
        map_width, 
        map_height, 
        d_map
    );

    cudaDeviceSynchronize();


    //main loop
    bool running = true;
    while (running) {





        #ifdef ENABLE_DISPLAY
        display.renderFrame(d_map);

        if(display.shouldClose()){
            running = false;
        }
        #endif

        if (interrupted) {
            running = false;
        }
    }

    printf("\nQuitting the simulation. Shutting down...\n");
    fflush(stdout);

    CUDA_CHECK(cudaFree(d_map));
    CUDA_CHECK(cudaFree(d_creature_x));
    CUDA_CHECK(cudaFree(d_creature_y));
    CUDA_CHECK(cudaFree(d_creature_energy));
    CUDA_CHECK(cudaFree(d_random_states));
    return 0;
}