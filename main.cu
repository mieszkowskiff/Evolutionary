#include <iostream>
#include <curand_kernel.h>
#include <csignal>
#include <unistd.h>

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

#define seed 1234
#define FOOD_SPAWN_RATE 0.1f
#define INITIAL_CREATURE_ENERGY 1

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



__device__ int get_cell_index(int x, int y, int map_width, int map_height) {
    int nx = x;
    int ny = y;

    if (nx < 0) {
        nx += map_width;
    } else if (nx >= map_width) {
        nx -= map_width;
    }

    if (ny < 0) {
        ny += map_height;
    } else if (ny >= map_height) {
        ny -= map_height;
    }

    return ny * map_width + nx;
}

__global__ void place_food(
    unsigned int* map,
    curandState* random_states,
    int map_width,
    int map_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= map_width * map_height) return;

    float rand_val = curand_uniform(&random_states[idx]);
    map[idx] |= (rand_val < FOOD_SPAWN_RATE ? 1 << BIT_FOOD : 0);
}

__global__ void initialize_creatures(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    curandState* random_states,
    int creature_n,
    int max_creature_n,
    int map_width,
    int map_height,
    unsigned int* map,
    float* creature_matrix,
    float* creature_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    for(int i = 0; i < 6 * 6; i++) {
        creature_matrix[idx + i * max_creature_n] = curand_uniform(&random_states[idx]) * 2 - 1; // random weights in [-1, 1]
    }
    for(int i = 0; i < 6; i++) {
        creature_bias[idx + i * max_creature_n] = curand_uniform(&random_states[idx]) * 2 - 1; // random biases in [-1, 1]
    }

    creature_x[idx] = curand(&random_states[idx]) % map_width;
    creature_y[idx] = curand(&random_states[idx]) % map_height;
    creature_energy[idx] = INITIAL_CREATURE_ENERGY;

    atomicOr(&map[get_cell_index(creature_x[idx], creature_y[idx], map_width, map_height)], 1 << BIT_CREATURE);
}

__global__ void creature_action_step(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int creature_n,
    int max_creature_n,
    int map_width,
    int map_height,
    unsigned int* map,
    float* creature_matrix,
    float* creature_bias,
    int* creature_by_actions,
    int* action_counts,
    curandState* random_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    extern __shared__ float neuron_values[];
    /*
    Sensors:
    0: Energy level (normalized)
    1: Food presence in current cell
    2: Food presence in right cell
    3: Food presence in left cell
    4: Food presence in down cell
    5: Food presence in up cell
    */
    // Load sensor parameters into shared memory
    neuron_values[threadIdx.x + 0 * blockDim.x] = creature_energy[idx]; // Energy as input

    // Food presence as input
    neuron_values[threadIdx.x + 1 * blockDim.x] = 
    (map[get_cell_index(creature_x[idx], creature_y[idx], map_width, map_height)] & (1 << BIT_FOOD)) ? 1.0f : 0.0f;

    neuron_values[threadIdx.x + 2 * blockDim.x] = 
    (map[get_cell_index(creature_x[idx] + 1, creature_y[idx], map_width, map_height)] & (1 << BIT_FOOD)) ? 1.0f : 0.0f;
    neuron_values[threadIdx.x + 3 * blockDim.x] = 
    (map[get_cell_index(creature_x[idx] - 1, creature_y[idx], map_width, map_height)] & (1 << BIT_FOOD)) ? 1.0f : 0.0f;
    neuron_values[threadIdx.x + 4 * blockDim.x] = 
    (map[get_cell_index(creature_x[idx], creature_y[idx] + 1, map_width, map_height)] & (1 << BIT_FOOD)) ? 1.0f : 0.0f;
    neuron_values[threadIdx.x + 5 * blockDim.x] = 
    (map[get_cell_index(creature_x[idx], creature_y[idx] - 1, map_width, map_height)] & (1 << BIT_FOOD)) ? 1.0f : 0.0f;

    // Compute outputs
    float sum = 0.0f;
    for(int i = 0; i < 6; i++) {
        neuron_values[threadIdx.x + blockDim.x * (i + 6)] = 0.0f;
        for(int j = 0; j < 6; j++) {
            neuron_values[threadIdx.x + blockDim.x * (i + 6)] += 
                neuron_values[threadIdx.x + blockDim.x * j] * creature_matrix[idx + max_creature_n * (i * 6 + j)];
        }
        neuron_values[threadIdx.x + blockDim.x * (i + 6)] += creature_bias[idx + max_creature_n * i];
        sum += expf(neuron_values[threadIdx.x + blockDim.x * (i + 6)]); // For softmax
    }

    /*
    Actions:
    0: Eat
    1: Reproduce
    2: Move Right
    3: Move Left
    4: Move Down
    5: Move Up
    */

    // Softmax activation
    for(int i = 0; i < 6; i++) {
        neuron_values[threadIdx.x + blockDim.x * (i + 6)] = expf(neuron_values[threadIdx.x + blockDim.x * (i + 6)]) / sum;
    }

    float u = curand_uniform(&random_states[idx]);
    
    float cdf = 0.0f;

    int action = 5;

    for (int i = 0; i < 6; ++i) {
        cdf += neuron_values[threadIdx.x + blockDim.x * (i + 6)];
        
        if (u <= cdf) {
            action = i;
            break;
        }
    }

    int creature_index = atomicAdd(&action_counts[action], 1);
    creature_by_actions[action * max_creature_n + creature_index] = idx;
}

__global__ void move_right(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int max_creature_n,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[2]) return;
    int new_x = creature_x[creature_by_actions[2 * max_creature_n + idx]] + 1;
    if (new_x >= map_width) {
        new_x -= map_width;
    }
    creature_x[creature_by_actions[2 * max_creature_n + idx]] = new_x;
}

__global__ void move_left(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int max_creature_n,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[3]) return;
    int new_x = creature_x[creature_by_actions[3 * max_creature_n + idx]] - 1;
    if (new_x < 0) {
        new_x += map_width;
    }
    creature_x[creature_by_actions[3 * max_creature_n + idx]] = new_x;
}

__global__ void move_down(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int max_creature_n,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[4]) return;
    int new_y = creature_y[creature_by_actions[4 * max_creature_n + idx]] + 1;
    if (new_y >= map_height) {
        new_y -= map_height;
    }
    creature_y[creature_by_actions[4 * max_creature_n + idx]] = new_y;
}

__global__ void move_up(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int max_creature_n,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[5]) return;
    int new_y = creature_y[creature_by_actions[5 * max_creature_n + idx]] - 1;
    if (new_y < 0) {
        new_y += map_height;
    }
    creature_y[creature_by_actions[5 * max_creature_n + idx]] = new_y;
}


__global__ void remove_creatures_from_map(
    unsigned int* map,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    map[idx] &= ~(1 << BIT_CREATURE); // Clear creature bit
}

__global__ void place_creatures_on_map(
    unsigned int* map,
    unsigned int* creature_x,
    unsigned int* creature_y,
    int creature_n,
    int max_creature_n,
    int map_width,
    int map_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    int x = creature_x[idx];
    int y = creature_y[idx];

    atomicOr(&map[get_cell_index(x, y, map_width, map_height)], 1 << BIT_CREATURE);

    atomicOr(&map[get_cell_index(x + 1, y, map_width, map_height)], 1 << BIT_CREATURE);
    atomicOr(&map[get_cell_index(x - 1, y, map_width, map_height)], 1 << BIT_CREATURE);
    atomicOr(&map[get_cell_index(x, y + 1, map_width, map_height)], 1 << BIT_CREATURE);
    atomicOr(&map[get_cell_index(x, y - 1, map_width, map_height)], 1 << BIT_CREATURE);
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
    CUDA_CHECK(cudaMemset(d_map, 0, map_width * map_height * sizeof(unsigned int)));

    // Creatures
    int creatures_n = 1024;
    int max_creature_n = 1024 * 1024; // maximum number of creatures
    unsigned int* d_creature_x;
    unsigned int* d_creature_y;
    float* d_creature_energy;
    float* d_creature_matrix;
    float* d_creature_bias;
    int* d_creatures_by_action;
    int* d_action_counts;
    CUDA_CHECK(cudaMalloc(&d_creature_x, max_creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_y, max_creature_n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_energy, max_creature_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_matrix, max_creature_n * 6 * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_bias, max_creature_n * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creatures_by_action, max_creature_n * 6 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_action_counts, 6 * sizeof(int)));

    // we will use this on the host to reset action counts before each step
    int* h_action_counts = new int[6];



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


    place_food<<<(map_width * map_height + 255) / 256, 256>>>(
        d_map, 
        d_random_states, 
        map_width, 
        map_height
    );

    

    cudaDeviceSynchronize();

    printf("Initializing creatures...\n");
    initialize_creatures<<<(creatures_n + 255) / 256, 256>>>(
        d_creature_x, 
        d_creature_y, 
        d_creature_energy, 
        d_random_states, 
        creatures_n,
        max_creature_n, 
        map_width, 
        map_height, 
        d_map,
        d_creature_matrix,
        d_creature_bias
    );

    cudaDeviceSynchronize();

    //main loop
    bool running = true;
    while (running) {

        CUDA_CHECK(cudaMemset(d_action_counts, 0, 6 * sizeof(int)));
        size_t shared_memory_size = 256 * 6 * 2 * sizeof(float); // 256 threads, 6 inputs + 6 outputs per creature
        creature_action_step<<<(creatures_n + 255) / 256, 256, shared_memory_size>>>(
            d_creature_x,
            d_creature_y,
            d_creature_energy,
            creatures_n,
            max_creature_n,
            map_width,
            map_height,
            d_map,
            d_creature_matrix,
            d_creature_bias,
            d_creatures_by_action,
            d_action_counts,
            d_random_states
        );

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(h_action_counts, d_action_counts, 6 * sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        std::cout << std::endl;

        move_right<<<(h_action_counts[2] + 255) / 256, 256>>>(
            d_creature_x,
            d_creature_y,
            max_creature_n,
            map_width,
            map_height,
            d_creatures_by_action,
            d_action_counts
        );
        move_left<<<(h_action_counts[3] + 255) / 256, 256>>>(
            d_creature_x,
            d_creature_y,
            max_creature_n,
            map_width,
            map_height,
            d_creatures_by_action,
            d_action_counts
        );
        move_down<<<(h_action_counts[4] + 255) / 256, 256>>>(
            d_creature_x,
            d_creature_y,
            max_creature_n,
            map_width,
            map_height,
            d_creatures_by_action,
            d_action_counts
        );
        move_up<<<(h_action_counts[5] + 255) / 256, 256>>>(
            d_creature_x,
            d_creature_y,
            max_creature_n,
            map_width,
            map_height,
            d_creatures_by_action,
            d_action_counts
        );

        cudaDeviceSynchronize();

        // for display purposes
        remove_creatures_from_map<<<(map_width * map_height + 255) / 256, 256>>>(
            d_map,
            map_width,
            map_height
        );

        cudaDeviceSynchronize();

        place_creatures_on_map<<<(creatures_n + 255) / 256, 256>>>(
            d_map,
            d_creature_x,
            d_creature_y,
            creatures_n,
            max_creature_n,
            map_width,
            map_height
        );

        cudaDeviceSynchronize();

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
    CUDA_CHECK(cudaFree(d_creature_matrix));
    CUDA_CHECK(cudaFree(d_creature_bias));
    CUDA_CHECK(cudaFree(d_creatures_by_action));
    CUDA_CHECK(cudaFree(d_action_counts));
    return 0;
}