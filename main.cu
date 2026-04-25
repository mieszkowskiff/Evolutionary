#include <iostream>
#include <curand_kernel.h>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "utils/utils.cuh"
#include "actions/actions.cuh"
#include "constants.h"

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif


// # define ENERGY_LEVEL_SENSOR 0
// # define FOOD_PRESENCE_HERE_SENSOR 1
// # define FOOD_PRESENCE_RIGHT_SENSOR 2
// # define FOOD_PRESENCE_LEFT_SENSOR 3
// # define FOOD_PRESENCE_DOWN_SENSOR 4
// # define FOOD_PRESENCE_UP_SENSOR 5

// # define EAT_ACTION 0
// # define REPRODUCE_ACTION 1
// # define MOVE_RIGHT_ACTION 2
// # define MOVE_LEFT_ACTION 3
// # define MOVE_DOWN_ACTION 4
// # define MOVE_UP_ACTION 5


#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \


#define CHECK_CUDA_STATE(msg) \
do { \
    cudaDeviceSynchronize(); \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cout << "\n[!!!] KATASTROFA W: " << msg << "\nPowod: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } else { \
        std::cout << "[OK] Czysto po: " << msg << std::endl; \
    } \
} while(0)


volatile std::sig_atomic_t interrupted = 0;

extern "C" void signal_handler(int signum) {
    interrupted = 1;
}

__global__ void initialize_random_states(curandState* random_states, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    curand_init(1290, idx, 0, &random_states[idx]);
}





__global__ void place_food(
    unsigned int* map,
    curandState* random_states,
    int map_width,
    int map_height,
    float food_spawn_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)(map_width * map_height * food_spawn_rate)) return;

    int rand_x = (int)(curand_uniform(&random_states[idx]) * map_width);
    int rand_y = (int)(curand_uniform(&random_states[idx]) * map_height);

    rand_x = min(rand_x, map_width - 1);
    rand_y = min(rand_y, map_height - 1);

    //here we can add logic to rebalance food distribution


    map[get_cell_index(rand_x, rand_y, map_width, map_height)] |= (1u << BIT_FOOD); // Set food bit
}

__global__ void initialize_creatures(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    curandState* random_states,
    int creature_n,
    int map_width,
    int map_height,
    unsigned int* map,
    float* creature_matrix,
    float* creature_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    for(int i = 0; i < 6 * 6; i++) {
        creature_matrix[idx + i * MAX_CREATURE_N] = curand_uniform(&random_states[idx]) * 2 - 1; // random weights in [-1, 1]
    }
    for(int i = 0; i < 6; i++) {
        creature_bias[idx + i * MAX_CREATURE_N] = curand_uniform(&random_states[idx]) * 2 - 1; // random biases in [-1, 1]
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
    int map_width,
    int map_height,
    unsigned int* map,
    float* creature_matrix,
    float* creature_bias,
    int* creature_by_actions,
    int* action_counts,
    int* creature_alive,
    curandState* random_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;

    creature_energy[idx] -= COST_OF_LIVING; // Energy cost for living, can be adjusted
    creature_alive[idx] = creature_energy[idx] > 0.0f ? 1 : 0; // Mark creature as dead if energy is depleted

    if (!creature_alive[idx]) {
        // Creature dies, we can just skip it for now. It will be overwritten when new creatures are initialized.
        return;
    }

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
                neuron_values[threadIdx.x + blockDim.x * j] * creature_matrix[idx + MAX_CREATURE_N * (i * 6 + j)];
        }
        neuron_values[threadIdx.x + blockDim.x * (i + 6)] += creature_bias[idx + MAX_CREATURE_N * i];
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
    creature_by_actions[action * MAX_CREATURE_N + creature_index] = idx;
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
    int map_width,
    int map_height,
    float* creature_energy,
    int* creature_alive
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;
    if (!creature_alive[idx]) {
        return; // Skip dead creatures
    }

    int x = creature_x[idx];
    int y = creature_y[idx];

    atomicOr(&map[get_cell_index(x, y, map_width, map_height)], 1 << BIT_CREATURE);
}


__global__ void contract(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    float* creature_matrix,
    float* creature_bias,
    unsigned int* creature_x_save,
    unsigned int* creature_y_save,
    float* creature_energy_save,
    float* creature_matrix_save,
    float* creature_bias_save,
    int* contracted_creature_indices,
    int *creature_alive,
    int creature_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;
    if (!creature_alive[idx]) {
        return; // Skip dead creatures
    }
    int new_idx = contracted_creature_indices[idx];

    creature_x_save[new_idx] = creature_x[idx];
    creature_y_save[new_idx] = creature_y[idx];
    creature_energy_save[new_idx] = creature_energy[idx];
    for(int i = 0; i < 6 * 6; i++) {
        creature_matrix_save[new_idx + i * MAX_CREATURE_N] = creature_matrix[idx + i * MAX_CREATURE_N];
    }
    for(int i = 0; i < 6; i++) {
        creature_bias_save[new_idx + i * MAX_CREATURE_N] = creature_bias[idx + i * MAX_CREATURE_N];
    }
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
    int* h_creatures_n = new int(INITIAL_CREATURE_N);
    int* d_creatures_n;
    CUDA_CHECK(cudaMalloc(&d_creatures_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_creatures_n, h_creatures_n, sizeof(int), cudaMemcpyHostToDevice));

    // we define everything twice in order to play ping-pong
    // features of creatures
    unsigned int* d_creature_x_alpha;
    unsigned int* d_creature_x_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_x_alpha, MAX_CREATURE_N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_x_beta, MAX_CREATURE_N * sizeof(unsigned int)));
    unsigned int* d_creature_x = d_creature_x_alpha; // ping-pong pointer
    unsigned int* d_creature_x_save = d_creature_x_beta; // for contraction step


    unsigned int* d_creature_y_alpha;
    unsigned int* d_creature_y_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_y_alpha, MAX_CREATURE_N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_y_beta, MAX_CREATURE_N * sizeof(unsigned int)));
    unsigned int* d_creature_y = d_creature_y_alpha; // ping-pong pointer
    unsigned int* d_creature_y_save = d_creature_y_beta; // for contraction step

    float* d_creature_energy_alpha;
    float* d_creature_energy_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_energy_alpha, MAX_CREATURE_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_energy_beta, MAX_CREATURE_N * sizeof(float)));
    float* d_creature_energy = d_creature_energy_alpha; // ping-pong pointer
    float* d_creature_energy_save = d_creature_energy_beta; // for contraction step

    float* d_creature_matrix_alpha;
    float* d_creature_matrix_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_matrix_alpha, MAX_CREATURE_N * 6 * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_matrix_beta, MAX_CREATURE_N * 6 * 6 * sizeof(float)));
    float* d_creature_matrix = d_creature_matrix_alpha; // ping-pong pointer
    float* d_creature_matrix_save = d_creature_matrix_beta; // for contraction step


    float* d_creature_bias_alpha;
    float* d_creature_bias_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_bias_alpha, MAX_CREATURE_N * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_bias_beta, MAX_CREATURE_N * 6 * sizeof(float)));
    float* d_creature_bias = d_creature_bias_alpha; // ping-pong pointer
    float* d_creature_bias_save = d_creature_bias_beta; // for contraction step

    // Actions
    int* h_action_counts = new int[6];
    int* d_action_counts;
    int* d_creatures_by_action;
    CUDA_CHECK(cudaMalloc(&d_creatures_by_action, MAX_CREATURE_N * 6 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_action_counts, 6 * sizeof(int)));

    // For contraction step
    int* d_creature_alive;
    int* d_contracted_creature_indices;
    CUDA_CHECK(cudaMalloc(&d_creature_alive, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contracted_creature_indices, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_creature_alive, 1, *h_creatures_n * sizeof(int)));

    
    // Randomness source
    curandState* d_random_states;
    CUDA_CHECK(cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState)));

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(map_width, map_height);
    #endif

    printf("Initializing random states...\n");
    initialize_random_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(
        d_random_states, 
        MAX_CREATURE_N
    );
    
    cudaDeviceSynchronize();

    printf("Initializing map...\n");


    place_food<<<((int)(map_width * map_height * INITIAL_FOOD_SPAWN_RATE) + 255) / 256, 256>>>(
        d_map, 
        d_random_states, 
        map_width, 
        map_height,
        INITIAL_FOOD_SPAWN_RATE
    );

    

    cudaDeviceSynchronize();

    printf("Initializing creatures...\n");
    initialize_creatures<<<(*h_creatures_n + 255) / 256, 256>>>(
        d_creature_x, 
        d_creature_y, 
        d_creature_energy, 
        d_random_states, 
        *h_creatures_n,
        map_width, 
        map_height, 
        d_map,
        d_creature_matrix,
        d_creature_bias
    );

    cudaDeviceSynchronize();

    //main loop
    bool running = true;
    int max_creature_in_simulation = 0;
    int t = 1;
    while (t < 100000 && running) {

        CUDA_CHECK(cudaMemset(d_action_counts, 0, 6 * sizeof(int)));
        size_t shared_memory_size = 256 * 6 * 2 * sizeof(float); // 256 threads, 6 inputs + 6 outputs per creature
        creature_action_step<<<(*h_creatures_n + 255) / 256, 256, shared_memory_size>>>(
            d_creature_x,
            d_creature_y,
            d_creature_energy,
            *h_creatures_n,
            map_width,
            map_height,
            d_map,
            d_creature_matrix,
            d_creature_bias,
            d_creatures_by_action,
            d_action_counts,
            d_creature_alive,
            d_random_states
        );


        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(h_action_counts, d_action_counts, 6 * sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        // std::cout << "All creatures: " << *h_creatures_n << " Action counts: ";
        // for (int i = 0; i < 6; i++) {
        //     std::cout << h_action_counts[i] << " ";
        // }
        // std::cout << std::endl;

        if (h_action_counts[2] > 0) {
            move_right<<<(h_action_counts[2] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                map_width,
                map_height,
                d_creatures_by_action,
                d_action_counts
            );
        }

        if (h_action_counts[3] > 0) {
            move_left<<<(h_action_counts[3] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                map_width,
                map_height,
                d_creatures_by_action,
                d_action_counts
            );
        }

        if (h_action_counts[4] > 0) {
            move_down<<<(h_action_counts[4] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                map_width,
                map_height,
                d_creatures_by_action,
                d_action_counts
            );
        }

        if (h_action_counts[5] > 0) {
            move_up<<<(h_action_counts[5] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                map_width,
                map_height,
                d_creatures_by_action,
                d_action_counts
            );
        }

        if (h_action_counts[0] > 0) {
            eat_food<<<(h_action_counts[0] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                d_creature_energy,
                map_width,
                map_height,
                d_map,
                d_creatures_by_action,
                d_action_counts
            );
        }

        if (h_action_counts[1] > 0) {
            reproduce<<<(h_action_counts[1] + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                d_creature_energy,
                d_creatures_n,
                map_width,
                map_height,
                d_map,
                d_creature_matrix,
                d_creature_bias,
                d_creatures_by_action,
                d_action_counts,
                d_creature_alive,
                d_random_states
            );
        }


        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(h_creatures_n, d_creatures_n, sizeof(int), cudaMemcpyDeviceToHost));
        if (*h_creatures_n > max_creature_in_simulation) {
            max_creature_in_simulation = *h_creatures_n;
        }

        // for display purposes
        remove_creatures_from_map<<<(map_width * map_height + 255) / 256, 256>>>(
            d_map,
            map_width,
            map_height
        );

        cudaDeviceSynchronize();

        place_creatures_on_map<<<(*h_creatures_n + 255) / 256, 256>>>(
            d_map,
            d_creature_x,
            d_creature_y,
            *h_creatures_n,
            map_width,
            map_height,
            d_creature_energy,
            d_creature_alive
        );

        place_food<<<((int)(map_width * map_height * FOOD_SPAWN_RATE) + 255) / 256, 256>>>(
            d_map,
            d_random_states,
            map_width,
            map_height,
            FOOD_SPAWN_RATE
        );

        cudaDeviceSynchronize();

        if (*h_creatures_n == 0) {
            printf("All creatures died. Ending simulation.\n");
            break;
        }
        if (!(t % 5)) {
            
            thrust::device_ptr<int> dev_flags(d_creature_alive);
            thrust::device_ptr<int> dev_indices(d_contracted_creature_indices);
            thrust::exclusive_scan(thrust::device, dev_flags, dev_flags + *h_creatures_n, dev_indices);


            cudaDeviceSynchronize();

            contract<<<(*h_creatures_n + 255) / 256, 256>>>(
                d_creature_x,
                d_creature_y,
                d_creature_energy,
                d_creature_matrix,
                d_creature_bias,
                d_creature_x_save,
                d_creature_y_save,
                d_creature_energy_save,
                d_creature_matrix_save,
                d_creature_bias_save,
                d_contracted_creature_indices,
                d_creature_alive,
                *h_creatures_n
            );


            int* last_creature_alive = new int[1];

            CUDA_CHECK(cudaMemcpy(last_creature_alive, &d_creature_alive[*h_creatures_n - 1], sizeof(int), cudaMemcpyDeviceToHost));
            *h_creatures_n = dev_indices[*h_creatures_n - 1] + last_creature_alive[0];
            CUDA_CHECK(cudaMemcpy(d_creatures_n, h_creatures_n, sizeof(int), cudaMemcpyHostToDevice));

            std::swap(d_creature_x, d_creature_x_save);
            std::swap(d_creature_y, d_creature_y_save);
            std::swap(d_creature_energy, d_creature_energy_save);
            std::swap(d_creature_matrix, d_creature_matrix_save);
            std::swap(d_creature_bias, d_creature_bias_save);

            cudaDeviceSynchronize();
        }

        #ifdef ENABLE_DISPLAY
        display.renderFrame(d_map);

        if(display.shouldClose()){
            running = false;
        }

        cudaDeviceSynchronize();
        #endif

        if (interrupted) {
            running = false;
        }

        t++;
    }

    printf("\nQuitting the simulation. Shutting down...\n");
    fflush(stdout);

    CUDA_CHECK(cudaFree(d_map));

    CUDA_CHECK(cudaFree(d_creature_x_alpha));
    CUDA_CHECK(cudaFree(d_creature_x_beta));

    CUDA_CHECK(cudaFree(d_creature_y_alpha));
    CUDA_CHECK(cudaFree(d_creature_y_beta));


    CUDA_CHECK(cudaFree(d_creature_energy_alpha));
    CUDA_CHECK(cudaFree(d_creature_energy_beta));

    CUDA_CHECK(cudaFree(d_creature_matrix_alpha));
    CUDA_CHECK(cudaFree(d_creature_matrix_beta));

    CUDA_CHECK(cudaFree(d_creature_bias_alpha));
    CUDA_CHECK(cudaFree(d_creature_bias_beta));

    CUDA_CHECK(cudaFree(d_random_states));
    CUDA_CHECK(cudaFree(d_creatures_by_action));
    CUDA_CHECK(cudaFree(d_action_counts));
    CUDA_CHECK(cudaFree(d_creatures_n));
    CUDA_CHECK(cudaFree(d_creature_alive));
    CUDA_CHECK(cudaFree(d_contracted_creature_indices));

    printf("Maximum creatures in simulation: %d\n", max_creature_in_simulation);
    return 0;
}