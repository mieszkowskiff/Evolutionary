#include "action_step/action_step.cuh"
#include "constants.h"
#include <curand_kernel.h>
#include "utils/utils.cuh"




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