#include "actions/actions.cuh"



__global__ void move_right(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[2]) return;
    int new_x = creature_x[creature_by_actions[2 * MAX_CREATURE_N + idx]] + 1;
    if (new_x >= map_width) {
        new_x -= map_width;
    }
    creature_x[creature_by_actions[2 * MAX_CREATURE_N + idx]] = new_x;
}

__global__ void move_left(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[3]) return;
    int new_x = creature_x[creature_by_actions[3 * MAX_CREATURE_N + idx]] - 1;
    if (new_x < 0) {
        new_x += map_width;
    }
    creature_x[creature_by_actions[3 * MAX_CREATURE_N + idx]] = new_x;
}

__global__ void move_down(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[4]) return;
    int new_y = creature_y[creature_by_actions[4 * MAX_CREATURE_N + idx]] + 1;
    if (new_y >= map_height) {
        new_y -= map_height;
    }
    creature_y[creature_by_actions[4 * MAX_CREATURE_N + idx]] = new_y;
}

__global__ void move_up(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[5]) return;
    int new_y = creature_y[creature_by_actions[5 * MAX_CREATURE_N + idx]] - 1;
    if (new_y < 0) {
        new_y += map_height;
    }
    creature_y[creature_by_actions[5 * MAX_CREATURE_N + idx]] = new_y;
}

__global__ void eat_food(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int map_width,
    int map_height,
    unsigned int* map,
    int* creature_by_actions,
    int* action_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= action_counts[0]) return;

    int creature_idx = creature_by_actions[0 * MAX_CREATURE_N + idx];
    int x = creature_x[creature_idx];
    int y = creature_y[creature_idx];

    int cell_index = get_cell_index(x, y, map_width, map_height);

    if (map[cell_index] & (1 << BIT_FOOD)) {
    unsigned int food_bit = (1 << BIT_FOOD);
    unsigned int old_val = atomicAnd(&map[cell_index], ~food_bit);

    if (old_val & food_bit) {
        creature_energy[creature_idx] = 1.0f;
    }
}
}

__global__ void reproduce(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int* creature_n,
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
    if (idx >= action_counts[1]) return;

    int new_creature_idx = atomicAdd(creature_n, 1);

    if (new_creature_idx >= MAX_CREATURE_N) {
        return; // No more space for new creatures, skip reproduction
    }

    int parent_idx = creature_by_actions[1 * MAX_CREATURE_N + idx];

    creature_x[new_creature_idx] = creature_x[parent_idx];
    creature_y[new_creature_idx] = creature_y[parent_idx];

    creature_alive[new_creature_idx] = 1;

    float parent_energy = creature_energy[parent_idx];
    creature_energy[new_creature_idx] = parent_energy / 2.0f; // Split energy between parent and offspring
    creature_energy[parent_idx] = parent_energy / 2.0f;
    for(int i = 0; i < 6 * 6; i++) {
        creature_matrix[new_creature_idx + i * MAX_CREATURE_N] = 
        max(-MAX_PARAMETER_VALUE, min(MAX_PARAMETER_VALUE, creature_matrix[parent_idx + i * MAX_CREATURE_N] + curand_uniform(&random_states[new_creature_idx]) * 0.1f)); // Copy weights
    }
    for(int i = 0; i < 6; i++) {
        creature_bias[new_creature_idx + i * MAX_CREATURE_N] = 
        max(-MAX_PARAMETER_VALUE, min(MAX_PARAMETER_VALUE, creature_bias[parent_idx + i * MAX_CREATURE_N] + curand_uniform(&random_states[new_creature_idx]) * 0.1f)); // Copy biases
    }
    
}

