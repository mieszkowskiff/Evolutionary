# ifndef ACTIONS_CUH
# define ACTIONS_CUH

# include "constants.h"
#include <curand_kernel.h>
#include "utils/utils.cuh"

__global__ void move_right(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
);
__global__ void move_left(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
);
__global__ void move_down(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
);
__global__ void move_up(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int map_width,
    int map_height,
    int* creature_by_actions,
    int* action_counts
);
__global__ void eat_food(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int map_width,
    int map_height,
    unsigned int* map,
    int* creature_by_actions,
    int* action_counts
);
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
);

# endif