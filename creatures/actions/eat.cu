#include "creatures/creatures.cuh"

__global__ void d_EatAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[EAT_ACTION]) return;
    int creature_index = d_creatures->eat_queue_creatures[idx];

    float energy = d_creatures->energy[creature_index];
    if (energy <= 0) return;
    if (d_creatures->water[creature_index] <= 0) return;

    int8_t action_index = d_creatures->eat_queue_actions[idx];

    int8_t action_x = d_creatures->action_x[action_index * MAX_CREATURE_N + creature_index];
    int8_t action_y = d_creatures->action_y[action_index * MAX_CREATURE_N + creature_index];

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y + action_y;

    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    float food_value = atomicExch(&d_map->food[get_cell_index(absolute_action_x, absolute_action_y)], 0.0f);

    d_creatures->energy[creature_index] = energy + food_value;
}