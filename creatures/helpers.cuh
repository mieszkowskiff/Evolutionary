#ifndef HELPERS_CUH
#define HELPERS_CUH


#include "constants.h"
#include "map/map.cuh"

__device__ inline float get_cell(MapData* map, int layer, int index) {
    switch (layer) {
        case 0: return map->food[index];
        case 1: return map->danger[index];
        case 2: return map->creature[index];
        case 3: return map->water[index];
        default: return 0.0f; 
    }
}

__device__ inline int get_sensor_idx(int creature_index, int sensor_index){
    return sensor_index * MAX_CREATURE_N + creature_index;
}

__device__ inline int get_input_layer_value_idx(int creature_index, int input_layer_value_index) {
    return input_layer_value_index * MAX_CREATURE_N + creature_index;
}

__device__ inline size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (hidden_idx * INPUT_NEURONS_N * MAX_CREATURE_N) + (sensor_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ inline int get_hidden_layer_value_idx(int creature_index, int hidden_layer_value_index) {
    return hidden_layer_value_index * MAX_CREATURE_N + creature_index;
}

__device__ inline int get_hidden_layer_bias_idx(int creature_index, int hidden_layer_bias_index) {
    return hidden_layer_bias_index * MAX_CREATURE_N + creature_index;
}

__device__ inline  size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (output_idx * HIDDEN_NEURONS_N * MAX_CREATURE_N) + (hidden_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ inline int get_output_layer_value_idx(int creature_index, int output_layer_value_index) {
    return output_layer_value_index * MAX_CREATURE_N + creature_index;
}

__device__ inline int get_action_idx(int creature_index, int action_index){
    return action_index * MAX_CREATURE_N + creature_index;
}

#endif