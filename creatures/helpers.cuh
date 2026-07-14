#ifndef HELPERS_CUH
#define HELPERS_CUH

#include "constants.h"
#include "map/map.cuh"

// Helper functions for Tensor Core memory layout
// Memory is now grouped per creature to allow contiguous memory loads by a single warp


#if true
__host__ __device__ inline int get_sensor_idx(int creature_index, int sensor_index){
    return sensor_index * MAX_CREATURE_N + creature_index;
}

__host__ __device__ inline int get_input_layer_value_idx(int creature_index, int input_layer_value_index) {
    return input_layer_value_index * MAX_CREATURE_N + creature_index;
}

__host__ __device__ inline size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (hidden_idx * INPUT_NEURONS_N * MAX_CREATURE_N) + (sensor_idx * MAX_CREATURE_N) + creature_idx;
}

__host__ __device__ inline int get_hidden_layer_value_idx(int creature_index, int hidden_layer_value_index) {
    return hidden_layer_value_index * MAX_CREATURE_N + creature_index;
}

__host__ __device__ inline int get_hidden_layer_bias_idx(int creature_index, int hidden_layer_bias_index) {
    return hidden_layer_bias_index * MAX_CREATURE_N + creature_index;
}

__host__ __device__ inline  size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (output_idx * HIDDEN_NEURONS_N * MAX_CREATURE_N) + (hidden_idx * MAX_CREATURE_N) + creature_idx;
}

__host__ __device__ inline int get_output_layer_value_idx(int creature_index, int output_layer_value_index) {
    return output_layer_value_index * MAX_CREATURE_N + creature_index;
}

__host__ __device__ inline int get_action_idx(int creature_index, int action_index){
    return action_index * MAX_CREATURE_N + creature_index;
}
#else

__host__ __device__ inline int get_sensor_idx(int creature_index, int sensor_index){
    return (creature_index * MILIEU_SENSORS_N) + sensor_index;
}

__host__ __device__ inline int get_input_layer_value_idx(int creature_index, int input_layer_value_index) {
    return (creature_index * INPUT_NEURONS_N) + input_layer_value_index;
}

__host__ __device__ inline size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    // New layout: [creature_idx][hidden_idx][sensor_idx]
    return (creature_idx * HIDDEN_NEURONS_N * INPUT_NEURONS_N) + (hidden_idx * INPUT_NEURONS_N) + sensor_idx;
}

__host__ __device__ inline int get_hidden_layer_value_idx(int creature_index, int hidden_layer_value_index) {
    return (creature_index * HIDDEN_NEURONS_N) + hidden_layer_value_index;
}

__host__ __device__ inline int get_hidden_layer_bias_idx(int creature_index, int hidden_layer_bias_index) {
    return (creature_index * HIDDEN_NEURONS_N) + hidden_layer_bias_index;
}

__host__ __device__ inline size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    // New layout: [creature_idx][output_idx][hidden_idx]
    return (creature_idx * OUTPUT_NEURONS_N * HIDDEN_NEURONS_N) + (output_idx * HIDDEN_NEURONS_N) + hidden_idx;
}

__host__ __device__ inline int get_output_layer_value_idx(int creature_index, int output_layer_value_index) {
    return (creature_index * OUTPUT_NEURONS_N) + output_layer_value_index;
}

__host__ __device__ inline int get_action_idx(int creature_index, int action_index){
    return (creature_index * OUTPUT_NEURONS_N) + action_index;
}
#endif


#endif