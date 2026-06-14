#include "save/save.cuh"



void SaveManager::Save(const Creatures* creatures, int start_creature_index) {
    int new_creatures_count = creatures->count - start_creature_index;

    // parameters for the whole population (both old and new creatures)
    unsigned int* x;
    cudaMemcpy(x, creatures->h_data->x, sizeof(unsigned int) * creatures->count, cudaMemcpyDeviceToHost);

    unsigned int* y;
    cudaMemcpy(y, creatures->h_data->y, sizeof(unsigned int) * creatures->count, cudaMemcpyDeviceToHost);

    float* energy;
    cudaMemcpy(energy, creatures->h_data->energy, sizeof(float) * creatures->count, cudaMemcpyDeviceToHost);

    long long* ids;
    cudaMemcpy(ids, creatures->h_data->ids, sizeof(long long) * creatures->count, cudaMemcpyDeviceToHost);

    // actions, only for the old creatures, as the new ones haven't performed any actions yet
    int8_t* chosen_action;
    cudaMemcpy(chosen_action, creatures->h_data->chosen_action, sizeof(int8_t) * start_creature_index, cudaMemcpyDeviceToHost);

    for(int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
        __nv_fp8_e4m3* input_layer_values;
        cudaMemcpy(input_layer_values, creatures->h_data->input_layer_values + get_sensor_idx(start_creature_index, input_index), sizeof(__nv_fp8_e4m3) * start_creature_index, cudaMemcpyDeviceToHost);
    }

    for (int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        __nv_fp8_e4m3* hidden_layer_values;
        cudaMemcpy(hidden_layer_values, creatures->h_data->hidden_layer_values + get_hidden_layer_bias_idx(start_creature_index, hidden_index), sizeof(__nv_fp8_e4m3) * start_creature_index, cudaMemcpyDeviceToHost);
    }

    float* output_layer_values;
    for (int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
        cudaMemcpy(output_layer_values, creatures->h_data->output_layer_values + get_action_idx(start_creature_index, output_index), sizeof(float) * start_creature_index, cudaMemcpyDeviceToHost);
    }
    
    // parameters for the new creatures only

    // Sensors data
    for (int sensor_index = 0; sensor_index < MILIEU_SENSORS_N; sensor_index++) {
        int8_t* sensor_x;
        int8_t* sensor_y;
        int8_t* sensor_type;
        cudaMemcpy(sensor_x, creatures->h_data->sensor_x + get_sensor_idx(start_creature_index, sensor_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(sensor_y, creatures->h_data->sensor_y + get_sensor_idx(start_creature_index, sensor_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(sensor_type, creatures->h_data->sensor_type + get_sensor_idx(start_creature_index, sensor_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
    }

    // Network data
    for (int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        for (int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
            __nv_fp8_e4m3* first_matrix;
            cudaMemcpy(first_matrix, creatures->h_data->first_matrix + get_first_matrix_idx(start_creature_index, hidden_index, input_index), sizeof(__nv_fp8_e4m3) * new_creatures_count, cudaMemcpyDeviceToHost);
        }

        for (int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
            __nv_fp8_e4m3* second_matrix;
            cudaMemcpy(second_matrix, creatures->h_data->second_matrix + get_second_matrix_idx(start_creature_index, output_index, hidden_index), sizeof(__nv_fp8_e4m3) * new_creatures_count, cudaMemcpyDeviceToHost);
        }

        __nv_fp8_e4m3* bias;
        cudaMemcpy(bias, creatures->h_data->bias + get_hidden_layer_bias_idx(start_creature_index, hidden_index), sizeof(__nv_fp8_e4m3) * new_creatures_count, cudaMemcpyDeviceToHost);
    }

    // Action data
    for (int action_index = 0; action_index < OUTPUT_NEURONS_N; action_index++) {
        int8_t* action_x;
        int8_t* action_y;
        int8_t* action_type;
        cudaMemcpy(action_x, creatures->h_data->action_x + get_action_idx(start_creature_index, action_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(action_y, creatures->h_data->action_y + get_action_idx(start_creature_index, action_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(action_type, creatures->h_data->action_type + get_action_idx(start_creature_index, action_index), sizeof(int8_t) * new_creatures_count, cudaMemcpyDeviceToHost);
    }
}   


void SaveManager::SaveMap(const Map* map) {
    float* food;
    float* water;

    cudaMemcpy(food, map->h_data->food, sizeof(float) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaMemcpy(water, map->h_data->water, sizeof(float) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
}