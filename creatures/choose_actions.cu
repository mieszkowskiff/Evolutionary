#include "creatures/creatures.cuh"
#include <cuda_fp8.h>
#include "iostream"

__global__ void d_PerceveMap(MapData* d_map, CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int sensor_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (sensor_index >= MILIEU_SENSORS_N) return;

    int8_t sensor_x = d_creatures->sensor_x[get_sensor_idx(creature_index, sensor_index)];
    int8_t sensor_y = d_creatures->sensor_y[get_sensor_idx(creature_index, sensor_index)];
    int8_t sensor_type = d_creatures->sensor_type[get_sensor_idx(creature_index, sensor_index)];

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];

    int sensor_position = get_cell_index(creature_x + sensor_x, creature_y + sensor_y);

    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, sensor_index)] = __nv_fp8_e4m3(get_cell(d_map, sensor_type, sensor_position));
}

__global__ void d_PerceveSimulation(MapData* d_map, CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 0)] = __nv_fp8_e4m3(d_map->season_cos);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 1)] = __nv_fp8_e4m3(d_map->season_sin);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 2)] = __nv_fp8_e4m3(energy);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 3)] = __nv_fp8_e4m3(water);
}

__global__ void d_PopulateHiddenLayer(CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;
    
    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int hidden_neuron_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (hidden_neuron_index >= HIDDEN_NEURONS_N) return;

    float acummulation = static_cast<float>(d_creatures->bias[get_hidden_layer_bias_idx(creature_index, hidden_neuron_index)]);

    #pragma unroll
    for(int input_idx = 0; input_idx < INPUT_NEURONS_N; input_idx++) {
        size_t weight_idx = get_first_matrix_idx(creature_index, hidden_neuron_index, input_idx);
        acummulation += static_cast<float>(d_creatures->first_matrix[weight_idx]) * static_cast<float>(d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, input_idx)]);
    }

    d_creatures->hidden_layer_values[get_hidden_layer_value_idx(creature_index, hidden_neuron_index)] = __nv_fp8_e4m3(acummulation);
}

__global__ void d_PopulateOutputLayer(CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int output_neuron_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (output_neuron_index >= OUTPUT_NEURONS_N) return;

   float acummulation = 0.0f;

    #pragma unroll
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        size_t weight_idx = get_second_matrix_idx(creature_index, output_neuron_index, hidden_idx);
        acummulation += static_cast<float>(d_creatures->second_matrix[weight_idx]) * static_cast<float>(d_creatures->hidden_layer_values[get_hidden_layer_value_idx(creature_index, hidden_idx)]);
    }

    d_creatures->output_layer_values[get_output_layer_value_idx(creature_index, output_neuron_index)] = acummulation;
}

__global__ void d_ActionSelection(CreatureData* d_creatures, unsigned long long seed, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    unsigned long long local_seed = derive_seed(seed, creature_index);

    float max_output = -__FLT_MAX__;

    float neuron_values[OUTPUT_NEURONS_N];
    
    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        neuron_values[output_idx] = d_creatures->output_layer_values[get_output_layer_value_idx(creature_index, output_idx)];
        if (neuron_values[output_idx] > max_output) {
            max_output = neuron_values[output_idx];
        }
    }

    float exp_values[OUTPUT_NEURONS_N];
    float exp_sum = 0.0f;

    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        float exp_val = expf(neuron_values[output_idx] - max_output);
        exp_values[output_idx] = exp_val;
        exp_sum += exp_val;
    }

    float random_val = rand_float(derive_seed(local_seed, 4097)) * exp_sum;
    float cumulative_sum = 0.0f;
    int selected_action = -1;

    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        cumulative_sum += exp_values[output_idx];
        if (random_val <= cumulative_sum) {
            selected_action = output_idx;
            break;
        }
    }

    if (selected_action != -1) {
        unsigned int type = d_creatures->action_type[get_action_idx(creature_index, selected_action)];

        unsigned int queue_index = atomicAdd(&d_creatures->action_types_counts[type], 1);

        d_creatures->chosen_action[creature_index] = selected_action;

        switch (type) {
            case 0: // Move
                d_creatures->move_queue_creatures[queue_index] = creature_index;
                d_creatures->move_queue_actions[queue_index] = selected_action;
                break;
            case 1: // Eat
                d_creatures->eat_queue_creatures[queue_index] = creature_index;
                d_creatures->eat_queue_actions[queue_index] = selected_action;
                break;
            case 2: // Attack
                d_creatures->attack_queue_creatures[queue_index] = creature_index;
                d_creatures->attack_queue_actions[queue_index] = selected_action;
                break;
            case 3: // Reproduce
                d_creatures->reproduce_queue_creatures[queue_index] = creature_index;
                d_creatures->reproduce_queue_actions[queue_index] = selected_action;
                break;
            case 4: // Drink
                d_creatures->drink_queue_creatures[queue_index] = creature_index;
                d_creatures->drink_queue_actions[queue_index] = selected_action;
                break;
        }
    }
}

void Creatures::ChooseAction(Map* map, unsigned long long seed, float season_cos, float season_sin) {
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));

    MapData* h_map_data = map->d_data;
    CreatureData* h_creature_data = d_data;

    d_PerceveMap<<<dim3((count + 127) / 128, (MILIEU_SENSORS_N + 7) / 8), dim3(128, 8)>>>(map->d_data, d_data, count);
    d_PerceveSimulation<<<(count + 255) / 256, 256>>>(map->d_data, d_data, count);
    d_PopulateHiddenLayer<<<dim3((count + 127) / 128, (HIDDEN_NEURONS_N + 7) / 8), dim3(128, 8)>>>(d_data, count);
    d_PopulateOutputLayer<<<dim3((count + 127) / 128, (OUTPUT_NEURONS_N + 7) / 8), dim3(128, 8)>>>(d_data, count);
    d_ActionSelection<<<(count + 255) / 256, 256>>>(d_data, seed, count);

    cudaDeviceSynchronize();

    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //TODO: sort actions per type to make it more efficient (partial coalescing)
}