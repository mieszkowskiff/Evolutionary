#include "creatures/creatures.cuh"


void Creatures::ChooseAction(Map* map, unsigned long long seed, float season_cos, float season_sin) {
    cudaDeviceSynchronize();
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));
    if (count > 0) d_ActionStep<<<(count + 255) / 256, 256>>>(map->d_data, d_data, seed, count, season_cos, season_sin);
    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //TODO: sort actions per type to make it more efficient (partial coalescing)
}


__global__ void d_ActionStep(MapData* d_map, CreatureData* d_creatures, unsigned long long seed, int count, float season_cos, float season_sin) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;
    if (d_creatures->energy[creature_index] <= 0.0f) return;
    if (d_creatures->water[creature_index] <= 0.0f) return;

    unsigned long long local_seed = derive_seed(seed, creature_index);

    __nv_fp8_e4m3 input_neurons[TOTAL_SENSORS_N];

    #pragma unroll
    for(int sensor_index = 0; sensor_index < SENSORS_N; sensor_index++) {
        int8_t sensor_x = d_creatures->sensor_x[sensor_index * MAX_CREATURE_N + creature_index];
        int8_t sensor_y = d_creatures->sensor_y[sensor_index * MAX_CREATURE_N + creature_index];
        int8_t sensor_type = d_creatures->sensor_type[sensor_index * MAX_CREATURE_N + creature_index];

        unsigned int creature_x = d_creatures->x[creature_index];
        unsigned int creature_y = d_creatures->y[creature_index];

        int sensor_position = get_cell_index(creature_x + sensor_x, creature_y + sensor_y);

        input_neurons[sensor_index] = __nv_fp8_e4m3(get_cell(d_map, sensor_type, sensor_position));
    }

    input_neurons[SENSORS_N] = __nv_fp8_e4m3(season_cos);
    input_neurons[SENSORS_N + 1] = __nv_fp8_e4m3(season_sin);

    __nv_fp8_e4m3 hidden_neurons[HIDDEN_N];

    #pragma unroll
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        float sum = 0.0f;

        #pragma unroll
        for(int sensor_idx = 0; sensor_idx < TOTAL_SENSORS_N; sensor_idx++) {
            size_t weight_idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            float weight = (float)d_creatures->first_matrix[weight_idx];
            float input_val = (float)input_neurons[sensor_idx];

            sum += weight * input_val;
        }
        size_t bias_idx = creature_index * HIDDEN_N + hidden_idx;
        sum += (float)d_creatures->bias[bias_idx];

        hidden_neurons[hidden_idx] = __nv_fp8_e4m3(sum > 0.0f ? sum : 0.0f);
    }

    __nv_fp8_e4m3 output_neurons[ACTIONS_N];
    float exp_values[ACTIONS_N];
    float total_exp_sum = 0.0f;

    #pragma unroll
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        float sum = 0.0f;
        #pragma unroll
        for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t weight_idx = get_second_matrix_idx(creature_index, action_idx, hidden_idx);
            float weight = (float)d_creatures->second_matrix[weight_idx];
            float hidden_val = (float)hidden_neurons[hidden_idx];

            sum += weight * hidden_val;
        }
        
        if (d_creatures->action_type[action_idx * MAX_CREATURE_N + creature_index] < ACTION_TYPES_N) { 
            float exp_val = expf(sum); 
            exp_values[action_idx] = exp_val;
            
            total_exp_sum += exp_val; 
        } else {
            exp_values[action_idx] = 0.0f; 
        }
    }


    #pragma unroll
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        if (d_creatures->action_type[action_idx * MAX_CREATURE_N + creature_index] < ACTION_TYPES_N) {
            float probability = exp_values[action_idx] / total_exp_sum;
            output_neurons[action_idx] = __nv_fp8_e4m3(probability);
        } else {
            output_neurons[action_idx] = __nv_fp8_e4m3(0.0f);
        }
    }

    //float random_val = curand_uniform(&random_states[creature_index]);
    float random_val = rand_float(derive_seed(local_seed, 4097));
    
    int selected_action = -1;
    float cumulative_probability = 0.0f;

    #pragma unroll
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        if (d_creatures->action_type[action_idx * MAX_CREATURE_N + creature_index] < ACTION_TYPES_N) {
            float action_prob = (float)output_neurons[action_idx];
            cumulative_probability += action_prob;

            if (random_val <= cumulative_probability && selected_action == -1) {
                selected_action = action_idx;
            }
        }
    }

    if (selected_action != -1) {
        unsigned int type = d_creatures->action_type[selected_action * MAX_CREATURE_N + creature_index];

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