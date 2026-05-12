#include "creatures/creatures.cuh"
#include "constants.h"
#include <stdio.h>


Creatures::Creatures(curandState* state, int count) {
    this->count = count;

    h_data = new CreatureData;

    cudaMalloc(&h_data->x, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->y, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->energy, MAX_CREATURE_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&h_data->sensor_x, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_y, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_type, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));

    cudaMalloc(&h_data->first_matrix, MAX_CREATURE_N * SENSORS_N * HIDDEN_N * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->second_matrix, MAX_CREATURE_N * HIDDEN_N * ACTIONS_N * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->bias, MAX_CREATURE_N * HIDDEN_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&h_data->action_x, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_y, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_type, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));


    cudaMalloc(&h_data->move_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->eat_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->attack_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->reproduce_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));

    cudaMalloc(&h_data->move_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->eat_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->attack_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->reproduce_queue_actions, MAX_CREATURE_N * sizeof(int8_t));

    cudaMalloc(&h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int));

    cudaMalloc(&d_data, sizeof(CreatureData));
    cudaMemcpy(d_data, h_data, sizeof(CreatureData), cudaMemcpyHostToDevice);

    action_types_counts = new unsigned int[ACTION_TYPES_N];

    cudaDeviceSynchronize();
    InitializeRandomCreatures<<<(count + 255) / 256, 256>>>(d_data, count, state);
    cudaDeviceSynchronize();
}

Creatures::~Creatures() {
    cudaFree(h_data->x);
    cudaFree(h_data->y);
    cudaFree(h_data->energy);

    cudaFree(h_data->sensor_x);
    cudaFree(h_data->sensor_y);
    cudaFree(h_data->sensor_type);

    cudaFree(h_data->first_matrix);
    cudaFree(h_data->second_matrix);
    cudaFree(h_data->bias);

    cudaFree(h_data->action_x);
    cudaFree(h_data->action_y);
    cudaFree(h_data->action_type);

    cudaFree(h_data->move_queue_creatures);
    cudaFree(h_data->eat_queue_creatures);
    cudaFree(h_data->attack_queue_creatures);
    cudaFree(h_data->reproduce_queue_creatures);

    cudaFree(h_data->move_queue_actions);
    cudaFree(h_data->eat_queue_actions);
    cudaFree(h_data->attack_queue_actions);
    cudaFree(h_data->reproduce_queue_actions);

    cudaFree(h_data->action_types_counts);

    cudaFree(h_data);
    delete h_data;
}

void Creatures::ActionStep(Map* map, curandState* random_states) {
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));

    d_ActionStep<<<(count + 255) / 256, 256>>>(map->d_data, d_data, random_states);
    cudaDeviceSynchronize();

    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

__global__ void d_ActionStep(MapData* d_map, CreatureData* d_creatures, curandState* random_states) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((float)d_creatures->energy[creature_index] <= 0.0f) return;

    __nv_fp8_e4m3 input_neurons[SENSORS_N];

    #pragma unroll
    for(int sensor_index = 0; sensor_index < SENSORS_N; sensor_index++) {
        int8_t sensor_x = d_creatures->sensor_x[sensor_index * MAX_CREATURE_N + creature_index];
        int8_t sensor_y = d_creatures->sensor_y[sensor_index * MAX_CREATURE_N + creature_index];
        int8_t sensor_type = d_creatures->sensor_type[sensor_index * MAX_CREATURE_N + creature_index];

        unsigned int creature_x = d_creatures->x[creature_index];
        unsigned int creature_y = d_creatures->y[creature_index];

        int sensor_position = get_cell_index(creature_x + sensor_x, creature_y + sensor_y);

        input_neurons[sensor_index] = get_cell(d_map, sensor_type, sensor_position);
    }

    __nv_fp8_e4m3 hidden_neurons[HIDDEN_N];

    #pragma unroll
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        float sum = 0.0f;

        #pragma unroll
        for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
            size_t weight_idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            float weight = (float)d_creatures->first_matrix[weight_idx];
            float input_val = (float)input_neurons[sensor_idx];

            sum += weight * input_val;
        }
        size_t bias_idx = (hidden_idx * MAX_CREATURE_N) + creature_index;
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

    float random_val = curand_uniform(&random_states[creature_index]);
    
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
        }
    }
}

__global__ void InitializeRandomCreatures(CreatureData* creatures, int count, curandState* states) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    curandState state = states[creature_index];

    //Initialize position and energy
    creatures->x[creature_index] = curand(&state) % WIDTH;
    creatures->y[creature_index] = curand(&state) % HEIGHT;
    creatures->energy[creature_index] = __nv_fp8_e4m3(1.0f);

    // Initialize sensors
    for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
        AddRandomSensors(creatures, creature_index, sensor_idx, state);
    }

    // Initialize network
    AddRandomNetwork(creatures, creature_index, state);

    // Initialize actions
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        SetRandomAction(creatures, creature_index, action_idx, state);
    }
}

__device__ void AddRandomSensors(CreatureData* creatures, int creature_index, int sensor_index, curandState& state) {
        float x_normal = curand_normal(&state) * SENSOR_STDDEV;
        float y_normal = curand_normal(&state) * SENSOR_STDDEV;
        
        int8_t x = static_cast<int8_t>(roundf(x_normal));
        int8_t y = static_cast<int8_t>(roundf(y_normal));
        int8_t type = curand(&state) % 10; // 0: food, 1: danger, 2: creature, 3-9: empty

        creatures->sensor_x[sensor_index * MAX_CREATURE_N + creature_index] = x;
        creatures->sensor_y[sensor_index * MAX_CREATURE_N + creature_index] = y;
        creatures->sensor_type[sensor_index * MAX_CREATURE_N + creature_index] = type;
}

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, curandState &state) {
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
            size_t idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            creatures->first_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < ACTIONS_N; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t idx = get_second_matrix_idx(creature_index, output_idx, hidden_idx);
            creatures->second_matrix[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        size_t idx = (creature_index * HIDDEN_N) + hidden_idx;
        creatures->bias[idx] = __nv_fp8_e4m3(curand_uniform(&state) * 2 - 1); // Random value between -1 and 1
    }
}

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx) {
    return (hidden_idx * SENSORS_N * MAX_CREATURE_N) + (sensor_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (output_idx * HIDDEN_N * MAX_CREATURE_N) + (hidden_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, curandState& state) {
    float x_normal = curand_normal(&state) * ACTION_STDDEV;
    float y_normal = curand_normal(&state) * ACTION_STDDEV;
    
    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = curand(&state) % 10; // 0: move, 1: eat, 2: attack, 3: reproduce, 4-9 no action (placeholder)

    creatures->action_x[action_index * MAX_CREATURE_N + creature_index] = x;
    creatures->action_y[action_index * MAX_CREATURE_N + creature_index] = y;
    creatures->action_type[action_index * MAX_CREATURE_N + creature_index] = type;
}

