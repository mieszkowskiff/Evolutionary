#include "creatures/creatures.cuh"
#include "constants.h"
#include <stdio.h>
#include <thrust/device_ptr.h>


Creatures::Creatures(curandState* state, int count) {

    h_data = new CreatureData;
    h_data->count = count;

    cudaMalloc(&h_data->x, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->y, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->energy, MAX_CREATURE_N * sizeof(float));

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

void Creatures::ChooseAction(Map* map, curandState* random_states) {
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));
    if (h_data->count > 0) d_ActionStep<<<(h_data->count + 255) / 256, 256>>>(map->d_data, d_data, random_states);
    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //TODO: sort actions per type to make it more efficient (partial coalescing)
}

void Creatures::RunActions(Map* map, curandState* random_states) {
    if (action_types_counts[ATTACK_ACTION] > 0) d_AttackAction<<<(action_types_counts[ATTACK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (h_data->count > 0) d_ProcessEnergy<<<(h_data->count + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[MOVE_ACTION] > 0) d_MoveAction<<<(action_types_counts[MOVE_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[EAT_ACTION] > 0) d_EatAction<<<(action_types_counts[EAT_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[REPRODUCE_ACTION]) d_ReproduceAction<<<(action_types_counts[REPRODUCE_ACTION] + 255) / 256, 256>>>(map->d_data, d_data, random_states);

    cudaDeviceSynchronize();
    cudaMemcpy(&h_data->count, &d_data->count, sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void d_MoveAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[MOVE_ACTION]) return;
    int creature_index = d_creatures->move_queue_creatures[idx];
    if (d_creatures->energy[creature_index] <= 0) return;

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];

    int8_t action_index = d_creatures->move_queue_actions[idx];

    int8_t action_x = d_creatures->action_x[action_index * MAX_CREATURE_N + creature_index];
    int8_t action_y = d_creatures->action_y[action_index * MAX_CREATURE_N + creature_index];


    d_map->creature[get_cell_index(creature_x, creature_y)] = __float2half(0.0f);
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y+ action_y;


    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    d_creatures->x[creature_index] = absolute_action_x;
    d_creatures->y[creature_index] = absolute_action_y;

    __syncthreads();

    atomicAdd(&d_map->creature[get_cell_index(absolute_action_x, absolute_action_y)], 1.0f); // maybe we should add energy instead of increment
}

__global__ void d_EatAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[EAT_ACTION]) return;
    int creature_index = d_creatures->move_queue_creatures[idx];

    float energy = d_creatures->energy[creature_index];
    if (energy <= 0) return;

    int8_t action_index = d_creatures->move_queue_actions[idx];

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

__global__ void d_AttackAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[ATTACK_ACTION]) return;
    int creature_index = d_creatures->move_queue_creatures[idx];
    int8_t action_index = d_creatures->move_queue_actions[idx];

    int8_t action_x = d_creatures->action_x[action_index * MAX_CREATURE_N + creature_index];
    int8_t action_y = d_creatures->action_y[action_index * MAX_CREATURE_N + creature_index];

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y+ action_y;

    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    atomicAdd(&d_map->danger[get_cell_index(absolute_action_x, absolute_action_y)], 1.0f);
}

__global__ void d_ProcessEnergy(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->count) return;

    float energy = d_creatures->energy[idx];
    if (energy <= 0) return;

    int creature_x = d_creatures->x[idx];
    int creature_y = d_creatures->y[idx];

    float damage = d_map->danger[get_cell_index(creature_x, creature_y)];
    energy -= ENERGY_DECAY;

    if (damage > 0) {
        energy -= damage;
        atomicAdd(&d_map->food[get_cell_index(creature_x, creature_y)], damage);
    }

    d_creatures->energy[idx] = energy;

    
}

__global__ void d_ReproduceAction(MapData* d_map, CreatureData* d_creatures, curandState* random_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[REPRODUCE_ACTION]) return;

    int parent_creature_index = d_creatures->move_queue_creatures[idx];
    int8_t action_index = d_creatures->move_queue_actions[idx];

    if (d_creatures->energy[parent_creature_index] <= 0.0f) return;

    int8_t action_x = d_creatures->action_x[action_index * MAX_CREATURE_N + parent_creature_index];
    int8_t action_y = d_creatures->action_y[action_index * MAX_CREATURE_N + parent_creature_index];

    int creature_x = d_creatures->x[parent_creature_index];
    int creature_y = d_creatures->y[parent_creature_index];
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y+ action_y;

    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    int new_creature_idx = atomicAdd(&d_creatures->count, 1);

    if (new_creature_idx >= MAX_CREATURE_N) {
        return;
    }

    d_creatures->x[new_creature_idx] = absolute_action_x;
    d_creatures->y[new_creature_idx] = absolute_action_y;
    
    reproduce_creature(d_creatures, parent_creature_index, new_creature_idx, random_states[parent_creature_index]);
}

__device__ void reproduce_creature(CreatureData* d_creatures, int parent_creature_index, int new_creature_idx, curandState& state) {

    d_creatures->energy[new_creature_idx] = d_creatures->energy[parent_creature_index] * 0.5f;
    d_creatures->energy[parent_creature_index] = d_creatures->energy[parent_creature_index] * 0.5f;

    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
            size_t parent_idx = get_first_matrix_idx(parent_creature_index, hidden_idx, sensor_idx);
            size_t new_idx = get_first_matrix_idx(new_creature_idx, hidden_idx, sensor_idx);

            d_creatures->first_matrix[new_idx] = __nv_fp8_e4m3((float)d_creatures->first_matrix[parent_idx] + curand_normal(&state) * PARAMETER_MUTATION_STDDEV);
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < ACTIONS_N; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t parent_idx = get_second_matrix_idx(parent_creature_index, output_idx, hidden_idx);
            size_t new_idx = get_second_matrix_idx(new_creature_idx, output_idx, hidden_idx);

            d_creatures->second_matrix[new_idx] = __nv_fp8_e4m3((float)d_creatures->second_matrix[parent_idx] + curand_normal(&state) * PARAMETER_MUTATION_STDDEV);
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t parent_idx = parent_creature_index * HIDDEN_N + hidden_idx;
            size_t new_idx = new_creature_idx * HIDDEN_N + hidden_idx;

            d_creatures->bias[new_idx] = __nv_fp8_e4m3((float)d_creatures->bias[parent_idx] + curand_normal(&state) * PARAMETER_MUTATION_STDDEV);
    }

    // Actions
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        d_creatures->action_x[action_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->action_x[action_idx * MAX_CREATURE_N + parent_creature_index];
        d_creatures->action_y[action_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->action_y[action_idx * MAX_CREATURE_N + parent_creature_index];
        d_creatures->action_type[action_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->action_type[action_idx * MAX_CREATURE_N + parent_creature_index];
    }

    // Sensors
    for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
        d_creatures->sensor_x[sensor_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->sensor_x[sensor_idx * MAX_CREATURE_N + parent_creature_index];
        d_creatures->sensor_y[sensor_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->sensor_y[sensor_idx * MAX_CREATURE_N + parent_creature_index];
        d_creatures->sensor_type[sensor_idx * MAX_CREATURE_N + new_creature_idx] = d_creatures->sensor_type[sensor_idx * MAX_CREATURE_N + parent_creature_index];
    }

    // Mutate sensors
    int new_sensors_n = curand(&state) % SENSORS_MUTATION_PACE;
    for (int sensor = 0; sensor < new_sensors_n; sensor++)
    {
        int sensor_idx = curand(&state) % SENSORS_N;
        AddRandomSensors(d_creatures, new_creature_idx, sensor_idx, state);
    }

    //Mutate actions
    int new_actions_n = curand(&state) % ACTIONS_MUTATION_PACE;
    for (int action = 0; action < new_actions_n; action++)
    {
        int action_idx = curand(&state) % ACTIONS_N;
        SetRandomAction(d_creatures, new_creature_idx, action_idx, state);
    }
}


__global__ void d_ActionStep(MapData* d_map, CreatureData* d_creatures, curandState* random_states) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_creatures->energy[creature_index] <= 0.0f) return;

    __nv_fp8_e4m3 input_neurons[SENSORS_N];

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
    creatures->energy[creature_index] = 1.0f;

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

