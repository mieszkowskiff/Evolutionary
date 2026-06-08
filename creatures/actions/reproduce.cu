#include "creatures/creatures.cuh"

__global__ void d_ReproduceAction(MapData* d_map, CreatureData* d_creatures, unsigned long long seed, unsigned int* d_successful_births, long long global_id_counter, int count, int max_children, int reproduce_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    unsigned long long local_seed = derive_seed(seed, idx);

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    int8_t action_index = d_creatures->reproduce_queue_actions[idx];

    if (d_creatures->energy[parent_creature_index] <= MIN_REPRODUCE_ENERGY) return;
    if (d_creatures->water[parent_creature_index] <= MIN_REPRODUCE_WATER) return;

    unsigned int child_offset = atomicAdd(d_successful_births, 1);
    if (child_offset >= max_children) return;

    int new_creature_idx = count + child_offset;
    long long new_id = global_id_counter + child_offset;

    // int new_creature_idx = count + idx;
    // long long new_id = global_id_counter + idx;

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

    if (new_creature_idx >= MAX_CREATURE_N) {
        return;
    }

    d_creatures->x[new_creature_idx] = absolute_action_x;
    d_creatures->y[new_creature_idx] = absolute_action_y;
    
    reproduce_creature(d_creatures, parent_creature_index, new_creature_idx, local_seed, new_id);
}


__device__ void reproduce_creature(CreatureData* d_creatures, int parent_creature_index, int new_creature_idx, unsigned long long local_seed, long long new_id) {

    float energy = d_creatures->energy[parent_creature_index] - REPRODUCE_COST;
    float water = d_creatures->water[parent_creature_index] - REPRODUCE_WATER_COST;

    d_creatures->energy[new_creature_idx] = energy * CHILD_ENERGY_SHARE;
    d_creatures->energy[parent_creature_index] = energy * (1.0f - CHILD_ENERGY_SHARE);
    d_creatures->water[new_creature_idx] = water * CHILD_WATER_SHARE;
    d_creatures->water[parent_creature_index] = water * (1.0f - CHILD_WATER_SHARE);
    d_creatures->ids[new_creature_idx] = new_id;
    d_creatures->age[new_creature_idx] = 0;

    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < TOTAL_SENSORS_N; sensor_idx++) {
            size_t parent_idx = get_first_matrix_idx(parent_creature_index, hidden_idx, sensor_idx);
            size_t new_idx = get_first_matrix_idx(new_creature_idx, hidden_idx, sensor_idx);

            //d_creatures->first_matrix[new_idx] = __nv_fp8_e4m3((float)d_creatures->first_matrix[parent_idx] + curand_normal(&state) * PARAMETER_MUTATION_STDDEV);
            d_creatures->first_matrix[new_idx] = __nv_fp8_e4m3(rand_normal(derive_seed(local_seed, hidden_idx * TOTAL_SENSORS_N + sensor_idx), (float)d_creatures->first_matrix[parent_idx], PARAMETER_MUTATION_STDDEV));

        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < ACTIONS_N; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t parent_idx = get_second_matrix_idx(parent_creature_index, output_idx, hidden_idx);
            size_t new_idx = get_second_matrix_idx(new_creature_idx, output_idx, hidden_idx);

            //d_creatures->second_matrix[new_idx] = __nv_fp8_e4m3((float)d_creatures->second_matrix[parent_idx] + curand_normal(&state) * PARAMETER_MUTATION_STDDEV);
            d_creatures->second_matrix[new_idx] = __nv_fp8_e4m3(rand_normal(derive_seed(local_seed, HIDDEN_N * TOTAL_SENSORS_N + output_idx * HIDDEN_N + hidden_idx), (float)d_creatures->second_matrix[parent_idx], PARAMETER_MUTATION_STDDEV));
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            size_t parent_idx = parent_creature_index * HIDDEN_N + hidden_idx;
            size_t new_idx = new_creature_idx * HIDDEN_N + hidden_idx;

            d_creatures->bias[new_idx] = __nv_fp8_e4m3((float)d_creatures->bias[parent_idx] + rand_normal(derive_seed(local_seed, HIDDEN_N * TOTAL_SENSORS_N + hidden_idx), 0.0f, PARAMETER_MUTATION_STDDEV));
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
    int new_sensors_n = rand_int(derive_seed(local_seed, 98765), SENSORS_MUTATION_PACE);
    for (int sensor = 0; sensor < new_sensors_n; sensor++)
    {
        int sensor_idx = rand_int(derive_seed(local_seed, 54321 + sensor), SENSORS_N);
        AddRandomSensors(d_creatures, new_creature_idx, sensor_idx, derive_seed(local_seed, 98043 + sensor_idx));
    }

    //Mutate actions
    int new_actions_n = rand_int(derive_seed(local_seed, 98765), ACTIONS_MUTATION_PACE);
    for (int action = 0; action < new_actions_n; action++)
    {
        int action_idx = rand_int(derive_seed(local_seed, 54321 + action), ACTIONS_N);
        SetRandomAction(d_creatures, new_creature_idx, action_idx, derive_seed(local_seed, 98043 + action_idx));
    }
}
