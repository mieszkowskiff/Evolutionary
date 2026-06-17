#include "creatures/creatures.cuh"
#include "creatures/helpers.cuh"
#include "iostream"

__global__ void d_CopySensors(CreatureData* d_creatures, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;

    int sensor_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (sensor_idx >= MILIEU_SENSORS_N) return;

    d_creatures->sensor_x[get_sensor_idx(new_creature_idx, sensor_idx)] = d_creatures->sensor_x[get_sensor_idx(parent_creature_index, sensor_idx)];
    d_creatures->sensor_y[get_sensor_idx(new_creature_idx, sensor_idx)] = d_creatures->sensor_y[get_sensor_idx(parent_creature_index, sensor_idx)];
    d_creatures->sensor_type[get_sensor_idx(new_creature_idx, sensor_idx)] = d_creatures->sensor_type[get_sensor_idx(parent_creature_index, sensor_idx)];
}

__global__ void d_CopyActions(CreatureData* d_creatures, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;

    int action_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (action_idx >= ACTIONS_N) return;

    d_creatures->action_x[get_action_idx(new_creature_idx, action_idx)] = d_creatures->action_x[get_action_idx(parent_creature_index, action_idx)];
    d_creatures->action_y[get_action_idx(new_creature_idx, action_idx)] = d_creatures->action_y[get_action_idx(parent_creature_index, action_idx)];
    d_creatures->action_type[get_action_idx(new_creature_idx, action_idx)] = d_creatures->action_type[get_action_idx(parent_creature_index, action_idx)];
}

__global__ void d_ReproduceFirstMatrix(CreatureData* d_creatures, unsigned long long seed, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;

    int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int input_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (hidden_idx >= HIDDEN_NEURONS_N || input_idx >= INPUT_NEURONS_N) return;

    size_t parent_parameter_idx = get_first_matrix_idx(parent_creature_index, hidden_idx, input_idx);
    size_t child_parameter_idx = get_first_matrix_idx(new_creature_idx, hidden_idx, input_idx);

    unsigned long long local_seed = derive_seed(seed, (int)parent_parameter_idx);

    d_creatures->first_matrix[child_parameter_idx] = __nv_fp8_e4m3(rand_normal(local_seed, (float)d_creatures->first_matrix[parent_parameter_idx], PARAMETER_MUTATION_STDDEV));
}

__global__ void d_ReproduceBias(CreatureData* d_creatures, unsigned long long seed, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;

    int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (hidden_idx >= HIDDEN_NEURONS_N) return;

    size_t parent_parameter_idx = get_hidden_layer_bias_idx(parent_creature_index, hidden_idx);
    size_t child_parameter_idx = get_hidden_layer_bias_idx(new_creature_idx, hidden_idx);

    unsigned long long local_seed = derive_seed(seed, (int)parent_parameter_idx);

    d_creatures->bias[child_parameter_idx] = __nv_fp8_e4m3(rand_normal(local_seed, (float)d_creatures->bias[parent_parameter_idx], PARAMETER_MUTATION_STDDEV));
}

__global__ void d_ReproduceSecondMatrix(CreatureData* d_creatures, unsigned long long seed, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;

    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (hidden_idx >= HIDDEN_NEURONS_N || output_idx >= OUTPUT_NEURONS_N) return;

    size_t parent_parameter_idx = get_second_matrix_idx(parent_creature_index, output_idx, hidden_idx);
    size_t child_parameter_idx = get_second_matrix_idx(new_creature_idx, output_idx, hidden_idx);

    unsigned long long local_seed = derive_seed(seed, (int)parent_parameter_idx);

    d_creatures->second_matrix[child_parameter_idx] = __nv_fp8_e4m3(rand_normal(local_seed, (float)d_creatures->second_matrix[parent_parameter_idx], PARAMETER_MUTATION_STDDEV));
}

__global__ void d_MutateSensors(CreatureData* d_creatures, unsigned long long seed, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;
    unsigned long long local_seed = derive_seed(seed, idx);

    int new_sensors_n = rand_int(derive_seed(local_seed, 548397), SENSORS_MUTATION_PACE);
    
    for (int sensor = 0; sensor < new_sensors_n; sensor++)
    {
        int sensor_idx = rand_int(derive_seed(local_seed, 54321 + sensor), MILIEU_SENSORS_N);
        SetRandomSensor(d_creatures, new_creature_idx, sensor_idx, derive_seed(local_seed, 98043 + sensor_idx));
    }
}

__global__ void d_MutateActions(CreatureData* d_creatures, unsigned long long seed, int reproduce_count, int old_creatures_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    if (parent_energy <= MIN_REPRODUCE_ENERGY) return;
    if (parent_water <= MIN_REPRODUCE_WATER) return;

    int new_creature_idx = old_creatures_count + idx;
    unsigned long long local_seed = derive_seed(seed, idx);

    int new_actions_n = rand_int(derive_seed(local_seed, 548397), ACTIONS_MUTATION_PACE);
    
    for (int action = 0; action < new_actions_n; action++)
    {
        int action_idx = rand_int(derive_seed(local_seed, 54321 + action), ACTIONS_N);
        SetRandomAction(d_creatures, new_creature_idx, action_idx, derive_seed(local_seed, 5783 + action_idx));
    }
}

__global__ void d_Reproduce(CreatureData* d_creatures, int reproduce_count, int old_creatures_count, long long global_id_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

    int new_creature_idx = old_creatures_count + idx;

    int parent_creature_index = d_creatures->reproduce_queue_creatures[idx];
    float parent_energy = d_creatures->energy[parent_creature_index];
    float parent_water = d_creatures->water[parent_creature_index];
    
    if (parent_energy <= MIN_REPRODUCE_ENERGY || parent_water <= MIN_REPRODUCE_WATER) {
        d_creatures->energy[new_creature_idx] = -1;
        return;
    }

    int8_t action_index = d_creatures->reproduce_queue_actions[idx];

    
    long long new_id = global_id_counter + idx;

    int8_t action_x = d_creatures->action_x[get_action_idx(parent_creature_index, action_index)];
    int8_t action_y = d_creatures->action_y[get_action_idx(parent_creature_index, action_index)];

    int creature_x = d_creatures->x[parent_creature_index];
    int creature_y = d_creatures->y[parent_creature_index];
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y + action_y;

    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    if (new_creature_idx >= MAX_CREATURE_N) {
        return;
    }

    d_creatures->x[new_creature_idx] = absolute_action_x;
    d_creatures->y[new_creature_idx] = absolute_action_y;

    float energy = parent_energy - REPRODUCE_COST;
    float water = parent_water - REPRODUCE_WATER_COST;

    d_creatures->energy[new_creature_idx] = energy * CHILD_ENERGY_SHARE;
    d_creatures->energy[parent_creature_index] = energy * (1.0f - CHILD_ENERGY_SHARE);
    d_creatures->water[new_creature_idx] = water * CHILD_WATER_SHARE;
    d_creatures->water[parent_creature_index] = water * (1.0f - CHILD_WATER_SHARE);
    d_creatures->ids[new_creature_idx] = new_id;
    d_creatures->age[new_creature_idx] = 0;
}


void Creatures::ReproduceAction(unsigned long long seed) {
    int reproduce_count = action_types_counts[REPRODUCE_ACTION];
    if (reproduce_count + count > MAX_CREATURE_N) {
        reproduce_count = MAX_CREATURE_N - count;
    }
    //std::cout << "Reproducing " << reproduce_count << " creatures. count: " << count << " MAX_CREATURE_N: " << MAX_CREATURE_N << std::endl;
    if (reproduce_count <= 0) return;

    d_CopySensors<<<dim3((reproduce_count + 127) / 128, (MILIEU_SENSORS_N + 7) / 8), dim3(128, 8)>>>(d_data, reproduce_count, count);
    d_ReproduceFirstMatrix<<<dim3((reproduce_count + 63) / 64, (HIDDEN_NEURONS_N + 3) / 4, (INPUT_NEURONS_N + 3) / 4), dim3(64, 4, 4)>>>(d_data, derive_seed(seed, 39482), reproduce_count, count);
    d_ReproduceBias<<<dim3((reproduce_count + 127) / 128, (HIDDEN_NEURONS_N + 7) / 8), dim3(128, 8)>>>(d_data, derive_seed(seed, 34892), reproduce_count, count);
    d_ReproduceSecondMatrix<<<dim3((reproduce_count + 63) / 64, (OUTPUT_NEURONS_N + 3) / 4, (HIDDEN_NEURONS_N + 3) / 4), dim3(64, 4, 4)>>>(d_data, derive_seed(seed, 57829), reproduce_count, count);
    d_CopyActions<<<dim3((reproduce_count + 127) / 128, (ACTIONS_N + 7) / 8), dim3(128, 8)>>>(d_data, reproduce_count, count);

    d_MutateSensors<<<(reproduce_count + 255) / 256, 256>>>(d_data, derive_seed(seed, 57887), reproduce_count, count);
    d_MutateActions<<<(reproduce_count + 255) / 256, 256>>>(d_data, derive_seed(seed, 57823), reproduce_count, count);

    cudaDeviceSynchronize();

    d_Reproduce<<<(reproduce_count + 255) / 256, 256>>>(d_data, reproduce_count, count, *global_id_counter);

    cudaDeviceSynchronize();
    
    count += reproduce_count;
    global_id_counter[0] += reproduce_count;
}
