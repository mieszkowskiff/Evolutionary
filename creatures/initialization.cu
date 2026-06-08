#include "creatures/creatures.cuh"



Creatures::Creatures(unsigned long long seed, int count, long long *global_id_counter) {

    h_data = new CreatureData;
    this->count = count;
    this->global_id_counter = global_id_counter;

    cudaMalloc(&d_successful_births, sizeof(unsigned int));
    cudaMalloc(&d_attack_damage_kills, sizeof(unsigned int));
    h_attack_damage_kills = 0;

    cudaMalloc(&h_data->x, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->y, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->energy, MAX_CREATURE_N * sizeof(float));
    cudaMalloc(&h_data->water, MAX_CREATURE_N * sizeof(float));
    cudaMalloc(&h_data->ids, MAX_CREATURE_N * sizeof(long long));
    cudaMalloc(&h_data->age, MAX_CREATURE_N * sizeof(int));

    cudaMalloc(&h_data->sensor_x, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_y, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMalloc(&h_data->sensor_type, MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));

    cudaMalloc(&h_data->first_matrix, MAX_CREATURE_N * TOTAL_SENSORS_N * HIDDEN_N * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->second_matrix, MAX_CREATURE_N * HIDDEN_N * ACTIONS_N * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&h_data->bias, MAX_CREATURE_N * HIDDEN_N * sizeof(__nv_fp8_e4m3));

    cudaMalloc(&h_data->action_x, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_y, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMalloc(&h_data->action_type, MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));

    cudaMalloc(&h_data->chosen_action, MAX_CREATURE_N * sizeof(int8_t));

    cudaMalloc(&h_data->move_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->eat_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->attack_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->reproduce_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));
    cudaMalloc(&h_data->drink_queue_creatures, MAX_CREATURE_N * sizeof(unsigned int));

    cudaMalloc(&h_data->move_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->eat_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->attack_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->reproduce_queue_actions, MAX_CREATURE_N * sizeof(int8_t));
    cudaMalloc(&h_data->drink_queue_actions, MAX_CREATURE_N * sizeof(int8_t));

    cudaMalloc(&h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int));

    cudaMalloc(&d_data, sizeof(CreatureData));
    cudaMemcpy(d_data, h_data, sizeof(CreatureData), cudaMemcpyHostToDevice);

    action_types_counts = new unsigned int[ACTION_TYPES_N];

    h_pinned = new CreatureData;
    cudaMallocHost(&h_pinned->x,            MAX_CREATURE_N * sizeof(unsigned int));
    cudaMallocHost(&h_pinned->y,            MAX_CREATURE_N * sizeof(unsigned int));
    cudaMallocHost(&h_pinned->energy,       MAX_CREATURE_N * sizeof(float));
    cudaMallocHost(&h_pinned->water,        MAX_CREATURE_N * sizeof(float));
    cudaMallocHost(&h_pinned->ids,          MAX_CREATURE_N * sizeof(long long));
    cudaMallocHost(&h_pinned->age,          MAX_CREATURE_N * sizeof(int));
    cudaMallocHost(&h_pinned->chosen_action,MAX_CREATURE_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->sensor_x,     MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->sensor_y,     MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->sensor_type,  MAX_CREATURE_N * SENSORS_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->action_x,      MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->action_y,      MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));
    cudaMallocHost(&h_pinned->action_type,   MAX_CREATURE_N * ACTIONS_N * sizeof(int8_t));



    cudaDeviceSynchronize();
    if (count > 0) InitializeRandomCreatures<<<(count + 255) / 256, 256>>>(d_data, count, seed, *global_id_counter);
    *global_id_counter += count;
    cudaDeviceSynchronize();
}


__global__ void InitializeRandomCreatures(CreatureData* creatures, int count, unsigned long long seed, long long global_id_counter) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    unsigned long long local_seed = derive_seed(seed, creature_index);

    //Initialize position and energy
    creatures->x[creature_index] = rand_int(derive_seed(local_seed, 0), WIDTH);
    creatures->y[creature_index] = rand_int(derive_seed(local_seed, 1), HEIGHT);
    creatures->energy[creature_index] = INITIAL_CREATURE_ENERGY;
    creatures->water[creature_index] = INITIAL_CREATURE_WATER;
    creatures->ids[creature_index] = global_id_counter + creature_index;
    creatures->age[creature_index] = 0;

    // Initialize sensors
    for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
        AddRandomSensors(creatures, creature_index, sensor_idx, derive_seed(local_seed, 2 + sensor_idx));
    }

    // Initialize network
    AddRandomNetwork(creatures, creature_index, derive_seed(local_seed, 10000));

    // Initialize actions
    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        SetRandomAction(creatures, creature_index, action_idx, derive_seed(local_seed, 20000 + action_idx));
    }
}


Creatures::~Creatures() {
    cudaFree(d_successful_births);

    cudaFree(h_data->x);
    cudaFree(h_data->y);
    cudaFree(h_data->energy);
    cudaFree(h_data->water);
    cudaFree(h_data->ids);
    cudaFree(h_data->age);
    cudaFree(d_attack_damage_kills);

    cudaFree(h_data->sensor_x);
    cudaFree(h_data->sensor_y);
    cudaFree(h_data->sensor_type);

    cudaFree(h_data->first_matrix);
    cudaFree(h_data->second_matrix);
    cudaFree(h_data->bias);

    cudaFree(h_data->action_x);
    cudaFree(h_data->action_y);
    cudaFree(h_data->action_type);

    cudaFree(h_data->chosen_action);

    cudaFree(h_data->move_queue_creatures);
    cudaFree(h_data->eat_queue_creatures);
    cudaFree(h_data->attack_queue_creatures);
    cudaFree(h_data->reproduce_queue_creatures);
    cudaFree(h_data->drink_queue_creatures);

    cudaFree(h_data->move_queue_actions);
    cudaFree(h_data->eat_queue_actions);
    cudaFree(h_data->attack_queue_actions);
    cudaFree(h_data->reproduce_queue_actions);
    cudaFree(h_data->drink_queue_actions);

    cudaFree(h_data->action_types_counts);

    cudaFree(d_data);
    delete h_data;

    cudaFreeHost(h_pinned->x);
    cudaFreeHost(h_pinned->y);
    cudaFreeHost(h_pinned->energy);
    cudaFreeHost(h_pinned->water);
    cudaFreeHost(h_pinned->ids);
    cudaFreeHost(h_pinned->age);
    cudaFreeHost(h_pinned->chosen_action);

    cudaFreeHost(h_pinned->sensor_x);
    cudaFreeHost(h_pinned->sensor_y);
    cudaFreeHost(h_pinned->sensor_type);
    cudaFreeHost(h_pinned->action_x);
    cudaFreeHost(h_pinned->action_y);
    cudaFreeHost(h_pinned->action_type);

    delete h_pinned;
    delete[] action_types_counts;
}
