#include "creatures/creatures.cuh"
#include "constants.h"
#include <stdio.h>
#include <thrust/device_ptr.h>

static void SaveMapAfterDamage(Map* map, int tick) {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    cudaMemcpy(map->h_pinned->food,     map->h_data->food,     bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->water,    map->h_data->water,    bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->danger,   map->h_data->danger,   bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->creature, map->h_data->creature, bytes, cudaMemcpyDeviceToHost);

    char fname[128];
    snprintf(fname, sizeof(fname), "map_%06d_after_damage.bin", tick);

    FILE* f = fopen(fname, "wb");

    int width = WIDTH, height = HEIGHT;
    fwrite(&width,  sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(map->h_pinned->food,     sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->danger,   sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->creature, sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->water,    sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);
}

Creatures::Creatures(curandState* state, int count, long long *global_id_counter) {

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
    if (count > 0) InitializeRandomCreatures<<<(count + 255) / 256, 256>>>(d_data, count, state, *global_id_counter);
    *global_id_counter += count;
    cudaDeviceSynchronize();
}

void Creatures::Save_tick(int tick) {
    size_t bytes_count = count * sizeof(unsigned int);
    size_t bytes_energy = count * sizeof(float);
    size_t bytes_id = count * sizeof(long long);

    cudaMemcpy(h_pinned->x,      h_data->x,      bytes_count,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->y,      h_data->y,      bytes_count,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->energy, h_data->energy, bytes_energy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->water,  h_data->water,  bytes_energy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->ids,    h_data->ids,    bytes_id,     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->chosen_action, h_data->chosen_action, count * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_x, h_data->sensor_x, count * SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_y, h_data->sensor_y, count * SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_type, h_data->sensor_type, count * SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_x, h_data->action_x, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_y, h_data->action_y, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_type, h_data->action_type, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);

    char fname[64];
    snprintf(fname, sizeof(fname), "creatures_%06d.bin", tick);
    FILE* f = fopen(fname, "wb");
    int sensors_n = SENSORS_N, actions_n = ACTIONS_N;
    fwrite(&count,     sizeof(int), 1, f);
    fwrite(&sensors_n, sizeof(int), 1, f);
    fwrite(&actions_n, sizeof(int), 1, f);
    fwrite(h_pinned->x,     sizeof(unsigned int), count, f);
    fwrite(h_pinned->y,     sizeof(unsigned int), count, f);
    fwrite(h_pinned->energy,sizeof(float),        count, f);
    fwrite(h_pinned->water, sizeof(float),        count, f);
    fwrite(h_pinned->ids,   sizeof(long long),    count, f);
    fwrite(h_pinned->chosen_action, sizeof(int8_t), count, f);
    fwrite(h_pinned->sensor_x, sizeof(int8_t), count * SENSORS_N, f);
    fwrite(h_pinned->sensor_y, sizeof(int8_t), count * SENSORS_N, f);
    fwrite(h_pinned->sensor_type, sizeof(int8_t), count * SENSORS_N, f);
    fwrite(h_pinned->action_x, sizeof(int8_t), count * ACTIONS_N, f);
    fwrite(h_pinned->action_y, sizeof(int8_t), count * ACTIONS_N, f);
    fwrite(h_pinned->action_type, sizeof(int8_t), count * ACTIONS_N, f);
    fclose(f);
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

__global__ void d_RebuildCreatureMap(MapData* d_map, CreatureData* d_creatures, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (d_creatures->energy[idx] <= 0.0f) return;
    if (d_creatures->water[idx] <= 0.0f) return;

    int x = d_creatures->x[idx];
    int y = d_creatures->y[idx];

    atomicAdd(&d_map->creature[get_cell_index(x, y)], 1.0f);
}

void Creatures::RebuildCreatureMap(Map* map) {
    cudaMemset(map->h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));

    if (count > 0) {
        d_RebuildCreatureMap<<<(count + 255) / 256, 256>>>(
            map->d_data,
            d_data,
            count
        );
    }
}

void Creatures::ChooseAction(Map* map, curandState* random_states, float season_cos, float season_sin) {
    cudaDeviceSynchronize();
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));
    if (count > 0) d_ActionStep<<<(count + 255) / 256, 256>>>(map->d_data, d_data, random_states, count, season_cos, season_sin);
    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //TODO: sort actions per type to make it more efficient (partial coalescing)
}

void Creatures::RunActions(Map* map, curandState* random_states) {
    static int action_tick = 0;

    h_attack_damage_kills = 0;
    cudaMemset(d_attack_damage_kills, 0, sizeof(unsigned int));

    if (action_types_counts[ATTACK_ACTION] > 0) d_AttackAction<<<(action_types_counts[ATTACK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    
    if (count > 0) d_ProcessEnergy<<<(count + 255) / 256, 256>>>(map->d_data, d_data, count, d_attack_damage_kills);

    cudaMemcpy(
        &h_attack_damage_kills,
        d_attack_damage_kills,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );

    if (SAVE_AFTER_DAMAGE_MAP && !(action_tick % SAVE_AFTER_DAMAGE_MAP_EVERY)) {
        cudaDeviceSynchronize();
        SaveMapAfterDamage(map, action_tick);
    }

    if (action_types_counts[MOVE_ACTION] > 0) d_MoveAction<<<(action_types_counts[MOVE_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[EAT_ACTION] > 0) d_EatAction<<<(action_types_counts[EAT_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[DRINK_ACTION] > 0) d_DrinkAction<<<(action_types_counts[DRINK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    int reproduce_count = action_types_counts[REPRODUCE_ACTION];

    if (count + reproduce_count > MAX_CREATURE_N) reproduce_count = MAX_CREATURE_N - count;
    if (reproduce_count < 0) reproduce_count = 0;

    unsigned int h_successful_births = 0;
    cudaMemset(d_successful_births, 0, sizeof(unsigned int));

    int max_children = MAX_CREATURE_N - count;

    if (reproduce_count > 0) d_ReproduceAction<<<(reproduce_count + 255) / 256, 256>>>(map->d_data, d_data, random_states, d_successful_births, *global_id_counter, count, max_children, reproduce_count);
    
    cudaMemcpy(&h_successful_births, d_successful_births, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    if (h_successful_births > (unsigned int)max_children) {
        h_successful_births = max_children;
    }
    
    *global_id_counter += h_successful_births;
    count += h_successful_births;

    action_tick++;

    cudaDeviceSynchronize();
}

__global__ void d_MoveAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[MOVE_ACTION]) return;
    int creature_index = d_creatures->move_queue_creatures[idx];
    if (d_creatures->energy[creature_index] <= 0) return;
    if (d_creatures->water[creature_index] <= 0) return;

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];

    int8_t action_index = d_creatures->move_queue_actions[idx];

    int8_t action_x = d_creatures->action_x[action_index * MAX_CREATURE_N + creature_index];
    int8_t action_y = d_creatures->action_y[action_index * MAX_CREATURE_N + creature_index];


    //d_map->creature[get_cell_index(creature_x, creature_y)] = __float2half(0.0f);
        
    int absolute_action_x = creature_x + action_x;
    int absolute_action_y = creature_y + action_y;


    if (absolute_action_x < 0) absolute_action_x += WIDTH;
    if (absolute_action_y < 0) absolute_action_y += HEIGHT;

    if (absolute_action_x >= WIDTH) absolute_action_x -= WIDTH;
    if (absolute_action_y >= HEIGHT) absolute_action_y -= HEIGHT;

    d_creatures->x[creature_index] = absolute_action_x;
    d_creatures->y[creature_index] = absolute_action_y;

    //__syncthreads();

    //atomicAdd(&d_map->creature[get_cell_index(absolute_action_x, absolute_action_y)], 1.0f); // maybe we should add energy instead of increment
}

__global__ void d_EatAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[EAT_ACTION]) return;
    int creature_index = d_creatures->eat_queue_creatures[idx];

    float energy = d_creatures->energy[creature_index];
    if (energy <= 0) return;
    if (d_creatures->water[creature_index] <= 0) return;

    int8_t action_index = d_creatures->eat_queue_actions[idx];

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

__global__ void d_DrinkAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[DRINK_ACTION]) return;
    int creature_index = d_creatures->drink_queue_creatures[idx];

    float water = d_creatures->water[creature_index];
    if (d_creatures->energy[creature_index] <= 0) return;
    if (water <= 0) return;

    int8_t action_index = d_creatures->drink_queue_actions[idx];

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

    float water_value = atomicExch(&d_map->water[get_cell_index(absolute_action_x, absolute_action_y)], 0.0f);

    d_creatures->water[creature_index] = water + water_value;
}

__global__ void d_AttackAction(MapData* d_map, CreatureData* d_creatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_creatures->action_types_counts[ATTACK_ACTION]) return;
    int creature_index = d_creatures->attack_queue_creatures[idx];

    float energy = d_creatures->energy[creature_index];
    if (energy <= 0) return;
    if (d_creatures->water[creature_index] <= 0) return;

    energy -= ATTACK_COST;
    d_creatures->energy[creature_index] = energy;
    if (energy <= 0) return;

    int8_t action_index = d_creatures->attack_queue_actions[idx];

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

    atomicAdd(&d_map->danger[get_cell_index(absolute_action_x, absolute_action_y)], ATTACK_DAMAGE);
}

__global__ void d_ProcessEnergy(MapData* d_map, CreatureData* d_creatures, int count, unsigned int* d_attack_damage_kills) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float energy = d_creatures->energy[idx];
    float water = d_creatures->water[idx];
    if (energy <= 0 || water <= 0) return;

    int age = d_creatures->age[idx];
    int creature_x = d_creatures->x[idx];
    int creature_y = d_creatures->y[idx];
    int cell = get_cell_index(creature_x, creature_y);

    // Decay - Energy and water dissipation
    energy -= ENERGY_DECAY;
    water -= WATER_DECAY;
    if (age >= OLD_AGE_START) {
        energy -= OLD_AGE_EXTRA_DECAY;
    }
    d_creatures->energy[idx] = energy;
    d_creatures->water[idx] = water;
    
    // Aging
    age += 1;
    d_creatures->age[idx] = age;
    
    // Energy, water and age check
    if (energy <= 0 || water <= 0) return;
    
    if (age >= MAX_AGE) {
        atomicAdd(&d_map->food[cell], energy);
        atomicAdd(&d_map->water[cell], water);
        d_creatures->energy[idx] = -0.1f;
        d_creatures->water[idx] = -0.1f;
        return;
    }

    // Damage distribution
    float damage = d_map->danger[cell];

    if (damage <= 0) {
        return;
    }

    if (energy - damage <= 0) {
        atomicAdd(d_attack_damage_kills, 1u);
        atomicAdd(&d_map->food[cell], energy);
        atomicAdd(&d_map->water[cell], water);
        d_creatures->energy[idx] = -0.1f;
        d_creatures->water[idx] = -0.1f;
        return;
    } else {
        d_creatures->energy[idx] = (energy - damage);
        return;
    }
}

__global__ void d_ReproduceAction(MapData* d_map, CreatureData* d_creatures, curandState* random_states, unsigned int* d_successful_births, long long global_id_counter, int count, int max_children, int reproduce_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reproduce_count) return;

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
    
    reproduce_creature(d_creatures, parent_creature_index, new_creature_idx, random_states[parent_creature_index], new_id);
}

__device__ void reproduce_creature(CreatureData* d_creatures, int parent_creature_index, int new_creature_idx, curandState& state, long long new_id) {

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


__global__ void d_ActionStep(MapData* d_map, CreatureData* d_creatures, curandState* random_states, int count, float season_cos, float season_sin) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;
    if (d_creatures->energy[creature_index] <= 0.0f) return;
    if (d_creatures->water[creature_index] <= 0.0f) return;

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

__global__ void InitializeRandomCreatures(CreatureData* creatures, int count, curandState* states, long long global_id_counter) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    curandState state = states[creature_index];

    //Initialize position and energy
    creatures->x[creature_index] = curand(&state) % WIDTH;
    creatures->y[creature_index] = curand(&state) % HEIGHT;
    creatures->energy[creature_index] = INITIAL_CREATURE_ENERGY;
    creatures->water[creature_index] = INITIAL_CREATURE_WATER;
    creatures->ids[creature_index] = global_id_counter + creature_index;
    creatures->age[creature_index] = 0;

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
        int8_t type = curand(&state) % 10; // 0: food, 1: danger, 2: creature, 3: water, 4-9: empty

        creatures->sensor_x[sensor_index * MAX_CREATURE_N + creature_index] = x;
        creatures->sensor_y[sensor_index * MAX_CREATURE_N + creature_index] = y;
        creatures->sensor_type[sensor_index * MAX_CREATURE_N + creature_index] = type;
}

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, curandState &state) {
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < TOTAL_SENSORS_N; sensor_idx++) {
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
    return (hidden_idx * TOTAL_SENSORS_N * MAX_CREATURE_N) + (sensor_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx) {
    return (output_idx * HIDDEN_N * MAX_CREATURE_N) + (hidden_idx * MAX_CREATURE_N) + creature_idx;
}

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, curandState& state) {
    float x_normal = curand_normal(&state) * ACTION_STDDEV;
    float y_normal = curand_normal(&state) * ACTION_STDDEV;
    
    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = curand(&state) % ACTION_TYPES_N; // 0: move, 1: eat, 2: attack, 3: reproduce, 4: drink

    creatures->action_x[action_index * MAX_CREATURE_N + creature_index] = x;
    creatures->action_y[action_index * MAX_CREATURE_N + creature_index] = y;
    creatures->action_type[action_index * MAX_CREATURE_N + creature_index] = type;
}

