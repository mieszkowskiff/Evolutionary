#include "creatures/creatures.cuh"
#include "constants.h"
#include <thrust/device_ptr.h>


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
        d_RebuildCreatureMap<<<(count + 255) / 256, 256, 0, compute_stream>>>(
            map->d_data,
            d_data,
            count
        );
    }
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

__device__ void SetRandomSensor(CreatureData* creatures, int creature_index, int sensor_index, unsigned long long local_seed) {
        float x_normal = rand_normal(derive_seed(local_seed, 87943), 0.0f, SENSOR_STDDEV);
        float y_normal = rand_normal(derive_seed(local_seed, 57839), 0.0f, SENSOR_STDDEV);

        int8_t x = static_cast<int8_t>(roundf(x_normal));
        int8_t y = static_cast<int8_t>(roundf(y_normal));
        int8_t type = rand_int(derive_seed(local_seed, 12345), 10); // 0: food, 1: danger, 2: creature, 3: water, 4-9: empty

        creatures->sensor_x[get_sensor_idx(creature_index, sensor_index)] = x;
        creatures->sensor_y[get_sensor_idx(creature_index, sensor_index)] = y;
        creatures->sensor_type[get_sensor_idx(creature_index, sensor_index)] = type;
}

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, unsigned long long local_seed) {
    
    // First matrix
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < INPUT_NEURONS_N; sensor_idx++) {
            size_t idx = get_first_matrix_idx(creature_index, hidden_idx, sensor_idx);
            creatures->first_matrix[idx] = __nv_fp8_e4m3(rand_float(derive_seed(local_seed, 100000 + hidden_idx * INPUT_NEURONS_N + sensor_idx)) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Second matrix
    for(int output_idx = 0; output_idx < ACTIONS_N; output_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
            size_t idx = get_second_matrix_idx(creature_index, output_idx, hidden_idx);
            creatures->second_matrix[idx] = __nv_fp8_e4m3(rand_float(derive_seed(local_seed, 200000 + output_idx * HIDDEN_NEURONS_N + hidden_idx)) * 2 - 1); // Random value between -1 and 1
        }
    }

    // Bias
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        size_t idx = (creature_index * HIDDEN_NEURONS_N) + hidden_idx;
        creatures->bias[idx] = __nv_fp8_e4m3(rand_float(derive_seed(local_seed, 300000 + creature_index * HIDDEN_NEURONS_N + hidden_idx)) * 2 - 1); // Random value between -1 and 1
    }
}

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, unsigned long long local_seed) {
    float x_normal = rand_normal(derive_seed(local_seed, 89347), 0.0f, ACTION_STDDEV);
    float y_normal = rand_normal(derive_seed(local_seed, 29838), 0.0f, ACTION_STDDEV);

    int8_t x = static_cast<int8_t>(roundf(x_normal));
    int8_t y = static_cast<int8_t>(roundf(y_normal));
    int8_t type = rand_int(derive_seed(local_seed, 12345), ACTION_TYPES_N); // 0: move, 1: eat, 2: attack, 3: reproduce, 4: drink

    creatures->action_x[action_index * MAX_CREATURE_N + creature_index] = x;
    creatures->action_y[action_index * MAX_CREATURE_N + creature_index] = y;
    creatures->action_type[action_index * MAX_CREATURE_N + creature_index] = type;
}


