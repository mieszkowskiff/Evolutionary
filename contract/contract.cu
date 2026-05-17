#include "contract/contract.cuh"
#include "constants.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


void contract(Creatures* old_creatures, Creatures* new_creatures) {

    int* d_contracted_creature_indices;
    int* d_creature_alive;
    cudaMalloc(&d_contracted_creature_indices, old_creatures->count * sizeof(int));
    cudaMalloc(&d_creature_alive, old_creatures->count * sizeof(int));
    
    d_calculate_live_creatures<<<(old_creatures->count + 255) / 256, 256>>>(old_creatures->d_data, d_creature_alive, old_creatures->count);

    thrust::device_ptr<int> dev_flags(d_creature_alive);
    thrust::device_ptr<int> dev_indices(d_contracted_creature_indices);
    thrust::exclusive_scan(thrust::device, dev_flags, dev_flags + old_creatures->count, dev_indices);

    contract<<<(old_creatures->count + 255) / 256, 256>>>(old_creatures->d_data, new_creatures->d_data, d_contracted_creature_indices, d_creature_alive, old_creatures->count);

    int last_creature_alive;// = d_creature_alive[old_creatures->count - 1];
    int new_count;// = d_contracted_creature_indices[old_creatures->count - 1] + last_creature_alive;

    cudaMemcpy(&last_creature_alive, d_creature_alive + old_creatures->count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_count, d_contracted_creature_indices + old_creatures->count - 1, sizeof(int), cudaMemcpyDeviceToHost);

    new_count += last_creature_alive;

    new_creatures->count = new_count;
    
    cudaMemcpy(new_creatures->d_data, new_creatures->h_data, sizeof(CreatureData), cudaMemcpyHostToDevice);

    cudaFree(d_contracted_creature_indices);
    cudaFree(d_creature_alive);
}


__global__ void d_calculate_live_creatures(CreatureData* d_creatures, int* d_creature_alive, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;
    d_creature_alive[creature_index] = d_creatures->energy[creature_index] > 0.0f ? 1 : 0;
}

__global__ void contract(CreatureData* d_old_creatures, CreatureData* d_new_creatures, int* d_contracted_creature_indices, int* d_creature_alive, int count) {
    int old_creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (old_creature_index >= count) return;
    if (!d_creature_alive[old_creature_index]) {
        return;
    }

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    //Copying data

    d_new_creatures->x[new_creature_idx] = d_old_creatures->x[old_creature_index];
    d_new_creatures->y[new_creature_idx] = d_old_creatures->y[old_creature_index];
    d_new_creatures->energy[new_creature_idx] = d_old_creatures->energy[old_creature_index];
    d_new_creatures->ids[new_creature_idx] = d_old_creatures->ids[old_creature_index];

    for (int i = 0; i < SENSORS_N; i++) {
        d_new_creatures->sensor_x[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_x[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->sensor_y[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_y[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->sensor_type[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_type[i * MAX_CREATURE_N + old_creature_index];
    }

    for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < SENSORS_N; sensor_idx++) {
            d_new_creatures->first_matrix[get_first_matrix_idx(new_creature_idx, hidden_idx, sensor_idx)] = d_old_creatures->first_matrix[get_first_matrix_idx(old_creature_index, hidden_idx, sensor_idx)];
        }
    }

    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_N; hidden_idx++) {
            d_new_creatures->second_matrix[get_second_matrix_idx(new_creature_idx, action_idx, hidden_idx)] = d_old_creatures->second_matrix[get_second_matrix_idx(old_creature_index, action_idx, hidden_idx)];
        }
    }

    for (int i = 0; i < ACTIONS_N; i++) {
        d_new_creatures->action_x[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_x[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->action_y[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_y[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->action_type[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_type[i * MAX_CREATURE_N + old_creature_index];
    }
}