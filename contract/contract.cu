#include "contract/contract.cuh"
#include "constants.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// #################################################################################################################################
// Mode 0, initial, stable version, allocates the memory in each call dynamically, scans to establish new ids, 1 thread copies 1 creatue

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
    d_creature_alive[creature_index] = (d_creatures->energy[creature_index] > 0.0f && d_creatures->water[creature_index] > 0.0f) ? 1 : 0;
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
    d_new_creatures->water[new_creature_idx] = d_old_creatures->water[old_creature_index];
    d_new_creatures->ids[new_creature_idx] = d_old_creatures->ids[old_creature_index];
    d_new_creatures->age[new_creature_idx] = d_old_creatures->age[old_creature_index];

    for (int i = 0; i < MILIEU_SENSORS_N; i++) {
        d_new_creatures->sensor_x[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_x[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->sensor_y[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_y[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->sensor_type[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->sensor_type[i * MAX_CREATURE_N + old_creature_index];
    }

    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        for(int sensor_idx = 0; sensor_idx < INPUT_NEURONS_N; sensor_idx++) {
            d_new_creatures->first_matrix[get_first_matrix_idx(new_creature_idx, hidden_idx, sensor_idx)] = d_old_creatures->first_matrix[get_first_matrix_idx(old_creature_index, hidden_idx, sensor_idx)];
        }
    }

    for(int action_idx = 0; action_idx < ACTIONS_N; action_idx++) {
        for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
            d_new_creatures->second_matrix[get_second_matrix_idx(new_creature_idx, action_idx, hidden_idx)] = d_old_creatures->second_matrix[get_second_matrix_idx(old_creature_index, action_idx, hidden_idx)];
        }
    }

    for (int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        d_new_creatures->bias[new_creature_idx * HIDDEN_NEURONS_N + hidden_idx] =
            d_old_creatures->bias[old_creature_index * HIDDEN_NEURONS_N + hidden_idx];
    }

    for (int i = 0; i < ACTIONS_N; i++) {
        d_new_creatures->action_x[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_x[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->action_y[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_y[i * MAX_CREATURE_N + old_creature_index];
        d_new_creatures->action_type[i * MAX_CREATURE_N + new_creature_idx] = d_old_creatures->action_type[i * MAX_CREATURE_N + old_creature_index];
    }
}

// #################################################################################################################################
// Mode 1, introduce workspace memory allocation, instead of allocating the memory in each contract call, allocate it once and reuse

static void ensure_contract_workspace(ContractWorkspace* workspace, int required_capacity) {
    if (required_capacity > workspace->capacity) {
        if (workspace->d_contracted_creature_indices != nullptr) {
            cudaFree(workspace->d_contracted_creature_indices);
        }

        if (workspace->d_creature_alive != nullptr) {
            cudaFree(workspace->d_creature_alive);
        }

        cudaMalloc(&workspace->d_contracted_creature_indices, required_capacity * sizeof(int));
        cudaMalloc(&workspace->d_creature_alive, required_capacity * sizeof(int));

        workspace->capacity = required_capacity;
    }

    if (workspace->d_new_count == nullptr) {
        cudaMalloc(&workspace->d_new_count, sizeof(int));
    }
}


void free_contract_workspace(ContractWorkspace* workspace) {
    if (workspace->d_contracted_creature_indices != nullptr) {
        cudaFree(workspace->d_contracted_creature_indices);
        workspace->d_contracted_creature_indices = nullptr;
    }

    if (workspace->d_creature_alive != nullptr) {
        cudaFree(workspace->d_creature_alive);
        workspace->d_creature_alive = nullptr;
    }

    if (workspace->d_new_count != nullptr) {
        cudaFree(workspace->d_new_count);
        workspace->d_new_count = nullptr;
    }

    workspace->capacity = 0;
}


void contract_optimized(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
) {
    if (old_creatures->count <= 0) {
        new_creatures->count = 0;
        return;
    }

    ensure_contract_workspace(workspace, old_creatures->count);

    int* d_contracted_creature_indices = workspace->d_contracted_creature_indices;
    int* d_creature_alive = workspace->d_creature_alive;

    d_calculate_live_creatures<<<(old_creatures->count + 255) / 256, 256>>>(
        old_creatures->d_data,
        d_creature_alive,
        old_creatures->count
    );

    thrust::device_ptr<int> dev_flags(d_creature_alive);
    thrust::device_ptr<int> dev_indices(d_contracted_creature_indices);

    thrust::exclusive_scan(
        thrust::device,
        dev_flags,
        dev_flags + old_creatures->count,
        dev_indices
    );

    contract<<<(old_creatures->count + 255) / 256, 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        old_creatures->count
    );

    int last_creature_alive;
    int new_count;

    cudaMemcpy(
        &last_creature_alive,
        d_creature_alive + old_creatures->count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        &new_count,
        d_contracted_creature_indices + old_creatures->count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    new_count += last_creature_alive;

    new_creatures->count = new_count;

    cudaMemcpy(
        new_creatures->d_data,
        new_creatures->h_data,
        sizeof(CreatureData),
        cudaMemcpyHostToDevice
    );
}

// ##################################################################################################
// Mode 2, copy the creatures using multiple threads instead of 1, keep the workspace allocation idea

static int blocks_for_items(size_t items) {
    return (int)((items + 255) / 256);
}


__global__ void d_contract_copy_scalars(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count
) {
    int old_creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (old_creature_index >= count) return;
    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    d_new_creatures->x[new_creature_idx] = d_old_creatures->x[old_creature_index];
    d_new_creatures->y[new_creature_idx] = d_old_creatures->y[old_creature_index];
    d_new_creatures->energy[new_creature_idx] = d_old_creatures->energy[old_creature_index];
    d_new_creatures->water[new_creature_idx] = d_old_creatures->water[old_creature_index];
    d_new_creatures->ids[new_creature_idx] = d_old_creatures->ids[old_creature_index];
    d_new_creatures->age[new_creature_idx] = d_old_creatures->age[old_creature_index];
}


__global__ void d_contract_copy_sensors(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count,
    size_t total_items
) {
    size_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item >= total_items) return;

    int old_creature_index = item % count;
    int sensor_idx = item / count;

    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    size_t old_idx = sensor_idx * MAX_CREATURE_N + old_creature_index;
    size_t new_idx = sensor_idx * MAX_CREATURE_N + new_creature_idx;

    d_new_creatures->sensor_x[new_idx] = d_old_creatures->sensor_x[old_idx];
    d_new_creatures->sensor_y[new_idx] = d_old_creatures->sensor_y[old_idx];
    d_new_creatures->sensor_type[new_idx] = d_old_creatures->sensor_type[old_idx];
}


__global__ void d_contract_copy_first_matrix(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count,
    size_t total_items
) {
    size_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item >= total_items) return;

    int old_creature_index = item % count;
    size_t local_idx = item / count;

    int sensor_idx = local_idx % INPUT_NEURONS_N;
    int hidden_idx = local_idx / INPUT_NEURONS_N;

    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    d_new_creatures->first_matrix[get_first_matrix_idx(new_creature_idx, hidden_idx, sensor_idx)] =
        d_old_creatures->first_matrix[get_first_matrix_idx(old_creature_index, hidden_idx, sensor_idx)];
}


__global__ void d_contract_copy_second_matrix(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count,
    size_t total_items
) {
    size_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item >= total_items) return;

    int old_creature_index = item % count;
    size_t local_idx = item / count;

    int hidden_idx = local_idx % HIDDEN_NEURONS_N;
    int action_idx = local_idx / HIDDEN_NEURONS_N;

    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    d_new_creatures->second_matrix[get_second_matrix_idx(new_creature_idx, action_idx, hidden_idx)] =
        d_old_creatures->second_matrix[get_second_matrix_idx(old_creature_index, action_idx, hidden_idx)];
}


__global__ void d_contract_copy_bias(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count,
    size_t total_items
) {
    size_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item >= total_items) return;

    int old_creature_index = item % count;
    int hidden_idx = item / count;

    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    d_new_creatures->bias[get_hidden_layer_bias_idx(new_creature_idx, hidden_idx)] =
        d_old_creatures->bias[get_hidden_layer_bias_idx(old_creature_index, hidden_idx)];
}


__global__ void d_contract_copy_actions(
    CreatureData* d_old_creatures,
    CreatureData* d_new_creatures,
    int* d_contracted_creature_indices,
    int* d_creature_alive,
    int count,
    size_t total_items
) {
    size_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item >= total_items) return;

    int old_creature_index = item % count;
    int action_idx = item / count;

    if (!d_creature_alive[old_creature_index]) return;

    int new_creature_idx = d_contracted_creature_indices[old_creature_index];

    size_t old_idx = action_idx * MAX_CREATURE_N + old_creature_index;
    size_t new_idx = action_idx * MAX_CREATURE_N + new_creature_idx;

    d_new_creatures->action_x[new_idx] = d_old_creatures->action_x[old_idx];
    d_new_creatures->action_y[new_idx] = d_old_creatures->action_y[old_idx];
    d_new_creatures->action_type[new_idx] = d_old_creatures->action_type[old_idx];
}


void contract_optimized_split_copy(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
) {
    if (old_creatures->count <= 0) {
        new_creatures->count = 0;
        return;
    }

    ensure_contract_workspace(workspace, old_creatures->count);

    int* d_contracted_creature_indices = workspace->d_contracted_creature_indices;
    int* d_creature_alive = workspace->d_creature_alive;

    d_calculate_live_creatures<<<(old_creatures->count + 255) / 256, 256>>>(
        old_creatures->d_data,
        d_creature_alive,
        old_creatures->count
    );

    thrust::device_ptr<int> dev_flags(d_creature_alive);
    thrust::device_ptr<int> dev_indices(d_contracted_creature_indices);

    thrust::exclusive_scan(
        thrust::device,
        dev_flags,
        dev_flags + old_creatures->count,
        dev_indices
    );

    int count = old_creatures->count;

    d_contract_copy_scalars<<<(count + 255) / 256, 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count
    );

    size_t sensor_items = (size_t)count * MILIEU_SENSORS_N;
    d_contract_copy_sensors<<<blocks_for_items(sensor_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count,
        sensor_items
    );

    size_t first_matrix_items = (size_t)count * HIDDEN_NEURONS_N * INPUT_NEURONS_N;
    d_contract_copy_first_matrix<<<blocks_for_items(first_matrix_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count,
        first_matrix_items
    );

    size_t second_matrix_items = (size_t)count * ACTIONS_N * HIDDEN_NEURONS_N;
    d_contract_copy_second_matrix<<<blocks_for_items(second_matrix_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count,
        second_matrix_items
    );

    size_t bias_items = (size_t)count * HIDDEN_NEURONS_N;
    d_contract_copy_bias<<<blocks_for_items(bias_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count,
        bias_items
    );

    size_t action_items = (size_t)count * ACTIONS_N;
    d_contract_copy_actions<<<blocks_for_items(action_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_contracted_creature_indices,
        d_creature_alive,
        count,
        action_items
    );

    int last_creature_alive;
    int new_count;

    cudaMemcpy(
        &last_creature_alive,
        d_creature_alive + old_creatures->count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        &new_count,
        d_contracted_creature_indices + old_creatures->count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    new_count += last_creature_alive;

    new_creatures->count = new_count;

    cudaMemcpy(
        new_creatures->d_data,
        new_creatures->h_data,
        sizeof(CreatureData),
        cudaMemcpyHostToDevice
    );
}

// ##########################################################################################
// Mode 3, keep the mode 2 improvements, use AtomicAdd for new ids instead of the thrust scan

__global__ void d_contract_assign_atomic_indices(
    CreatureData* d_old_creatures,
    int* d_old_to_new,
    int* d_creature_alive,
    int* d_new_count,
    int count
) {
    int old_creature_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (old_creature_index >= count) return;

    int alive = (
        d_old_creatures->energy[old_creature_index] > 0.0f &&
        d_old_creatures->water[old_creature_index] > 0.0f
    ) ? 1 : 0;

    d_creature_alive[old_creature_index] = alive;

    if (!alive) {
        d_old_to_new[old_creature_index] = -1;
        return;
    }

    int new_creature_idx = atomicAdd(d_new_count, 1);
    d_old_to_new[old_creature_index] = new_creature_idx;
}

void contract_optimized_atomic(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
) {
    if (old_creatures->count <= 0) {
        new_creatures->count = 0;
        return;
    }

    ensure_contract_workspace(workspace, old_creatures->count);

    int count = old_creatures->count;

    int* d_old_to_new = workspace->d_contracted_creature_indices;
    int* d_creature_alive = workspace->d_creature_alive;
    int* d_new_count = workspace->d_new_count;

    cudaMemset(d_new_count, 0, sizeof(int));

    d_contract_assign_atomic_indices<<<(count + 255) / 256, 256>>>(
        old_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        d_new_count,
        count
    );

    d_contract_copy_scalars<<<(count + 255) / 256, 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count
    );

    size_t sensor_items = (size_t)count * MILIEU_SENSORS_N;
    d_contract_copy_sensors<<<blocks_for_items(sensor_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count,
        sensor_items
    );

    size_t first_matrix_items = (size_t)count * HIDDEN_NEURONS_N * INPUT_NEURONS_N;
    d_contract_copy_first_matrix<<<blocks_for_items(first_matrix_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count,
        first_matrix_items
    );

    size_t second_matrix_items = (size_t)count * ACTIONS_N * HIDDEN_NEURONS_N;
    d_contract_copy_second_matrix<<<blocks_for_items(second_matrix_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count,
        second_matrix_items
    );

    size_t bias_items = (size_t)count * HIDDEN_NEURONS_N;
    d_contract_copy_bias<<<blocks_for_items(bias_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count,
        bias_items
    );

    size_t action_items = (size_t)count * ACTIONS_N;
    d_contract_copy_actions<<<blocks_for_items(action_items), 256>>>(
        old_creatures->d_data,
        new_creatures->d_data,
        d_old_to_new,
        d_creature_alive,
        count,
        action_items
    );

    int new_count;

    cudaMemcpy(
        &new_count,
        d_new_count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    new_creatures->count = new_count;

    cudaMemcpy(
        new_creatures->d_data,
        new_creatures->h_data,
        sizeof(CreatureData),
        cudaMemcpyHostToDevice
    );
}