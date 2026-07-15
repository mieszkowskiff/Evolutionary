#include "creatures/creatures.cuh"
#include <cuda_fp8.h>
#include "iostream"
#include "map/map.cuh"
#include <mma.h>
#include <cuda_fp16.h>


__global__ void d_PerceveMap(MapData* d_map, CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int sensor_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (sensor_index >= MILIEU_SENSORS_N) return;

    int8_t sensor_x = d_creatures->sensor_x[get_sensor_idx(creature_index, sensor_index)];
    int8_t sensor_y = d_creatures->sensor_y[get_sensor_idx(creature_index, sensor_index)];
    int8_t sensor_type = d_creatures->sensor_type[get_sensor_idx(creature_index, sensor_index)];

    int creature_x = d_creatures->x[creature_index];
    int creature_y = d_creatures->y[creature_index];

    int sensor_position = get_cell_index(creature_x + sensor_x, creature_y + sensor_y);

    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, sensor_index)] = __nv_fp8_e4m3(get_cell(d_map, sensor_type, sensor_position));
}

__global__ void d_PerceveSimulation(MapData* d_map, CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 0)] = __nv_fp8_e4m3(d_map->season_cos);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 1)] = __nv_fp8_e4m3(d_map->season_sin);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 2)] = __nv_fp8_e4m3(energy);
    d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, MILIEU_SENSORS_N + 3)] = __nv_fp8_e4m3(water);
}

__global__ void d_PopulateHiddenLayer(CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;
    
    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int hidden_neuron_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (hidden_neuron_index >= HIDDEN_NEURONS_N) return;

    float acummulation = static_cast<float>(d_creatures->bias[get_hidden_layer_bias_idx(creature_index, hidden_neuron_index)]);

    #pragma unroll
    for(int input_idx = 0; input_idx < INPUT_NEURONS_N; input_idx++) {
        size_t weight_idx = get_first_matrix_idx(creature_index, hidden_neuron_index, input_idx);
        acummulation += static_cast<float>(d_creatures->first_matrix[weight_idx]) * static_cast<float>(d_creatures->input_layer_values[get_input_layer_value_idx(creature_index, input_idx)]);
    }

    d_creatures->hidden_layer_values[get_hidden_layer_value_idx(creature_index, hidden_neuron_index)] = __nv_fp8_e4m3(acummulation);
}

__global__ void d_PopulateOutputLayer(CreatureData* d_creatures, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    int output_neuron_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (output_neuron_index >= OUTPUT_NEURONS_N) return;

   float acummulation = 0.0f;

    #pragma unroll
    for(int hidden_idx = 0; hidden_idx < HIDDEN_NEURONS_N; hidden_idx++) {
        size_t weight_idx = get_second_matrix_idx(creature_index, output_neuron_index, hidden_idx);
        acummulation += static_cast<float>(d_creatures->second_matrix[weight_idx]) * static_cast<float>(d_creatures->hidden_layer_values[get_hidden_layer_value_idx(creature_index, hidden_idx)]);
    }

    d_creatures->output_layer_values[get_output_layer_value_idx(creature_index, output_neuron_index)] = acummulation;
}

__global__ void d_ActionSelection(CreatureData* d_creatures, unsigned long long seed, int count) {
    int creature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f) return;
    if (water <= 0.0f) return;

    unsigned long long local_seed = derive_seed(seed, creature_index);

    float max_output = -__FLT_MAX__;

    float neuron_values[OUTPUT_NEURONS_N];
    
    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        neuron_values[output_idx] = d_creatures->output_layer_values[get_output_layer_value_idx(creature_index, output_idx)];
        if (neuron_values[output_idx] > max_output) {
            max_output = neuron_values[output_idx];
        }
    }

    float exp_values[OUTPUT_NEURONS_N];
    float exp_sum = 0.0f;

    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        float exp_val = expf(neuron_values[output_idx] - max_output);
        exp_values[output_idx] = exp_val;
        exp_sum += exp_val;
    }

    float random_val = rand_float(derive_seed(local_seed, 4097)) * exp_sum;
    float cumulative_sum = 0.0f;
    int selected_action = -1;

    #pragma unroll
    for(int output_idx = 0; output_idx < OUTPUT_NEURONS_N; output_idx++) {
        cumulative_sum += exp_values[output_idx];
        if (random_val <= cumulative_sum) {
            selected_action = output_idx;
            break;
        }
    }

    if (selected_action != -1) {
        unsigned int type = d_creatures->action_type[get_action_idx(creature_index, selected_action)];

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


using namespace nvcuda;

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32
#define BLOCK_SIZE (WARPS_PER_BLOCK * WARP_SIZE)

__global__ void d_PopulateHiddenLayer_WMMA(CreatureData* d_creatures, int count) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_id_local = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int creature_index = warp_id_global;

    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f || water <= 0.0f) return;

    // Shared Memory buffered as 'half' (FP16) to satisfy WMMA templates
    // A: 16 rows (padded), 64 inputs
    __shared__ half smem_A[WARPS_PER_BLOCK][16][64];
    // B: 32 hidden neurons, 64 inputs (Row-Major layout)
    __shared__ half smem_B[WARPS_PER_BLOCK][32][64];
    // C: 16 rows (padded), 32 hidden neurons output
    __shared__ float smem_C[WARPS_PER_BLOCK][16][32];

    // 1. Initialize A to zeros and load specific creature data (Casting FP8 -> Float -> Half)
    #pragma unroll
    for (int row = 0; row < 16; row++) {
        smem_A[warp_id_local][row][lane_id] = __float2half(0.0f);
        smem_A[warp_id_local][row][lane_id + 32] = __float2half(0.0f);
    }

    int in_idx_1 = get_input_layer_value_idx(creature_index, lane_id);
    int in_idx_2 = get_input_layer_value_idx(creature_index, lane_id + 32);
    smem_A[warp_id_local][0][lane_id] = __float2half(static_cast<float>(d_creatures->input_layer_values[in_idx_1]));
    smem_A[warp_id_local][0][lane_id + 32] = __float2half(static_cast<float>(d_creatures->input_layer_values[in_idx_2]));

    // 2. Load Weights from Global (FP8) to Shared (Half)
    __nv_fp8_e4m3* base_B = d_creatures->first_matrix + get_first_matrix_idx(creature_index, 0, 0);
    
    // 32 threads load 32 neurons. Each thread loads all 64 sensors for its assigned neuron.
    for (int sensor_idx = 0; sensor_idx < 64; sensor_idx++) {
        float weight_val = static_cast<float>(base_B[lane_id * 64 + sensor_idx]);
        smem_B[warp_id_local][lane_id][sensor_idx] = __float2half(weight_val);
    }

    __syncwarp();

    // 3. Declare WMMA fragments with 'half' (M=16, N=16, K=16)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0; // Neurons 0-15
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag1; // Neurons 16-31
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;

    wmma::fill_fragment(c_frag0, 0.0f);
    wmma::fill_fragment(c_frag1, 0.0f);

    // 4. K-dimension tiling: K is 64, step is 16 (4 iterations)
    for (int k_step = 0; k_step < 64; k_step += 16) {
        wmma::load_matrix_sync(a_frag, &smem_A[warp_id_local][0][k_step], 64);
        wmma::load_matrix_sync(b_frag0, &smem_B[warp_id_local][0][k_step], 64);
        wmma::load_matrix_sync(b_frag1, &smem_B[warp_id_local][16][k_step], 64);

        wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
    }

    // 5. Store back and write to Global Memory in FP8
    wmma::store_matrix_sync(&smem_C[warp_id_local][0][0], c_frag0, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_C[warp_id_local][0][16], c_frag1, 32, wmma::mem_row_major);

    __syncwarp();

    float result = smem_C[warp_id_local][0][lane_id];
    float bias_val = static_cast<float>(d_creatures->bias[get_hidden_layer_bias_idx(creature_index, lane_id)]);
    
    d_creatures->hidden_layer_values[get_hidden_layer_value_idx(creature_index, lane_id)] = __nv_fp8_e4m3(result + bias_val);
}

__global__ void d_PopulateOutputLayer_WMMA(CreatureData* d_creatures, int count) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_id_local = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int creature_index = warp_id_global;

    if (creature_index >= count) return;

    float energy = d_creatures->energy[creature_index];
    float water = d_creatures->water[creature_index];
    if (energy <= 0.0f || water <= 0.0f) return;

    __shared__ half smem_A[WARPS_PER_BLOCK][16][32];
    __shared__ half smem_B[WARPS_PER_BLOCK][32][32]; // 32 actions, 32 hidden neurons
    __shared__ float smem_C[WARPS_PER_BLOCK][16][32];

    #pragma unroll
    for (int row = 0; row < 16; row++) {
        smem_A[warp_id_local][row][lane_id] = __float2half(0.0f);
    }

    int hidden_val_idx = get_hidden_layer_value_idx(creature_index, lane_id);
    smem_A[warp_id_local][0][lane_id] = __float2half(static_cast<float>(d_creatures->hidden_layer_values[hidden_val_idx]));

    __nv_fp8_e4m3* base_B = d_creatures->second_matrix + get_second_matrix_idx(creature_index, 0, 0);
    
    // Load Weights
    for (int hidden_idx = 0; hidden_idx < 32; hidden_idx++) {
        float weight_val = static_cast<float>(base_B[lane_id * 32 + hidden_idx]);
        smem_B[warp_id_local][lane_id][hidden_idx] = __float2half(weight_val);
    }

    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0; 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag1; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;

    wmma::fill_fragment(c_frag0, 0.0f);
    wmma::fill_fragment(c_frag1, 0.0f);

    // K is 32, step is 16 (2 iterations)
    for (int k_step = 0; k_step < 32; k_step += 16) {
        wmma::load_matrix_sync(a_frag, &smem_A[warp_id_local][0][k_step], 32);
        wmma::load_matrix_sync(b_frag0, &smem_B[warp_id_local][0][k_step], 32);
        wmma::load_matrix_sync(b_frag1, &smem_B[warp_id_local][16][k_step], 32);

        wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
    }

    wmma::store_matrix_sync(&smem_C[warp_id_local][0][0], c_frag0, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_C[warp_id_local][0][16], c_frag1, 32, wmma::mem_row_major);

    __syncwarp();

    d_creatures->output_layer_values[get_output_layer_value_idx(creature_index, lane_id)] = smem_C[warp_id_local][0][lane_id];
}



#if true //change to false to use tensor cores

void Creatures::ChooseAction(Map* map, unsigned long long seed, float season_cos, float season_sin) {
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));

    d_PerceveMap<<<dim3((count + 127) / 128, (MILIEU_SENSORS_N + 7) / 8), dim3(128, 8), 0, compute_stream>>>(map->d_data, d_data, count);
    d_PerceveSimulation<<<(count + 255) / 256, 256, 0, compute_stream>>>(map->d_data, d_data, count);
    d_PopulateHiddenLayer<<<dim3((count + 127) / 128, (HIDDEN_NEURONS_N + 7) / 8), dim3(128, 8), 0, compute_stream>>>(d_data, count);
    d_PopulateOutputLayer<<<dim3((count + 127) / 128, (OUTPUT_NEURONS_N + 7) / 8), dim3(128, 8), 0, compute_stream>>>(d_data, count);
    d_ActionSelection<<<(count + 255) / 256, 256, 0, compute_stream>>>(d_data, seed, count);

    cudaDeviceSynchronize();

    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //TODO: sort actions per type to make it more efficient (partial coalescing)
}

#else
void Creatures::ChooseAction(Map* map, unsigned long long seed, float season_cos, float season_sin) {
    cudaMemset(h_data->action_types_counts, 0, ACTION_TYPES_N * sizeof(unsigned int));
    d_PerceveMap<<<dim3((count + 127) / 128, (MILIEU_SENSORS_N + 7) / 8), dim3(128, 8), 0, compute_stream>>>(map->d_data, d_data, count);
    d_PerceveSimulation<<<(count + 255) / 256, 256, 0, compute_stream>>>(map->d_data, d_data, count);

    int threads_per_block = 128;
    int blocks = (count * 32 + threads_per_block - 1) / threads_per_block;

    d_PopulateHiddenLayer_WMMA<<<blocks, threads_per_block, 0, compute_stream>>>(d_data, count);
    d_PopulateOutputLayer_WMMA<<<blocks, threads_per_block, 0, compute_stream>>>(d_data, count);

    d_ActionSelection<<<(count + 255) / 256, 256, 0, compute_stream>>>(d_data, seed, count);

    cudaDeviceSynchronize();

    cudaMemcpy(this->action_types_counts, h_data->action_types_counts, ACTION_TYPES_N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}
#endif