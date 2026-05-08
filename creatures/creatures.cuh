#ifndef CREATURES_CUH
# define CREATURES_CUH

#include <curand_kernel.h>
#include <cuda_fp8.h>
#include "constants.h"

struct CreatureData {

    //General data
    unsigned int* x;
    unsigned int* y;
    __nv_fp8_e4m3* energy;

    // Sensors data
    int8_t* sensor_x;
    int8_t* sensor_y;
    int8_t* sensor_type;

    //Network data
    int8_t* output_neurons_n;
    __nv_fp8_e4m3* first_matrix;
    __nv_fp8_e4m3* second_matrix;
    __nv_fp8_e4m3* bias;

    // Action data
    int8_t* action_x;
    int8_t* action_y;
    int8_t* action_type;

    // Temporary data for calculations
    __nv_fp8_e4m3* hidden_neuron_values;
    __nv_fp8_e4m3* output_neuron_values;
};

class Creatures {
    int count;
    
    CreatureData* d_data;
    CreatureData* h_data;

    public:
    Creatures(curandState* state, int count);
    ~Creatures();
};

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx);

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx);

__global__ void InitializeRandomCreatures(CreatureData* d_data, int count, curandState* states);

__device__ void AddRandomSensors(CreatureData* creatures, int creature_index, int sensor_index, curandState& state);

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, curandState &state);

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, curandState& state);


# endif