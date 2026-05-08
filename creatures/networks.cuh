# ifndef NETWORKS_CUH
# define NETWORKS_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

class Networks {
    public:
    int8_t* output_neurons_n;
    __nv_fp8_e4m3* first_matrix;
    __nv_fp8_e4m3* second_matrix;
    __nv_fp8_e4m3* bias;

    __nv_fp8_e4m3* hidden_neuron_values;
    __nv_fp8_e4m3* output_neuron_values;

    Networks(curandState* state, int creatures_n);
    ~Networks();

    __device__ void AddRandomNetwork(int creature_index, curandState &state);
};

__global__ void InitializeRandomNetworks(Networks* networks, int creatures_n, curandState* states);

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx);

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx);

# endif