# ifndef NETWORKS_CUH
# define NETWORKS_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

class Networks {
    public:
    int8_t* hidden_neurons_n;
    int8_t* output_neurons_n;
    __nv_fp8_e4m3* first_matrix;
    __nv_fp8_e4m3* second_matrix;
    __nv_fp8_e4m3* bias;

    __nv_fp8_e4m3* hidden_neuron_values;
    __nv_fp8_e4m3* output_neuron_values;

    Networks(int8_t* input_neurons_n, int8_t* output_neurons_n, int creature_n);
    ~Networks();

    private:
    __device__ void AddRandomNetwork(int creature_index, curandState &state);
};

# endif