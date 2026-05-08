# ifndef ACTIONS_CUH
# define ACTIONS_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

class Actions {
    public:
    int8_t* action_x;
    int8_t* action_y;
    int8_t* action_type;

    Actions(curandState* state, int creatures_n);
    ~Actions();

    __device__ void SetRandomAction(int creature_index, int action_index, curandState& state);
};

__global__ void SetActions(Actions* actions, int creatures_n, curandState* state);

# endif