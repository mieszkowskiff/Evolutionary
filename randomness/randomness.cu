#include <curand_kernel.h>
#include <csignal>
#include "randomness/randomness.cuh"
#include "constants.h"

__global__ void init_curand_states(curandState* states, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_CREATURE_N)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}