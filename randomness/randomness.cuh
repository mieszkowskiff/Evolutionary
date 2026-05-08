#ifndef RANDOMNESS_CUH
#define RANDOMNESS_CUH

__global__ void init_curand_states(curandState* states, unsigned long long seed);

#endif