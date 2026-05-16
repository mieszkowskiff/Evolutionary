# ifndef CONTRACT_CUH
# define CONTRACT_CUH

#include "creatures/creatures.cuh"
#include <curand_kernel.h>
#include <cuda_fp8.h>

void contract(Creatures* old_creatures, Creatures* new_creatures);

__global__ void d_calculate_live_creatures(CreatureData* d_creatures, int* d_creature_alive);

__global__ void contract(CreatureData* d_old_creatures, CreatureData* d_new_creatures, int* d_contracted_creature_indices, int* d_creature_alive);

# endif // CONTRACT_CUH