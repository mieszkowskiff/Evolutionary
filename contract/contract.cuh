# ifndef CONTRACT_CUH
# define CONTRACT_CUH

#include "creatures/creatures.cuh"
#include <curand_kernel.h>
#include <cuda_fp8.h>

void contract(Creatures* old_creatures, Creatures* new_creatures);

struct ContractWorkspace {
    int capacity = 0;
    int* d_contracted_creature_indices = nullptr;
    int* d_creature_alive = nullptr;
    int* d_new_count = nullptr;
};

void free_contract_workspace(ContractWorkspace* workspace);

void contract_optimized(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
);

void contract_optimized_split_copy(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
);

void contract_optimized_atomic(
    Creatures* old_creatures,
    Creatures* new_creatures,
    ContractWorkspace* workspace
);

__global__ void d_calculate_live_creatures(CreatureData* d_creatures, int* d_creature_alive, int count);

__global__ void contract(CreatureData* d_old_creatures, CreatureData* d_new_creatures, int* d_contracted_creature_indices, int* d_creature_alive, int count);

# endif // CONTRACT_CUH