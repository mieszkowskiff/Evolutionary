#include "contract/contract.cuh"
#include "constants.h"


__global__ void contract(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    float* creature_matrix,
    float* creature_bias,
    unsigned int* creature_x_save,
    unsigned int* creature_y_save,
    float* creature_energy_save,
    float* creature_matrix_save,
    float* creature_bias_save,
    int* contracted_creature_indices,
    int *creature_alive,
    int creature_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= creature_n) return;
    if (!creature_alive[idx]) {
        return; // Skip dead creatures
    }
    int new_idx = contracted_creature_indices[idx];

    creature_x_save[new_idx] = creature_x[idx];
    creature_y_save[new_idx] = creature_y[idx];
    creature_energy_save[new_idx] = creature_energy[idx];
    for(int i = 0; i < 6 * 6; i++) {
        creature_matrix_save[new_idx + i * MAX_CREATURE_N] = creature_matrix[idx + i * MAX_CREATURE_N];
    }
    for(int i = 0; i < 6; i++) {
        creature_bias_save[new_idx + i * MAX_CREATURE_N] = creature_bias[idx + i * MAX_CREATURE_N];
    }
}
