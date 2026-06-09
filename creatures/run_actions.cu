#include "creatures/creatures.cuh"

void Creatures::RunActions(Map* map, unsigned long long seed) {
    if (action_types_counts[ATTACK_ACTION] > 0) d_AttackAction<<<(action_types_counts[ATTACK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    
    if (count > 0) d_ProcessEnergy<<<(count + 255) / 256, 256>>>(map->d_data, d_data, count, d_attack_damage_kills);

    if (action_types_counts[MOVE_ACTION] > 0) d_MoveAction<<<(action_types_counts[MOVE_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[EAT_ACTION] > 0) d_EatAction<<<(action_types_counts[EAT_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[DRINK_ACTION] > 0) d_DrinkAction<<<(action_types_counts[DRINK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    
    ReproduceAction(seed);

    cudaDeviceSynchronize();
}