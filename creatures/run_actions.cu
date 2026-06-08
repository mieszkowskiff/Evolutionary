#include "creatures/creatures.cuh"

void Creatures::RunActions(Map* map, unsigned long long seed) {
    static int action_tick = 0;

    h_attack_damage_kills = 0;
    cudaMemset(d_attack_damage_kills, 0, sizeof(unsigned int));

    if (action_types_counts[ATTACK_ACTION] > 0) d_AttackAction<<<(action_types_counts[ATTACK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    
    if (count > 0) d_ProcessEnergy<<<(count + 255) / 256, 256>>>(map->d_data, d_data, count, d_attack_damage_kills);

    cudaMemcpy(
        &h_attack_damage_kills,
        d_attack_damage_kills,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );

    if (SAVE_AFTER_DAMAGE_MAP && !(action_tick % SAVE_AFTER_DAMAGE_MAP_EVERY)) {
        cudaDeviceSynchronize();
        SaveMapAfterDamage(map, action_tick);
    }

    if (action_types_counts[MOVE_ACTION] > 0) d_MoveAction<<<(action_types_counts[MOVE_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[EAT_ACTION] > 0) d_EatAction<<<(action_types_counts[EAT_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    if (action_types_counts[DRINK_ACTION] > 0) d_DrinkAction<<<(action_types_counts[DRINK_ACTION] + 255) / 256, 256>>>(map->d_data, d_data);
    int reproduce_count = action_types_counts[REPRODUCE_ACTION];

    if (count + reproduce_count > MAX_CREATURE_N) reproduce_count = MAX_CREATURE_N - count;
    if (reproduce_count < 0) reproduce_count = 0;

    unsigned int h_successful_births = 0;
    cudaMemset(d_successful_births, 0, sizeof(unsigned int));

    int max_children = MAX_CREATURE_N - count;

    if (reproduce_count > 0) d_ReproduceAction<<<(reproduce_count + 255) / 256, 256>>>(map->d_data, d_data, derive_seed(seed, 4093), d_successful_births, *global_id_counter, count, max_children, reproduce_count);
    
    cudaMemcpy(&h_successful_births, d_successful_births, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    if (h_successful_births > (unsigned int)max_children) {
        h_successful_births = max_children;
    }
    
    *global_id_counter += h_successful_births;
    count += h_successful_births;

    action_tick++;

    cudaDeviceSynchronize();
}