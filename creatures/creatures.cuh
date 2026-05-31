#ifndef CREATURES_CUH
# define CREATURES_CUH

#include <curand_kernel.h>
#include <cuda_fp8.h>
#include "constants.h"
#include "map/map.cuh"

struct CreatureData {

    //General data
    unsigned int* x;
    unsigned int* y;
    float* energy;
    float* water;
    long long* ids;
    int* age;

    // Sensors data
    int8_t* sensor_x;
    int8_t* sensor_y;
    int8_t* sensor_type;

    //Network data
    __nv_fp8_e4m3* first_matrix;
    __nv_fp8_e4m3* second_matrix;
    __nv_fp8_e4m3* bias;

    // Action data
    int8_t* action_x;
    int8_t* action_y;
    int8_t* action_type;

    int8_t* chosen_action;

    unsigned int* move_queue_creatures;
    unsigned int* eat_queue_creatures;
    unsigned int* attack_queue_creatures;
    unsigned int* reproduce_queue_creatures;
    unsigned int* drink_queue_creatures;

    int8_t* move_queue_actions;
    int8_t* eat_queue_actions;
    int8_t* attack_queue_actions;
    int8_t* reproduce_queue_actions;
    int8_t* drink_queue_actions;

    unsigned int* action_types_counts;
};

class Creatures {
    public:

    int count;
    long long* global_id_counter;
    unsigned int* action_types_counts;
    unsigned int* d_successful_births;
    unsigned int* d_attack_damage_kills;
    unsigned int h_attack_damage_kills;
    
    CreatureData* d_data;
    CreatureData* h_data;

    CreatureData* h_pinned;
    
    Creatures(curandState* state, int count, long long *global_id_counter);
    ~Creatures();

    void ChooseAction(Map* map, curandState* random_states, float season_cos, float season_sin);    

    void RebuildCreatureMap(Map* map);

    void RunActions(Map* map, curandState* random_states);

    void Save_tick(int tick);
};

__device__ size_t get_second_matrix_idx(int creature_idx, int output_idx, int hidden_idx);

__device__ size_t get_first_matrix_idx(int creature_idx, int hidden_idx, int sensor_idx);

__global__ void InitializeRandomCreatures(CreatureData* d_data, int count, curandState* states, long long global_id_counter);

__global__ void d_ActionStep(MapData* d_map, CreatureData* d_creatures, curandState* random_states, int count, float season_cos, float season_sin);

__device__ void AddRandomSensors(CreatureData* creatures, int creature_index, int sensor_index, curandState& state);

__device__ void AddRandomNetwork(CreatureData* creatures, int creature_index, curandState &state);

__device__ void SetRandomAction(CreatureData* creatures, int creature_index, int action_index, curandState& state);


__global__ void d_MoveAction(MapData* d_map, CreatureData* d_creatures);
__global__ void d_EatAction(MapData* d_map, CreatureData* d_creatures);
__global__ void d_AttackAction(MapData* d_map, CreatureData* d_creatures);
__global__ void d_DrinkAction(MapData* d_map, CreatureData* d_creatures);
__global__ void d_ReproduceAction(MapData* d_map, CreatureData* d_creatures, curandState* random_states, unsigned int* d_successful_births, long long global_id_counter, int count,  int max_children, int reproduce_count);
__global__ void d_ProcessEnergy(MapData* d_map, CreatureData* d_creatures, int count, unsigned int* d_attack_damage_kills);

__device__ void reproduce_creature(CreatureData* d_creatures, int parent_creature_index, int new_creature_idx, curandState& state, long long new_id);

__global__ void d_RebuildCreatureMap(MapData* d_map, CreatureData* d_creatures, int count);
# endif