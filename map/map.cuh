# ifndef MAP_CUH
# define MAP_CUH

#include <cuda_fp8.h>
#include "random/random.cuh"

struct MapData {
    float* food;
    float* water;
    float* danger;
    float* creature;

    float season_sin;
    float season_cos;
};



class Map {

    public:
    MapData* d_data;
    MapData* h_data;

    Map();
    ~Map();

    void Save(int tick);

    void remove_creatures_from_map();

    void refresh(unsigned long long seed, int max_food_count);
};

__global__ void place_food(MapData* map, int max_food_count, unsigned long long seed);
__global__ void place_water(MapData* map, int max_water_count, unsigned long long seed);

__device__ int get_cell_index(int x, int y);

__device__ float get_cell(MapData* map, int layer, int index);

#endif
