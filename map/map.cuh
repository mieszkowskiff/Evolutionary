# ifndef MAP_CUH
# define MAP_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

struct MapData {
    __nv_fp8_e4m3* food;
    __nv_fp8_e4m3* danger;
    __nv_fp8_e4m3* creature;
};



class Map {

    public:
    MapData* d_data;
    MapData* h_data;

    Map();
    ~Map();

    void remove_creatures_from_map();
};

__global__ void place_food(Map* map, int max_food_count, curandState* random_states);

__device__ int get_cell_index(int x, int y);

__device__ __nv_fp8_e4m3 get_cell(MapData* map, int layer, int index);

#endif