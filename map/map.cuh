# ifndef MAP_CUH
# define MAP_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

class Map {

    public:
    __nv_fp8_e4m3* food;
    __nv_fp8_e4m3* danger;
    __nv_fp8_e4m3* creature;

    Map();
    ~Map();

    __device__ int get_cell_index(int x, int y);

    __device__ __nv_fp8_e4m3 get_cell(int layer, int index) const;

    void remove_creatures_from_map();
};

__global__ void place_food(Map* map, int max_food_count, curandState* random_states);

#endif