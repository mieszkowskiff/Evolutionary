# ifndef MAP_CUH
# define MAP_CUH

#include <cuda_fp8.h>
#include <curand_kernel.h>

class Map {

    public:
    int width;
    int height;
    __nv_fp8_e4m3* food;
    __nv_fp8_e4m3* danger;
    __nv_fp8_e4m3* creature;

    Map(int width, int height);
    ~Map();

    __device__ int get_cell_index(int x, int y);

    __global__ void place_food(int max_food_count, curandState* random_states);

    __device__ __nv_fp8_e4m3 get_cell(int layer, int index) const;

    void remove_creatures_from_map();
};

#endif