#include "utils/utils.cuh"


__device__ int get_cell_index(int x, int y, int map_width, int map_height) {
    int nx = x;
    int ny = y;

    if (nx < 0) {
        nx += map_width;
    } else if (nx >= map_width) {
        nx -= map_width;
    }

    if (ny < 0) {
        ny += map_height;
    } else if (ny >= map_height) {
        ny -= map_height;
    }

    return ny * map_width + nx;
}