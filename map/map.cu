#include "map/map.cuh"
#include "constants.h"
#include <fstream>
#include <math.h>

Map::Map() {

    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    h_data = new MapData;

    h_data->season_sin = 0.0f;
    h_data->season_cos = 1.0f;

    cudaMalloc(&h_data->food, bytes);
    cudaMalloc(&h_data->water, bytes);
    cudaMalloc(&h_data->danger, bytes);
    cudaMalloc(&h_data->creature, bytes);

    cudaMalloc(&d_data, sizeof(MapData));
    cudaMemcpy(d_data, h_data, sizeof(MapData), cudaMemcpyHostToDevice);

    cudaMemset(h_data->food, 0, bytes);
    cudaMemset(h_data->water, 0, bytes);
    cudaMemset(h_data->danger, 0, bytes);
    cudaMemset(h_data->creature, 0, bytes);
}

Map::~Map() {
    cudaFree(h_data->food);
    cudaFree(h_data->water);
    cudaFree(h_data->danger);
    cudaFree(h_data->creature);
    cudaFree(d_data);
    delete h_data;
}

void Map::refresh(unsigned long long seed, int max_food_count) {
    cudaMemset(h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->food, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->water, 0, WIDTH * HEIGHT * sizeof(float));

    place_food<<<(max_food_count + 255) / 256, 256>>>(d_data, max_food_count, derive_seed(seed, 67890));
    place_water<<<(max_food_count + 255) / 256, 256>>>(d_data, max_food_count, derive_seed(seed, 54321));
}

__device__ int get_cell_index(int x, int y) {
    int nx = x;
    int ny = y;

    if (nx < 0) {
        nx += WIDTH;
    } else if (nx >= WIDTH) {
        nx -= WIDTH;
    }

    if (ny < 0) {
        ny += HEIGHT;
    } else if (ny >= HEIGHT) {
        ny -= HEIGHT;
    }

    return ny * WIDTH + nx;
}

__device__ int get_food_curve_y(int x, int curve_id) {
    float xf = (float)x;
    float center = xf - 0.5f * WIDTH;
    float y = 0.0f;

    switch (curve_id % FOOD_CURVES_N) {
        case 0:
            y = 0.30f * HEIGHT + 30.0f * sinf(2.0f * 3.14159265358979323846f * xf / WIDTH);
            break;
        case 1:
            y = 0.62f * HEIGHT + 22.0f * sinf(4.0f * 3.14159265358979323846f * xf / WIDTH + 0.8f);
            break;
        case 2:
            y = 20.0f + 0.0030f * center * center;
            break;
        default:
            y = HEIGHT - 35.0f - 0.0025f * center * center;
            break;
    }

    int iy = (int)roundf(y);
    return get_cell_index(x, iy) / WIDTH;
}

__device__ int get_water_curve_y(int x, int curve_id) {
    float xf = (float)x;
    float center = xf - 0.5f * WIDTH;
    float y = 0.0f;

    switch (curve_id % WATER_CURVES_N) {
        case 0:
            y = 0.48f * HEIGHT + 35.0f * sinf(2.0f * 3.14159265358979323846f * xf / WIDTH + 1.7f);
            break;
        case 1:
            y = 0.78f * HEIGHT + 18.0f * sinf(6.0f * 3.14159265358979323846f * xf / WIDTH);
            break;
        case 2:
            y = 70.0f + 0.0020f * center * center;
            break;
        default:
            y = HEIGHT - 75.0f - 0.0032f * center * center;
            break;
    }

    int iy = (int)roundf(y);
    return get_cell_index(x, iy) / WIDTH;
}

__global__ void place_food(MapData* map, int max_food_count, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_food_count || idx >= MAX_CREATURE_N) return;

    
    unsigned long long local_seed = derive_seed(seed, idx);

    int rand_x = rand_int(derive_seed(local_seed, 12345), WIDTH);
    int rand_y;

    if (rand_float(derive_seed(local_seed, 98765)) < RESOURCE_RANDOM_FRACTION) {
        rand_y = rand_int(derive_seed(local_seed, 54321), HEIGHT);
    } else {
        int curve_id = rand_int(derive_seed(local_seed, 67890), FOOD_CURVES_N);
        int curve_y = get_food_curve_y(rand_x, curve_id);
        rand_y = rand_normal(derive_seed(local_seed, 13579), curve_y, RESOURCE_CURVE_STDDEV);
    }

    map->food[get_cell_index(rand_x, rand_y)] = 1.0f;
}

__global__ void place_water(MapData* map, int max_water_count, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_water_count || idx >= MAX_CREATURE_N) return;

    unsigned long long local_seed = derive_seed(seed, idx);

    int rand_x = rand_int(derive_seed(local_seed, 54321), WIDTH);
    int rand_y;

    if (rand_float(derive_seed(local_seed, 98765)) < RESOURCE_RANDOM_FRACTION) {
        rand_y = rand_int(derive_seed(local_seed, 54321), HEIGHT);
    } else {
        int curve_id = rand_int(derive_seed(local_seed, 67890), WATER_CURVES_N);
        int curve_y = get_water_curve_y(rand_x, curve_id);
        rand_y = rand_normal(derive_seed(local_seed, 13579), curve_y, RESOURCE_CURVE_STDDEV);
    }

    map->water[get_cell_index(rand_x, rand_y)] = 1.0f;
}

