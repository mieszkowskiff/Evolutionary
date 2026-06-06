#include "map/map.cuh"
#include "constants.h"
#include <fstream>
#include <math.h>

Map::Map() {

    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    h_data = new MapData;

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

    h_pinned = new MapData;
    cudaMallocHost(&h_pinned->food,     bytes);
    cudaMallocHost(&h_pinned->water,    bytes);
    cudaMallocHost(&h_pinned->danger,   bytes);
    cudaMallocHost(&h_pinned->creature, bytes);
}

Map::~Map() {
    cudaFree(h_data->food);
    cudaFree(h_data->water);
    cudaFree(h_data->danger);
    cudaFree(h_data->creature);
    cudaFree(d_data);
    delete h_data;

    cudaFreeHost(h_pinned->food);
    cudaFreeHost(h_pinned->water);
    cudaFreeHost(h_pinned->danger);
    cudaFreeHost(h_pinned->creature);
    delete h_pinned;
}

void Map::Save(int tick) {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    cudaMemcpy(h_pinned->food,     h_data->food,     bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->water,    h_data->water,    bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->danger,   h_data->danger,   bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->creature, h_data->creature, bytes, cudaMemcpyDeviceToHost);

    char fname[64];
    snprintf(fname, sizeof(fname), "save/map_%06d.bin", tick);

    FILE* f = fopen(fname, "wb");
    
    int width = WIDTH, height = HEIGHT;
    fwrite(&width,  sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(h_pinned->food,     sizeof(float), WIDTH * HEIGHT, f);
    fwrite(h_pinned->danger,   sizeof(float), WIDTH * HEIGHT, f);
    fwrite(h_pinned->creature, sizeof(float), WIDTH * HEIGHT, f);
    fwrite(h_pinned->water,    sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);
}

void Map::refresh(curandState* random_states, int max_food_count) {
    cudaMemset(h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->food, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->water, 0, WIDTH * HEIGHT * sizeof(float));

    place_food<<<(max_food_count + 255) / 256, 256>>>(d_data, max_food_count, random_states);
    place_water<<<(max_food_count + 255) / 256, 256>>>(d_data, max_food_count, random_states);
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

__global__ void place_food(MapData* map, int max_food_count, curandState* random_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_food_count || idx >= MAX_CREATURE_N) return;

    curandState state = random_states[idx];

    int rand_x = (int)(curand_uniform(&state) * WIDTH);
    int rand_y;

    if (curand_uniform(&state) < RESOURCE_RANDOM_FRACTION) {
        rand_y = (int)(curand_uniform(&state) * HEIGHT);
    } else {
        int curve_id = curand(&state) % FOOD_CURVES_N;
        int curve_y = get_food_curve_y(rand_x, curve_id);
        rand_y = curve_y + (int)roundf(curand_normal(&state) * RESOURCE_CURVE_STDDEV);
    }

    map->food[get_cell_index(rand_x, rand_y)] = 1.0f;
    random_states[idx] = state;
}

__global__ void place_water(MapData* map, int max_water_count, curandState* random_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_water_count || idx >= MAX_CREATURE_N) return;

    curandState state = random_states[idx];

    int rand_x = (int)(curand_uniform(&state) * WIDTH);
    int rand_y;

    if (curand_uniform(&state) < RESOURCE_RANDOM_FRACTION) {
        rand_y = (int)(curand_uniform(&state) * HEIGHT);
    } else {
        int curve_id = curand(&state) % WATER_CURVES_N;
        int curve_y = get_water_curve_y(rand_x, curve_id);
        rand_y = curve_y + (int)roundf(curand_normal(&state) * RESOURCE_CURVE_STDDEV);
    }

    map->water[get_cell_index(rand_x, rand_y)] = 1.0f;
    random_states[idx] = state;
}

__device__ float get_cell(MapData* map, int layer, int index) {
    switch (layer) {
        case 0: return map->food[index];
        case 1: return map->danger[index];
        case 2: return map->creature[index];
        case 3: return map->water[index];
        default: return 0.0f; 
    }
}
