#include "map/map.cuh"
#include "constants.h"
#include <fstream>

Map::Map() {

    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    h_data = new MapData;

    cudaMalloc(&h_data->food, bytes);
    cudaMalloc(&h_data->danger, bytes);
    cudaMalloc(&h_data->creature, bytes);

    cudaMalloc(&d_data, sizeof(MapData));
    cudaMemcpy(d_data, h_data, sizeof(MapData), cudaMemcpyHostToDevice);

    cudaMemset(h_data->food, 0, bytes);
    cudaMemset(h_data->danger, 0, bytes);
    cudaMemset(h_data->creature, 0, bytes);

    h_pinned = new MapData;
    cudaMallocHost(&h_pinned->food,     bytes);
    cudaMallocHost(&h_pinned->danger,   bytes);
    cudaMallocHost(&h_pinned->creature, bytes);
}

Map::~Map() {
    cudaFree(h_data->food);
    cudaFree(h_data->danger);
    cudaFree(h_data->creature);
    cudaFree(d_data);
    delete h_data;

    cudaFreeHost(h_pinned->food);
    cudaFreeHost(h_pinned->danger);
    cudaFreeHost(h_pinned->creature);
    delete h_pinned;
}

void Map::Save(int tick) {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    cudaMemcpy(h_pinned->food,     h_data->food,     bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->danger,   h_data->danger,   bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->creature, h_data->creature, bytes, cudaMemcpyDeviceToHost);

    char fname[64];
    snprintf(fname, sizeof(fname), "map_%06d.bin", tick);

    FILE* f = fopen(fname, "wb");
    
    int width = WIDTH, height = HEIGHT;
    fwrite(&width,  sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(h_pinned->food,     sizeof(float), WIDTH * HEIGHT, f);
    fwrite(h_pinned->danger,   sizeof(float), WIDTH * HEIGHT, f);
    fwrite(h_pinned->creature, sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);
}

void Map::refresh(curandState* random_states, int max_food_count) {
    cudaMemset(h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));
    place_food<<<(max_food_count + 255) / 256, 256>>>(d_data, max_food_count, random_states);
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

__global__ void place_food(MapData* map, int max_food_count, curandState* random_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_food_count) return;

    int rand_x = (int)(curand_uniform(&random_states[idx]) * WIDTH);
    int rand_y = (int)(curand_uniform(&random_states[idx]) * HEIGHT);

    map->food[get_cell_index(rand_x, rand_y)] = 1.0f;
}

__device__ float get_cell(MapData* map, int layer, int index) {
    switch (layer) {
        case 0: return map->food[index];
        case 1: return map->danger[index];
        case 2: return map->creature[index];
        default: return 0.0f; 
    }
}