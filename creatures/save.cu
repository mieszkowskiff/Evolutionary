#include "creatures/creatures.cuh"
#include <stdio.h>

static void SaveMapAfterDamage(Map* map, int tick) {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    cudaMemcpy(map->h_pinned->food,     map->h_data->food,     bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->water,    map->h_data->water,    bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->danger,   map->h_data->danger,   bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(map->h_pinned->creature, map->h_data->creature, bytes, cudaMemcpyDeviceToHost);

    char fname[128];
    snprintf(fname, sizeof(fname), "save/map_%06d_after_damage.bin", tick);

    FILE* f = fopen(fname, "wb");

    int width = WIDTH, height = HEIGHT;
    fwrite(&width,  sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(map->h_pinned->food,     sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->danger,   sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->creature, sizeof(float), WIDTH * HEIGHT, f);
    fwrite(map->h_pinned->water,    sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);
}


void Creatures::Save_tick(int tick) {
    size_t bytes_count = count * sizeof(unsigned int);
    size_t bytes_energy = count * sizeof(float);
    size_t bytes_id = count * sizeof(long long);

    cudaMemcpy(h_pinned->x,      h_data->x,      bytes_count,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->y,      h_data->y,      bytes_count,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->energy, h_data->energy, bytes_energy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->water,  h_data->water,  bytes_energy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->ids,    h_data->ids,    bytes_id,     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->chosen_action, h_data->chosen_action, count * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_x, h_data->sensor_x, count * MILIEU_SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_y, h_data->sensor_y, count * MILIEU_SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->sensor_type, h_data->sensor_type, count * MILIEU_SENSORS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_x, h_data->action_x, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_y, h_data->action_y, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinned->action_type, h_data->action_type, count * ACTIONS_N * sizeof(int8_t), cudaMemcpyDeviceToHost);

    char fname[64];
    snprintf(fname, sizeof(fname), "save/creatures_%06d.bin", tick);
    FILE* f = fopen(fname, "wb");
    int sensors_n = MILIEU_SENSORS_N, actions_n = ACTIONS_N;
    fwrite(&count,     sizeof(int), 1, f);
    fwrite(&sensors_n, sizeof(int), 1, f);
    fwrite(&actions_n, sizeof(int), 1, f);
    fwrite(h_pinned->x,     sizeof(unsigned int), count, f);
    fwrite(h_pinned->y,     sizeof(unsigned int), count, f);
    fwrite(h_pinned->energy,sizeof(float),        count, f);
    fwrite(h_pinned->water, sizeof(float),        count, f);
    fwrite(h_pinned->ids,   sizeof(long long),    count, f);
    fwrite(h_pinned->chosen_action, sizeof(int8_t), count, f);
    fwrite(h_pinned->sensor_x, sizeof(int8_t), count * MILIEU_SENSORS_N, f);
    fwrite(h_pinned->sensor_y, sizeof(int8_t), count * MILIEU_SENSORS_N, f);
    fwrite(h_pinned->sensor_type, sizeof(int8_t), count * MILIEU_SENSORS_N, f);
    fwrite(h_pinned->action_x, sizeof(int8_t), count * ACTIONS_N, f);
    fwrite(h_pinned->action_y, sizeof(int8_t), count * ACTIONS_N, f);
    fwrite(h_pinned->action_type, sizeof(int8_t), count * ACTIONS_N, f);
    fclose(f);
}