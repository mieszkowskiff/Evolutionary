#include <iostream>
#include <curand_kernel.h>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <unistd.h>
#include "constants.h"
#include "creatures/creatures.cuh"
#include "map/map.cuh"
#include "randomness/randomness.cuh"
#include "constants.h"
#include "contract/contract.cuh"
#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

volatile std::sig_atomic_t interrupted = 0;

extern "C" void signal_handler(int signum) {
    interrupted = 1;
}

int main() {
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    cudaError cudaStatus;
 

    curandState* d_random_states;
    cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState));

    init_curand_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(d_random_states, 1234);

    cudaDeviceSynchronize();

    long long* global_id_counter = new long long;

    *global_id_counter = 0;

    Creatures creatures1 = Creatures(d_random_states, 4096, global_id_counter);
    Creatures creatures2 = Creatures(d_random_states, 0, global_id_counter);

    Creatures* current_creatures = &creatures1;
    Creatures* next_creatures = &creatures2;
    
    Map map = Map();

    float* food_save = new float[WIDTH * HEIGHT];
    float* creature_save = new float[WIDTH * HEIGHT];
    float* danger_save = new float[WIDTH * HEIGHT];

    cudaDeviceSynchronize();

    map.refresh(d_random_states, FOOD_SPAWN_QUANTITY * 128);

    cudaDeviceSynchronize();

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(WIDTH, HEIGHT);
    #endif
    bool running = true;

    int t = 0;

    while (running) {

        current_creatures->ChooseAction(&map, d_random_states);

        #ifdef ENABLE_DISPLAY
        display.renderFrame(&map);

        if(display.shouldClose()){
            running = false;
        }

        #endif

        if (interrupted) {
            running = false;
        }

        map.Save(t);

        cudaDeviceSynchronize();

        std::cout << t << " " << *global_id_counter << " " << current_creatures->count << " " << current_creatures->action_types_counts[0] << " " << current_creatures->action_types_counts[1] << " " << current_creatures->action_types_counts[2] << " " << current_creatures->action_types_counts[3] << std::endl;

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

        cudaMemset(map.h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));
        cudaMemset(map.h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));

        current_creatures->Save_tick(t);

        cudaDeviceSynchronize();

        current_creatures->RunActions(&map, d_random_states);

        cudaDeviceSynchronize();

        place_food<<<(FOOD_SPAWN_QUANTITY + 255) / 256, 256>>>(map.d_data, FOOD_SPAWN_QUANTITY, d_random_states);


        t++;

        if (!(t % 10)) {
            std::cout << "Contracting..." << std::endl;
            contract(current_creatures, next_creatures);
            std::swap(current_creatures, next_creatures);
            if (current_creatures->count == 0) {
                std::cout << "All creatures died. Ending simulation." << std::endl;
                running = false;
            }
        }
    }

    cudaFree(d_random_states);
    delete global_id_counter;

    return 0;
};