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



    curandState* d_random_states;
    cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState));

    init_curand_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(d_random_states, 1234);

    cudaDeviceSynchronize();

    Creatures creatures1 = Creatures(d_random_states, 16);
    Creatures creatures2 = Creatures(d_random_states, 0);

    Creatures* current_creatures = &creatures1;
    Creatures* next_creatures = &creatures2;
    
    Map map = Map();

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

        // here we can copy map to host and save

        cudaDeviceSynchronize();

        std::cout << current_creatures->h_data->count << " " << current_creatures->action_types_counts[0] << " " << current_creatures->action_types_counts[1] << " " << current_creatures->action_types_counts[2] << " " << current_creatures->action_types_counts[3] << std::endl;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "ERROR: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "No errors found" << std::endl;
        }

        map.refresh(d_random_states, FOOD_SPAWN_QUANTITY);

        // here we can copy actions to host and save

        cudaDeviceSynchronize();

        current_creatures->RunActions(&map, d_random_states);

        t++;

        if (!(t % 100)) {
            contract(current_creatures, next_creatures);
            std::swap(current_creatures, next_creatures);
        }
    }

    cudaFree(d_random_states);

    return 0;
};