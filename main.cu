#include <iostream>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include <cstdlib>
#include <cstring>

#include "profiling/profiling.cuh"
#include <cuda_profiler_api.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <unistd.h>
#include "constants.h"
#include "creatures/creatures.cuh"
#include "map/map.cuh"
#include "constants.h"
#include "contract/contract.cuh"
#include <thread>
#include <chrono>

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

volatile std::sig_atomic_t interrupted = 0;

extern "C" void signal_handler(int signum) {
    interrupted = 1;
}

int main(int argc, char** argv) {
    cudaError cudaStatus;
    
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    cudaDeviceSynchronize();

    long long* global_id_counter = new long long;

    *global_id_counter = 0;

    Creatures creatures1 = Creatures(SEED, INITIAL_CREATURE_N, global_id_counter, SAVE_DIRECTORY + std::string("stream1.bin"));
    Creatures creatures2 = Creatures(SEED, 0, global_id_counter, SAVE_DIRECTORY + std::string("stream2.bin"));

    Creatures* current_creatures = &creatures1;
    Creatures* next_creatures = &creatures2;

    ContractWorkspace contract_workspace;
    
    Map map = Map(SAVE_DIRECTORY + std::string("map_stream.bin"));
    float* food_save = new float[WIDTH * HEIGHT];
    float* creature_save = new float[WIDTH * HEIGHT];
    float* danger_save = new float[WIDTH * HEIGHT];

    cudaDeviceSynchronize();

    map.refresh(derive_seed(SEED, 0), FOOD_SPAWN_QUANTITY * INITIAL_FOOD_MULTIPLIER);

    cudaStreamSynchronize(map.map_stream);

    current_creatures->RebuildCreatureMap(&map);
    
    cudaStreamSynchronize(current_creatures->compute_stream);

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(WIDTH, HEIGHT);
    #endif
    bool running = true;

    int t = 0;

    std::thread save_map_worker;


    cudaProfilerStart();

    while (running && t != MAX_TICKS) {

        unsigned long long seed = derive_seed(SEED, t + 54);
        float season_phase = 2.0f * 3.14159265358979323846f * t / SEASON_PERIOD;
        float season_cos = cosf(season_phase);
        float season_sin = sinf(season_phase);

        if  (30000 < t && t < 30100){
            map.transfer_thread = std::thread(&Map::Save, &map, t);
            map.transfer_thread = std::thread(&Map::Save, &map, t);
        } 

        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "ChooseAction");
        current_creatures->ChooseAction(&map, derive_seed(seed, 98423), season_cos, season_sin);
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

        if (map.transfer_thread.joinable()) {
            NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "WaitMapTransfer");

            auto start_time = std::chrono::high_resolution_clock::now();
            map.transfer_thread.join();
            auto end_time = std::chrono::high_resolution_clock::now();
            NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (duration.count() > 0) {
                std::cout << "Time spent waiting in map transfer: " << duration.count() << " ms, t = " << t << std::endl;
            }
        }

        #ifdef ENABLE_DISPLAY
        display.renderFrame(&map);

        if(display.shouldClose()){
            running = false;
        }

        #endif

        if (interrupted) {
            running = false;
        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        
        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "ClearDanger");
        cudaMemset(map.h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);
        
        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "RunActions");
        current_creatures->RunActions(&map, derive_seed(seed, 4096));
        cudaStreamSynchronize(current_creatures->compute_stream);
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "PlaceResources");
        float seasonal_factor = SEASON_OFFSET + SEASON_AMPLITUDE * season_sin;

        if (seasonal_factor < 0.0f) seasonal_factor = 0.0f;

        int food_this_tick = (int)roundf(FOOD_SPAWN_QUANTITY * seasonal_factor);
        int water_this_tick = (int)roundf(WATER_SPAWN_QUANTITY * seasonal_factor);

        if (food_this_tick > 0) {
            place_food<<<(food_this_tick + 255) / 256, 256, 0, map.map_stream>>>(
                map.d_data,
                food_this_tick,
                derive_seed(seed, 12345)
            );
        }

        if (water_this_tick > 0) {
            place_water<<<(water_this_tick + 255) / 256, 256, 0, map.map_stream>>>(
                map.d_data,
                water_this_tick,
                derive_seed(seed, 54321)
            );
        }

        cudaStreamSynchronize(map.map_stream);
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "SyncCreatureTransferStream");
        cudaStreamSynchronize(next_creatures->transfer_stream);
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

        if (current_creatures->count > 0 && t % CONTRACT_EVERY_N_TICKS == 0) {
            NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "Contract");

            if (CONTRACTION_TYPE == 0) {
                contract(current_creatures, next_creatures);
            } else if (CONTRACTION_TYPE == 1) {
                contract_optimized(current_creatures, next_creatures, &contract_workspace);
            } else if (CONTRACTION_TYPE == 2) {
                contract_optimized_split_copy(current_creatures, next_creatures, &contract_workspace);
            } else if (CONTRACTION_TYPE == 3) {
                contract_optimized_atomic(current_creatures, next_creatures, &contract_workspace);
            }

            std::swap(current_creatures, next_creatures);

            NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);
        }


        if (next_creatures->transfer_thread.joinable()) {
            NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "WaitCreatureTransfer");

            auto start_time = std::chrono::high_resolution_clock::now();
            next_creatures->transfer_thread.join();
            auto end_time = std::chrono::high_resolution_clock::now();
            NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "WaitCreatureTransfer");

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (duration.count() > 0) {
                std::cout << "Time spent waiting in creature transfer: " << duration.count() << " ms, t = " << t << std::endl;
            }
        }

        if (t >= SAVE_START_TICK && t < SAVE_END_TICK) {
            NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "StartCreatureSaveThread");
            next_creatures->transfer_thread = std::thread(&Creatures::Save, next_creatures, next_creatures->count - next_creatures->action_types_counts[REPRODUCE_ACTION], t, true, false, false);
            NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);
        }
        
        NVTX_PUSH_ENABLED(NSYS_PROFILING_ENABLED, "RebuildCreatureMap");
        current_creatures->RebuildCreatureMap(&map);
        cudaStreamSynchronize(current_creatures->compute_stream);
        NVTX_POP_ENABLED(NSYS_PROFILING_ENABLED);

        if (current_creatures->count == 0) {
            std::cout << "All creatures died. Ending simulation." << std::endl;
            running = false;
        }

        t++;
        if (t % 50 == 0) {
            std::cout << "Tick: " << t << ", Creatures: " << current_creatures->count << std::endl;
        }
    }

    cudaProfilerStop();

    free_contract_workspace(&contract_workspace);

    delete global_id_counter;

    return 0;
};