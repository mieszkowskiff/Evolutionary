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

struct RunConfig {
    int initial_creatures = INITIAL_CREATURE_N;
    unsigned long long seed = 1234ULL;
    int food_spawn_quantity = FOOD_SPAWN_QUANTITY;
    int initial_food_multiplier = INITIAL_FOOD_MULTIPLIER;
    int save_every = 1;
    int max_ticks = -1;
    int contract_every = 2;
    int save_creatures = 1;
    int nvtx = 0;
    int nsys_start_tick = -1;
    int nsys_end_tick = -1;
    int contract_mode = 0;
};

int parse_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    long v = std::strtol(value, &end, 10);

    if (end == value || *end != '\0') {
        std::cerr << "Invalid integer for " << name << ": " << value << std::endl;
        std::exit(2);
    }

    return static_cast<int>(v);
}

unsigned long long parse_ull_arg(const char* value, const char* name) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(value, &end, 10);

    if (end == value || *end != '\0') {
        std::cerr << "Invalid unsigned integer for " << name << ": " << value << std::endl;
        std::exit(2);
    }

    return v;
}

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --initial-creatures N\n"
        << "  --seed N\n"
        << "  --food-spawn-quantity N\n"
        << "  --initial-food-multiplier N\n"
        << "  --save-every N\n"
        << "  --max-ticks N\n"
        << "  --contract-every N\n"
        << "  --save-creatures 0|1\n"
        << "  --nvtx 0|1\n"
        << "  --nsys-start-tick N\n"
        << "  --nsys-end-tick N\n"
        << "  --contract-mode 0|1|2|3\n"
        << "  --help\n\n"
        << "Compile-time constants:\n"
        << "  WIDTH=" << WIDTH << "\n"
        << "  HEIGHT=" << HEIGHT << "\n"
        << "  MAX_CREATURE_N=" << MAX_CREATURE_N << "\n";
}

RunConfig parse_args(int argc, char** argv) {
    RunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        auto value_after = [&](const char* arg) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << arg << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(argv[i], "--initial-creatures") == 0) {
            cfg.initial_creatures = parse_int_arg(value_after(argv[i]), "--initial-creatures");
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            cfg.seed = parse_ull_arg(value_after(argv[i]), "--seed");
        } else if (std::strcmp(argv[i], "--food-spawn-quantity") == 0) {
            cfg.food_spawn_quantity = parse_int_arg(value_after(argv[i]), "--food-spawn-quantity");
        } else if (std::strcmp(argv[i], "--initial-food-multiplier") == 0) {
            cfg.initial_food_multiplier = parse_int_arg(value_after(argv[i]), "--initial-food-multiplier");
        } else if (std::strcmp(argv[i], "--save-every") == 0) {
            cfg.save_every = parse_int_arg(value_after(argv[i]), "--save-every");
        } else if (std::strcmp(argv[i], "--max-ticks") == 0) {
            cfg.max_ticks = parse_int_arg(value_after(argv[i]), "--max-ticks");
        } else if (std::strcmp(argv[i], "--contract-every") == 0) {
            cfg.contract_every = parse_int_arg(value_after(argv[i]), "--contract-every");
        } else if (std::strcmp(argv[i], "--save-creatures") == 0) {
            cfg.save_creatures = parse_int_arg(value_after(argv[i]), "--save-creatures");
        } else if (std::strcmp(argv[i], "--nvtx") == 0) {
            cfg.nvtx = parse_int_arg(value_after(argv[i]), "--nvtx");
        } else if (std::strcmp(argv[i], "--nsys-start-tick") == 0) {
            cfg.nsys_start_tick = parse_int_arg(value_after(argv[i]), "--nsys-start-tick");
        } else if (std::strcmp(argv[i], "--nsys-end-tick") == 0) {
            cfg.nsys_end_tick = parse_int_arg(value_after(argv[i]), "--nsys-end-tick");
        } else if (std::strcmp(argv[i], "--contract-mode") == 0) {
            cfg.contract_mode = parse_int_arg(value_after(argv[i]), "--contract-mode");
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            print_usage(argv[0]);
            std::exit(2);
        }
    }

    if (cfg.initial_creatures < 0 || cfg.initial_creatures > MAX_CREATURE_N) {
        std::cerr << "--initial-creatures must be between 0 and MAX_CREATURE_N" << std::endl;
        std::exit(2);
    }

    if (cfg.save_every < 0) {
        std::cerr << "--save-every must be >= 0" << std::endl;
        std::exit(2);
    }

    if (cfg.contract_every <= 0) {
        std::cerr << "--contract-every must be > 0" << std::endl;
        std::exit(2);
    }

    if (cfg.save_creatures != 0 && cfg.save_creatures != 1) {
        std::cerr << "--save-creatures must be 0 or 1" << std::endl;
        std::exit(2);
    }

    if (cfg.nvtx != 0 && cfg.nvtx != 1) {
        std::cerr << "--nvtx must be 0 or 1" << std::endl;
        std::exit(2);
    }

    if (cfg.nsys_start_tick < -1) {
        std::cerr << "--nsys-start-tick must be >= -1" << std::endl;
        std::exit(2);
    }

    if (cfg.nsys_end_tick < -1) {
        std::cerr << "--nsys-end-tick must be >= -1" << std::endl;
        std::exit(2);
    }

    if (cfg.nsys_start_tick >= 0 && cfg.nsys_end_tick >= 0 && cfg.nsys_end_tick < cfg.nsys_start_tick) {
        std::cerr << "--nsys-end-tick must be >= --nsys-start-tick" << std::endl;
        std::exit(2);
    }

    if (cfg.contract_mode < 0 || cfg.contract_mode > 3) {
        std::cerr << "--contract-mode must be 0, 1, 2 or 3" << std::endl;
        std::exit(2);
    }

    return cfg;
}

int main(int argc, char** argv) {
    cudaError cudaStatus;
    RunConfig cfg = parse_args(argc, argv);
    
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    cudaDeviceSynchronize();

    long long* global_id_counter = new long long;

    *global_id_counter = 0;

    Creatures creatures1 = Creatures(cfg.seed, cfg.initial_creatures, global_id_counter, "save/stream1.bin");
    Creatures creatures2 = Creatures(cfg.seed, 0, global_id_counter, "save/stream2.bin");

    Creatures* current_creatures = &creatures1;
    Creatures* next_creatures = &creatures2;

    ContractWorkspace contract_workspace;
    
    Map map = Map("save/map_stream.bin");
    float* food_save = new float[WIDTH * HEIGHT];
    float* creature_save = new float[WIDTH * HEIGHT];
    float* danger_save = new float[WIDTH * HEIGHT];

    cudaDeviceSynchronize();

    map.refresh(derive_seed(cfg.seed, 0), cfg.food_spawn_quantity * cfg.initial_food_multiplier);

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


    while (running && t != cfg.max_ticks) {
        if (cfg.nsys_start_tick >= 0 && t == cfg.nsys_start_tick) {
            cudaProfilerStart();
        }
        NVTX_PUSH_ENABLED(cfg.nvtx, "Tick");

        unsigned long long seed = derive_seed(cfg.seed, t + 54);
        float season_phase = 2.0f * 3.14159265358979323846f * t / SEASON_PERIOD;
        float season_cos = cosf(season_phase);
        float season_sin = sinf(season_phase);

        if  (30000 < t && t < 30100){
            map.transfer_thread = std::thread(&Map::Save, &map, t);
            map.transfer_thread = std::thread(&Map::Save, &map, t);
            NVTX_POP_ENABLED(cfg.nvtx);
        } 

        NVTX_PUSH_ENABLED(cfg.nvtx, "ChooseAction");
        current_creatures->ChooseAction(&map, derive_seed(seed, 98423), season_cos, season_sin);
        NVTX_POP_ENABLED(cfg.nvtx);

        if (map.transfer_thread.joinable()) {
            NVTX_PUSH_ENABLED(cfg.nvtx, "WaitMapTransfer");

            auto start_time = std::chrono::high_resolution_clock::now();
            map.transfer_thread.join();
            auto end_time = std::chrono::high_resolution_clock::now();
            NVTX_POP_ENABLED(cfg.nvtx);

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
        
        NVTX_PUSH_ENABLED(cfg.nvtx, "ClearDanger");
        cudaMemset(map.h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));
        NVTX_POP_ENABLED(cfg.nvtx);
        
        NVTX_PUSH_ENABLED(cfg.nvtx, "RunActions");
        current_creatures->RunActions(&map, derive_seed(seed, 4096));
        cudaStreamSynchronize(current_creatures->compute_stream);
        NVTX_POP_ENABLED(cfg.nvtx);

        NVTX_PUSH_ENABLED(cfg.nvtx, "PlaceResources");
        float seasonal_factor = SEASON_OFFSET + SEASON_AMPLITUDE * season_sin;

        if (seasonal_factor < 0.0f) seasonal_factor = 0.0f;

        int food_this_tick = (int)roundf(cfg.food_spawn_quantity * seasonal_factor);
        int water_this_tick = (int)roundf(cfg.food_spawn_quantity * seasonal_factor);

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
        NVTX_POP_ENABLED(cfg.nvtx);

        NVTX_PUSH_ENABLED(cfg.nvtx, "SyncCreatureTransferStream");
        cudaStreamSynchronize(next_creatures->transfer_stream);
        NVTX_POP_ENABLED(cfg.nvtx);

        if (current_creatures->count > 0) {
            NVTX_PUSH_ENABLED(cfg.nvtx, "Contract");

            if (cfg.contract_mode == 0) {
                contract(current_creatures, next_creatures);
            } else if (cfg.contract_mode == 1) {
                contract_optimized(current_creatures, next_creatures, &contract_workspace);
            } else if (cfg.contract_mode == 2) {
                contract_optimized_split_copy(current_creatures, next_creatures, &contract_workspace);
            } else if (cfg.contract_mode == 3) {
                contract_optimized_atomic(current_creatures, next_creatures, &contract_workspace);
            }

            std::swap(current_creatures, next_creatures);

            NVTX_POP_ENABLED(cfg.nvtx);
        }


        if (next_creatures->transfer_thread.joinable()) {
            NVTX_PUSH_ENABLED(cfg.nvtx, "WaitCreatureTransfer");

            auto start_time = std::chrono::high_resolution_clock::now();
            next_creatures->transfer_thread.join();
            auto end_time = std::chrono::high_resolution_clock::now();
            NVTX_PUSH_ENABLED(cfg.nvtx, "WaitCreatureTransfer");

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (duration.count() > 0) {
                std::cout << "Time spent waiting in creature transfer: " << duration.count() << " ms, t = " << t << std::endl;
            }
        }

        if (30000 < t && t < 30100) {
            NVTX_PUSH_ENABLED(cfg.nvtx, "StartCreatureSaveThread");
            next_creatures->transfer_thread = std::thread(&Creatures::Save, next_creatures, next_creatures->count - next_creatures->action_types_counts[REPRODUCE_ACTION], t, true, true, true);
            NVTX_POP_ENABLED(cfg.nvtx);
        }
        //else next_creatures->transfer_thread = std::thread(&Creatures::Save, next_creatures, next_creatures->count - next_creatures->action_types_counts[REPRODUCE_ACTION], t, false, false, false);
        
        NVTX_PUSH_ENABLED(cfg.nvtx, "RebuildCreatureMap");
        current_creatures->RebuildCreatureMap(&map);
        cudaStreamSynchronize(current_creatures->compute_stream);
        NVTX_POP_ENABLED(cfg.nvtx);

        if (current_creatures->count == 0) {
            std::cout << "All creatures died. Ending simulation." << std::endl;
            running = false;
        }

        t++;
        if (t % 50 == 0) {
            std::cout << "Tick: " << t << ", Creatures: " << current_creatures->count << std::endl;
        }
        if (t > 30100) running = false;

        NVTX_POP_ENABLED(cfg.nvtx);

        if (cfg.nsys_end_tick >= 0 && t == cfg.nsys_end_tick) {
            cudaProfilerStop();
        }
    }

    free_contract_workspace(&contract_workspace);

    delete global_id_counter;

    return 0;
};