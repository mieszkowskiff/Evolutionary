#include <iostream>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include <cstdlib>
#include <cstring>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <unistd.h>
#include "constants.h"
#include "creatures/creatures.cuh"
#include "map/map.cuh"
#include "constants.h"
#include "contract/contract.cuh"
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
    int contract_every = 10;
    int save_creatures = 1;
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

    return cfg;
}

int main(int argc, char** argv) {
    RunConfig cfg = parse_args(argc, argv);
    
    struct sigaction action;
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);

    cudaError cudaStatus;

    cudaDeviceSynchronize();

    long long* global_id_counter = new long long;

    *global_id_counter = 0;

    Creatures creatures1 = Creatures(cfg.seed, cfg.initial_creatures, global_id_counter);
    Creatures creatures2 = Creatures(cfg.seed, 0, global_id_counter);

    Creatures* current_creatures = &creatures1;
    Creatures* next_creatures = &creatures2;
    
    Map map = Map();

    float* food_save = new float[WIDTH * HEIGHT];
    float* creature_save = new float[WIDTH * HEIGHT];
    float* danger_save = new float[WIDTH * HEIGHT];

    cudaDeviceSynchronize();

    map.refresh(derive_seed(cfg.seed, 0), cfg.food_spawn_quantity * cfg.initial_food_multiplier);

    cudaDeviceSynchronize();

    current_creatures->RebuildCreatureMap(&map);
    
    cudaDeviceSynchronize();

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(WIDTH, HEIGHT);
    #endif
    bool running = true;

    int t = 0;

    while (running && (cfg.max_ticks < 0 || t < cfg.max_ticks)) {

        unsigned long long seed = derive_seed(cfg.seed, t + 54);
        float season_phase = 2.0f * 3.14159265358979323846f * t / SEASON_PERIOD;
        float season_cos = cosf(season_phase);
        float season_sin = sinf(season_phase);

        current_creatures->ChooseAction(&map, derive_seed(seed, 98423), season_cos, season_sin);

        #ifdef ENABLE_DISPLAY
        display.renderFrame(&map);

        if(display.shouldClose()){
            running = false;
        }

        #endif

        if (interrupted) {
            running = false;
        }

        bool should_save = (cfg.save_every > 0) && (t % cfg.save_every == 0);

        // should_save
        if (should_save) {
            map.Save(t);
            cudaDeviceSynchronize();
        }

        std::cout << t << " " << *global_id_counter << " " << current_creatures->count << " " << current_creatures->action_types_counts[0] << " " << current_creatures->action_types_counts[1] << " " << current_creatures->action_types_counts[2] << " " << current_creatures->action_types_counts[3] << " " << current_creatures->action_types_counts[4] << " " << " kills= " << current_creatures->h_attack_damage_kills << std::endl;

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

        // cudaMemset(map.h_data->creature, 0, WIDTH * HEIGHT * sizeof(float));
        cudaMemset(map.h_data->danger, 0, WIDTH * HEIGHT * sizeof(float));

        if (should_save && cfg.save_creatures) {
            current_creatures->Save_tick(t);
            cudaDeviceSynchronize();
        }

        current_creatures->RunActions(&map, derive_seed(seed, 4096));

        cudaDeviceSynchronize();

        float seasonal_factor = SEASON_OFFSET + SEASON_AMPLITUDE * season_sin;

        if (seasonal_factor < 0.0f) seasonal_factor = 0.0f;

        int food_this_tick = (int)roundf(cfg.food_spawn_quantity * seasonal_factor);
        int water_this_tick = (int)roundf(cfg.food_spawn_quantity * seasonal_factor);

        if (food_this_tick > 0) {
            place_food<<<(food_this_tick + 255) / 256, 256>>>(
                map.d_data,
                food_this_tick,
                derive_seed(seed, 12345)
            );
        }

        if (water_this_tick > 0) {
            place_water<<<(water_this_tick + 255) / 256, 256>>>(
                map.d_data,
                water_this_tick,
                derive_seed(seed, 54321)
            );
        }

        cudaDeviceSynchronize();

        t++;

        if (!(t % cfg.contract_every)) {
            std::cout << "Contracting..." << std::endl;
            contract(current_creatures, next_creatures);
            std::swap(current_creatures, next_creatures);
            if (current_creatures->count == 0) {
                std::cout << "All creatures died. Ending simulation." << std::endl;
                running = false;
            }
        }

        current_creatures->RebuildCreatureMap(&map);
        cudaDeviceSynchronize();
    }

    delete global_id_counter;

    return 0;
};