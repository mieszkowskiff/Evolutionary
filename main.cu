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

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

volatile std::sig_atomic_t interrupted = 0;

extern "C" void signal_handler(int signum) {
    interrupted = 1;
}

int main() {
    curandState* d_random_states;
    cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState));

    init_curand_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(d_random_states, 1234);

    cudaDeviceSynchronize();

    Creatures creatures = Creatures(d_random_states, MAX_CREATURE_N);
    
    Map map = Map();

    cudaDeviceSynchronize();

    creatures.ActionStep(&map, d_random_states);

    cudaDeviceSynchronize();

    cudaFree(d_random_states);

    std::cout << creatures.action_types_counts[0] << " " << creatures.action_types_counts[1] << " " << creatures.action_types_counts[2] << " " << creatures.action_types_counts[3] << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "ERROR: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "No errors found" << std::endl;
    }

    return 0;
};


//         cudaDeviceSynchronize();
//         CUDA_CHECK(cudaMemcpy(h_action_counts, d_action_counts, 6 * sizeof(int), cudaMemcpyDeviceToHost));
//         cudaDeviceSynchronize();
//         // std::cout << "All creatures: " << *h_creatures_n << " Action counts: ";
//         // for (int i = 0; i < 6; i++) {
//         //     std::cout << h_action_counts[i] << " ";
//         // }curandState* d_random_states;
//     CUDA_CHECK(cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState)));

//     #ifdef ENABLE_DISPLAY
//     // Displaying
//     Renderer display(map_width, map_height);
//     #endif

//     printf("Initializing random states...\n");
//     initialize_random_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(
//         d_random_states, 
//         MAX_CREATURE_N
//     );
    
//     cudaDeviceSynchronize();

//     printf("Initializing map...\n");


//     place_food<<<((int)(map_width * map_height * INITIAL_FOOD_SPAWN_RATE) + 255) / 256, 256>>>(
//         d_map, 
//         d_random_states, 
//         map_width, 
//         map_height,
//         INITIAL_FOOD_SPAWN_RATE
//     );

//     cudaDeviceSynchronize();

//     printf("Initializing creatures...\n");
//     initialize_creatures<<<(*h_creatures_n + 255) / 256, 256>>>(
//     d_creature_x,
//     d_creature_y,
//     d_creature_energy,
//     d_creature_sensors_n,
//     d_creature_hidden_neurons_n,
//     d_creature_sensor_x,
//     d_creature_sensor_y,
//     d_creature_sensor_type,
//     d_random_states,
//     *h_creatures_n,
//     map_width,
//     map_height
// );

//     cudaDeviceSynchronize();

//     //main loop
//     bool running = true;
//     int max_creature_in_simulation = 0;
//     int t = 1;
//     while (t < 100000 && running) {

//         CUDA_CHECK(cudaMemset(d_action_counts, 0, 6 * sizeof(int)));
//         size_t shared_memory_size = 256 * 6 * 2 * sizeof(float); // 256 threads, 6 inputs + 6 outputs per creature
//         creature_action_step<<<(*h_creatures_n + 255) / 256, 256, shared_memory_size>>>(
//             d_creature_x,
//             d_creature_y,
//             d_creature_energy,
//             *h_creatures_n,
//             map_width,
//             map_height,
//             d_map,
//             d_creature_matrix,
//             d_creature_bias,
//             d_creatures_by_action,
//             d_action_counts,
//             d_creature_alive,
//             d_random_states
//         );


//         cudaDeviceSynchronize();
//         CUDA_CH
//         // std::cout << std::endl;

//         if (h_action_counts[2] > 0) {
//             move_right<<<(h_action_counts[2] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 map_width,
//                 map_height,
//                 d_creatures_by_action,
//                 d_action_counts
//             );
//         }

//         if (h_action_counts[3] > 0) {
//             move_left<<<(h_action_counts[3] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 map_width,
//                 map_height,
//                 d_creatures_by_action,
//                 d_action_counts
//             );
//         }

//         if (h_action_counts[4] > 0) {
//             move_down<<<(h_action_counts[4] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 map_width,
//                 map_height,
//                 d_creatures_by_action,
//                 d_action_counts
//             );
//         }

//         if (h_action_counts[5] > 0) {
//             move_up<<<(h_action_counts[5] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 map_width,
//                 map_height,
//                 d_creatures_by_action,
//                 d_action_counts
//             );
//         }

//         if (h_action_counts[0] > 0) {
//             eat_food<<<(h_action_counts[0] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 d_creature_energy,
//                 map_width,
//                 map_height,
//                 d_map,
//                 d_creatures_by_action,
//                 d_action_counts
//             );
//         }

//         if (h_action_counts[1] > 0) {
//             reproduce<<<(h_action_counts[1] + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 d_creature_energy,
//                 d_creatures_n,
//                 map_width,
//                 map_height,
//                 d_map,
//                 d_creature_matrix,
//                 d_creature_bias,
//                 d_creatures_by_action,
//                 d_action_counts,
//                 d_creature_alive,
//                 d_random_states
//             );
//         }


//         cudaDeviceSynchronize();

//         CUDA_CHECK(cudaMemcpy(h_creatures_n, d_creatures_n, sizeof(int), cudaMemcpyDeviceToHost));
//         if (*h_creatures_n > max_creature_in_simulation) {
//             max_creature_in_simulation = *h_creatures_n;
//         }

//         // for display purposes
//         remove_creatures_from_map<<<(map_width * map_height + 255) / 256, 256>>>(
//             d_map,
//             map_width,
//             map_height
//         );

//         cudaDeviceSynchronize();

//         place_creatures_on_map<<<(*h_creatures_n + 255) / 256, 256>>>(
//             d_map,
//             d_creature_x,
//             d_creature_y,
//             *h_creatures_n,
//             map_width,
//             map_height,
//             d_creature_energy,
//             d_creature_alive
//         );

//         place_food<<<((int)(map_width * map_height * FOOD_SPAWN_RATE) + 255) / 256, 256>>>(
//             d_map,
//             d_random_states,
//             map_width,
//             map_height,
//             FOOD_SPAWN_RATE
//         );

//         cudaDeviceSynchronize();

//         if (*h_creatures_n == 0) {
//             printf("All creatures died. Ending simulation.\n");
//             break;
//         }
//         if (!(t % 5)) {
            
//             thrust::device_ptr<int> dev_flags(d_creature_alive);
//             thrust::device_ptr<int> dev_indices(d_contracted_creature_indices);
//             thrust::exclusive_scan(thrust::device, dev_flags, dev_flags + *h_creatures_n, dev_indices);


//             cudaDeviceSynchronize();

//             contract<<<(*h_creatures_n + 255) / 256, 256>>>(
//                 d_creature_x,
//                 d_creature_y,
//                 d_creature_energy,
//                 d_creature_matrix,
//                 d_creature_bias,
//                 d_creature_x_save,
//                 d_creature_y_save,
//                 d_creature_energy_save,
//                 d_creature_matrix_save,
//                 d_creature_bias_save,
//                 d_contracted_creature_indices,
//                 d_creature_alive,
//                 *h_creatures_n
//             );


//             int* last_creature_alive = new int[1];

//             CUDA_CHECK(cudaMemcpy(last_creature_alive, &d_creature_alive[*h_creatures_n - 1], sizeof(int), cudaMemcpyDeviceToHost));
//             *h_creatures_n = dev_indices[*h_creatures_n - 1] + last_creature_alive[0];
//             CUDA_CHECK(cudaMemcpy(d_creatures_n, h_creatures_n, sizeof(int), cudaMemcpyHostToDevice));

//             std::swap(d_creature_x, d_creature_x_save);
//             std::swap(d_creature_y, d_creature_y_save);
//             std::swap(d_creature_energy, d_creature_energy_save);
//             std::swap(d_creature_matrix, d_creature_matrix_save);
//             std::swap(d_creature_bias, d_creature_bias_save);

//             cudaDeviceSynchronize();
//         }

//         #ifdef ENABLE_DISPLAY
//         display.renderFrame(d_map);

//         if(display.shouldClose()){
//             running = false;
//         }

//         cudaDeviceSynchronize();
//         #endif

//         if (interrupted) {
//             running = false;
//         }

//         t++;
//     }

//     printf("\nQuitting the simulation. Shutting down...\n");
//     fflush(stdout);
