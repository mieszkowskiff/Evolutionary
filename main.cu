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
#include "action_step/action_step.cuh"
#include "contract/contract.cuh"
#include "initialization/initialization.cuh"

#ifdef ENABLE_DISPLAY
#include "display/renderer.h"
#endif

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \



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

    printf("Starting simulation...\n");

    // Map
    int map_width = 1024;
    int map_height = 1024;

    unsigned int* d_map; // d_map is considered as a bit vector (see CellFlags)
    CUDA_CHECK(cudaMalloc(&d_map, map_width * map_height * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_map, 0, map_width * map_height * sizeof(unsigned int)));

    // Creatures
    int* h_creatures_n = new int(INITIAL_CREATURE_N);
    int* d_creatures_n;
    CUDA_CHECK(cudaMalloc(&d_creatures_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_creatures_n, h_creatures_n, sizeof(int), cudaMemcpyHostToDevice));

    // we define everything twice in order to play ping-pong
    // features of creatures
    unsigned int* d_creature_x_alpha;
    unsigned int* d_creature_x_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_x_alpha, MAX_CREATURE_N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_x_beta, MAX_CREATURE_N * sizeof(unsigned int)));
    unsigned int* d_creature_x = d_creature_x_alpha; // ping-pong pointer
    unsigned int* d_creature_x_save = d_creature_x_beta; // for contraction step

    unsigned int* d_creature_y_alpha;
    unsigned int* d_creature_y_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_y_alpha, MAX_CREATURE_N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_creature_y_beta, MAX_CREATURE_N * sizeof(unsigned int)));
    unsigned int* d_creature_y = d_creature_y_alpha; // ping-pong pointer
    unsigned int* d_creature_y_save = d_creature_y_beta; // for contraction step

    float* d_creature_energy_alpha;
    float* d_creature_energy_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_energy_alpha, MAX_CREATURE_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_energy_beta, MAX_CREATURE_N * sizeof(float)));
    float* d_creature_energy = d_creature_energy_alpha; // ping-pong pointer
    float* d_creature_energy_save = d_creature_energy_beta; // for contraction step

    int* d_creature_sensor_x_alpha;
    int* d_creature_sensor_x_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_x_alpha, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_x_beta, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    int* d_creature_sensor_x = d_creature_sensor_x_alpha; // ping-pong pointer
    int* d_creature_sensor_x_save = d_creature_sensor_x_beta; // for contraction step

    int* d_creature_sensor_y_alpha;
    int* d_creature_sensor_y_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_y_alpha, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_y_beta, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    int* d_creature_sensor_y = d_creature_sensor_y_alpha; // ping-pong pointer
    int* d_creature_sensor_y_save = d_creature_sensor_y_beta; // for contraction step

    int *d_creature_sensor_type_alpha;
    int *d_creature_sensor_type_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_type_alpha, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_type_beta, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(int)));
    int* d_creature_sensor_type = d_creature_sensor_type_alpha; // ping-pong pointer
    int* d_creature_sensor_type_save = d_creature_sensor_type_beta; // for contraction step

    int *d_creature_sensors_n_alpha;
    int *d_creature_sensors_n_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_sensors_n_alpha, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_creature_sensors_n_beta, MAX_CREATURE_N * sizeof(int)));
    int* d_creature_sensors_n = d_creature_sensors_n_alpha; // ping-pong pointer
    int* d_creature_sensors_n_save = d_creature_sensors_n_beta; // for contraction step

    int *d_creature_hidden_neurons_n_alpha;
    int *d_creature_hidden_neurons_n_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_neurons_n_alpha, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_neurons_n_beta, MAX_CREATURE_N * sizeof(int)));
    int* d_creature_hidden_neurons_n = d_creature_hidden_neurons_n_alpha; // ping-pong pointer
    int* d_creature_hidden_neurons_n_save = d_creature_hidden_neurons_n_beta; // for contraction step

    float* d_creature_hidden_layer_matrix_alpha;
    float* d_creature_hidden_layer_matrix_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_layer_matrix_alpha, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * MAX_INPUT_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_layer_matrix_beta, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * MAX_INPUT_NEURONS * sizeof(float)));
    float* d_creature_hidden_layer_matrix = d_creature_hidden_layer_matrix_alpha; // ping-pong pointer
    float* d_creature_hidden_layer_matrix_save = d_creature_hidden_layer_matrix_beta; // for contraction step

    float* d_creature_output_layer_matrix_alpha;
    float* d_creature_output_layer_matrix_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_output_layer_matrix_alpha, MAX_CREATURE_N * OUTPUT_NEURONS * MAX_HIDDEN_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_output_layer_matrix_beta, MAX_CREATURE_N * OUTPUT_NEURONS * MAX_HIDDEN_NEURONS * sizeof(float)));
    float* d_creature_output_layer_matrix = d_creature_output_layer_matrix_alpha; // ping-pong pointer
    float* d_creature_output_layer_matrix_save = d_creature_output_layer_matrix_beta; // for contraction step

    float* d_creature_hidden_layer_bias_alpha;
    float* d_creature_hidden_layer_bias_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_layer_bias_alpha, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_layer_bias_beta, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * sizeof(float)));
    float* d_creature_hidden_layer_bias = d_creature_hidden_layer_bias_alpha; // ping-pong pointer
    float* d_creature_hidden_layer_bias_save = d_creature_hidden_layer_bias_beta; // for contraction step

    float* d_creature_output_layer_bias_alpha;
    float* d_creature_output_layer_bias_beta;
    CUDA_CHECK(cudaMalloc(&d_creature_output_layer_bias_alpha, MAX_CREATURE_N * OUTPUT_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_creature_output_layer_bias_beta, MAX_CREATURE_N * OUTPUT_NEURONS * sizeof(float)));
    float* d_creature_output_layer_bias = d_creature_output_layer_bias_alpha; // ping-pong pointer
    float* d_creature_output_layer_bias_save = d_creature_output_layer_bias_beta; // for contraction step

    // Sensor values
    float* d_creature_sensor_values;
    CUDA_CHECK(cudaMalloc(&d_creature_sensor_values, MAX_CREATURE_N * MAX_INPUT_NEURONS * sizeof(float)));

    // Hidden layer neuron values
    float * d_creature_hidden_layer_neuron_values;
    CUDA_CHECK(cudaMalloc(&d_creature_hidden_layer_neuron_values, MAX_CREATURE_N * MAX_HIDDEN_NEURONS * sizeof(float)));

    // Actions
    int* h_action_counts = new int[ACTIONS_N];
    int* d_action_counts;
    int* d_creatures_by_action;
    CUDA_CHECK(cudaMalloc(&d_creatures_by_action, MAX_CREATURE_N * ACTIONS_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_action_counts, ACTIONS_N * sizeof(int)));

    // For contraction step
    int* d_creature_alive;
    int* d_contracted_creature_indices;
    CUDA_CHECK(cudaMalloc(&d_creature_alive, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contracted_creature_indices, MAX_CREATURE_N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_creature_alive, 1, *h_creatures_n * sizeof(int)));

    
    // Randomness source
    curandState* d_random_states;
    CUDA_CHECK(cudaMalloc(&d_random_states, MAX_CREATURE_N * sizeof(curandState)));

    

    #ifdef ENABLE_DISPLAY
    // Displaying
    Renderer display(map_width, map_height);
    #endif

    initialize_random_states<<<(MAX_CREATURE_N + 255) / 256, 256>>>(
        d_random_states, 
        MAX_CREATURE_N
    );


    place_food<<<((int)(map_width * map_height * INITIAL_FOOD_SPAWN_RATE) + 255) / 256, 256>>>(
        d_map, 
        d_random_states, 
        map_width, 
        map_height,
        INITIAL_FOOD_SPAWN_RATE
    );

    initialize_creatures<<<(*h_creatures_n + 255) / 256, 256>>>(
        d_creature_x,
        d_creature_y,
        d_creature_energy,
        d_creature_sensors_n,
        d_creature_hidden_neurons_n,
        d_creature_sensor_x,
        d_creature_sensor_y,
        d_creature_sensor_type,
        d_random_states,
        *h_creatures_n,
        map_width,
        map_height
    );

    cudaDeviceSynchronize();

    //main loop
    bool running = true;
    int max_creature_in_simulation = 0;
    int t = 1;
    while (t < 100000 && running) {

        CUDA_CHECK(cudaMemset(d_action_counts, 0, 6 * sizeof(int)));


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

    sleep(20);

    CUDA_CHECK(cudaFree(d_map));

    CUDA_CHECK(cudaFree(d_creature_x_alpha));
    CUDA_CHECK(cudaFree(d_creature_x_beta));

    CUDA_CHECK(cudaFree(d_creature_y_alpha));
    CUDA_CHECK(cudaFree(d_creature_y_beta));


    CUDA_CHECK(cudaFree(d_creature_energy_alpha));
    CUDA_CHECK(cudaFree(d_creature_energy_beta));

    CUDA_CHECK(cudaFree(d_creature_hidden_layer_matrix_alpha));
    CUDA_CHECK(cudaFree(d_creature_hidden_layer_matrix_beta));
    CUDA_CHECK(cudaFree(d_creature_output_layer_matrix_alpha));
    CUDA_CHECK(cudaFree(d_creature_output_layer_matrix_beta));

    CUDA_CHECK(cudaFree(d_creature_sensors_n_alpha));
    CUDA_CHECK(cudaFree(d_creature_sensors_n_beta));
    CUDA_CHECK(cudaFree(d_creature_hidden_neurons_n_alpha));
    CUDA_CHECK(cudaFree(d_creature_hidden_neurons_n_beta));

    CUDA_CHECK(cudaFree(d_creature_hidden_layer_bias_alpha));
    CUDA_CHECK(cudaFree(d_creature_hidden_layer_bias_beta));
    CUDA_CHECK(cudaFree(d_creature_output_layer_bias_alpha));
    CUDA_CHECK(cudaFree(d_creature_output_layer_bias_beta));

    CUDA_CHECK(cudaFree(d_creature_sensor_x_alpha));
    CUDA_CHECK(cudaFree(d_creature_sensor_x_beta));
    CUDA_CHECK(cudaFree(d_creature_sensor_y_alpha));
    CUDA_CHECK(cudaFree(d_creature_sensor_y_beta));
    CUDA_CHECK(cudaFree(d_creature_sensor_type_alpha));
    CUDA_CHECK(cudaFree(d_creature_sensor_type_beta));

    CUDA_CHECK(cudaFree(d_creature_sensor_values));
    CUDA_CHECK(cudaFree(d_creature_hidden_layer_neuron_values));

    CUDA_CHECK(cudaFree(d_random_states));
    CUDA_CHECK(cudaFree(d_creatures_by_action));
    CUDA_CHECK(cudaFree(d_action_counts));
    CUDA_CHECK(cudaFree(d_creatures_n));
    CUDA_CHECK(cudaFree(d_creature_alive));
    CUDA_CHECK(cudaFree(d_contracted_creature_indices));

    //printf("Maximum creatures in simulation: %d\n", max_creature_in_simulation);
    return 0;
}