# ifndef INITIALIZATION_CUH
# define INITIALIZATION_CUH

__global__ void initialize_random_states(
    curandState* random_states, 
    int num_states
);

__global__ void initialize_creatures(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,
    int* creature_sensors_n,
    int* creature_hidden_neurons_n,
    int* creature_sensor_x,
    int* creature_sensor_y,
    int* creature_sensor_type,
    curandState* random_states,
    int creature_n,
    int map_width,
    int map_height
);




#endif