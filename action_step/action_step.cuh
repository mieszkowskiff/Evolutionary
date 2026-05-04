# ifndef ACTION_STEP_CUH
# define ACTION_STEP_CUH

__global__ void creature_action_step(
    unsigned int* creature_x,
    unsigned int* creature_y,
    int creature_n,
    int input_neurons_n,
    int hidden_neurons_n,
    float* creature_sensor_x,
    float* creature_sensor_y,
    float* creature_sensor_type,

    float* creature_hidden_matrix,
    float* creature_hidden_bias,
    float* creature_output_matrix,
    float* creature_output_bias,

    int map_width,
    int map_height,
    unsigned int* map,
    int* creature_by_actions,
    int* action_counts,
    int* creature_alive,
    curandState* random_states
);






# endif // ACTION_STEP_CUH