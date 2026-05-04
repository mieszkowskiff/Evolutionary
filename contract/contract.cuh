# ifndef CONTRACT_CUH
# define CONTRACT_CUH

__global__ void contract(
    unsigned int* creature_x,
    unsigned int* creature_y,
    float* creature_energy,

    int *creature_sensors_n,
    int *creature_hidden_neurons_n,

    float* creature_sensor_x,
    float* creature_sensor_y,
    float* creature_sensor_type,

    float* creature_hidden_matrix,
    float* creature_hidden_bias,
    float* creature_output_matrix,
    float* creature_output_bias,

    unsigned int* creature_x_save,
    unsigned int* creature_y_save,
    float* creature_energy_save,

    int* creature_sensors_n_save,
    int* creature_hidden_neurons_n_save,

    float* creature_sensor_x_save,
    float* creature_sensor_y_save,
    float* creature_sensor_type_save,

    float* creature_hidden_matrix_save,
    float* creature_hidden_bias_save,
    float* creature_output_matrix_save,
    float* creature_output_bias_save,

    int* contracted_creature_indices,
    int *creature_alive,
    int creature_n
);

# endif // CONTRACT_CUH