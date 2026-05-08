# ifndef SENSORS_CUH
# define SENSORS_CUH

#include <cstdint>
#include <curand_kernel.h>

class Sensors {
    public:
    int8_t* sensor_x;
    int8_t* sensor_y;
    int8_t* sensor_type;

    Sensors(curandState* state, int creatures_n);
    ~Sensors();

    __device__ void AddRandomSensors(int creature_index, int sensor_index, curandState &state);
};

__global__ void SetSensors(Sensors* sensors, int creatures_n, curandState* state);


# endif