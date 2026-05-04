# ifndef CREATURES_CUH
# define CREATURES_CUH

#include <curand_kernel.h>
#include <cuda_fp8.h>
#include "constants.h"
#include "creatures/sensors.cuh"
#include "creatures/networks.cuh"
#include "creatures/actions.cuh"


class Creatures {
    public:
    int count;
    unsigned int* x;
    unsigned int* y;
    __nv_fp8_e4m3* energy;

    Sensors sensors;
    Networks networks;
    Actions actions;

    

};


# endif