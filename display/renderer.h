#pragma once
#include <cuda_runtime.h>
#include "map/map.cuh"

class Renderer {
public:
    Renderer(int width, int height);
    
    ~Renderer();

    bool shouldClose();

    void renderFrame(Map* map);

private:
    int width;
    int height;
    void* window;
    unsigned int pbo; 
    cudaGraphicsResource* cuda_pbo_resource;
};