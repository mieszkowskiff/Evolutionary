#pragma once
#include <cuda_runtime.h>

class Renderer {
public:
    Renderer(int width, int height);
    
    ~Renderer();

    bool shouldClose();

    void renderFrame(unsigned int* d_logic_map);

private:
    int width;
    int height;
    void* window;
    unsigned int pbo; 
    cudaGraphicsResource* cuda_pbo_resource;
};