#include "renderer.h"
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

enum CellFlags { BIT_FOOD = 0, BIT_CREATURE = 1 };

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

__global__ void render_kernel(MapData* map_data, unsigned int* pbo_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = get_cell_index(x, y);
        float food = map_data->food[idx];
        float creature = map_data->creature[idx];
        float danger = map_data->danger[idx];
        float water = map_data->water[idx];

        unsigned char r = 255, g = 255, b = 255, a = 255;

        if (creature > 0) {r = 255, g = 0, b = 0, a = 255;}
        if (food > 0) {r = 0, g = 255, b = 0, a = 255;}
        //if (danger > 0) {r = 255;}
        if (water > 0) {r = 0, g = 0, b = 255, a = 255;}

        pbo_out[idx] = (a << 24) | (b << 16) | (g << 8) | r;
    }
}

Renderer::Renderer(int w, int h) : width(w), height(h) {
    if (!glfwInit()) exit(-1);
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    GLFWwindow* win = glfwCreateWindow(width, height, "Evolutionary", NULL, NULL);
    if (!win) { glfwTerminate(); exit(-1); }
    glfwMakeContextCurrent(win);
    this->window = (void*)win;

    // --- NOWE: GLEW musi mieć włączony tryb eksperymentalny na profilu Core ---
    glewExperimental = GL_TRUE;
    // --------------------------------------------------------------------------

    if (glewInit() != GLEW_OK) exit(-1);
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

Renderer::~Renderer() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glfwDestroyWindow((GLFWwindow*)window);
    glfwTerminate();
}

bool Renderer::shouldClose() {
    return glfwWindowShouldClose((GLFWwindow*)window);
}

void Renderer::renderFrame(Map* map) {
    unsigned int* d_pbo_ptr;
    size_t num_bytes;
    
    gpuErrchk(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, cuda_pbo_resource));

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    render_kernel<<<blocks, threads>>>(map->d_data, d_pbo_ptr, width, height);

    gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers((GLFWwindow*)window);
    glfwPollEvents();
}