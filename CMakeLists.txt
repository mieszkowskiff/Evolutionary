cmake_minimum_required(VERSION 3.18)

project(Evolutionary LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89-real 89-virtual)

find_package(CUDAToolkit REQUIRED)


option(ENABLE_DISPLAY "Build with OpenGL graphical display" ON)

if(ENABLE_DISPLAY)
    message(STATUS "GUI Display is ENABLED. Looking for OpenGL libraries...")
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)

    add_executable(Evolutionary main.cu display/renderer.cu)

    target_link_libraries(Evolutionary PRIVATE 
        CUDA::cudart
        CUDA::curand
        OpenGL::GL
        glfw
        GLEW::GLEW
    )
    
    target_compile_definitions(Evolutionary PRIVATE ENABLE_DISPLAY)
else()
    message(STATUS "GUI Display is DISABLED (Headless mode). Skipping OpenGL.")
    
    add_executable(Evolutionary main.cu)

    target_link_libraries(Evolutionary PRIVATE 
        CUDA::cudart
        CUDA::curand
    )
endif()

target_include_directories(Evolutionary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/display)
target_compile_options(Evolutionary PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)