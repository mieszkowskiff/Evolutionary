cmake_minimum_required(VERSION 3.18)

project(Evolutionary LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(Evolutionary main.cu)

target_link_libraries(Evolutionary PRIVATE CUDA::curand)

target_compile_options(Evolutionary PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)