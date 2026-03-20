# Artificial Life Simulation

This project implements a high-performance artificial life simulation. It was developed as part of the Evolutionary Algorithms and High Performance Computing courses during the 2025/2026 summer semester at the Faculty of Mathematics and Information Science, Warsaw University of Technology.

## Main Goal

The primary objective of this project is to model and observe emergent evolutionary behaviors and social phenomena within a highly optimized, GPU-accelerated artificial environment.

## Setup and Build Instructions

This project uses CMake and can be built in two different modes depending on your environment: **GUI Mode** (default, renders the simulation to a window) and **Headless Mode** (no graphical output, ideal for servers and compute clusters).

### Prerequisites

* **NVIDIA GPU** with a compatible Display Driver (e.g., version 535, 550, or 580+)
* **CUDA Toolkit** (version compatible with your driver, e.g., 12.x or 13.x)
* **CMake** (version 3.18 or higher)
* **C++ Compiler** (GCC or Clang compatible with your CUDA version)

*(For GUI Mode only)* **OpenGL Development Headers**:
On Ubuntu/Debian, install them via:
```bash
sudo apt update
sudo apt install libgl1-mesa-dev libglfw3-dev libglew-dev
```

---

### Option 1: Build with Graphical Display (Default)

This mode compiles the project with the CUDA-OpenGL Interoperability layer. It requires a local display attached to the NVIDIA GPU.

**1. Configure and build the project:**
```bash
mkdir build
cd build
cmake ..
make
```

**2. Run the simulation:**
```bash
./Evolutionary
```

### Option 2: Build in Headless Mode (No GUI)

This mode strips out all OpenGL, GLFW, and GLEW dependencies. It compiles only the core CUDA simulation logic. Use this mode if you are running the code on a remote server via SSH or benchmarking performance.

**1. Configure and build the project:**
Pass the `ENABLE_DISPLAY=OFF` flag to CMake.
```bash
mkdir build
cd build
cmake -DENABLE_DISPLAY=OFF ..
make
```

**2. Run the simulation:**
```bash
./Evolutionary
```