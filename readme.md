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

## Simulation Rules

**The Map and Environment**
The simulation takes place on a flat, discrete square plane, with its edges connected to form a torus shape. Each field on the map can simultaneously contain one or more agents, a danger level, as well as independent resources of food and water. The appearance of these resources is irregular in both space and time. The amount of food and water added at simulation step $t$ is calculated using the formula $round(Q\cdot(S_{o}+S_{a}\cdot \sin(\frac{2\pi t}{P})))$. In this formula, the base resource quantity is 1024, the constant season offset is 1, the seasonal amplitude is 0.5, and the season period is 500. Furthermore, resources are highly concentrated spatially, as 90% of them appear along long, narrow areas known as rivers. The intersections of these rivers create resource-rich areas referred to as oases.

**Agents and Anatomy**
Every agent is defined by its sensors, actions, and a neural network serving as a decision-making center. None of these components, including the neural network weights, change during the agent's lifetime. Agents possess 62 external sensors, each defined by a specific type and relative coordinates targeting a particular field. These external sensors can detect agents, food, water, and danger. Additionally, agents have two internal, non-mutating sensors that read the sine and cosine of the current season. Agents can choose from 32 available actions, which include moving, consuming food, consuming water, placing danger, and reproducing. The decision-making process relies on a built-in neural network featuring two dense layers, including a hidden layer of size 32. The network strictly maps 64 input neurons to output neurons corresponding to the available actions, utilizing a softmax function on the output layer to determine the probability of executing a specific action.

**Survival and Combat**
Survival requires managing internal food and water levels, both of which decrease by 0.02 every simulation step. If either level drops to 0, the agent dies. Agents can replenish these levels by consuming resources from their current field. An agent can also engage in combat by executing a fight action, which places 3 units of danger on a targeted field. If another agent is on that field, it loses internal energy equal to the danger amount. If this energy loss kills the agent, its remaining resources transform into food and water dropped on the map.

**Reproduction and Mutation**
To successfully reproduce, an agent's internal food and water levels must both exceed a threshold of 0.6. The reproduction process deducts a flat cost of 0.2 from both resources, and the parent transfers half of its remaining food and water to the new offspring. The newborn inherits the parent's actions, sensors, and neural network weights, subject to random mutations. The neural network weights mutate by adding a random value drawn from the normal distribution $\mathcal{N}(0,0.2)$. During this mutation phase, 0, 1, or 2 randomly selected sensors are removed and replaced with entirely new ones featuring random types and coordinates. Similarly, 0, 1, or 2 actions are also randomly replaced with new ones.

## Parameters and behaviour

Modify parameters of the simulation in the `constants.h` file. The phenomena observed in the project are explained in `evolucja_otwarta.pdf` file (in polish).

