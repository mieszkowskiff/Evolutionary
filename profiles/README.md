# Profiling

To profile the program, you must first build the project.

## 1. Build the Project

From the root directory of the repository, run the following commands:

```bash
mkdir build
cd build
mkdir save
cmake ..
make
```

## 2. Run the Profiler

After building, navigate to the `profiles` folder:

```bash
cd ../profiles
```

Run the profiler using NVIDIA Nsight Systems (`nsys`):

```bash
DEBUGINFOD_URLS="" nsys profile --export=sqlite -o profile_name --resolve=false ../build/Evolutionary
```

> **Note:** You can replace `profile_name` with your desired output filename.

## 3. View Statistics

To read the profiling results directly from the terminal, use the following command (make sure the filename matches your output from the previous step):

```bash
nsys stats profile_name.sqlite
```

## 4. Plotting

To generate plots from multiple profiles, gather them in the same directory and ensure their filenames match the regular expression `([a-zA-Z]+)(\d+)\.sqlite` (for example, `run01.sqlite`, `run02.sqlite`). 

Then, execute the plotting script:

```bash
python3 main.py
```