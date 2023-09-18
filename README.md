nvcc -O3 -shared -std=c++11 --compiler-options '-fPIC' `python3 -m pybind11 --includes` wrapper.cpp particle_filter.cu -o particle_filter_lib`python3-config --extension-suffix`

