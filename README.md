nvcc -std=c++11 -shared -o particle_filter_module`python3-config --extension-suffix`      -Xcompiler '-fPIC'      -I/path/to/pybind11/include `python3 -m pybind11 --includes`      wrapper.cpp particle_filter.cu

