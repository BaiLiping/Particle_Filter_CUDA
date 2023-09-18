#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "particle_filter.cuh"

namespace py = pybind11;

// Define Python bindings for Particle Filter functions
void predict_py(py::array_t<float> particles, py::array_t<float> u, py::array_t<float> std, float dt) {
    py::buffer_info particles_info = particles.request();
    py::buffer_info u_info = u.request();
    py::buffer_info std_info = std.request();

    if (particles_info.size != u_info.size || particles_info.size != std_info.size || particles_info.size % 3 != 0) {
        throw std::runtime_error("Input array sizes do not match.");
    }

    float* particles_ptr = static_cast<float*>(particles_info.ptr);
    float* u_ptr = static_cast<float*>(u_info.ptr);
    float* std_ptr = static_cast<float*>(std_info.ptr);
    int N = particles_info.size / 3;

    predict(particles_ptr, u_ptr, std_ptr, N, dt);
}

void update_py(py::array_t<float> particles, py::array_t<float> weights, py::array_t<float> z, float R, py::array_t<float> landmarks, int num_landmarks) {
    py::buffer_info particles_info = particles.request();
    py::buffer_info weights_info = weights.request();
    py::buffer_info z_info = z.request();
    py::buffer_info landmarks_info = landmarks.request();

    if (particles_info.size != weights_info.size || particles_info.size % 3 != 0) {
        throw std::runtime_error("Input array sizes do not match.");
    }

    float* particles_ptr = static_cast<float*>(particles_info.ptr);
    float* weights_ptr = static_cast<float*>(weights_info.ptr);
    float* z_ptr = static_cast<float*>(z_info.ptr);
    float* landmarks_ptr = static_cast<float*>(landmarks_info.ptr);
    int N = particles_info.size / 3;

    update(particles_ptr, weights_ptr, z_ptr, R, landmarks_ptr, N, num_landmarks);
}

void estimate_py(py::array_t<float> particles, py::array_t<float> weights, py::array_t<float> mean, py::array_t<float> var) {
    py::buffer_info particles_info = particles.request();
    py::buffer_info weights_info = weights.request();
    py::buffer_info mean_info = mean.request();
    py::buffer_info var_info = var.request();

    if (particles_info.size != weights_info.size || particles_info.size % 3 != 0) {
        throw std::runtime_error("Input array sizes do not match.");
    }

    float* particles_ptr = static_cast<float*>(particles_info.ptr);
    float* weights_ptr = static_cast<float*>(weights_info.ptr);
    float* mean_ptr = static_cast<float*>(mean_info.ptr);
    float* var_ptr = static_cast<float*>(var_info.ptr);
    int N = particles_info.size / 3;

    estimate(particles_ptr, weights_ptr, mean_ptr, var_ptr, N);
}

void resample_py(py::array_t<float> particles, py::array_t<float> weights) {
    py::buffer_info particles_info = particles.request();
    py::buffer_info weights_info = weights.request();

    if (particles_info.size != weights_info.size || particles_info.size % 3 != 0) {
        throw std::runtime_error("Input array sizes do not match.");
    }

    float* particles_ptr = static_cast<float*>(particles_info.ptr);
    float* weights_ptr = static_cast<float*>(weights_info.ptr);
    int N = particles_info.size / 3;

    resample(particles_ptr, weights_ptr, N);
}

// Define the Python module
PYBIND11_MODULE(particle_filter, m) {
    m.doc() = "Particle Filter Implementation";

    // Expose Particle Filter functions to Python
    m.def("predict", &predict_py, "Predict particle movement");
    m.def("update", &update_py, "Update particle weights based on measurement");
    m.def("estimate", &estimate_py, "Estimate weighted mean and variance");
    m.def("resample", &resample_py, "Resample particles based on weights");
}
