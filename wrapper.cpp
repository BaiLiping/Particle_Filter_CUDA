
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "particle_filter.cuh"

namespace py = pybind11;

// Wrap the predict function
void py_predict(py::array_t<float> particles, py::array_t<float> u, py::array_t<float> std, int N, float dt) {
    predict(particles.mutable_data(), u.data(), std.data(), N, dt);
}

// Wrap the update function
void py_update(py::array_t<float> particles, py::array_t<float> weights, py::array_t<float> z, float R, py::array_t<float> landmarks, int N, int num_landmarks) {
    update(particles.mutable_data(), weights.mutable_data(), z.data(), R, landmarks.data(), N, num_landmarks);
}

// Wrap the estimate function
void py_estimate(py::array_t<float> particles, py::array_t<float> weights, py::array_t<float> mean, py::array_t<float> var, int N) {
    estimate(particles.data(), weights.data(), mean.mutable_data(), var.mutable_data(), N);
}

// Wrap the resample function
void py_resample(py::array_t<float> particles, py::array_t<float> weights, int N) {
    resample(particles.mutable_data(), weights.data(), N);
}

PYBIND11_MODULE(particle_filter, m) {
    m.def("predict", &py_predict, "Predict particle movement");
    m.def("update", &py_update, "Update particle weights");
    m.def("estimate", &py_estimate, "Compute weighted mean and variance");
    m.def("resample", &py_resample, "Resample particles");
}

