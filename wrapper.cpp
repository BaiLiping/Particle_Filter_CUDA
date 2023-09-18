#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "particle_filter.cuh"

namespace py = pybind11;

void prediction_wrapper(py::array_t<float> particles, float move_radius) {
    py::buffer_info buf_info = particles.request();
    Particle* ptr = static_cast<Particle*>(buf_info.ptr);
    int num_particles = buf_info.size;

    prediction_step(ptr, move_radius, num_particles);
}

void update_wrapper(py::array_t<float> particles, float measurement_x, float measurement_y) {
    py::buffer_info buf_info = particles.request();
    Particle* ptr = static_cast<Particle*>(buf_info.ptr);
    int num_particles = buf_info.size;

    update_step(ptr, measurement_x, measurement_y, num_particles);
}

void resample_wrapper(py::array_t<float> particles, py::array_t<float> new_particles, py::array_t<float> cumulative_weights) {
    py::buffer_info buf_info_particles = particles.request();
    py::buffer_info buf_info_new_particles = new_particles.request();
    py::buffer_info buf_info_weights = cumulative_weights.request();

    Particle* ptr_particles = static_cast<Particle*>(buf_info_particles.ptr);
    Particle* ptr_new_particles = static_cast<Particle*>(buf_info_new_particles.ptr);
    float* ptr_weights = static_cast<float*>(buf_info_weights.ptr);

    int num_particles = buf_info_particles.size;

    resample_step(ptr_particles, ptr_new_particles, ptr_weights, num_particles);
}

PYBIND11_MODULE(particle_filter_module, m) {
    m.def("prediction", &prediction_wrapper, "Predict particle positions");
    m.def("update", &update_wrapper, "Update particle weights based on measurements");
    m.def("resample", &resample_wrapper, "Resample particles based on their weights");
}
