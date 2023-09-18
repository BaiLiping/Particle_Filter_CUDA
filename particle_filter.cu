#include "particle_filter.cuh"
#include <cuda_runtime.h>

// CUDA kernel implementations
__global__ void prediction_kernel(Particle* particles, float move_radius, int num_particles) {
    // ... kernel code ...
}

__global__ void update_kernel(Particle* particles, float measurement_x, float measurement_y, int num_particles) {
    // ... kernel code ...
}

__global__ void resample_kernel(Particle* particles, Particle* new_particles, float* cumulative_weights, int num_particles) {
    // ... kernel code ...
}

extern "C" {
    void prediction_step(Particle* particles, float move_radius, int num_particles) {
        prediction_kernel<<<(num_particles + 255) / 256, 256>>>(particles, move_radius, num_particles);
        cudaDeviceSynchronize();
    }

    void update_step(Particle* particles, float measurement_x, float measurement_y, int num_particles) {
        update_kernel<<<(num_particles + 255) / 256, 256>>>(particles, measurement_x, measurement_y, num_particles);
        cudaDeviceSynchronize();
    }

    void resample_step(Particle* particles, Particle* new_particles, float* cumulative_weights, int num_particles) {
        resample_kernel<<<(num_particles + 255) / 256, 256>>>(particles, new_particles, cumulative_weights, num_particles);
        cudaDeviceSynchronize();
    }
}
