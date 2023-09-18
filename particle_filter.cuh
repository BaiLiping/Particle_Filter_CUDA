#ifndef PARTICLE_FILTER_CUH
#define PARTICLE_FILTER_CUH

struct Particle {
    float x;
    float y;
    float weight;
};

// Only the external function declarations are needed
extern "C" {
    void prediction_step(Particle* particles, float move_radius, int num_particles);
    void update_step(Particle* particles, float measurement_x, float measurement_y, int num_particles);
    void resample_step(Particle* particles, Particle* new_particles, float* cumulative_weights, int num_particles);
}

#endif  // PARTICLE_FILTER_CUH
