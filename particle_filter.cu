#include "particle_filter.cuh"
#include <curand_kernel.h>

// Define helper function to get index
__device__ int get_global_idx_1d_1d() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Predict kernel
__global__ void predict_kernel(float* particles, const float* u, const float* std, int N, float dt) {
    int idx = get_global_idx_1d_1d();
    if (idx >= N) return;

    curandState state;
    curand_init(1234, idx, 0, &state);

    // Update heading
    particles[idx*3 + 2] += u[0] + curand_normal(&state) * std[0];
    particles[idx*3 + 2] = fmodf(particles[idx*3 + 2], 2.0f * M_PI);

    // Move in the (noisy) commanded direction
    float dist = (u[1] * dt) + curand_normal(&state) * std[1];
    particles[idx*3] += cosf(particles[idx*3 + 2]) * dist;
    particles[idx*3 + 1] += sinf(particles[idx*3 + 2]) * dist;
}

// Update kernel
__global__ void update_kernel(float* particles, float* weights, const float* z, float R, const float* landmarks, int N, int num_landmarks) {
    int idx = get_global_idx_1d_1d();
    if (idx >= N) return;

    for (int i = 0; i < num_landmarks; i++) {
        float dx = particles[idx*3] - landmarks[i*2];
        float dy = particles[idx*3 + 1] - landmarks[i*2 + 1];
        float distance = sqrtf(dx*dx + dy*dy);
        float weight = expf(-(distance - z[i])*(distance - z[i]) / (2.0f * R*R));
        weights[idx] *= weight;
    }
}


// Estimate kernel
__global__ void estimate_kernel(const float* particles, const float* weights, float* mean, float* var, int N) {
    // NOTE: This is a simplified version to compute weighted mean and variance.
    // Proper parallel reduction should be implemented for optimal performance.
    int idx = get_global_idx_1d_1d();
    if (idx >= N) return;

    atomicAdd(&mean[0], particles[idx*3] * weights[idx]);
    atomicAdd(&mean[1], particles[idx*3 + 1] * weights[idx]);
    atomicAdd(&var[0], particles[idx*3] * particles[idx*3] * weights[idx]);
    atomicAdd(&var[1], particles[idx*3 + 1] * particles[idx*3 + 1] * weights[idx]);
}

// Resample kernel
__global__ void resample_kernel(float* particles, const float* weights, int N) {
    // TODO: Implement the resampling logic. 
    // Due to the complexity of resampling in parallel, this is a placeholder.
}

// Wrapper functions
void predict(float* particles, const float* u, const float* std, int N, float dt) {
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    predict_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(particles, u, std, N, dt);
}

void update(float* particles, float* weights, const float* z, float R, const float* landmarks, int N, int num_landmarks) {
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    update_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(particles, weights, z, R, landmarks, N, num_landmarks);
}

void estimate(const float* particles, const float* weights, float* mean, float* var, int N) {
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    estimate_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(particles, weights, mean, var, N);
}

void resample(float* particles, const float* weights, int N) {
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    resample_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(particles, weights, N);
}