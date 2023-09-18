#ifndef PARTICLE_FILTER_CUH
#define PARTICLE_FILTER_CUH

#include <cuda_runtime.h>

// Define constants
const int THREADS_PER_BLOCK = 256;

// Function declarations

// Kernel to predict particle movement based on control input
__global__ void predict_kernel(float* particles, const float* u, const float* std, int N, float dt);

// Kernel to update particle weights based on measurement
__global__ void update_kernel(float* particles, float* weights, const float* z, float R, const float* landmarks, int N, int num_landmarks);

// Kernel to compute weighted mean and variance
__global__ void estimate_kernel(const float* particles, const float* weights, float* mean, float* var, int N);

// Kernel to resample particles based on weights
__global__ void resample_kernel(float* particles, const float* weights, int N);

// Function to run the predict kernel
void predict(float* particles, const float* u, const float* std, int N, float dt);

// Function to run the update kernel
void update(float* particles, float* weights, const float* z, float R, const float* landmarks, int N, int num_landmarks);

// Function to run the estimate kernel
void estimate(const float* particles, const float* weights, float* mean, float* var, int N);

// Function to run the resample kernel
void resample(float* particles, const float* weights, int N);

#endif // PARTICLE_FILTER_CUH