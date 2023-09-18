import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import particle_filter_module as pf

# Constants
NUM_PARTICLES = 5000
WORLD_SIZE = 500
NOISE_STD = 10.0
FRAME_NUMBER = 1000

# Simulated target movement
class Target:
    def __init__(self):
        self.x = WORLD_SIZE / 2
        self.y = WORLD_SIZE / 2
        self.direction = np.random.rand() * 20 * np.pi

    def move(self):
        self.x += np.cos(self.direction)
        self.y += np.sin(self.direction)
        if np.random.rand() > 0.95:
            self.direction = np.random.rand() * 20 * np.pi

# Create figure for plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Initializations
target = Target()
particles = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
particles[:, 0] = np.random.uniform(0, WORLD_SIZE, NUM_PARTICLES)  # x-coordinates
particles[:, 1] = np.random.uniform(0, WORLD_SIZE, NUM_PARTICLES)  # y-coordinates
measurements = []
particles_weights = np.ones(NUM_PARTICLES, dtype=np.float32) / NUM_PARTICLES
particles = np.column_stack((particles, particles_weights))


# Animation update function
def update(num):
    ax.clear()
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)

    # Move target and get noisy measurement
    target.move()
    noisy_x = target.x + NOISE_STD * np.random.randn()
    noisy_y = target.y + NOISE_STD * np.random.randn()
    measurements.append((noisy_x, noisy_y))

    print(f"Frame {num}:")
    print("Average particle position before prediction:", np.mean(particles[:, 0]), np.mean(particles[:, 1]))

    # Particle filter steps
    pf.prediction(particles, 1.0)
    
    print("Average particle position after prediction:", np.mean(particles[:, 0]), np.mean(particles[:, 1]))

    pf.update(particles, noisy_x, noisy_y)
    print("Particle weights after update - Min:", np.min(particles[:, 2]), "Max:", np.max(particles[:, 2]), "Average:", np.mean(particles[:, 2]))

    new_particles = np.zeros_like(particles)
    cumulative_weights = np.cumsum(particles[:, 1])
    pf.resample(particles, new_particles, cumulative_weights)
    particles[:] = new_particles

    print("Average particle position after resampling:", np.mean(particles[:, 0]), np.mean(particles[:, 1]))
    print("-" * 50)
# 

    # For the initial frame, just plot the initialized particles.
    if num == 0:
        ax.plot(particles[:, 0], particles[:, 1], 'b.', markersize=2, label="Particles (Initial)")
        ax.legend()
        return
    ax.plot([t[0] for t in measurements], [t[1] for t in measurements], 'r.', markersize=2, label="Measurements")
    ax.plot(target.x, target.y, 'go', markersize=10, label="True Position")
    ax.plot(particles[:, 0], particles[:, 1], 'b.', markersize=2, label="Particles")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=FRAME_NUMBER, repeat=False)
plt.show()
