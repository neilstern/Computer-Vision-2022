import numpy as np

def resample(particles, particles_w):
    particles_new = np.zeros(particles.shape)
    weights_new = np.zeros(particles_w.shape)
    particles_i = np.random.choice(particles.shape[0], size=particles.shape[0], p=particles_w.T[0])
    for i in range(particles.shape[0]):
        particles_new[i] = particles[particles_i[i]]
        weights_new[i] = particles_w[particles_i[i]]
    if np.sum(weights_new) != 0:
        return particles_new, weights_new / np.sum(weights_new)
    return particles, np.ones(particles_w.shape) * 1.0/particles_w.shape[0]