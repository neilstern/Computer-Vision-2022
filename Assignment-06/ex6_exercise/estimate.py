import numpy as np

def estimate(particles, particles_w):
    return np.sum((particles_w * particles), axis=0)