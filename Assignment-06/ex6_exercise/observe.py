import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram
import math

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    particles_w = np.zeros((particles.shape[0], 1))
    for i in range(particles.shape[0]):
        xmin = round(particles[i][0] - 0.5*bbox_width)
        ymin = round(particles[i][1] - 0.5*bbox_height)
        xmax = round(particles[i][0] + 0.5*bbox_width)
        ymax = round(particles[i][1] + 0.5*bbox_height)
        hist_e = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        dist = chi2_cost(hist_e, hist)
        particles_w[i] = math.exp(-1.0 * dist**2 / (2 * sigma_observe**2)) / (math.sqrt(2 * math.pi) * sigma_observe)
    if np.sum(particles_w) != 0:
        return particles_w / np.sum(particles_w)
    return np.ones(particles_w.shape) * 1.0/particles_w.shape[0]
