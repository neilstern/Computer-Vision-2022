import numpy as np

def propagate(particles, frame_height, frame_width, params):
    A = np.zeros((particles.shape[1], particles.shape[1]))
    w = np.zeros((particles.shape[1], particles.shape[1]))
    generator = np.random.default_rng()
    if params["model"] == 0:
        A = np.array([[1, 0], [0, 1]])
        ran_x = generator.normal(0, params["sigma_position"], particles.shape[0])
        ran_y = generator.normal(0, params["sigma_position"], particles.shape[0])
        w = np.stack((ran_x, ran_y)).T
    else:
        A = np.array([[1, 0, 1, 0], [0, 1, 0 , 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        ran_x = generator.normal(0, params["sigma_position"], particles.shape[0])
        ran_y = generator.normal(0, params["sigma_position"], particles.shape[0])
        ran_vx = generator.normal(0, params["sigma_velocity"], particles.shape[0])
        ran_vy = generator.normal(0, params["sigma_velocity"], particles.shape[0])
        w = np.stack((ran_x, ran_y, ran_vx, ran_vy)).T
    
    for i in range(particles.shape[0]):
        particles[i] = np.matmul(A, particles[i]) + w[i]
        particles[i][0] = max(0, min(particles[i][0], frame_width - 1))
        particles[i][1] = max(0, min(particles[i][1], frame_height - 1))
    return particles