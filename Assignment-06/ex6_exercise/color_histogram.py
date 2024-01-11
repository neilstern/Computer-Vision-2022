import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    hist = np.zeros((3, hist_bin))
    xmin = min(max(0, xmin), frame.shape[0] - 2)
    ymin = min(max(0, ymin), frame.shape[1] - 2)
    xmax = min(max(xmin + 1, xmax), frame.shape[0] - 1)
    ymax = min(max(ymin + 1, ymax), frame.shape[1] - 1)
    for c in range(3):
        hist[c], _edges = np.histogram(frame[ymin:ymax, xmin:xmax, c], hist_bin, density=True)
    if np.sum(hist) != 0:
        return hist / np.sum(hist)
    return np.zeros((3, hist_bin))