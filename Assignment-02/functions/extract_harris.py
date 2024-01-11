import numpy as np
import scipy
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    # kernels
    convx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    convy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    
    # gradients
    Ix = scipy.signal.convolve2d(img, convx, mode='same')
    Iy = scipy.signal.convolve2d(img, convy, mode='same')

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    
    # kernelsize ???
    kernelsize = (0, 0)
    Ix2 = cv2.GaussianBlur(Ix**2, kernelsize, sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix*Iy, kernelsize, sigma, borderType=cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(Iy**2, kernelsize, sigma, borderType=cv2.BORDER_REPLICATE)
    
    # not needed
    Mp = np.hstack((np.vstack((Ix2, Ixy)), np.vstack((Ixy, Iy2))))

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    detMp = Ix2 * Iy2 - Ixy**2
    trMp = Ix2 + Iy2
    C = detMp - k * (trMp**2)

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # create a mask for, where the maximum values are
    C_max = scipy.ndimage.maximum_filter(C, (3, 3))
    mask = (C_max == C)
    C = C * mask
    # print(C)

    # filter corners above threshold
    corners = np.argwhere(C > thresh)
    # I somehow messed up the x, y coordinates
    corners[:, [1, 0]] = corners[:, [0, 1]]
    # print(corners)

    return corners, C

