import numpy as np
import math

def filter_keypoints(img, keypoints, patch_size = 9):
    # TODO: Filter out keypoints that are too close to the edges
    size = img.shape
    # calculate the rim
    cutoff = int(np.ceil(patch_size / 2.0))
    # remove rim from all sides (could probably be done simpler)
    keypoints = keypoints[np.where(keypoints[:, 0] >= cutoff)]
    keypoints = keypoints[np.where(keypoints[:, 0] < size[1] - cutoff - 1)]
    keypoints = keypoints[np.where(keypoints[:, 1] >= cutoff)]
    keypoints = keypoints[np.where(keypoints[:, 1] < size[0] - cutoff - 1)]
    return keypoints

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

