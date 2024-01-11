import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    q1 = desc1.shape[0]
    q2 = desc2.shape[0]
    
    #sorry for the for loop (I tried)
    dist = np.zeros((q1, q2))
    for i in range(q1):
        for j in range(q2):
            dist[i][j] = np.sum((desc1[i] - desc2[j])**2)
            

    # mat1 = np.tile(desc1[:, 1], (q2, 1)).transpose()
    # mat2 = np.tile(desc2[:, 1].transpose(), (q1, 1))
    # mat = (mat1 - mat2)**2
    # sumcol = mat[:, ::3] + mat[:, 1::3] + mat[:, 2::3]
    # sum = sumcol[::3, :] + sumcol[1::3, :] + sumcol[2::3, :]
    return dist

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here

        # check smallest element in each column
        mindist = np.argmin(distances, axis=1)
        result = np.column_stack((np.arange(q1), mindist))
        return result
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here

        # check smallest element in each column
        mindist1 = np.argmin(distances, axis=1)
        # check smallest element in each row
        mindist2 = np.argmin(distances, axis=0)
        result = np.empty([0,2], dtype=int)
        for i in range(q1):
            if (i == mindist2[mindist1[i]]):
                result = np.append(result, np.array([[i, mindist1[i]]]), axis=0)
        return result
    elif method == "ratio":
        # TODO: implement the ratio test matching here

        # check smallest two element in each column
        mindist = np.argmin(distances, axis=1)
        min1dist = np.partition(distances, 0, axis=1)[:, 0]
        min2dist = np.partition(distances, 1, axis=1)[:, 1]
        result = np.empty([0,2], dtype=int)
        for i in range(q1):
            if (min1dist[i] < ratio_thresh*min2dist[i]):
                result = np.append(result, np.array([[i, mindist[i]]]), axis=0)
        return result
    else:
        raise NotImplementedError
    return matches