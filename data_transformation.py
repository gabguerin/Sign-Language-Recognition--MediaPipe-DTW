import numpy as np
from scipy.linalg import expm, norm


def translate_keypoints(keypoints: np.array, idx: int, target_v: np.array):
    """
    Function that will centered all keypoints
        to better compare hand movement for example
    :param keypoints: list of keypoints
    :param idx: idx of the keypoint which will be moved
    :return: list of keypoints centered
    """
    if np.sum(keypoints) == 0:
        return keypoints

    kp0 = keypoints[idx]
    for i in range(keypoints.shape[0]):
        keypoints[i, 0] -= kp0[0] - target_v[0]
        keypoints[i, 1] -= kp0[1] - target_v[1]
        keypoints[i, 2] -= kp0[2] - target_v[2]
    return keypoints


def rotate_keypoints(keypoints: np.array, axis: np.array, angle: float):
    rot_M = expm(np.cross(np.eye(3), axis / norm(axis) * angle))
    return np.dot(rot_M, keypoints.T).T

def fix_keypoints(keypoints: np.array, axis: np.array):
    """
    Function that will fix one landmark connexion (axis)
    to the unit vector y, so that results are standardized
    :param keypoints: landmark keypoints
    :param axis: landmark connexion that we want to move to the vector (0,1,0)
    :return: new keypoints rotated toward (0,1,0)
    """
    u = np.array([0, -1, 0])
    # Vector orthogonal to axis & u
    v = np.cross(u, axis)
    # The angle between axis & (0,1,0) is the arccos of the dot product of the two vectors
    dot_product = np.dot(u, axis / np.linalg.norm(axis))
    angle = np.arccos(dot_product)
    # Rotate keypoints
    return rotate_keypoints(keypoints, v, angle)

