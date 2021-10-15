import cv2
import os
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, save_array
from scipy.linalg import expm, norm


def landmark_to_array(landmark_list):
    keypoints = []
    for landmark in landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


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
    u = np.array([0, 1, 0])
    # Vector orthogonal to axis & u
    v = np.cross(u, axis)
    # The angle between axis & (0,1,0) is the arccos of the dot product of the two vectors
    dot_product = np.dot(u, axis / np.linalg.norm(axis))
    angle = np.arccos(dot_product)
    # Rotate keypoints
    return rotate_keypoints(keypoints, v, angle)


def extract_keypoints(results):
    """
    Transform results in a list of standardized keypoints to be able to compute dtw distances
    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: A tuple containing hand landmarks or NaN if a hand doesn't appear in the results
    """
    # this list contains keypoints of the left and right hand
    pose = landmark_to_array(results.pose_landmarks)
    axis = np.array((0, 1, (pose[11, 2] + pose[12, 2]) / 2))

    lh = np.zeros(63 * 3)
    if results.left_hand_landmarks:
        lh1 = landmark_to_array(results.left_hand_landmarks)

        lh2 = rotate_keypoints(lh1, axis, 0.05)
        lh2 = translate_keypoints(lh2, 0, pose[15])

        lh3 = rotate_keypoints(lh1, axis, -0.05)
        lh3 = translate_keypoints(lh3, 0, pose[15])

        lh4 = translate_keypoints(lh1, 0, [0.5, 0.5, 0])

        lh5 = fix_keypoints(lh4, lh4[5] - lh4[0])

        lh = lh1.reshape(63).tolist() + lh2.reshape(63).tolist() + lh3.reshape(63).tolist() + lh4.reshape(63).tolist() + lh5.reshape(63).tolist()

    rh = np.zeros(63 * 3)
    if results.right_hand_landmarks:
        rh1 = landmark_to_array(results.right_hand_landmarks)

        rh2 = rotate_keypoints(rh1, axis, 0.05)
        rh2 = translate_keypoints(rh2, 0, pose[15])

        rh3 = rotate_keypoints(rh1, axis, -0.05)
        rh3 = translate_keypoints(rh3, 0, pose[15])

        rh4 = translate_keypoints(rh1, 0, [0.5, 0.5, 0])

        rh5 = fix_keypoints(rh4, rh4[5] - rh4[0])

        rh = rh1.reshape(63).tolist() + rh2.reshape(63).tolist() + rh3.reshape(63).tolist() + rh4.reshape(63).tolist() + rh5.reshape(63).tolist()
    return lh, rh


def extract_landmarks(video, save=True):
    lh_list = []
    rh_list = []

    cap = cv2.VideoCapture(os.path.join("Videos",video))
    # Set mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Store results
                lh, rh = extract_keypoints(results)
                lh_list.append(lh)
                rh_list.append(rh)
            else:
                break
        cap.release()


    if save:
        # Saving landmarks
        fname = video.split('.')[0] if '.' in video else video
        path = os.path.join("landmarks", fname + ".pickle")
        save_array([lh_list, rh_list], path)