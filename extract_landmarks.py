import cv2
import os
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, save_array
from data_transformation import *


def landmark_to_array(landmark_list):
    keypoints = []
    for landmark in landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_keypoints(results, proportions=None):
    """
    Transform results in a list of standardized keypoints to be able to compute dtw distances
    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: A tuple containing hand landmarks or NaN if a hand doesn't appear in the results
    """
    # Proportions of the different keypoint transformations
    if proportions is None:
        proportions = [1, 1, 1, 1, 1]
    p1, p2, p3, p4, p5 = proportions

    # this list contains keypoints of the left and right hand
    pose = landmark_to_array(results.pose_landmarks)
    axis = np.array((0, 1, (pose[11, 2] + pose[12, 2]) / 2))

    lh = np.zeros(63 * 3)
    if results.left_hand_landmarks:
        lh1 = landmark_to_array(results.left_hand_landmarks)

        lh2 = rotate_keypoints(lh1, axis, 0.1)
        lh2 = translate_keypoints(lh2, 4, [0, 0, 0])

        lh3 = rotate_keypoints(lh1, axis, -0.1)
        lh3 = translate_keypoints(lh3, 4, [0, 0, 0])

        lh4 = translate_keypoints(lh1, 4, [0, 0, 0])

        lh5 = fix_keypoints(lh4, lh4[4] - lh4[0])

        lh = p1 * lh1.reshape(63).tolist() + \
             p2 * lh2.reshape(63).tolist() + \
             p3 * lh3.reshape(63).tolist() + \
             p4 * lh4.reshape(63).tolist() + \
             p5 * lh5.reshape(63).tolist()

    rh = np.zeros(63 * 3)
    if results.right_hand_landmarks:
        rh1 = landmark_to_array(results.right_hand_landmarks)

        rh2 = rotate_keypoints(rh1, axis, 0.1)
        rh2 = translate_keypoints(rh2, 4, [0, 0, 0])

        rh3 = rotate_keypoints(rh1, axis, -0.1)
        rh3 = translate_keypoints(rh3, 4, [0, 0, 0])

        rh4 = translate_keypoints(rh1, 4, [0, 0, 0])

        rh5 = fix_keypoints(rh4, rh4[4] - rh4[0])


        rh = p1 * rh1.reshape(63).tolist() + \
             p2 * rh2.reshape(63).tolist() + \
             p3 * rh3.reshape(63).tolist() + \
             p4 * rh4.reshape(63).tolist() + \
             p5 * rh5.reshape(63).tolist()

    return lh, rh


def extract_landmarks(video, save=True):
    lh_list = []
    rh_list = []

    cap = cv2.VideoCapture(os.path.join("Videos",video))
    # Set mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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