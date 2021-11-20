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


def extract_keypoints(results):
    """
    Transform results in a list of standardized keypoints to be able to compute dtw distances
    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: A tuple containing hand landmarks or NaN if a hand doesn't appear in the results
    """
    # this list contains keypoints of the left and right hand
    pose = landmark_to_array(results.pose_landmarks)
    axis = np.array((0, 1, (pose[11, 2] + pose[12, 2]) / 2))

    lh = np.zeros(63)
    if results.left_hand_landmarks:
        lh = landmark_to_array(results.left_hand_landmarks)

        lh = translate_keypoints(lh, 4, [0, 0, 0])
        lh = fix_keypoints(lh, lh[5] - lh[0])
        lh = translate_keypoints(lh, 4, [0, 0, 0])

        lh = lh.reshape(63).tolist()

    rh = np.zeros(63)
    if results.right_hand_landmarks:
        rh = landmark_to_array(results.right_hand_landmarks)

        rh = translate_keypoints(rh, 4, [0, 0, 0])
        rh = fix_keypoints(rh, rh[4] - rh[0])
        rh = translate_keypoints(rh, 4, [0, 0, 0])

        rh = rh.reshape(63).tolist()
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