import cv2
import os
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, save_array


def flatten_landmarks(landmark_list, s, e):
    if landmark_list:
         return np.array([[res.x, res.y, res.z] for res in landmark_list.landmark[s:e]]).flatten()
    return np.zeros((e - s - 1) * 3) # Return zero matrix if the landmark doesn't exist

def extract_keypoints(results):
    pose = flatten_landmarks(results.pose_landmarks, 9, 18)
    lh = flatten_landmarks(results.pose_landmarks, 0, 21)
    rh = flatten_landmarks(results.pose_landmarks, 0, 21)
    return np.concatenate([pose, lh, rh])


def extract_landmarks(video, save=True):
    keypoint_list = []

    cap = cv2.VideoCapture(os.path.join("Videos",video))
    # Set mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Store results
                keypoint_list.append(extract_keypoints(results))

                if save:
                    # Saving landmarks
                    fname = video.split('.')[0] if '.' in video else video
                    path = os.path.join("landmarks", fname+".pickle")
                    save_array(keypoint_list, path)
            else:
                break
        cap.release()

    return keypoint_list
