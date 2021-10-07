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
    standardize_results(results)

    pose = flatten_landmarks(results.pose_landmarks, 9, 18)
    lh = flatten_landmarks(results.left_hand_landmarks, 0, 21)
    rh = flatten_landmarks(results.right_hand_landmarks, 0, 21)
    return np.concatenate([pose, lh, rh])


def standardize_results(results):
    if results.left_hand_landmarks:
        left_wrist = results.left_hand_landmarks.landmark[0]
        for landmark in results.left_hand_landmarks.landmark:
            landmark.x -= left_wrist.x - 0.5
            landmark.y -= left_wrist.y - 0.5
            landmark.z -= left_wrist.z
    if results.right_hand_landmarks:
        right_wrist = results.right_hand_landmarks.landmark[0]
        for landmark in results.right_hand_landmarks.landmark:
            landmark.x -= right_wrist.x - 0.5
            landmark.y -= right_wrist.y - 0.5
            landmark.z -= right_wrist.z


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
