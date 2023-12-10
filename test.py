import cv2
import mediapipe
import numpy as np
import pandas as pd

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.test_dataset_utils import load_test_dataset, load_test_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from test_sign_recorder import TestSignRecorder
from webcam_manager import WebcamManager
from collections import Counter
from utils.dtw import dtw_distances, soft_dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks

def process_results(self, results) -> (str, bool):
    """
    If the SignRecorder is in the recording state:
        it stores the landmarks during seq_len frames and then computes the sign distances
    :param results: mediapipe output
    :return: Return the word predicted (blank text if there is no distances)
            & the recording state
    """
    if self.is_recording:
        # # if len(self.recorded_results) < self.seq_len:
        # #     self.recorded_results.append(results)
        # else:
            self.compute_distances()
            print(self.reference_signs)

    if np.sum(self.reference_signs["distance"].values) == 0:
        return "", self.is_recording
    return self._get_sign_predicted(), self.is_recording

def compute_distances(signs):
    """
    Updates the distance column of the reference_signs
    and resets recording variables
    """
    left_hand_list, right_hand_list = [], []
    for results in signs:
        _, left_hand, right_hand = extract_landmarks(results)
        left_hand_list.append(left_hand)
        right_hand_list.append(right_hand)

    # Create a SignModel object with the landmarks gathered during recording
    recorded_sign = SignModel(left_hand_list, right_hand_list)

    # Compute sign similarity with DTW or SoftDTW (ascending order)
    # self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)
    self.reference_signs = soft_dtw_distances(recorded_sign, self.reference_signs)

    # Reset variables
    self.recorded_results = []
    self.is_recording = False

def _get_sign_predicted(self, batch_size=5, threshold=0.5):
    """
    Method that outputs the sign that appears the most in the list of closest
    reference signs, only if its proportion within the batch is greater than the threshold

    :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
    :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                    we output the sign_name
                        If not,
                    we output "Sign not found"
    :return: The name of the predicted sign
    """
    # Get the list (of size batch_size) of the most similar reference signs
    sign_names = self.reference_signs.iloc[:batch_size]["name"].values

    # Count the occurrences of each sign and sort them by descending order
    sign_counter = Counter(sign_names).most_common()

    predicted_sign, count = sign_counter[0]
    if count / batch_size < threshold:
        return "Signe inconnu"
    return predicted_sign


if __name__ == "__main__":
    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    videos = load_dataset()
    test_videos = load_test_dataset()
    reference_signs = load_reference_signs(videos)
    print()
    test_signs = load_test_reference_signs(test_videos)
    print(test_signs)

    # Object that stores mediapipe results and computes sign similarities
    # sign_recorder = SignRecorder(reference_signs)
    # test_sign_recorder = SignRecorder(test_signs)
    
    # # Object that draws keypoints & displays results
    # webcam_manager = WebcamManager()

    # # Set up the Mediapipe environment
    # with mediapipe.solutions.holistic.Holistic(
    #     min_detection_confidence=0.5, min_tracking_confidence=0.5
    # ) as holistic:
    #     while cap.isOpened():

    #         # Read frame from the video file
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Make detections
    #         image, results = mediapipe_detection(frame, holistic)
    #         # print(results.face_landmarks)
    #         # print(results.pose_landmarks)
    #         # print(len(results.pose_landmarks[0]))
    #         # Process results
    #         sign_detected, is_recording = sign_recorder.process_results(results)

    #         # Update the frame (draw landmarks & display result)
    #         webcam_manager.update(frame, results, sign_detected, is_recording)

    #         pressedKey = cv2.waitKey(1) & 0xFF
