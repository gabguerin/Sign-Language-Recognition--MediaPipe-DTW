import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_keypoints


class SignRecorder(object):
    def __init__(self, sign_dictionary: pd.DataFrame, seq_len=40):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.sign_dictionary = sign_dictionary

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.sign_dictionary["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.record_movement(results)
            else:
                self.compute_distances()
                print(self.sign_dictionary)

        if np.sum(self.sign_dictionary["distance"].values) == 0:
            return "", self.is_recording

        return self._get_sign_predicted(), self.is_recording

    def record_movement(self, results):
        """
        Stores pose, left_hand and right_hand landmarks
        """
        self.recorded_results.append(results)

    def compute_distances(self):
        """
        Updates the distance column of the sign_dictionary
        and resets recording variables
        """
        pose_list, left_hand_list, right_hand_list = [], [], []
        for results in self.recorded_results:
            pose, left_hand, right_hand = extract_keypoints(results)
            pose_list.append(pose)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(pose_list, left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.sign_dictionary = dtw_distances(recorded_sign, self.sign_dictionary)

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
        sign_names = self.sign_dictionary.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / batch_size > threshold:
            return predicted_sign
        return "Signe inconnu"
