from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_keypoints


class SignRecorder(object):
    def __init__(self, sign_dictionary: dict, sign_distances: dict, seq_len=40):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of landmarks of size (seq_len x nb_landmarks)
        self.left_hand_list = []
        self.right_hand_list = []
        self.pose_list = []

        # Dictionary storing the SignModels of the reference signs
        self.sign_dictionary = sign_dictionary
        # Dictionary storing the distances between this SignModel object & the reference signs
        self.sign_distances = sign_distances

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.sign_distances = {k: 0 for k, _ in self.sign_dictionary.items()}
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
            if len(self.pose_list) < self.seq_len:
                self.record_movement(results)
            else:
                self.compute_distances()
                print(self.sign_distances)
        if sum(self.sign_distances.values()) == 0:
            return "", self.is_recording
        return list(self.sign_distances.keys())[0], self.is_recording

    def record_movement(self, results):
        """
        Record pose, left_hand and right_hand landmarks
        """
        pose, left_hand, right_hand = extract_keypoints(results)
        self.left_hand_list.append(left_hand)
        self.right_hand_list.append(right_hand)
        self.pose_list.append(pose)

    def compute_distances(self):
        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(self.pose_list, self.left_hand_list, self.right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.sign_distances = dtw_distances(recorded_sign, self.sign_dictionary)

        # Reset variables
        self.left_hand_list = []
        self.right_hand_list = []
        self.pose_list = []
        self.is_recording = False
