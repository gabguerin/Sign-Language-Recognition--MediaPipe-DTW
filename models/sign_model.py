import numpy as np

from models.hand_model import HandModel
from models.pose_model import PoseModel


class SignModel(object):
    def __init__(self, pose_list, left_hand_list, right_hand_list):
        """
        Args
            landmarks_list: numpy array of shape (n_frames, n_keypoints, 3) containing
                            the 3D coordinates of the keypoints in a video
        """
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.feature_vector = self._get_embedding_from_landmark_list(
            pose_list, left_hand_list, right_hand_list
        )

    def _get_embedding_from_landmark_list(
        self, pose_list, left_hand_list, right_hand_list
    ):
        embeddings = []
        for pose, left_hand, right_hand in zip(
            pose_list, left_hand_list, right_hand_list
        ):
            lh_gesture = HandModel(left_hand)
            rh_gesture = HandModel(right_hand)
            pose = PoseModel(pose)

            connections = []
            if self.has_left_hand:
                connections += lh_gesture.connections
                connections += pose.left_arm_connections
            if self.has_right_hand:
                connections += rh_gesture.connections
                connections += pose.right_arm_connections

            embeddings.append(embedding)
        return embeddings
