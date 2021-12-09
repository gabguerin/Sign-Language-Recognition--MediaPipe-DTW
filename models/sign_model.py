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

        self.lh_embedding = self._get_embedding_from_landmark_list(
            left_hand_list, pose_list, hand="left"
        )
        self.rh_embedding = self._get_embedding_from_landmark_list(
            right_hand_list, pose_list, hand="right"
        )

    @staticmethod
    def _get_embedding_from_landmark_list(hand_list, pose_list, hand):
        embeddings = []
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            pose = PoseModel(pose_list[frame_idx])

            embedding = hand_gesture.feature_vector
            # if hand == "left":
            #    embedding += pose.left_arm_embedding
            # elif hand == "right":
            #    embedding += pose.right_arm_embedding
            # else:
            #    raise ValueError(f"Error in the hand type: {hand} type does not exist.")

            embeddings.append(embedding)
        return embeddings
