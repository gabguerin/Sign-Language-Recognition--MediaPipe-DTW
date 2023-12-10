from typing import List
import numpy as np
from models.hand_model import HandModel
from models.total_model import TotalModel  # Import the TotalModel class

class SignModel(object):
    def __init__(
        self, left_hand_list: List[List[float]], right_hand_list: List[List[float]], pose_list: List[List[float]]
    ):
        """
        Params
            left_hand_list: List of all landmarks for the left hand for each frame of a video
            right_hand_list: List of all landmarks for the right hand for each frame of a video
            pose_list: List of all landmarks for the pose for each frame of a video
        Args
            has_left_hand: bool; True if left hand is detected in the video, otherwise False
            has_right_hand: bool; True if right hand is detected in the video, otherwise False
            has_pose: bool; True if pose is detected in the video, otherwise False
            lh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections) for the left hand
            rh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections) for the right hand
            pose_embedding: ndarray; Array of shape (n_frame, n_connections * n_connections) for the pose
        """
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0
        self.has_pose = self.has_left_hand or self.has_right_hand

        # Use HandModel and TotalModel to calculate embeddings for both hands and pose
        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)
        # if True:
        #     raise ValueError((self.lh_embedding))  
        # print(pose_list.shape)
        # print(right_hand_list.shape)

        self.pose_embedding = self._get_total_embedding_from_landmark_list(pose_list, left_hand_list, right_hand_list)
        # if True:
        #     raise ValueError(self.pose_embedding)
    @staticmethod
    def _get_embedding_from_landmark_list(
        hand_list: List[List[float]],
    ) -> List[List[float]]:
        """
        Params
            hand_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
            # if True:
            #     raise ValueError(embedding)  
        return embedding


    @staticmethod
    def _get_total_embedding_from_landmark_list(
    pose_list: List[List[float]],
    left_hand_list: List[List[float]],
    right_hand_list: List[List[float]]
) -> List[List[float]]:
        embedding = []
        for frame_idx in range(len(pose_list)):
            if np.sum(left_hand_list[frame_idx]) == 0:
                if np.sum(right_hand_list[frame_idx]) == 0:
                    continue
            # if True:
            #     raise ValueError(type(left_hand_list[frame_idx]))
            # Create a TotalModel instance with both sets of landmarks
            concatenated_landmarks = np.concatenate((pose_list[frame_idx], left_hand_list[frame_idx], right_hand_list[frame_idx]))
            # if True:
            #     raise ValueError((concatenated_landmarks))
            total_model = TotalModel(concatenated_landmarks)
            embedding.append(total_model.feature_vector)
            # if True:
            #     raise ValueError(embedding)
        return embedding