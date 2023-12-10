from typing import List

import numpy as np
import mediapipe as mp


class TotalModel(object):
    """
    Params
        landmarks: List of positions
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: List of length 21 * 21 = 441 containing the angles between all connections
    """

    def __init__(self, landmarks: List[float]):

        # Define the connections
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS

        # Create feature vector (list of the angles between all the connections)
        landmarks = np.array(landmarks).reshape((15 + 21 + 21, 3))
        pose_list = landmarks[:15]  # (15, 3) 배열
        left_hand_list = landmarks[15:36]  # (21, 3) 배열
        right_hand_list = landmarks[36:]  # (21, 3) 배열
        
        v_st = pose_list[[11], :] - pose_list[[12], :]
        v1 = pose_list[[9,9,9,9,9], :]
        v2 = pose_list[[10,10,10,10,10],:]
        v3 = left_hand_list[[4,8,12,16,20], :]
        v4 = right_hand_list[[4,8,12,16,20], :]

        v5 = np.concatenate((v3 - v1, v4 - v2))

        v = np.concatenate((v5, pose_list[[13], :] - pose_list[[11], :], pose_list[[14], :] - pose_list[[12], :]))
        angles = []
        for _ in range(12):
            angle = self._get_angle_between_vectors(v_st[0], v[[_], :][0])
            angles.append(angle)
        self.feature_vector = np.concatenate((angles, self._get_feature_vector(left_hand_list), self._get_feature_vector(right_hand_list)))
        # if True:
        #     raise ValueError(self.feature_vector)
            # raise ValueError(len(self.feature_vector), len(angles), len(self._get_feature_vector(left_hand_list)), len(self._get_feature_vector(right_hand_list)))
    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """
        Params
            landmarks: numpy array of shape (15 + 21 + 21, 3)
        Return
            List of length nb_connections * nb_connections containing
            all the angles between the connections
        """
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for connection_from in connections:
            for connection_to in connections:
                angle = self._get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not NaN we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    def _get_connections_from_landmarks(
        self, landmarks: np.ndarray
    ) -> List[np.ndarray]:
        """
        Params
            landmarks: numpy array of shape (15 + 21 + 21, 3)
        Return
            List of vectors representing hand connections
        """
        return list(
            map(
                lambda t: landmarks[t[1]] - landmarks[t[0]],
                self.connections,
            )
        )

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Args
            u, v: 3D vectors representing two connections
        Return
            Angle between the two vectors
        """
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
