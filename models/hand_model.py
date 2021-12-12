import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic
CONNECTIONS = [connection for connection in mp_holistic.HAND_CONNECTIONS]


class HandModel(object):
    def __init__(self, landmarks):
        """
        Args
            dataset: numpy array of shape (21,3) containing
                        the 3D coordinates of the 21 hand keypoints
        """
        # Reshape landmarks
        self.landmarks = np.array(landmarks).reshape((21, 3))

        # Connexions between hand keypoints
        self.connections = list(
            map(
                lambda t: self.landmarks[t[1]] - self.landmarks[t[0]],
                CONNECTIONS,
            )
        )
