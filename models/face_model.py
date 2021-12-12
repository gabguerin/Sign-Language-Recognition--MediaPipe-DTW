import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
LEFT_ARM_CONNECTIONS = [
    (
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
    ),
    (
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
    ),
    (
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
        mp_holistic.PoseLandmark.LEFT_WRIST.value,
    ),
]


class FaceModel(object):
    def __init__(self, landmarks):
        # Reshape landmarks
        self.landmarks = np.array(landmarks).reshape((33, 3))

        # Left and right arms connections
        self.connections = list(
            map(
                lambda t: self.landmarks[t[1]] - self.landmarks[t[0]],
                CONNECTIONS,
            )
        )
