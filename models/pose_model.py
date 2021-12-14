import numpy as np


class PoseModel(object):
    def __init__(self, landmarks):

        self.landmark_names = [
            "shoulder",
            "elbow",
            "wrist",
        ]

        # Reshape landmarks
        landmarks = np.array(landmarks).reshape((33, 3))

        self.left_arm_landmarks = self._normalize_landmarks(
            [landmarks[lmk_idx] for lmk_idx in [11, 13, 15]]
        )
        self.right_arm_landmarks = self._normalize_landmarks(
            [landmarks[lmk_idx] for lmk_idx in [12, 14, 16]]
        )

        self.left_arm_embedding = self.left_arm_landmarks[
            self.landmark_names.index("wrist")
        ].tolist()
        self.right_arm_embedding = self.right_arm_landmarks[
            self.landmark_names.index("wrist")
        ].tolist()

    def _normalize_landmarks(self, landmarks):
        """
        Normalizes dataset translation and scale
        """
        # Take shoulder's position as origin
        shoulder_ = landmarks[self.landmark_names.index("shoulder")]
        landmarks -= shoulder_

        # Divide positions by the distance between the wrist & the middle finger
        arm_size = self._get_distance_by_names(landmarks, "shoulder", "elbow")
        landmarks /= arm_size

        return landmarks

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        landmark_from = landmarks[self.landmark_names.index(name_from)]
        landmark_to = landmarks[self.landmark_names.index(name_to)]
        distance = np.linalg.norm(landmark_to - landmark_from)
        return distance
