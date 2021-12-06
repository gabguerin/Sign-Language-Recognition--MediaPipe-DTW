import numpy as np

KEYPOINTS = [
    "wrist",
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "index_0",
    "index_1",
    "index_2",
    "index_3",
    "middle_0",
    "middle_1",
    "middle_2",
    "middle_3",
    "ring_0",
    "ring_1",
    "ring_2",
    "ring_3",
    "pinky_0",
    "pinky_1",
    "pinky_2",
    "pinky_3",
]

CONNEXIONS = [
    # Connexions between wrist and fingers
    ("wrist", "thumb_0"),
    ("wrist", "index_0"),
    ("wrist", "middle_0"),
    ("wrist", "ring_0"),
    ("wrist", "pinky_0"),
    # Connexions between fingers
    ("index_0", "middle_0"),
    ("middle_0", "ring_0"),
    ("ring_0", "pinky_0"),
    # Connexions between phalanxes
    ("thumb_0", "thumb_1"),
    ("thumb_1", "thumb_2"),
    ("thumb_2", "thumb_3"),
    ("index_0", "index_1"),
    ("index_1", "index_2"),
    ("index_2", "index_3"),
    ("middle_0", "middle_1"),
    ("middle_1", "middle_2"),
    ("middle_2", "middle_3"),
    ("ring_0", "ring_1"),
    ("ring_1", "ring_2"),
    ("ring_2", "ring_3"),
    ("pinky_0", "pinky_1"),
    ("pinky_1", "pinky_2"),
    ("pinky_2", "pinky_3"),
]


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
        self.connexions = list(
            map(
                lambda t: self.landmarks[KEYPOINTS.index(t[1])] - self.landmarks[KEYPOINTS.index(t[0])],
                CONNEXIONS
            )
        )

        # Compute embeddings
        self.feature_vector = self._get_angles_embedding()

    def _get_angles_embedding(self):
        """
        Computes a (M,1) vector where each element represents an angle between
        two hand connexions.
        """
        embedding = []
        for connexion_from in self.connexions:
            for connexion_to in self.connexions:
                embedding.append(self._get_angle_between_vectors(connexion_from, connexion_to))
        return embedding

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray):
        """
        The angle between two vector is:
            Theta = ArcCOS(u_^v_ / ||u.v||2)
        """
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)

    def _normalize_landmarks(self, landmarks):
        """
        Normalizes dataset translation and scale
        """
        # Take wrist's position as origin
        wrist_ = landmarks[KEYPOINTS.index("wrist")]
        landmarks -= wrist_

        # Divide positions by the distance between the wrist & the middle finger
        palm_size = self._get_distance_by_names(landmarks, "wrist", "middle_0")
        landmarks /= palm_size

        return landmarks

    @staticmethod
    def _get_distance_by_names(landmarks, name_from, name_to):
        landmark_from = landmarks[KEYPOINTS.index(name_from)]
        landmark_to = landmarks[KEYPOINTS.index(name_to)]
        distance = np.linalg.norm(landmark_to - landmark_from)
        return distance
