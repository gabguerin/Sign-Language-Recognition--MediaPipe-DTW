import numpy as np


class HandModel(object):
    def __init__(self, landmarks):
        """
        Args
            dataset: numpy array of shape (21,3) containing
                        the 3D coordinates of the 21 hand keypoints
        """
        self.landmark_names = [
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

        # Normalize dataset
        self.landmarks = self.normalize_landmarks(landmarks)

        # Compute embeddings
        self.embedding = self.get_distance_embedding(self.landmarks)

    def normalize_landmarks(self, landmarks):
        """
        Normalizes dataset translation and scale
        """
        # Take wrist's position as origin
        wrist_ = landmarks[self.landmark_names.index("wrist")]
        landmarks -= wrist_

        # Divide positions by the distance between the wrist & the middle finger
        palm_size = self.get_distance_by_names(landmarks, "wrist", "middle_0")
        landmarks /= palm_size

        return landmarks

    def get_distance_embedding(self, landmarks):
        """
        Computes a (M,1) vector where each element represents a distance between
        two keypoints of the hand.
        """
        tuple_names = [
            # Distances between finger tip and finger base
            ("index_0", "index_3"),
            ("middle_0", "middle_3"),
            ("ring_0", "ring_3"),
            ("pinky_0", "pinky_3"),
            # Distances between first phalanx
            ("thumb_1", "index_1"),
            ("thumb_1", "middle_1"),
            ("thumb_1", "ring_1"),
            ("thumb_1", "pinky_1"),
            ("index_1", "middle_1"),
            ("index_1", "ring_1"),
            ("index_1", "pinky_1"),
            ("middle_1", "ring_1"),
            ("middle_1", "pinky_1"),
            ("ring_1", "pinky_1"),
            # Distances between finger tips
            ("thumb_3", "index_3"),
            ("thumb_3", "middle_3"),
            ("thumb_3", "ring_3"),
            ("thumb_3", "pinky_3"),
            ("index_3", "middle_3"),
            ("index_3", "ring_3"),
            ("index_3", "pinky_3"),
            ("middle_3", "ring_3"),
            ("middle_3", "pinky_3"),
            ("ring_3", "pinky_3"),
        ]
        return np.array(list(map(lambda t: self.get_distance_by_names(landmarks, t[0], t[1]), tuple_names)))

    def get_distance_by_names(self, landmarks, name_from, name_to):
        landmark_from = landmarks[self.landmark_names.index(name_from)]
        landmark_to = landmarks[self.landmark_names.index(name_to)]
        distance = np.linalg.norm(landmark_to - landmark_from)
        # assert len(distance) == 1
        return distance
