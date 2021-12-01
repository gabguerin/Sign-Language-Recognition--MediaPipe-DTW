from models.hand_model import HandModel


class SignModel(object):

    def __init__(self, left_hand_list, right_hand_list):
        """
        Args
            landmarks_list: numpy array of shape (N,21,3) containing
                        the 3D coordinates of the 21 hand keypoints of the N frames of a video
        """
        self.left_hand_list = left_hand_list
        self.right_hand_list = right_hand_list

        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)

    @staticmethod
    def _get_embedding_from_landmark_list(landmarks_list):
        embeddings = []
        for landmarks in landmarks_list:
            hand_gesture = HandModel(landmarks)
            embeddings.append(hand_gesture.embedding)
        return embeddings

